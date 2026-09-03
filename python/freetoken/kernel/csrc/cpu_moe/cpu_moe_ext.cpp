// CPU-compute MoE executor for the "cpu" offload backend.
//
// Decode ships activations to the CPU, computes the routed experts here (reading
// the pinned host expert banks at full RAM bandwidth), and ships the results
// back. To keep the whole decode path inside a single CUDA graph we expose
// submit/sync as host nodes via cudaLaunchHostFunc -- the callbacks only touch a
// CPU worker pool + pinned host buffers and never call any CUDA API.
//
// One task is in flight at a time (per MoE layer): submit() wakes the pool,
// sync() blocks the host-func thread until the pool drains. The heavy GEMV runs
// on the persistent worker threads, not the host-func thread.
//
// Weight formats: bf16, NVFP4, MXFP4, ds_fp4 and Q4_0 expert banks (see WFmt and
// the per-format bank schemas). Compute is FP32-accumulate; the intermediate is
// stored bf16 to match the GPU decode path. ISA is chosen once at construction
// (AVX-512-BF16 dpbf16 -> AVX-512F widening -> AVX2+FMA -> scalar).

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <torch/extension.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define CPU_MOE_HAS_AFFINITY 1
#else
#define CPU_MOE_HAS_AFFINITY 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_MOE_X86 1
#else
#define CPU_MOE_X86 0
#endif

namespace {

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round-to-nearest-even
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

// ACT_SWIGLUOAI is the clamped (up + 1) swiglu (gpt-oss "swigluoai" /
// MiniMax-M3 and clamped_silu for GLM-5.3): gate/up are combined jointly with
// the runtime limit/alpha scalars, so do_pass1 handles them (act_apply never
// sees either; the mxfp4 kernel additionally fuses its own swigluoai math).
enum ActKind {
  ACT_SILU = 0,
  ACT_GELU = 1,
  ACT_GELU_TANH = 2,
  ACT_SWIGLUOAI = 3,
  ACT_CLAMPED_SILU = 4,
};

inline float act_apply(int act, float x) {
  if (act == ACT_SILU) return x / (1.0f + std::exp(-x));
  if (act == ACT_GELU)
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
  // gelu_tanh
  const float k0 = 0.79788456080286535588f;  // sqrt(2/pi)
  const float inner = k0 * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// ------------------------------- dot products -------------------------------
// dot(weight[bf16], act[bf16], n) -> fp32. The selected impl is a function
// pointer chosen at runtime; the per-row call overhead is negligible vs n.

using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

// Software prefetch distance (bytes) ahead of the current weight row stream. The
// weight stream is the bandwidth bottleneck (read once, never reused); nudging the
// HW prefetcher with a few cache lines of lookahead raises sustained throughput.
constexpr int PF_AHEAD = 512;

#if CPU_MOE_X86
__attribute__((target("avx512f")))
float dot_avx512f(const bf16_t* w, const bf16_t* x, int n) {
  // 4 independent accumulators -> more in-flight loads (memory-level parallelism),
  // which is what lifts a bandwidth-bound GEMV toward peak.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 64; j += 16) {
      __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j));
      __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + j));
      __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
      __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
      __m512& acc = (j == 0) ? a0 : (j == 16) ? a1 : (j == 32) ? a2 : a3;
      acc = _mm512_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 16 <= n; i += 16) {
    __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
    __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
    a0 = _mm512_fmadd_ps(wf, xf, a0);
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOE_HAS_AVX512BF16 1
__attribute__((target("avx512bf16,avx512f")))
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16(const bf16_t* w, const bf16_t* x, int n) {
  // 4 accumulators (128 bf16/iter) for memory-level parallelism + a prefetch nudge.
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // avx512bf16 available

// --------- AVX2 + FMA (256-bit fallback for CPUs without AVX-512) ----------
// Covers Intel 12-14th gen / Arrow Lake (AVX-512 fused off) and AMD Zen<4. The
// bf16->fp32 widen is a zero-extend + <<16; with 4 independent accumulators the
// GEMV is memory-bandwidth bound, same as the AVX-512 path (just half the width).
__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
float dot_avx2(const bf16_t* w, const bf16_t* x, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 32; j += 8) {
      __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + j));
      __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + j));
      __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
      __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
      __m256& acc = (j == 0) ? a0 : (j == 8) ? a1 : (j == 16) ? a2 : a3;
      acc = _mm256_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
    __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
    __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
    __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
    a0 = _mm256_fmadd_ps(wf, xf, a0);
  }
  float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // CPU_MOE_X86

// --------------------------- NVFP4 (W4A16) dequant ---------------------------
// Weights: e2m1 4-bit codes (2/byte, low nibble first), per-16 block scale in
// fp8-e4m3, per-output-row global scale in fp16. Dequant matches the GPU kernels
// (freetoken/kernel/triton/nvfp4_dequant.py): w = E2M1[code] * e4m3(scale) * global.
// Activations stay bf16 (W4A16); the GEMV dequantizes weights inside the K-loop.

const float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// e2m1 * 2 as exact int8 (all e2m1 values are multiples of 0.5). Used by the AVX-VNNI
// W4A8 path: nibble -> int8 weight via PSHUFB LUT, then VPDPBUSD against int8 activations;
// the *2 is undone by a 0.5 folded into the final scale. Mirrors ggml's kvalues_mxfp4.
alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// e4m3 (OCP "fn": finite, max-normal 448, exp bias 7). Decoded into a 256-entry LUT.
inline float e4m3_decode(uint8_t v) {
  const float sign = (v & 0x80u) ? -1.0f : 1.0f;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t man = v & 0x7u;
  if (exp == 0) return sign * (man / 8.0f) * 0.015625f;  // 2^(1-7) = 2^-6
  return sign * (1.0f + man / 8.0f) * std::ldexp(1.0f, (int)exp - 7);
}

// Activations pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]); see the
// ds_fp4 dot below for why (drops the hot loop to ~1.5 shuffle ops / 16 weights).
using nvdot_fn = float (*)(const uint8_t*, const uint8_t*, float, const float*, const float*,
                           int, const float*, const float*);

float dot_nvfp4_scalar(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const float bs = e4m3[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;  // 16 K -> 8 even + 8 odd
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];
      bsum += e2m1[byte >> 4] * xob[j];
    }
    acc += bs * bsum;
  }
  return acc * global;
}

// ---- NVFP4 W4A8 (int8 activations) dot: nibble->int8 via LUT, per-16 act scale ----
// asi8: int8 activations laid out per-16 block as [even(8), odd(8)]; asb[b] = per-block
// activation scale (absmax/127). Result folds the e2m1*2 -> *0.5 into the scale.
using nvi8dot_fn = float (*)(const uint8_t*, const uint8_t*, float, const int8_t*, int,
                             const float*, const float*);

// Row-batched W4A8 entry point. ``acts`` contains M quantized activation rows and
// ``act_scales`` contains their per-16 scales. The weight block is decoded once,
// then reused across every row before the kernel advances through the packed
// stream. This is a real MxK by Kx1 kernel, not a wrapper around M GEMVs.
using nvi8batch_fn = void (*)(float*, const uint8_t*, const uint8_t*, float,
                              const int8_t*, int, int, const float*, const float*);

// Weight-row tile for expert prefill. At the measured H=2560, M=160 geometry,
// the largest gate/up tile occupies 500 KiB of activation data + scales,
// 45 KiB of packed weights + scales + globals, and 40 KiB for both fp32
// projection tiles: 599,104 bytes (585 KiB), leaving headroom in Zen 4's
// 1 MiB private L2.
constexpr int kNvi8WeightRows = 32;
using nvi8batch_rows_fn = void (*)(float*, const uint8_t*, const uint8_t*,
                                   const uint16_t*, int, const int8_t*, int, int,
                                   const float*, const float*);

[[maybe_unused]] float dot_nvfp4_i8_scalar(const uint8_t* packed, const uint8_t* scale,
                          float global, const int8_t* asi8, int K, const float* e4m3,
                          const float* asb) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16;       // even(8)
    const int8_t* ao = ae + 8;                      // odd(8)
    int isum = 0;
    for (int j = 0; j < 8; ++j) {
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j];
      isum += (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    }
    acc += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return acc * (0.5f * global);
}

void batch_nvfp4_i8_scalar(float* out, const uint8_t* packed, const uint8_t* scale,
                           float global, const int8_t* acts, int M, int K,
                           const float* e4m3, const float* act_scales) {
  std::fill(out, out + M, 0.0f);
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    int8_t wb[16];
    for (int j = 0; j < 8; ++j) {
      wb[j] = kE2M1x2[pk[j] & 0xF];
      wb[8 + j] = kE2M1x2[pk[j] >> 4];
    }
    const float ws = 0.5f * global * e4m3[scale[b]];
    for (int m = 0; m < M; ++m) {
      const int8_t* a = acts + (size_t)m * K + (size_t)b * 16;
      int isum = 0;
      for (int j = 0; j < 16; ++j) isum += (int)wb[j] * (int)a[j];
      out[m] += ws * act_scales[(size_t)m * nb + b] * (float)isum;
    }
  }
}

void batch_nvfp4_i8_scalar_rows(float* out, const uint8_t* packed,
                                const uint8_t* scale, const uint16_t* globals,
                                int R, const int8_t* acts, int M, int K,
                                const float* e4m3, const float* act_scales) {
  const size_t packed_stride = static_cast<size_t>(K) / 2;
  const size_t scale_stride = static_cast<size_t>(K) / 16;
  for (int r = 0; r < R; ++r) {
    batch_nvfp4_i8_scalar(
        out + static_cast<size_t>(r) * M, packed + static_cast<size_t>(r) * packed_stride,
        scale + static_cast<size_t>(r) * scale_stride, fp16_to_f32(globals[r]),
        acts, M, K, e4m3, act_scales);
  }
}

#if CPU_MOE_X86
// AVX2 e2m1 nibble decode: codes (0..15) in 8 int32 lanes -> fp32. AVX2 vpermps is
// only 8-wide, so instead of a 16-entry LUT we use the e2m1 sign/magnitude symmetry:
// value = (code&8 ? - : +) * mag8[code&7], mag8 = e2m1[0..7]. The sign is bit 3 of
// the code shifted into the fp32 sign bit (bit 31). Bit-identical to the e2m1 LUT.
__attribute__((target("avx2,fma")))
inline __m256 e2m1_decode8(__m256i codes, __m256 mag8) {
  __m256 mag = _mm256_permutevar8x32_ps(mag8, _mm256_and_si256(codes, _mm256_set1_epi32(7)));
  __m256i sgn = _mm256_slli_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(8)), 28);
  return _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
}

// Two 16-K blocks (16 packed bytes) per iter: lo nibbles -> even-K, hi -> odd-K,
// gathered via two vpermps. The per-16 e4m3 scale differs across the two blocks, so
// it is applied per lane (low 8 lanes = block b, high 8 = block b+1).
__attribute__((target("avx512f")))
inline __m512 nvfp4_blk2(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                         __m512i loma, float s0, float s1) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  // lanes 0-7 (block b) -> s0, lanes 8-15 (block b+1) -> s1  (pure AVX512F mask move)
  __m512 scv = _mm512_mask_mov_ps(_mm512_set1_ps(s0), 0xFF00, _mm512_set1_ps(s1));
  return _mm512_mul_ps(prod, scv);
}

__attribute__((target("avx512f")))
float dot_nvfp4_avx512(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  const __m512 lut = _mm512_loadu_ps(e2m1);  // 16 e2m1 values
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 16;  // 8 packed bytes + one e4m3 scale per block
  int b = 0;
  for (; b + 4 <= nb; b += 4) {  // two blk2 calls -> 4 blocks, 2 accumulators
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
    acc1 = _mm512_add_ps(acc1, nvfp4_blk2(packed + (size_t)(b + 2) * 8, xe + (size_t)(b + 2) * 8,
                                          xo + (size_t)(b + 2) * 8, lut, loma, e4m3[scale[b + 2]], e4m3[scale[b + 3]]));
  }
  for (; b + 2 <= nb; b += 2)
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
  float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  for (; b < nb; ++b) {  // odd final 16-K block
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      bsum += e2m1[pk[j] & 0xF] * xeb[j] + e2m1[pk[j] >> 4] * xob[j];
    }
    s += e4m3[scale[b]] * bsum;
  }
  return s * global;
}

// AVX2: one 16-K block (8 packed bytes) per call, 8 even + 8 odd lanes.
__attribute__((target("avx2,fma")))
inline __m256 nvfp4_blk_avx2(const uint8_t* pk, const float* xeb, const float* xob,
                             __m256 mag8, float sc) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  __m256 prod = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
  return _mm256_mul_ps(prod, _mm256_set1_ps(sc));
}

__attribute__((target("avx2,fma")))
float dot_nvfp4_avx2(const uint8_t* packed, const uint8_t* scale, float global,
                     const float* xe, const float* xo, int K, const float* e2m1,
                     const float* e4m3) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);  // e2m1[0..7] magnitudes
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
    acc1 = _mm256_add_ps(acc1, nvfp4_blk_avx2(packed + (size_t)(b + 1) * 8, xe + (size_t)(b + 1) * 8,
                                              xo + (size_t)(b + 1) * 8, mag8, e4m3[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
  return hsum256(_mm256_add_ps(acc0, acc1)) * global;
}

// AVX-VNNI W4A8: decode 8 packed bytes (16 nibbles) of one 16-block to int8 [lo(8),hi(8)]
// via PSHUFB against the e2m1*2 LUT (replaces the 2 vpermps fp32 expands -- ~4x less
// port-5 traffic). lo=even-K weights, hi=odd-K, matching the [even(8),odd(8)] act layout.
__attribute__((target("avx2,avxvnni,fma")))
inline __m128i nvfp4_decode_block_i8(const uint8_t* pk, __m128i lut) {
  __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk));   // 8 bytes
  __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
  __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
  return _mm_shuffle_epi8(lut, _mm_unpacklo_epi64(lo, hi));            // [lo(8),hi(8)] -> int8
}

// Two 16-blocks per VPDPBUSD (32 int8). Sign trick (ggml mul_add_epi8): |w|*(sign(w)*a)=w*a,
// so u8*s8 needs no offset/correction term. Per-block scale (e4m3 * act-scale) folded in fp32
// (lanes 0-3 -> block b, 4-7 -> block b+1). Bit-faithful weight; only the int8 activation quant
// (W4A8) differs from the bf16 reference.
__attribute__((target("avx2,avxvnni,fma")))
float dot_nvfp4_i8_vnni(const uint8_t* packed, const uint8_t* scale, float global,
                        const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2));
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    __m128i wb = nvfp4_decode_block_i8(packed + (size_t)b * 8, lut);
    __m128i wb1 = nvfp4_decode_block_i8(packed + (size_t)(b + 1) * 8, lut);
    __m256i w = _mm256_set_m128i(wb1, wb);                              // [blk b | blk b+1]
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(asi8 + (size_t)b * 16));
    __m256i aw = _mm256_sign_epi8(w, w);                               // |w| (u8 operand)
    __m256i sa = _mm256_sign_epi8(a, w);                               // sign(w)*a (s8 operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    __m256 scv = _mm256_blend_ps(_mm256_set1_ps(e4m3[scale[b]] * asb[b]),
                                 _mm256_set1_ps(e4m3[scale[b + 1]] * asb[b + 1]), 0xF0);
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), scv, accF);
  }
  float s = hsum256(accF);
  for (; b < nb; ++b) {  // tail (odd block count)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}

// Expert-prefill GEMM. Packed weights are the outer stream, while all routed
// activation rows reuse each decoded pair of 16-K blocks. Each VPDPBUSD therefore
// serves M rows before the next weight bytes are loaded and decoded.
__attribute__((target("avx2,avxvnni,fma")))
void batch_nvfp4_i8_vnni(float* out, const uint8_t* packed, const uint8_t* scale,
                         float global, const int8_t* acts, int M, int K,
                         const float* e4m3, const float* act_scales) {
  std::fill(out, out + M, 0.0f);
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2));
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    const __m128i wb = nvfp4_decode_block_i8(packed + (size_t)b * 8, lut);
    const __m128i wb1 = nvfp4_decode_block_i8(packed + (size_t)(b + 1) * 8, lut);
    const __m256i w = _mm256_set_m128i(wb1, wb);
    const __m256i aw = _mm256_sign_epi8(w, w);
    const float ws0 = 0.5f * global * e4m3[scale[b]];
    const float ws1 = 0.5f * global * e4m3[scale[b + 1]];
    for (int m = 0; m < M; ++m) {
      const int8_t* arow = acts + (size_t)m * K;
      const __m256i a = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(arow + (size_t)b * 16));
      const __m256i sa = _mm256_sign_epi8(a, w);
      const __m256i di = _mm256_dpbusd_avx_epi32(
          _mm256_setzero_si256(), aw, sa);
      const float* as = act_scales + (size_t)m * nb;
      const __m256 scv = _mm256_blend_ps(
          _mm256_set1_ps(ws0 * as[b]),
          _mm256_set1_ps(ws1 * as[b + 1]), 0xF0);
      out[m] += hsum256(_mm256_mul_ps(_mm256_cvtepi32_ps(di), scv));
    }
  }
  for (; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const float ws = 0.5f * global * e4m3[scale[b]];
    for (int m = 0; m < M; ++m) {
      const int8_t* a = acts + (size_t)m * K + (size_t)b * 16;
      int isum = 0;
      for (int j = 0; j < 8; ++j) {
        isum += (int)kE2M1x2[pk[j] & 0xF] * (int)a[j];
        isum += (int)kE2M1x2[pk[j] >> 4] * (int)a[8 + j];
      }
      out[m] += ws * act_scales[(size_t)m * nb + b] * (float)isum;
    }
  }
}

__attribute__((target("avx2,avxvnni,fma")))
void batch_nvfp4_i8_vnni_rows(float* out, const uint8_t* packed,
                              const uint8_t* scale, const uint16_t* globals,
                              int R, const int8_t* acts, int M, int K,
                              const float* e4m3, const float* act_scales) {
  std::fill(out, out + static_cast<size_t>(R) * M, 0.0f);
  const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2));
  const int nb = K / 16;
  const size_t packed_stride = static_cast<size_t>(K) / 2;
  const size_t scale_stride = static_cast<size_t>(nb);
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    for (int r = 0; r < R; ++r) {
      const uint8_t* row_packed = packed + static_cast<size_t>(r) * packed_stride;
      const uint8_t* row_scale = scale + static_cast<size_t>(r) * scale_stride;
      const __m128i wb = nvfp4_decode_block_i8(
          row_packed + static_cast<size_t>(b) * 8, lut);
      const __m128i wb1 = nvfp4_decode_block_i8(
          row_packed + static_cast<size_t>(b + 1) * 8, lut);
      const __m256i w = _mm256_set_m128i(wb1, wb);
      const __m256i aw = _mm256_sign_epi8(w, w);
      const float row_global = 0.5f * fp16_to_f32(globals[r]);
      const float ws0 = row_global * e4m3[row_scale[b]];
      const float ws1 = row_global * e4m3[row_scale[b + 1]];
      float* row_out = out + static_cast<size_t>(r) * M;
      for (int m = 0; m < M; ++m) {
        const int8_t* arow = acts + static_cast<size_t>(m) * K;
        const __m256i a = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(arow + static_cast<size_t>(b) * 16));
        const __m256i sa = _mm256_sign_epi8(a, w);
        const __m256i di = _mm256_dpbusd_avx_epi32(
            _mm256_setzero_si256(), aw, sa);
        const float* as = act_scales + static_cast<size_t>(m) * nb;
        const __m256 scv = _mm256_blend_ps(
            _mm256_set1_ps(ws0 * as[b]),
            _mm256_set1_ps(ws1 * as[b + 1]), 0xF0);
        row_out[m] += hsum256(
            _mm256_mul_ps(_mm256_cvtepi32_ps(di), scv));
      }
    }
  }
  for (; b < nb; ++b) {
    for (int r = 0; r < R; ++r) {
      const uint8_t* row_packed = packed + static_cast<size_t>(r) * packed_stride;
      const uint8_t* row_scale = scale + static_cast<size_t>(r) * scale_stride;
      const uint8_t* pk = row_packed + static_cast<size_t>(b) * 8;
      const float ws = 0.5f * fp16_to_f32(globals[r]) * e4m3[row_scale[b]];
      float* row_out = out + static_cast<size_t>(r) * M;
      for (int m = 0; m < M; ++m) {
        const int8_t* a = acts + static_cast<size_t>(m) * K + static_cast<size_t>(b) * 16;
        int isum = 0;
        for (int j = 0; j < 8; ++j) {
          isum += (int)kE2M1x2[pk[j] & 0xF] * (int)a[j];
          isum += (int)kE2M1x2[pk[j] >> 4] * (int)a[8 + j];
        }
        row_out[m] += ws * act_scales[static_cast<size_t>(m) * nb + b] * (float)isum;
      }
    }
  }
}

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOE_HAS_AVX512VNNI 1

// Software-prefetch distance for the W4A8 weight stream, in 16-K blocks (8 packed
// bytes each). Returns -1 when FREETOKEN_CPU_MOE_PF_BLOCKS is unset: the kernel then
// uses the built-in default min(512 blocks = 4 KB, 2 rows) -- 4 KB is the empirical
// optimum on large-row machines (Emerald Rapids sweep), while the 2-row cap keeps a small-row
// model's overshoot bounded (the executor works in 32-row tiles, so a fixed byte
// distance otherwise prefetches another worker's tile: duplicated DRAM traffic that
// regresses at the bandwidth ceiling). An EXPLICIT env value is honored verbatim
// (no clamp; 0 disables): the per-machine optimum can sit past the safe default (+20%
// at 4 KB on a 24-thread Ice Lake with 256B rows), so the escape hatch must reach it.
// Prefetch never faults, so overshooting a row/bank tail is safe.
static int nvfp4_pf_blocks() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_PF_BLOCKS");
    return (s && s[0]) ? atoi(s) : -1;
  }();
  return v;
}

// AVX-512 VNNI W4A8: FOUR 16-K blocks per VPDPBUSD (64 int8) -- 2x the AVX-VNNI
// (256-bit) path. Decode 32 packed bytes -> 64 int8 with a single _mm512_shuffle_epi8
// (e2m1*2 LUT replicated to all 4 128-bit lanes). AVX-512 has no _mm512_sign_epi8, so
// the u8*s8 sign trick (|w| as u8, sign(w)*a as s8) uses abs_epi8 + a masked negate.
// Bit-faithful weight; only the int8 activation quant (W4A8) differs from bf16.
// One 4-block group (64 int8) -> scaled fp32 partial (16 lanes). Isolated as a helper so
// the caller can run several independent chains into separate accumulators (the decode ->
// dpbusd -> scale chain is long, so a single accumulator leaves the core latency-bound).
__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
static inline __m512 nvfp4_i8_grp4(const uint8_t* packed, const uint8_t* scale,
                                   const int8_t* asi8, const float* e4m3, const float* asb,
                                   int b, __m512i lut, __m512i idx, __m512i mask0F,
                                   __m512i idxsc) {
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;  // bytes 8-15 of each 128b lane -> hi nibbles
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + (size_t)b * 8));
  __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
  __m512i lo = _mm512_and_si512(src, mask0F);
  __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
  __m512i comb = _mm512_mask_blend_epi8(hi_half, lo, hi);
  __m512i w = _mm512_shuffle_epi8(lut, comb);  // 64 int8 weights (e2m1*2)
  // u8*s8 sign trick without _mm512_sign_epi8: aw=|w|, sa = (w<0 ? -a : a) (a moot at w==0)
  __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(asi8 + (size_t)b * 16));
  __m512i aw = _mm512_abs_epi8(w);
  __mmask64 neg = _mm512_movepi8_mask(w);
  __m512i sa = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
  __m512i di = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa);  // 16 int32 (groups of 4)
  // int32[0..3]->blk b, [4..7]->b+1, [8..11]->b+2, [12..15]->b+3. The 4 block scales
  // (e4m3 LUT x per-block act scale) are computed vectorized: a 4-byte load + epu8->epi32
  // widen + one 4-lane LUT gather + one mul replaces 8 scalar loads + 4 scalar muls +
  // a set_ps assembly, which otherwise dominates the per-group op count.
  int sc_raw;
  memcpy(&sc_raw, scale + b, 4);
  __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
  __m128 s4 = _mm_mul_ps(_mm_i32gather_ps(e4m3, sc4, 4), _mm_loadu_ps(asb + b));
  __m512 scv = _mm512_permutexvar_ps(idxsc, _mm512_castps128_ps512(s4));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv);
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
float dot_nvfp4_i8_avx512vnni(const uint8_t* packed, const uint8_t* scale, float global,
                              const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);  // block i -> 128b lane i
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
  const int nb = K / 16;
  // Built-in default: min(4 KB, 2 rows) -- see nvfp4_pf_blocks(). An explicitly set
  // env value is used verbatim so a per-machine sweep can reach operating points past
  // the conservative default.
  const int pfb = nvfp4_pf_blocks();
  const int pf = (pfb < 0) ? std::min(512, 2 * nb) : pfb;
  int b = 0;
  // Four independent 4-block groups per iter -> four accumulator chains hide the ~10-op
  // decode->dpbusd->scale latency (1-2 chains leave the loop latency-bound: the per-core
  // rate sat at ~60% of the core's achievable DRAM stream rate).
  for (; b + 16 <= nb; b += 16) {
    // Prefetch the weight stream ahead: the interleaved decode lowers the L1-miss
    // concurrency the HW prefetcher sustains on its own (134 -> 167 GB/s on Emerald Rapids).
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8 + 64),
                   _MM_HINT_T0);
    }
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
    acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx,
                                             mask0F, idxsc));
    acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx,
                                             mask0F, idxsc));
    acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx,
                                             mask0F, idxsc));
  }
  for (; b + 4 <= nb; b += 4)
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
  float s = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
  for (; b < nb; ++b) {  // tail (<4 remaining 16-K blocks)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}

// AVX-512 counterpart of the expert-prefill kernel. Keep the decoded weight group
// outside the M-row loop, as in the AVX-VNNI implementation, but use EVEX VPDPBUSD
// so AVX-512 VNNI CPUs do not depend on the distinct AVX-VNNI feature bit.
__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
void batch_nvfp4_i8_avx512vnni(float* out, const uint8_t* packed,
                               const uint8_t* scale, float global,
                               const int8_t* acts, int M, int K,
                               const float* e4m3, const float* act_scales) {
  std::fill(out, out + M, 0.0f);
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(
      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;
  const int nb = K / 16;
  int b = 0;
  for (; b + 4 <= nb; b += 4) {
    const __m256i raw = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(packed + (size_t)b * 8));
    const __m512i src = _mm512_permutexvar_epi64(
        idx, _mm512_castsi256_si512(raw));
    const __m512i lo = _mm512_and_si512(src, mask0F);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
    const __m512i w = _mm512_shuffle_epi8(
        lut, _mm512_mask_blend_epi8(hi_half, lo, hi));
    const __m512i aw = _mm512_abs_epi8(w);
    const __mmask64 neg = _mm512_movepi8_mask(w);

    int sc_raw;
    memcpy(&sc_raw, scale + b, 4);
    const __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
    const __m128 ws4 = _mm_mul_ps(
        _mm_i32gather_ps(e4m3, sc4, 4), _mm_set1_ps(0.5f * global));
    for (int m = 0; m < M; ++m) {
      const int8_t* arow = acts + (size_t)m * K;
      const __m512i a = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(arow + (size_t)b * 16));
      const __m512i sa = _mm512_mask_sub_epi8(
          a, neg, _mm512_setzero_si512(), a);
      const __m512i di = _mm512_dpbusd_epi32(
          _mm512_setzero_si512(), aw, sa);
      const __m128 as4 = _mm_loadu_ps(act_scales + (size_t)m * nb + b);
      const __m512 scv = _mm512_permutexvar_ps(
          idxsc, _mm512_castps128_ps512(_mm_mul_ps(ws4, as4)));
      out[m] += _mm512_reduce_add_ps(
          _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv));
    }
  }
  for (; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const float ws = 0.5f * global * e4m3[scale[b]];
    for (int m = 0; m < M; ++m) {
      const int8_t* a = acts + (size_t)m * K + (size_t)b * 16;
      int isum = 0;
      for (int j = 0; j < 8; ++j) {
        isum += (int)kE2M1x2[pk[j] & 0xF] * (int)a[j];
        isum += (int)kE2M1x2[pk[j] >> 4] * (int)a[8 + j];
      }
      out[m] += ws * act_scales[(size_t)m * nb + b] * (float)isum;
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
void batch_nvfp4_i8_avx512vnni_rows(
    float* out, const uint8_t* packed, const uint8_t* scale,
    const uint16_t* globals, int R, const int8_t* acts, int M, int K,
    const float* e4m3, const float* act_scales) {
  std::fill(out, out + static_cast<size_t>(R) * M, 0.0f);
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(
      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;
  const int nb = K / 16;
  const size_t packed_stride = static_cast<size_t>(K) / 2;
  const size_t scale_stride = static_cast<size_t>(nb);
  int b = 0;
  for (; b + 4 <= nb; b += 4) {
    for (int r = 0; r < R; ++r) {
      const uint8_t* row_packed = packed + static_cast<size_t>(r) * packed_stride;
      const uint8_t* row_scale = scale + static_cast<size_t>(r) * scale_stride;
      const __m256i raw = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(row_packed + static_cast<size_t>(b) * 8));
      const __m512i src = _mm512_permutexvar_epi64(
          idx, _mm512_castsi256_si512(raw));
      const __m512i lo = _mm512_and_si512(src, mask0F);
      const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
      const __m512i w = _mm512_shuffle_epi8(
          lut, _mm512_mask_blend_epi8(hi_half, lo, hi));
      const __m512i aw = _mm512_abs_epi8(w);
      const __mmask64 neg = _mm512_movepi8_mask(w);
      int sc_raw;
      memcpy(&sc_raw, row_scale + b, 4);
      const __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
      const __m128 ws4 = _mm_mul_ps(
          _mm_i32gather_ps(e4m3, sc4, 4),
          _mm_set1_ps(0.5f * fp16_to_f32(globals[r])));
      float* row_out = out + static_cast<size_t>(r) * M;
      for (int m = 0; m < M; ++m) {
        const int8_t* arow = acts + static_cast<size_t>(m) * K;
        const __m512i a = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(arow + static_cast<size_t>(b) * 16));
        const __m512i sa = _mm512_mask_sub_epi8(
            a, neg, _mm512_setzero_si512(), a);
        const __m512i di = _mm512_dpbusd_epi32(
            _mm512_setzero_si512(), aw, sa);
        const __m128 as4 = _mm_loadu_ps(
            act_scales + static_cast<size_t>(m) * nb + b);
        const __m512 scv = _mm512_permutexvar_ps(
            idxsc, _mm512_castps128_ps512(_mm_mul_ps(ws4, as4)));
        row_out[m] += _mm512_reduce_add_ps(
            _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv));
      }
    }
  }
  for (; b < nb; ++b) {
    for (int r = 0; r < R; ++r) {
      const uint8_t* row_packed = packed + static_cast<size_t>(r) * packed_stride;
      const uint8_t* row_scale = scale + static_cast<size_t>(r) * scale_stride;
      const uint8_t* pk = row_packed + static_cast<size_t>(b) * 8;
      const float ws = 0.5f * fp16_to_f32(globals[r]) * e4m3[row_scale[b]];
      float* row_out = out + static_cast<size_t>(r) * M;
      for (int m = 0; m < M; ++m) {
        const int8_t* a = acts + static_cast<size_t>(m) * K + static_cast<size_t>(b) * 16;
        int isum = 0;
        for (int j = 0; j < 8; ++j) {
          isum += (int)kE2M1x2[pk[j] & 0xF] * (int)a[j];
          isum += (int)kE2M1x2[pk[j] >> 4] * (int)a[8 + j];
        }
        row_out[m] += ws * act_scales[static_cast<size_t>(m) * nb + b] * (float)isum;
      }
    }
  }
}
#endif  // avx512vnni available
#endif

// =====================================================================================
// CUDA stream memory operations (driver API, resolved via dlopen -- no link-time or
// toolchain dependence). The GPU side of the flag handshake: submit = WRITE_VALUE
// (done[slot]=0 then ready[slot]=1), sync = WAIT_VALUE(done[slot] >= 1). The wait is
// executed by the GPU front-end (no SM-resident kernel), so GPU "utilization" stays
// truthful during CPU compute windows -- a resident spin kernel pinned it at 99%,
// which laptop CPU/GPU dynamic power schedulers answered by clamping the CPU's max
// frequency (GEMV workers -1.5x: the reported edge regression). Availability is
// probed functionally at startup (memops_probe); anything unsupported (Windows WDDM,
// vGPU, old drivers) falls back to the cudaLaunchHostFunc path.
#if defined(_WIN32)
#include <windows.h>
static void* cumemop_dlopen() { return (void*)::LoadLibraryA("nvcuda.dll"); }
static void* cumemop_dlsym(void* h, const char* n) {
  return (void*)::GetProcAddress((HMODULE)h, n);
}
#else
#include <dlfcn.h>
static void* cumemop_dlopen() {
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* cumemop_dlsym(void* h, const char* n) { return dlsym(h, n); }
#endif

using cuMemOp64_fn = int (*)(void* stream, unsigned long long addr, unsigned long long value,
                             unsigned int flags);
static cuMemOp64_fn g_cu_write64 = nullptr;
static cuMemOp64_fn g_cu_wait64 = nullptr;
static constexpr unsigned int kCuWaitValueGeq = 0x0;   // CU_STREAM_WAIT_VALUE_GEQ
static constexpr unsigned int kCuWriteDefault = 0x0;   // CU_STREAM_WRITE_VALUE_DEFAULT

static bool cumemop_resolve() {
  static bool resolved = [] {
    void* h = cumemop_dlopen();
    if (h == nullptr) return false;
    // 11.7+ made the v2 entry points the default; older drivers export only the v1
    // names with the same signature.
    g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64_v2"));
    if (g_cu_write64 == nullptr)
      g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64_v2"));
    if (g_cu_wait64 == nullptr)
      g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64"));
    return g_cu_write64 != nullptr && g_cu_wait64 != nullptr;
  }();
  return resolved;
}

// Functional probe on a scratch pinned int64: enqueue WRITE(7) + WAIT(>=7) + sync.
// Returns true only if the whole memop path works on THIS stream/device/driver.
static bool cumemops_probe(uintptr_t stream, uintptr_t scratch_addr) {
  if (!cumemop_resolve()) return false;
  auto* s = reinterpret_cast<void*>(stream);
  if (g_cu_write64(s, (unsigned long long)scratch_addr, 7ULL, kCuWriteDefault) != 0) return false;
  if (g_cu_wait64(s, (unsigned long long)scratch_addr, 7ULL, kCuWaitValueGeq) != 0) return false;
  return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
}

// GPU side of the flag handshake (see the block comment above): enqueued on the
// caller's (possibly capturing) stream; the WAIT immediate is the constant 1,
// replay-safe under CUDA graphs.
// The startup probe validates EAGER memops; a driver could still reject them at graph
// capture time. Those enqueue errors would otherwise be swallowed here and surface only
// as a later EndCapture failure -- log the first CUresult so triage is one step.
static void cumemop_check(int rc, const char* what) {
  static std::atomic<bool> warned{false};
  if (rc != 0 && !warned.exchange(true)) {
    std::fprintf(stderr,
                 "[freetoken/cpu_moe] %s failed with CUresult=%d (first occurrence; "
                 "subsequent errors are not repeated). If this happened during CUDA "
                 "graph capture, the driver lacks capture support for stream memops -- "
                 "set FREETOKEN_CPU_MOE_FLAG_SYNC=0.\n",
                 what, rc);
  }
}

static void cumemop_submit(uintptr_t stream, uintptr_t done_addr, uintptr_t ready_addr,
                           int64_t slot) {
  auto* s = reinterpret_cast<void*>(stream);
  // Order matters and is preserved by the front end: reset done BEFORE raising ready,
  // so the coordinator's completion write for THIS step can never be wiped.
  cumemop_check(g_cu_write64(s, (unsigned long long)(done_addr + (size_t)slot * 8), 0ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(done)");
  cumemop_check(g_cu_write64(s, (unsigned long long)(ready_addr + (size_t)slot * 8), 1ULL,
                             kCuWriteDefault),
                "cuStreamWriteValue64(ready)");
}

static void cumemop_sync(uintptr_t stream, uintptr_t done_addr, int64_t slot) {
  cumemop_check(g_cu_wait64(reinterpret_cast<void*>(stream),
                            (unsigned long long)(done_addr + (size_t)slot * 8), 1ULL,
                            kCuWaitValueGeq),
                "cuStreamWaitValue64(done)");
}

struct DotChoice {
  dot_fn fn;
  const char* name;
};

// SIMD tiers, ascending. Each format picks the highest tier <= the one chosen by
// pick_isa() that it implements (fp4 formats have no bf16-specific tier, so the
// avx512bf16 tier maps to their avx512 kernel).
enum IsaTier { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2, ISA_AVX512BF16 = 3 };

// Best tier the CPU+build supports, optionally capped DOWN by
// FREETOKEN_CPU_MOE_ISA={scalar,avx2,avx512,avx512bf16} (A/B testing on a machine
// that supports more). FREETOKEN_CPU_MOE_SCALAR=1 forces scalar (legacy alias).
inline IsaTier pick_isa() {
#if CPU_MOE_X86
  if (getenv("FREETOKEN_CPU_MOE_SCALAR")) return ISA_SCALAR;
  IsaTier best = ISA_SCALAR;
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) best = ISA_AVX2;
  if (best >= ISA_AVX2 && __builtin_cpu_supports("avx512f")) best = ISA_AVX512;
#ifdef CPU_MOE_HAS_AVX512BF16
  if (best >= ISA_AVX512 && __builtin_cpu_supports("avx512bf16")) best = ISA_AVX512BF16;
#endif
  if (const char* f = getenv("FREETOKEN_CPU_MOE_ISA")) {
    IsaTier want = best;
    if (!std::strcmp(f, "scalar")) want = ISA_SCALAR;
    else if (!std::strcmp(f, "avx2")) want = ISA_AVX2;
    else if (!std::strcmp(f, "avx512")) want = ISA_AVX512;
    else if (!std::strcmp(f, "avx512bf16")) want = ISA_AVX512BF16;
    if (want < best) best = want;  // cap downward; never force above hw/build support
  }
  return best;
#else
  return ISA_SCALAR;
#endif
}

DotChoice select_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
#ifdef CPU_MOE_HAS_AVX512BF16
  if (t >= ISA_AVX512BF16) return {dot_avx512bf16, "avx512bf16"};
#endif
  if (t >= ISA_AVX512) return {dot_avx512f, "avx512f"};
  if (t >= ISA_AVX2) return {dot_avx2, "avx2"};
#endif
  (void)t;
  return {dot_scalar, "scalar"};
}

nvdot_fn select_nvdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_nvfp4_avx512;
  if (t >= ISA_AVX2) return dot_nvfp4_avx2;
#endif
  (void)t;
  return dot_nvfp4_scalar;
}

// AVX-VNNI (VEX-256 VPDPBUSD) availability: Alder/Raptor Lake, Sapphire Rapids+, Zen5.
// Distinct from AVX-512 VNNI. Opt out with FREETOKEN_CPU_MOE_NO_VNNI=1 (A/B the W4A8 path).
inline bool cpu_has_avxvnni() {
#if CPU_MOE_X86
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no && no[0] && no[0] != '0') return false;  // ignore unset/empty/"0"
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni");
#else
  return false;
#endif
}

// AVX-512 VNNI (512-bit VPDPBUSD): Cascade Lake+, Ice Lake, Sapphire/Emerald Rapids,
// Zen4+. 2x the 256-bit AVX-VNNI width. FREETOKEN_CPU_MOE_NO_AVX512VNNI=1 forces the
// 256-bit path (A/B the two W4A8 kernels on the same box); FREETOKEN_CPU_MOE_NO_VNNI=1
// still disables the whole W4A8 family (back to the faithful fp32 nvdot), so it is
// honored here too.
inline bool cpu_has_avx512vnni() {
#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512VNNI)
  const char* no = getenv("FREETOKEN_CPU_MOE_NO_AVX512VNNI");
  if (no && no[0] && no[0] != '0') return false;
  const char* no_vnni = getenv("FREETOKEN_CPU_MOE_NO_VNNI");
  if (no_vnni && no_vnni[0] && no_vnni[0] != '0') return false;
  return __builtin_cpu_supports("avx512vnni");
#else
  return false;
#endif
}

// Serial and expert-batched W4A8 must resolve from this one feature decision. In
// particular, AVX-512 VNNI and AVX-VNNI are distinct CPUID features: probing only
// AVX-VNNI silently sent AVX-512 VNNI servers to the scalar batch kernel.
enum Nvi8Tier { NVI8_NONE = 0, NVI8_AVXVNNI = 1, NVI8_AVX512VNNI = 2 };

inline Nvi8Tier nvi8_tier_from_flags(bool has_avx512vnni, bool has_avxvnni) {
#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512VNNI)
  if (has_avx512vnni) return NVI8_AVX512VNNI;
#else
  (void)has_avx512vnni;
#endif
#if CPU_MOE_X86
  if (has_avxvnni) return NVI8_AVXVNNI;
#else
  (void)has_avxvnni;
#endif
  return NVI8_NONE;
}

inline Nvi8Tier detect_nvi8_tier() {
  // Keep this probe order identical to the serial path's historical preference.
  if (cpu_has_avx512vnni()) return NVI8_AVX512VNNI;
  if (cpu_has_avxvnni()) return NVI8_AVXVNNI;
  return NVI8_NONE;
}

struct Nvi8Dispatch {
  nvi8dot_fn dot;
  nvi8batch_fn batch;
  nvi8batch_rows_fn batch_rows;
  const char* batch_name;
};

Nvi8Dispatch select_nvi8_dispatch(Nvi8Tier tier) {
#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512VNNI)
  if (tier == NVI8_AVX512VNNI)
    return {dot_nvfp4_i8_avx512vnni, batch_nvfp4_i8_avx512vnni,
            batch_nvfp4_i8_avx512vnni_rows, "vnni_rows32"};
#endif
#if CPU_MOE_X86
  if (tier == NVI8_AVXVNNI)
    return {dot_nvfp4_i8_vnni, batch_nvfp4_i8_vnni,
            batch_nvfp4_i8_vnni_rows, "vnni_rows32"};
#endif
  return {nullptr, batch_nvfp4_i8_scalar, batch_nvfp4_i8_scalar_rows,
          "scalar_rows32"};
}

// ----------------------- DeepSeek-V4 ds_fp4 (W4A8) ---------------------------
// Row-major e2m1 (2/byte, low nibble first) + e8m0 per-32 block scale, no global
// (w = E2M1[code] * 2^(e8m0-127)); activations are FP8-e4m3 round-tripped (per-128
// block, ue8m0 scale) before each GEMM. Matches kernel/triton/dsv4 (fused_moe +
// fp8_linear): silu(clamp(gate,max=lim)) * clamp(up,-lim,lim), router weight on the
// down output.

// Activations are pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]) once per
// token/route and reused across all output rows. This drops the hot dot to one
// vpmovzxbd + two vpermps per 32 weights (1.5 shuffle ops / 16 vs 4 for a bf16,
// dup-permute, per-element gather), so the row-major fp4 GEMV stops being port-5
// bound and approaches the bf16 memory-bandwidth ceiling.
using dsdot_fn = float (*)(const uint8_t*, const uint8_t*, const float*, const float*, int,
                           const float*, const float*);

float dot_dsfp4_scalar(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const float sc = e8m0[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 16;  // 16 bytes = 32 codes
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    float bsum = 0.0f;
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];   // low nibble  -> even-K activation
      bsum += e2m1[byte >> 4] * xob[j];    // high nibble -> odd-K activation
    }
    acc += sc * bsum;
  }
  return acc;
}

#if CPU_MOE_X86
// One 32-block: 16 bytes -> 16 low + 16 high nibble values via two vpermps, times
// the pre-split even/odd fp32 activations, folded by the per-32 e8m0 scale.
__attribute__((target("avx512f")))
inline __m512 dsfp4_blk(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                        __m512i loma, float sc) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  return _mm512_mul_ps(prod, _mm512_set1_ps(sc));
}

__attribute__((target("avx512f")))
float dot_dsfp4_avx512(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 32;  // 16 packed bytes + one e8m0 scale per block
  int b = 0;
  for (; b + 2 <= nb; b += 2) {  // two independent accumulators hide FMA latency
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
    acc1 = _mm512_add_ps(acc1, dsfp4_blk(packed + (size_t)(b + 1) * 16, xe + (size_t)(b + 1) * 16,
                                         xo + (size_t)(b + 1) * 16, lut, loma, e8m0[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

// AVX2: a 32-K block is 16 bytes -> two 8-lane halves (8 even + 8 odd each).
__attribute__((target("avx2,fma")))
inline __m256 dsfp4_half_avx2(const uint8_t* pk, const float* xeb, const float* xob, __m256 mag8) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  return _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
}

__attribute__((target("avx2,fma")))
float dot_dsfp4_avx2(const uint8_t* packed, const uint8_t* scale, const float* xe,
                     const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 16;
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    const __m256 sc = _mm256_set1_ps(e8m0[scale[b]]);
    acc0 = _mm256_fmadd_ps(dsfp4_half_avx2(pk, xeb, xob, mag8), sc, acc0);
    acc1 = _mm256_fmadd_ps(dsfp4_half_avx2(pk + 8, xeb + 8, xob + 8, mag8), sc, acc1);
  }
  return hsum256(_mm256_add_ps(acc0, acc1));
}
#endif

dsdot_fn select_dsdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_dsfp4_avx512;
  if (t >= ISA_AVX2) return dot_dsfp4_avx2;
#endif
  (void)t;
  return dot_dsfp4_scalar;
}

// ------------------------- mxfp4 (gpt-oss) GEMV -----------------------------
// Transposed split-K layout: blk[Kpairs, N2] (N innermost), scl[Kpairs/16, N2]
// e8m0 per 32-K. Computes out[c] = sum_kb (E2M1[lo]*x[2kb] + E2M1[hi]*x[2kb+1])
// * 2^(e8m0-127) for a contiguous column tile (blk/scl already offset to col 0 of
// the tile). Vectorized over N (16 columns / __m512), K stays the outer (cache-
// sequential) loop. Used by both gate_up (K=H) and down (K=I).
using mxgemv_fn = void (*)(float*, const uint8_t*, const uint8_t*, const bf16_t*, int, int,
                           int, const float*, const float*);

void mxfp4_gemv_scalar(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  for (int c = 0; c < ncol; ++c) out[c] = 0.0f;
  for (int kb = 0; kb < Kpairs; ++kb) {
    const uint8_t* w = blk + (size_t)kb * N2;
    const uint8_t* s = scl + (size_t)(kb >> 4) * N2;
    const float xl = bf16_to_f32(x[2 * kb]);
    const float xh = bf16_to_f32(x[2 * kb + 1]);
    for (int c = 0; c < ncol; ++c) {
      const uint8_t byte = w[c];
      out[c] += (e2m1[byte & 0xF] * xl + e2m1[byte >> 4] * xh) * e8m0[s[c]];
    }
  }
}

#if CPU_MOE_X86
__attribute__((target("avx512f")))
void mxfp4_gemv_avx512(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[c]=2^(c-127) computed via bit construction (no gather)
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  // K-outer / N-inner: each kb cache line is read once and all live column chunks
  // (up to 4 -> 64 cols) accumulate from registers, so DRAM/L2 stream the tile once.
  int c0 = 0;
  for (; c0 + 16 <= ncol; c0 += 64) {
    const int nchunk = std::min(4, (ncol - c0) / 16);
    __m512 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = 32 K = one scale row
      __m512 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 16));
        sc[ci] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(sraw), 23));
      }
      __m512 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm512_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        // The transposed layout strides K by N2 bytes; prefetch ahead so the strided
        // reads are not exposed to DRAM latency (the HW streamer misses big strides).
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m512 xl = _mm512_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m512 xh = _mm512_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m512i wi = _mm512_cvtepu8_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(wbase + ci * 16)));
          __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
          __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
          blk_acc[ci] = _mm512_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm512_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm512_storeu_ps(out + c0 + ci * 16, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 16)
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}

__attribute__((target("avx2,fma")))
void mxfp4_gemv_avx2(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                     int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[s]=2^(s-127) built via s<<23 (no gather)
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  int c0 = 0;
  for (; c0 + 8 <= ncol; c0 += 32) {  // up to 4 chunks of 8 = 32 cols
    const int nchunk = std::min(4, (ncol - c0) / 8);
    __m256 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = one scale row
      __m256 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 8));
        sc[ci] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu8_epi32(sraw), 23));
      }
      __m256 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm256_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m256 xl = _mm256_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m256 xh = _mm256_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m256i wi = _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wbase + ci * 8)));
          __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
          __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
          blk_acc[ci] = _mm256_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm256_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm256_storeu_ps(out + c0 + ci * 8, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 8); none when ncol%8==0
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}
#endif

mxgemv_fn select_mxgemv() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return mxfp4_gemv_avx512;
  if (t >= ISA_AVX2) return mxfp4_gemv_avx2;
#endif
  (void)t;
  return mxfp4_gemv_scalar;
}

// Round a clamped |x|<=448 to nearest float8-e4m3 (RNE), back to fp32. Matches
// torch.float8_e4m3fn / triton .to(float8e4nv).
inline float e4m3_round(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float a = std::fabs(x);
  if (a == 0.0f) return 0.0f;
  if (a >= 448.0f) return sign * 448.0f;
  int e;
  std::frexp(a, &e);  // a in [2^(e-1), 2^e)
  float step = std::ldexp(1.0f, e - 4);
  const float min_step = std::ldexp(1.0f, -9);  // e4m3 subnormal step (2^-9)
  if (step < min_step) step = min_step;
  float r = std::nearbyint(a / step) * step;
  if (r > 448.0f) r = 448.0f;
  return sign * r;
}

// IEEE ceil(log2(v)) for v>0 (matches dsv4 _log2_ceil / fast_round_scale).
inline int ceil_log2_pos(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  const int exp = (int)((bits >> 23) & 0xFF);
  const int man = (int)(bits & 0x7FFFFF);
  return exp - 127 + (man != 0 ? 1 : 0);
}

// Split an interleaved bf16 row into fp32 even/odd halves (even[m]=src[2m]).
// bf16->fp32 is exact, so this only reorders -- done once per token/route and
// reused across every output row of the GEMV.
inline void deinterleave_bf16_f32(const bf16_t* src, float* even, float* odd, int K) {
  for (int m = 0; m < K / 2; ++m) {
    even[m] = bf16_to_f32(src[2 * m]);
    odd[m] = bf16_to_f32(src[2 * m + 1]);
  }
}

// DeepSeek-V4 activation FP8 round-trip (bf16 in/out): per 128-block,
// s = 2^ceil(log2(max(|x|,1e-4)/448)); y = round_e4m3(clamp(x/s,+-448)) * s.
void fp8_roundtrip_bf16(const bf16_t* src, bf16_t* dst, int K) {
  for (int b0 = 0; b0 < K; b0 += 128) {
    const int b1 = std::min(K, b0 + 128);
    float amax = 1e-4f;
    for (int i = b0; i < b1; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(src[i])));
    const float s = std::ldexp(1.0f, ceil_log2_pos(amax * (1.0f / 448.0f)));
    const float inv_s = 1.0f / s;
    for (int i = b0; i < b1; ++i) {
      float q = bf16_to_f32(src[i]) * inv_s;
      q = std::min(448.0f, std::max(-448.0f, q));
      dst[i] = f32_to_bf16(e4m3_round(q) * s);
    }
  }
}

// --------------------------------- executor ---------------------------------

struct CpuMoeExecutor;

struct MoeTask {
  CpuMoeExecutor* exec;
  int layer_id;
  int num_tokens;
  bool group_routes;      // persistent decode tasks retain their existing grouped schedule
  bool prefill_batch;     // synchronous prefill uses the expert-batched W4A8 schedule
  const bf16_t* x;     // [num_tokens, H]
  const int32_t* ids;  // [num_tokens, top_k]  (raw expert ids; <0 = skip)
  const float* w;      // [num_tokens, top_k]
  bf16_t* y;           // [num_tokens, H]
  // Native doorbell-to-completion span for optional per-step diagnostics. Atomics
  // let Python read the last completed replay after synchronizing its CUDA fence.
  std::atomic<int64_t> t_doorbell_ns{0};
  std::atomic<int64_t> t_groups_done_ns{0};
  std::atomic<int64_t> t_gil_acquired_ns{0};
  std::atomic<int64_t> t_precb_done_ns{0};
  std::atomic<int64_t> t_seen_ns{0};
  std::atomic<int64_t> t_done_stored_ns{0};
  std::atomic<int64_t> t_notified_ns{0};
  // Set per submission: the pre-run callback ran after notify (--moe-cpu-precb after),
  // so its GIL and body spans overlap compute and are not part of wake.
  bool timing_precb_deferred = false;
  std::atomic<int64_t> timing_last_run_ns{0};
  std::atomic<int64_t> t_first_worker_ns{0};
  std::atomic<int64_t> t_compute_done_ns{0};
  std::atomic<int64_t> t_signalled_ns{0};
  bool timing_active = false;
  uint64_t timing_experts = 0;
  uint64_t timing_bytes = 0;
};

struct StepTimingAccum {
  int64_t wake_ns = 0;
  int64_t groups_ns = 0;
  int64_t gil_ns = 0;
  int64_t precb_ns = 0;
  int64_t notify_ns = 0;
  int64_t seen_to_doorbell_ns = 0;
  int64_t done_store_ns = 0;
  int64_t last_seen_ns = 0;
  int64_t last_done_stored_ns = 0;
  int64_t compute_ns = 0;
  int64_t signal_ns = 0;
  int64_t wake_max_ns = 0;
  int64_t gil_max_ns = 0;
  int64_t precb_max_ns = 0;
  int64_t compute_max_ns = 0;
  int64_t signal_max_ns = 0;
  uint64_t tasks = 0;
  uint64_t experts = 0;
  uint64_t bytes = 0;
};

// One graph-stable DISK -> pinned staging task. The GPU copies num_rows and the
// LRU's layer-local source row ids into mapped-pinned control buffers before it
// rings the normal coordinator doorbell. The coordinator then faults/copies only
// those rows from the read-only FTW mappings into the pinned bank ring. A later
// captured H2D gather installs ring row i into the LRU-selected destination slot.
struct GpuFetchTask {
  CpuMoeExecutor* exec;
  int layer_id;
  int num_experts;
  int capacity;
  const int64_t* num_rows;
  const int32_t* row_ids;
  std::vector<uintptr_t> source_ptrs;
  std::vector<uintptr_t> staging_ptrs;
  std::vector<int64_t> row_bytes;
};

// Output-row tiling. Small enough to give every worker independent work even at
// batch size 1; large enough to amortize the atomic work-grab.
//
// Bandwidth notes (Sapphire Rapids 8480+, 13 cores): at bs=1 the two passes read
// every expert weight byte exactly once (each output row block is owned by one
// worker), and x stays hot in L1 across a route's rows. Decode batches group routes
// by expert below, keeping each expert tile hot while it is applied to every routed
// token. Synchronous prefill retains the original route-major tiling.
// One worker per *physical* core, pinned, is the sweet spot; SMT oversubscription
// thrashes the spin-barrier. Deferred (not worth it here / for this workload):
//   - AMX-bf16: a GEMM tile engine; decode is M=1 GEMV so tiles sit idle. It would
//     only pay off in a grouped/batched (dedup) path.
//   - NUMA: a single node is assumed. Multi-socket machines would split each
//     expert's K dimension per node (banks are already per-row contiguous).
constexpr int IBLK = 32;
constexpr int HBLK = 32;

// -------------------------------- Q4_0 (W4A8) --------------------------------
// Native GGUF Q4_0 experts (gemma4 GGUF): per-32 block = fp16 scale d + 16 packed
// bytes; byte j holds element j in its low nibble and j+16 in its high nibble, so a
// block's storage order is [lo0..lo15, hi0..hi15] and w = (nibble - 8) * d. Matches
// the reference dequant (models/gguf/dequant.py) and the packed banks the GPU offload
// path streams.
//
// llama.cpp ggml_vec_dot_q4_0_q8_0: W4A8. The activation is pre-quantized to Q8_0
// (per-32-block int8 ``aq`` + fp32 scale ``asb``); each block unpacks its 16 bytes to
// 32 int8 weights in [-8,7] (bytes_from_nibbles_32: low nibbles -> elems 0..15, high
// -> 16..31) and runs an integer block dot -- VPDPBUSD (AVX-VNNI) or VPMADDUBSW+VPMADDWD
// (AVX2) with the ggml sign trick |w|*(sign(w)*a)=w*a, or a scalar int loop -- then
// scales the block sum by wd*xd in fp32. No fp weight dequant / shuffle chain. The GPU
// offload path (ggml_moe_a8_vec / MMVQ) is also W4A8, so cpu and hybrid stay close.
using q4dot_fn = float (*)(const uint8_t*, const int8_t*, const float*, int);

float q4_0_dot_i8_scalar(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const uint8_t* q = blk + 2;  // 16 nibble bytes
    const int8_t* a = aq + (size_t)b * 32;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((int)(q[j] & 0x0F) - 8) * (int)a[j];       // elem j
      isum += ((int)(q[j] >> 4) - 8) * (int)a[16 + j];    // elem 16+j
    }
    acc += fp16_to_f32(dh) * asb[b] * (float)isum;
  }
  return acc;
}

#if CPU_MOE_X86
// fp16 block scale -> fp32 via HW F16C (single value in lane 0).
__attribute__((target("f16c")))
static inline float q4_scale(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// Unpack one Q4_0 block's 16 bytes -> 32 int8 weights in [-8,7] (elems 0..15 = low
// nibbles, 16..31 = high nibbles). ``eight`` = _mm256_set1_epi8(8).
__attribute__((target("avx2")))
static inline __m256i q4_unpack32(const uint8_t* blk, __m128i mask, __m256i eight) {
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i lo = _mm_and_si128(qb, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
  return _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
}

// AVX2 W4A8 (llama.cpp non-VNNI mul_sum_i8_pairs): integer block dot via VPMADDUBSW +
// VPMADDWD (sign trick), scaled by wd*xd. |aw*sa| pair sums <= 8*127*2 < 32767 -> no
// int16 saturation. This is the fast path on AVX2 CPUs without AVX-VNNI (and the
// avx512-tier fallback, since the block dot is 256-bit either way).
__attribute__((target("avx2,fma,f16c")))
float q4_0_dot_i8_avx2(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);              // |wq|          (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);               // sign(wq) * a  (signed operand)
    __m256i d32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), ones16);  // 8 int32
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}

// AVX-VNNI W4A8: one VPDPBUSD per block (the fast path on modern CPUs).
__attribute__((target("avx2,avxvnni,fma,f16c")))
float q4_0_dot_i8_vnni(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);   // |wq|            (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);    // sign(wq) * a    (signed operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    // All 32 elems of the block share wd*xd; distribute over di's 8 partial sums and
    // reduce at the end (equivalent to scale * block_total).
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}
#endif  // CPU_MOE_X86

// All tiers are W4A8 (int8 activations pre-quantized to Q8_0). AVX-VNNI is orthogonal to
// the ISA tier (gated by cpu_has_avxvnni() / FREETOKEN_CPU_MOE_NO_VNNI), so it wins when
// present; otherwise the 256-bit VPMADDUBSW kernel covers both the avx2 and avx512 tiers.
q4dot_fn select_q4dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (cpu_has_avxvnni()) return q4_0_dot_i8_vnni;
  if (t >= ISA_AVX2) return q4_0_dot_i8_avx2;
#endif
  (void)t;
  return q4_0_dot_i8_scalar;
}

enum WFmt { WF_BF16 = 0, WF_NVFP4 = 1, WF_MXFP4 = 2, WF_DSFP4 = 3, WF_Q4_0 = 4 };

// Each ctor pointer arg is the address of a CPU int64 array of length
// num_layers (one base address per layer, built by cpu_executor.py's
// _make_table), not a single flat bank. tbl_at resolves
// tbl[layer_id] once per task/pass; a null table (bank unused by this fmt, ptr
// arg 0) resolves to nullptr without dereferencing.
inline const void* tbl_at(const uint64_t* tbl, int layer_id) {
  return tbl ? reinterpret_cast<const void*>(tbl[layer_id]) : nullptr;
}

struct CpuMoeExecutor {
  int num_threads;
  int num_layers, num_experts, top_k;
  int H, I;
  int act, apply_on_input;
  int fmt;                // WFmt
  bool needs_di = false;  // pre-deinterleave activations to fp32 (nvfp4/ds_fp4)
  // Per-layer pointer tables (one base address per layer, see tbl_at). gate_up_tbl
  // doubles as the bf16 gate_up table and the nvfp4/mxfp4/q4_0/ds_fp4 packed-gate_up
  // table (down_tbl likewise for down); which reinterpretation applies is picked by
  // fmt at each resolve site (see gemm1_dot/gemm2_dot and the format-specific passes).
  const uint64_t* gate_up_tbl;   // bf16: [E,2I,H] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* down_tbl;      // bf16: [E,H,I] rows; else: packed e2m1/mxfp4-blocks
  const uint64_t* gu_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,2I,*] block scales
  const uint64_t* gu_global_tbl; // nvfp4: [E,2I] fp16 row globals
  const uint64_t* dn_scale_tbl;  // nvfp4/mxfp4/ds_fp4: [E,H,*] block scales
  const uint64_t* dn_global_tbl; // nvfp4: [E,H] fp16 row globals
  const uint64_t* gu_bias_tbl;   // mxfp4: [E,2I] bf16 biases
  const uint64_t* dn_bias_tbl;   // mxfp4: [E,H] bf16 biases
  float swiglu_alpha;
  float swiglu_limit;          // +inf == no clamp
  dot_fn dot;
  nvdot_fn nvdot;
  nvi8dot_fn nvi8dot = nullptr;  // AVX-VNNI W4A8 nvfp4 dot (nullptr -> use fp32 nvdot)
  nvi8batch_rows_fn nvi8batch_rows = nullptr;  // R_w weight rows x M activations
  const char* nvi8batch_name = "scalar";
  bool use_vnni = false;         // nvfp4 + AVX-VNNI: decode via int8 VPDPBUSD (W4A8)
  bool use_q4a8 = false;       // q4_0: always W4A8 (llama.cpp Q4_0 x Q8_0); int8 pre-quant
  dsdot_fn dsdot;
  mxgemv_fn mxgemv;
  q4dot_fn q4dot;
  // ds_fp4: the caller already FP8-round-tripped the input activations on the GPU
  // (same reference grid), so submit() must not repeat it on the host-callback
  // thread. That scalar per-element pass is single-threaded ON THE DECODE CRITICAL
  // PATH (~0.3ms/layer at H=4096, every worker and the GPU waiting on it); moving
  // it to a captured GPU elementwise kernel removes it while keeping the official
  // W4A8 numerics bit-exact. Set via set_input_prequant (see cpu_executor.py).
  bool input_prequant = false;
  std::atomic<bool> task_timing_enabled{false};
  bool timed_worker_mode = false;
  // Q4_0 packed-row byte strides (H/32*18 for gate_up over K=H, I/32*18 for down over K=I).
  int q4_gu_row_bytes = 0, q4_dn_row_bytes = 0;
  float e2m1_lut[16];
  float e4m3_lut[256];
  float e8m0_lut[256];         // mxfp4 block scale: 2^(s-127), s clamped to [0,254]
  const char* isa;

  std::vector<bf16_t> g_scratch;   // [max_tokens * top_k * I] intermediate
  // Decode-only expert grouping. grouped_routes is stable within each expert, so
  // the final route reduction can retain the original top-k accumulation order.
  std::vector<int> expert_offsets;    // [num_experts + 1]
  std::vector<int> expert_cursor;     // submit-time fill cursor, allocation-stable
  std::vector<int> grouped_routes;    // flattened route ids (token * top_k + k)
  std::vector<int> distinct_experts;  // experts with at least one valid route
  std::vector<float> route_y_scratch; // [num_tokens * top_k, H] unweighted down outputs
  // NVFP4 prefill-only bounded workspace. The bf16 input gather buffer is reused
  // in place for W4A8 activations after gathering. All
  // route-indexed arrays are capped by prefill_capacity * top_k and allocated once
  // by setup_prefill_batch(), outside the per-layer hot path.
  int prefill_capacity = 0;
  std::vector<bf16_t> prefill_x_scratch;  // [routes,H], compacted to int8 in-place
  std::vector<bf16_t> prefill_g_scratch;  // [routes,I] activated intermediate
  std::vector<int8_t> prefill_gi8_scratch;  // [routes,I] W4A8 down input
  std::vector<bf16_t> prefill_y_scratch;  // [routes,H] per-route down result
  std::vector<float> prefill_x_scale_scratch;  // [routes,H/16]
  std::vector<float> prefill_g_scale_scratch;  // [routes,I/16]
  // [R_w,routes] projection tiles. Two buffers retain gate rows while the
  // corresponding up rows are computed, then the down projection reuses gate.
  std::vector<float> prefill_gate_scratch, prefill_up_scratch;
  std::vector<int> route_to_group;  // original token-major route -> grouped row
  std::vector<bf16_t> xq_scratch;  // [max_tokens * H] ds_fp4 fp8-roundtripped input
  // ds_fp4 activations pre-deinterleaved to fp32 (even/odd K) for the row-major dot.
  std::vector<float> xe_scratch, xo_scratch;  // [max_tokens * H/2]   (input)
  std::vector<float> ge_scratch, go_scratch;  // [max_tokens*top_k*I/2] (intermediate)
  // AVX-VNNI W4A8: per-16-block int8 activations [even(8),odd(8)] + per-block scale.
  std::vector<int8_t> xi8_scratch, gi8_scratch;  // [max_tokens*H], [max_tokens*top_k*I]
  std::vector<float> xas_scratch, gas_scratch;   // [max_tokens*H/16], [..*top_k*I/16]
  std::string isa_str;

  std::vector<std::thread> workers;
  std::mutex task_mtx;
  std::condition_variable task_cv;
  std::mutex sync_mtx;
  std::condition_variable sync_cv;

  bool stop = false;
  uint64_t cur_gen = 0;
  MoeTask* cur_task = nullptr;
  std::atomic<uint64_t> submitted{0};
  std::atomic<uint64_t> completed{0};

  std::atomic<int64_t> p1_next{0};
  std::atomic<int64_t> p2_next{0};
  std::atomic<int64_t> p3_next{0};  // grouped decode's stable per-token reduction
  std::atomic<int64_t> prt_next{0};  // ds_fp4 intermediate fp8 round-trip phase
  int64_t p1_total = 0, p2_total = 0, p3_total = 0, prt_total = 0;
  int n_iblk = 0, n_hblk = 0;
  std::atomic<int> done_count{0};
  std::atomic<int> bar_count{0};
  std::atomic<int> bar_sense{0};

  std::mutex step_timing_mtx;
  std::vector<StepTimingAccum> step_timing;
  std::atomic<uint64_t> step_timing_inflight{0};
  std::atomic<int> pre_run_callback_mode{0};  // 0 = before notify, 1 = after notify

  std::vector<MoeTask*> owned_tasks;  // persistent task descriptors (graph-stable)
  std::vector<GpuFetchTask*> owned_gpufetch_tasks;
  std::vector<int> core_ids;          // worker tid -> logical CPU to pin to (may be empty)
  // Optional Python pre-run hook. DISK banks use it to issue mmap advice or wait for
  // UFFD row fills after routing D2H and before workers first read expert weights.
  pybind11::function pre_run_callback;

  // ---- Flag-based GPU<->CPU handshake (replaces the per-layer cudaLaunchHostFunc pair) ----
  // A tiny GPU kernel bumps ready_flags[slot] at submit; this coordinator thread busy-polls
  // it, runs the slot's task on the worker pool, and sets done_flags[slot], which a GPU
  // spin-wait kernel polls at sync. This removes the ~2x30-50us host-func dispatch round
  // trips per MoE layer per decode step that otherwise idle the GPU (~6 ms/step on a
  // 75-layer model). One slot per (layer, decode batch size) pair -- the Python side
  // allocates slots as tasks are created. Flags live in mapped-pinned host memory (UVA:
  // the same pointers are used by the GPU kernels and by this thread).
  std::thread coord_thread;
  std::atomic<bool> coord_stop{false};
  volatile int64_t* ready_flags = nullptr;  // GPU increments, this thread polls
  volatile int64_t* done_flags = nullptr;   // this thread sets, GPU spin-waits
  int coord_num_slots = 0;
  std::vector<MoeTask*> flag_task;           // slot -> task (registered lazily)
  std::vector<GpuFetchTask*> flag_gpufetch_task;  // slot -> DISK staging task
  std::vector<int64_t> flag_served;          // slot -> completed dispatch count (tests/debug)
  std::mutex flag_task_mtx;
  std::atomic<int64_t> gpufetch_fills{0};
  std::atomic<int64_t> gpufetch_steps{0};
  std::atomic<int64_t> gpufetch_fill_ns{0};
  std::atomic<int64_t> gpufetch_error{0};

  // Portable ordering for the flag handshake: "ready observed => the DMA'd inputs that
  // preceded the bump are visible" and "y stores are visible before done". Plain
  // volatile loads lean on x86 TSO; acquire/release makes it hold on aarch64 too
  // (GH200/Jetson) at zero x86 cost. (MSVC branch is x86-only today: compiler barrier
  // + TSO.)
  static int64_t flag_load_acquire(const volatile int64_t* p) {
#if defined(_MSC_VER)
    const int64_t v = *p;
    _ReadWriteBarrier();
    return v;
#else
    return __atomic_load_n(const_cast<const int64_t*>(p), __ATOMIC_ACQUIRE);
#endif
  }

  static void flag_store_release(volatile int64_t* p, int64_t v) {
#if defined(_MSC_VER)
    _ReadWriteBarrier();
    *p = v;
#else
    __atomic_store_n(const_cast<int64_t*>(p), v, __ATOMIC_RELEASE);
#endif
  }

  CpuMoeExecutor(int num_threads_, int num_layers_, int num_experts_, int top_k_,
                 int hidden_size, int inter_size, int max_tokens, int activation_id,
                 int apply_router_weight_on_input, int weight_format,
                 uintptr_t gate_up_ptr, uintptr_t down_ptr, uintptr_t gate_up_scale_ptr,
                 uintptr_t gate_up_global_ptr, uintptr_t down_scale_ptr,
                 uintptr_t down_global_ptr, uintptr_t gate_up_bias_ptr,
                 uintptr_t down_bias_ptr, double swiglu_alpha_, double swiglu_limit_,
                 std::vector<int> core_ids_, bool task_timing_enabled_ = false)
      : num_threads(num_threads_ > 0 ? num_threads_ : 1),
        num_layers(num_layers_),
        num_experts(num_experts_),
        top_k(top_k_),
        H(hidden_size),
        I(inter_size),
        act(activation_id),
        apply_on_input(apply_router_weight_on_input),
        fmt(weight_format),
        gate_up_tbl(reinterpret_cast<const uint64_t*>(gate_up_ptr)),
        down_tbl(reinterpret_cast<const uint64_t*>(down_ptr)),
        gu_scale_tbl(reinterpret_cast<const uint64_t*>(gate_up_scale_ptr)),
        gu_global_tbl(reinterpret_cast<const uint64_t*>(gate_up_global_ptr)),
        dn_scale_tbl(reinterpret_cast<const uint64_t*>(down_scale_ptr)),
        dn_global_tbl(reinterpret_cast<const uint64_t*>(down_global_ptr)),
        gu_bias_tbl(reinterpret_cast<const uint64_t*>(gate_up_bias_ptr)),
        dn_bias_tbl(reinterpret_cast<const uint64_t*>(down_bias_ptr)),
        swiglu_alpha(static_cast<float>(swiglu_alpha_)),
        swiglu_limit(static_cast<float>(swiglu_limit_)),
        task_timing_enabled(task_timing_enabled_),
        timed_worker_mode(task_timing_enabled_),
        step_timing(static_cast<size_t>(num_layers_)),
        core_ids(std::move(core_ids_)) {
    DotChoice c = select_dot();
    dot = c.fn;
    nvdot = select_nvdot();
    const Nvi8Tier nvi8_tier = detect_nvi8_tier();
    const Nvi8Dispatch nvi8_dispatch = select_nvi8_dispatch(nvi8_tier);
    nvi8batch_rows = nvi8_dispatch.batch_rows;
    nvi8batch_name = nvi8_dispatch.batch_name;
    dsdot = select_dsdot();
    mxgemv = select_mxgemv();
    q4dot = select_q4dot();
    if (weight_format == WF_Q4_0) {
      if (H % 32 != 0 || I % 32 != 0)
        throw std::runtime_error("Q4_0 CPU MoE requires H and I to be multiples of 32");
      q4_gu_row_bytes = (H / 32) * 18;  // K = H (gate_up rows)
      q4_dn_row_bytes = (I / 32) * 18;  // K = I (down rows)
    }
    isa = c.name;
    // nvfp4 (AVX-VNNI only): W4A8 int8 decode when the CPU supports it. q4_0 is always
    // W4A8 (activations pre-quantized to Q8_0); select_q4dot picks VPDPBUSD / VPMADDUBSW
    // / scalar for the tier, so the tag reflects which of those q4dot resolved to.
    nvi8dot = nvi8_dispatch.dot;
    use_vnni = (weight_format == WF_NVFP4) && (nvi8dot != nullptr);
    use_q4a8 = (weight_format == WF_Q4_0);
    const char* q4tag = use_q4a8 ? (cpu_has_avxvnni() ? "+vnni(q4_0-w4a8)" : "+q4_0-w4a8") : "";
    const char* vnni_tag = nvi8_tier == NVI8_AVX512VNNI
                               ? "+avx512vnni(nvfp4-w4a8)"
                               : "+vnni(nvfp4-w4a8)";
    isa_str = std::string(c.name) + (use_vnni ? vnni_tag : "") + q4tag;
    isa = isa_str.c_str();
    for (int i = 0; i < 16; ++i) e2m1_lut[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) e4m3_lut[i] = e4m3_decode((uint8_t)i);
    // e8m0 (mxfp4 block scale) = 2^(s-127); the GPU GEMV clamps s to [0,254].
    for (int i = 0; i < 256; ++i) e8m0_lut[i] = std::ldexp(1.0f, std::min(i, 254) - 127);
    g_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
    expert_offsets.assign(static_cast<size_t>(num_experts) + 1, 0);
    expert_cursor.assign(static_cast<size_t>(num_experts), 0);
    grouped_routes.reserve(static_cast<size_t>(max_tokens) * top_k);
    distinct_experts.reserve(std::min(num_experts, max_tokens * top_k));
    route_y_scratch.assign(static_cast<size_t>(max_tokens) * top_k * H, 0.0f);
    // Row-major fp4 (nvfp4/ds_fp4) pre-deinterleaves activations to fp32 even/odd.
    needs_di = (fmt == WF_NVFP4 || fmt == WF_DSFP4);
    if (needs_di) {
      if (fmt == WF_DSFP4) xq_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xe_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      xo_scratch.assign(static_cast<size_t>(max_tokens) * (H / 2), 0);
      ge_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      go_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 2), 0);
      if (use_vnni) {
        xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
        xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 16), 0);
        gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
        gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 16), 0);
      }
    }
    // q4_0 W4A8: per-32-block Q8_0 activations (int8 + fp32 scale) for input + intermediate.
    if (use_q4a8) {
      xi8_scratch.assign(static_cast<size_t>(max_tokens) * H, 0);
      xas_scratch.assign(static_cast<size_t>(max_tokens) * (H / 32), 0);
      gi8_scratch.assign(static_cast<size_t>(max_tokens) * top_k * I, 0);
      gas_scratch.assign(static_cast<size_t>(max_tokens) * top_k * (I / 32), 0);
    }
    for (int t = 0; t < num_threads; ++t) {
      if (timed_worker_mode)
        workers.emplace_back([this, t] { timed_worker_loop(t); });
      else
        workers.emplace_back([this, t] { worker_loop(t); });
    }
  }

  // Quantize a bf16 activation row to Q8_0 (llama.cpp): per-32-block symmetric int8 in
  // natural order + one fp32 scale (amax/127) per block. Done once per token/route,
  // amortized over every output row that the q4_0 W4A8 GEMV reads. K % 32 == 0.
  void quant_q8_0(const bf16_t* x, int K, int8_t* aq, float* asb) {
    const int nb = K / 32;
    for (int b = 0; b < nb; ++b) {
      const bf16_t* xb = x + (size_t)b * 32;
      float xf[32], amax = 0.0f;
      for (int j = 0; j < 32; ++j) {
        xf[j] = bf16_to_f32(xb[j]);
        amax = std::max(amax, std::fabs(xf[j]));
      }
      const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = d;
      const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
      int8_t* o = aq + (size_t)b * 32;
      for (int j = 0; j < 32; ++j)
        o[j] = (int8_t)std::max(-127, std::min(127, (int)std::lround(xf[j] * inv)));
    }
  }

  // Quantize the pre-deinterleaved fp32 even/odd activations to per-16-block int8 in the
  // [even(8),odd(8)] layout the VNNI dot expects. Done once per token/route (amortized over
  // every output row), so a scalar pass is fine relative to the GEMV.
  void quant_i8_pg16(const float* xe, const float* xo, int K, int8_t* asi8, float* asb) {
    const int nb = K / 16;
    for (int b = 0; b < nb; ++b) {
      const float* xeb = xe + (size_t)b * 8;
      const float* xob = xo + (size_t)b * 8;
      float amax = 0.0f;
      for (int j = 0; j < 8; ++j)
        amax = std::max(amax, std::max(std::fabs(xeb[j]), std::fabs(xob[j])));
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = s;
      const float inv = 1.0f / s;
      int8_t* ae = asi8 + (size_t)b * 16;
      for (int j = 0; j < 8; ++j) {
        int qe = (int)std::lround(xeb[j] * inv), qo = (int)std::lround(xob[j] * inv);
        ae[j] = (int8_t)std::max(-127, std::min(127, qe));
        ae[8 + j] = (int8_t)std::max(-127, std::min(127, qo));
      }
    }
  }

  // Quantize an interleaved bf16 row directly into the W4A8 [even(8),odd(8)]
  // block layout. Reading each 16-value block into locals before writing makes
  // in-place compaction from bf16 to int8 safe.
  void quant_i8_bf16_pg16(const bf16_t* x, int K, int8_t* asi8, float* asb) {
    const int nb = K / 16;
    for (int b = 0; b < nb; ++b) {
      float xf[16], amax = 0.0f;
      for (int j = 0; j < 16; ++j) {
        xf[j] = bf16_to_f32(x[(size_t)b * 16 + j]);
        amax = std::max(amax, std::fabs(xf[j]));
      }
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
      asb[b] = s;
      const float inv = 1.0f / s;
      int8_t* o = asi8 + (size_t)b * 16;
      for (int j = 0; j < 8; ++j) {
        const int qe = (int)std::lround(xf[2 * j] * inv);
        const int qo = (int)std::lround(xf[2 * j + 1] * inv);
        o[j] = (int8_t)std::max(-127, std::min(127, qe));
        o[8 + j] = (int8_t)std::max(-127, std::min(127, qo));
      }
    }
  }

  bool setup_prefill_batch(int max_prefill_tokens) {
    if (fmt != WF_NVFP4 || nvi8batch_rows == nullptr || max_prefill_tokens <= 0 ||
        H % 16 != 0 || I % 16 != 0) {
      return false;
    }
    if (prefill_capacity == max_prefill_tokens) return true;
    if (prefill_capacity != 0) return false;
    try {
      const size_t routes = static_cast<size_t>(max_prefill_tokens) * top_k;
      std::vector<bf16_t> x(routes * H);
      std::vector<bf16_t> g(routes * I);
      std::vector<int8_t> gi8(routes * I);
      std::vector<bf16_t> y(routes * H);
      std::vector<float> xs(routes * (H / 16));
      std::vector<float> gs(routes * (I / 16));
      std::vector<float> gate(routes * kNvi8WeightRows);
      std::vector<float> up(routes * kNvi8WeightRows);
      std::vector<int> route_map(routes, -1);
      grouped_routes.reserve(routes);
      prefill_x_scratch.swap(x);
      prefill_g_scratch.swap(g);
      prefill_gi8_scratch.swap(gi8);
      prefill_y_scratch.swap(y);
      prefill_x_scale_scratch.swap(xs);
      prefill_g_scale_scratch.swap(gs);
      prefill_gate_scratch.swap(gate);
      prefill_up_scratch.swap(up);
      route_to_group.swap(route_map);
      prefill_capacity = max_prefill_tokens;
      return true;
    } catch (const std::exception&) {
      prefill_capacity = 0;
      return false;
    }
  }

  int64_t prefill_batch_buffer_bytes() const {
    return static_cast<int64_t>(
        prefill_x_scratch.size() * sizeof(bf16_t) +
        prefill_g_scratch.size() * sizeof(bf16_t) +
        prefill_gi8_scratch.size() * sizeof(int8_t) +
        prefill_y_scratch.size() * sizeof(bf16_t) +
        prefill_x_scale_scratch.size() * sizeof(float) +
        prefill_g_scale_scratch.size() * sizeof(float) +
        prefill_gate_scratch.size() * sizeof(float) +
        prefill_up_scratch.size() * sizeof(float) +
        route_to_group.size() * sizeof(int) +
        static_cast<size_t>(prefill_capacity) * top_k * sizeof(int));
  }

  // gate_up output row `row` (in [0, 2I)) dotted with activation over K = H. ``e`` is
  // the layer-local expert row (0..num_experts); the layer bases (already resolved
  // once per task/pass by the caller via tbl_at) pick the layer's own tensors.
  // bf16 uses the interleaved bf16 row; nvfp4 uses the pre-split fp32 even/odd halves
  // (or, with AVX-VNNI, the per-16-block int8 activations).
  inline float gemm1_dot(const bf16_t* gate_up_l, const uint8_t* gu_packed_l,
                         const uint8_t* gu_scale_l, const uint16_t* gu_global_l, int e, int row,
                         const bf16_t* x, const float* xe, const float* xo, const int8_t* xi8,
                         const float* xas) {
    if (fmt == WF_BF16) {
      const bf16_t* w = gate_up_l + ((size_t)e * (2 * I) + row) * H;
      return dot(w, x, H);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w =
          gu_packed_l + ((size_t)e * (2 * I) + row) * (size_t)q4_gu_row_bytes;
      return q4dot(w, xi8, xas, H);  // W4A8: int8 activations (Q8_0), scale in xas
    }
    const size_t r = (size_t)e * (2 * I) + row;
    if (use_vnni)
      return nvi8dot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                     fp16_to_f32(gu_global_l[r]), xi8, H, e4m3_lut, xas);
    return nvdot(gu_packed_l + r * (size_t)(H / 2), gu_scale_l + r * (size_t)(H / 16),
                 fp16_to_f32(gu_global_l[r]), xe, xo, H, e2m1_lut, e4m3_lut);
  }

  // down output row `row` (in [0, H)) dotted with the intermediate over K = I. Same
  // layer-local-base convention as gemm1_dot.
  inline float gemm2_dot(const bf16_t* down_l, const uint8_t* dn_packed_l,
                         const uint8_t* dn_scale_l, const uint16_t* dn_global_l, int e, int row,
                         const bf16_t* g, const float* ge, const float* go, const int8_t* gi8,
                         const float* gas) {
    if (fmt == WF_BF16) {
      const bf16_t* w = down_l + ((size_t)e * H + row) * I;
      return dot(w, g, I);
    }
    if (fmt == WF_Q4_0) {
      const uint8_t* w = dn_packed_l + ((size_t)e * H + row) * (size_t)q4_dn_row_bytes;
      return q4dot(w, gi8, gas, I);  // W4A8: int8 activations (Q8_0), scale in gas
    }
    const size_t r = (size_t)e * H + row;
    if (use_vnni)
      return nvi8dot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                     fp16_to_f32(dn_global_l[r]), gi8, I, e4m3_lut, gas);
    return nvdot(dn_packed_l + r * (size_t)(I / 2), dn_scale_l + r * (size_t)(I / 16),
                 fp16_to_f32(dn_global_l[r]), ge, go, I, e2m1_lut, e4m3_lut);
  }

  void pin_self(int tid) {
#if CPU_MOE_HAS_AFFINITY
    if (core_ids.empty()) return;
    const int cpu = core_ids[tid % static_cast<int>(core_ids.size())];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)tid;
#endif
  }

  ~CpuMoeExecutor() {
    coord_stop.store(true);
    if (coord_thread.joinable()) coord_thread.join();
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      stop = true;
    }
    task_cv.notify_all();
    for (auto& th : workers)
      if (th.joinable()) th.join();
    for (MoeTask* t : owned_tasks) delete t;
    for (GpuFetchTask* t : owned_gpufetch_tasks) delete t;
  }

  uintptr_t create_task(int layer_id, int num_tokens, uintptr_t x_ptr,
                        uintptr_t ids_ptr, uintptr_t w_ptr, uintptr_t y_ptr) {
    MoeTask* t = new MoeTask{this,
                             layer_id,
                             num_tokens,
                             true,
                             false,
                             reinterpret_cast<const bf16_t*>(x_ptr),
                             reinterpret_cast<const int32_t*>(ids_ptr),
                             reinterpret_cast<const float*>(w_ptr),
                             reinterpret_cast<bf16_t*>(y_ptr)};
    owned_tasks.push_back(t);
    return reinterpret_cast<uintptr_t>(t);
  }

  uintptr_t create_gpufetch_task(
      int layer_id, int capacity, uintptr_t num_rows_ptr, uintptr_t row_ids_ptr,
      std::vector<uintptr_t> source_ptrs, std::vector<uintptr_t> staging_ptrs,
      std::vector<int64_t> row_bytes) {
    if (capacity <= 0 || source_ptrs.empty() || source_ptrs.size() != staging_ptrs.size() ||
        source_ptrs.size() != row_bytes.size())
      throw std::invalid_argument("invalid GPU-fetch staging task geometry");
    GpuFetchTask* t = new GpuFetchTask{
        this, layer_id, num_experts, capacity,
        reinterpret_cast<const int64_t*>(num_rows_ptr),
        reinterpret_cast<const int32_t*>(row_ids_ptr),
        std::move(source_ptrs), std::move(staging_ptrs), std::move(row_bytes)};
    owned_gpufetch_tasks.push_back(t);
    return reinterpret_cast<uintptr_t>(t);
  }

  void run_gpufetch(GpuFetchTask* t) {
    const auto begin = std::chrono::steady_clock::now();
    const int64_t count = *t->num_rows;
    if (count < 0 || count > t->capacity) {
      gpufetch_error.store(1, std::memory_order_release);
      return;
    }
    for (int64_t i = 0; i < count; ++i) {
      const int row = t->row_ids[i];
      if (row < 0 || row >= t->num_experts) {
        gpufetch_error.store(2, std::memory_order_release);
        return;
      }
      for (size_t b = 0; b < t->row_bytes.size(); ++b) {
        const size_t bytes = static_cast<size_t>(t->row_bytes[b]);
        std::memcpy(
            reinterpret_cast<void*>(t->staging_ptrs[b] + static_cast<size_t>(i) * bytes),
            reinterpret_cast<const void*>(
                t->source_ptrs[b] + static_cast<size_t>(row) * bytes),
            bytes);
      }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count();
    gpufetch_fills.fetch_add(count, std::memory_order_relaxed);
    gpufetch_steps.fetch_add(1, std::memory_order_relaxed);
    gpufetch_fill_ns.fetch_add(elapsed, std::memory_order_relaxed);
  }

  const char* isa_name() const { return isa; }
  const char* prefill_batch_kernel_name() const { return nvi8batch_name; }

  void barrier(int& local_sense) {
    local_sense ^= 1;
    if (bar_count.fetch_add(1) + 1 == num_threads) {
      bar_count.store(0);
      bar_sense.store(local_sense);
    } else {
      while (bar_sense.load() != local_sense) {
#if CPU_MOE_X86
        _mm_pause();
#endif
      }
    }
  }

  void do_pass1_route(const MoeTask* t, int route, int ib) {
    if (fmt == WF_MXFP4) {
      do_pass1_mxfp4_route(t, route, ib);
      return;
    }
    if (fmt == WF_DSFP4) {
      do_pass1_dsfp4_route(t, route, ib);
      return;
    }
    const int k = route % top_k;
    const int tok = route / top_k;
    const int e = t->ids[route];
    if (e < 0 || e >= num_experts) return;
    const float w_in = apply_on_input ? t->w[static_cast<size_t>(tok) * top_k + k] : 1.0f;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* gate_up_l = reinterpret_cast<const bf16_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const uint16_t* gu_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(gu_global_tbl, t->layer_id));
    const bf16_t* x_row = t->x + (size_t)tok * H;
    const float* xe = needs_di ? xe_scratch.data() + (size_t)tok * (H / 2) : nullptr;
    const float* xo = needs_di ? xo_scratch.data() + (size_t)tok * (H / 2) : nullptr;
    const int8_t* xi8 =
        (use_vnni || use_q4a8) ? xi8_scratch.data() + (size_t)tok * H : nullptr;
    const float* xas = use_vnni ? xas_scratch.data() + (size_t)tok * (H / 16)
                     : use_q4a8 ? xas_scratch.data() + (size_t)tok * (H / 32)
                                  : nullptr;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const bool swigluoai = act == ACT_SWIGLUOAI;
    const bool clamped_silu = act == ACT_CLAMPED_SILU;
    const float lim = swiglu_limit, alpha = swiglu_alpha;
    for (int i = i0; i < i1; ++i) {
      // gate = row i, up = row I+i
      float gate =
          gemm1_dot(gate_up_l, gu_packed_l, gu_scale_l, gu_global_l, e, i, x_row, xe, xo, xi8, xas) * w_in;
      float up = gemm1_dot(gate_up_l, gu_packed_l, gu_scale_l, gu_global_l, e, I + i, x_row,
                           xe, xo, xi8, xas) * w_in;
      if (swigluoai || clamped_silu) {
        // Both variants clamp gate/up. swigluoai uses sigmoid(alpha*gate)*(up+1),
        // while clamped_silu uses sigmoid(gate)*up (lim == +inf means no clamp).
        if (gate > lim) gate = lim;
        if (up > lim) up = lim;
        else if (up < -lim) up = -lim;
        const float glu = gate / (1.0f + std::exp(-gate * (swigluoai ? alpha : 1.0f)));
        g_row[i] = f32_to_bf16(glu * (swigluoai ? up + 1.0f : up));
      } else {
        g_row[i] = f32_to_bf16(act_apply(act, gate) * up);
      }
    }
  }

  void do_pass1(const MoeTask* t, int64_t p) {
    const int ib = static_cast<int>(p % n_iblk);
    const int route = static_cast<int>(p / n_iblk);
    do_pass1_route(t, route, ib);
  }

  void do_pass2(const MoeTask* t, int64_t p) {
    if (fmt == WF_MXFP4) {
      do_pass2_mxfp4(t, p);
      return;
    }
    if (fmt == WF_DSFP4) {
      do_pass2_dsfp4(t, p);
      return;
    }
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const bf16_t* down_l = reinterpret_cast<const bf16_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const uint16_t* dn_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(dn_global_tbl, t->layer_id));
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
        if (e < 0 || e >= num_experts) continue;
        const float w_out = apply_on_input ? 1.0f : t->w[static_cast<size_t>(tok) * top_k + k];
        const size_t gr = (size_t)tok * top_k + k;
        const bf16_t* g_row = g_scratch.data() + gr * I;
        const float* ge = needs_di ? ge_scratch.data() + gr * (I / 2) : nullptr;
        const float* go = needs_di ? go_scratch.data() + gr * (I / 2) : nullptr;
        const int8_t* gi8 = (use_vnni || use_q4a8) ? gi8_scratch.data() + gr * I : nullptr;
        const float* gas = use_vnni ? gas_scratch.data() + gr * (I / 16)
                         : use_q4a8 ? gas_scratch.data() + gr * (I / 32)
                                      : nullptr;
        acc += gemm2_dot(down_l, dn_packed_l, dn_scale_l, dn_global_l, e, h, g_row, ge, go, gi8,
                         gas) * w_out;
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  // ----------------------------- mxfp4 (gpt-oss) -----------------------------
  // Transposed split-K layout (N innermost), so the GEMV streams K and accumulates
  // a contiguous block of N output columns -> cache-line-efficient, no repack, no
  // extra host memory. Pass 1 fuses gate_up + clamped-swiglu(+bias); pass 2 fuses
  // down(+bias) * router-weight, summed over the token's routes.
  //
  // Dequant: w = E2M1[code] * 2^(e8m0_scale - 127); two codes per byte (low nibble
  // first), one e8m0 scale per 32 contiguous K. Matches kernel/triton/mxfp4_moe.py.

  void do_pass1_mxfp4_route(const MoeTask* t, int route, int ib) {
    const int k = route % top_k;
    const int tok = route / top_k;
    const int e = t->ids[route];
    if (e < 0 || e >= num_experts) return;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const bf16_t* gu_bias_l = reinterpret_cast<const bf16_t*>(tbl_at(gu_bias_tbl, t->layer_id));
    const int N2 = 2 * I;            // gate_up output width (gate/up interleaved)
    const int Hh = H / 2;            // packed-K rows
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const int nunit = i1 - i0;       // intermediate units owned by this tile
    const int col0 = 2 * i0;         // first gate_up column
    const int ncol = 2 * nunit;      // gate_up columns owned by this tile
    const bf16_t* x_row = t->x + (size_t)tok * H;
    const uint8_t* blk_e = gu_packed_l + (size_t)e * Hh * N2;
    const uint8_t* scl_e = gu_scale_l + (size_t)e * (size_t)(H / 32) * N2;
    float gu[2 * IBLK];
    mxgemv(gu, blk_e + col0, scl_e + col0, x_row, Hh, N2, ncol, e2m1_lut, e8m0_lut);
    const bf16_t* bias_e = gu_bias_l + (size_t)e * N2 + col0;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const float lim = swiglu_limit, alpha = swiglu_alpha;
    for (int j = 0; j < nunit; ++j) {
      float gate = gu[2 * j] + bf16_to_f32(bias_e[2 * j]);
      float up = gu[2 * j + 1] + bf16_to_f32(bias_e[2 * j + 1]);
      if (gate > lim) gate = lim;
      if (up > lim) up = lim;
      else if (up < -lim) up = -lim;
      const float glu = gate / (1.0f + std::exp(-gate * alpha));  // gate * sigmoid(alpha*gate)
      g_row[i0 + j] = f32_to_bf16(glu * (up + 1.0f));
    }
  }

  void do_pass2_mxfp4(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int Ih = I / 2;            // packed-K rows
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    const int nh = h1 - h0;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const bf16_t* dn_bias_l = reinterpret_cast<const bf16_t*>(tbl_at(dn_bias_tbl, t->layer_id));
    float acc[HBLK];
    for (int c = 0; c < nh; ++c) acc[c] = 0.0f;
    for (int k = 0; k < top_k; ++k) {
      const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
      if (e < 0 || e >= num_experts) continue;
      const float wt = t->w[static_cast<size_t>(tok) * top_k + k];
      const bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
      const uint8_t* blk_e = dn_packed_l + (size_t)e * Ih * H;
      const uint8_t* scl_e = dn_scale_l + (size_t)e * (size_t)(I / 32) * H;
      float part[HBLK];
      mxgemv(part, blk_e + h0, scl_e + h0, g_row, Ih, H, nh, e2m1_lut, e8m0_lut);
      const bf16_t* bias_e = dn_bias_l + (size_t)e * H + h0;
      for (int c = 0; c < nh; ++c) acc[c] += (part[c] + bf16_to_f32(bias_e[c])) * wt;
    }
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int c = 0; c < nh; ++c) y_row[h0 + c] = f32_to_bf16(acc[c]);
  }

  // ----------------------------- ds_fp4 (DSV4) -------------------------------
  // Row-major e2m1 + e8m0/32 (no global); silu-swiglu with clamp; FP8-roundtripped
  // activations (x once in submit -> xq_scratch; the intermediate g in a dedicated
  // round-trip phase between the two passes). Router weight applies on the down output.

  void do_pass1_dsfp4_route(const MoeTask* t, int route, int ib) {
    const int k = route % top_k;
    const int tok = route / top_k;
    const int e = t->ids[route];
    if (e < 0 || e >= num_experts) return;
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* gu_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const int N2 = 2 * I, Hh = H / 2, Hs = H / 32;
    // fp8-roundtripped input, pre-deinterleaved to fp32 even/odd halves.
    const float* xe = xe_scratch.data() + (size_t)tok * (H / 2);
    const float* xo = xo_scratch.data() + (size_t)tok * (H / 2);
    const uint8_t* gp = gu_packed_l + (size_t)e * N2 * Hh;
    const uint8_t* gs = gu_scale_l + (size_t)e * N2 * Hs;
    bf16_t* g_row = g_scratch.data() + ((size_t)tok * top_k + k) * I;
    const int i0 = static_cast<int>(ib) * IBLK;
    const int i1 = std::min(I, i0 + IBLK);
    const float lim = swiglu_limit;
    for (int i = i0; i < i1; ++i) {
      // gate_up is stored bf16 by the reference GEMV before swiglu; round to match.
      float gate = bf16_to_f32(f32_to_bf16(
          dsdot(gp + (size_t)i * Hh, gs + (size_t)i * Hs, xe, xo, H, e2m1_lut, e8m0_lut)));
      float up = bf16_to_f32(f32_to_bf16(dsdot(
          gp + (size_t)(I + i) * Hh, gs + (size_t)(I + i) * Hs, xe, xo, H, e2m1_lut, e8m0_lut)));
      if (lim > 0.0f) {
        if (gate > lim) gate = lim;
        if (up > lim) up = lim;
        else if (up < -lim) up = -lim;
      }
      const float glu = gate / (1.0f + std::exp(-gate));  // silu(gate)
      g_row[i] = f32_to_bf16(glu * up);
    }
  }

  // Prepare one intermediate row (token,route) for the down GEMV: ds_fp4 first FP8
  // round-trips it (DSV4 act_quant), then both formats deinterleave to fp32 even/odd
  // (reused across every down output row).
  void prep_g_row(int64_t r) {
    bf16_t* g = g_scratch.data() + (size_t)r * I;
    if (use_q4a8) {  // q4_0 W4A8: Q8_0-quantize the intermediate row for the down GEMV.
      quant_q8_0(g, I, gi8_scratch.data() + (size_t)r * I,
                 gas_scratch.data() + (size_t)r * (I / 32));
      return;
    }
    if (fmt == WF_DSFP4) fp8_roundtrip_bf16(g, g, I);
    float* ge = ge_scratch.data() + (size_t)r * (I / 2);
    float* go = go_scratch.data() + (size_t)r * (I / 2);
    deinterleave_bf16_f32(g, ge, go, I);
    if (use_vnni)
      quant_i8_pg16(ge, go, I, gi8_scratch.data() + (size_t)r * I,
                    gas_scratch.data() + (size_t)r * (I / 16));
  }

  void do_pass2_dsfp4(const MoeTask* t, int64_t p) {
    const int64_t hb = p % n_hblk;
    const int tok = static_cast<int>(p / n_hblk);
    const int Ih = I / 2, Is = I / 32;
    const int h0 = static_cast<int>(hb) * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    // Resolve this task's layer base once; row indexing below is layer-local (e).
    const uint8_t* dn_packed_l = reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l = reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    bf16_t* y_row = t->y + (size_t)tok * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int e = t->ids[static_cast<size_t>(tok) * top_k + k];
        if (e < 0 || e >= num_experts) continue;
        const float wt = t->w[static_cast<size_t>(tok) * top_k + k];
        const float* ge = ge_scratch.data() + ((size_t)tok * top_k + k) * (I / 2);
        const float* go = go_scratch.data() + ((size_t)tok * top_k + k) * (I / 2);
        const uint8_t* dp = dn_packed_l + (size_t)e * (size_t)H * Ih + (size_t)h * Ih;
        const uint8_t* ds = dn_scale_l + (size_t)e * (size_t)H * Is + (size_t)h * Is;
        // The reference rounds each route's weighted down output to bf16 before the
        // fp32 sum over routes (down [T, top_k, H] bf16 -> .sum(dim=1)).
        acc += bf16_to_f32(f32_to_bf16(dsdot(dp, ds, ge, go, I, e2m1_lut, e8m0_lut) * wt));
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  // ----------------------- grouped decode (expert-major) -----------------------
  // One work item owns an (expert, output-row tile) and applies that hot tile to
  // every route selecting the expert. The per-route math is unchanged. Pass 2
  // writes independent fp32 route outputs, then pass 3 weights and sums them in the
  // token's original top-k order, preserving the old deterministic reduction.

  void do_pass1_grouped(const MoeTask* t, int64_t p) {
    const int ib = static_cast<int>(p % n_iblk);
    const int di = static_cast<int>(p / n_iblk);
    const int e = distinct_experts[di];
    for (int pos = expert_offsets[e]; pos < expert_offsets[e + 1]; ++pos)
      do_pass1_route(t, grouped_routes[pos], ib);
  }

  void do_pass2_grouped_route(const MoeTask* t, int route, int e, int hb) {
    const int h0 = hb * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    float* route_y = route_y_scratch.data() + static_cast<size_t>(route) * H;

    if (fmt == WF_MXFP4) {
      const int Ih = I / 2;
      const int nh = h1 - h0;
      const uint8_t* dn_packed_l =
          reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
      const uint8_t* dn_scale_l =
          reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
      const bf16_t* dn_bias_l =
          reinterpret_cast<const bf16_t*>(tbl_at(dn_bias_tbl, t->layer_id));
      const bf16_t* g_row = g_scratch.data() + static_cast<size_t>(route) * I;
      const uint8_t* blk_e = dn_packed_l + static_cast<size_t>(e) * Ih * H;
      const uint8_t* scl_e = dn_scale_l + static_cast<size_t>(e) * (I / 32) * H;
      float part[HBLK];
      mxgemv(part, blk_e + h0, scl_e + h0, g_row, Ih, H, nh, e2m1_lut, e8m0_lut);
      const bf16_t* bias_e = dn_bias_l + static_cast<size_t>(e) * H + h0;
      for (int c = 0; c < nh; ++c)
        route_y[h0 + c] = part[c] + bf16_to_f32(bias_e[c]);
      return;
    }

    if (fmt == WF_DSFP4) {
      const int Ih = I / 2, Is = I / 32;
      const uint8_t* dn_packed_l =
          reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
      const uint8_t* dn_scale_l =
          reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
      const float* ge = ge_scratch.data() + static_cast<size_t>(route) * (I / 2);
      const float* go = go_scratch.data() + static_cast<size_t>(route) * (I / 2);
      for (int h = h0; h < h1; ++h) {
        const uint8_t* dp =
            dn_packed_l + static_cast<size_t>(e) * H * Ih + static_cast<size_t>(h) * Ih;
        const uint8_t* ds =
            dn_scale_l + static_cast<size_t>(e) * H * Is + static_cast<size_t>(h) * Is;
        route_y[h] = dsdot(dp, ds, ge, go, I, e2m1_lut, e8m0_lut);
      }
      return;
    }

    const bf16_t* down_l = reinterpret_cast<const bf16_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_packed_l =
        reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l =
        reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const uint16_t* dn_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(dn_global_tbl, t->layer_id));
    const bf16_t* g_row = g_scratch.data() + static_cast<size_t>(route) * I;
    const float* ge = needs_di ? ge_scratch.data() + static_cast<size_t>(route) * (I / 2) : nullptr;
    const float* go = needs_di ? go_scratch.data() + static_cast<size_t>(route) * (I / 2) : nullptr;
    const int8_t* gi8 =
        (use_vnni || use_q4a8) ? gi8_scratch.data() + static_cast<size_t>(route) * I : nullptr;
    const float* gas = use_vnni ? gas_scratch.data() + static_cast<size_t>(route) * (I / 16)
                       : use_q4a8 ? gas_scratch.data() + static_cast<size_t>(route) * (I / 32)
                                    : nullptr;
    for (int h = h0; h < h1; ++h)
      route_y[h] = gemm2_dot(down_l, dn_packed_l, dn_scale_l, dn_global_l, e, h, g_row,
                             ge, go, gi8, gas);
  }

  void do_pass2_grouped(const MoeTask* t, int64_t p) {
    const int hb = static_cast<int>(p % n_hblk);
    const int di = static_cast<int>(p / n_hblk);
    const int e = distinct_experts[di];
    for (int pos = expert_offsets[e]; pos < expert_offsets[e + 1]; ++pos)
      do_pass2_grouped_route(t, grouped_routes[pos], e, hb);
  }

  void do_pass3_grouped(const MoeTask* t, int64_t p) {
    const int hb = static_cast<int>(p % n_hblk);
    const int tok = static_cast<int>(p / n_hblk);
    const int h0 = hb * HBLK;
    const int h1 = std::min(H, h0 + HBLK);
    bf16_t* y_row = t->y + static_cast<size_t>(tok) * H;
    for (int h = h0; h < h1; ++h) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int route = tok * top_k + k;
        const int e = t->ids[route];
        if (e < 0 || e >= num_experts) continue;
        const float down = route_y_scratch[static_cast<size_t>(route) * H + h];
        if (fmt == WF_DSFP4) {
          // Preserve ds_fp4's per-route weighted bf16 rounding before the sum.
          acc += bf16_to_f32(f32_to_bf16(down * t->w[route]));
        } else {
          const float w_out = (fmt == WF_MXFP4 || !apply_on_input) ? t->w[route] : 1.0f;
          acc += down * w_out;
        }
      }
      y_row[h] = f32_to_bf16(acc);
    }
  }

  void prepare_prefill_batch(const MoeTask* t) {
    const size_t rows = grouped_routes.size();
    if (t->num_tokens > prefill_capacity ||
        rows > static_cast<size_t>(prefill_capacity) * top_k) {
      throw std::runtime_error("CPU MoE prefill batch exceeds configured capacity");
    }
    for (size_t pos = 0; pos < rows; ++pos) {
      const int route = grouped_routes[pos];
      const int tok = route / top_k;
      std::memcpy(prefill_x_scratch.data() + pos * H,
                  t->x + static_cast<size_t>(tok) * H,
                  static_cast<size_t>(H) * sizeof(bf16_t));
    }
    // Compact the gathered bf16 rows into the first half of the same allocation.
    // Increasing row order is overlap-safe because every int8 destination trails
    // the bf16 source row it replaces. Block scales live in a separate array.
    int8_t* xi8 = reinterpret_cast<int8_t*>(prefill_x_scratch.data());
    for (size_t pos = 0; pos < rows; ++pos) {
      quant_i8_bf16_pg16(
          prefill_x_scratch.data() + pos * H, H, xi8 + pos * H,
          prefill_x_scale_scratch.data() + pos * (H / 16));
    }
  }

  void do_prefill_batch_expert(const MoeTask* t, int distinct_index) {
    const int e = distinct_experts[distinct_index];
    const int begin = expert_offsets[e];
    const int count = expert_offsets[e + 1] - begin;
    if (count <= 0) return;

    const uint8_t* gu_packed_l =
        reinterpret_cast<const uint8_t*>(tbl_at(gate_up_tbl, t->layer_id));
    const uint8_t* gu_scale_l =
        reinterpret_cast<const uint8_t*>(tbl_at(gu_scale_tbl, t->layer_id));
    const uint16_t* gu_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(gu_global_tbl, t->layer_id));
    const uint8_t* dn_packed_l =
        reinterpret_cast<const uint8_t*>(tbl_at(down_tbl, t->layer_id));
    const uint8_t* dn_scale_l =
        reinterpret_cast<const uint8_t*>(tbl_at(dn_scale_tbl, t->layer_id));
    const uint16_t* dn_global_l =
        reinterpret_cast<const uint16_t*>(tbl_at(dn_global_tbl, t->layer_id));

    const int8_t* xi8 = reinterpret_cast<const int8_t*>(prefill_x_scratch.data());
    const int8_t* expert_x = xi8 + static_cast<size_t>(begin) * H;
    const float* expert_xs =
        prefill_x_scale_scratch.data() + static_cast<size_t>(begin) * (H / 16);
    // Each expert owns [begin*R_w, (begin+count)*R_w), so distinct-expert
    // work stealing never aliases projection tiles.
    float* gate = prefill_gate_scratch.data() + static_cast<size_t>(begin) * kNvi8WeightRows;
    float* up = prefill_up_scratch.data() + static_cast<size_t>(begin) * kNvi8WeightRows;
    const bool swigluoai = act == ACT_SWIGLUOAI;
    const bool clamped_silu = act == ACT_CLAMPED_SILU;
    for (int i0 = 0; i0 < I; i0 += kNvi8WeightRows) {
      const int rows = std::min(kNvi8WeightRows, I - i0);
      const size_t gr = static_cast<size_t>(e) * (2 * I) + i0;
      const size_t ur = gr + I;
      nvi8batch_rows(gate, gu_packed_l + gr * (H / 2),
                     gu_scale_l + gr * (H / 16), gu_global_l + gr, rows,
                     expert_x, count, H, e4m3_lut, expert_xs);
      nvi8batch_rows(up, gu_packed_l + ur * (H / 2),
                     gu_scale_l + ur * (H / 16), gu_global_l + ur, rows,
                     expert_x, count, H, e4m3_lut, expert_xs);
      for (int r = 0; r < rows; ++r) {
        for (int m = 0; m < count; ++m) {
          const int pos = begin + m;
          const int route = grouped_routes[pos];
          const float w_in = apply_on_input ? t->w[route] : 1.0f;
          float gv = gate[static_cast<size_t>(r) * count + m] * w_in;
          float uv = up[static_cast<size_t>(r) * count + m] * w_in;
          float activated;
          if (swigluoai || clamped_silu) {
            if (gv > swiglu_limit) gv = swiglu_limit;
            if (uv > swiglu_limit) uv = swiglu_limit;
            else if (uv < -swiglu_limit) uv = -swiglu_limit;
            const float alpha = swigluoai ? swiglu_alpha : 1.0f;
            activated = gv / (1.0f + std::exp(-gv * alpha)) *
                        (swigluoai ? uv + 1.0f : uv);
          } else {
            activated = act_apply(act, gv) * uv;
          }
          prefill_g_scratch[static_cast<size_t>(pos) * I + i0 + r] =
              f32_to_bf16(activated);
        }
      }
    }

    for (int m = 0; m < count; ++m) {
      const size_t pos = static_cast<size_t>(begin + m);
      quant_i8_bf16_pg16(
          prefill_g_scratch.data() + pos * I, I,
          prefill_gi8_scratch.data() + pos * I,
          prefill_g_scale_scratch.data() + pos * (I / 16));
    }

    const int8_t* expert_g =
        prefill_gi8_scratch.data() + static_cast<size_t>(begin) * I;
    const float* expert_gs =
        prefill_g_scale_scratch.data() + static_cast<size_t>(begin) * (I / 16);
    for (int h0 = 0; h0 < H; h0 += kNvi8WeightRows) {
      const int rows = std::min(kNvi8WeightRows, H - h0);
      const size_t dr = static_cast<size_t>(e) * H + h0;
      nvi8batch_rows(gate, dn_packed_l + dr * (I / 2),
                     dn_scale_l + dr * (I / 16), dn_global_l + dr, rows,
                     expert_g, count, I, e4m3_lut, expert_gs);
      for (int r = 0; r < rows; ++r) {
        for (int m = 0; m < count; ++m) {
          prefill_y_scratch[(static_cast<size_t>(begin + m) * H) + h0 + r] =
              f32_to_bf16(gate[static_cast<size_t>(r) * count + m]);
        }
      }
    }
  }

  void scatter_prefill_batch(const MoeTask* t) {
    for (int tok = 0; tok < t->num_tokens; ++tok) {
      bf16_t* y = t->y + static_cast<size_t>(tok) * H;
      for (int h = 0; h < H; ++h) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
          const int route = tok * top_k + k;
          const int pos = route_to_group[route];
          if (pos < 0) continue;
          const float w_out = apply_on_input ? 1.0f : t->w[route];
          acc += bf16_to_f32(
              prefill_y_scratch[static_cast<size_t>(pos) * H + h]) * w_out;
        }
        y[h] = f32_to_bf16(acc);
      }
    }
  }

  void run_task_body(const MoeTask* t) {
    if (t->prefill_batch) {
      for (;;) {
        const int64_t p = p1_next.fetch_add(1, std::memory_order_relaxed);
        if (p >= p1_total) break;
        do_prefill_batch_expert(t, static_cast<int>(p));
      }
      return;
    }
    int local_sense = 0;
    for (;;) {
      int64_t p = p1_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p1_total) break;
      if (t->group_routes) do_pass1_grouped(t, p);
      else do_pass1(t, p);
    }
    barrier(local_sense);
    // Row-major fp4: prepare the intermediate rows (per token,route) before the down
    // GEMV -- ds_fp4 FP8 round-trips (DSV4 act_quant), both deinterleave to fp32; q4_0
    // W4A8 Q8_0-quantizes. Needs all of pass1 done (a full row spans every iblk).
    if (needs_di || use_q4a8) {
      for (;;) {
        int64_t r = prt_next.fetch_add(1, std::memory_order_relaxed);
        if (r >= prt_total) break;
        prep_g_row(r);
      }
      barrier(local_sense);
    }
    for (;;) {
      int64_t p = p2_next.fetch_add(1, std::memory_order_relaxed);
      if (p >= p2_total) break;
      if (t->group_routes) do_pass2_grouped(t, p);
      else do_pass2(t, p);
    }
    if (t->group_routes) {
      barrier(local_sense);
      for (;;) {
        int64_t p = p3_next.fetch_add(1, std::memory_order_relaxed);
        if (p >= p3_total) break;
        do_pass3_grouped(t, p);
      }
    }
  }

  void worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    for (;;) {
      MoeTask* t;
      {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop || cur_gen != my_gen; });
        if (stop) return;
        my_gen = cur_gen;
        t = cur_task;
      }
      run_task_body(t);
      if (done_count.fetch_add(1) + 1 == num_threads) {
        completed.store(my_gen, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  static int64_t steady_now_ns() {
    // std::chrono::steady_clock is CLOCK_MONOTONIC on Linux, matching Python's
    // time.monotonic_ns() used to calibrate the surrounding CUDA events.
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void timed_worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    for (;;) {
      MoeTask* t;
      {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop || cur_gen != my_gen; });
        if (stop) return;
        my_gen = cur_gen;
        t = cur_task;
      }
      if (t->timing_active) {
        int64_t expected = 0;
        if (t->t_first_worker_ns.compare_exchange_strong(
                expected, -1, std::memory_order_acq_rel)) {
          t->t_first_worker_ns.store(steady_now_ns(), std::memory_order_release);
        }
      }
      run_task_body(t);
      if (done_count.fetch_add(1) + 1 == num_threads) {
        if (t->timing_active)
          t->t_compute_done_ns.store(steady_now_ns(), std::memory_order_release);
        completed.store(my_gen, std::memory_order_release);
        if (t->timing_active) {
          t->t_signalled_ns.store(
              steady_now_ns(), std::memory_order_release);
        }
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  void build_decode_groups(const MoeTask* t) {
    std::fill(expert_offsets.begin(), expert_offsets.end(), 0);
    const int routes = t->num_tokens * top_k;
    for (int route = 0; route < routes; ++route) {
      const int e = t->ids[route];
      if (e >= 0 && e < num_experts) ++expert_offsets[e + 1];
    }
    distinct_experts.clear();
    for (int e = 0; e < num_experts; ++e) {
      expert_offsets[e + 1] += expert_offsets[e];
      if (expert_offsets[e + 1] != expert_offsets[e]) distinct_experts.push_back(e);
      expert_cursor[e] = expert_offsets[e];
    }
    grouped_routes.resize(static_cast<size_t>(expert_offsets[num_experts]));
    // Route ids are visited in token-major/top-k order. Insertion is therefore
    // stable within each expert group, which also makes grouping deterministic.
    for (int route = 0; route < routes; ++route) {
      const int e = t->ids[route];
      if (e >= 0 && e < num_experts) {
        const int pos = expert_cursor[e]++;
        grouped_routes[pos] = route;
        if (t->prefill_batch) route_to_group[route] = pos;
      } else if (t->prefill_batch) {
        route_to_group[route] = -1;
      }
    }
  }

  uint64_t expert_weight_bytes() const {
    const uint64_t h = static_cast<uint64_t>(H);
    const uint64_t i = static_cast<uint64_t>(I);
    const uint64_t hi = h * i;
    switch (fmt) {
      case WF_BF16:
        return 6 * hi;
      case WF_NVFP4:
        return (3 * hi) / 2 + (3 * hi) / 16 + 4 * i + 2 * h;
      case WF_MXFP4:
        return (3 * hi) / 2 + (3 * hi) / 32 + 4 * i + 2 * h;
      case WF_DSFP4:
        return (3 * hi) / 2 + (3 * hi) / 32;
      case WF_Q4_0:
        return (54 * hi) / 32;
      default:
        return 0;
    }
  }

  void finish_task_timing(MoeTask* t, int64_t signalled_ns) {
    if (!t->timing_active) return;
    const int64_t doorbell_ns = t->t_doorbell_ns.load(std::memory_order_acquire);
    const int64_t groups_done_ns =
        t->t_groups_done_ns.load(std::memory_order_acquire);
    const int64_t gil_acquired_ns =
        t->t_gil_acquired_ns.load(std::memory_order_acquire);
    const int64_t precb_done_ns =
        t->t_precb_done_ns.load(std::memory_order_acquire);
    const int64_t seen_ns = t->t_seen_ns.load(std::memory_order_acquire);
    const int64_t worker_signalled_ns =
        t->t_signalled_ns.load(std::memory_order_acquire);
    const int64_t done_stored_ns =
        t->t_done_stored_ns.load(std::memory_order_acquire);
    const int64_t first_worker_ns =
        t->t_first_worker_ns.load(std::memory_order_acquire);
    const int64_t compute_done_ns =
        t->t_compute_done_ns.load(std::memory_order_acquire);
    t->t_signalled_ns.store(signalled_ns, std::memory_order_release);
    const int64_t wake_ns = std::max<int64_t>(0, first_worker_ns - doorbell_ns);
    const int64_t groups_ns = std::max<int64_t>(0, groups_done_ns - doorbell_ns);
    // Deferred callback (after notify): the GIL wait starts at the notify stamp and
    // wake is groups plus notify; gil and precb then overlap compute. Default mode:
    // groups + gil + precb + notify == wake.
    const bool deferred = t->timing_precb_deferred;
    const int64_t notified_ns = t->t_notified_ns.load(std::memory_order_acquire);
    const int64_t gil_ns = std::max<int64_t>(
        0, gil_acquired_ns - (deferred ? notified_ns : groups_done_ns));
    const int64_t precb_ns = std::max<int64_t>(0, precb_done_ns - gil_acquired_ns);
    const int64_t notify_ns = std::max<int64_t>(
        0, first_worker_ns - (deferred ? groups_done_ns : precb_done_ns));
    const int64_t seen_to_doorbell_ns =
        seen_ns > 0 ? std::max<int64_t>(0, doorbell_ns - seen_ns) : 0;
    const int64_t done_store_ns =
        done_stored_ns > 0
            ? std::max<int64_t>(0, done_stored_ns - worker_signalled_ns)
            : 0;
    const int64_t compute_ns = std::max<int64_t>(0, compute_done_ns - first_worker_ns);
    const int64_t signal_ns = std::max<int64_t>(0, signalled_ns - compute_done_ns);
    t->timing_last_run_ns.store(
        std::max<int64_t>(0, signalled_ns - doorbell_ns), std::memory_order_release);
    {
      std::lock_guard<std::mutex> lk(step_timing_mtx);
      StepTimingAccum& row = step_timing.at(static_cast<size_t>(t->layer_id));
      row.wake_ns += wake_ns;
      row.groups_ns += groups_ns;
      row.gil_ns += gil_ns;
      row.precb_ns += precb_ns;
      row.notify_ns += notify_ns;
      row.seen_to_doorbell_ns += seen_to_doorbell_ns;
      row.done_store_ns += done_store_ns;
      row.last_seen_ns = seen_ns;
      row.last_done_stored_ns = done_stored_ns;
      row.compute_ns += compute_ns;
      row.signal_ns += signal_ns;
      row.wake_max_ns = std::max(row.wake_max_ns, wake_ns);
      row.gil_max_ns = std::max(row.gil_max_ns, gil_ns);
      row.precb_max_ns = std::max(row.precb_max_ns, precb_ns);
      row.compute_max_ns = std::max(row.compute_max_ns, compute_ns);
      row.signal_max_ns = std::max(row.signal_max_ns, signal_ns);
      ++row.tasks;
      row.experts += t->timing_experts;
      row.bytes += t->timing_bytes;
    }
    t->timing_active = false;
    step_timing_inflight.fetch_sub(1, std::memory_order_release);
  }

  pybind11::dict step_timing_snapshot_and_reset() {
    pybind11::dict result;
    while (step_timing_inflight.load(std::memory_order_acquire) != 0)
      std::this_thread::yield();
    std::lock_guard<std::mutex> lk(step_timing_mtx);
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
      StepTimingAccum& row = step_timing[static_cast<size_t>(layer_id)];
      if (row.tasks == 0) continue;
      pybind11::dict values;
      values["wake_us"] = static_cast<double>(row.wake_ns) / 1000.0;
      values["groups_us"] = static_cast<double>(row.groups_ns) / 1000.0;
      values["gil_us"] = static_cast<double>(row.gil_ns) / 1000.0;
      values["precb_us"] = static_cast<double>(row.precb_ns) / 1000.0;
      values["notify_us"] = static_cast<double>(row.notify_ns) / 1000.0;
      values["coord_pre_us"] =
          static_cast<double>(row.seen_to_doorbell_ns) / 1000.0;
      values["coord_post_us"] = static_cast<double>(row.done_store_ns) / 1000.0;
      values["last_seen_ns"] = row.last_seen_ns;
      values["last_done_stored_ns"] = row.last_done_stored_ns;
      values["compute_us"] = static_cast<double>(row.compute_ns) / 1000.0;
      values["signal_us"] = static_cast<double>(row.signal_ns) / 1000.0;
      values["wake_max_us"] = static_cast<double>(row.wake_max_ns) / 1000.0;
      values["gil_max_us"] = static_cast<double>(row.gil_max_ns) / 1000.0;
      values["precb_max_us"] = static_cast<double>(row.precb_max_ns) / 1000.0;
      values["compute_max_us"] = static_cast<double>(row.compute_max_ns) / 1000.0;
      values["signal_max_us"] = static_cast<double>(row.signal_max_ns) / 1000.0;
      values["tasks"] = row.tasks;
      values["experts"] = row.experts;
      values["bytes"] = row.bytes;
      result[pybind11::int_(layer_id)] = std::move(values);
      row = StepTimingAccum{};
    }
    return result;
  }

  void submit(MoeTask* t, bool run_pre_callback = true,
              bool coordinator_submission = false) {
    // Persistent group_routes tasks are decode tasks. Prefill remains outside the
    // per-step snapshot even when diagnostics are enabled.
    const bool time_task = t->group_routes &&
        task_timing_enabled.load(std::memory_order_relaxed);
    if (time_task) {
      t->timing_active = true;
      t->t_doorbell_ns.store(steady_now_ns(), std::memory_order_release);
      t->t_groups_done_ns.store(0, std::memory_order_relaxed);
      t->t_gil_acquired_ns.store(0, std::memory_order_relaxed);
      t->t_precb_done_ns.store(0, std::memory_order_relaxed);
      if (!coordinator_submission)
        t->t_seen_ns.store(0, std::memory_order_relaxed);
      t->t_done_stored_ns.store(0, std::memory_order_relaxed);
      t->t_notified_ns.store(0, std::memory_order_relaxed);
      t->timing_precb_deferred = false;
      t->t_first_worker_ns.store(0, std::memory_order_relaxed);
      t->t_compute_done_ns.store(0, std::memory_order_relaxed);
      t->t_signalled_ns.store(0, std::memory_order_relaxed);
    }
    if (t->group_routes || t->prefill_batch) build_decode_groups(t);
    if (time_task)
      t->t_groups_done_ns.store(steady_now_ns(), std::memory_order_release);
    if (time_task) {
      // Each valid route applies one expert and streams that expert's weights once.
      t->timing_experts = static_cast<uint64_t>(grouped_routes.size());
      t->timing_bytes = t->timing_experts * expert_weight_bytes();
    }
    const bool defer_pre_callback =
        coordinator_submission && t->group_routes && !t->prefill_batch &&
        run_pre_callback && pre_run_callback &&
        pre_run_callback_mode.load(std::memory_order_relaxed) == 1;
    if (time_task) t->timing_precb_deferred = defer_pre_callback;
    if (!defer_pre_callback && run_pre_callback && pre_run_callback) {
      pybind11::gil_scoped_acquire gil;
      if (time_task)
        t->t_gil_acquired_ns.store(steady_now_ns(), std::memory_order_release);
      if (t->group_routes || t->prefill_batch) {
        // The same route D2H used by compute is enough to build the expert union.
        // Hand WILLNEED the compact list plus the original valid-pair count, with
        // no second transfer and no repeated page-advice requests.
        pre_run_callback(t->layer_id, distinct_experts, grouped_routes.size());
      } else {
        std::vector<int32_t> ids(
            t->ids, t->ids + static_cast<size_t>(t->num_tokens) * top_k);
        pre_run_callback(t->layer_id, std::move(ids));
      }
      if (time_task)
        t->t_precb_done_ns.store(steady_now_ns(), std::memory_order_release);
    } else if (time_task && !defer_pre_callback) {
      const int64_t groups_done_ns =
          t->t_groups_done_ns.load(std::memory_order_relaxed);
      t->t_gil_acquired_ns.store(groups_done_ns, std::memory_order_relaxed);
      t->t_precb_done_ns.store(groups_done_ns, std::memory_order_relaxed);
    }
    if (t->prefill_batch) {
      prepare_prefill_batch(t);
      p1_total = static_cast<int64_t>(distinct_experts.size());
      p2_total = p3_total = prt_total = 0;
      p1_next.store(0, std::memory_order_relaxed);
      done_count.store(0, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lk(task_mtx);
        cur_task = t;
        ++cur_gen;
        submitted.store(cur_gen, std::memory_order_release);
      }
      task_cv.notify_all();
      return;
    }
    n_iblk = (I + IBLK - 1) / IBLK;
    n_hblk = (H + HBLK - 1) / HBLK;
    // Grow the per-token intermediate scratch if a larger batch shows up than the
    // construction-time hint. Decode graph warmup covers its fixed batch sizes;
    // synchronous prefill can grow this further as larger chunks arrive.
    const size_t need = static_cast<size_t>(t->num_tokens) * top_k * I;
    if (need > g_scratch.size()) g_scratch.resize(need);
    const size_t route_y_need = static_cast<size_t>(t->num_tokens) * top_k * H;
    if (t->group_routes && route_y_need > route_y_scratch.size())
      route_y_scratch.resize(route_y_need);
    p1_total = t->group_routes
        ? static_cast<int64_t>(distinct_experts.size()) * n_iblk
        : static_cast<int64_t>(t->num_tokens) * top_k * n_iblk;
    p2_total = t->group_routes
        ? static_cast<int64_t>(distinct_experts.size()) * n_hblk
        : static_cast<int64_t>(t->num_tokens) * n_hblk;
    p3_total = t->group_routes ? static_cast<int64_t>(t->num_tokens) * n_hblk : 0;
    prt_total = (needs_di || use_q4a8) ? static_cast<int64_t>(t->num_tokens) * top_k : 0;
    p1_next.store(0, std::memory_order_relaxed);
    p2_next.store(0, std::memory_order_relaxed);
    p3_next.store(0, std::memory_order_relaxed);
    prt_next.store(0, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    bar_count.store(0, std::memory_order_relaxed);
    bar_sense.store(0, std::memory_order_relaxed);
    // ds_fp4: FP8 round-trip the per-token input once, up front (single-threaded;
    // tiny for decode, and done before the workers are woken below).
    if (needs_di) {
      const size_t xn = static_cast<size_t>(t->num_tokens) * H;
      if (xn / 2 > xe_scratch.size()) {
        xe_scratch.resize(xn / 2);
        xo_scratch.resize(xn / 2);
        ge_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 2));
        go_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 2));
        if (use_vnni) {
          xi8_scratch.resize(xn);
          xas_scratch.resize(xn / 16);
          gi8_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * I);
          gas_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 16));
        }
      }
      const bool ds = (fmt == WF_DSFP4) && !input_prequant;
      if (ds && xn > xq_scratch.size()) xq_scratch.resize(xn);
      for (int tok = 0; tok < t->num_tokens; ++tok) {
        const bf16_t* src = t->x + (size_t)tok * H;
        if (ds) {  // DSV4 FP8-round-trips the input before the gate_up GEMV
          bf16_t* xq = xq_scratch.data() + (size_t)tok * H;
          fp8_roundtrip_bf16(src, xq, H);
          src = xq;
        }
        float* xe = xe_scratch.data() + (size_t)tok * (H / 2);
        float* xo = xo_scratch.data() + (size_t)tok * (H / 2);
        deinterleave_bf16_f32(src, xe, xo, H);
        if (use_vnni)
          quant_i8_pg16(xe, xo, H, xi8_scratch.data() + (size_t)tok * H,
                        xas_scratch.data() + (size_t)tok * (H / 16));
      }
    }
    // q4_0 W4A8: Q8_0-quantize the per-token input once (single-threaded, tiny for decode).
    if (use_q4a8) {
      const size_t xn = static_cast<size_t>(t->num_tokens) * H;
      if (xn > xi8_scratch.size()) {
        xi8_scratch.resize(xn);
        xas_scratch.resize(xn / 32);
        gi8_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * I);
        gas_scratch.resize(static_cast<size_t>(t->num_tokens) * top_k * (I / 32));
      }
      for (int tok = 0; tok < t->num_tokens; ++tok)
        quant_q8_0(t->x + (size_t)tok * H, H, xi8_scratch.data() + (size_t)tok * H,
                   xas_scratch.data() + (size_t)tok * (H / 32));
    }
    if (time_task) step_timing_inflight.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_task = t;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
    if (time_task) t->t_notified_ns.store(steady_now_ns(), std::memory_order_release);
    if (defer_pre_callback) {
      // Group construction finishes before notify. Workers only read these vectors,
      // so the callback can inspect them while grouped decode compute is running.
      pybind11::gil_scoped_acquire gil;
      if (time_task)
        t->t_gil_acquired_ns.store(steady_now_ns(), std::memory_order_release);
      if (t->group_routes || t->prefill_batch) {
        // The same route D2H used by compute is enough to build the expert union.
        // Hand WILLNEED the compact list plus the original valid-pair count, with
        // no second transfer and no repeated page-advice requests.
        pre_run_callback(t->layer_id, distinct_experts, grouped_routes.size());
      } else {
        std::vector<int32_t> ids(
            t->ids, t->ids + static_cast<size_t>(t->num_tokens) * top_k);
        pre_run_callback(t->layer_id, std::move(ids));
      }
      if (time_task)
        t->t_precb_done_ns.store(steady_now_ns(), std::memory_order_release);
    }
  }

  void sync(MoeTask* timing_task = nullptr) {
    const uint64_t target = submitted.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lk(sync_mtx);
    sync_cv.wait(lk, [&] { return completed.load(std::memory_order_acquire) >= target; });
    (void)timing_task;
  }

  void submit_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::submit_cb,
                       reinterpret_cast<void*>(task));
  }

  void sync_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMoeExecutor::sync_cb,
                       reinterpret_cast<void*>(task));
  }

  // Register a (layer, batch-size) slot's task so the coordinator can dispatch it on a
  // flag bump.
  void register_flag_task(int slot, uintptr_t task) {
    std::lock_guard<std::mutex> lk(flag_task_mtx);
    if (static_cast<int>(flag_task.size()) <= slot) flag_task.resize(slot + 1, nullptr);
    flag_task[slot] = reinterpret_cast<MoeTask*>(task);
  }

  void register_flag_gpufetch_task(int slot, uintptr_t task) {
    std::lock_guard<std::mutex> lk(flag_task_mtx);
    if (static_cast<int>(flag_gpufetch_task.size()) <= slot)
      flag_gpufetch_task.resize(slot + 1, nullptr);
    flag_gpufetch_task[slot] = reinterpret_cast<GpuFetchTask*>(task);
  }

  int64_t flag_served_count(int slot) const {
    return (slot >= 0 && slot < static_cast<int>(flag_served.size())) ? flag_served[slot] : 0;
  }

  // Start the busy-poll coordinator over the mapped-pinned flag arrays. ``pin_core`` >= 0
  // pins the coordinator to that logical CPU (the worker auto-sizing reserves it), so its
  // polling never migrates onto / contends with a GEMV worker's core.
  void start_flag_coordinator(uintptr_t ready_ptr, uintptr_t done_ptr, int num_slots,
                              int pin_core) {
    ready_flags = reinterpret_cast<volatile int64_t*>(ready_ptr);
    done_flags = reinterpret_cast<volatile int64_t*>(done_ptr);
    coord_num_slots = num_slots;
    {
      std::lock_guard<std::mutex> lk(flag_task_mtx);
      if (static_cast<int>(flag_task.size()) < num_slots) flag_task.resize(num_slots, nullptr);
      if (static_cast<int>(flag_gpufetch_task.size()) < num_slots)
        flag_gpufetch_task.resize(num_slots, nullptr);
    }
    flag_served.assign(num_slots, 0);
    coord_stop.store(false);
    coord_thread = std::thread([this, pin_core] {
#if CPU_MOE_HAS_AFFINITY
      if (pin_core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pin_core, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
#endif
      coordinator_loop();
    });
  }

  void coordinator_loop() {
    // Idle backoff (edge-device friendliness): spin hot only while decode traffic is
    // flowing, with a TIME-based hot window (an iteration count varies 2-6x with slot
    // count and pause cost, so some configs dozed every token). 50 ms since the last
    // flag comfortably covers intra- and inter-token gaps -- including host-func-only
    // stretches (prefill bursts, --moe-cpu-layers subsets) -- so steady decode never
    // sleeps and keeps the ~us-level wakeup. Past it, i.e. the engine is actually idle,
    // escalate to timed sleeps capped at 2 ms: a dozing coordinator costs <0.1% of a
    // core instead of 100%, and the only price is a <=2 ms discovery delay on the FIRST
    // MoE layer after an idle period (irrelevant next to prefill). The clock is sampled
    // every 1024 empty polls (~0.1-1 ms) to keep the hot loop cheap.
    using coord_clock = std::chrono::steady_clock;
    constexpr auto kHotWindow = std::chrono::milliseconds(50);
    constexpr int64_t kSleepCapUs = 2000;
    auto last_active = coord_clock::now();
    unsigned empty_polls = 0;  // unsigned: the hot-phase ++ must not overflow into UB
    int64_t sleep_us = 100;
    bool dozing = false;
    while (!coord_stop.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int L = 0; L < coord_num_slots; ++L) {
        // Binary handshake (memop-compatible: the GPU-side WAIT compares against an
        // immediate baked at graph capture, so the protocol resets per step instead of
        // counting). Acquire: everything the GPU made visible before setting ready --
        // the D2H input copies -- is visible to the worker pool after this read.
        if (flag_load_acquire(&ready_flags[L]) != 0) {
          flag_store_release(&ready_flags[L], 0);  // consume this step's doorbell
          MoeTask* t;
          GpuFetchTask* fetch;
          {
            std::lock_guard<std::mutex> lk(flag_task_mtx);
            t = (L < static_cast<int>(flag_task.size())) ? flag_task[L] : nullptr;
            fetch = (L < static_cast<int>(flag_gpufetch_task.size()))
                        ? flag_gpufetch_task[L] : nullptr;
          }
          if (fetch != nullptr) {
            run_gpufetch(fetch);
          } else if (t != nullptr) {
            if (t->group_routes &&
                task_timing_enabled.load(std::memory_order_relaxed)) {
              t->t_seen_ns.store(steady_now_ns(), std::memory_order_release);
            } else {
              t->t_seen_ns.store(0, std::memory_order_relaxed);
            }
            submit(t, true, true);
            sync(t);
          }
          // Release: the workers' y stores are visible before the GPU sees done.
          flag_store_release(&done_flags[L], 1);
          if (t != nullptr && t->timing_active) {
            const int64_t done_stored_ns = steady_now_ns();
            t->t_done_stored_ns.store(done_stored_ns, std::memory_order_release);
            finish_task_timing(t, done_stored_ns);
          }
          if (L < static_cast<int>(flag_served.size())) ++flag_served[L];
          any = true;
        }
      }
      if (any) {
        last_active = coord_clock::now();
        empty_polls = 0;
        sleep_us = 100;
        dozing = false;
        continue;
      }
      // Hot phase: pause-spin, consulting the clock only every 1024 empty polls (the
      // amortization must gate the CLOCK, not the sleep -- gating the sleep decision
      // let 1023/1024 idle iterations hot-scan all slots, ~20% of a core at 992 slots).
      // Doze phase: sleep on EVERY iteration until traffic returns; one wake-up scan
      // (~0.5 us) per 2 ms sleep is ~0.03% duty.
      if (!dozing) {
        if ((++empty_polls & 1023u) != 0 ||
            coord_clock::now() - last_active < kHotWindow) {
#if CPU_MOE_X86
          _mm_pause();
#endif
          continue;
        }
        dozing = true;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min<int64_t>(sleep_us * 2, kSleepCapUs);
    }
    // Teardown: release any in-flight (or future) spin-wait immediately so a replay
    // caught mid-shutdown exits its sync kernel now instead of owning the watchdog
    // stall. Runs before the destructor's join() returns, while the flag arrays are
    // still alive on the Python side.
    for (int L = 0; L < coord_num_slots; ++L) {
      flag_store_release(&done_flags[L], INT64_MAX);
    }
  }

  // Eager (non-graph) path: run one task to completion on the pool.
  void run_task(uintptr_t task) {
    MoeTask* t = reinterpret_cast<MoeTask*>(task);
    submit(t);
    sync(t);
    if (t->timing_active) {
      finish_task_timing(
          t, t->t_signalled_ns.load(std::memory_order_acquire));
    }
  }

  // Synchronous task whose descriptor lives only for this call. Prefill uses one
  // grow-to-largest IO buffer and passes the current token count here, avoiding the
  // persistent exact-batch task cache required by CUDA-graph decode.
  void run_task_sync(int layer_id, int num_tokens, uintptr_t x_ptr,
                     uintptr_t ids_ptr, uintptr_t w_ptr, uintptr_t y_ptr,
                     bool run_pre_callback) {
    MoeTask task{this,
                 layer_id,
                 num_tokens,
                 false,
                 false,
                 reinterpret_cast<const bf16_t*>(x_ptr),
                 reinterpret_cast<const int32_t*>(ids_ptr),
                 reinterpret_cast<const float*>(w_ptr),
                 reinterpret_cast<bf16_t*>(y_ptr)};
    submit(&task, run_pre_callback);
    sync(&task);
    if (task.timing_active) {
      finish_task_timing(
          &task, task.t_signalled_ns.load(std::memory_order_acquire));
    }
  }

  std::vector<int64_t> run_prefill_batch_sync(
      int layer_id, int num_tokens, uintptr_t x_ptr, uintptr_t ids_ptr,
      uintptr_t w_ptr, uintptr_t y_ptr) {
    if (prefill_capacity <= 0 || num_tokens <= 0 || num_tokens > prefill_capacity)
      throw std::runtime_error("CPU MoE prefill batch is unavailable or over capacity");
    MoeTask task{this,
                 layer_id,
                 num_tokens,
                 false,
                 true,
                 reinterpret_cast<const bf16_t*>(x_ptr),
                 reinterpret_cast<const int32_t*>(ids_ptr),
                 reinterpret_cast<const float*>(w_ptr),
                 reinterpret_cast<bf16_t*>(y_ptr)};
    submit(&task, false);
    sync(&task);
    if (task.timing_active) {
      finish_task_timing(
          &task, task.t_signalled_ns.load(std::memory_order_acquire));
    }
    scatter_prefill_batch(&task);
    // Logical GEMMs: one fused gate_up projection and one down projection for each
    // non-empty expert group. Activation is fused between them.
    return {static_cast<int64_t>(grouped_routes.size()),
            static_cast<int64_t>(distinct_experts.size()) * 2};
  }

  int64_t task_last_run_ns(uintptr_t task) const {
    const MoeTask* t = reinterpret_cast<const MoeTask*>(task);
    return t->timing_last_run_ns.load(std::memory_order_acquire);
  }

  void set_task_timing(bool enabled) {
    if (enabled && !timed_worker_mode)
      throw std::runtime_error("task timing must be enabled when the executor is created");
    task_timing_enabled.store(enabled, std::memory_order_relaxed);
  }

  void set_pre_run_callback(pybind11::function callback) {
    pre_run_callback = std::move(callback);
  }

  void set_pre_run_callback_mode(int mode) {
    if (mode != 0 && mode != 1)
      throw std::invalid_argument("pre-run callback mode must be 0 or 1");
    pre_run_callback_mode.store(mode, std::memory_order_relaxed);
  }

  void gpufetch_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream),
                       &CpuMoeExecutor::gpufetch_cb,
                       reinterpret_cast<void*>(task));
  }

  std::vector<int64_t> gpufetch_stats(bool reset) {
    const int64_t fills = reset ? gpufetch_fills.exchange(0) : gpufetch_fills.load();
    const int64_t steps = reset ? gpufetch_steps.exchange(0) : gpufetch_steps.load();
    const int64_t ns = reset ? gpufetch_fill_ns.exchange(0) : gpufetch_fill_ns.load();
    return {fills, steps, ns};
  }

  int64_t gpufetch_error_code() const {
    return gpufetch_error.load(std::memory_order_acquire);
  }

  static void CUDART_CB submit_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->submit(t);
  }
  static void CUDART_CB sync_cb(void* ud) {
    MoeTask* t = reinterpret_cast<MoeTask*>(ud);
    t->exec->sync(t);
    if (t->timing_active) {
      t->exec->finish_task_timing(
          t, t->t_signalled_ns.load(std::memory_order_acquire));
    }
  }
  static void CUDART_CB gpufetch_cb(void* ud) {
    GpuFetchTask* t = reinterpret_cast<GpuFetchTask*>(ud);
    t->exec->run_gpufetch(t);
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  py::class_<CpuMoeExecutor>(m, "CpuMoeExecutor")
      .def(py::init<int, int, int, int, int, int, int, int, int, int, uintptr_t, uintptr_t,
                    uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                    double, double, std::vector<int>, bool>(),
           py::arg("num_threads"), py::arg("num_layers"), py::arg("num_experts"),
           py::arg("top_k"), py::arg("hidden_size"), py::arg("inter_size"),
           py::arg("max_tokens"), py::arg("activation_id"),
           py::arg("apply_router_weight_on_input"), py::arg("weight_format"),
           py::arg("gate_up_ptr"), py::arg("down_ptr"), py::arg("gate_up_scale_ptr"),
           py::arg("gate_up_global_ptr"), py::arg("down_scale_ptr"),
           py::arg("down_global_ptr"), py::arg("gate_up_bias_ptr"),
           py::arg("down_bias_ptr"), py::arg("swiglu_alpha"), py::arg("swiglu_limit"),
           py::arg("core_ids"), py::arg("task_timing_enabled") = false)
      .def("create_task", &CpuMoeExecutor::create_task, py::arg("layer_id"),
           py::arg("num_tokens"), py::arg("x_ptr"), py::arg("ids_ptr"), py::arg("w_ptr"),
           py::arg("y_ptr"))
      .def("create_gpufetch_task", &CpuMoeExecutor::create_gpufetch_task,
           py::arg("layer_id"), py::arg("capacity"), py::arg("num_rows_ptr"),
           py::arg("row_ids_ptr"), py::arg("source_ptrs"),
           py::arg("staging_ptrs"), py::arg("row_bytes"))
      .def("submit_with_cuda_stream", &CpuMoeExecutor::submit_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("sync_with_cuda_stream", &CpuMoeExecutor::sync_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("gpufetch_with_cuda_stream", &CpuMoeExecutor::gpufetch_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("run_task", &CpuMoeExecutor::run_task, py::arg("task"),
           py::call_guard<py::gil_scoped_release>())
      .def("run_task_sync", &CpuMoeExecutor::run_task_sync,
           py::arg("layer_id"), py::arg("num_tokens"), py::arg("x_ptr"),
           py::arg("ids_ptr"), py::arg("w_ptr"), py::arg("y_ptr"),
           py::arg("run_pre_callback"), py::call_guard<py::gil_scoped_release>())
      .def("setup_prefill_batch", &CpuMoeExecutor::setup_prefill_batch,
           py::arg("max_prefill_tokens"))
      .def("prefill_batch_buffer_bytes", &CpuMoeExecutor::prefill_batch_buffer_bytes)
      .def("run_prefill_batch_sync", &CpuMoeExecutor::run_prefill_batch_sync,
           py::arg("layer_id"), py::arg("num_tokens"), py::arg("x_ptr"),
           py::arg("ids_ptr"), py::arg("w_ptr"), py::arg("y_ptr"),
           py::call_guard<py::gil_scoped_release>())
      .def("set_pre_run_callback", &CpuMoeExecutor::set_pre_run_callback,
           py::arg("callback"))
      .def("set_pre_run_callback_mode", &CpuMoeExecutor::set_pre_run_callback_mode,
           py::arg("mode"))
      .def("register_flag_task", &CpuMoeExecutor::register_flag_task,
           py::arg("slot"), py::arg("task"))
      .def("register_flag_gpufetch_task", &CpuMoeExecutor::register_flag_gpufetch_task,
           py::arg("slot"), py::arg("task"))
      .def("flag_served_count", &CpuMoeExecutor::flag_served_count, py::arg("slot"))
      .def("task_last_run_ns", &CpuMoeExecutor::task_last_run_ns, py::arg("task"))
      .def("set_task_timing", &CpuMoeExecutor::set_task_timing, py::arg("enabled"))
      .def("step_timing_snapshot_and_reset",
           &CpuMoeExecutor::step_timing_snapshot_and_reset)
      .def("gpufetch_stats", &CpuMoeExecutor::gpufetch_stats, py::arg("reset"))
      .def("gpufetch_error_code", &CpuMoeExecutor::gpufetch_error_code)
      .def("start_flag_coordinator", &CpuMoeExecutor::start_flag_coordinator,
           py::arg("ready_ptr"), py::arg("done_ptr"), py::arg("num_slots"),
           py::arg("pin_core"))
      .def("set_input_prequant",
           [](CpuMoeExecutor& e, bool v) { e.input_prequant = v; },
           py::arg("value"))
      .def("isa_name", &CpuMoeExecutor::isa_name)
      .def("prefill_batch_kernel_name",
           &CpuMoeExecutor::prefill_batch_kernel_name);
  // Detection seam for GPU-free dispatch tests. Pointer identity is checked here,
  // while the Python test controls only the CPUID results supplied to the selector.
  m.def("prefill_batch_kernel_for_isa_flags",
        [](bool has_avx512vnni, bool has_avxvnni) {
          const Nvi8Dispatch d = select_nvi8_dispatch(
              nvi8_tier_from_flags(has_avx512vnni, has_avxvnni));
#if CPU_MOE_X86 && defined(CPU_MOE_HAS_AVX512VNNI)
          if (d.batch_rows == batch_nvfp4_i8_avx512vnni_rows)
            return std::string("batch_nvfp4_i8_avx512vnni_rows");
#endif
#if CPU_MOE_X86
          if (d.batch_rows == batch_nvfp4_i8_vnni_rows)
            return std::string("batch_nvfp4_i8_vnni_rows");
#endif
          if (d.batch_rows == batch_nvfp4_i8_scalar_rows)
            return std::string("batch_nvfp4_i8_scalar_rows");
          return std::string("unknown");
        },
        py::arg("has_avx512vnni"), py::arg("has_avxvnni"));
  // GPU-free numerical seam. The detected ISA is safe to execute on this host;
  // both outputs traverse every K block in the same order and must match exactly.
  m.def("run_prefill_batch_rows_parity",
        [](uintptr_t rows_out_ptr, uintptr_t singles_out_ptr,
           uintptr_t packed_ptr, uintptr_t scale_ptr, uintptr_t globals_ptr,
           uintptr_t acts_ptr, uintptr_t act_scales_ptr, int R, int M, int K) {
          if (R <= 0 || M <= 0 || K <= 0 || K % 16 != 0)
            throw std::runtime_error(
                "rows parity dimensions must be positive and K divisible by 16");
          const Nvi8Dispatch d = select_nvi8_dispatch(detect_nvi8_tier());
          float e4m3[256];
          for (int i = 0; i < 256; ++i)
            e4m3[i] = e4m3_decode(static_cast<uint8_t>(i));
          float* rows_out = reinterpret_cast<float*>(rows_out_ptr);
          float* singles_out = reinterpret_cast<float*>(singles_out_ptr);
          const uint8_t* packed = reinterpret_cast<const uint8_t*>(packed_ptr);
          const uint8_t* scale = reinterpret_cast<const uint8_t*>(scale_ptr);
          const uint16_t* globals = reinterpret_cast<const uint16_t*>(globals_ptr);
          const int8_t* acts = reinterpret_cast<const int8_t*>(acts_ptr);
          const float* act_scales = reinterpret_cast<const float*>(act_scales_ptr);
          d.batch_rows(rows_out, packed, scale, globals, R, acts, M, K,
                       e4m3, act_scales);
          for (int r = 0; r < R; ++r) {
            d.batch(singles_out + static_cast<size_t>(r) * M,
                    packed + static_cast<size_t>(r) * (K / 2),
                    scale + static_cast<size_t>(r) * (K / 16),
                    fp16_to_f32(globals[r]), acts, M, K, e4m3,
                    act_scales);
          }
          return std::string(d.batch_name);
        },
        py::arg("rows_out_ptr"), py::arg("singles_out_ptr"),
        py::arg("packed_ptr"), py::arg("scale_ptr"), py::arg("globals_ptr"),
        py::arg("acts_ptr"), py::arg("act_scales_ptr"), py::arg("rows"),
        py::arg("activation_rows"), py::arg("hidden_size"));
  m.def("memops_probe", &cumemops_probe, py::arg("stream"), py::arg("scratch_addr"));
  m.def("memop_submit", &cumemop_submit, py::arg("stream"), py::arg("done_addr"),
        py::arg("ready_addr"), py::arg("slot"));
  m.def("memop_sync", &cumemop_sync, py::arg("stream"), py::arg("done_addr"),
        py::arg("slot"));
  // ABI capability marker: the highest ActKind this build implements in the
  // GENERIC epilogue. CpuMoeExecutor.__init__ probes it before requesting an act
  // id the epilogue must handle -- a prebuilt .so from before ACT_SWIGLUOAI
  // accepts a newer id without error and silently computes the wrong activation
  // (act_apply falls through to gelu_tanh); the probe turns a stale extension
  // into a loud rebuild instruction instead of wrong model outputs.
  m.def("max_generic_act_id", []() { return static_cast<int>(ACT_CLAMPED_SILU); });
}
