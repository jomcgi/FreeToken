# Supported models

FreeToken loads HF safetensors checkpoints directly (plus native GGUF for
Gemma-4). The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.8 / Qwen3.6 dense | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)), [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

### File-backed expert banks

`--moe-disk-layers` keeps the selected MoE layers as read-only mappings of their
checkpoint regions. Those layers decode on the CPU executor, while Linux loads only
touched expert pages into the page cache. The flag accepts the same explicit id list,
count, or fraction grammar as `--moe-cpu-layers`, for example
`--moe-disk-layers 48,49,50` or `--moe-disk-layers 0.25`.

DISK layers also prefill on the CPU executor by default. FreeToken first prefetches
the union of routed expert pages for the chunk, then computes only those routes without
copying the whole bank to the GPU cache. `--moe-disk-prefill copy` restores the prior
whole-layer pageable copy path for benchmarking. LOCKED and PAGEABLE layers keep their
existing GPU prefill copy behavior.

`--moe-disk-pager madvise` is the default and preserves that file-mapping behavior.
During decode, `--moe-disk-lookahead on` (the default) issues each DISK layer's
previous route set before the next step starts, then advises only newly routed experts
when that layer's current routing arrives. `--moe-disk-lookahead off` restores fully
reactive advice. The option is a no-op with `uffd` because its logical-row prefetch path
already manages residency differently.
On Linux, `--moe-disk-pager uffd --moe-pager-budget-gib 40` instead registers anonymous
bank regions with userfaultfd. The route hook accepts logical expert rows, deduplicates
the physical pages covering them, and installs missing pages with positional I/O before
compute. A process-wide userspace LRU evicts pages with `MADV_DONTNEED` at the configured
resident-byte ceiling. Decode status reports proactive and fault-driven logical fills,
installed pages, evictions, resident bytes, and a fill-latency histogram.

At offload startup, the host-memory governor reads `MemTotal` and `MemAvailable` from
`/proc/meminfo`. It reserves `max(8 GiB, 15% of MemTotal)` for the OS and the
expert-tier file cache by default, then fits the expert pin and UFFD pager budgets
under the remaining available memory. With neither budget explicit, the split uses
the measured 28:22 pin-to-pager ratio. `FREETOKEN_PIN_BUDGET_GB`,
`--moe-pager-budget-gib`, and `--host-cache-reserve-gib` remain explicit overrides.
Startup rejects an explicit pin plus pager sum above the fitted ceiling instead of
clamping either value.

UFFD mode requires `/proc/sys/vm/unprivileged_userfaultfd=1` or `CAP_SYS_PTRACE`.
Startup probes the kernel API and reports the current sysctl value on failure. It also
requires page-aligned anonymous mapping boundaries. Expert row sizes and FTW file
offsets may be arbitrary.
The `madvise` backend remains available on every supported platform and is the bench
baseline when comparing warm-decode fault counts.

The DISK tier requires FreeToken's per-layer FTW layout. Convert a raw
safetensors checkpoint first with `ft checkpoint`; raw safetensors and GGUF banks are
not supported. When expert banks exceed `FREETOKEN_PIN_BUDGET_GB`, an FTW checkpoint
automatically spills enough head and tail layers to DISK, following the assumption that
middle-layer traffic is higher. Non-FTW checkpoints use the same head and tail split for
the existing OS-locked fallback.

For traffic-aware selection, collect a representative decode window with
`--moe-collect-stats --moe-disk-layers 0`, then save the running server's profile:

```bash
curl -s http://127.0.0.1:1919/v1/moe-layer-profile > moe-layer-profile.json
```

The endpoint writes profile version 2. Its `layers` object maps every MoE layer id to
realized decode misses per step, and `expert_hits` contains one route-count array per
layer. Pass it on later boots with
`--moe-disk-layer-profile moe-layer-profile.json`; automatic spill picks the lowest
layer scores, with ties resolved by layer id. Legacy unversioned layer-only profiles
remain valid for whole-layer spill selection.

`--moe-hot-expert-budget-gib N` uses the version 2 `expert_hits` section to pin a
compact top-N expert bank inside every DISK layer. The same N is used for each layer and
is derived from the byte budget after accounting for whole-layer pins and the
`FREETOKEN_PIN_BUDGET_GB` ceiling. HOT pairs use the normal GPU slot cache and fused
copy plan. COLD pairs remain file-backed and run in the CPU executor; their weighted
partial is combined with the GPU partial through the hybrid decode merge. The default
budget is 0, which preserves pure whole-layer residency. Expert-granular residency
currently requires `--moe-disk-decode cpu`.

With `--moe-cold-fetch-max N > 0`, DISK layers apply a per-step residency policy:
when the distinct cold experts for the step number at most N, they are fetched into
non-protected GPU slots and the entire layer computes on the GPU, eliminating CPU
round-trip latency (~1.9 ms per trip). When the cold set exceeds the budget, up to N
cold experts are fetched and the remaining routes use the split path (GPU hot + CPU
cold). Fetched experts consume non-protected LRU slots and fall back gracefully if no
free slot or staging-ring capacity is available. This mode requires
`--moe-disk-decode cpu` and a non-zero `--moe-hot-expert-budget-gib`.

Explicit `--moe-disk-layers` always takes precedence for layer selection. A malformed
or incomplete layer-score profile produces a warning and falls back to the head and
tail split. HOT residency treats a missing or malformed per-expert section as a startup
error because guessing a partition would silently spend pinned memory. Startup logs
include selected layer ids, top-N, pinned bytes, and the profiled hot-pair rate.

### File-backed PLE table

Qwen3.8-Flash-Next defaults to `--ple-backend pinned`, which keeps its PLE n-gram
table in pinned host RAM and preserves the original CUDA graph path.
`--ple-backend cached --ple-cache-gib 8` instead maps the checkpoint read-only and
uses the GiB budget for a CLOCK-managed pinned hot-row bank. The bank is allocated
in fixed row-group slabs, so a future runtime resize can add or retire slabs without
reloading the checkpoint. Decode resolves row ids in the existing pre-replay hook,
installs only misses, updates fixed pinned slot ids, and keeps the captured gather
restricted to the cache bank. The `ple_hits`, `ple_misses`, `ple_evictions`,
`ple_installed_rows`, and `ple_hit_rate` fields are printed in decode status lines.

For traffic-derived warmup, pass a JSON row-frequency object such as
`{"123": 40, "987": 12}` with `--ple-cache-warm profile.json`. To collect that
format from live traffic, set `--ple-cache-profile-out profile.json`; the cumulative
profile is atomically refreshed at each decode status interval. Prefill installs the
chunk row union in bulk. If one chunk's union exceeds the cache capacity, that chunk
is logged and served through the pure staged disk path.

`--ple-backend disk` instead maps each PLE safetensors payload read-only, applies
random-access advice, and page-prefetches the deduplicated union of requested rows
before copying them through a small pinned staging bank. Disk PLE bytes are not
reserved from the expert-bank pin budget, so automatic MoE spilling can keep more
expert layers pinned.

`--ple-backend hmm` uses the same read-only per-shard mappings and zero pin-budget
reservation, but the in-graph GPU gather reads the mapped rows directly through Linux
Heterogeneous Memory Management. It requires the NVIDIA open GPU kernel modules. If the
startup readback probe fails, use `--ple-backend disk` as the staged fallback. HMM keeps
row ids on the GPU, performs no host staging or sampled-token synchronization, and keeps
CUDA graphs enabled. The reported `ple_major_faults` procfs delta includes host-side HMM
fault servicing, but procfs does not directly expose GPU-side page residency.

The PLE format is detected from the checkpoint tensors. Supported layouts are FP8 e4m3
with scalar or per-row scales, INT4 group-16 with fp16 scales, and NVFP4-style e2m1
group-16 with e4m3 scales plus `weight_scale_2`. Packed 4-bit rows use low-nibble-first
storage. All three formats support `pinned`, `cached`, `disk`, and `hmm`; the backend flag does not
select or override precision.

Disk PLE keeps CUDA graph decode enabled: a pre-replay host hook derives the next
n-gram row ids from request token history, stages their deduplicated rows, and updates
fixed pinned compact-id buffers read by the captured gather. Set
`FREETOKEN_PLE_DISK_NO_GRAPHS=1` to restore eager decode for debugging. Prefill stages
the full requested-row union for each chunk through the unchanged eager path.

### Native MTP speculative decoding

Qwen3.8-Flash-Next can load its checkpoint-native multi-token prediction head with
`--speculative-mtp on`. The default is `off`, which retains the previous model shape,
weight loading, scheduling, and CUDA graph behavior. The shipped head is one decoder
layer used for one draft step. FreeToken verifies `[seed, draft]` in one target-model
forward. A matching draft emits the draft and one target bonus token; a rejection emits
only the target token for the seed. `--mtp-draft-tokens` is fixed at `1` in this patch,
and any other value is rejected at startup.

This first implementation supports TP 1, greedy sampling, and one request in the decode
batch. Unsupported sampling modes and larger decode batches fall back to ordinary
one-token decoding. Enabling the feature selects the non-overlapped scheduler loop so
request-owned recurrent state can be restored safely after a rejected draft.

The target verification forward is eager and remains classified as decode, so target MoE
experts use the existing decode-routed path for both rows. GDN, PLE convolution, and QSA
run their established width-one decode operations in order inside that model call. Before
the call, FreeToken snapshots the request's GDN, PLE, and QSA continuation tensors. A
rejection restores that snapshot and recomputes only the seed row. This recovery preserves
bit-identical greedy state without a multi-position recurrent rollback, but reduces the
benefit of rejected steps. Decode logs report the snapshot copy time as `snapshot_us`.
CUDA graph capture remains unchanged when the feature is off. It is disabled while MTP is
enabled because the next draft step consumes target hidden state that is not currently
exposed by graph replay.

Pass `--speculative-mtp on` to `ft checkpoint` to store MTP tensors under a separate
optional tensor kind. The head is BF16 by default. Add `--mtp-quant nvfp4` to quantize
only its routed experts with the same packed E2M1, e4m3 group-16 scale, and fp16 row-global
layout used by the main NVFP4 experts. Serving reads `mtp_quant` from the FTW metadata, so
no serve-side precision flag is needed. An FTW checkpoint created without the MTP conversion
flag must be reconverted before it can be served with `--speculative-mtp on`. Default-off
conversions do not store the large head, and default-off FTW loads do not read or allocate it.

The MTP head is always GPU-resident, even when the target model uses an offload-family MoE
backend. It never enters the expert bank, LRU, CPU, or disk tiers. Startup logs its exact
resident tensor bytes, and the engine charges those bytes with the other dense weights before
sizing the target expert cache and KV pool. For the RadixArk geometry (`E=512`, `H=2560`,
`I=640`), the two BF16 expert tensors use `6*E*H*I = 5,033,164,800` bytes (4.6875 GiB).
The native NVFP4 expert banks use 1,419,509,760 bytes (1.3220 GiB), plus the unchanged BF16
non-expert portion of the head. The v1 draft attention keeps the current one-token
speculative chain but does not yet retain an independent MTP prompt KV and QSA cache.
Verification remains lossless, while acceptance can be lower than an implementation with
the complete draft cache.

Decode status lines report `drafted`, `accepted`, `acceptance rate`, and `tokens/step`.
Here `accepted` counts matching draft tokens; the target bonus token contributes to
`tokens/step` but not the acceptance numerator.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Qwen3.8-Flash-Next keeps its PLE table pinned by default. Use
  `--ple-backend cached --ple-cache-gib N` for a bounded pinned hot set, or
  `--ple-backend disk` when no persistent PLE row bank fits.
- Multimodal checkpoints are served text-only.
