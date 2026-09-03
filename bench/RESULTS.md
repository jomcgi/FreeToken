# FreeToken DISK expert-bank tier: implementation + bench results

Date: 2026-08-29. Branch: jomcgi/FreeToken `feat/moe-disk-tier` (5932caf).
Bench box: GCP g2-standard-32 spot (L4 24 GB, 125 GiB RAM, 375 GB local NVMe),
europe-west2-a. Total compute spend: ~$6.50.

## Correctness

- 11/11 disk-tier tests pass on GPU, including the bitwise parity test
  (disk-mapped vs pinned banks through the real `_cpu_moe` executor) and the
  CUDA copy-plan skip.
- `tests/moe` + `tests/engine`: zero branch-only failures vs upstream/main on
  the same box (45 env-caused failures identical on both).

## Qwen3.6-35B-A3B-NVFP4 (banks ~17 GiB, all RAM-cached, 0 major faults)

| config | warm tok/s | delta |
|---|---:|---|
| baseline-pinned | 56.2 | |
| disk4 | 39.5 | -30% |
| disk8 | 31.7 | -44% |
| disk16 | 22.8 | -59% |

Measures pure CPU-executor cost of a routed layer: ~1.9 ms/token/layer.

## Qwen3.8-Flash-Next-NVFP4 (first Flash serve through the fork)

FTW = 72.7 GiB (banks 63.5); PLE table (47.7 GiB) loads from raw safetensors
shards, which must sit next to the FTW dir (symlinks work) - `ft checkpoint`
does not convert the PLE.

| config | warm tok/s | delta |
|---|---:|---|
| baseline-pinned | 11.7 | (L4; 4090 upstream benchmark = 36) |
| disk4 | 10.9 | -7% |
| disk8 | 9.6 | -18% |
| disk16 | 8.0 | -31% |

Flash CPU-layer cost ~0.9 ms/token/layer - much gentler than 35B because
Flash's per-token time is larger. All RAM-cached (125 GiB box).

## node-4 simulation (the actual question)

`FREETOKEN_PIN_BUDGET_GB=52` + cgroup `MemoryMax=64G` (verified pegged at
64.0 GiB): 47.7 PLE pinned + 3.97 GiB banks pinned (3 layers) + 59.5 GiB
file-backed (45 DISK layers).

**Decode-side prefetch mechanics are sound** (MADV_RANDOM + page-deduped
WILLNEED; counters live in the decode log). **Prefill is the blocker**: any
prefill batch triggers the per-#112 whole-layer pageable copy for every
non-pinned layer - all 59.5 GiB of disk banks per batch, regardless of prompt
length. A 6-token prompt took 17+ minutes under eviction pressure. Unusable at
this spill ratio.

## Verdict for node-4 (RTX 4090 24 GB + 64 GB RAM)

- Decode: plausibly ~20-27 tok/s with ~10-12 spilled layers (36 baseline minus
  CPU-layer cost), less with real fault traffic. Promising.
- Prefill: current path makes the tier unusable. The needed upstream change:
  prefill for DISK layers must stream only the routed experts (or run prefill
  on the CPU executor), never whole-layer copies.
- PLE: still needs upstream's PLE-to-disk work (roadmap item) - our tier only
  covers expert banks.

## Next moves

1. Patch the prefill path for DISK layers (stream routed experts only), rebench.
2. Or hand these findings + branch upstream (issue #214 / roadmap #79) and let
   them fold it into their PLE-disk work.

## Infra notes

- VM `freetoken-bench` STOPPED (not deleted). Persistent disk `ft-data` (300 GB,
  ~$33/mo if kept) holds models, FTW, venv, results; `~/recover.sh` rebuilds
  the wiped local SSD in ~15 min after any restart/preemption.
- Delete when done: `gcloud compute instances delete freetoken-bench --zone=europe-west2-a`
  and `gcloud compute disks delete ft-data --zone=europe-west2-a`.

---

# Round 2 (2026-08-29, later): CPU prefill + disk-backed PLE

Commits: 0fe89d5 (DISK-layer prefill on the CPU executor, --moe-disk-prefill),
ef052ba (--ple-backend disk: mmap'd PLE shards + staged UVA gather, CUDA
graphs disabled as v1). All on jomcgi/FreeToken feat/moe-disk-tier.

Node-4 config (Flash-Next NVFP4, 64G cgroup, FREETOKEN_PIN_BUDGET_GB=52, L4):

| metric | round 1 | + CPU prefill | + PLE disk |
|---|---:|---:|---:|
| layers spilled | 45/48 | 45/48 | 9/48 |
| prefill 441 tok (warm) | 17+ min for 6 tok | 29.2s (15 tok/s) | 5.5s (80 tok/s) |
| decode | wedged | 1.8 tok/s | 4.5-4.8 tok/s |
| warm majflt/step | n/a | 1047 | 2.5 |

Findings:
- Disk tier is SILENT at steady state after both patches (2.5 faults/step).
- Decode is now bound by the graphs-disabled eager path (~100 ms/step), not
  by disk: patch 4a (graph-compatible staged gather) is the next lever;
  projected ~9-10 tok/s on L4, ~20+ on the 4090.
- UFFD pager (patch 4b) re-scoped: capacity play (bigger-than-RAM banks,
  row-granular residency), not a current-speed play.

Bench-harness lessons: sudo secure_path strips venv (ninja JIT failures);
non-editable reinstall shadows the repo (stale-code test failures); orphaned
servers stack and CUDA-OOM each other (bench2.sh now traps EXIT and retries
warmup only while the server pid is alive).

---

# Round 3 (2026-08-29/30): HMM, staging fast path, spill selection

Commits: c93c2e9+85b5695 (graph-replayable staged gather), 04f6d88 (staging
vectorization), b48297d (--ple-backend hmm), 2e66f7f (miss-aware spill).

Decisive numbers (L4, warm decode):
- pinned PLE reference: 11.7 tok/s
- staged PLE, any variant: ~4.9-5.1
- HMM PLE, 0 spill, uncapped: 5.7  <- ~105ms/step GPU-side, NOT cold faults
  (pre-touch neutral), NOT host staging (staging_us=0), NOT spill-related
  (0-spill identical to 9-spill). The GPU re-faults file-backed mappings
  every replay on this driver/silicon.
- profile-guided spill + pretouch: neutral at bs=1.

Shape economics rewrite: -6+fork needs a pinned hot-row PLE cache (patch 9
candidate) or better Blackwell HMM behavior; -12 + primitive-ai quantized
table (28.8GB NVFP4-g16, fits 45GB RAM pinned) is the value pick at $0.53/hr.
Quality gate for the 4-bit table still pending (quality.sh comparison).

Quality harness (chat endpoint): arith/recall/reason PASS through full tier;
longgen needs thinking-budget fix (model reasons by default, xhigh).

---

# Round 4/5 (2026-08-30): feature-complete milestone

Fork: jomcgi/FreeToken feat/moe-disk-tier, 21 commits, based on upstream
58f4b9e (upstream unmoved since branch). All features live and tested.

## Shipped since round 3

- 4f137db concurrency hardening (5-target audit, 1 real fix: staging bank
  vs explicit graph sizes; the report doubles as the concurrency doc)
- fa280cd quantized PLE tables (FP8 / INT4 g16 / e2m1 g16, auto-detected)
- 5164b25 pinned hot-row PLE cache (--ple-backend cached, CLOCK eviction,
  warm profiles, full stats)
- 906574a per-shard PLE global scales (real-checkpoint layout fix)
- 4f0be4d + 410a494 MTP speculative decode (greedy bs=1, lossless;
  LM-head row-selection and is_greedy fixes; latent QSA ring bug fixed)

## Key numbers (L4, 64G-cap node-4 envelope unless noted)

| config | decode tok/s | notes |
|---|---:|---|
| pinned PLE reference | 11.7 | uncapped |
| HMM PLE, 9-layer spill | 5.7 | disk silent |
| cached PLE (fp8), cold | 4.2-4.8 | 62->70% hit rate warming |
| cached PLE aggregate x4 / x8 | 10.9 / 11.6 | batching amortizes per-step cost |
| e2m1 table + cache | 2.4 | packed miss-install ~4x cost (open item) |
| MTP on | 0.20 | 31.7% acceptance; BF16 draft experts stream per draft (open item) |

Quality: arith/recall/reason PASS through the full tier (chat endpoint);
longgen needs thinking-budget handling in the harness.

## Open performance items (ranked)

1. MTP economics: quantize/pin the 4.69 GiB draft experts (fits VRAM),
   graphs under MTP, persistent draft KV (acceptance 32% -> ?). Until then
   MTP stays default-off.
2. Packed-table miss-install cost (~4x fp8) — batch data+scale copies.
3. Hot-row cache long-uptime hit-rate measurement (short benches cap ~70%).

## Target configs (current best knowledge)

- node-4 (24G VRAM / 64G RAM / 1TB NVMe, bare metal post-migration, open
  driver): budget-52 + cached PLE (quantized table) + profile-guided spill;
  batched agent traffic ~2x single-stream. GPU-fetch decode is the next
  node-4-specific lever.
- G4 -6 (96G VRAM / 22G RAM): banks in VRAM; cached PLE on the 28.8G e2m1
  table (page-cache friendly) once miss-install is optimized; batching
  carries throughput. -12 with the table fully pinned needs zero further
  work today.

Bench-harness debt: zombie spawn-children survive pkill patterns (three
incidents); kill by venv path. pkill self-match keeps killing SSH sessions
(bracket the pattern or split kill/launch).

---

# Round 5 (2026-08-30/31): Blackwell (G4) and the four outcomes

Hardware learned the hard way: GCP G4 small shapes are FRACTIONAL vGPU
slices of the RTX PRO 6000 (g4-standard-6 = 12 GB "DC-1-12Q"; -24 = 48 GB
"DC-2-48Q"; full 96 GB needs -48). vGPU guests refuse the open kernel
module AND the plain GRID driver: only GCP's grid-gcp build works
(cuda_installer.pyz fetches it). vGPU also rejects CUDA VMM, so
FreeToken's expandable_segments allocator must be disabled
(PYTORCH_CUDA_ALLOC_CONF=backend:native) - patch candidate: auto-detect.

## The headline table (Flash-Next 125B-A6B NVFP4, full tier)

G4 -24 slice (48 GB VRAM + 88 GB RAM, $1.05/hr spot), e2m1 quantized table
hot-row cached (8 GiB pin, 62% hits cold and warming), ALL expert banks
RAM-pinned, 48 GB VRAM LRU cache, CUDA graphs on:

- prefill 441-token prompt: 1.3 s (340-352 tok/s)
- decode single stream: ~29 tok/s
- aggregate: 81 tok/s at 4 streams, 88.6 at 8
- majflt/step: 0 (the tier is invisible at steady state)

## Outcome 1: what this means for a 96 GB PRO 6000 at home

The -24 slice is HALF the card plus a vGPU tax. At home (full 96 GB, no
vGPU, open driver so HMM also works): banks go fully VRAM-resident,
removing the last host-path costs -> project 100-150+ tok/s aggregate,
400+ prefill. The tier still earns its keep there for GLM-5.3-class
models (150 GB banks > any single card) via the RAM tier + future pager.

## Outcome 2: what this means for a 4090 at home

The L4 rounds (rounds 1-4) are exactly the 4090's shape: 24 GB-class VRAM
+ 64 GB RAM. Measured there: the full stack serves the 125B model in that
envelope at 5.7-11.7 tok/s single / 11.6 aggregate on ~1/3 of a 4090's
bandwidth. Scaled to the 4090 (and its native-NVFP4 sibling numbers
upstream: 36 tok/s plain offload), expect ~20-35 tok/s with the tier and
quantized table - a genuinely usable private 125B server from a gaming
card, IF the RAM is 64 GB+ and the table is quantized.

## Outcome 3: disk vs RAM (the measured ladder)

- RAM-pinned PLE: 11.7 tok/s reference (L4).
- ANY disk-backed PLE: ~5 tok/s - a flat ~105 ms/step tax, and NOT from
  I/O: the GPU re-faults file-backed mappings every CUDA-graph replay
  (pretouch does nothing, staging_us proved the host idle). Disk tables
  are for capacity, never for speed; quantize (47.7 -> 28.8 GB) and pin
  or hot-row-cache instead.
- Expert banks on disk are DIFFERENT: with router-driven page-deduped
  MADV_WILLNEED prefetch, spilled banks run at <10 major faults/step -
  effectively free at steady state. Disk is fine for cold experts, fatal
  for per-token lookup tables.
- Prefill through disk tiers must never copy whole layers: the naive path
  took 17 MINUTES for 6 tokens; routing prefill through the CPU executor
  (touched experts only) made it 5.5 s for 441 tokens.

## Outcome 4: RAM tiering (what actually works)

- Pinned hot-row cache over a Zipf lookup table: 62% hit rate cold,
  70%+ within minutes, miss cost now 1.2x fp8 after interleaved-copy
  fix. The cache budget is the RAM allocation knob.
- Pin-budget auto-spill + per-layer traffic profiles pick WHICH layers
  leave RAM (profile endpoint shipped; matters at higher spill counts).
- Batching is the great equalizer: fixed per-step tier costs amortize
  across concurrent streams - 4.9 -> 11.6 tok/s (L4 x8), 29 -> 88.6
  (Blackwell x8). An agent factory should never serve bs=1.
- MTP speculative decode: lossless and working, but parked - draft-head
  streaming + verify routing made it net-negative until the resident-head
  + batched-verify follow-ups; only worth revisiting on big-VRAM boxes.

## Economics postscript

Cloud GPU self-hosting loses to hosted APIs at every measured shape
(GLM-5.3-Flash: $0.15/$0.50 per M list). The winning architecture:
GLM-Flash API as the 24/7 orchestrator (~$10-30/mo), subscription pools
(Claude/Codex) as implementation muscle, owned metal for private lanes.
The fork's value is metal utilisation, not cloud arbitrage.

Fork: jomcgi/FreeToken feat/moe-disk-tier, 25 commits, based on upstream
58f4b9e. Total experiment spend: ~$39 of $50.

---

# Attribution: what FreeToken had, what we added

## Upstream FreeToken (before the fork - credit where due)

- The engine itself: MoE offload backend with a GPU LRU expert-slot cache
  and PCIe miss streaming; CUDA-graph decode; radix prefix cache.
- Qwen3.8-Flash-Next support (#257): hybrid GDN/QSA layers, PINNED host
  PLE table with in-graph UVA gather, NVFP4/FP8 kernels, FTW fast-load
  format, MTP weights present-but-dropped.
- Per-layer host-bank residency (#112): PINNED/LOCKED/PAGEABLE classes,
  the CPU MoE executor (C++ worker pool, flag-handshake coordination with
  captured graphs), pin budgets for WSL/WDDM, --moe-cpu-layers.
- Elastic VRAM pools (live cache rebuild), bandwidth calibration.

Everything we built stands on those seams - especially #112's residency
plumbing and the CPU executor.

## Our fork (25 commits on feat/moe-disk-tier)

1. **DISK residency for expert banks** - read-only file mmaps of FTW
   regions, MADV_RANDOM + router-driven page-deduped MADV_WILLNEED before
   compute. The "disk banks are free" result.
2. **CPU prefill for DISK layers** - replaced whole-layer pageable copies
   (17 min for 6 tokens) with routed-experts-only CPU compute (5.5 s for
   441). Plus --moe-disk-prefill escape hatch.
3. **Four PLE-off-RAM backends**: staged gather; CUDA-graph-replayable
   staging (bit-exact host reimplementation of the n-gram hash);
   vectorized staging (process_vm_readv batched copies); HMM direct GPU
   reads of file mmaps; and the pinned hot-row cache (CLOCK eviction,
   warm profiles, hit-rate stats).
4. **Quantized PLE tables** - INT4/e2m1 group-16 in-kernel dequant,
   per-shard global scales folded at load, interleaved data+scale
   miss-install (1.2x fp8 cost).
5. **Miss-aware spill selection** - /v1/moe-layer-profile endpoint +
   profile-guided lowest-traffic layer choice.
6. **Concurrency hardening** - bs>1 audit with one real fix (staging
   capacity vs explicit graph sizes).
7. **MTP speculative decode v1** - head loading, greedy draft/verify
   (lossless), resident+quantizable head, decode-routed verify; parked at
   break-even pending batched verify.
8. **Compat + measurement corpus** - GCP vGPU survival guide (grid-gcp
   driver, VMM-free allocator), is_greedy semantics fix, and the
   five-round benchmark methodology itself.

Upstreaming posture: items 1-5 are coherent PR candidates if we choose;
the measured findings (GPU-refault law, prefill pathology) are useful to
upstream regardless of code.

## node-4 BARE METAL (2026-08-31): the real 4090 + 64GB numbers

Hardware: RTX 4090 24GB, Ryzen 7800X3D 8C/16T, 64GB DDR5, 1.9TB NVMe
(2.5GbE). Driver: NVIDIA OPEN kernel module 610.57.04 + CUDA 13.0. Model:
Qwen3.8-Flash-Next NVFP4 (FTW 72.7 GiB) + e2m1 quantized PLE table (27 GiB).
Conversion on bare NVMe: 94 seconds (vs OOM saga on cloud boxes).
These numbers replace the L4-proxy projections. Serving dir: e2m1 (quantized
table, sidecar discovery). All runs `--moe-disk-prefill cpu`,
`--moe-backend offload --moe-cache-auto`, max-running-requests 8.

### Config sweep (load.sh: x1 = 1 stream x 256 tok, x4/x8 = 192 tok/stream)

| config | pinned banks | PLE | x1 | x4 | x8 agg | warm majflt/step |
|---|---|---|---:|---:|---:|---:|
| budget52 + cached10 | 33.0 (25 lyr) | cached 10G | 3.8 | 7.2 | 9.6 | ~96,000 |
| budget40 + cached6 | 33.0 (25 lyr) | cached 6G | 9.3 | 14.0 | 26.8 | ~4,400 |
| budget44 + cached8 | 36.0 (30 lyr) | cached 8G | 5.1 | 11.7 | 17.3 | ~12,500 |
| budget36 + cached6 | 30.0 | cached 6G | 9.1 | 16.6 | 24.6 | ~6,300 |
| budget40 + **hmm** | 39.7 (30 lyr) | HMM | 10.8 | 26.5 | 32.8 | ~670 |
| budget46 + hmm | 42+ (34 lyr) | HMM | **17.3** | 24.0 | 30.4 | ~4,400 |
| budget40 + hmm + profile | 39.7 (30 lyr) | HMM | 12.3 | **26.9** | **33.9** | ~580 |

Prefill (441-tok prompt, budget52+cached10 run): 18.3 tok/s cold, 53.8 warm.
Warmup to first served token: 32-110s from cold NVMe.

### Laws measured on bare metal

1. **The page cache is a tier: pin budget competes with it.** 52G pinned on a
   64G box leaves ~0 page cache for the ~57G of disk-resident data; every
   disk access refaults (96k majflt/step, 3.8 tok/s). Dropping to 40G pinned
   freed ~16G of cache and tripled throughput. Size pin budget to leave the
   spill working set cached, not to maximize pinned bytes.
2. **HMM beats the cached/staged PLE backends on bare metal open-driver**:
   x4 nearly doubled (14.0 -> 26.5). The L4 measurement of ~105ms/step HMM
   refault tax was a vGPU artifact, not an HMM property. HMM also frees the
   PLE cache's pin allocation, so more banks pin under the same budget.
3. **Workload-dependent optimum**: single-stream wants more pinned banks
   (budget46: 17.3 x1); concurrent throughput wants more page cache
   (budget40: 33.9 x8). One knob, two optima - pick per deployment.
4. **Profile-guided spill** (--moe-disk-layer-profile, needs
   --moe-collect-stats on the profiling server) kept the same head+tail
   spill set but still cut faults ~15% and set the x4/x8 records.
5. **GPU-fetch decode is VRAM-bound negative on 24GB**: correct (5/5
   quality) but 10.4/9.4/13.1 - the ~990 fills/step x ~2.6MB/expert over
   PCIe plus slot-cache thrash across 43 layers loses to CPU decode + page
   cache. Keep the flag for >=48GB VRAM boxes; do not use it on a 4090.

### Quality (quality.sh + no-thinking longgen variant)

5/5 PASS (arith, recall, reason, longgen-diversity, longgen-length) on every
config benched: cached b40c6, b36c6, hmm40, hmm46, hmm+profile, gpufetch.
The e2m1 quantized PLE table is validated at smoke level: first quality gate
it has ever passed, closing the top open item from the cloud rounds.

### New fork work landed tonight (branches, pending merge)

- feat/gpu-fetch-decode (95648ea + edce35e fix): --moe-disk-decode gpufetch.
  One correction round: install indices had to be bounded to the staging
  ring capacity (kernel requires equal-length index buffers). Benched:
  negative on 24GB (law 5), retained for big-VRAM targets.
- feat/prefill-overlap-split (f60d70a): per-layer prefill overlap under
  split residency (pinned layers overlap, DISK layers CPU-prefill and chain
  the next pinned layer's prefetch). UNVALIDATED on hardware yet.
- feat/uffd-pager (66134e0): --moe-disk-pager uffd, userfaultfd +
  io_uring row-granular pager with byte-budgeted LRU, targets the 4KiB
  fault amplification (27M pages requested for ~6M rows needed per load
  run). Linux-only. UNVALIDATED on hardware yet.
- MTP batched verify (11b/11c): DEFERRED - two Codex workers died silently
  mid-task (the GDN state rollback makes this the hardest patch). Partial
  attempt-1 patch preserved in the worktree. Speculation stays parked at
  break-even.

### Ops notes (bare metal specific)

- Node was dual-homed (2.5GbE + WiFi, same subnet): ARP flux routed bulk
  traffic over WiFi at ~15-20 MB/s while the wire crawled. Downing wlp13s0
  took the HF download from 21 to 107 MB/s. Check `ip route show default`
  count FIRST on any homelab box.
- Xet-backed HF repos: the hf CLI's Xet path wedged silently mid-download
  (0 B/s, progress frozen); HF_HUB_DISABLE_XET=1 + hf_transfer recovered it.
- ft serve leaves the torch-dist port (server port + 1) held by an orphaned
  scheduler child if the frontend is SIGKILLed: EADDRINUSE on relaunch.
  fuser -k both ports between serves.
- flashinfer JIT needs ninja AND nvcc on PATH at pytest time; a venv
  python invoked by absolute path does not put .venv/bin on PATH (219
  cascade failures that look real but are env).

## Overnight optimization round (2026-08-31, continued): four patches + close-out

All on the bare-metal champion base (budget40 + HMM PLE + layer profile,
MRR=8) unless stated. Suite state at close: 1664+ passed, 6 known
ULP-parity fails only. Everything merged to feat/moe-disk-tier.

| lane | result | verdict |
|---|---|---|
| prefill overlap (split residency) | warm prefill 54 -> **116 tok/s**; x1 12.3 -> 16.8 with x4 27.6 / x8 33.8 held | **BIG WIN, merged.** Collapsed the x1-vs-concurrency config tradeoff |
| expert-dedup CPU decode | x8 33.9 -> **34.0**; measured dedup_ratio 1.39 | marginal win, merged (stats alone worth it) |
| UFFD pager (after page-granular correction) | faults 580 -> **12.9/step**, 17.5GiB fully resident, 0 evictions; x8 30.1 | works; slightly under madvise throughput, so madvise stays default. UFFD = the bigger-than-RAM capacity lane |
| GPU-fetch decode | 10.4/9.4/13.1 | negative on 24GB (slot-cache thrash), merged default-off for big-VRAM targets |
| concurrency ladder | x16 does not fit (GDN state scales per slot); x12 peaks 32.9 < x8 | MRR=8 is the 24GB sweet spot |
| MTP batched verify | two Codex workers died mid-task | DEFERRED (hardest patch: exact GDN rollback) |

### Endurance + power (final merged tier, sustained x8 x 512-tok gens)

31.0-31.2 tok/s aggregate held across consecutive 130s runs.
GPU power: **avg 125W, peak 154W** (450W card at ~28% power) -> ~0.25
tok/s/W. Decode is host/PCIe-bound; the GPU coasts. A 125B-A6B model
serving 31 tok/s at wall power comparable to a bright lightbulb is the
efficiency headline.

### Final validated bests (4090 24GB + 64GB RAM, quality 5/5 throughout)

- warm prefill 116 tok/s (441-tok prompt)
- x1 16.8 / x4 27.6 / x8 34.0 tok/s (single config)
- sustained x8: ~31 tok/s at 125W avg GPU power

### Remaining known-but-unpursued items

- UFFD for the PLE table (marginal: PLE faults already ~13/step warm).
- CUDA graph-bs tuning (low value at MRR=8 optimum).
- MTP speculative decode (parked, Sol-failed-twice; would multiply x1).
- +64GB RAM: still the biggest single lever (~36 x1 all-pinned upstream).

### Single-stream hunt (stage 15, same day)

- **budget46 + merged tier: x1 19.5 tok/s (new interactive champion), x8 33.2.**
  Prefill-overlap compounds with the bigger pin budget.
- budget50 + uffd(6GiB): x1 10.0 - the pager budget was far below the ~24G
  spill working set, so every install evicted (1.6M evictions). Law: UFFD
  pays only when its budget covers the working set; it is the
  bigger-than-RAM lane, not a throughput lane.
- hybrid backend (PCIe-fetch split): x1 13.8 / x8 29.9 - loses to plain
  offload, consistent with the gpufetch result: on this box CPU compute
  beats PCIe expert fetch.

Interactive serving config of record: FREETOKEN_PIN_BUDGET_GB=46, HMM PLE,
layer profile, MRR=8: **19.5 x1 / 116 prefill / 33 x8**.

## Predictive round (2026-08-31 evening): expert-granular residency breaks both ceilings

- **Expert-granular residency** (--moe-hot-expert-budget-gib, v2 per-expert
  profiles): profile-hot expert rows of DISK layers pin under a byte budget
  and route through the GPU slot cache; the cold tail keeps CPU decode.
  **x1 19.5 -> 23.1, x8 34.0 -> 42.2** (profile-matched workload, hot rate
  72.5%, distinct CPU experts/step 58 -> 18). Broke the computed DDR wall
  by moving the Zipf head off the CPU tier. 2D sweep: pin40 + hot6 is the
  true optimum (44/4 and 36/8 both lose - the same three-tier balance law).
- Sustained real-world (diverse-traffic profile, hot rate 62%): x1 ~21,
  **x8 35.6-37.4 tok/s at 138W avg / 197W peak**, quality 5/5.
- Profile-workload fit matters: hot rate 72% -> 62% when traffic diverges
  from the capture. Direct motivation for online hot-set adaptation
  (top open item).
- **Lookahead prefetch**: negative (48% next-step prediction accuracy
  nearly doubles advised pages). Merged default-off with stats.
- **MTP K=1 re-test on the hot config**: still negative (12.6 vs 18.0).
  Final verdict: speculation does not pay while any meaningful share of
  marginal-token cost lands on the CPU tier. Closed.

### Numbers of record (4090 24GB + 64GB RAM, 125B-A6B, quality 5/5)

| metric | value |
|---|---|
| warm prefill (441 tok) | 116 tok/s |
| x1 peak / sustained-diverse | 23.1 / ~21 tok/s |
| x8 peak / sustained-diverse | 42.2 / ~37 tok/s |
| GPU power under x8 | 138W avg |
| day-1 baseline for contrast | 3.8 x1 / 9.6 x8 |

### Open items after this round

1. Online hot-set adaptation (evidence: the 72->62% drift above).
2. CPU/GPU layer pipelining (structural x1 lever, unattempted).
3. UFFD stays the bigger-than-RAM capacity lane.

## Adaptation round: the hierarchy tunes itself

**Online hot-set adaptation** (--moe-hot-adapt-interval-steps): decayed
per-expert counters, bounded three-phase background swaps (retire, copy,
flip - no torn mappings). Under drifted traffic hot_pair_rate recovered
**62.6% -> 73.3%** across four adaptation windows while static stayed at
62%; quality 5/5. Profile-free startup works (all-cold, warms itself), so
deployment no longer needs a capture step. One correction round
(inference-mode guard on the background copy thread). Known cosmetic bug:
hot_swaps/interval stat prints 0.00 despite live swaps.

Ops: node-4 CPU governor pinned to performance (was powersave/EPP-perf);
DDR confirmed at 6000 MT/s EXPO; GPU never throttles (138W of 450W) - the
step is host-bound, watts are not the constraint.

## Pipelining investigation (program close)

--moe-step-timing (merged) measured the step anatomy at the final config
under x8: cpu_head+cpu_tail ~57-100ms per interval vs gpu_mid ~19-40ms,
overlap already 20-45ms. The doorbell was already late-bound; the engines
overlap wherever the layer dataflow permits. The only remaining overlap
(cross-wave staggering) requires breaking the scheduler's
one-forward-in-flight contract - documented as the seam, out of scope.
Timing mode costs ~10% (event syncs): diagnostic only.

PROGRAM CLOSED. Every identified software lever on this hardware is now
implemented, measured, or bounded with a documented reason.

## Hardware predictions, revised by the software program

The program changed the hardware ranking, not just the numbers:

- **+64GB RAM (128 total): downgraded from "obvious" to "nice".** The old
  case was 9.3 -> ~36 x1 (4x). Software closed most of it: 23.1 -> ~36 is
  now +55%, and the all-pinned config no longer gets exclusive credit for
  GPU-computing hot routes - we already do that for 72% of them.
- **EPYC 8-channel: thesis weakened for this model class.** Its case was
  "aggregate is DDR-bound, buy channels"; expert-granular residency broke
  that ceiling by migrating the Zipf head to the GPU tier instead. The
  8-channel build retreats to true capacity plays: models whose COLD TAIL
  alone overwhelms 2-channel bandwidth.
- **RTX PRO 6000 96GB: thesis strengthened.** GPU compute of hot routes
  is where throughput lives; 96GB holds the entire hot working set,
  GPU-fetch (measured negative <48GB, kept for exactly this) becomes the
  cold-tail path, and the GDN-state concurrency cap (the x16 wall) lifts
  with VRAM. Every shipped mechanism compounds with this card.
- **DSV4-class bigger models: the registered 2-6 tok/s "unusable"
  prediction is now stale-pessimistic.** It assumed layer-granular spill
  with CPU decode dominating; hot-expert pinning + online adaptation
  change that math IF their routing is also Zipf. Re-register a
  prediction before testing. (Tier still verified on qwen4_exp only.)

## Serving program (round 2, 2026-09-01): from benchmark box to production endpoint

The throughput program made the box fast; this round made it a SERVER.
Trigger: node-4 went live as the platform's inference endpoint
(tailscale bridge to GKE) and Joe's actual chat experience exposed
everything a load generator cannot.

### Commit trajectory (feat/moe-disk-tier)

- 7641002 expose --linear-state-cache-ratio (multi-turn prefix reuse
  died because GDN snapshot slots were unreachable-by-flag; 48 -> 64)
- 3eeecbd disk-backed prefix state cache ("LMCache lane": 500G NVMe
  store, fingerprint-keyed, crash-safe, async write-back)
- 7da2bc6 bench/realworld.py scenario scorecard (the acceptance gate:
  conversation TTFT, agent resume across restart, contention, JSON
  validity)
- 27d3f91 request priority scheduling (priority field/header, aging
  bound, reorder at forward boundaries)
- 8547bd1 guided decoding via optional XGrammar (response_format
  json_object/json_schema, tool_choice required)
- 9c21d11 fix: first constrained request crashed the scheduler
  (GuidedState wrappers vs raw matchers)
- pending: session-conditioned expert prefetch (park each session's
  routed-expert profile with its state; prefetch at admission; protect
  live sessions' experts from LRU thrash), prefill-transient OOM guard

### Scorecard: broken first run vs deployed stack

| leg | first run | deployed |
|---|---|---|
| conversation first-turn TTFT | 13.3s | 7.2s |
| follow-up TTFT p50 | 4.2s | 4.1s |
| agent 10k-context cold TTFT | 870s | 14.8s |
| post-restart session resume | - | 13.8s (restarts ~free) |
| interactive TTFT under load | 86s | 23.0s |
| structured output | engine crash | 5/5 valid JSON |

Live UX proof: a 6.3k-token conversation matched 6272 cached tokens,
prefilled 24, streamed within ~2s. Joe: "Wow!! immediately streaming at
a useable speed."

### Production lessons (each cost real debugging)

1. **Hidden client requests are the silent killer**: Open WebUI fires
   2-4 auxiliary LLM calls per message (title/tags/follow-ups); on a
   one-forward engine they queue ahead of the user AND their prompts
   thrash the GDN snapshot slots, zeroing prefix reuse. Fix: disable at
   env level; structurally, session-aware eviction protection.
2. **KV is 64KB/token on this model, not 24KB** (the 0.19GiB/8256 log
   line undercounts). True 24G ceiling: ~100k tokens single-lane
   (1568 pages + activation headroom); 106k booted but OOM'd on GDN
   conv-prefill transients under load. 1M context = 64G KV = 96G-card
   or host-KV territory, not 48G as first estimated.
3. **A request-level OOM kills the scheduler process** (prefill
   transient at full KV) - the frontend survives as a zombie answering
   GETs. Guard patch queued: fail the request, not the process.
4. Session park/restore to NVMe makes restarts and (by extension) spot
   preemptions nearly free: the KV-swap-single-lane-session-multiplex
   pattern is the cloud-transition design (L4 spot ~$0.25/hr mirrors
   the 4090's 24G exactly).
5. Chunked prefill trades: 512-token chunks re-read disk banks ~16x
   (congestion collapse, measured); 8192 blocks interactive for tens of
   seconds. 1024-2048 is the band; adaptive chunking remains future work.

### Model-bench attempt (aborted, rerun queued)

First run at concurrency 3 collided with the 32k KV pool (agentic
contexts + 16k max_tokens headroom per stream): instant harness errors.
Lesson: bench concurrency must respect kv_pool / (context + max_tokens).
Rerun queued serial on the finished stack.

## Session-prefetch + guard round (2026-09-01, goal pipeline)

- **Session-conditioned expert prefetch MERGED + DEPLOYED**: each session's
  routed-expert profile (few KB) parks with its state; admission-time
  prefetch turns queue wait into warm-up; live sessions' experts carry
  bounded LRU protection. Advisory invariance PROVEN on hardware (the
  worker's CUDA test failed at construction three ways - Module .to,
  fp32 inputs, fp32 banks - and only tested anything after repair; 1778
  total green).
- **Request-level OOM guard MERGED + DEPLOYED**: forward OOM aborts the
  request with an actionable 503 and continues; decode OOM sheds the
  youngest and retries; poisoned CUDA context exits nonzero for systemd.
  137 scheduler/server tests green. Live-fire at the crash geometry did
  NOT reproduce the 07:19 OOM (fragmentation-dependent) - the guard is
  unit-proven, standing watch.
- **100k context validated end to end**: 93,430-token needle recall PASS
  (exact retrieval), profile of record: --num-pages 1568 (100,352 tok),
  MRR 1, --max-extend-length 2048. Novel-content prefill at ~47 tok/s
  (33 min for a full-context cold load) remains the honest cost.
- **drop-qwen MERGED** (homelab PR #5515): monolith drains + synthetics
  now on Luna; qwen session lane retired.
- Open: 6GB-session state skipped the async store write (bounded queue
  drop suspected) - giant sessions may silently not park; investigate
  before relying on session-swap for 90k+ contexts.
- Ops: repeated 1Password locks mid-pipeline; the dedicated node-4 key
  carried everything except GitHub pushes (queued). git bundles over scp
  bridged code to node-4 during the lock.

## Prefill program and the serving profile (2026-09-01 evening to 09-03)

RESULTS.md stopped at the session-prefetch round; this section covers the
commits after it (0b367d6..10dc8e2), measured on node-4's serving unit
unless stated. The "PROGRAM CLOSED" line above described the decode
program. The prefill path was rebuilt afterwards because the 100k-context,
single-lane serving profile made novel long prompts the dominant wait: a
25k-token prompt ran at 30 to 60 tok/s (7 to 14 minutes), and the first
request after every restart ran at ~15% hot coverage and 54 to 64 tok/s.

Probe of record: `ft probe-prefill` (41436b1): seeded, per-run unique
prompts sized in tokens, TTFT from the first delta, median of 3, and the
engine's own prefill line beside it. Client tok/s = prompt tokens / TTFT;
engine tok/s is the scheduler's input-throughput line. 2,012-token cold
prompts unless stated. Raw outputs: `~/repos/ft-worktrees/tools/evidence-0903/`
on the Mac, node-4 `/tmp/*.out`.

### What shipped (commit, mechanism, measured motivation)

- 5b05e97 chunk-ahead staged PLE gather: prefill n-gram ids are known from
  the input tokens, so each chunk dedupes its rows host-side and ships one
  H2D copy instead of millions of HMM faults (6.27M measured per prefill;
  idealised 49 -> ~133 tok/s).
- d5f819f, 1d5465a, 723c316: per-layer coalesced expert sweep, then
  populate-read (preadv into a bounded scratch) replacing advisory
  MADV_WILLNEED, which the kernel throttled on ~590 MB scattered sets. The
  ~3M demand faults per 2,048-token chunk became minor faults.
- e0d5b04: eager per-layer page release made opt-in. Consecutive chunks
  share most experts; releasing after each layer sent the next chunk back
  to disk: 5-11 tok/s live vs 56 inert (it had been a silent no-op behind a
  stale extension).
- c872e12, d228c81, 04e4c3d: one row-batched nvfp4-w4a8 GEMM per expert per
  chunk replacing ~1M per-token GEMVs; the batched entry had silently
  dispatched to the scalar kernel (96.5% of cycles) until the ISA fix;
  row-blocked tiles cut per-expert calls 3,840 -> 120, and a background
  thread populates the next chunk's layer-0 union behind tail-layer compute.
- 21d4b42, 61e5b43: one host-memory governor (page-cache reserve, pin and
  pager budgets fitted inside MemAvailable, oversubscription refuses
  startup) and GPU prefill layers reserved before the disk tier takes the
  rest (a 35G budget had pinned zero GPU layers; now 7 layers, 26.6G).
- 9b13711, 2821a71: hot rows hold protected GPU slots; the full pinned host
  mirror is gone (a 48G hot set had cost 47G of shmem on the 176G cloud
  box); the HOT plan is bounded by GPU slots, not the pin budget.
- 81163b2, 5e04cac, c7a667a, ee68166, ee48dfc: hot/cold split for
  DISK-layer prefill. Routes whose expert sits in a protected slot run the
  GPU expert GEMM (grouped prefill kernel after the moe_align 1,024-bin
  fix), cold routes stay on the CPU batched path. Motivation: 87% hot-pair
  at the decode plateau while prefill used none of it. Three correctness
  rounds on the way: the bf16 GPU kernel wrote over its input so the CPU
  partial consumed the GPU output; slot-count vs slot-index geometry
  (illegal memory access); cold routes clamped to slot 0 reading unwritten
  e4m3 scales as NaN.
- 02f7498, 355c2b0, abd2056: adaptation ticks on routed tokens including
  prefill (32-token answers had left prefill coverage at 9%, 250-token
  answers reached 55%); interval derived from the allocation (20 for a 48G
  set, 166 for 6G, the two values found by hand).
- 059057f, d2c684f: FP8 e4m3 KV cache on the QSA layers; freed VRAM becomes
  expert slots under cache-auto; disk prefix entries carry the dtype.
- eb1b923, 42fc950: abort on client disconnect (three timed-out prefills
  had held the single lane for 2.5 h); the ASGI cycle completes on
  disconnect.
- f36169f, b522bc1, b19413d: transparent hugepages on bank mappings, then
  the fair-test rework (advise before fill, never on UVM-registered ranges,
  tmpfs mirror arm for file-backed banks, grouped startup report).
- fb0007d: hot plan persisted across restarts (versioned file, atomic,
  every N minutes and on shutdown; seeds protected slots and counters when
  the FTW identity matches; bounded shutdown).
- ebd0cab: upstream ports (#342 lm_head rows, #339 GDN conv sync, #338
  Triton PLE hash, #231 routing-oracle stat). #89 skipped (tile table
  mismatch with the NVFP4 kernel).
- de3bf0b: `--ple-backend uring`, port of upstream #311: row reads from the
  quantized table through a native io_uring reader into governor-charged
  pinned staging, O_DIRECT with bounce buffers, strict failure, ring
  drained before teardown.
- a5e6f41: idle-time adaptation (after 500 ms idle, one bounded tick,
  repeated while swapping; an arriving request wakes the scheduler).
- 057ce31: KV ladder, port of upstream #300 with the #340 floor fix: the
  pool starts at a floor and grows in 32,768-token steps from expert slots
  at request boundaries; an explicit --num-pages becomes the cap.
- 7a61335: NVFP4 activations for the sm_120 expert GEMMs
  (--moe-activation-dtype auto = nvfp4 on Blackwell when every expert
  carries an input scale). Unmeasured; needs the G4 window.

### Numbers of record (node-4 serving unit, 2,012-token cold unique prompts)

| arm (09-03) | client tok/s min / median / max | engine tok/s | prefill hot-route | decode x1 (batches) |
|---|---|---|---|---|
| tier with `--ple-backend cached` (04:08) | 58 / 68 / 73 | 47-53 | 69-90% | 1.4-6.4 |
| `--ple-backend disk` + chunk-ahead gather (04:12) | 276 / 359 / 371 | 152-202 | 93-96% | 4.3-10.9 |
| KV reservation 163,840 -> 65,536, pre-ladder (05:26) | 132 / 365 / 374 | 93-235 | 93-96% | |
| hot plan persistence, first two requests after a restart (06:17) | 351, 391 (unseeded: 267, 333) | 172-195 | 96.3% (unseeded 93.1%) | |
| `--ple-backend uring` (07:57) | 316 / 371 / 402 | 196-283 | 91-93% | 5.4-12.8 |
| 10dc8e2, KV ladder + idle ticks (09:14) | 231 / 413 / 426 | 168-294 | 92-94% | 7.0-14.2 |
| THP on the pinned banks, 27.7 GB backed (09:29) | 174 / 296 / 341 | 137-229 | 93% | 5.4-12.5 |

26,812-token novel prompt (05:28, pre-ladder): TTFT 77.4 s, 85.3 s wall,
346 tok/s; the first 2,048-token chunk ran at 107 tok/s and the rest at
320 to 450. The same prompt shape was 7 to 14 minutes on 09-01.

Decode did not move in any arm: 9 to 14 tok/s single-stream once warm. A
40-token chat turn is 4.6 to 6.1 s wall at 10dc8e2 (deploy2, which by
accident measured the pre-ladder code, saw 9.4 s for the first turn and
~5 s after). Chat is decode-bound now. The 21 tok/s x1 of the adaptation
round was the MRR-8 load profile with a 32k KV pool; the 100k-context
profile spends those expert slots on KV (3,290 at a fixed 163,840-token
reservation, 3,753 with the ladder's 65,536-token floor, ~3,590 once grown
to the 100,352 cap).

### Verdicts

- PLE off the pinned budget is the single biggest prefill lever on this
  box: cached -> disk was 68 -> 359 tok/s on the same code, because the
  chunk-ahead gather feeds prefill in one batched read and the 8 GiB pin
  goes back to expert banks.
- Hot plan persistence: the cold first request is gone (351 vs 267 tok/s,
  96% coverage from the first prompt). One test fix on node-4 (fa1b11e:
  a captured graph must replay before its indices are read).
- io_uring PLE: +3% client-side and +24% engine-side over the mmap disk
  backend, 0.47 ms gather per decode step, 0 major faults, no UVM in the
  path. The extension must be built on the host (setup.py build_ext
  --inplace in the served worktree): the first deploy crashed at load
  because it was not (uringab1).
- KV ladder: +14% expert slots at startup for the 100k profile with no
  ceiling change; a 2k prompt with the default 32k output budget needs no
  growth.
- Idle ticks fire between chat turns (hot_adapt_ticks_idle: 2 in a
  four-turn test); the hot set assembles while the GPU would otherwise idle.
- THP, fair test: -28% client-side and -18 to -44% engine-side prefill,
  decode unchanged. Off stays default. The tmpfs mirror arm for file-backed
  banks needs RAM headroom a 64 GB box does not have; the wiring stays for
  hosts with file THP or more RAM.
- MTP with the resident draft head (flash-e2m1-mtp.ftw, K=1): 52 to 78%
  acceptance on 40-draft windows, yet 300-token essays took 37 to 48 s with
  the head on vs 19 to 31 s off (decode batches 10-14 vs 15-23). Still net
  negative: the verify pass runs outside CUDA graphs. The open lever is
  graphs under MTP, then batched verify at K=2-4. The first attempt crashed
  at load because the MTP FTW directory lacked the PLE shard links.
- HMM PLE retired on node-4: two UVM oopses on 09-02 traced to the HMM PLE
  backend itself, not THP (dmesg captures in node-4
  freetoken/dmesg-uvm-oops-*.txt). The blog's section 5.2 recommendation
  is superseded by the uring backend.
- Deploy trap: a git bundle whose base commit the target lacks fails to
  fetch, and the deploy script measured the old code (deploy2, 08:54).
  Read the target's log before cutting a bundle.

### GLM 5.3 Flash on the tier (09-01/09-02, GCE G4 box, us-central1-b)

dfd6cc8, 5408598, 766b889, a712aae, dd11d99, 0a58702, 7832c52: the 45-layer
hybrid (KDA linear layers plus MLA/DSA, 288 routed + 1 shared experts,
sigmoid noaux_tc) loads through the unified offload layer with disk-tier
support; CPU-streaming FTW conversion for the NVFP4 W4A4 layout; nope-only
MLA kernels (the first decode step died on tl.arange(0, 0)); the pooled
DSA index for checkpoint fidelity; nine real Linux-suite failures fixed on
the first RTX PRO 6000 run. Instances `glm-convert` and `glm-flash-test`
(pytorch-2-9-cu129 image family) were deleted on 09-02. No throughput
numbers were recorded: that is the pending G4 window, together with the
sm_120 NVFP4 activation path.

Fork state: feat/moe-disk-tier 10dc8e2, 166 commits on upstream 58f4b9e.
Upstream has 8 commits since: #311 (PLE from disk) and #342/#339/#338/#231
are ported; #332 (GLM-5.3-Flash, upstream's own implementation), #329
(exact Triton sampling), #336 (safetensors index download), #343
(mixed-precision NVFP4 detection) and #319 (Triton router) are not.
Reconcile GLM against #332 before any upstream PR.

## GLM 5.3 Flash on Blackwell (2026-09-03, GCE g4-standard-48 spot, RTX PRO 6000 96 GB, 180 GB RAM)

Model: RedHatAI/GLM-5.3-Flash-NVFP4 (45 layers, 288 routed + 1 shared experts,
top-8; about 160 GB of expert banks, 177 GiB FTW). Bench recipe: FTW and venv
from gs://h0melab-glm-ftw, tier branch on the box, `ft probe-prefill` 2k cold
prompts plus a 300-token essay and `load.sh` x4 per rung, quality via the
thinking-aware `quality2.sh` (thinking off at 768 tokens, thinking on at 2048).
Raw logs: ~/repos/ft-worktrees/tools/evidence-0903/ (g4-rungs.log, rungs.log,
diag*.log, g4-journals.tgz).

### The prefill bug that invalidated the first rungs

The first two rungs (R1 pin 112 / hot 48, R2 pin 64 under a 128 GB cap) ran
with fluent decode and a broken prompt: with thinking off the model answered
the codeword prompt as if it had seen only "ZEPHYR", not the digits or the
counting instruction, and the word-list prompt as if no list had been given.
Isolation arms (hot split off, batched CPU prefill off) failed identically on
40-token prompts that barely touch the CPU path, which pointed above the
experts. Cause: `glm5_next/linear.py` computed the KDA per-channel forget gate
`[1, T, H, D]` and handed it to `chunk_gated_delta_rule`, whose contract is one
gate per head `[B, T, H]`; the wrapper asserts only beta's shape. Decode uses
`fused_sigmoid_gating_delta_rule_update(is_kda=True)` in-kernel and was right,
which is why throughput looked normal. Fix (e029865, f47f5b2, merged 8ba441e):
upstream's KDA chunk kernels vendored verbatim from a2538a4 and the prefill
branch rewired to `chunk_kda_with_fused_gate` (gathered initial state in,
final state scattered back). Four prompt-fidelity probes (repeat-exactly,
codeword recall, list lookup, short) pass after the fix. Lesson recorded: run a
prompt-fidelity probe before any throughput rung on a new model family; the
five-check quality script missed it because thinking consumed `max_tokens`.

### Numbers on the fixed build

| rung | config | 2k cold prefill | decode x1 | x4 aggregate | quality |
|---|---|---:|---:|---:|---|
| F1 | 112 GB pinned, 48 GB hot, no cap | 257 tok/s (settled; 28 and 49 during the hot-set fill) | 11 to 12 tok/s | 19.3 tok/s | 8/8 (thinking off and on) |
| F3 | 35 GB pinned, 48 GB hot, cgroup MemoryMax 64 GB | probe timed out (0.06 to 7.6 tok/s per chunk) | 0.03 to 0.04 tok/s, 140,835 major faults per step | not run | not run |
| R2 (pre-fix, decode only valid) | 64 GB pinned, 48 GB hot, cap 128 GB | (invalid) | ~14 tok/s | 10.0 tok/s | (invalid) |

Reading F1 correctly: with 112 GB pinned, 29 layers are PINNED and stream
experts over PCIe through a 17 GB LRU every step (4,904 slots, 3,580 of them
protected hot rows), so the ~70 ms step is PCIe streaming, not GPU compute.
"Everything resident" is not available on a 96 GB card for a 160 GB expert
set; GLM inherits the same decode shape as Qwen on the 4090, a cold tail off
the GPU. F3 is the honest 64 GB-box answer for this model with this method:
unusable. The 48 GB hot set starts cold and needs about 5k routed tokens to
fill at 0.5 GiB per tick, so a 1,500-token probe warmup measures the fill; the
later rung scripts use 8,000.

### Not measured (issue #12 on the fork)

- All 42 MoE layers on the DISK path with the cold tail fetched over PCIe
  (`--moe-disk-decode gpufetch`, no hot budget): was mid CUDA-graph capture
  (bs=4 capture took 159 s) when the window was closed.
- The same with the cold tail on the CPU (pin 8, hot 60): the server got
  SIGKILL about 10 s after uvicorn started, twice, no OOM in dmesg.
- bf16 versus NVFP4 activations on sm_120: the bucket FTW predates the
  sidecar support, and reconversion on a CPU host writes a bf16 file with no
  sidecars because `ft checkpoint` resolves the activation policy against the
  converting host (issue #11); gate and up input scales in the source are
  identical, so the checkpoint itself is usable. Convert with `--gpu 0` on an
  sm_120 host.
- Step timing on the F1 config.

### Cost and ops

About 13 hours of g4-standard-48 spot across six sessions since 1 September
(roughly 25 to 35 GBP), 3.7 hours of n2 convert VMs, bucket about 6 GBP a
month. Traps hit: `pkill -f "ft serve"` inside a gcloud ssh command kills the
ssh session itself (bracket the pattern); a git bundle whose base the box
lacks fails to fetch (push the branch, `git fetch origin`); systemd-run scripts
need /snap/bin on PATH for gcloud; `--moe-hot-expert-budget-gib` requires
`--moe-disk-decode cpu`; the io_uring extension needs `<linux/time_types.h>`
on 22.04 headers (aa818cb).

## Decode program, night of 2026-09-03 (node-4, fork issues #2 to #15)

Harness: `tools/n4-decode-ab.sh` on the Mac side (a second worktree on node-4
with its own extension build, sharing the serving venv), each arm run as a
detached systemd unit with a 15-minute watchdog, the production unit stopped
for the arm and restored after it, and from 20:00 the hot plan file backed
up before and restored after every arm. Metric: three 300-token essays
(thinking off) after three warm-up essays, plus the decode stats line.
Production profile throughout (hot 6 GiB, uring PLE, single lane, 100k cap).

### The number that matters first: a warm control

The first control of the evening measured 13.5 tok/s mean with 1,000 to
1,400 major faults per step; every later arm ran at 130 to 500. Re-measured
warm at the end, the same merged tier gave 23.7 / 22.9 / 17.9 tok/s (mean
21.5, steady batches 24 to 30). Every "gain" quoted against the cold control
evaporated against the warm one. Rule adopted: the control runs last (or
twice), and arms are compared only when major faults per step are in the
same band.

### Arms (essay means; warm control 21.5)

| arm | branch | mean tok/s | batches | verdict |
|---|---|---:|---|---|
| timers on (#3) | 04149da | 16.2 (cold cache) | 17 to 22 | diagnostic, merged 0215585 |
| cold fetch N = 4 (#6) | 8147037 | 12.0 | 7 to 17 | negative |
| cold fetch N = 12, twice | 8147037 | 14.8, 13.6 | 8 to 25 | neutral |
| N = 12 plus hot 7 GiB | 8147037 | 15.3 | 10 to 23 | the budget, not the fetch |
| hot 7 GiB alone | 8147037 | 21.5 | 23 to 29 | equals the warm control |
| spin executor, first cut (#4) | a625863 | 2.6 | 2 to 3 | 14 spinners on 8 cores starved the GPU thread |
| spin, bounded and placed, 3 ms idle | 60055ff | 20.9 | 22 to 30 | neutral; wake 378 to 273 us only |
| spin plus hot 7 | 60055ff | 20.1 | 24 to 29 | neutral |

### What the timers say (per DISK layer per step, 28 layers)

wake 270 to 380 us, compute 400 to 450 us for one to three experts, signal
about 5 us; the CPU windows around the executor task (cpu_head plus cpu_tail,
36 to 40 ms per step) hold another ~0.6 ms per layer of handoff (D2H,
doorbell, H2D, GPU-side wait). That handoff is now the largest line item
(issue #13). The cold-fetch policy reuses the same flag-handshake task for
its staging copy, which is why it cannot remove a round trip (its payload
changed, not its latency). Batched speculative verify (#9) remains the only
lever that moves the ~40 ms step by more than a fraction.

### The long-prompt scare, and what it actually was

A 72k-token `ft probe-prefill` prompt (seeded random word salad) left decode
at 0.14 to 0.25 tok/s with the hot-pair rate at 10 to 14 percent against an
oracle of 94 to 100 and PLE major faults in the hundreds of thousands per
step. It reproduced with the KV ladder on, off (fixed 100k pool), on the
ladder-fix branch, and with hot 6, 6.5 and 7. With adaptation disabled the
realised hot rate was 14 percent for the whole prefill and prefill fell from
450 to 80 tok/s: random words route to experts no hot set tuned on real text
holds, adaptation re-tunes the set to them during prefill, and random n-gram
rows defeat the page cache (5,000 to 14,000 disk major faults per step). The
instrument, not the ladder.

The real-text version (a 76,570-token document of fork docs and sources):
prefill 551 s (139 tok/s average, chunks 170 to 335, 59 percent of prefill
routes hot), then three essays at 9.1 to 12.3 tok/s (batches about 14.5)
with the realised hot rate at 13 percent against a 93 percent oracle and
disk major faults at 14 per step. So a long document does re-tune the hot
set to itself and the following conversation runs at roughly two thirds of
the warm rate until adaptation catches up (issue #16); it does not collapse.

Two things that were real: with a 7 GiB budget the ladder's growth rebuild
evicted protected rows (it handed the rebuild the trimmed plan owners and
discarded a completed adaptation future unpublished); fix on
`fix/ladder-growth-hot-set`, whose first cut the review blocked because its
refusal path parked requests forever (correction in progress, issue #15).
And every A/B arm persisted its hot plan on shutdown, so the production unit
was reseeded with a document-tuned or salad-tuned set after each long-probe
arm until the runner learned to back the plan up.

### Production

Unchanged: hot 6 GiB, ladder on. Hot 7 is neutral against a warm control
and leaves the ladder 41 reclaimable slots against the ~163 it needs to
reach the cap. Serving box downtime from the arms: one 13-minute strand at
18:10 UTC when a spin arm hung and the driver died with the ssh session
(the runner is detached and watchdogged since), plus about 100 s per arm.

## Round two, late 2026-09-03 (node-4, fork issues #13, #16, #17)

Same box, same serving profile, same A/B protocol (detached driver, hot plan
backed up and restored around every arm, control run last on a warm page
cache, compare only arms with similar major faults per step). The evening's
timers said the executor task was about 850 us of a 1,460 us per-layer window
and the rest was "handoff outside the task". That framing was wrong, and
finding out why is most of this section.

### The wake bucket was the WILLNEED callback

`submit()` stamps the doorbell first and then, before any worker is notified,
builds the decode groups, takes the GIL, and runs the Python pre-run callback
(`prefetch_experts`: dedupe, stats, and one `madvise(MADV_WILLNEED)` per
coalesced expert range on every DISK bank of the layer). So "wake" was four
things. Splitting it (branch `feat/handoff-timers`, merged as f560cd0), per
DISK layer, 28 layers per step, `--moe-step-timing`:

| arm | x1 wall tok/s | batches (median) | CPU windows/step | wake | precb | compute | major faults/step |
|---|---|---|---|---|---|---|---|
| control, default order | 19.9, 16.3, 19.8 | 21 to 25 (22.3) | 41.3 ms | 409 us | 384 | 466 | 135 |
| `--moe-cpu-precb after` (callback after notify) | 18.5, 18.8, 17.7 | 18 to 22 | 45.2 ms | 17 | 475, overlapped | 788 | 626 |
| host-func handshake (`FREETOKEN_CPU_MOE_FLAG_SYNC=0`) | 14.8, 18.5, 16.7 | 17 to 24 (19.6) | 54.1 ms | 451 | 425 | 838 | 993 |
| `--moe-cpu-willneed recent` | 22.0, 22.6, 16.4 | 25 to 29 (24) | 33.4 ms | 64 | 40 | 513 | 476 |

Groups, GIL and notify are 1, 1 and 22 us; the coordinator's own overhead is
5 us; the host-clock-calibrated GPU-side latencies (doorbell to coordinator,
done flag to the GPU wait releasing) are under 10 us in both directions. The
callback is the bucket, and its cost is the kernel walking the page cache for
every page of the advised range even when the pages are resident.

Moving the callback after the notify removes it from wake and loses anyway:
the workers fault the pages themselves, major faults double and compute rises
by 300 us. So the advice is doing real work whenever a page is not resident.
The host-func handshake is a clear loss and memops stay.

`--moe-cpu-willneed recent` skips the advice for experts this executor
computed on the same layer within the last 256 decode steps (their pages are
resident by construction), with a rolling major-fault ceiling that falls back
to always-advise for 256 steps after a prefill has destroyed the page cache.
It takes about 8 ms off a 41 ms step. The price is more faults during compute
and the guard tripped twice in a three-minute arm at the 2,000 ceiling, so the
code default stays `always` and the production unit carries the flag.

### What actually fills the step

With the callback out of the way the DISK layers cost about 0.6 ms each, 17 ms
of a 37 ms step, and the GPU-side handoff is not the remainder. The other 20
MoE layers are PINNED: their routed experts stream over PCIe every step (one
fused gather kernel at about 31 GB/s, on the same stream, immediately before
the GEMM) through what is left of the slot pool after the DISK hot rows are
carved off, about 1,450 LRU slots shared by all 20 layers. Those layers have
no hot set and no frequency signal: every gate in the code says the hot set is
DISK-only, and nothing writes the decayed counters for a PINNED layer. That is
issue #17 (`feat/pinned-hot-set`, `--moe-pinned-hot-budget-gib`, in test as
this is written). The slot pool is fixed on a 24 GB card, so PINNED hot rows
only exist by lowering the DISK budget; the arms trade 1 and 2 GiB across.

### The long document, valid this time

The evening's real-text number (a 76,570-token document, then three essays
at 9.1 to 12.3 tok/s) had one control and no arms. Three knobs, each default
off, on `feat/hot-adapt-phase-weight` (merged as ea0e4ae): a multiplier on
prefill route counts before they reach the decayed counters, a cap on the
total swap over a run of consecutive prefill boundaries, and a forced tick at
the first decode boundary after a prefill run.

The first chain of arms measured nothing: the document hit the KV disk cache
and "prefilled" in 11 to 16 s, so no prefill routes were ever counted and the
knobs never ran. A nonce at the start of the prompt fixed it, and the wall
time of the long request is now the first thing to check on any prefill arm.
Valid arms, post-document turns of 200 tokens, wall tok/s, then the realised
hot-pair rate against the oracle, control last:

| arm | prefill | turn 1 | turn 2 | turn 3 | realised / oracle |
|---|---|---|---|---|---|
| `--moe-hot-adapt-prefill-weight 0.1` | 630 s | 11.0 | 15.2 | 15.8 | 47 to 53% / 93% |
| `--moe-hot-adapt-prefill-run-cap-frac 0.5` | 679 s | 10.9 | 12.8 | 13.7 | 35% / 94% |
| `--moe-hot-adapt-post-prefill-tick on` | 588 s | 8.8 | 12.2 | 8.8 | 15 to 16% / 93% |
| all three | 640 s | 10.1 | 11.6 | 15.5 | 46 to 48% / 93% |
| control | 570 s | 4.9 | 12.3 | 7.2 | 15 to 16% / 94% |

The weight is the whole effect. The counters are document-dominated because
a 2,048-token chunk adds up to 2,048 route counts per invocation while a
decode step adds ten, so a forced tick or a churn cap only re-ranks toward the
document. The production unit carries the weight at 0.1; a weight that
normalises by tokens per invocation is the obvious follow-up.

### Production

Deployed at 23:51 UTC: tier f560cd0 with `--moe-cpu-willneed recent` in the
unit (post-deploy probe 11.6 tok/s cold, warming). Queued behind the #17
arms: tier ea0e4ae with `--moe-hot-adapt-prefill-weight 0.1` added. The
previous checkout and the unit file are kept for rollback. Box time lost to
the harness tonight: about 15 minutes to the cache-hit chain and 11 minutes to
an idle check that matched the word "running" inside a dead unit's command
line (`--max-running-requests`).
