# Quickstart: Qwen3.8-Flash-Next on a 24 GB GPU with 64 GB of RAM

This fork adds a disk tier to FreeToken so a model whose expert weights do not
fit in pinned host RAM can still serve. This page is the exact path that runs
on the box the numbers were measured on: an RTX 4090, a Ryzen 7800X3D, 64 GB
of DDR5, and an NVMe. Upstream's own docs cover everything else
([install](docs/install.md), [cli](docs/cli.md), [models](docs/models.md)).

What to expect at the end: about 100 GB of weights served, 21 tokens a second
on one stream and about 37 across eight, with the GPU averaging 138 W.

## 1. Requirements

- Linux x86_64, an NVIDIA GPU with 24 GB, 64 GB of RAM, and about 250 GB free
  on an NVMe (the raw checkpoint is 126 GB, the quantized lookup table 27 GB,
  the converted weights 73 GB; the raw checkpoint can go after conversion).
- The NVIDIA **open** kernel modules, not the proprietary ones. The lookup
  table is read by the GPU straight out of the file mapping through Linux HMM,
  and HMM needs the open modules. Check with `modinfo nvidia | grep -i license`
  (open modules report a dual MIT/GPL license).
- A CUDA 13 toolkit with `nvcc` on PATH, Python 3.12, `uv`, and `ninja`.

## 2. Install the fork

```bash
git clone https://github.com/jomcgi-org/freetoken-fork.git freetoken && cd freetoken
git checkout feat/moe-disk-tier
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[accel]" ninja
ft --version
```

CUDA kernels compile on first use. Anything under `csrc/` needs
`python setup.py build_ext --inplace` after a pull; the JIT does not cover it,
and a stale `.so` silently degrades the CPU executor.

## 3. Download the model and the quantized lookup table

```bash
export HF_HUB_DISABLE_XET=1   # the Xet path has wedged mid-download
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir models/flash-raw
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_nvfp4/*" --local-dir models/ple-quant
```

The second download is the per-token lookup table already quantized to e2m1
(27 GB instead of 47.7 GB). Without it the table alone will not leave room for
enough expert layers in RAM.

If the download crawls on a box with more than one route (say wired plus
WiFi), check `ip route show default` and take the second one down; ARP flux
between two interfaces on one subnet dropped one download from 107 to 21 MB/s.

## 4. Convert to FTW

```bash
ft checkpoint --model models/flash-raw --out models/flash.ftw
```

The converter streams one expert layer at a time. On the reference box it
took 94 seconds. It has run out of memory on 64 GB cloud boxes with less
headroom; if it does, add a swapfile on the NVMe for the conversion, or convert
once on any machine with 96 GB or more.

`models/flash.ftw` now holds the FTW shards, `freetoken_weight.json`, the
model's config and tokenizer files, and still the raw safetensors with their
index. Serving from it as is would use the unquantized fp8 table. The serving
directory is a second directory of symlinks that leaves out the raw shards and
the index and adds the quantized table shards. With no index file present the
loader discovers the table shards beside the weights.

```bash
mkdir models/flash-e2m1.ftw && cd models/flash-e2m1.ftw
for f in ../flash.ftw/*; do
  case "$(basename "$f")" in
    model-*.safetensors|model.safetensors.index.json) ;;
    *) ln -s "$f" . ;;
  esac
done
ln -s ../ple-quant/ples_nvfp4/*.safetensors .
cd -
```

## 5. Serve

This is the unit that runs on the reference box, with the paths made relative.
It is the interactive configuration: one request at a time with a 100k
context, and a prefix cache on disk so a restart or a long conversation does
not re-prefill.

```bash
ft serve \
  --model models/flash-e2m1.ftw \
  --moe-backend offload --moe-cache-auto \
  --moe-disk-prefill cpu \
  --ple-backend hmm \
  --moe-hot-expert-budget-gib 6 \
  --moe-pinned-hot-budget-gib 0 \
  --moe-hot-adapt-interval-steps auto \
  --moe-hot-plan-persist auto \
  --moe-hot-plan-dir models/flash-e2m1.ftw \
  --moe-hot-plan-interval-minutes 10 \
  --moe-collect-stats \
  --max-running-requests 1 --linear-state-cache-ratio 4.0 \
  --num-pages 1568 --max-seq-len-override 100352 --max-extend-length 2048 \
  --kv-disk-cache-dir prefix-cache --kv-disk-cache-gib 500 \
  --host 127.0.0.1 --port 8090
```

What each line does:

- `--moe-backend offload --moe-cache-auto`: upstream's expert offload with the
  GPU slot cache sized from what VRAM is left after the KV pool.
- `--moe-disk-prefill cpu`: prefill for disk-resident layers runs on the CPU
  executor and touches only the routed experts. Without it a prefill pages in
  whole expert layers and a six-token prompt takes minutes.
- `--ple-backend hmm`: the GPU reads lookup-table rows through the file
  mapping. If the startup readback probe fails you are on the proprietary
  modules; `--ple-backend disk` is the staged fallback, slower.
- `--moe-hot-expert-budget-gib 6`: the hot set. The most-routed expert rows of
  the disk layers are pinned and go through the GPU slot cache; the cold tail
  stays on CPU decode. 6 GiB was the optimum in a two-dimensional sweep with a
  40 GB pin budget; 44/4 and 36/8 both lose.
- `--moe-pinned-hot-budget-gib N`: optionally reserves `N` GiB of the same
  protected GPU slot pool for frequently routed experts from PINNED layers.
  Misses still refill from pinned host memory through the normal offload path.
  The default is zero, which preserves the original DISK-only hot set. With
  `--moe-collect-stats`, status reports protected hit coverage plus PINNED
  misses and PCIe bytes per decode step.
- `--moe-hot-adapt-interval-steps auto` with `--moe-collect-stats`: the hot
  set follows the traffic. The interval is measured in routed tokens, shared
  by counted HOT-split prefill chunks and decode batches. The automatic default
  derives its fill and steady cadences from the HOT allocation. At most 0.5 GiB
  of rows swap per due interval, while the default boundary cap stages no more
  than half of the HOT budget at one request boundary. This also means no
  profile capture step: start cold and it warms itself. If you want a static
  budget instead of auto-adapting, set the adaptation interval to zero and
  supply a captured profile with `--moe-disk-layer-profile <json>`.
- `--moe-hot-adapt-idle-ms N`: starts an adaptation tick after the scheduler
  has been idle for `N` milliseconds. The default is 500; zero disables idle
  adaptation without disabling token-boundary adaptation. Idle adaptation is
  supported only when TP == 1 and is inert under tensor parallelism.
- `--moe-hot-adapt-idle-min-interval-ms N`: sets the minimum time between
  repeated idle adaptation ticks. The default is 2000 milliseconds.
- `--moe-hot-adapt-prefill-weight N`: scales prefill route counts before they
  update HOT adaptation counters. The default is 1.0.
- `--moe-hot-adapt-prefill-run-cap-frac N`: caps total row swaps across one
  consecutive prefill run as a fraction of the HOT budget. The default is 0,
  which disables the run cap.
- `--moe-hot-adapt-post-prefill-tick {on,off}`: runs one immediate standard-cap
  adaptation tick at the first decode boundary after prefill. The default is
  `off`.
- `--moe-hot-plan-persist {auto,on,off}`: controls HOT plan loading and
  snapshots. The default is `auto`, which enables writes when the plan
  directory is writable. `on` requests persistence explicitly, and `off`
  disables both loading and writing.
- `--moe-hot-plan-dir <path>`: overrides the default plan location beside the
  model.
- `--moe-hot-plan-interval-minutes N`: sets how often periodic HOT plan
  snapshots are written. The default is 10 minutes.

After a clean shutdown, the hot set is seeded from the persisted plan and the
first request runs at the previous session's coverage (measurement pending on
node-4). Without a persisted plan, the initial fill lands at about the second
request.

- The KV block: 100,352 tokens of context on a single lane. KV is 64 KB a
  token on this model, so this is close to the 24 GB ceiling. The KV ladder
  starts the pool at its floor, then grows it on demand to this cap. For
  throughput instead of context, drop the three KV flags and set
  `--max-running-requests 8`, which is where the 37 tokens a second aggregate
  comes from.
- `--kv-ladder`: default on when `--moe-cache-auto` is on; it turns
  `--num-pages` into a growth cap. Growth is one-way until restart: expert
  slots surrendered to reach the cap do not return, so one long conversation
  lowers the slot count for later short requests.
- `--kv-disk-cache-dir`: prefix and recurrent state parked on the NVMe,
  fingerprint keyed, so a restart is nearly free and a returning conversation
  prefills only its new tokens.

Host memory has two budgets, the pinned expert banks and the pager's
resident bytes, and they must sum to less than the RAM left after the OS. The
startup governor sizes both from `--host-cache-reserve-gib` (default: the
larger of 8 GiB or 15% of RAM) and prints what it chose. To override, set
`FREETOKEN_PIN_BUDGET_GB` (40 for throughput, 46 for single-stream latency on
this box). Pinning 52 of 64 GB leaves no page cache for the disk tier and
collapses throughput to under 4 tokens a second; that was day one.

## 6. Check it

The server binds its port before the engine is ready, so readiness is a real
completion, not a TCP connect. First token from a cold NVMe takes 30 to 110
seconds.

```bash
curl -s http://127.0.0.1:8090/v1/models
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<id from /v1/models>","messages":[{"role":"user","content":"Explain page cache in two sentences."}],"max_tokens":128}'
```

Use the exact model id from `/v1/models`. A wrong model name hangs the request
instead of returning an error.

Watch the decode status lines in the log. At steady state on this
configuration: `disk major faults` under 20 a step, `hot_pair_rate` climbing
past 60% within a few adaptation windows, and `ple_major_faults` near zero
once the table's working set is cached.

## 7. When it goes wrong

- **Thousands of major faults a step, single digits of tokens a second:** the
  pin budget is eating the page cache. Lower `FREETOKEN_PIN_BUDGET_GB` or
  raise `--host-cache-reserve-gib`.
- **`--ple-backend hmm` fails its readback probe:** proprietary kernel modules.
  Install the open ones, or fall back to `--ple-backend disk`.
- **Relaunch fails with the port in use:** a killed frontend leaves the
  scheduler child holding the port plus one. `fuser -k <port>/tcp
  <port+1>/tcp` between runs.
- **Request-level out-of-memory at full context:** the guard fails the request
  with a 503 and keeps serving. If it recurs at the same geometry, lower
  `--num-pages`.
- **Prefill of a novel 25k-token prompt runs at 30 to 60 tokens a second:**
  that is the current cold-prefill ceiling on this hardware, bound by the CPU
  expert path. Warm prefill of cached prefixes is much faster; the disk prefix
  cache is what makes long conversations usable.

## What is measured and what is not

Every configuration above passed a five-check smoke suite (arithmetic, recall,
reasoning, two long generations) and a 93k-token needle recall. That is a
floor, not benchmark parity. The tier is verified on the Qwen3.8-Flash-Next
architecture; the GLM MoE families load through the same tier but have not
been benchmarked on this box.
