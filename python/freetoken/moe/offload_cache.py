from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat
from freetoken.exit_safe_executor import ExitSafeThreadPoolExecutor

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Shutdown allows one conservative 512 MiB/s transfer window plus fixed setup
# headroom after asking a catch-up staging loop to stop at its next row boundary.
_HOT_ADAPT_SHUTDOWN_BYTES_PER_SECOND = 512 << 20
_HOT_ADAPT_SHUTDOWN_FIXED_SECONDS = 2.0
# Leave enough supervisor grace for the drain, final write, and executor join.
_HOT_ADAPT_STOP_WAIT_MAX_SECONDS = 20.0
_HOT_PLAN_FINAL_WRITE_SECONDS = 1.0
_HOT_PLAN_FENCE_CANCEL_SECONDS = 0.5
_HOT_ADAPT_EXECUTOR_JOIN_SECONDS = 1.0


class _HotPlanWriteFence:
    """Serialize cancellation with the final atomic plan rename."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False

    def cancel(self, timeout_seconds: float | None = None) -> bool:
        """Fence publication, or abandon confirmation when the lock stays busy."""
        if timeout_seconds is None:
            timeout_seconds = _HOT_PLAN_FENCE_CANCEL_SECONDS
        acquired = self._lock.acquire(timeout=max(0.0, timeout_seconds))
        if not acquired:
            self._cancelled = True
            logger.warning_rank0(
                "MoE HOT plan fence lock acquire timed out; abandoning write "
                "without confirmation"
            )
            return False
        try:
            self._cancelled = True
        finally:
            self._lock.release()
        return True

    def publish(self, source: str, target: str) -> bool:
        with self._lock:
            if self._cancelled:
                return False
            os.replace(source, target)
            return True


def hot_dynamic_slot_reserve(
    cache_size: int, num_experts: int, prefill_overlap: bool,
) -> int:
    """Headroom protected HOT rows must leave for ordinary cache traffic.

    The normal one-layer or two-layer prefill floor is appropriate for a full-sized
    cache, but applying it literally to a small cache can reserve every slot. Keep
    at most half of the configured cache for this floor so HOT rows and dynamic
    traffic can coexist in synthetic and deliberately small configurations.
    """
    floor = (2 if prefill_overlap else 1) * int(num_experts)
    return min(floor, int(cache_size) // 2)


def disk_gpufetch_capacity(
    *, max_tokens: int, top_k: int, num_experts: int, cache_size: int,
) -> int:
    """Maximum distinct decode misses that can need staging in one layer call."""
    values = (max_tokens, top_k, num_experts, cache_size)
    if any(int(value) <= 0 for value in values):
        raise ValueError("GPU-fetch staging geometry must be positive")
    return min(int(max_tokens) * int(top_k), int(num_experts), int(cache_size))


MOE_LAYER_PROFILE_VERSION = 2


def serialize_moe_layer_profile(
    stats: dict, expert_hits: list[list[int]] | None = None,
) -> dict:
    """Serialize traffic into the versioned layer and per-expert profile format."""
    layers: dict[str, float] = {}
    for row in stats["per_layer"]:
        layers[str(int(row["layer"]))] = float(row["missing_per_step"])
    profile = {"version": MOE_LAYER_PROFILE_VERSION, "layers": layers}
    if expert_hits is not None:
        profile["expert_hits"] = {
            str(layer_id): [int(count) for count in counts]
            for layer_id, counts in enumerate(expert_hits)
        }
    return profile

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

def fp8_block_scale_pad(rows: int, cols: int) -> int:
    """Trailing scale-bank dim padded so per-expert row bytes are 16B-aligned (fused copy)."""
    while (rows * cols * 2) % 16:
        cols += 1
    return cols


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # DISK-only prefill policy. LOCKED/PAGEABLE layers always keep the whole-layer
    # pageable copy path.
    moe_disk_prefill: str = "cpu"
    moe_prefill_coalesce: str = "populate"
    moe_prefill_hot_split: str = "on"
    moe_prefill_split_kernel: str = "grouped"
    # DISK-only decode policy. gpufetch keeps the mmap as the authoritative host
    # bank but fills LRU misses through a bounded pinned staging ring.
    moe_disk_decode: str = "cpu"
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = 0.0

    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        assert self.moe_disk_prefill in ("cpu", "copy"), self.moe_disk_prefill
        assert self.moe_prefill_coalesce in (
            "populate", "on", "off"
        ), self.moe_prefill_coalesce
        assert self.moe_prefill_hot_split in ("on", "off"), self.moe_prefill_hot_split
        assert self.moe_prefill_split_kernel in (
            "grouped", "decode"
        ), self.moe_prefill_split_kernel
        assert self.moe_disk_decode in ("cpu", "gpufetch"), self.moe_disk_decode
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-backend cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # HOT rows occupy protected slots in bank_caches. The full bank_sources
        # remain authoritative for the CPU executor and bounded staging fills.
        # ``hot_bank_sources`` is retained as an empty compatibility surface for
        # older cache/test doubles that probe the attribute.
        self.hot_bank_sources: dict[str, list[torch.Tensor | None]] = {}
        self.hot_expert_ids: dict[int, tuple[int, ...]] = {}
        self.hot_expert_capacity: dict[int, int] = {}
        self._pinned_hot_layer_ids: frozenset[int] = frozenset()
        self.hot_row_for_expert = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Online HOT adaptation keeps fixed protected GPU slots and a reusable
        # host staging bank. Rows are retired before the worker stages replacements,
        # then published only after every staged row reaches its GPU slot.
        self.hot_adapt_enabled = False
        self.hot_adapt_interval_steps = 0
        self._hot_adapt_interval_controller = None
        self.hot_adapt_max_swap_bytes = 0
        self.hot_adapt_expert_bytes = 0
        self.hot_adapt_hot_budget_bytes = 0
        self.hot_adapt_boundary_cap_frac = 0.5
        self.hot_adapt_prefill_weight = 1.0
        self.hot_adapt_prefill_run_cap_frac = 0.0
        self.hot_adapt_post_prefill_tick = False
        self._hot_adapt_prefill_run_swapped_bytes = 0
        self._hot_adapt_prefill_run_swaps = 0
        self._hot_adapt_prefill_run_active = False
        self._hot_adapt_prefill_run_generation = 0
        self._hot_adapt_tick_prefill_run_generation: int | None = None
        self._hot_adapt_after_prefill_pending = False
        self._hot_adapt_prefill_tokens_counted = 0
        self._hot_adapt_token_clock = None
        self.hot_adapt_ticks = 0
        self.hot_adapt_ticks_prefill = 0
        self.hot_adapt_ticks_decode = 0
        self.hot_adapt_ticks_idle = 0
        self.hot_adapt_swaps = 0
        self.hot_adapt_idle_swaps = 0
        self._hot_adapt_ticks_reported = 0
        self._hot_adapt_ticks_prefill_reported = 0
        self._hot_adapt_ticks_decode_reported = 0
        self._hot_adapt_ticks_idle_reported = 0
        self._hot_adapt_swaps_reported = 0
        self._hot_adapt_idle_swaps_reported = 0
        self._hot_decay_factor = 1.0
        self.decayed_decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.float32, device=self.device
        )
        # Session profiles are configured by Engine before graph capture. The fixed
        # table-indexed sketch is address-stable, so decode collection is capturable;
        # admission prefetch and protection updates remain outside graph capture.
        self.session_profile_enabled = False
        self.session_profile_topk = 0
        self.session_profile_ids: torch.Tensor | None = None
        self.session_profile_counts: torch.Tensor | None = None
        self._session_decay_factor = 1.0
        self._session_protect_limit = 0
        from freetoken.moe.session_profile import SessionProtectionRegistry

        self._session_protections = SessionProtectionRegistry()
        self._session_prefetch_experts = 0
        self._resume_timing: dict[int, dict[str, float]] = {}
        self._last_resume_warm_rate = 0.0
        self._last_resume_steady_rate = 0.0
        self._hot_mapping_host: torch.Tensor | None = None
        self._hot_slot_owners: dict[int, list[int | None]] = {}
        self._hot_slot_for_row: dict[int, tuple[int, ...]] = {}
        self._hot_slots_device: torch.Tensor | None = None
        self._hot_staging: list[torch.Tensor] = []
        self._hot_staging_rows = 0
        self.hot_staging_bytes = 0
        self._hot_adapt_executor: ThreadPoolExecutor | None = None
        self._hot_adapt_future: Future | None = None
        self._hot_adapt_stop_event = threading.Event()
        self._hot_adapt_wake_event = threading.Event()
        self._hot_adapt_phase: str | None = None
        self._hot_adapt_swaps_pending = ()
        self._hot_adapt_worker_installs = False
        self._hot_adapt_snapshot_host: torch.Tensor | None = None
        self._hot_adapt_snapshot_ready = None
        self._hot_adapt_copy_stream: torch.cuda.Stream | None = None
        self._hot_adapt_tick_interval_tokens = 0
        self._hot_adapt_tick_staged_bytes = 0
        self._hot_adapt_tick_covered_seconds = 0.0
        self._hot_adapt_window_started_at: float | None = None
        self._hot_adapt_deferred_logged = False
        self._hot_adapt_idle_tracker = None
        self._hot_adapt_tick_boundary: str | None = None
        self._hot_adapt_tick_planned_swaps = 0
        self._hot_adapt_tick_executed_swaps = 0
        self._hot_adapt_tick_rate_before = 0.0
        # Persisted HOT plan state. The adaptation and persistence executors are
        # separate so JSON encoding and fsync never occupy the rerank worker.
        self._hot_plan_counter_seed: dict[int, tuple[float, ...]] = {}
        self._hot_plan_path: str | None = None
        self._hot_plan_identity = None
        self._hot_plan_tier_commit = ""
        self._hot_plan_write_enabled = False
        self._hot_plan_interval_seconds = 600.0
        self._hot_plan_last_snapshot = 0.0
        self._hot_plan_executor: ThreadPoolExecutor | None = None
        self._hot_plan_future: Future | None = None
        self._hot_plan_write_fence: _HotPlanWriteFence | None = None
        self._hot_plan_stop_event = threading.Event()
        self._hot_plan_zero_logged = False
        self._hot_plan_last_published_owners: dict[int, tuple[int | None, ...]] = {}
        # per-layer host residency: direct GPU movement requires "pinned";
        # LOCKED/PAGEABLE decode on CPU, while DISK may use CPU or the staging ring
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # Effective per-layer overlap plan. Non-pinned layers never appear here.
        # Buffer ids alternate across pinned layers, except that a synchronous
        # pageable layer reserves buffer 0 for its whole-layer materialization.
        self._prefill_overlap_buffer_ids: list[int] = [-1] * self.num_layers
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        self.moe_activation_dtype = "bf16"
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_hot_pairs = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_hot_total_pairs = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_pinned_hot_pairs = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        self.stat_pinned_hot_total_pairs = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        self._prefill_hot_pairs = 0
        self._prefill_route_pairs = 0
        self._prefill_cpu_experts = 0
        # Decode routing histogram (per layer, per expert) for cache-skew analysis and
        # v2 profiles. The device scatter is captured and replays with each step's raw
        # ids whenever collect_stats is enabled. collect_decode_freq remains a separate
        # benchmark opt-in for callers that want only concentration stats.
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # Host snapshot used only by periodic status reporting. Device route counts keep
        # accumulating without a per-step sync; each report subtracts this baseline to
        # compare the current protected set with the best set of the same per-layer size.
        self._protected_route_baseline: list[list[int]] | None = None
        self._pinned_lru_baseline: list[tuple[int, int]] | None = None
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst/feat are layer-invariant.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # DISK gpufetch is initialized after the CPU executor exists, because its
        # existing coordinator owns the graph-safe doorbell. The staging tensors are
        # host-pinned; only tiny pointer/index descriptors live on the GPU.
        self._gpufetch_capacity = 0
        self._gpufetch_num_host: torch.Tensor | None = None
        self._gpufetch_ids_host: torch.Tensor | None = None
        self._gpufetch_stage_indices: torch.Tensor | None = None
        self._gpufetch_staging: list[torch.Tensor] = []
        self._gpufetch_dst_ptrs: torch.Tensor | None = None
        self._gpufetch_src_ptrs: torch.Tensor | None = None
        self._gpufetch_feat_bytes: torch.Tensor | None = None
        self._gpufetch_fused_ok = False
        # Per-bank [2, num_experts, ...] double-buffer views over the slot cache's
        # first 2 * num_experts slots (set up when prefill_overlap is enabled).
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
        hot_sources: dict[str, list[torch.Tensor | None]] | None = None,
        hot_expert_ids: dict[int, tuple[int, ...]] | None = None,
        hot_expert_capacity: dict[int, int] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.moe.nvfp4_backends`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value
        (default: all pinned). Non-pinned layers have no direct device address.
        LOCKED/PAGEABLE layers must route to the CPU executor (``cpu_layer_ids``,
        set BEFORE this call); DISK layers may instead use
        ``moe_disk_decode=gpufetch`` and pass through a pinned staging ring. The
        direct copy plan skips all of them, and prefill overlap remains enabled
        only for the pinned layers (per-layer schedule).
        """
        from freetoken.moe.host_banks import HostResidency

        assert set(sources) == set(self.bank_schema), (
            f"banks {sorted(sources)} do not match the {self.quant_format!r} "
            f"schema {self.bank_schema}"
        )
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        for label in residency:
            HostResidency(label)
        unpinned = frozenset(
            i for i, r in enumerate(residency) if r != HostResidency.PINNED.value
        )
        if unpinned:
            gpufetch_disk = frozenset(
                i for i, r in enumerate(residency)
                if r == HostResidency.DISK.value and self.moe_disk_decode == "gpufetch"
            )
            unsupported = unpinned - self.cpu_layer_ids - gpufetch_disk
            if unsupported:
                raise ValueError(
                    f"non-pinned layers {sorted(unsupported)} are neither in "
                    "cpu_layer_ids nor DISK gpufetch layers: a layer without a "
                    "device address can only decode on the CPU executor or, for "
                    "DISK residency, via --moe-disk-decode gpufetch (set "
                    "cache.cpu_layer_ids before set_bank_sources)"
                )
        hot_sources = hot_sources or {}
        hot_expert_ids = {
            int(layer_id): tuple(int(expert_id) for expert_id in expert_ids)
            for layer_id, expert_ids in (hot_expert_ids or {}).items()
        }
        hot_expert_capacity = {
            int(layer_id): int(capacity)
            for layer_id, capacity in (hot_expert_capacity or {}).items()
        }
        for layer_id, expert_ids in hot_expert_ids.items():
            hot_expert_capacity.setdefault(layer_id, len(expert_ids))
        if hot_sources and set(hot_sources) != set(self.bank_schema):
            raise ValueError(
                f"HOT banks {sorted(hot_sources)} do not match schema {self.bank_schema}"
            )
        for layer_id, capacity in hot_expert_capacity.items():
            expert_ids = hot_expert_ids.get(layer_id, ())
            if not 0 <= layer_id < self.num_layers:
                raise ValueError(f"HOT layer id {layer_id} is out of range")
            if residency[layer_id] not in (
                HostResidency.DISK.value,
                HostResidency.PINNED.value,
            ):
                raise ValueError(
                    f"HOT layer {layer_id} is neither DISK nor PINNED resident"
                )
            if capacity <= 0 or capacity > self.num_experts or len(expert_ids) > capacity:
                raise ValueError(
                    f"HOT layer {layer_id} capacity must be in [1, {self.num_experts}] "
                    "and cover its seeds"
                )
            if len(set(expert_ids)) != len(expert_ids):
                raise ValueError(f"HOT expert ids for layer {layer_id} must be unique")
            if any(expert_id < 0 or expert_id >= self.num_experts for expert_id in expert_ids):
                raise ValueError(
                    f"HOT expert ids for layer {layer_id} must be in [0, {self.num_experts})"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        self.hot_expert_capacity = hot_expert_capacity
        self._pinned_hot_layer_ids = frozenset(
            layer_id for layer_id in hot_expert_capacity
            if residency[layer_id] == HostResidency.PINNED.value
        )
        self.hot_expert_ids = {
            layer_id: hot_expert_ids.get(layer_id, ())
            for layer_id in hot_expert_capacity
        }
        # Legacy loaders may still supply compact rows. They are validated below
        # but never retained: the authoritative rows are the ordinary layer banks,
        # and configure_hot_adaptation streams seeds through bounded staging.
        self.hot_bank_sources = {}
        self.hot_row_for_expert.fill_(-1)
        self._configure_prefill_overlap_layers()
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                assert source.shape == head.shape and source.dtype == head.dtype, (
                    name, layer_id, source.shape, source.dtype,
                )
            self.bank_sources[name] = list(per_layer)
            self.bank_caches[name] = torch.empty(
                (self.cache_size, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
            if hot_sources:
                compact = hot_sources[name]
                if len(compact) != self.num_layers:
                    raise ValueError(
                        f"HOT bank {name!r} has {len(compact)} layers, expected {self.num_layers}"
                    )
                for layer_id, capacity in hot_expert_capacity.items():
                    hot = compact[layer_id]
                    if hot is None:
                        raise ValueError(f"HOT bank {name!r} is missing layer {layer_id}")
                    if not hot.is_contiguous() or hot.shape != (
                        capacity, *head.shape[1:]
                    ) or hot.dtype != head.dtype:
                        raise ValueError(
                            f"HOT bank {name!r} layer {layer_id} has incompatible "
                            f"shape/dtype {hot.shape}/{hot.dtype}"
                        )
        total_hot_rows = sum(hot_expert_capacity.values())
        dynamic_reserve = hot_dynamic_slot_reserve(
            self.cache_size, self.num_experts, self.prefill_overlap
        )
        if total_hot_rows and total_hot_rows + dynamic_reserve > self.cache_size:
            raise ValueError(
                f"HOT residency needs {total_hot_rows} protected slots plus "
                f"{dynamic_reserve} dynamic/prefill slots, but "
                f"moe_cache_size={self.cache_size}"
            )
        first_hot_slot = self.cache_size - total_hot_rows
        next_hot_slot = first_hot_slot
        self._hot_slot_for_row = {}
        for layer_id, capacity in sorted(hot_expert_capacity.items()):
            slots = tuple(range(next_hot_slot, next_hot_slot + capacity))
            self._hot_slot_for_row[layer_id] = slots
            next_hot_slot += capacity
        hot_slots = list(range(first_hot_slot, self.cache_size)) if total_hot_rows else []
        self._hot_slots_device = torch.tensor(
            hot_slots, dtype=torch.long, device=self.device
        )
        self._hot_slot_owners = {
            layer_id: [
                self.hot_expert_ids[layer_id][row]
                if row < len(self.hot_expert_ids[layer_id]) else None
                for row in range(capacity)
            ]
            for layer_id, capacity in hot_expert_capacity.items()
        }
        self._hot_mapping_host = self.hot_row_for_expert.detach().cpu()
        if self.device.type == "cuda":
            self._hot_mapping_host = self._hot_mapping_host.pin_memory()
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()
        if any(buffer_id >= 0 for buffer_id in self._prefill_overlap_buffer_ids):
            self._init_prefill_overlap_buffers()

    def _configure_prefill_overlap_layers(self) -> None:
        """Build the pinned-layer double-buffer schedule.

        A synchronous pageable prefill materializes into slots ``[0, E)``, which
        alias overlap buffer 0. The next pinned layer therefore uses buffer 1 so
        its asynchronous copy can overlap that pageable layer's GEMM safely.
        DISK layers using CPU prefill do not touch either buffer.
        """
        self._prefill_overlap_buffer_ids = [-1] * self.num_layers
        if not self.prefill_overlap:
            return
        next_buffer = 0
        for layer_id, residency in enumerate(self.layer_residency):
            if layer_id not in self._unpinned_layers:
                self._prefill_overlap_buffer_ids[layer_id] = next_buffer
                next_buffer ^= 1
            elif residency != "disk" or self.moe_disk_prefill != "cpu":
                next_buffer = 1

    def prefill_overlap_for_layer(self, layer_id: int) -> bool:
        """Whether this pinned layer uses the prefill double-buffer path."""
        return (
            0 <= layer_id < self.num_layers
            and self._prefill_overlap_buffer_ids[layer_id] >= 0
        )

    def prefill_path_counts(self) -> tuple[int, int, int]:
        """Return boot-time counts for overlap, synchronous, and CPU prefill."""
        overlap = sum(buffer_id >= 0 for buffer_id in self._prefill_overlap_buffer_ids)
        cpu = sum(
            residency == "disk" and self.moe_disk_prefill == "cpu"
            for residency in self.layer_residency
        )
        return overlap, self.num_layers - overlap - cpu, cpu

    def init_disk_gpufetch(self, executor, *, max_tokens: int, top_k: int) -> None:
        """Allocate/register the bounded pinned row ring used by DISK decode misses."""
        disk_layers = [
            i for i, residency in enumerate(self.layer_residency)
            if residency == "disk"
        ]
        if self.moe_disk_decode != "gpufetch" or not disk_layers:
            return
        if self.device.type != "cuda":
            raise RuntimeError("--moe-disk-decode gpufetch requires CUDA")
        from freetoken.kernel.pinned import alloc_pinned_tensor

        capacity = disk_gpufetch_capacity(
            max_tokens=max_tokens,
            top_k=top_k,
            num_experts=self.num_experts,
            cache_size=self.cache_size,
        )
        self._gpufetch_capacity = capacity
        self._gpufetch_num_host = alloc_pinned_tensor(1, dtype=torch.int64)
        self._gpufetch_ids_host = alloc_pinned_tensor(capacity, dtype=torch.int32)
        self._gpufetch_stage_indices = torch.arange(
            capacity, dtype=torch.int32, device=self.device,
        )
        self._gpufetch_staging = [
            alloc_pinned_tensor(capacity, *per_layer[0].shape[1:], dtype=per_layer[0].dtype)
            for per_layer, _ in self.banks
        ]
        self._build_gpufetch_copy_plan()
        row_bytes = [
            math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            for per_layer, _ in self.banks
        ]
        staging_ptrs = [stage.data_ptr() for stage in self._gpufetch_staging]
        for layer_id in disk_layers:
            executor.register_gpufetch_layer(
                layer_id,
                capacity=capacity,
                num_rows_ptr=self._gpufetch_num_host.data_ptr(),
                row_ids_ptr=self._gpufetch_ids_host.data_ptr(),
                source_ptrs=[per_layer[layer_id].data_ptr() for per_layer, _ in self.banks],
                staging_ptrs=staging_ptrs,
                row_bytes=row_bytes,
            )

    def _build_gpufetch_copy_plan(self) -> None:
        """Refresh H2D descriptors; destination addresses change after cache rebuild."""
        self._gpufetch_fused_ok = False
        if not self._gpufetch_staging or self.device.type != "cuda":
            return
        from freetoken.kernel.pinned import device_ptr

        dst = [cache.data_ptr() for _, cache in self.banks]
        src = [device_ptr(stage) for stage in self._gpufetch_staging]
        feats = [
            math.prod(cache.shape[1:]) * cache.element_size() for _, cache in self.banks
        ]
        if _FUSED_COPY and all(
            value % 16 == 0 for value in (*dst, *src, *feats)
        ):
            self._gpufetch_dst_ptrs = torch.tensor(dst, dtype=torch.int64, device=self.device)
            self._gpufetch_src_ptrs = torch.tensor(src, dtype=torch.int64, device=self.device)
            self._gpufetch_feat_bytes = torch.tensor(
                feats, dtype=torch.int64, device=self.device,
            )
            self._gpufetch_fused_ok = True

    def _build_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        dst_ptrs, feats = [], []
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for bank_id, (per_layer, cache) in enumerate(self.banks):
            feat = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                return  # leave fused disabled; copy_missing uses the per-bank path
            for layer_id, source in enumerate(per_layer):
                if self.is_hot_split_layer(layer_id):
                    # HOT decode is always a protected GPU-slot hit. DISK source
                    # rows reach the GPU only through the bounded staging bank.
                    layer_src_ptrs[layer_id].append(0)
                    continue
                elif layer_id in self._unpinned_layers:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
            dst_ptrs.append(cache.data_ptr())
            feats.append(feat)
        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        if self.quant_format == "nvfp4_marlin" and cache_size > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --nvfp4-backend triton"
            )

    def drain_hot_adaptation_for_rebuild(self) -> None:
        """Stop at a row boundary and publish copied rows before a rebuild."""
        future = self._hot_adapt_future
        if future is None:
            return
        self._hot_adapt_stop_event.set()
        phase = self._hot_adapt_phase
        started_at = time.monotonic()
        computed_budget = max(0.0, self._hot_adapt_stop_wait_seconds())
        wait_budget = min(computed_budget, _HOT_ADAPT_STOP_WAIT_MAX_SECONDS)
        if wait_budget < computed_budget:
            logger.warning_rank0(
                "MoE HOT rebuild drain wait clamped from "
                f"{computed_budget:g} to {wait_budget:g} s"
            )
        cancelled = future.cancel()
        try:
            result = None if cancelled else future.result(timeout=wait_budget)
        except TimeoutError as exc:
            if future.done():
                raise
            elapsed = time.monotonic() - started_at
            logger.warning_rank0(
                f"MoE HOT rebuild drain timed out: phase={phase}, "
                f"elapsed={elapsed:.3f}s, budget={wait_budget:.3f}s"
            )
            raise TimeoutError(
                f"MoE HOT rebuild drain exceeded its {wait_budget:.3f}s wait budget"
            ) from exc

        if phase == "copy":
            copied_rows, staging_seconds = (
                result if result is not None else (set(), 0.0)
            )
            self._finish_hot_adaptation_swaps(copied_rows, staging_seconds)
        elif phase == "plan":
            # A completed plan is intentionally discarded. Polling it here would
            # submit a copy phase whose result the rebuild immediately replaces.
            self._hot_adapt_future = None
            self._hot_adapt_phase = None
            self._hot_adapt_tick_executed_swaps = 0
            if getattr(self, "_hot_adapt_tick_boundary", None) == "idle":
                self.hot_adapt_ticks -= 1
                self.hot_adapt_ticks_idle -= 1
        else:
            raise RuntimeError("HOT adaptation future has an invalid rebuild phase")
        self._hot_adapt_stop_event.clear()
        elapsed = time.monotonic() - started_at
        outcome = (
            "cancelled"
            if cancelled
            else ("discarded" if phase == "plan" else "published")
        )
        logger.info_rank0(
            f"MoE HOT rebuild drain: phase={phase}, elapsed={elapsed:.3f}s, "
            f"budget={wait_budget:.3f}s, "
            f"outcome={outcome}"
        )

    def rebuild(self, cache_size: int, *, preserve_hot_state: bool = False) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Protected HOT rows are reloaded into their new fixed
        slots. Every rebuild with protected rows first stops adaptation at a row
        boundary and publishes completed copies; rebuilds without protected rows do
        not pay that drain cost. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid. ``preserve_hot_state`` additionally keeps
        the current realized-hit window, adaptation controller, decayed counters,
        published owner plan, and plan persistence state used by ladder growth.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        self.validate_rebuild(cache_size)
        total_hot_rows = sum(self.hot_expert_capacity.values())
        dynamic_reserve = hot_dynamic_slot_reserve(
            cache_size, self.num_experts, self.prefill_overlap
        )
        if total_hot_rows and total_hot_rows + dynamic_reserve > cache_size:
            raise ValueError(
                f"HOT residency needs {total_hot_rows} protected slots plus "
                f"{dynamic_reserve} dynamic/prefill slots, but moe_cache_size={cache_size}"
            )
        # Catch-up copies can install rows from their worker thread. Finish their
        # normal publication path while the old slot tensors and row map are still
        # valid. Clearing a completed future directly would retain its retired None
        # owners and silently drop those protected rows at the rebuild boundary.
        if total_hot_rows:
            self.drain_hot_adaptation_for_rebuild()
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self.cache_size = cache_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            self.bank_caches[name] = torch.empty(
                (cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        self._build_gpufetch_copy_plan()
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        first_hot_slot = cache_size - total_hot_rows
        next_hot_slot = first_hot_slot
        for layer_id, capacity in sorted(self.hot_expert_capacity.items()):
            self._hot_slot_for_row[layer_id] = tuple(
                range(next_hot_slot, next_hot_slot + capacity)
            )
            next_hot_slot += capacity
        self._hot_slots_device = torch.arange(
            first_hot_slot, cache_size, dtype=torch.long, device=self.device
        )
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        if self.cpu_executor is not None:
            # A rebuilt cache is a cold boundary for route prediction as well as LRU.
            self.cpu_executor.reset_disk_lookahead()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        if not preserve_hot_state:
            self.stat_hot_pairs.zero_()
            self.stat_hot_total_pairs.zero_()
            self.stat_pinned_hot_pairs.zero_()
            self.stat_pinned_hot_total_pairs.zero_()
            self._prefill_hot_pairs = 0
            self._prefill_route_pairs = 0
            self._prefill_cpu_experts = 0
            self.decode_freq.zero_()
            self._protected_route_baseline = None
            self._pinned_lru_baseline = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and cache_size < 2 * self.num_experts:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< 2*num_experts {2 * self.num_experts}."
            )
            self.prefill_overlap = False
        self._configure_prefill_overlap_layers()
        if any(buffer_id >= 0 for buffer_id in self._prefill_overlap_buffer_ids):
            self._init_prefill_overlap_buffers()
        if self._hot_staging:
            self._reload_hot_slots()

    def set_alphas(
        self,
        gate_up_alpha: torch.Tensor | None,
        down_alpha: torch.Tensor | None,
        gate_up_input_scale: torch.Tensor | None = None,
        down_input_scale: torch.Tensor | None = None,
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        gate_up_alpha = gate_up_alpha.to(self.device)
        down_alpha = down_alpha.to(self.device)
        if gate_up_input_scale is not None or down_input_scale is not None:
            assert gate_up_input_scale is not None and down_input_scale is not None
            assert gate_up_input_scale.shape == down_input_scale.shape == (total,)
            self.gate_up_alpha = torch.stack(
                (gate_up_alpha, gate_up_input_scale.to(self.device).float())
            )
            self.down_alpha = torch.stack(
                (down_alpha, down_input_scale.to(self.device).float())
            )
            self.moe_activation_dtype = "nvfp4"
        else:
            self.gate_up_alpha = gate_up_alpha
            self.down_alpha = down_alpha
            self.moe_activation_dtype = "bf16"

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid") or self.moe_disk_decode == "gpufetch", (
            "set_cpu_executor requires CPU/hybrid decode or DISK gpufetch"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_hot_split_layer(self, layer_id: int) -> bool:
        """Whether a DISK layer has a compact pinned HOT expert partition."""
        return (
            layer_id in self.hot_expert_capacity
            and layer_id not in getattr(self, "_pinned_hot_layer_ids", ())
        )

    def is_pinned_hot_layer(self, layer_id: int) -> bool:
        """Whether a PINNED layer owns protected rows but keeps plain decode."""
        return (
            layer_id in self.hot_expert_capacity
            and layer_id in getattr(self, "_pinned_hot_layer_ids", ())
        )

    def configure_hot_adaptation(
        self,
        *,
        half_life_steps: int,
        interval_steps: str | int,
        max_swap_bytes: int,
        expert_bytes: int,
        boundary_cap_frac: float = 0.5,
        prefill_weight: float = 1.0,
        prefill_run_cap_frac: float = 0.0,
        post_prefill_tick: bool = False,
        persisted_counter_seed: Mapping[int, tuple[float, ...]] | None = None,
        persisted_seeded_layers: frozenset[int] = frozenset(),
        hot_plan_path: str | None = None,
        hot_plan_identity=None,
        hot_plan_tier_commit: str = "",
        hot_plan_write_enabled: bool = False,
        hot_plan_interval_seconds: float = 600.0,
        idle_ms: int = 500,
        idle_min_interval_ms: int = 2000,
        tp_size: int = 1,
    ) -> None:
        """Allocate bounded staging, load seeds, and arm online adaptation."""
        if not self.hot_expert_capacity:
            return
        if half_life_steps <= 0:
            raise ValueError("HOT adaptation half-life must be positive")
        if max_swap_bytes <= 0 or expert_bytes <= 0:
            raise ValueError("HOT adaptation byte geometry must be positive")
        if (
            isinstance(boundary_cap_frac, bool)
            or not math.isfinite(boundary_cap_frac)
            or not 0 < boundary_cap_frac <= 1
        ):
            raise ValueError("HOT boundary cap fraction must be finite and in (0, 1]")
        if (
            isinstance(prefill_weight, bool)
            or not math.isfinite(prefill_weight)
            or not 0 <= prefill_weight <= 1
        ):
            raise ValueError("HOT prefill weight must be finite and in [0, 1]")
        if (
            isinstance(prefill_run_cap_frac, bool)
            or not math.isfinite(prefill_run_cap_frac)
            or not 0 <= prefill_run_cap_frac <= 1
        ):
            raise ValueError(
                "HOT prefill run cap fraction must be 0 or finite and in (0, 1]"
            )
        if not isinstance(post_prefill_tick, bool):
            raise ValueError("HOT post-prefill tick must be a boolean")
        if (
            isinstance(idle_ms, bool)
            or not isinstance(idle_ms, int)
            or idle_ms < 0
        ):
            raise ValueError("HOT adaptation idle delay must be a non-negative integer")
        if (
            isinstance(idle_min_interval_ms, bool)
            or not isinstance(idle_min_interval_ms, int)
            or idle_min_interval_ms < 0
        ):
            raise ValueError(
                "HOT adaptation idle minimum interval must be a non-negative integer"
            )
        from freetoken.moe.host_banks import alloc_pinned_row_staging
        from freetoken.moe.hot_adapt import (
            HotAdaptIdleTracker,
            HotAdaptIntervalController,
            HotAdaptTokenClock,
            decay_multiplier,
            hot_staging_budget_bytes,
            hot_staging_rows,
        )

        self.hot_adapt_max_swap_bytes = int(max_swap_bytes)
        self.hot_adapt_expert_bytes = int(expert_bytes)
        self.hot_adapt_hot_budget_bytes = (
            sum(self.hot_expert_capacity.values()) * self.hot_adapt_expert_bytes
        )
        self.hot_adapt_boundary_cap_frac = float(boundary_cap_frac)
        self.hot_adapt_prefill_weight = float(prefill_weight)
        self.hot_adapt_prefill_run_cap_frac = float(prefill_run_cap_frac)
        self.hot_adapt_post_prefill_tick = post_prefill_tick
        self._hot_adapt_prefill_run_swapped_bytes = 0
        self._hot_adapt_prefill_run_swaps = 0
        self._hot_adapt_prefill_run_active = False
        self._hot_adapt_prefill_run_generation = 0
        self._hot_adapt_tick_prefill_run_generation = None
        self._hot_adapt_after_prefill_pending = False
        controller = HotAdaptIntervalController.create(
            interval_steps,
            hot_budget_bytes=self.hot_adapt_hot_budget_bytes,
            max_swap_bytes=self.hot_adapt_max_swap_bytes,
        )
        seed = {
            int(layer_id): tuple(float(value) for value in row)
            for layer_id, row in (persisted_counter_seed or {}).items()
        }
        for layer_id, row in seed.items():
            if layer_id not in self.hot_expert_capacity or len(row) != self.num_experts:
                raise ValueError(f"persisted HOT counter layer {layer_id} has invalid geometry")
        self._hot_plan_counter_seed = seed
        self._apply_hot_plan_counter_seed()
        fully_seeded = (
            bool(self.hot_expert_capacity)
            and persisted_seeded_layers == frozenset(self.hot_expert_capacity)
            and all(
                len(self.hot_expert_ids[layer_id]) == capacity
                for layer_id, capacity in self.hot_expert_capacity.items()
            )
        )
        if fully_seeded:
            controller.fill_complete = True
            if controller.auto:
                controller.current_interval = max(
                    controller.current_interval, controller.steady_interval
                )
        self._hot_adapt_interval_controller = controller
        self.hot_adapt_interval_steps = controller.current_interval
        self._hot_adapt_tick_interval_tokens = self.hot_adapt_interval_steps
        self.hot_adapt_enabled = self.hot_adapt_interval_steps > 0
        self._hot_adapt_token_clock = (
            HotAdaptTokenClock(self.hot_adapt_interval_steps)
            if self.hot_adapt_enabled else None
        )
        self._hot_adapt_idle_tracker = (
            None
            if idle_ms == 0 or tp_size != 1
            else HotAdaptIdleTracker(
                idle_seconds=idle_ms / 1000.0,
                min_interval_seconds=idle_min_interval_ms / 1000.0,
            )
        )
        self._hot_decay_factor = decay_multiplier(half_life_steps)
        self._hot_staging_rows = hot_staging_rows(max_swap_bytes, expert_bytes)
        self._hot_staging = alloc_pinned_row_staging(
            [self.bank_sources[name][0] for name in self.bank_schema],
            self._hot_staging_rows,
            pinned=self.device.type == "cuda",
        )
        payload_bytes = sum(t.numel() * t.element_size() for t in self._hot_staging)
        budget_bytes = hot_staging_budget_bytes(max_swap_bytes)
        if payload_bytes > budget_bytes:
            raise ValueError(
                f"one HOT expert row needs {payload_bytes} staging bytes, exceeding "
                f"max_swap plus fixed headroom ({budget_bytes} bytes)"
            )
        self.hot_staging_bytes = budget_bytes
        # These pools use daemon workers omitted from concurrent.futures' atexit
        # join registry. Explicit shutdown still cleans up responsive workers,
        # while process exit may abandon a wedged copy, fsync, or temporary file.
        self._hot_adapt_executor = ExitSafeThreadPoolExecutor(
            max_workers=1, thread_name_prefix="freetoken-hot-adapt"
        )
        self._hot_adapt_stop_event.clear()
        self._hot_adapt_wake_event.clear()
        self._hot_plan_path = hot_plan_path
        self._hot_plan_identity = hot_plan_identity
        self._hot_plan_tier_commit = hot_plan_tier_commit
        self._hot_plan_write_enabled = bool(
            hot_plan_write_enabled and hot_plan_path and hot_plan_identity
        )
        self._hot_plan_interval_seconds = float(hot_plan_interval_seconds)
        self._hot_plan_last_snapshot = time.monotonic()
        self._hot_plan_stop_event.clear()
        if self._hot_plan_write_enabled:
            self._hot_plan_executor = ExitSafeThreadPoolExecutor(
                max_workers=1, thread_name_prefix="freetoken-hot-plan"
            )
        self._reload_hot_slots()
        self._checkpoint_published_hot_slot_owners()
        logger.info_rank0(
            f"MoE HOT expert residency: {sum(self.hot_expert_capacity.values())} "
            f"protected GPU rows across {len(self.hot_expert_capacity)} layers, "
            f"hot_staging_gib={self.hot_staging_bytes / 2**30:.2f}, "
            f"hot_staging_rows={self._hot_staging_rows}"
        )
        mode = "auto" if controller.auto else f"fixed({controller.current_interval})"
        if tp_size != 1:
            idle = "off (tensor parallel)"
        elif idle_ms == 0:
            idle = "off"
        else:
            idle = f"{idle_ms} ms"
        logger.info_rank0(
            f"MoE HOT adaptation intervals: mode={mode}, "
            f"unit=routed_tokens, "
            f"hot_budget_gib={self.hot_adapt_hot_budget_bytes / 2**30:.2f}, "
            f"max_swap_gib={self.hot_adapt_max_swap_bytes / 2**30:.2f}, "
            f"boundary_cap_frac={self.hot_adapt_boundary_cap_frac:.2f}, "
            f"fill_ticks={controller.fill_ticks}, "
            f"target_fill_tokens={controller.target_fill_tokens}, "
            f"fill_interval={controller.fill_interval}, "
            f"steady_interval={controller.steady_interval}, "
            f"current_interval={controller.current_interval}, "
            f"idle={idle}, "
            f"idle_min_interval_ms={idle_min_interval_ms}"
        )
        if not self.hot_adapt_enabled:
            return
        pin = self.device.type == "cuda"
        self._hot_adapt_snapshot_host = torch.empty(
            (self.num_layers, self.num_experts), dtype=torch.float32,
            device="cpu", pin_memory=pin,
        )
        self._hot_adapt_snapshot_device = torch.empty_like(self.decayed_decode_freq)
        if self.device.type == "cuda":
            self._hot_adapt_copy_stream = torch.cuda.Stream(device=self.device)
            self._hot_adapt_snapshot_ready = torch.cuda.Event()

    def _apply_hot_plan_counter_seed(self) -> None:
        """Restore the startup seed after graph capture and prefill warmup resets."""
        for layer_id, row in self._hot_plan_counter_seed.items():
            self.decayed_decode_freq[layer_id].copy_(
                torch.tensor(row, dtype=torch.float32, device=self.device)
            )

    def _protect_hot_slots(self) -> None:
        """Keep permanent HOT rows outside every ordinary LRU victim set."""
        slots = self._hot_slots_device
        if slots is not None and slots.numel():
            self.usage.index_fill_(0, slots, torch.iinfo(torch.int64).max)

    def _protect_hot_layer_slots(self, layer_id: int) -> None:
        """Restore one layer's contiguous protected range after an LRU hit."""
        slots = self._hot_slot_for_row.get(layer_id, ())
        if slots:
            self.usage[slots[0]:slots[-1] + 1].fill_(torch.iinfo(torch.int64).max)

    def _restore_hot_slot_metadata(self) -> None:
        """Restore protected slot ownership after reset without reloading weights."""
        self._protect_hot_slots()
        for layer_id, owners in self._hot_slot_owners.items():
            for row, expert in enumerate(owners):
                if expert is None:
                    continue
                slot = self._hot_slot_for_row[layer_id][row]
                self.slot_for_id[layer_id, expert] = slot
                self.id_of_slot[slot] = layer_id * self.num_experts + expert

    def _stage_hot_rows(self, ready, swaps, stop_event=None):
        if ready is not None:
            ready.synchronize()
        if len(swaps) > self._hot_staging_rows:
            raise RuntimeError("HOT swap plan exceeds the allocated staging rows")
        started_at = time.perf_counter()
        copied: set[tuple[int, int]] = set()
        with torch.inference_mode():
            for stage_row, swap in enumerate(swaps):
                # Check once per whole expert row. Once a row starts, every bank
                # for that row is copied. Preemption bounds additional host staging
                # to one row, but the H2D install of the staged prefix, up to
                # self._hot_staging_rows rows, still lands on the scheduler stream
                # before the next forward and costs 25 to 50 ms on node-4.
                if stop_event is not None and stop_event.is_set():
                    break
                for bank_id, name in enumerate(self.bank_schema):
                    source = self.bank_sources[name][swap.layer_id]
                    self._hot_staging[bank_id][stage_row].copy_(
                        source[swap.incoming_expert]
                    )
                copied.add((swap.layer_id, swap.row))
        return copied, time.perf_counter() - started_at

    def _install_staged_hot_rows(self, swaps) -> None:
        """Copy staged rows into their fixed GPU slots on the publication stream."""
        # Runtime rebuild allocates bank_caches under Engine's inference-mode
        # context. The catch-up path can call this method from a worker, and
        # inference mode is thread-local, so enter it again around the writes.
        with torch.inference_mode():
            for stage_row, swap in enumerate(swaps):
                slot = self._hot_slot_for_row[swap.layer_id][swap.row]
                for bank_id, name in enumerate(self.bank_schema):
                    self.bank_caches[name][slot].copy_(
                        self._hot_staging[bank_id][stage_row],
                        non_blocking=self.device.type == "cuda",
                    )

    def _stage_hot_rows_batched(self, ready, swaps, stop_event=None):
        """Stream catch-up ticks through the one-tick staging allocation."""
        if ready is not None:
            ready.synchronize()
        if stop_event is None:
            stop_event = self._hot_adapt_stop_event
        started_at = time.perf_counter()
        copied: set[tuple[int, int]] = set()
        for start in range(0, len(swaps), self._hot_staging_rows):
            batch = tuple(swaps[start:start + self._hot_staging_rows])
            batch_copied, _ = self._stage_hot_rows(
                None, batch, stop_event
            )
            batch = batch[:len(batch_copied)]
            if not batch:
                break
            if self.device.type == "cuda":
                assert self._hot_adapt_copy_stream is not None
                with torch.cuda.stream(self._hot_adapt_copy_stream):
                    self._install_staged_hot_rows(batch)
                self._hot_adapt_copy_stream.synchronize()
            else:
                self._install_staged_hot_rows(batch)
            copied.update(batch_copied)
            if stop_event.is_set():
                break
        return copied, time.perf_counter() - started_at

    def _checkpoint_published_hot_slot_owners(self) -> None:
        self._hot_plan_last_published_owners = {
            layer_id: tuple(owners)
            for layer_id, owners in self._hot_slot_owners.items()
        }

    def _reload_hot_slots(self) -> None:
        """Stream every published/seeded HOT row through the bounded stage."""
        from freetoken.moe.hot_adapt import HotSwap, finish_hot_swaps

        mapping = [[-1] * self.num_experts for _ in range(self.num_layers)]
        items = [
            HotSwap(layer_id, row, expert, None)
            for layer_id, owners in sorted(self._hot_slot_owners.items())
            for row, expert in enumerate(owners)
            if expert is not None
        ]
        for start in range(0, len(items), self._hot_staging_rows):
            batch = tuple(items[start:start + self._hot_staging_rows])
            copied, _staging_seconds = self._stage_hot_rows(None, batch)
            self._install_staged_hot_rows(batch)
            # The next batch reuses the same pinned rows. Complete this H2D before
            # the CPU overwrites them; this path runs only at startup/rebuild.
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            mapping = finish_hot_swaps(mapping, batch, copied)
        self.slot_for_id.fill_(-1)
        self.id_of_slot.fill_(-1)
        self._replace_hot_mapping(mapping)
        self._restore_hot_slot_metadata()

    def configure_session_profiles(
        self,
        *,
        max_sessions: int,
        enabled: bool,
        protect_experts: int,
        half_life_steps: int,
    ) -> None:
        """Allocate the fixed per-table top-k sketch before CUDA graph capture."""
        from freetoken.moe.hot_adapt import decay_multiplier
        from freetoken.moe.session_profile import SESSION_EXPERT_PROFILE_TOPK

        self.session_profile_enabled = bool(enabled)
        self.session_profile_topk = SESSION_EXPERT_PROFILE_TOPK
        self._session_protect_limit = max(0, int(protect_experts))
        self._session_decay_factor = decay_multiplier(half_life_steps)
        if not self.session_profile_enabled:
            self.session_profile_ids = None
            self.session_profile_counts = None
            return
        # One extra row is the scheduler's graph-padding request. It is never exported.
        shape = (int(max_sessions) + 1, self.num_layers, self.session_profile_topk)
        self.session_profile_ids = torch.full(
            shape, -1, dtype=torch.int32, device=self.device
        )
        self.session_profile_counts = torch.zeros(
            shape, dtype=torch.float32, device=self.device
        )

    def _record_session_profile(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Merge this decode layer's routes into each request's bounded top-k sketch."""
        if not self.session_profile_enabled or self.session_profile_ids is None:
            return
        from freetoken.core import get_global_ctx

        try:
            batch = get_global_ctx().batch
        except AssertionError:
            return
        if not batch.is_decode or batch.active_table_idx is None:
            return
        from freetoken.moe.offload_kernels import update_session_profile

        update_session_profile(
            self, layer_id, expert_ids, batch.active_table_idx.reshape(-1)
        )

    def activate_session_profile(
        self, uid: int, table_idx: int, profile, *, restored: bool
    ) -> None:
        """Seed the live sketch after resource admission; prefetch already ran at queue entry."""
        if not self.session_profile_enabled or self.session_profile_ids is None:
            return
        self.session_profile_ids[table_idx].fill_(-1)
        self.session_profile_counts[table_idx].zero_()
        if profile is not None:
            tensors = profile.to_tensors()
            ids = tensors["expert_profile.ids"].to(self.device, dtype=torch.int32)
            counts = tensors["expert_profile.counts"].to(self.device, dtype=torch.float32)
            layers = min(ids.shape[0], self.num_layers)
            self.session_profile_ids[table_idx, :layers].copy_(ids[:layers])
            self.session_profile_counts[table_idx, :layers].copy_(counts[:layers])
        if restored and profile is not None:
            self._resume_timing[int(uid)] = {
                "steps": 0.0,
                "first": 0.0,
                "step64": 0.0,
                "step65": 0.0,
                "last": 0.0,
            }

    def export_session_profile(self, table_idx: int):
        """Synchronously compact one parked table row into its few-KB host object."""
        if not self.session_profile_enabled or self.session_profile_ids is None:
            return None
        from freetoken.moe.session_profile import SessionExpertProfile

        ids = self.session_profile_ids[table_idx].detach().cpu().tolist()
        counts = self.session_profile_counts[table_idx].detach().cpu().tolist()
        out_ids: list[tuple[int, ...]] = []
        out_counts: list[tuple[float, ...]] = []
        for layer_ids, layer_counts in zip(ids, counts):
            pairs = [
                (int(expert), float(count))
                for expert, count in zip(layer_ids, layer_counts)
                if int(expert) >= 0 and math.isfinite(float(count)) and float(count) > 0
            ]
            pairs.sort(key=lambda pair: (-pair[1], pair[0]))
            out_ids.append(tuple(expert for expert, _ in pairs))
            out_counts.append(tuple(count for _, count in pairs))
        if not any(out_ids):
            return None
        return SessionExpertProfile(tuple(out_ids), tuple(out_counts))

    def admit_session_profile(self, uid: int, profile) -> int:
        """Plan and enqueue advisory warming at waiting-queue admission."""
        if not self.session_profile_enabled or profile is None:
            return 0
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return 0
        from freetoken.moe.session_profile import (
            SESSION_ADAPT_INJECTION_WEIGHT,
            plan_session_prefetch,
        )

        plan = plan_session_prefetch(
            profile,
            self.layer_residency,
            hot_experts=self.hot_expert_ids,
            protect_limit=self._session_protect_limit,
        )
        self._session_protections.admit(uid, plan.protected)
        from freetoken.moe import offload_kernels

        try:
            for layer_id, experts in plan.promote:
                ids = torch.tensor(experts, dtype=torch.int32, device=self.device)
                self._pending_src_layer = layer_id
                self._pending_whole_layer = False
                if self.is_hot_split_layer(layer_id):
                    offload_kernels.ensure_experts_hot(
                        self, layer_id, ids, record_stats=False
                    )
                else:
                    offload_kernels.ensure_experts(
                        self, layer_id, ids, record_stats=False
                    )
                self.copy_missing()
                self._boost_protected_slots()
        except Exception as exc:
            # ensure_experts publishes slot ownership before the copy. If the copy
            # fails, invalidate the advisory slot map so no forward can consume an
            # incompletely filled row.
            if self.device.type == "cuda":
                offload_kernels.reset_cache(self)
            else:
                self.slot_for_id.fill_(-1)
                self.id_of_slot.fill_(-1)
                self.usage.zero_()
            logger.warning(
                f"session expert H2D prefetch for request {uid} was skipped: {exc!r}"
            )
        try:
            if self.cpu_executor is not None:
                for layer_id, experts in plan.willneed:
                    self.cpu_executor._prefetch_selected(layer_id, experts)
        except Exception as exc:
            logger.warning(
                f"session expert WILLNEED for request {uid} was skipped: {exc!r}"
            )
        try:
            for layer_id, expert, count in profile.ranked_pairs():
                if layer_id < self.num_layers and expert < self.num_experts:
                    self.decayed_decode_freq[layer_id, expert] += (
                        float(count) * SESSION_ADAPT_INJECTION_WEIGHT
                    )
            self._boost_protected_slots()
        except Exception as exc:
            logger.warning(
                f"session expert adaptation hint for request {uid} was skipped: {exc!r}"
            )
        self._session_prefetch_experts += plan.expert_count
        return plan.expert_count

    def _boost_protected_slots(self) -> None:
        if not self.session_profile_enabled or self.session_profile_ids is None:
            return
        protected = self._session_protections.all()
        if not protected:
            return
        flat_ids = torch.tensor(
            [layer * self.num_experts + expert for layer, expert in sorted(protected)],
            dtype=torch.long,
            device=self.device,
        )
        slots = self.slot_for_id.view(-1).index_select(0, flat_ids)
        slots = slots[slots >= 0].to(torch.long)
        if slots.numel():
            self.usage.index_fill_(0, slots, (1 << 60))

    def release_session_profile(self, uid: int) -> None:
        old = self._session_protections.release(uid)
        still = self._session_protections.all()
        released = [pair for pair in old if pair not in still]
        if released:
            flat = torch.tensor(
                [layer * self.num_experts + expert for layer, expert in released],
                dtype=torch.long,
                device=self.device,
            )
            slots = self.slot_for_id.view(-1).index_select(0, flat)
            slots = slots[slots >= 0].to(torch.long)
            if slots.numel():
                self.usage.index_fill_(0, slots, int(self.step.item()))
        timing = self._resume_timing.pop(int(uid), None)
        if timing is not None:
            steps, first, step64, step65, last = (
                timing["steps"], timing["first"], timing["step64"],
                timing["step65"], timing["last"],
            )
            if steps >= 64 and last > first:
                self._last_resume_warm_rate = 63.0 / max(step64 - first, 1e-9)
            if steps > 65 and last > step65:
                self._last_resume_steady_rate = (steps - 65.0) / (last - step65)

    def record_resume_decode_batch(self, reqs) -> None:
        now = time.perf_counter()
        for req in reqs:
            timing = self._resume_timing.get(int(req.uid))
            if timing is None:
                continue
            timing["steps"] += 1.0
            if timing["steps"] == 1:
                timing["first"] = now
            if timing["steps"] == 64:
                timing["step64"] = now
            if timing["steps"] == 65:
                timing["step65"] = now
            timing["last"] = now

    def session_profile_stats(self, *, reset: bool = False) -> dict[str, float | int]:
        protected = self._session_protections.all()
        result = {
            "resume_prefetch_experts": self._session_prefetch_experts,
            "protected_experts": len(protected),
            "resume_first64_tok_s": self._last_resume_warm_rate,
            "resume_steady_tok_s": self._last_resume_steady_rate,
            "resume_warmup_ratio": (
                self._last_resume_warm_rate / self._last_resume_steady_rate
                if self._last_resume_steady_rate > 0 else 0.0
            ),
        }
        if reset:
            self._session_prefetch_experts = 0
        return result

    def _publish_hot_mapping(self) -> None:
        """Update the fixed device table in place, preserving graph addresses."""
        assert self._hot_mapping_host is not None
        self.hot_row_for_expert.copy_(
            self._hot_mapping_host, non_blocking=self.device.type == "cuda"
        )

    def _replace_hot_mapping(self, mapping: list[list[int]]) -> None:
        assert self._hot_mapping_host is not None
        # This pinned source is fixed too. Retirement's row-copy worker waits on
        # the mapping-copy event, and the next plan waits on a later snapshot event,
        # so no phase can mutate it while an earlier H2D copy is still in flight.
        self._hot_mapping_host.copy_(torch.tensor(mapping, dtype=torch.int32))
        self._publish_hot_mapping()

    def _hot_mapping_lists(self) -> list[list[int]]:
        assert self._hot_mapping_host is not None
        return self._hot_mapping_host.tolist()

    def _plan_hot_adaptation(
        self,
        ready,
        token: int,
        swap_budget_bytes: int | None,
        boundary: str,
        tick_count: int,
    ):
        if ready is not None:
            ready.synchronize()
        assert self._hot_adapt_snapshot_host is not None
        from freetoken.moe.hot_adapt import (
            hot_catchup_swap_bytes,
            plan_hot_swaps,
            recompute_hot_partition,
        )

        counts = {
            layer_id: tuple(float(value) for value in self._hot_adapt_snapshot_host[layer_id])
            for layer_id in self.hot_expert_capacity
        }
        budget_bytes = sum(self.hot_expert_capacity.values()) * self.hot_adapt_expert_bytes
        desired = recompute_hot_partition(
            counts,
            frozenset(self.hot_expert_capacity),
            budget_bytes=budget_bytes,
            expert_bytes=self.hot_adapt_expert_bytes,
            num_experts=self.num_experts,
            capacities=self.hot_expert_capacity,
        )
        owners = {layer_id: tuple(rows) for layer_id, rows in self._hot_slot_owners.items()}
        if swap_budget_bytes is None:
            swap_budget_bytes = hot_catchup_swap_bytes(
                self.hot_adapt_max_swap_bytes,
                self.hot_adapt_expert_bytes,
                tick_count,
                hot_budget_bytes=self.hot_adapt_hot_budget_bytes,
                boundary_cap_frac=self.hot_adapt_boundary_cap_frac,
            )
        swaps = plan_hot_swaps(
            counts,
            owners,
            desired,
            expert_bytes=self.hot_adapt_expert_bytes,
            max_swap_bytes=swap_budget_bytes,
        )
        total = sum(sum(layer) for layer in counts.values())
        hot = sum(
            counts[layer_id][expert]
            for layer_id, rows in owners.items()
            for expert in rows if expert is not None
        )
        rate = hot / total if total else 0.0
        if boundary != "idle":
            logger.info_rank0(
                f"MoE HOT adaptation tick token={token}, boundary={boundary}: "
                f"decayed_hot_pair_rate={rate:.2%}, "
                f"ticks={tick_count}, planned_swaps={len(swaps)}, "
                f"max_swap_gib_per_tick="
                f"{self.hot_adapt_max_swap_bytes / 2**30:.2f}, "
                f"boundary_cap_frac={self.hot_adapt_boundary_cap_frac:.2f}"
            )
        return swaps, rate, tick_count

    def _retire_hot_adaptation_swaps(
        self,
        swaps,
        *,
        tick_count: int = 1,
    ) -> None:
        from freetoken.moe.hot_adapt import (
            hot_boundary_interval_tokens,
            retire_hot_swaps,
        )

        # Back-off must represent all bytes actually staged at this boundary.
        # Scaling the nominal token interval by the equivalent number of full
        # swap batches avoids treating a multi-tick catch-up as one 250-token tick.
        staged_bytes = len(swaps) * self.hot_adapt_expert_bytes
        self._hot_adapt_tick_staged_bytes = staged_bytes
        self._hot_adapt_tick_interval_tokens = hot_boundary_interval_tokens(
            self._hot_adapt_tick_interval_tokens,
            self.hot_adapt_max_swap_bytes,
            staged_bytes,
        )
        retired = retire_hot_swaps(self._hot_mapping_lists(), swaps)
        self._replace_hot_mapping(retired)
        for swap in swaps:
            self._hot_slot_owners[swap.layer_id][swap.row] = None
        self.hot_expert_ids = {
            layer_id: tuple(sorted(owner for owner in owners if owner is not None))
            for layer_id, owners in self._hot_slot_owners.items()
        }
        ready = None
        if self.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        assert self._hot_adapt_executor is not None
        self._hot_adapt_swaps_pending = swaps
        self._hot_adapt_phase = "copy"
        self._hot_adapt_worker_installs = tick_count > 1
        stage = (
            self._stage_hot_rows_batched
            if self._hot_adapt_worker_installs else self._stage_hot_rows
        )
        self._hot_adapt_future = self._hot_adapt_executor.submit(
            stage, ready, swaps, self._hot_adapt_stop_event
        )
        self._hot_adapt_future.add_done_callback(
            lambda _future: self._hot_adapt_wake_event.set()
        )

    def _finish_hot_adaptation_swaps(
        self, copied_rows, staging_seconds: float = 0.0
    ) -> None:
        from freetoken.moe.hot_adapt import finish_hot_swaps

        swaps = self._hot_adapt_swaps_pending
        executed = tuple(
            swap
            for swap in swaps
            if (swap.layer_id, swap.row) in copied_rows
        )
        abandoned = tuple(
            swap
            for swap in swaps
            if (swap.layer_id, swap.row) not in copied_rows
        )
        mapping = self._hot_mapping_lists()
        for swap in abandoned:
            if swap.outgoing_expert is not None:
                mapping[swap.layer_id][swap.outgoing_expert] = swap.row
            self._hot_slot_owners[swap.layer_id][swap.row] = swap.outgoing_expert
        finished = finish_hot_swaps(mapping, executed, copied_rows)
        if not self._hot_adapt_worker_installs:
            self._install_staged_hot_rows(executed)
        for swap in executed:
            slot = self._hot_slot_for_row[swap.layer_id][swap.row]
            if swap.outgoing_expert is not None:
                self.slot_for_id[swap.layer_id, swap.outgoing_expert] = -1
            self.id_of_slot[slot] = swap.layer_id * self.num_experts + swap.incoming_expert
            self.slot_for_id[swap.layer_id, swap.incoming_expert] = slot
            self._hot_slot_owners[swap.layer_id][swap.row] = swap.incoming_expert
        self._protect_hot_slots()
        # A single-batch bank copy is ordered here before the mapping H2D. The
        # batched worker host-synchronizes its copy stream before its future
        # completes. Decode can therefore observe either the retired set or the
        # fully installed set, never a mapping to an incomplete GPU row.
        self._replace_hot_mapping(finished)
        self.hot_expert_ids = {
            layer_id: tuple(sorted(owner for owner in owners if owner is not None))
            for layer_id, owners in self._hot_slot_owners.items()
        }
        self._checkpoint_published_hot_slot_owners()
        if getattr(self, "_hot_adapt_tick_boundary", None) == "idle":
            self.hot_adapt_idle_swaps += len(executed)
        else:
            self.hot_adapt_swaps += len(executed)
        if (
            getattr(self, "_hot_adapt_tick_boundary", None) == "prefill"
            and self._hot_adapt_tick_prefill_run_generation
            == self._hot_adapt_prefill_run_generation
        ):
            self._hot_adapt_prefill_run_swapped_bytes += (
                len(executed) * self.hot_adapt_expert_bytes
            )
            self._hot_adapt_prefill_run_swaps += len(executed)
        self._hot_adapt_tick_executed_swaps = len(executed)
        self._hot_adapt_swaps_pending = ()
        self._hot_adapt_worker_installs = False
        self._hot_adapt_phase = None
        self._hot_adapt_future = None
        self._complete_hot_adaptation_tick(staging_seconds=staging_seconds)

    def _complete_hot_adaptation_tick(self, *, staging_seconds: float) -> None:
        controller = self._hot_adapt_interval_controller
        if controller is None:
            return
        partition_full = all(
            owner is not None
            for owners in self._hot_slot_owners.values()
            for owner in owners
        )
        old_interval = controller.current_interval
        switched, backed_off, backoff_interval = controller.complete_tick(
            partition_full=partition_full,
            tick_interval=self._hot_adapt_tick_interval_tokens,
            staging_seconds=staging_seconds,
            covered_seconds=self._hot_adapt_tick_covered_seconds,
        )
        self.hot_adapt_interval_steps = controller.current_interval
        clock = self._hot_adapt_token_clock
        if (
            clock is not None
            and getattr(self, "_hot_adapt_tick_boundary", None) != "idle"
        ):
            clock.set_interval(self.hot_adapt_interval_steps)
        if backed_off:
            fraction = staging_seconds / self._hot_adapt_tick_covered_seconds
            logger.info_rank0(
                f"MoE HOT adaptation bandwidth back-off: staging_ms="
                f"{staging_seconds * 1000:.1f}, covered_token_wall_ms="
                f"{self._hot_adapt_tick_covered_seconds * 1000:.1f}, "
                f"staged_gib={self._hot_adapt_tick_staged_bytes / 2**30:.2f}, "
                f"staging_fraction={fraction:.1%}, interval_floor="
                f"{old_interval}->{backoff_interval}, next_interval="
                f"{controller.current_interval}"
            )
        if switched:
            logger.info_rank0(
                f"MoE HOT adaptation fill complete: fill_interval="
                f"{controller.fill_interval}, steady_interval="
                f"{controller.steady_interval}, current_interval="
                f"{controller.current_interval}"
            )
        if getattr(self, "_hot_adapt_tick_boundary", None) == "idle":
            self._checkpoint_published_hot_slot_owners()
            if self._hot_adapt_tick_executed_swaps:
                self.snapshot_hot_plan()
            after = self.decayed_hot_pair_rate()
            logger.info_rank0(
                f"MoE HOT adaptation idle tick token={self.hot_adapt_routed_tokens}: "
                f"planned_swaps={self._hot_adapt_tick_planned_swaps}, "
                f"executed_swaps={self._hot_adapt_tick_executed_swaps}, "
                f"decayed_hot_pair_rate="
                f"{self._hot_adapt_tick_rate_before:.2%}->{after:.2%}"
            )
            tracker = getattr(self, "_hot_adapt_idle_tracker", None)
            if tracker is not None:
                tracker.tick_completed(
                    time.monotonic(), self._hot_adapt_tick_executed_swaps
                )

    def _poll_hot_adaptation(self, *, preempt_idle: bool = False) -> None:
        future = self._hot_adapt_future
        if future is None or not future.done():
            return
        if self._hot_adapt_phase == "plan":
            swaps, rate, tick_count = future.result()
            self._hot_adapt_future = None
            self._hot_adapt_phase = None
            self._hot_adapt_tick_planned_swaps = len(swaps)
            self._hot_adapt_tick_rate_before = rate
            idle_tick = getattr(self, "_hot_adapt_tick_boundary", None) == "idle"
            if idle_tick and preempt_idle:
                self._hot_adapt_tick_executed_swaps = 0
                self.hot_adapt_ticks -= 1
                self.hot_adapt_ticks_idle -= 1
                logger.info_rank0(
                    f"MoE HOT adaptation idle tick token="
                    f"{self.hot_adapt_routed_tokens}: preempted after planning"
                )
                return
            if swaps:
                self._retire_hot_adaptation_swaps(
                    swaps,
                    tick_count=tick_count,
                )
            else:
                self._hot_adapt_tick_executed_swaps = 0
                self._complete_hot_adaptation_tick(staging_seconds=0.0)
        elif self._hot_adapt_phase == "copy":
            copied_rows, staging_seconds = future.result()
            self._finish_hot_adaptation_swaps(copied_rows, staging_seconds)
        else:
            raise RuntimeError("HOT adaptation future completed in an invalid phase")

    def _write_hot_plan_snapshot(
        self,
        owners,
        counters: torch.Tensor,
        fence: _HotPlanWriteFence,
    ) -> bool:
        from freetoken.moe.hot_adapt import atomic_write_hot_plan, make_hot_plan_document

        document = make_hot_plan_document(
            identity=self._hot_plan_identity,
            disk_layer_ids=tuple(
                layer_id for layer_id in sorted(self.hot_expert_capacity)
                if self.is_hot_split_layer(layer_id)
            ),
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            hot_budget_bytes=self.hot_adapt_hot_budget_bytes,
            tier_commit=self._hot_plan_tier_commit,
            protected_slots=owners,
            decayed_counters={
                layer_id: counters[layer_id].tolist()
                for layer_id in sorted(self.hot_expert_capacity)
            },
        )
        if document is None:
            if not self._hot_plan_zero_logged:
                logger.info_rank0("MoE HOT plan write skipped: decayed counters are all zero")
                self._hot_plan_zero_logged = True
            return False
        return atomic_write_hot_plan(
            self._hot_plan_path,
            document,
            publish=fence.publish,
        )

    def _collect_finished_hot_plan_write(self) -> None:
        future = self._hot_plan_future
        if future is None or not future.done():
            return
        self._hot_plan_future = None
        self._hot_plan_write_fence = None
        try:
            wrote = future.result()
        except Exception as exc:  # noqa: BLE001 - persistence must not stop serving
            logger.warning_rank0(f"MoE HOT plan write failed: {exc!r}")
        else:
            if wrote:
                logger.info_rank0(f"MoE HOT plan wrote {self._hot_plan_path!r}")

    def snapshot_hot_plan(
        self,
        *,
        force: bool = False,
        snapshot_at: Mapping[int, tuple[int | None, ...]] | None = None,
    ) -> Future | None:
        """Take the small D2H state snapshot and queue encoding plus fsync."""
        if (
            not getattr(self, "_hot_plan_write_enabled", False)
            or getattr(self, "_hot_plan_executor", None) is None
        ):
            return None
        self._collect_finished_hot_plan_write()
        if self._hot_plan_future is not None:
            return self._hot_plan_future
        now = time.monotonic()
        if not force and (
            self._hot_plan_stop_event.is_set()
            or now - self._hot_plan_last_snapshot < self._hot_plan_interval_seconds
            or self._hot_adapt_future is not None
        ):
            return None
        # The mapping owners are host bookkeeping. Copy counters to host here so
        # the background task never calls CUDA or touches mutable serving state.
        owner_source = self._hot_slot_owners if snapshot_at is None else snapshot_at
        owners = {layer_id: tuple(rows) for layer_id, rows in owner_source.items()}
        counters = self.decayed_decode_freq.detach().to("cpu", copy=True)
        self._hot_plan_last_snapshot = now
        fence = _HotPlanWriteFence()
        self._hot_plan_write_fence = fence
        self._hot_plan_future = self._hot_plan_executor.submit(
            self._write_hot_plan_snapshot, owners, counters, fence
        )
        return self._hot_plan_future

    def _start_hot_adaptation_tick(
        self,
        *,
        token: int,
        boundary: str,
        tick_count: int,
        swap_budget_bytes: int | None = None,
    ) -> None:
        """Snapshot counters and submit one bounded planner boundary."""
        self._hot_adapt_tick_boundary = boundary
        self._hot_adapt_tick_planned_swaps = 0
        self._hot_adapt_tick_executed_swaps = 0
        self._hot_adapt_tick_rate_before = 0.0
        self._hot_adapt_tick_interval_tokens = self.hot_adapt_interval_steps
        self._hot_adapt_tick_staged_bytes = 0
        self._hot_adapt_tick_prefill_run_generation = (
            getattr(self, "_hot_adapt_prefill_run_generation", 0)
            if boundary == "prefill" else None
        )
        if (
            boundary == "prefill"
            and getattr(self, "hot_adapt_prefill_run_cap_frac", 0.0) > 0
            and swap_budget_bytes is None
        ):
            from freetoken.moe.hot_adapt import (
                hot_catchup_swap_bytes,
                prefill_run_swap_budget,
            )

            per_boundary_bytes = hot_catchup_swap_bytes(
                self.hot_adapt_max_swap_bytes,
                self.hot_adapt_expert_bytes,
                tick_count,
                hot_budget_bytes=self.hot_adapt_hot_budget_bytes,
                boundary_cap_frac=self.hot_adapt_boundary_cap_frac,
            )
            swap_budget_bytes = prefill_run_swap_budget(
                per_boundary_bytes,
                self.hot_adapt_expert_bytes,
                getattr(self, "_hot_adapt_prefill_run_swapped_bytes", 0),
                hot_budget_bytes=self.hot_adapt_hot_budget_bytes,
                run_cap_frac=self.hot_adapt_prefill_run_cap_frac,
            )
            if swap_budget_bytes == 0:
                self._complete_hot_adaptation_tick(staging_seconds=0.0)
                return
        assert self._hot_adapt_snapshot_host is not None
        self._hot_adapt_snapshot_device.copy_(self.decayed_decode_freq)
        ready = None
        if self.device.type == "cuda":
            assert self._hot_adapt_copy_stream is not None
            assert self._hot_adapt_snapshot_ready is not None
            begin = torch.cuda.Event()
            begin.record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._hot_adapt_copy_stream):
                self._hot_adapt_copy_stream.wait_event(begin)
                self._hot_adapt_snapshot_host.copy_(
                    self._hot_adapt_snapshot_device, non_blocking=True
                )
                self._hot_adapt_snapshot_ready.record(self._hot_adapt_copy_stream)
            ready = self._hot_adapt_snapshot_ready
        else:
            self._hot_adapt_snapshot_host.copy_(self._hot_adapt_snapshot_device)
        assert self._hot_adapt_executor is not None
        self._hot_adapt_phase = "plan"
        self._hot_adapt_future = self._hot_adapt_executor.submit(
            self._plan_hot_adaptation,
            ready,
            token,
            swap_budget_bytes,
            boundary,
            tick_count,
        )
        self._hot_adapt_future.add_done_callback(
            lambda _future: self._hot_adapt_wake_event.set()
        )

    def _wait_for_hot_adaptation(self, timeout_seconds: float = 0.01) -> None:
        """Sleep until worker progress or the next bounded request poll."""
        self._hot_adapt_wake_event.wait(timeout=max(0.0, timeout_seconds))
        self._hot_adapt_wake_event.clear()

    def _finish_preempted_idle_tick(self) -> None:
        """Stop an idle tick and restore a safe published set before serving."""
        if getattr(self, "_hot_adapt_tick_boundary", None) != "idle":
            return
        stop_event = getattr(self, "_hot_adapt_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        future = self._hot_adapt_future
        if (
            future is not None
            and self._hot_adapt_phase == "plan"
            and future.cancel()
        ):
            self._hot_adapt_future = None
            self._hot_adapt_phase = None
            self._hot_adapt_tick_executed_swaps = 0
            self.hot_adapt_ticks -= 1
            self.hot_adapt_ticks_idle -= 1
            logger.info_rank0(
                f"MoE HOT adaptation idle tick token={self.hot_adapt_routed_tokens}: "
                "preempted before planning"
            )
        while self._hot_adapt_future is not None:
            future = self._hot_adapt_future
            if not future.done():
                self._wait_for_hot_adaptation()
                continue
            self._poll_hot_adaptation(preempt_idle=True)

    def _begin_hot_adapt_prefill_run(self) -> None:
        if not getattr(self, "_hot_adapt_prefill_run_active", False):
            self._hot_adapt_prefill_run_active = True
            self._hot_adapt_prefill_run_generation = (
                getattr(self, "_hot_adapt_prefill_run_generation", 0) + 1
            )
            self._hot_adapt_prefill_run_swapped_bytes = 0
            self._hot_adapt_prefill_run_swaps = 0
        self._hot_adapt_after_prefill_pending = True

    def _end_hot_adapt_prefill_run(self, *, clear_pending: bool) -> None:
        self._hot_adapt_prefill_run_active = False
        self._hot_adapt_prefill_run_swapped_bytes = 0
        if clear_pending:
            self._hot_adapt_after_prefill_pending = False

    def hot_adapt_while_idle(
        self,
        request_pending: Callable[[], bool],
        wait_for_request: Callable[[float], bool] | None = None,
    ) -> None:
        """Run eligible idle ticks until convergence or request arrival."""
        tracker = getattr(self, "_hot_adapt_idle_tracker", None)
        if not self.hot_adapt_enabled or tracker is None:
            return
        tracker.begin_idle(time.monotonic())
        try:
            self._poll_hot_adaptation()
            if not tracker.has_evidence():
                return

            while True:
                if request_pending():
                    self._finish_preempted_idle_tick()
                    self._hot_adapt_stop_event.clear()
                    return
                self._poll_hot_adaptation()
                now = time.monotonic()
                if self._hot_adapt_future is None and tracker.due(now):
                    self._end_hot_adapt_prefill_run(clear_pending=True)
                    self._hot_adapt_stop_event.clear()
                    tracker.tick_started()
                    self.hot_adapt_ticks += 1
                    self.hot_adapt_ticks_idle += 1
                    self._hot_adapt_tick_covered_seconds = 0.0
                    self._start_hot_adaptation_tick(
                        token=self.hot_adapt_routed_tokens,
                        boundary="idle",
                        tick_count=1,
                    )
                    continue
                if self._hot_adapt_future is None and not tracker.has_evidence():
                    return
                timeout_seconds = (
                    0.01
                    if self._hot_adapt_future is not None
                    else tracker.seconds_until_due(now)
                )
                if wait_for_request is None:
                    self._wait_for_hot_adaptation(timeout_seconds)
                elif wait_for_request(timeout_seconds):
                    self._finish_preempted_idle_tick()
                    self._hot_adapt_stop_event.clear()
                    return
        except BaseException:
            if (
                getattr(self, "_hot_adapt_tick_boundary", None) == "idle"
                and self._hot_adapt_future is not None
            ):
                self._hot_adapt_stop_event.set()
            raise
        finally:
            tracker.end_idle()

    def _hot_adapt_token_boundary(self, routed_tokens: int, boundary: str) -> None:
        """Advance the shared routed-token clock without waiting on staging work.

        Due thresholds are deferred while work is active, then consumed together
        at the next free boundary. Token tick counters include every consumed
        threshold, so ``hot_swaps_per_interval`` retains the accumulated token-tick
        denominator for both automatic and explicit fixed intervals. Idle ticks
        are tracked separately and excluded from that operator sizing metric.
        """
        if routed_tokens < 0:
            raise ValueError("HOT adaptation routed token count must be non-negative")
        if boundary not in ("prefill", "decode"):
            raise ValueError(f"invalid HOT adaptation boundary {boundary!r}")
        # This runs after model work, keeping protection maintenance outside capture.
        self._protect_hot_slots()
        self._boost_protected_slots()
        self.snapshot_hot_plan()
        if not self.hot_adapt_enabled:
            return
        clock = self._hot_adapt_token_clock
        assert clock is not None
        # An idle fill transition may change the controller interval. Apply it
        # only when routed-token accounting resumes so the idle tick itself does
        # not mutate any token-clock field.
        if clock.interval != self.hot_adapt_interval_steps:
            clock.set_interval(self.hot_adapt_interval_steps)
        clock.advance(routed_tokens)
        tracker = getattr(self, "_hot_adapt_idle_tracker", None)
        if routed_tokens > 0 and tracker is not None:
            tracker.note_routed_pairs()
        now = time.perf_counter()
        if self._hot_adapt_window_started_at is None:
            self._hot_adapt_window_started_at = now
        self._poll_hot_adaptation()
        if boundary == "prefill":
            self._begin_hot_adapt_prefill_run()
        else:
            self._end_hot_adapt_prefill_run(
                clear_pending=not getattr(
                    self, "hot_adapt_post_prefill_tick", False
                )
            )
        controller = self._hot_adapt_interval_controller
        force_post_prefill = (
            boundary == "decode"
            and getattr(self, "hot_adapt_post_prefill_tick", False)
            and getattr(self, "_hot_adapt_after_prefill_pending", False)
        )
        if controller is None or (
            not force_post_prefill
            and clock.routed_tokens < clock.next_tick_token
        ):
            return
        if self._hot_adapt_future is not None:
            if not self._hot_adapt_deferred_logged:
                logger.info_rank0(
                    f"MoE HOT adaptation tick token={clock.routed_tokens}, "
                    f"boundary={boundary}: "
                    f"copy_or_plan_in_progress, decayed_hot_pair_rate deferred"
                )
                self._hot_adapt_deferred_logged = True
            return
        if force_post_prefill:
            tick_count = 1
            clock.consume_forced_tick()
            self._hot_adapt_after_prefill_pending = False
        else:
            tick_count = clock.advance(0)
            for _ in range(tick_count):
                clock.consume_tick()
        self.hot_adapt_ticks += tick_count
        if boundary == "prefill":
            self.hot_adapt_ticks_prefill += tick_count
        else:
            self.hot_adapt_ticks_decode += tick_count
        self._hot_adapt_deferred_logged = False
        if tracker is not None:
            tracker.tick_started()
        self._hot_adapt_tick_covered_seconds = max(
            0.0, now - self._hot_adapt_window_started_at
        )
        self._hot_adapt_window_started_at = now
        self._start_hot_adaptation_tick(
            token=clock.routed_tokens,
            boundary=boundary,
            tick_count=tick_count,
            swap_budget_bytes=(
                self.hot_adapt_max_swap_bytes
                if force_post_prefill else None
            ),
        )

    def hot_adapt_prefill_boundary(self) -> None:
        """Account only prefill tokens observed by the HOT split counting path."""
        routed_tokens = self._hot_adapt_prefill_tokens_counted
        self._hot_adapt_prefill_tokens_counted = 0
        self._hot_adapt_token_boundary(routed_tokens, "prefill")

    def hot_adapt_step_boundary(self, batch_size: int = 1) -> None:
        """Account one decode batch and start any due tick."""
        if batch_size < 0:
            raise ValueError("HOT adaptation decode batch size must be non-negative")
        self._hot_adapt_token_boundary(batch_size, "decode")

    @property
    def hot_adapt_routed_tokens(self) -> int:
        clock = self._hot_adapt_token_clock
        return clock.routed_tokens if clock is not None else 0

    def decayed_hot_pair_rate(self) -> float:
        """Return DISK HOT coverage, excluding PINNED protected traffic."""
        if not self.hot_adapt_enabled:
            return 0.0
        counts = self.decayed_decode_freq.tolist()
        disk_layers = (
            [
                layer_id for layer_id in self.hot_expert_capacity
                if self.is_hot_split_layer(layer_id)
            ]
            if self.hot_expert_capacity
            else list(self._hot_slot_owners)
        )
        total = sum(sum(counts[layer_id]) for layer_id in disk_layers)
        hot = sum(
            counts[layer_id][expert]
            for layer_id, owners in self._hot_slot_owners.items()
            if layer_id in disk_layers
            for expert in owners if expert is not None
        )
        return hot / total if total else 0.0

    def request_hot_plan_stop(self) -> None:
        """Quiet periodic snapshots before the scheduler enters its drain path."""
        self._hot_plan_stop_event.set()

    def _drain_hot_adaptation(self, deadline: float) -> bool:
        """Publish completed rerank and copy work until empty or the deadline."""
        while self._hot_adapt_future is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            future = self._hot_adapt_future
            try:
                future.result(timeout=remaining)
            except Exception:  # noqa: BLE001 - shutdown must keep progressing
                if not future.done():
                    return False
                try:
                    future.result()
                except Exception as completed_exc:  # noqa: BLE001
                    logger.warning_rank0(
                        f"MoE HOT adaptation drain failed: {completed_exc!r}"
                    )
                    self._hot_adapt_future = None
                    self._hot_adapt_phase = None
                    return False
            try:
                self._poll_hot_adaptation()
            except Exception as exc:  # noqa: BLE001 - shutdown must keep progressing
                logger.warning_rank0(f"MoE HOT adaptation drain failed: {exc!r}")
                self._hot_adapt_future = None
                self._hot_adapt_phase = None
                return False
        return True

    def _hot_adapt_stop_wait_seconds(self) -> float:
        boundary_bytes = int(
            self.hot_adapt_boundary_cap_frac * self.hot_adapt_hot_budget_bytes
        )
        return (
            boundary_bytes / _HOT_ADAPT_SHUTDOWN_BYTES_PER_SECOND
            + _HOT_ADAPT_SHUTDOWN_FIXED_SECONDS
        )

    def _wait_for_hot_adaptation_worker(self, timeout_seconds: float) -> bool:
        future = self._hot_adapt_future
        if future is None:
            return True
        try:
            future.result(timeout=max(0.0, timeout_seconds))
        except Exception:  # noqa: BLE001 - shutdown must keep progressing
            if not future.done():
                return False
            try:
                future.result()
            except Exception as completed_exc:  # noqa: BLE001
                logger.warning_rank0(
                    "MoE HOT adaptation worker stopped with an error: "
                    f"{completed_exc!r}"
                )
            return True
        return True

    def _shutdown_hot_adaptation_executor(
        self, executor: ThreadPoolExecutor, timeout_seconds: float
    ) -> bool:
        """Join the executor without letting a stuck staging worker block shutdown."""
        joined = threading.Event()

        def join_executor() -> None:
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            finally:
                joined.set()

        join_thread = threading.Thread(
            target=join_executor,
            name="freetoken-hot-adapt-shutdown",
            daemon=True,
        )
        join_thread.start()
        return joined.wait(max(0.0, timeout_seconds))

    def _abandon_hot_plan_write(self, future: Future) -> None:
        """Cancel a queued write and fence a running write before its rename."""
        fence = getattr(self, "_hot_plan_write_fence", None)
        if fence is not None:
            fence.cancel()
        future.cancel()
        if getattr(self, "_hot_plan_future", None) is future:
            self._hot_plan_future = None
            self._hot_plan_write_fence = None

    @staticmethod
    def _hot_plan_owner_counts(
        owners: Mapping[int, tuple[int | None, ...]],
    ) -> tuple[int, int]:
        return (
            sum(owner is not None for rows in owners.values() for owner in rows),
            sum(any(owner is not None for owner in rows) for rows in owners.values()),
        )

    @staticmethod
    def _log_hot_shutdown_step(
        step: str,
        started_at: float,
        budget_seconds: float,
        outcome: str,
    ) -> None:
        elapsed = time.monotonic() - started_at
        logger.info_rank0(
            f"MoE HOT shutdown {step}: elapsed={elapsed:.3f}s, "
            f"budget={budget_seconds:.3f}s, outcome={outcome}"
        )

    def _persisted_hot_plan_counts(self) -> tuple[int, int] | None:
        """Read the plan file and report the protected expert and layer counts."""
        path = getattr(self, "_hot_plan_path", None)
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                protected = json.load(handle).get("protected_slots")
        except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(protected, dict) or any(
            not isinstance(owners, list) for owners in protected.values()
        ):
            return None
        return (
            sum(len(owners) for owners in protected.values()),
            sum(bool(owners) for owners in protected.values()),
        )

    def shutdown_hot_adaptation(self, timeout_seconds: float = 3.0) -> None:
        """Bound adaptation drain, persist one final snapshot, then stop workers."""
        self.request_hot_plan_stop()
        drain_budget = max(0.0, float(timeout_seconds))
        drain_started = time.monotonic()
        drain_deadline = drain_started + drain_budget
        if self.cpu_executor is not None:
            cancel = getattr(
                self.cpu_executor, "cancel_prefill_populate_overlap", None
            )
            if cancel is not None:
                cancel(wait=False)
        drained = self._drain_hot_adaptation(drain_deadline)
        self._log_hot_shutdown_step(
            "drain",
            drain_started,
            drain_budget,
            "completed" if drained else "timed_out_or_failed",
        )
        timed_out_owners = None
        worker_timed_out = False
        stop_wait_budget = 0.0
        stop_wait_started = time.monotonic()
        stopped = True
        if not drained:
            timed_out_owners = dict(self._hot_plan_last_published_owners)
            worker_timed_out = self._hot_adapt_future is not None
            if worker_timed_out:
                self._hot_adapt_stop_event.set()
                computed_stop_wait = max(
                    0.0, self._hot_adapt_stop_wait_seconds()
                )
                stop_wait_budget = min(
                    computed_stop_wait, _HOT_ADAPT_STOP_WAIT_MAX_SECONDS
                )
                if stop_wait_budget < computed_stop_wait:
                    logger.warning_rank0(
                        "MoE HOT adaptation stop wait clamped from "
                        f"{computed_stop_wait:g} to {stop_wait_budget:g} s"
                    )
                stopped = self._wait_for_hot_adaptation_worker(stop_wait_budget)
                if not stopped:
                    logger.warning_rank0(
                        "MoE HOT adaptation worker exceeded its bounded stop wait; "
                        "attempting a bounded executor join"
                    )
        self._log_hot_shutdown_step(
            "stop_wait",
            stop_wait_started,
            stop_wait_budget,
            "completed" if stopped else "timed_out",
        )
        final_write_started = time.monotonic()
        final_write_deadline = final_write_started + _HOT_PLAN_FINAL_WRITE_SECONDS
        existing_write = self._hot_plan_future
        if existing_write is not None:
            if existing_write.done():
                self._collect_finished_hot_plan_write()
            else:
                self._abandon_hot_plan_write(existing_write)
                logger.warning_rank0(
                    "MoE HOT periodic plan write was abandoned before final snapshot"
                )
        try:
            final_write = self.snapshot_hot_plan(
                force=True, snapshot_at=timed_out_owners
            )
        except Exception as exc:  # noqa: BLE001 - persistence must not stop shutdown
            logger.warning_rank0(f"MoE HOT final plan snapshot failed: {exc!r}")
            final_write = None
        final_wrote = False
        final_write_confirmed = final_write is None
        if final_write is not None:
            try:
                final_wrote = bool(
                    final_write.result(
                        timeout=max(0.0, final_write_deadline - time.monotonic())
                    )
                )
                final_write_confirmed = True
            except Exception:  # noqa: BLE001 - persistence must not stop shutdown
                if final_write.done():
                    final_write_confirmed = True
                else:
                    self._abandon_hot_plan_write(final_write)
                    logger.warning_rank0(
                        "MoE HOT final plan write did not finish before shutdown"
                    )
            self._collect_finished_hot_plan_write()
        self._log_hot_shutdown_step(
            "final_write",
            final_write_started,
            _HOT_PLAN_FINAL_WRITE_SECONDS,
            (
                "written"
                if final_wrote
                else ("completed_without_write" if final_write_confirmed else "timed_out")
            ),
        )
        if timed_out_owners is not None:
            drain_outcome = "timed out" if worker_timed_out else "failed"
            attempted_experts, attempted_layers = self._hot_plan_owner_counts(
                timed_out_owners
            )
            if final_wrote:
                persisted_counts = self._persisted_hot_plan_counts()
                if persisted_counts is not None:
                    seeded_experts, seeded_layers = persisted_counts
                    logger.warning_rank0(
                        f"MoE HOT adaptation drain {drain_outcome}; wrote the last "
                        f"published slot set containing {seeded_experts} seeded experts "
                        f"from {seeded_layers} layers"
                    )
                else:
                    logger.warning_rank0(
                        f"MoE HOT adaptation drain {drain_outcome}; plan written and "
                        "fsynced; readback failed; the file on disk is the new plan"
                    )
            elif final_write is not None and not final_write_confirmed:
                logger.warning_rank0(
                    f"MoE HOT adaptation drain {drain_outcome}; attempted "
                    f"{attempted_experts} experts across {attempted_layers} layers; "
                    f"write did not confirm within {_HOT_PLAN_FINAL_WRITE_SECONDS:g} s"
                )
            else:
                logger.warning_rank0(
                    f"MoE HOT adaptation drain {drain_outcome}; attempted "
                    f"{attempted_experts} experts across {attempted_layers} layers; "
                    "no plan was published"
                )
        executor = self._hot_adapt_executor
        join_started = time.monotonic()
        joined = True
        if executor is not None:
            joined = self._shutdown_hot_adaptation_executor(
                executor, _HOT_ADAPT_EXECUTOR_JOIN_SECONDS
            )
            if joined:
                self._hot_adapt_executor = None
                if hasattr(self, "_hot_adapt_copy_stream"):
                    self._hot_adapt_copy_stream = None
            else:
                logger.warning_rank0(
                    "MoE HOT adaptation executor join exceeded its "
                    f"{_HOT_ADAPT_EXECUTOR_JOIN_SECONDS:g} second reserve; "
                    "leaving the copy stream intact"
                )
        self._log_hot_shutdown_step(
            "executor_join",
            join_started,
            _HOT_ADAPT_EXECUTOR_JOIN_SECONDS,
            "completed" if joined else "timed_out",
        )
        plan_executor = self._hot_plan_executor
        if plan_executor is not None:
            plan_executor.shutdown(wait=False, cancel_futures=True)
            self._hot_plan_executor = None

    def is_gpufetch_layer(self, layer_id: int) -> bool:
        """Whether this file-backed layer decodes through the GPU slot cache."""
        return (
            self.moe_disk_decode == "gpufetch"
            and layer_id < len(self.layer_residency)
            and self.layer_residency[layer_id] == "disk"
        )

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE/DISK): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def prefetch_disk_experts(self, layer_id: int, expert_ids) -> int:
        """Prefetch selected file-backed rows, or no-op for a RAM-resident layer."""
        if self.layer_residency[layer_id] != "disk":
            return 0
        assert self.cpu_executor is not None, "DISK layer requires the CPU MoE executor"
        return self.cpu_executor.prefetch_experts(layer_id, expert_ids, is_prefill=True)

    def prepare_disk_prefill(self, layer_id: int, expert_ids):
        """Warm one bounded routed union before shared CPU prefill compute.

        Returns None whenever coalescing is not active for this layer (flag
        off, copy prefill, or an executor predating the seam - test doubles
        included); the caller treats a None lease as "run the original
        advisory sweep instead".
        """
        if self.layer_residency[layer_id] != "disk":
            return None
        assert self.cpu_executor is not None, "DISK layer requires the CPU MoE executor"
        prepare = getattr(self.cpu_executor, "prepare_prefill_layer", None)
        if (
            self.moe_disk_prefill != "cpu"
            or self.moe_prefill_coalesce == "off"
            or prepare is None
        ):
            return None
        return prepare(layer_id, expert_ids)

    def release_disk_prefill(self, lease) -> None:
        """Apply one-pass cache advice after the CPU layer has finished."""
        if lease is not None:
            assert self.cpu_executor is not None
            self.cpu_executor.release_prefill_layer(lease)

    def schedule_next_chunk_disk_prefill(self, layer_id: int, expert_ids):
        """Use this chunk's layer-0 routes to predict the next chunk's warm set."""
        if (
            layer_id != 0
            or self.layer_residency[layer_id] != "disk"
            or self.moe_disk_prefill != "cpu"
            or self.moe_prefill_coalesce != "populate"
        ):
            return None
        assert self.cpu_executor is not None
        schedule = getattr(
            self.cpu_executor, "schedule_prefill_layer_overlap", None
        )
        return schedule(layer_id, expert_ids) if schedule is not None else None

    def disk_prefetch_stats(self, *, reset: bool = False) -> dict:
        result = (
            self.cpu_executor.disk_prefetch_stats(reset=reset)
            if self.cpu_executor is not None else {}
        )
        hot_pairs = int(self.stat_hot_pairs.item())
        total_pairs = int(self.stat_hot_total_pairs.item())
        result["hot_pair_rate"] = hot_pairs / total_pairs if total_pairs else 0.0
        result["hot_pairs"] = hot_pairs
        result["routed_pairs"] = total_pairs
        result.update(
            self.protected_routing_stats(
                realized_hits=hot_pairs,
                routed_pairs=total_pairs,
                reset=reset,
            )
        )
        result["prefill_hot_route_frac"] = (
            self._prefill_hot_pairs / self._prefill_route_pairs
            if self._prefill_route_pairs else 0.0
        )
        result["prefill_cpu_experts"] = self._prefill_cpu_experts
        ticks = self.hot_adapt_ticks - self._hot_adapt_ticks_reported
        ticks_prefill = (
            self.hot_adapt_ticks_prefill - self._hot_adapt_ticks_prefill_reported
        )
        ticks_decode = (
            self.hot_adapt_ticks_decode - self._hot_adapt_ticks_decode_reported
        )
        ticks_idle = (
            self.hot_adapt_ticks_idle - self._hot_adapt_ticks_idle_reported
        )
        token_ticks = ticks - ticks_idle
        token_swaps = self.hot_adapt_swaps - self._hot_adapt_swaps_reported
        idle_swaps = (
            self.hot_adapt_idle_swaps - self._hot_adapt_idle_swaps_reported
        )
        # A background copy may complete after the status window that contained
        # its tick. Attribute such completions to one interval instead of dropping
        # them merely because this report window has no new tick.
        result["hot_swaps_per_interval"] = token_swaps / max(token_ticks, 1)
        result["hot_adapt_idle_swaps_per_tick"] = (
            idle_swaps / max(ticks_idle, 1)
        )
        result["decayed_hot_pair_rate"] = self.decayed_hot_pair_rate()
        result["hot_adapt_interval"] = self.hot_adapt_interval_steps
        result["hot_adapt_ticks_prefill"] = ticks_prefill
        result["hot_adapt_prefill_run_swaps"] = getattr(
            self, "_hot_adapt_prefill_run_swaps", 0
        )
        result["hot_adapt_ticks_decode"] = ticks_decode
        result["hot_adapt_ticks_idle"] = ticks_idle
        pinned_layers = [
            layer_id for layer_id in range(self.num_layers)
            if self.layer_residency
            and self.layer_residency[layer_id] == "pinned"
        ]
        pinned_hot_pairs = int(self.stat_pinned_hot_pairs.item())
        pinned_hot_total = int(self.stat_pinned_hot_total_pairs.item())
        result["pinned_hot_pair_rate"] = (
            pinned_hot_pairs / pinned_hot_total if pinned_hot_total else 0.0
        )
        pinned_missing = 0
        pinned_steps = 0
        if self.collect_stats and pinned_layers:
            lru = self.lru_stats.tolist()
            baseline = self._pinned_lru_baseline or [
                (0, 0) for _ in range(self.num_layers)
            ]
            pinned_missing = sum(
                max(0, int(lru[layer_id][Stat.MISS]) - baseline[layer_id][0])
                for layer_id in pinned_layers
            )
            pinned_steps = max(
                (
                    max(
                        0,
                        int(lru[layer_id][Stat.CALLS]) - baseline[layer_id][1],
                    )
                    for layer_id in pinned_layers
                ),
                default=0,
            )
            if reset:
                self._pinned_lru_baseline = [
                    (int(row[Stat.MISS]), int(row[Stat.CALLS])) for row in lru
                ]
        result["pinned_missing_per_step"] = (
            pinned_missing / pinned_steps if pinned_steps else 0.0
        )
        expert_bytes = int(getattr(self, "hot_adapt_expert_bytes", 0) or 0)
        if expert_bytes <= 0 and self.bank_sources:
            # No adaptation configured: derive one expert row's bytes from the first
            # bank that has a host source (DISK layers may carry None here).
            for per_layer in self.bank_sources.values():
                source = next((s for s in per_layer if s is not None), None)
                if source is not None:
                    expert_bytes += source[0].numel() * source.element_size()
        result["pinned_h2d_bytes_per_step"] = (
            result["pinned_missing_per_step"] * expert_bytes
        )
        result.update(self.session_profile_stats(reset=reset))
        if reset:
            self.stat_hot_pairs.zero_()
            self.stat_hot_total_pairs.zero_()
            self.stat_pinned_hot_pairs.zero_()
            self.stat_pinned_hot_total_pairs.zero_()
            self._prefill_hot_pairs = 0
            self._prefill_route_pairs = 0
            self._prefill_cpu_experts = 0
            self._hot_adapt_ticks_reported = self.hot_adapt_ticks
            self._hot_adapt_ticks_prefill_reported = self.hot_adapt_ticks_prefill
            self._hot_adapt_ticks_decode_reported = self.hot_adapt_ticks_decode
            self._hot_adapt_ticks_idle_reported = self.hot_adapt_ticks_idle
            self._hot_adapt_swaps_reported = self.hot_adapt_swaps
            self._hot_adapt_idle_swaps_reported = self.hot_adapt_idle_swaps
        return result

    def protected_routing_stats(
        self,
        *,
        realized_hits: int,
        routed_pairs: int,
        reset: bool = False,
    ) -> dict:
        """Compare protected-slot coverage with a same-capacity routing oracle.

        Route frequencies accumulate on the device under ``--moe-collect-stats``.
        This method is called only by the periodic status reporter, where one host
        read is acceptable. For each protected layer, the oracle takes the C most
        routed experts, where C is that layer's protected-slot capacity. Both oracle
        and realized hit rates use the exact routed-pair count as their denominator.
        """
        disk_capacity = {
            layer_id: capacity
            for layer_id, capacity in self.hot_expert_capacity.items()
            if self.is_hot_split_layer(layer_id)
        }
        if not self.collect_stats or not disk_capacity or routed_pairs <= 0:
            return {}

        snapshot = self.decode_freq.tolist()
        baseline = self._protected_route_baseline
        if baseline is None:
            baseline = [
                [0] * self.num_experts for _ in range(self.num_layers)
            ]

        oracle_hits = 0
        for layer_id, capacity in disk_capacity.items():
            counts = [
                max(0, current - previous)
                for current, previous in zip(
                    snapshot[layer_id], baseline[layer_id], strict=True
                )
            ]
            oracle_hits += sum(sorted(counts, reverse=True)[:capacity])

        if reset:
            self._protected_route_baseline = snapshot
        return {
            "oracle_hit": oracle_hits / routed_pairs,
            "realized_hit": realized_hits / routed_pairs,
            "oracle_hits": oracle_hits,
            "oracle_routed_pairs": routed_pairs,
        }

    def record_prefill_hot_split(
        self, raw_ids: torch.Tensor, hot_mask: torch.Tensor
    ) -> None:
        """Account actual GPU-served pairs and distinct cold experts per chunk."""
        hot_pairs = int(hot_mask.sum().item())
        cold_ids = raw_ids.masked_select(~hot_mask)
        cold_ids = cold_ids[cold_ids >= 0]
        self._prefill_hot_pairs += hot_pairs
        self._prefill_route_pairs += int(raw_ids.numel())
        self._prefill_cpu_experts += int(torch.unique(cold_ids).numel())

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        idx = layer_id * self.num_experts + (
            self.id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[..., idx], self.down_alpha[..., idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[..., lo:hi], self.down_alpha[..., lo:hi]

    def prefill_slot_tables(
        self, layer_id: int,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, torch.Tensor] | None,
        int,
    ]:
        """Protected-slot tables, per-slot alphas, and their shared row count."""
        views = self.bank_views()
        alphas = self.alphas_for_slots(layer_id)
        assert views, "grouped prefill requires at least one expert table"
        row_count = views[0].shape[0]
        for name, table in zip(self.bank_schema, views):
            assert table.shape[0] == row_count, (
                f"expert table {name} has {table.shape[0]} rows, expected {row_count}"
            )
        if alphas is not None:
            for name, table in zip(("gate_up_alpha", "down_alpha"), alphas):
                assert table.shape[0] == row_count, (
                    f"alpha table {name} has {table.shape[0]} rows, expected {row_count}"
                )
        return views, alphas, row_count

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the full ``[S]`` slot cache
        (decode), or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # The double buffers borrow the slot cache's first 2 * num_experts slots
        # (one full expert layer per buffer), one view per registered bank.
        self.prefill_bank_buffers = [
            cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
            for _, cache in self.banks
        ]
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
        # ensure_experts evicts them first.
        self.usage[slot_start:slot_end].zero_()

    def begin_prefill(self) -> None:
        if not any(buffer_id >= 0 for buffer_id in self._prefill_overlap_buffer_ids):
            return
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")
        if not self.prefill_overlap_for_layer(layer_id):
            raise RuntimeError(
                f"layer {layer_id} is unpinned and cannot use prefill overlap"
            )

        assert self.banks and self.prefill_bank_buffers

        buffer_id = self._prefill_overlap_buffer_ids[layer_id]
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
                buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)

        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= 2 * E
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_dst_ptrs is not None:
            prefill_hit_compact(self, layer_id, buffer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._copy_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(self._copy_dst_ptrs_host[b] + (buffer_id * E + starts) * feat)
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + starts * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        if not self.prefill_overlap_for_layer(layer_id):
            raise RuntimeError(
                f"layer {layer_id} is unpinned and cannot use prefill overlap"
            )
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        buffer_id = self._prefill_overlap_buffer_ids[layer_id]
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        if not self.prefill_overlap_for_layer(layer_id):
            raise RuntimeError(
                f"layer {layer_id} is unpinned and cannot use prefill overlap"
            )
        buffer_id = self._prefill_overlap_buffer_ids[layer_id]
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        from freetoken.moe.offload_kernels import bump_decayed_freq, ensure_experts

        self.record_decode_frequency(layer_id, expert_ids)
        pinned_hot = self.is_pinned_hot_layer(layer_id)
        if pinned_hot and self.collect_stats:
            raw_ids = expert_ids.reshape(-1).long()
            slots = self.slot_for_id[layer_id].index_select(0, raw_ids)
            first_hot_slot = self.cache_size - sum(self.hot_expert_capacity.values())
            self.stat_pinned_hot_pairs += (slots >= first_hot_slot).sum()
            self.stat_pinned_hot_total_pairs += raw_ids.numel()
        if pinned_hot and self.hot_adapt_enabled:
            bump_decayed_freq(self, layer_id, expert_ids)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts(self, layer_id, expert_ids)
        if pinned_hot:
            # A protected hit receives the ordinary LRU timestamp. Restore the
            # sentinel in the same captured stream before another layer can evict it.
            self._protect_hot_layer_slots(layer_id)

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        self.record_decode_frequency(layer_id, expert_ids)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction
        )

    def ensure_experts_hot(
        self,
        layer_id: int,
        expert_ids: torch.Tensor,
        *,
        route_weight: float = 1.0,
    ) -> int:
        """Route only this DISK layer's current HOT experts through the slot cache.

        HOT routes become cache slots and missing copies use compact pinned source
        rows. COLD routes become -1 for the CPU partial, exactly like hybrid overflow.
        Return the number of routed token rows counted by this update. Prefill
        callers record that count so the request boundary advances the shared
        clock only when this exact HOT split path ran.
        """
        from freetoken.moe.offload_kernels import ensure_experts_hot

        if layer_id not in self.hot_expert_capacity:
            raise ValueError(f"layer {layer_id} has no HOT expert partition")
        routed_tokens = (
            int(expert_ids.shape[0]) if expert_ids.ndim else int(expert_ids.numel())
        )
        self.record_decode_frequency(layer_id, expert_ids)
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts_hot(
            self,
            layer_id,
            expert_ids,
            route_weight=route_weight,
        )
        # The kernel updates hit timestamps for compatibility with ordinary LRU.
        # Restore the permanent sentinel inside graph capture so later layers in
        # the same decode step cannot evict HOT slots.
        self._protect_hot_slots()
        return routed_tokens

    def record_hot_adapt_prefill_tokens(self, routed_tokens: int) -> None:
        """Record a token count returned by the prefill HOT split counting path."""
        if routed_tokens < 0:
            raise ValueError("HOT adaptation prefill token count must be non-negative")
        if self.hot_adapt_enabled:
            # Every split layer sees the same flattened token rows. Use max so a
            # multi-layer prefill advances the request clock once, not once per layer.
            self._hot_adapt_prefill_tokens_counted = max(
                self._hot_adapt_prefill_tokens_counted,
                routed_tokens,
            )

    def record_decode_frequency(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Accumulate raw per-expert route counts before ids are rewritten to slots."""
        self._record_session_profile(layer_id, expert_ids)
        if self.collect_stats or self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        self._restore_hot_slot_metadata()
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)
        self.decayed_decode_freq.zero_()
        self._apply_hot_plan_counter_seed()
        if self.session_profile_ids is not None:
            self.session_profile_ids.fill_(-1)
            self.session_profile_counts.zero_()
        if self.cpu_executor is not None:
            # Graph capture calls reset before live serving. Do not let synthetic
            # warmup routes seed the first real decode prediction.
            self.cpu_executor.reset_disk_lookahead()

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.stat_hot_pairs.zero_()
        self.stat_hot_total_pairs.zero_()
        self.stat_pinned_hot_pairs.zero_()
        self.stat_pinned_hot_total_pairs.zero_()
        self._pinned_lru_baseline = None
        self._prefill_hot_pairs = 0
        self._prefill_route_pairs = 0
        self._prefill_cpu_experts = 0
        if self.cpu_executor is not None:
            self.cpu_executor.reset_disk_stats()

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and
        the active count. The CPU computes (missing - fetched) experts. Device-side;
        accumulates both the scalar totals and the per-layer breakdown."""
        assert 0 <= layer_id < self.num_layers, f"layer_id {layer_id} out of range [0, {self.num_layers})"
        missing = self.num_missing_full.sum()
        fetched = self.num_indices.sum()
        active = self.active_mask.sum()
        self.stat_missing += missing
        self.stat_fetched += fetched
        self.stat_active += active
        self.stat_calls += 1
        self.stat_missing_layer[layer_id] += missing
        self.stat_fetched_layer[layer_id] += fetched
        self.stat_active_layer[layer_id] += active
        self.stat_steps_layer[layer_id] += 1

    def decode_miss_stats(self) -> dict:
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        result = {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }
        if disk := self.disk_prefetch_stats():
            result["disk"] = disk
        return result

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_miss_layer_profile(self) -> dict:
        """JSON-ready layer traffic and per-expert hit profile."""
        return serialize_moe_layer_profile(
            self.decode_miss_stats_per_layer(), self.decode_freq.tolist()
        )

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.cache_size / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if layer_id in self.hot_expert_capacity and not self._pending_whole_layer:
            # Published HOT experts are permanent GPU hits; COLD experts route to
            # the CPU partial. No decode reload source exists by design.
            return
        if layer_id in self._unpinned_layers and not (
            layer_id in self.hot_expert_capacity and not self._pending_whole_layer
        ):
            if self.is_gpufetch_layer(layer_id) and not self._pending_whole_layer:
                assert self.cpu_executor is not None
                assert self._gpufetch_num_host is not None
                assert self._gpufetch_ids_host is not None
                assert self._gpufetch_stage_indices is not None
                # Fixed-size captured D2H control copies. num_indices tells the native
                # coordinator how many ids are valid; the ring capacity covers the
                # maximum distinct routes for every configured decode batch.
                self._gpufetch_num_host.copy_(self.num_indices, non_blocking=True)
                self._gpufetch_ids_host.copy_(
                    self.src_indices[: self._gpufetch_capacity], non_blocking=True,
                )
                self.cpu_executor.gpufetch(layer_id)
                # dst/src index buffers must be the same length for the copy
                # kernels; decode misses are bounded by the ring capacity, so the
                # capacity-long prefix of evict_slots covers every live entry.
                evict_prefix = self.evict_slots[: self._gpufetch_capacity]
                if self._gpufetch_fused_ok:
                    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

                    fast_index_copy_multi_jit(
                        self._gpufetch_dst_ptrs,
                        self._gpufetch_src_ptrs,
                        self._gpufetch_feat_bytes,
                        evict_prefix,
                        self._gpufetch_stage_indices,
                        self.num_indices,
                    )
                else:
                    from freetoken.kernel import fast_index_copy_jit

                    for stage, (_, cache) in zip(self._gpufetch_staging, self.banks):
                        fast_index_copy_jit(
                            cache,
                            evict_prefix,
                            stage,
                            self._gpufetch_stage_indices,
                            self.num_indices,
                        )
                return
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            # the only copy a non-pinned layer ever needs is the non-overlap prefill materialize, which schedules the whole layer into slots [0, num_experts) with position == expert id -- a plain synchronous pageable H2D copy
            # never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)
            for per_layer, cache in self.banks:
                cache[: self.num_experts].copy_(per_layer[layer_id])
            return
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            fast_index_copy_multi_jit(
                self._copy_dst_ptrs,
                self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes,
                self.evict_slots,
                self.src_indices,
                self.num_indices,
            )
            return

        from freetoken.kernel import fast_index_copy_jit

        for per_layer, cache in self.banks:
            source = per_layer[layer_id]
            fast_index_copy_jit(
                cache,
                self.evict_slots,
                source,
                self.src_indices,
                self.num_indices,
            )


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    # A model whose MoE blocks are bespoke nn.Modules (not OffloadMoELayer) declares its
    # offload layers explicitly via this hook (e.g. DeepSeek-V4-Flash); attach_offload_moe_cache
    # then sets .offload_cache on each yielded layer just like the OffloadMoELayer walk.
    hook = getattr(model, "_iter_offload_moe_layers", None)
    if hook is not None:
        yield from hook()
        return

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
