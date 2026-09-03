from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    # FULL-attention K/V storage. "auto" is resolved after the attention
    # backend and assigned GPU are known.
    kv_cache_dtype: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    # None is the implicit Triton default. Keeping the sentinel lets activation-scale
    # auto-selection distinguish it from an explicit --nvfp4-backend triton.
    nvfp4_backend: str | None = None
    # Routed activation precision for the SM120 flashinfer expert path.
    moe_activation_dtype: str = "auto"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    # Expert bank source: FTW, a byte-identical safetensors index, or automatic detection.
    bank_source: str = "auto"
    # Transparent huge pages for anonymous and eligible file-backed expert banks.
    # "auto" advises supported Linux mappings, "on" requires Linux support, and
    # "off" preserves ordinary page mappings.
    moe_bank_hugepages: str = "auto"
    # Optional tmpfs mirror for file-backed DISK expert banks. The mount must use
    # huge=always or huge=within_size. Existing matching mirrors count toward capacity.
    moe_bank_hugepages_tmpfs: str | None = None
    moe_bank_hugepages_tmpfs_margin_gib: float = 1.0
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # Automatic request-boundary KV growth funded by shrinking --moe-cache-auto.
    kv_ladder: str = "on"
    # Parser provenance used to distinguish an explicit request from the default-on policy.
    kv_ladder_explicit: bool = False
    # Startup geometry derived after the KV pool cost and final page size are known.
    kv_ladder_floor_tokens: int | None = field(default=None, init=False)
    kv_ladder_cap_tokens: int | None = field(default=None, init=False)
    kv_ladder_explicit_cap: bool = field(default=False, init=False)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Automatic host residency reserves these pinned layers for the GPU prefill
    # overlap path before assigning the remaining layers to CPU or DISK.
    # "auto" fits as many as the pin budget permits, "off" reserves none, and a
    # positive integer string forces that exact count.
    moe_gpu_prefill_layers: str = "auto"
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # File-backed FTW or indexed safetensors bank layers. Uses the same grammar as
    # moe_cpu_layers. Decode uses the CPU executor by default or the GPU slot cache.
    moe_disk_layers: str | None = None
    # Optional per-MoE-layer traffic scores used only by automatic DISK spill
    # selection. Explicit moe_disk_layers remains authoritative.
    moe_disk_layer_profile: str | None = None
    # Protected GPU HOT row capacity for DISK layers. A profile seeds the rows;
    # without one, online adaptation starts the fixed partition all-cold.
    moe_hot_expert_budget_gib: float = 0.0
    # Protected GPU HOT row capacity for PINNED layers. These rows share the
    # protected slot range and adaptation machinery with DISK HOT rows.
    moe_pinned_hot_budget_gib: float = 0.0
    # Online HOT-set adaptation. "auto" derives the fill cadence from the HOT
    # allocation and swap bound; an integer retains a fixed cadence, with 0 off.
    moe_hot_adapt_halflife_steps: int = 2000
    moe_hot_adapt_interval_steps: str | int = "auto"
    moe_hot_adapt_max_swap_gib: float = 0.5
    moe_hot_adapt_boundary_cap_frac: float = 0.5
    # Persist the adapted protected-slot assignment and its decayed routing counts.
    # auto reads an existing plan and writes only when its directory is writable.
    moe_hot_plan_persist: str = "auto"
    moe_hot_plan_dir: str | None = None
    moe_hot_plan_interval_minutes: float = 10.0
    # Zero disables scheduler-idle ticks without disabling token-boundary adaptation.
    moe_hot_adapt_idle_ms: int = 500
    moe_hot_adapt_idle_min_interval_ms: int = 2000
    # DISK-layer prefill compute: "cpu" reads only routed experts through the CPU
    # executor; "copy" restores the whole-layer pageable GPU-copy path for benchmarks.
    moe_disk_prefill: str = "cpu"
    # Warm a CPU prefill layer's bounded routed union. "populate" reads the file,
    # "on" keeps advisory WILLNEED, and "off" preserves the original seam behavior.
    # The setting is inert for the DISK copy path.
    moe_prefill_coalesce: str = "populate"
    # Split DISK prefill routes across protected HOT slots and the CPU executor.
    moe_prefill_hot_split: str = "on"
    # GPU kernel for the protected HOT partial. Grouped reuses each resident row
    # across the chunk; decode retains the route-at-a-time A/B baseline.
    moe_prefill_split_kernel: str = "grouped"
    # Group CPU-prefill routes by expert and use the native row-batched NVFP4
    # W4A8 kernel. Unsupported formats or setup failures retain the serial path.
    moe_cpu_prefill_batch: str = "on"
    # DISK-layer decode: "cpu" preserves the native CPU executor path; "gpufetch"
    # faults only LRU-missing routed expert rows into a pinned staging ring, copies
    # them into the existing GPU slot cache, and runs the normal GPU expert GEMM.
    moe_disk_decode: str = "cpu"
    # DISK bank residency backend. "madvise" preserves the file-mmap path; "uffd"
    # installs FTW expert rows into anonymous mappings under a userspace LRU.
    moe_disk_pager: str = "madvise"
    # Predict madvise WILLNEED rows from the preceding decode step. UFFD ignores this
    # setting because its userspace pager already prefetches logical expert rows.
    moe_disk_lookahead: str = "off"
    # Restore a parked session's compact routed-expert working set at request
    # admission. Protection is bounded per live session and is strictly advisory.
    session_expert_prefetch: str = "on"
    session_protect_experts: int = 64
    # Diagnostic decode instrumentation. Records CUDA phase boundaries and native
    # CPU task spans, then emits interval-averaged per-step timings on the decode log.
    moe_step_timing: bool = False
    # Host expert-tier budgets are resolved together at engine startup. The pin
    # budget is internal; its explicit input remains FREETOKEN_PIN_BUDGET_GB.
    host_cache_reserve_gib: float | None = None
    moe_pager_budget_gib: float | None = None
    moe_pin_budget_gib: float | None = field(default=None, init=False, repr=False)
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # Qwen3.8 Flash-Next PLE table storage. "pinned" is the original full-bank UVA
    # path; "cached" keeps hot mapped rows in a bounded pinned bank; "disk" stages
    # mapped rows; "uring" streams rows with Linux io_uring; "hmm" gathers from
    # mapped shards directly.
    ple_backend: str = "pinned"
    # Bulk-stage a prefill chunk's deduplicated PLE rows when HMM is selected.
    # Decode remains on the direct HMM path regardless of this setting.
    ple_prefill_gather: str = "on"
    ple_cache_gib: float = 8.0
    ple_cache_warm: str | None = None
    ple_cache_profile_out: str | None = None
    ple_uring_staging_mib: int = 64
    ple_uring_queue_depth: int = 64
    # Qwen3.8-Flash-Next native multi-token prediction head. Kept as an explicit
    # on/off choice instead of a boolean so the CLI and serialized daemon config
    # have one stable spelling.
    speculative_mtp: str = "off"
    # Patch 11-lite intentionally supports one draft only. Keep the explicit
    # knob so attempts to reuse older K>1 launch commands fail loudly.
    mtp_draft_tokens: int = 1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Whole-prefix QSA KV plus GDN/PLE state persisted outside VRAM. A zero byte budget keeps
    # the lane fully disabled. Version 2 adds a page index to whole-prefix entries.
    kv_disk_cache_dir: str | None = None
    kv_disk_cache_gib: float = 0.0
    # Repeated kind=prefix signatures recognized at the start of leading system or
    # developer content. The CLI replaces these defaults when any entries are supplied.
    kv_harness_prefixes: tuple[str, ...] = (
        "opencode=You are OpenCode,",
        "pi=You are a focused coding agent.",
    )
    # Demand-load page-indexed disk QSA KV. Older entries without an index fall back to eager.
    lazy_restore: str = "on"
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None

    def __post_init__(self) -> None:
        if self.kv_ladder not in ("on", "off"):
            raise ValueError(
                f"--kv-ladder must be 'on' or 'off', got {self.kv_ladder!r}"
            )
        if (
            self.kv_ladder == "on"
            and self.kv_ladder_explicit
            and self.moe_cache_auto
            and self.max_running_req != 1
        ):
            raise ValueError(
                "explicit --kv-ladder on requires --max-running-requests 1"
            )
        if self.moe_activation_dtype not in ("auto", "bf16", "nvfp4"):
            raise ValueError(
                "--moe-activation-dtype must be 'auto', 'bf16', or 'nvfp4', got "
                f"{self.moe_activation_dtype!r}"
            )
        from freetoken.spec_decode import (
            validate_mtp_draft_tokens,
            validate_speculative_mtp,
        )

        validate_speculative_mtp(self.speculative_mtp)
        validate_mtp_draft_tokens(self.mtp_draft_tokens)
        if self.kv_cache_dtype not in ("auto", "bf16", "fp8_e4m3"):
            raise ValueError(
                "--kv-cache-dtype must be 'auto', 'bf16', or 'fp8_e4m3', got "
                f"{self.kv_cache_dtype!r}"
            )
        if self.bank_source not in ("auto", "ftw", "index"):
            raise ValueError(
                "--bank-source must be 'auto', 'ftw', or 'index', got "
                f"{self.bank_source!r}"
            )
        if self.moe_bank_hugepages not in ("auto", "on", "off"):
            raise ValueError(
                "--moe-bank-hugepages must be 'auto', 'on', or 'off', got "
                f"{self.moe_bank_hugepages!r}"
            )
        if self.moe_bank_hugepages == "off" and self.moe_bank_hugepages_tmpfs:
            raise ValueError(
                "--moe-bank-hugepages-tmpfs requires "
                "--moe-bank-hugepages auto or on"
            )
        if (
            not math.isfinite(float(self.moe_bank_hugepages_tmpfs_margin_gib))
            or self.moe_bank_hugepages_tmpfs_margin_gib < 0
        ):
            raise ValueError(
                "--moe-bank-hugepages-tmpfs-margin-gib must be a finite "
                "non-negative number"
            )
        if (
            not math.isfinite(float(self.kv_disk_cache_gib))
            or self.kv_disk_cache_gib < 0
        ):
            raise ValueError("--kv-disk-cache-gib must be a finite non-negative number")
        if self.kv_disk_cache_gib > 0 and not self.kv_disk_cache_dir:
            raise ValueError(
                "--kv-disk-cache-dir is required when --kv-disk-cache-gib is positive"
            )
        for entry in self.kv_harness_prefixes:
            if not isinstance(entry, str):
                raise ValueError(
                    "--kv-harness-prefixes entries must be strings using kind=prefix syntax, "
                    f"got {entry!r}"
                )
            kind, separator, prefix = entry.partition("=")
            if not separator or not kind.strip() or not prefix.strip():
                raise ValueError(
                    "--kv-harness-prefixes entries must use non-empty kind=prefix syntax, "
                    f"got {entry!r}"
                )
        if self.lazy_restore not in ("on", "off"):
            raise ValueError(
                f"--lazy-restore must be 'on' or 'off', got {self.lazy_restore!r}"
            )
        if self.ple_backend not in ("pinned", "cached", "disk", "uring", "hmm"):
            raise ValueError(
                "--ple-backend must be 'pinned', 'cached', 'disk', 'uring', or 'hmm', got "
                f"{self.ple_backend!r}"
            )
        if self.ple_prefill_gather not in ("on", "off"):
            raise ValueError(
                "--ple-prefill-gather must be 'on' or 'off', got "
                f"{self.ple_prefill_gather!r}"
            )
        if not math.isfinite(float(self.ple_cache_gib)) or self.ple_cache_gib <= 0:
            raise ValueError("--ple-cache-gib must be a finite positive number")
        if self.ple_uring_staging_mib < 1:
            raise ValueError("--ple-uring-staging-mib must be a positive integer")
        if not 1 <= self.ple_uring_queue_depth <= 4096:
            raise ValueError(
                "--ple-uring-queue-depth must be an integer in [1, 4096]"
            )
        if self.moe_disk_prefill not in ("cpu", "copy"):
            raise ValueError(
                "--moe-disk-prefill must be 'cpu' or 'copy', got "
                f"{self.moe_disk_prefill!r}"
            )
        if self.moe_prefill_coalesce not in ("populate", "on", "off"):
            raise ValueError(
                "--moe-prefill-coalesce must be 'populate', 'on', or 'off', got "
                f"{self.moe_prefill_coalesce!r}"
            )
        if self.moe_prefill_hot_split not in ("on", "off"):
            raise ValueError(
                "--moe-prefill-hot-split must be 'on' or 'off', got "
                f"{self.moe_prefill_hot_split!r}"
            )
        if self.moe_prefill_split_kernel not in ("grouped", "decode"):
            raise ValueError(
                "--moe-prefill-split-kernel must be 'grouped' or 'decode', got "
                f"{self.moe_prefill_split_kernel!r}"
            )
        if self.moe_cpu_prefill_batch not in ("on", "off"):
            raise ValueError(
                "--moe-cpu-prefill-batch must be 'on' or 'off', got "
                f"{self.moe_cpu_prefill_batch!r}"
            )
        if self.moe_disk_decode not in ("cpu", "gpufetch"):
            raise ValueError(
                "--moe-disk-decode must be 'cpu' or 'gpufetch', got "
                f"{self.moe_disk_decode!r}"
            )
        if (
            not math.isfinite(float(self.moe_hot_expert_budget_gib))
            or self.moe_hot_expert_budget_gib < 0
        ):
            raise ValueError(
                "--moe-hot-expert-budget-gib must be a finite non-negative number"
            )
        if (
            not math.isfinite(float(self.moe_pinned_hot_budget_gib))
            or self.moe_pinned_hot_budget_gib < 0
        ):
            raise ValueError(
                "--moe-pinned-hot-budget-gib must be a finite non-negative number"
            )
        if (
            isinstance(self.moe_hot_adapt_halflife_steps, bool)
            or not isinstance(self.moe_hot_adapt_halflife_steps, int)
            or self.moe_hot_adapt_halflife_steps <= 0
        ):
            raise ValueError("--moe-hot-adapt-halflife-steps must be a positive integer")
        interval = self.moe_hot_adapt_interval_steps
        if interval != "auto" and (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval < 0
        ):
            raise ValueError(
                "--moe-hot-adapt-interval-steps must be 'auto' or a non-negative integer"
            )
        if (
            not math.isfinite(float(self.moe_hot_adapt_max_swap_gib))
            or self.moe_hot_adapt_max_swap_gib <= 0
        ):
            raise ValueError("--moe-hot-adapt-max-swap-gib must be a finite positive number")
        if (
            isinstance(self.moe_hot_adapt_boundary_cap_frac, bool)
            or not math.isfinite(float(self.moe_hot_adapt_boundary_cap_frac))
            or not 0 < self.moe_hot_adapt_boundary_cap_frac <= 1
        ):
            raise ValueError(
                "--moe-hot-adapt-boundary-cap-frac must be finite and in (0, 1]"
            )
        if self.moe_hot_plan_persist not in ("auto", "on", "off"):
            raise ValueError(
                "--moe-hot-plan-persist must be 'auto', 'on', or 'off', got "
                f"{self.moe_hot_plan_persist!r}"
            )
        if (
            isinstance(self.moe_hot_plan_interval_minutes, bool)
            or not math.isfinite(float(self.moe_hot_plan_interval_minutes))
            or self.moe_hot_plan_interval_minutes <= 0
        ):
            raise ValueError(
                "--moe-hot-plan-interval-minutes must be a finite positive number"
            )
        if (
            isinstance(self.moe_hot_adapt_idle_ms, bool)
            or not isinstance(self.moe_hot_adapt_idle_ms, int)
            or self.moe_hot_adapt_idle_ms < 0
        ):
            raise ValueError(
                "--moe-hot-adapt-idle-ms must be a non-negative integer"
            )
        if (
            isinstance(self.moe_hot_adapt_idle_min_interval_ms, bool)
            or not isinstance(self.moe_hot_adapt_idle_min_interval_ms, int)
            or self.moe_hot_adapt_idle_min_interval_ms < 0
        ):
            raise ValueError(
                "--moe-hot-adapt-idle-min-interval-ms must be a non-negative integer"
            )
        if self.moe_disk_pager not in ("madvise", "uffd"):
            raise ValueError(
                "--moe-disk-pager must be 'madvise' or 'uffd', got "
                f"{self.moe_disk_pager!r}"
            )
        if self.moe_disk_lookahead not in ("on", "off"):
            raise ValueError(
                "--moe-disk-lookahead must be 'on' or 'off', got "
                f"{self.moe_disk_lookahead!r}"
            )
        if self.session_expert_prefetch not in ("on", "off"):
            raise ValueError(
                "--session-expert-prefetch must be 'on' or 'off', got "
                f"{self.session_expert_prefetch!r}"
            )
        if (
            isinstance(self.session_protect_experts, bool)
            or not isinstance(self.session_protect_experts, int)
            or self.session_protect_experts < 0
        ):
            raise ValueError("--session-protect-experts must be a non-negative integer")
        if self.host_cache_reserve_gib is not None and (
            not math.isfinite(float(self.host_cache_reserve_gib))
            or self.host_cache_reserve_gib < 0
        ):
            raise ValueError(
                "--host-cache-reserve-gib must be a finite non-negative number"
            )
        if self.moe_pager_budget_gib is not None and (
            not math.isfinite(float(self.moe_pager_budget_gib))
            or self.moe_pager_budget_gib <= 0
        ):
            raise ValueError("--moe-pager-budget-gib must be a finite positive number")

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        return parse_config(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
