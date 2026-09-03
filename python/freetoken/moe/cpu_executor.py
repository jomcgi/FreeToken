"""Python wrapper around the ``_cpu_moe`` C++ executor (--moe-backend cpu).

Owns the persistent CPU worker pool, the per-batch-size pinned IO buffers, and
the per-(layer, batch-size) host-func task descriptors. ``decode`` issues the
whole CUDA-graph-capturable sequence on the current stream:

    D2H (hidden, topk_ids, topk_weights -> pinned)
      -> submit host node (cudaLaunchHostFunc: enqueue MoE task to the pool)
      -> sync host node   (cudaLaunchHostFunc: spin until the pool drains)
      -> H2D (pinned expert output -> GPU)

Buffers and tasks are allocated lazily per batch size. GraphRunner runs an eager
``model.forward()`` at each batch size immediately before capturing it, so the
first (eager) call materializes the stable pinned buffers + task pointers that
the subsequent capture embeds in its host/memcpy nodes.
"""

from __future__ import annotations

import mmap
import os
import platform
import threading
import time
import weakref
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Flag-based GPU<->CPU handshake for hybrid/cpu decode. The default host-func path
# (cudaLaunchHostFunc submit+sync per layer) pays ~30-50us of callback dispatch latency
# per call with the GPU stream idle -- 2 calls per MoE layer per decode step (~6 ms/step
# on a 75-layer model). Instead the GPU raises a mapped-pinned "ready" flag at submit; a
# persistent CPU coordinator (in _cpu_moe) polls it, runs the layer, and sets a "done"
# flag the GPU waits on at sync -- no host-func round-trip. Both GPU-side operations are
# STREAM MEMORY OPERATIONS (cuStreamWriteValue64 / cuStreamWaitValue64, resolved from the
# driver at runtime): they execute on the GPU front end with no SM-resident kernel, so
# GPU utilization stays truthful during the CPU compute window. (The first cut used a
# spin-wait kernel; that pinned reported utilization at 99% and laptop CPU/GPU dynamic
# power schedulers responded by clamping the CPU frequency -- a net decode regression on
# power-coupled edge devices.) Each (layer, decode batch size) pair gets its own flag
# slot, so every captured decode graph rides the handshake. Where memops are unavailable
# (Windows WDDM, vGPU, old drivers -- functionally probed at startup) or the slot
# capacity is exceeded, decode keeps the host-func path (functional, slower). A Python
# watchdog thread turns a wedged coordinator into a loud RuntimeError (via err[] +
# raise_if_unhealthy) instead of an indefinite stream stall.
# Caveat: the coordinator busy-polls one core while decode traffic flows (idle backoff
# otherwise); FREETOKEN_CPU_MOE_FLAG_SYNC=0 opts out entirely.
_FLAG_SYNC = os.getenv("FREETOKEN_CPU_MOE_FLAG_SYNC", "1") != "0"
# Flag slots per MoE layer: covers this many distinct decode batch sizes (captured graph
# sizes plus any eager padded sizes); more than that is unheard of, and the overflow
# just keeps the host-func path for the extra combos.
_FLAG_SLOTS_PER_LAYER = 16

# MoeTask::num_tokens and CpuMoeExecutor::create_task use a signed C++ int.
CPU_MOE_MAX_TASK_TOKENS = (1 << 31) - 1
_PREFILL_POPULATE_SCRATCH_BYTES = 32 << 20
_PREFILL_GEMM_WEIGHT_ROWS = 32


@dataclass(frozen=True)
class CpuTopology:
    architecture: str
    logical_cpus: tuple[int, ...]
    physical_cores: tuple[tuple[int, int, int], ...]
    socket_count: int
    threads_per_core: int
    ht_capable: bool
    total_physical_cores: int = 0


@dataclass(frozen=True)
class ExecutorModeDecision:
    mode: str
    reason: str
    single_socket_x86: bool
    cpu_count_le_32: bool
    cpus_free_ge_2: bool
    cpus_free: int


def parse_cpu_list(value: str) -> list[int]:
    """Parse Linux CPU-list syntax such as ``0-3,8-11``."""
    cpus: set[int] = set()
    for raw_part in value.strip().split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            cpus.add(int(part))
            continue
        first_text, last_text = part.split("-", 1)
        first, last = int(first_text), int(last_text)
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU range {part!r}")
        cpus.update(range(first, last + 1))
    return sorted(cpus)


def _cpuinfo_fields(cpuinfo_text: str) -> tuple[set[str], set[int]]:
    flags: set[str] = set()
    sockets: set[int] = set()
    for line in cpuinfo_text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        if key in ("flags", "features"):
            flags.update(value.strip().lower().split())
        elif key == "physical id":
            try:
                sockets.add(int(value.strip()))
            except ValueError:
                pass
    return flags, sockets


def read_cpu_topology(
    sys_cpu_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
    cpuinfo_path: str | os.PathLike[str] = "/proc/cpuinfo",
    *,
    machine: str | None = None,
    allowed_cpus: set[int] | None = None,
) -> CpuTopology:
    """Read the process-visible Linux socket, core, and SMT topology."""
    if allowed_cpus is None:
        try:
            allowed_cpus = set(os.sched_getaffinity(0))
        except AttributeError:
            allowed_cpus = set(range(os.cpu_count() or 1))

    try:
        cpuinfo_text = Path(cpuinfo_path).read_text(encoding="utf-8")
    except OSError:
        cpuinfo_text = ""
    flags, cpuinfo_sockets = _cpuinfo_fields(cpuinfo_text)

    rows: list[tuple[int, int, int, tuple[int, ...]]] = []
    system_core_keys: set[tuple[int, int]] = set()
    system_sockets: set[int] = set()
    root = Path(sys_cpu_root)
    for cpu_dir in root.glob("cpu[0-9]*"):
        suffix = cpu_dir.name[3:]
        if not suffix.isdigit():
            continue
        cpu = int(suffix)
        topology_dir = cpu_dir / "topology"
        try:
            core_id = int((topology_dir / "core_id").read_text().strip())
            package_id = int(
                (topology_dir / "physical_package_id").read_text().strip()
            )
        except (OSError, ValueError):
            continue
        system_core_keys.add((package_id, core_id))
        system_sockets.add(package_id)
        if cpu not in allowed_cpus:
            continue
        try:
            siblings = tuple(
                parse_cpu_list(
                    (topology_dir / "thread_siblings_list").read_text()
                )
            )
        except (OSError, ValueError):
            continue
        rows.append((package_id, core_id, cpu, siblings or (cpu,)))

    rows.sort(key=lambda row: row[2])
    logical_cpus = tuple(row[2] for row in rows)
    core_reps: dict[tuple[int, int], int] = {}
    threads_per_core = 1
    for package_id, core_id, cpu, siblings in rows:
        core_reps.setdefault((package_id, core_id), cpu)
        threads_per_core = max(threads_per_core, len(siblings))
    physical_cores = tuple(
        (package_id, core_id, cpu)
        for (package_id, core_id), cpu in core_reps.items()
    )
    socket_count = len(system_sockets or cpuinfo_sockets)
    architecture = (machine or platform.machine()).lower()
    return CpuTopology(
        architecture=architecture,
        logical_cpus=logical_cpus,
        physical_cores=physical_cores,
        socket_count=socket_count,
        threads_per_core=threads_per_core,
        ht_capable="ht" in flags or "smt" in flags or threads_per_core > 1,
        total_physical_cores=len(system_core_keys),
    )


def decide_cpu_executor_mode(
    topology: CpuTopology, executor_threads: int, *, reserved_cpus: int = 0,
) -> ExecutorModeDecision:
    """Resolve auto mode from topology and the CPUs left outside the executor."""
    is_x86 = topology.architecture in ("x86_64", "amd64")
    single_socket_x86 = is_x86 and topology.socket_count == 1
    physical_count = topology.total_physical_cores or len(topology.physical_cores)
    cpu_count_le_32 = 0 < physical_count <= 32
    cpus_free = max(
        0, len(topology.logical_cpus) - int(executor_threads) - int(reserved_cpus)
    )
    cpus_free_ge_2 = cpus_free >= 2
    if not topology.logical_cpus or not topology.physical_cores:
        reason = "CPU topology unavailable"
    elif not is_x86:
        reason = "non-x86 architecture detected"
    elif topology.socket_count != 1:
        reason = "multi-socket system detected"
    elif not cpu_count_le_32:
        reason = "more than 32 physical cores detected"
    elif not cpus_free_ge_2:
        reason = "fewer than 2 CPUs free"
    else:
        reason = "auto-detected suitable CPU topology"
    mode = (
        "spin"
        if single_socket_x86 and cpu_count_le_32 and cpus_free_ge_2
        else "sleep"
    )
    return ExecutorModeDecision(
        mode=mode,
        reason=reason,
        single_socket_x86=single_socket_x86,
        cpu_count_le_32=cpu_count_le_32,
        cpus_free_ge_2=cpus_free_ge_2,
        cpus_free=cpus_free,
    )


def _spin_core_cpus(topology: CpuTopology) -> list[int]:
    """Return one process-visible logical CPU for each physical core."""
    return [cpu for _package_id, _core_id, cpu in topology.physical_cores]


@dataclass(frozen=True)
class _StepTimingEvents:
    layer_start: torch.cuda.Event
    d2h_start: torch.cuda.Event
    overlap_start: torch.cuda.Event
    hot_done: torch.cuda.Event
    wait_done: torch.cuda.Event
    layer_end: torch.cuda.Event


@dataclass(frozen=True)
class _PrefillCoalesceLease:
    layer_id: int
    experts: tuple[int, ...]


@dataclass
class _PrefillPopulateOverlap:
    """One predicted next-chunk populate owned by its executor."""

    layer_id: int
    experts: tuple[int, ...]
    cancel: threading.Event
    thread: threading.Thread | None = None
    started_ns: int = 0
    finished_ns: int = 0
    joining: bool = False


def _split_step_timing_layers(
    layer_ids: set[int] | frozenset[int], num_layers: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split selected CPU layers at their largest interior gap.

    Head+tail DISK placement therefore maps to two phases. A single contiguous
    run maps to the edge it touches; a non-edge run maps to the nearer edge.
    """
    ordered = tuple(sorted(int(layer_id) for layer_id in layer_ids))
    if not ordered:
        return (), ()
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    if gaps and max(gaps) > 1:
        split = gaps.index(max(gaps)) + 1
        return ordered[:split], ordered[split:]
    if ordered[0] == 0:
        return ordered, ()
    if ordered[-1] == num_layers - 1:
        return (), ordered
    midpoint = (num_layers - 1) / 2
    return (ordered, ()) if sum(ordered) / len(ordered) <= midpoint else ((), ordered)


def validate_cpu_moe_task_tokens(num_tokens: int, *, source: str) -> int:
    """Validate a batch size before pybind narrows it to C++ ``int``."""
    num_tokens = int(num_tokens)
    if not 1 <= num_tokens <= CPU_MOE_MAX_TASK_TOKENS:
        raise ValueError(
            f"{source} is {num_tokens}, outside the native CPU MoE task token-field "
            f"range [1, {CPU_MOE_MAX_TASK_TOKENS}]"
        )
    return num_tokens

# Activation ids must match ActKind in csrc/cpu_moe/cpu_moe_ext.cpp. Id 3 is the
# clamped (up + 1) swiglu: "swigluoai" runs it in the generic GEMV epilogue,
# "gpt_oss_swiglu" is the same math fused inside the mxfp4 kernel.
_ACT_IDS = {
    "silu": 0,
    "swish": 0,
    "gelu": 1,
    "gelu_tanh": 2,
    "gelu_pytorch_tanh": 2,
    "gpt_oss_swiglu": 3,
    "swigluoai": 3,
    "clamped_silu": 4,
}

# Weight-format ids must match WFmt in csrc/cpu_moe/cpu_moe_ext.cpp.
_WFMT_IDS = {"bf16": 0, "nvfp4": 1, "mxfp4_triton": 2, "ds_fp4": 3, "q4_0": 4}


def _process_faults() -> tuple[int | None, int | None]:
    """Linux process minor/major fault events from one procfs read."""
    try:
        with open("/proc/self/stat", encoding="utf-8") as f:
            tail = f.read().rpartition(") ")[2].split()
        # minflt is field 10 and majflt is field 12; tail[0] is field 3 (state).
        return int(tail[7]), int(tail[9])
    except (OSError, ValueError, IndexError):
        return None, None


def _dedupe_decode_routes(expert_ids, num_experts: int) -> tuple[list[int], int]:
    """Return the sorted valid expert union and valid route-pair count.

    Decode routing has already arrived in the pinned D2H buffer when the native
    pre-run callback invokes this helper. Keeping the full route list in that
    existing buffer avoids another transfer; this CPU-side pass supplies the
    deduped list consumed by every bank's WILLNEED sweep.
    """
    if isinstance(expert_ids, torch.Tensor):
        expert_ids = expert_ids.detach().cpu().reshape(-1).tolist()
    valid = [int(i) for i in expert_ids if 0 <= int(i) < num_experts]
    return sorted(set(valid)), len(valid)


def _group_prefill_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> dict[int, list[tuple[int, float]]]:
    """Reference grouping used by GPU-free tests and diagnostics.

    The native hot path mirrors this stable token-major grouping without creating
    Python objects. Invalid route ids are skipped, matching the serial executor.
    """
    ids = topk_ids.detach().cpu().reshape(topk_ids.shape[0], -1)
    weights = topk_weights.detach().cpu().reshape_as(ids)
    groups: dict[int, list[tuple[int, float]]] = {}
    for token in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            if 0 <= expert < num_experts:
                groups.setdefault(expert, []).append(
                    (token, float(weights[token, slot]))
                )
    return groups


def _prefill_batch_buffer_nbytes(
    max_tokens: int, top_k: int, hidden_size: int, intermediate_size: int,
) -> int:
    """Bytes in the native route-bounded NVFP4 prefill workspace."""
    rows = int(max_tokens) * int(top_k)
    return rows * (
        4 * int(hidden_size)
        + 3 * int(intermediate_size)
        + int(hidden_size) // 4
        + int(intermediate_size) // 4
        + 8 * _PREFILL_GEMM_WEIGHT_ROWS
        + 8
    )


def _disk_lookahead_allowed(requested: bool, pagers: set) -> bool:
    """Return whether previous-step WILLNEED prediction applies to these banks."""
    return bool(requested and not pagers)


def _prefill_coalesce_limit(
    banks, num_experts: int, byte_ceiling: int, *, page_size: int = mmap.PAGESIZE,
) -> tuple[int, int]:
    """Return the bounded expert count and conservative page bytes per expert."""
    if num_experts <= 0 or byte_ceiling <= 0 or page_size <= 0:
        return 0, 0
    per_expert = 0
    for bank in banks:
        tensor = getattr(bank, "tensor", None)
        shape = getattr(tensor, "shape", ())
        rows = int(shape[0]) if shape else num_experts
        nbytes = int(getattr(bank, "nbytes", 0))
        if rows <= 0 or nbytes <= 0 or nbytes % rows:
            continue
        row_bytes = nbytes // rows
        view_offset = int(getattr(bank, "_view_offset", 0))
        max_pages = max(
            (
                (view_offset + (row + 1) * row_bytes + page_size - 1)
                // page_size
                - (view_offset + row * row_bytes) // page_size
            )
            for row in range(rows)
        )
        per_expert += max_pages * page_size
    if per_expert <= 0:
        return num_experts, 0
    return min(num_experts, byte_ceiling // per_expert), per_expert


def compiled_extension_supports(activation: str) -> bool:
    """Whether the compiled ``_cpu_moe`` extension can serve ``activation``
    through its generic epilogue. A stale prebuilt .so accepts newer act ids
    while silently computing the wrong math; the executor hard-errors on that,
    but the engine's auto offload->hybrid upgrade consults this first so a
    default boot degrades to offload instead of crashing after weight load."""
    if activation not in _ACT_IDS:
        return False
    if _ACT_IDS[activation] < 3:
        return True
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        return False
    return _ACT_IDS[activation] <= getattr(_cpu_moe, "max_generic_act_id", lambda: 2)()


def physical_core_cpus() -> list[int]:
    """One logical CPU per physical core, restricted to this process's affinity.

    MoE decode is memory-bandwidth-bound, so SMT siblings only contend for the
    same core's load ports without adding bandwidth. Picking one logical CPU per
    physical core (and pinning to it) gives the best, most stable bandwidth.
    Falls back to the full affinity set when sysfs topology is unavailable.
    """
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        allowed = list(range(os.cpu_count() or 1))
    reps: list[int] = []
    seen: set[str] = set()
    for cpu in allowed:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                key = f.read().strip()
        except OSError:
            reps.append(cpu)
            continue
        if key not in seen:
            seen.add(key)
            reps.append(cpu)
    return reps or allowed or [0]


def resolve_threads_and_affinity(requested: int) -> tuple[int, list[int]]:
    """Return (num_threads, core_ids) for the worker pool.

    ``requested == 0`` -> one thread per physical core, pinned to it (best for the
    bandwidth-bound GEMV: SMT siblings only contend for a core's load ports and the
    spin-barrier degrades badly when oversubscribed). An explicit count is honored,
    spreading first across physical cores, then across the remaining logical CPUs
    (so distinct hardware threads are used before any core is doubled up).
    """
    reps = physical_core_cpus()
    if requested and requested > 0:
        n = int(requested)
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except AttributeError:
            allowed = list(range(os.cpu_count() or 1))
        # physical-core reps first, then the rest of the logical CPUs.
        order = reps + [c for c in allowed if c not in set(reps)]
        if not order:
            order = [0]
        core_ids = [order[i % len(order)] for i in range(n)]
        return n, core_ids
    return len(reps), list(reps)


class CpuMoeExecutor:
    """Decode-time CPU expert compute over an ``OffloadMoeCache``'s host banks
    (bf16, nvfp4, mxfp4_triton, ds_fp4 or q4_0 — see ``_WFMT_IDS`` / ``_resolve_banks``)."""

    def __init__(
        self,
        cache,
        *,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        num_threads: int,
        max_tokens: int,
        device: torch.device,
        executor_mode: str = "sleep",
        spin_wait_us: int = 2000,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float | None = None,
        disk_lookahead: bool = True,
        step_timing: bool = False,
        prefill_coalesce: str | bool = "populate",
        prefill_coalesce_budget_bytes: int = 40 << 30,
        prefill_batch: str | bool = "on",
        max_prefill_tokens: int | None = None,
    ) -> None:
        if executor_mode not in ("sleep", "spin", "auto"):
            raise ValueError(
                "executor_mode must be 'sleep', 'spin', or 'auto', got "
                f"{executor_mode!r}"
            )
        if int(spin_wait_us) < 0:
            raise ValueError("spin_wait_us must be non-negative")
        self._requested_executor_mode = executor_mode

        from freetoken.kernel import _cpu_moe

        fmt = cache.quant_format
        if fmt not in _WFMT_IDS:
            raise NotImplementedError(
                f"--moe-backend cpu/hybrid computes experts on the CPU and supports "
                f"{sorted(_WFMT_IDS)} formats, but this checkpoint's experts are "
                f"{fmt!r}; use --moe-backend offload (GPU-side dequant) instead."
            )
        if activation not in _ACT_IDS:
            raise NotImplementedError(f"CPU MoE backend: unsupported activation {activation!r}")
        # ABI probe: a stale prebuilt _cpu_moe.so accepts newer act ids without
        # error and silently computes the wrong activation in the generic
        # epilogue -- fail loudly with the rebuild instruction instead. (mxfp4
        # handles its act inside the kernel and predates the marker.)
        if _ACT_IDS[activation] >= 3 and fmt != "mxfp4_triton":
            supported = getattr(_cpu_moe, "max_generic_act_id", lambda: 2)()
            if _ACT_IDS[activation] > supported:
                raise RuntimeError(
                    f"the compiled _cpu_moe extension predates activation "
                    f"{activation!r} (max generic act id {supported}); rebuild it "
                    "with `python setup.py build_ext --inplace` (or reinstall the "
                    "wheel) before serving this model on the cpu/hybrid backend."
                )

        self.num_layers = int(cache.num_layers)
        self.num_experts = int(cache.num_experts)
        self.top_k = int(top_k)
        self.quant_format = fmt
        self.device = device
        self.max_tokens = int(max_tokens)
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self._step_timing = bool(step_timing)
        timing_methods = ("task_last_run_ns", "step_timing_snapshot_and_reset")
        if self._step_timing and not all(
            hasattr(_cpu_moe.CpuMoeExecutor, name) for name in timing_methods
        ):
            raise RuntimeError(
                "the CPU MoE extension needs rebuilding for --moe-step-timing; run "
                "`python setup.py build_ext --inplace` or reinstall the wheel"
            )
        # The per-layer tensors and their pointer tables must outlive the executor
        # (C++ holds raw addresses into both).
        self._banks: list[torch.Tensor] = []
        self._disk_banks: dict[int, list] = {}
        self._disk_pagers: set = set()
        self._disk_prefetch_calls = [0] * self.num_layers
        self._disk_prefetch_pages = [0] * self.num_layers
        self._disk_decode_steps = 0
        self._disk_route_pairs = 0
        self._disk_distinct_experts = 0
        self._disk_lookahead_hits = 0
        self._disk_lookahead_routes = 0
        self._disk_delta_pages = 0
        self._disk_minor_fault_base, self._disk_major_fault_base = _process_faults()
        self._disk_prefetch_error: BaseException | None = None
        ptrs, (self.H, self.I) = self._resolve_banks(cache.bank_sources, fmt)
        if self._step_timing:
            ptrs["task_timing_enabled"] = True
        if isinstance(prefill_coalesce, bool):
            prefill_coalesce = "on" if prefill_coalesce else "off"
        if prefill_coalesce not in ("populate", "on", "off"):
            raise ValueError(
                "prefill_coalesce must be 'populate', 'on', or 'off', got "
                f"{prefill_coalesce!r}"
            )
        self._prefill_coalesce_mode = prefill_coalesce
        self._prefill_coalesce_enabled = prefill_coalesce != "off"
        self._prefill_coalesce_experts = 0
        self._prefill_coalesce_ns = 0
        self._prefill_coalesce_degrades = 0
        self._prefill_coalesce_warned = False
        self._prefill_populate_bytes = 0
        self._prefill_populate_skipped_tmpfs_bytes = 0
        self._prefill_populate_ns = 0
        self._prefill_populate_overlap_ns = 0
        self._prefill_release_pages = 0
        self._prefill_release_skipped_tmpfs_bytes = 0
        self._prefill_populate_scratch = None
        self._prefill_populate_overlap: _PrefillPopulateOverlap | None = None
        self._prefill_populate_overlap_lock = threading.Lock()
        if isinstance(prefill_batch, bool):
            prefill_batch = "on" if prefill_batch else "off"
        if prefill_batch not in ("on", "off"):
            raise ValueError(
                f"prefill_batch must be 'on' or 'off', got {prefill_batch!r}"
            )
        self._prefill_batch_requested = prefill_batch == "on"
        self._prefill_batch_enabled = False
        self._prefill_batch_warned = False
        self._prefill_batch_rows = 0
        self._prefill_batch_gemms = 0
        self._prefill_batch_degrades = 0
        self._prefill_batch_capacity = int(
            max_prefill_tokens if max_prefill_tokens is not None else max_tokens
        )
        self._prefill_batch_buffer_bytes = 0
        # Keep half of the pager budget available to the decode working set. Only one
        # prefill layer is live at a time and its lease is released after compute.
        self._prefill_coalesce_byte_ceiling = max(
            0, int(prefill_coalesce_budget_bytes) // 2
        )
        self._prefill_populate_scratch_bytes = min(
            _PREFILL_POPULATE_SCRATCH_BYTES,
            self._prefill_coalesce_byte_ceiling,
        )
        self._prefill_coalesce_limits = {
            layer_id: _prefill_coalesce_limit(
                banks,
                self.num_experts,
                self._prefill_coalesce_byte_ceiling,
            )[0]
            for layer_id, banks in self._disk_banks.items()
        }
        # UFFD already prefetches complete logical rows through its userspace pager.
        # Keep one-step prediction on the madvise path only.
        self._disk_lookahead_enabled = _disk_lookahead_allowed(
            disk_lookahead, self._disk_pagers
        )
        self._disk_previous_experts: dict[int, tuple[int, ...]] = {}
        self._disk_predicted_experts: dict[int, tuple[int, ...]] = {}
        for layer_id, banks in self._disk_banks.items():
            pagers = {getattr(bank, "_pager", None) for bank in banks}
            for pager in pagers - {None}:
                pager_banks = [bank for bank in banks if bank._pager is pager]
                pager.validate_working_set(
                    pager_banks,
                    min(self.num_experts, self.max_tokens * self.top_k),
                    context=f"layer {layer_id} decode",
                )

        # Decide the flag handshake up front (env + device + a functional stream-memop
        # probe): its coordinator needs a core of its own, which the auto thread sizing
        # below reserves (a coordinator time-slicing against the GEMV workers measurably
        # destabilizes throughput on fully-subscribed boxes).
        self._flag_sync = _FLAG_SYNC and device.type == "cuda"
        self._cpu_moe = _cpu_moe  # module ref for the decode-path memop calls
        if self._flag_sync:
            probe_scratch = alloc_pinned_tensor(1, dtype=torch.int64)
            probe_scratch.zero_()
            if not _cpu_moe.memops_probe(
                torch.cuda.current_stream().cuda_stream, probe_scratch.data_ptr()
            ):
                logger.info_rank0(
                    "cpu-moe flag handshake unavailable: CUDA stream memory operations "
                    "are not supported here (Windows WDDM / vGPU / old driver); using "
                    "the cudaLaunchHostFunc sync"
                )
                self._flag_sync = False

        nthreads, core_ids = resolve_threads_and_affinity(num_threads)
        coord_core = -1
        if self._flag_sync and num_threads == 0 and nthreads > 2:
            # Auto sizing: give the coordinator the last physical core instead of
            # oversubscribing (workers drop from N to N-1).
            coord_core = core_ids[-1]
            nthreads -= 1
            core_ids = core_ids[:-1]
        topology = None
        effective_mode = executor_mode
        if executor_mode != "sleep":
            topology = read_cpu_topology()
        if executor_mode == "auto":
            decision = decide_cpu_executor_mode(
                topology,
                nthreads,
                reserved_cpus=1 if coord_core >= 0 else 0,
            )
            effective_mode = decision.mode
            checks = (
                f"single-socket x86 "
                f"({'yes' if decision.single_socket_x86 else 'no'}), "
                f"CPU count <=32 "
                f"({'yes' if decision.cpu_count_le_32 else 'no'}), "
                f"CPUs free >=2 "
                f"({'yes' if decision.cpus_free_ge_2 else 'no'})"
            )
            if effective_mode == "spin":
                logger.info_rank0(
                    "MOE executor: spin mode enabled "
                    f"(auto-detected single-socket x86, "
                    f"{topology.total_physical_cores} cores, "
                    f"{topology.threads_per_core} threads per core, "
                    f"{decision.cpus_free} CPUs free)"
                )
            else:
                logger.info_rank0(f"MOE executor: sleep mode ({decision.reason})")
            logger.info_rank0(f"MOE executor auto checks: {checks}")
        elif executor_mode == "spin":
            logger.info_rank0("MOE executor: spin mode enabled (explicit)")

        if effective_mode == "spin" and topology is not None:
            physical_cpus = [
                cpu for cpu in _spin_core_cpus(topology) if cpu != coord_core
            ]
            if physical_cpus:
                core_ids = [
                    physical_cpus[index % len(physical_cpus)]
                    for index in range(nthreads)
                ]
        self._executor_mode = effective_mode
        self._report_spin_fallbacks = executor_mode == "spin"
        if effective_mode == "spin":
            logger.info_rank0(
                "Core pinning: "
                + ", ".join(
                    f"thread {thread_id} -> core {cpu}"
                    for thread_id, cpu in enumerate(core_ids)
                )
            )
        self._coord_core = coord_core
        self._ext = _cpu_moe.CpuMoeExecutor(
            num_threads=nthreads,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.H,
            inter_size=self.I,
            max_tokens=self.max_tokens,
            activation_id=_ACT_IDS[activation],
            apply_router_weight_on_input=1 if apply_router_weight_on_input else 0,
            weight_format=_WFMT_IDS[fmt],
            swiglu_alpha=float(swiglu_alpha),
            swiglu_limit=float(swiglu_limit) if swiglu_limit is not None else float("inf"),
            core_ids=core_ids,
            spin_mode=effective_mode == "spin",
            spin_wait_us=int(spin_wait_us),
            **ptrs,
        )
        self._configure_prefill_batch()
        if self._disk_banks:
            self._disk_callback = partial(_disk_prefetch_callback, weakref.ref(self))
            self._ext.set_pre_run_callback(self._disk_callback)
        self.num_threads = nthreads
        self.core_ids = core_ids
        self.isa = self._ext.isa_name()

        spare = len(physical_core_cpus()) - nthreads - (1 if coord_core >= 0 else 0) - 1
        clamp = max(1, min(torch.get_num_threads(), spare))
        if clamp < torch.get_num_threads():
            logger.info_rank0(
                f"torch intra-op threads: {torch.get_num_threads()} -> {clamp} "
                "(cores reserved for the pinned CPU MoE pool)"
            )
            torch.set_num_threads(clamp)

        self._io: dict[int, dict[str, torch.Tensor]] = {}
        self._tasks: dict[tuple[int, int], int] = {}
        self._step_timing_events: dict[tuple[int, int], _StepTimingEvents] = {}
        self._step_timing_hot_keys: set[tuple[int, int]] = set()
        self._gpufetch_tasks: dict[int, tuple[int, int | None]] = {}
        self._prefill_io: dict[str, torch.Tensor] | None = None
        self._prefill_capacity = 0

        # Flag-based handshake: mapped-pinned ready/done/err int64 arrays (one slot per
        # (MoE layer, decode batch size) pair, allocated as tasks are created) + a
        # persistent CPU coordinator that polls ready[], runs the slot's task on the
        # pool, and sets done[]. Binary per-step protocol (GPU memops: done=0, ready=1;
        # coordinator: consume ready, run, done=1; GPU waits done>=1 -- the WAIT
        # immediate is constant, so CUDA-graph replays are safe). err[] is raised by the
        # WATCHDOG thread when a ready flag stays unanswered (dead coordinator): it
        # poisons done to unblock the stream and raise_if_unhealthy() turns the step
        # into a loud error instead of silent stale output. Buffers are kept alive on
        # self so the coordinator's pinned pointers stay valid for the executor's
        # lifetime (flag_sync itself was decided above, before thread sizing).
        self._ready = self._done = self._err = None
        self._flag_slots: dict[tuple, int] = {}
        # CPU tasks use up to _FLAG_SLOTS_PER_LAYER batch-size variants. Reserve one
        # additional stable slot per layer for DISK GPU-fetch staging.
        self._flag_capacity = self.num_layers * (_FLAG_SLOTS_PER_LAYER + 1)
        if self._flag_sync:
            self._ready = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._done = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._err = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._ready.zero_()
            self._done.zero_()
            self._err.zero_()
            self._ext.start_flag_coordinator(
                self._ready.data_ptr(), self._done.data_ptr(), self._flag_capacity,
                self._coord_core,
            )
            self._watchdog_stop = False
            # The thread target holds a WEAKREF and re-derefs it each tick: a bound
            # method would strong-reference the executor forever (the loop never ends
            # on its own), pinning the C++ worker pool and the pinned banks against GC
            # in build-many-engines scenarios. NB: the stop flag / weakref death is
            # observed only between 2 s sleeps, so teardown of the THREAD can lag up to
            # ~2 s -- it is a daemon, so neither GC of the executor (weakref breaks the
            # cycle) nor process exit waits on it.
            self._watchdog = threading.Thread(
                target=_watchdog_main,
                args=(weakref.ref(self),),
                name="cpu-moe-flag-watchdog",
                daemon=True,
            )
            self._watchdog.start()

        # ds_fp4: the reference FP8 activation round-trip is a scalar per-element chain
        # that the C++ side runs single-threaded on the CUDA host-callback thread --
        # straight on the decode critical path (~0.3ms/layer at H=4096, every worker and
        # the GPU waiting on it). When a GPU is present we run the numerically identical
        # round-trip as a captured GPU kernel BEFORE the D2H (see decode_submit) and tell
        # the C++ side to skip its own. Measured on DeepSeek-V4-Flash bs=1 decode:
        # 12.85 -> 15.65 tok/s, output bit-identical (tests/moe/test_dsfp4_prequant.py).
        self._gpu_prequant = fmt == "ds_fp4" and device.type == "cuda"
        if self._gpu_prequant:
            self._ext.set_input_prequant(True)
            logger.info_rank0(
                "ds_fp4: input FP8 round-trip moved to the GPU "
                "(bit-identical grid; the CPU-side scalar round-trip is skipped)"
            )

        logger.info_rank0(
            f"CPU MoE executor ready: threads={nthreads} (pinned to cores "
            f"{core_ids[0]}..{core_ids[-1]}) isa={self.isa} fmt={fmt} "
            f"H={self.H} I={self.I} experts={self.num_experts} layers={self.num_layers} "
            f"top_k={self.top_k} act={activation} max_tokens={self.max_tokens}"
        )
        logger.info_rank0(
            f"CPU MoE prefill batch: "
            f"{'on' if self._prefill_batch_enabled else 'off'}, "
            f"kernel={getattr(self._ext, 'prefill_batch_kernel_name', lambda: 'unknown')()}, "
            f"capacity={self._prefill_batch_capacity} tokens, "
            f"buffers={self._prefill_batch_buffer_bytes / 2**20:.1f} MiB"
        )
        if self._disk_banks and self._prefill_coalesce_enabled:
            logger.info_rank0(
                f"CPU MoE prefill coalesce: {self._prefill_coalesce_mode}, "
                f"per-layer ceiling={self._prefill_coalesce_byte_ceiling / 2**30:.2f} GiB, "
                f"expert limits={self._prefill_coalesce_limits}, "
                f"populate scratch={self._prefill_populate_scratch_bytes / 2**20:.0f} MiB"
            )

    def _configure_prefill_batch(self) -> None:
        if not self._prefill_batch_requested:
            return
        setup = getattr(self._ext, "setup_prefill_batch", None)
        run_batch = getattr(self._ext, "run_prefill_batch_sync", None)
        try:
            if setup is None or run_batch is None:
                raise RuntimeError("compiled extension lacks the batched entry point")
            if not setup(self._prefill_batch_capacity):
                raise RuntimeError(
                    f"batched NVFP4 kernel unavailable for format "
                    f"{self.quant_format!r}"
                )
            self._prefill_batch_enabled = True
            buffer_bytes = getattr(
                self._ext, "prefill_batch_buffer_bytes", lambda: 0
            )
            self._prefill_batch_buffer_bytes = int(buffer_bytes())
        except Exception as exc:
            self._degrade_prefill_batch("setup degraded to serial", exc)

    def _make_table(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """Build a CPU int64 tensor of per-layer base addresses for one bank.

        ``layers`` is one ``[num_experts, ...]`` tensor per layer (the per-layer host
        bank contract). The C++ side stores this table's pointer and indexes
        ``tbl[layer_id]`` at call time; both the table and the layer tensors are kept
        on ``self._banks`` (GC guard) so the raw pointers stay valid for the
        executor's lifetime.
        """
        assert len(layers) == self.num_layers, (len(layers), self.num_layers)
        for layer_id, tensor in enumerate(layers):
            bank = getattr(tensor, "_freetoken_host_bank", None)
            if bank is not None and bank.residency.value == "disk":
                layer_banks = self._disk_banks.setdefault(layer_id, [])
                if bank not in layer_banks:
                    layer_banks.append(bank)
                if getattr(bank, "_pager", None) is not None:
                    self._disk_pagers.add(bank._pager)
        table = torch.tensor([t.data_ptr() for t in layers], dtype=torch.int64)
        self._banks.append(table)
        self._banks.extend(layers)
        return table

    def _resolve_banks(self, banks: dict, fmt: str) -> tuple[dict, tuple[int, int]]:
        """Return (pointer kwargs for the C++ ctor, (H, I)) for the given format.

        ``banks[name]`` is a list of ``num_layers`` ``[num_experts, ...]`` tensors
        (the per-layer host bank contract); shapes are read from the first layer so
        per-partition (TP) sizes are exact. Unused pointers are 0. Every pointer kwarg
        is actually a per-layer table's address (see ``_make_table``), not a single
        bank's -- the C++ ctor resolves ``tbl[layer_id]`` per task.
        """
        if fmt == "bf16":
            gate_up = banks["gate_up"]
            down = banks["down"]
            if gate_up[0].dtype != torch.bfloat16 or down[0].dtype != torch.bfloat16:
                raise NotImplementedError(
                    f"bf16 CPU MoE requires bf16 banks, got {gate_up[0].dtype}/{down[0].dtype}"
                )
            H = int(gate_up[0].shape[2])
            I = int(gate_up[0].shape[1] // 2)
            assert gate_up[0].shape[1] == 2 * I
            assert tuple(down[0].shape[1:]) == (H, I), (down[0].shape, H, I)
            ptrs = dict(
                gate_up_ptr=self._make_table(gate_up).data_ptr(),
                down_ptr=self._make_table(down).data_ptr(),
                gate_up_scale_ptr=0,
                gate_up_global_ptr=0,
                down_scale_ptr=0,
                down_global_ptr=0,
                gate_up_bias_ptr=0,
                down_bias_ptr=0,
            )
            return ptrs, (H, I)

        if fmt == "q4_0":
            return self._resolve_q4_0_banks(banks)

        if fmt == "mxfp4_triton":
            return self._resolve_mxfp4_banks(banks)

        if fmt == "ds_fp4":
            return self._resolve_dsfp4_banks(banks)

        # nvfp4: packed e2m1 (2/byte) + fp8-e4m3 per-16 block scales + fp16 row globals.
        gup, gus, gug = banks["gate_up_packed"], banks["gate_up_scale"], banks["gate_up_global"]
        dnp, dns, dng = banks["down_packed"], banks["down_scale"], banks["down_global"]
        assert gup[0].dtype == torch.uint8 and dnp[0].dtype == torch.uint8, (gup[0].dtype, dnp[0].dtype)
        assert gus[0].element_size() == 1 and dns[0].element_size() == 1, "block scales must be 1 byte"
        assert gug[0].dtype == torch.float16 and dng[0].dtype == torch.float16, (gug[0].dtype, dng[0].dtype)
        I = int(gup[0].shape[1] // 2)
        H = int(gup[0].shape[2] * 2)
        assert gup[0].shape[1] == 2 * I
        assert H % 16 == 0 and I % 16 == 0, (H, I)
        assert tuple(dnp[0].shape[1:]) == (H, I // 2), (dnp[0].shape, H, I)
        assert tuple(gus[0].shape[1:]) == (2 * I, H // 16), (gus[0].shape, I, H)
        assert tuple(dns[0].shape[1:]) == (H, I // 16), (dns[0].shape, H, I)
        assert tuple(gug[0].shape[1:]) == (2 * I,) and tuple(dng[0].shape[1:]) == (H,)
        ptrs = dict(
            gate_up_ptr=self._make_table(gup).data_ptr(),
            down_ptr=self._make_table(dnp).data_ptr(),
            gate_up_scale_ptr=self._make_table(gus).data_ptr(),
            gate_up_global_ptr=self._make_table(gug).data_ptr(),
            down_scale_ptr=self._make_table(dns).data_ptr(),
            down_global_ptr=self._make_table(dng).data_ptr(),
            gate_up_bias_ptr=0,
            down_bias_ptr=0,
        )
        return ptrs, (H, I)

    def _resolve_q4_0_banks(self, banks: dict) -> tuple[dict, tuple[int, int]]:
        """Native GGUF Q4_0 schema (gemma4 GGUF): per-32 blocks (fp16 scale + 16 nibble
        bytes), row-major over K -- the *same* packed banks the GPU offload path streams.
        gate_up is [S, 2I, H//32*18], down is [S, H, I//32*18]; the C++ W4A16 GEMV reads a
        row in place (18 bytes / 32 K) and dequantizes weights inside the K-loop."""
        gate_up, down = banks["gate_up"], banks["down"]
        assert gate_up[0].dtype == torch.uint8 and down[0].dtype == torch.uint8, (
            gate_up[0].dtype, down[0].dtype,
        )
        I = int(gate_up[0].shape[1] // 2)
        H = int(down[0].shape[1])
        assert gate_up[0].shape[1] == 2 * I
        assert H % 32 == 0 and I % 32 == 0, (H, I)
        assert int(gate_up[0].shape[2]) == (H // 32) * 18, (gate_up[0].shape, H)
        assert int(down[0].shape[2]) == (I // 32) * 18, (down[0].shape, I)
        ptrs = dict(
            gate_up_ptr=self._make_table(gate_up).data_ptr(),
            down_ptr=self._make_table(down).data_ptr(),
            gate_up_scale_ptr=0,
            gate_up_global_ptr=0,
            down_scale_ptr=0,
            down_global_ptr=0,
            gate_up_bias_ptr=0,
            down_bias_ptr=0,
        )
        return ptrs, (H, I)

    def _resolve_mxfp4_banks(self, banks: dict) -> tuple[dict, tuple[int, int]]:
        """gpt-oss mxfp4 ``mxfp4_triton`` schema: transposed split-K blocks/scales
        (N innermost) + per-output-row biases. The C++ kernel streams K and
        accumulates a contiguous N-block, so the GPU-tiled layout is read in place
        (no repack, no extra host memory). Block scales are e8m0 (1 byte / 32 K)."""
        gub, gus, gob = banks["gate_up_blocks"], banks["gate_up_scales"], banks["gate_up_bias"]
        dnb, dns, dob = banks["down_blocks"], banks["down_scales"], banks["down_bias"]
        assert gub[0].dtype == torch.uint8 and dnb[0].dtype == torch.uint8, (gub[0].dtype, dnb[0].dtype)
        assert gus[0].dtype == torch.uint8 and dns[0].dtype == torch.uint8, (gus[0].dtype, dns[0].dtype)
        assert gob[0].dtype == torch.bfloat16 and dob[0].dtype == torch.bfloat16, (gob[0].dtype, dob[0].dtype)
        # gate_up_blocks [E, H//2, 2I]; down_blocks [E, I//2, H]
        H = int(gub[0].shape[1] * 2)
        I = int(gub[0].shape[2] // 2)
        assert gub[0].shape[2] == 2 * I
        assert H % 32 == 0 and I % 32 == 0, (H, I)
        assert tuple(dnb[0].shape[1:]) == (I // 2, H), (dnb[0].shape, H, I)
        assert tuple(gus[0].shape[1:]) == (H // 32, 2 * I), (gus[0].shape, H, I)
        assert tuple(dns[0].shape[1:]) == (I // 32, H), (dns[0].shape, H, I)
        assert tuple(gob[0].shape[1:]) == (2 * I,) and tuple(dob[0].shape[1:]) == (H,)
        ptrs = dict(
            gate_up_ptr=self._make_table(gub).data_ptr(),
            down_ptr=self._make_table(dnb).data_ptr(),
            gate_up_scale_ptr=self._make_table(gus).data_ptr(),
            gate_up_global_ptr=0,
            down_scale_ptr=self._make_table(dns).data_ptr(),
            down_global_ptr=0,
            gate_up_bias_ptr=self._make_table(gob).data_ptr(),
            down_bias_ptr=self._make_table(dob).data_ptr(),
        )
        return ptrs, (H, I)

    def _resolve_dsfp4_banks(self, banks: dict) -> tuple[dict, tuple[int, int]]:
        """DeepSeek-V4 ``ds_fp4`` schema: row-major e2m1 (2/byte) + e8m0 per-32 block
        scales, no global, no bias. Layout matches nvfp4 (K contiguous per output row),
        so the C++ GEMV reads it in place. The kernel additionally FP8-round-trips the
        activations (block 128) to match DSV4's W4A8 reference, hence the %128 dims."""
        gup, gus = banks["gate_up_packed"], banks["gate_up_scale"]
        dnp, dns = banks["down_packed"], banks["down_scale"]
        assert gup[0].dtype == torch.uint8 and dnp[0].dtype == torch.uint8, (gup[0].dtype, dnp[0].dtype)
        assert gus[0].element_size() == 1 and dns[0].element_size() == 1, "block scales must be 1 byte"
        I = int(gup[0].shape[1] // 2)
        H = int(gup[0].shape[2] * 2)
        assert gup[0].shape[1] == 2 * I
        assert H % 128 == 0 and I % 128 == 0, (H, I)  # FP8 activation round-trip block=128
        assert tuple(dnp[0].shape[1:]) == (H, I // 2), (dnp[0].shape, H, I)
        assert tuple(gus[0].shape[1:]) == (2 * I, H // 32), (gus[0].shape, I, H)
        assert tuple(dns[0].shape[1:]) == (H, I // 32), (dns[0].shape, H, I)
        ptrs = dict(
            gate_up_ptr=self._make_table(gup).data_ptr(),
            down_ptr=self._make_table(dnp).data_ptr(),
            gate_up_scale_ptr=self._make_table(gus).data_ptr(),
            gate_up_global_ptr=0,
            down_scale_ptr=self._make_table(dns).data_ptr(),
            down_global_ptr=0,
            gate_up_bias_ptr=0,
            down_bias_ptr=0,
        )
        return ptrs, (H, I)

    def _io_for(self, bs: int) -> dict[str, torch.Tensor]:
        validate_cpu_moe_task_tokens(bs, source="CPU MoE batch size")
        io = self._io.get(bs)
        if io is None:
            alloc = alloc_pinned_tensor if self.device.type == "cuda" else torch.empty
            io = {
                "x": alloc(bs, self.H, dtype=torch.bfloat16),
                "ids": alloc(bs, self.top_k, dtype=torch.int32),
                "w": alloc(bs, self.top_k, dtype=torch.float32),
                "y": alloc(bs, self.H, dtype=torch.bfloat16),
            }
            self._io[bs] = io
        return io

    def _task_for(self, layer_id: int, bs: int) -> int:
        key = (layer_id, bs)
        task = self._tasks.get(key)
        if task is None:
            io = self._io_for(bs)
            task = self._ext.create_task(
                layer_id,
                bs,
                io["x"].data_ptr(),
                io["ids"].data_ptr(),
                io["w"].data_ptr(),
                io["y"].data_ptr(),
            )
            self._tasks[key] = task
            # Allocate this (layer, bs) combo a flag slot and register its task with the
            # coordinator. Combos past the slot capacity keep the host-func path.
            if self._flag_sync and key not in self._flag_slots:
                slot = len(self._flag_slots)
                if slot < self._flag_capacity:
                    self._flag_slots[key] = slot
                    self._ext.register_flag_task(slot, task)
        return task

    def _timing_for(self, layer_id: int, bs: int) -> _StepTimingEvents | None:
        if not self._step_timing:
            return None
        key = (int(layer_id), int(bs))
        timing = self._step_timing_events.get(key)
        if timing is None:
            # external=True makes the record visible as an event node in a CUDA graph,
            # so elapsed_time resolves the current replay rather than capture warmup.
            def event() -> torch.cuda.Event:
                return torch.cuda.Event(enable_timing=True, external=True)

            timing = _StepTimingEvents(
                event(), event(), event(), event(), event(), event()
            )
            self._step_timing_events[key] = timing
        return timing

    def step_timing_breakdown(self, bs: int | None = None) -> dict:
        """Return and reset native per-layer decode timings since the prior call."""
        zero_total = {
            "wake_us": 0.0,
            "compute_us": 0.0,
            "signal_us": 0.0,
            "total_tasks": 0,
            "total_experts": 0,
            "total_bytes": 0,
        }
        if not getattr(self, "_step_timing", False):
            return {
                "per_layer": {},
                "total": zero_total,
                "submit_d2h_us": {"per_layer": {}, "total": 0.0},
            }

        snapshot = self._ext.step_timing_snapshot_and_reset()
        spin_fallbacks = None
        if getattr(self, "_report_spin_fallbacks", False):
            spin_fallbacks = int(self._ext.spin_fallback_count(True))
        per_layer = {}
        d2h_per_layer = {}
        for raw_layer_id, raw in snapshot.items():
            layer_id = int(raw_layer_id)
            d2h_us = 0.0
            if bs is not None:
                timing = getattr(self, "_step_timing_events", {}).get((layer_id, bs))
                if timing is not None:
                    d2h_us = float(
                        timing.d2h_start.elapsed_time(timing.overlap_start) * 1000.0
                    )
            per_layer[layer_id] = {
                "wake_us": float(raw.get("wake_us", 0.0)),
                "compute_us": float(raw.get("compute_us", 0.0)),
                "signal_us": float(raw.get("signal_us", 0.0)),
                "tasks": int(raw.get("tasks", 0)),
                "experts": int(raw.get("experts", 0)),
                "bytes": int(raw.get("bytes", 0)),
            }
            d2h_per_layer[layer_id] = max(0.0, d2h_us)

        total = dict(zero_total)
        for row in per_layer.values():
            total["wake_us"] += row["wake_us"]
            total["compute_us"] += row["compute_us"]
            total["signal_us"] += row["signal_us"]
            total["total_tasks"] += row["tasks"]
            total["total_experts"] += row["experts"]
            total["total_bytes"] += row["bytes"]
        result = {
            "per_layer": per_layer,
            "total": total,
            "submit_d2h_us": {
                "per_layer": d2h_per_layer,
                "total": sum(d2h_per_layer.values()),
            },
        }
        if spin_fallbacks is not None:
            result["spin_fallbacks"] = spin_fallbacks
        return result

    def resolve_step_timing(
        self,
        bs: int,
        step_start: torch.cuda.Event,
        step_end: torch.cuda.Event,
    ) -> dict[str, float]:
        """Resolve one synchronized decode replay into phase and overlap spans."""
        step_us = float(step_start.elapsed_time(step_end) * 1000.0)
        phase_ids = {
            layer_id
            for layer_id in getattr(self, "_disk_banks", ())
            if (layer_id, bs) in self._step_timing_events
        }
        head, tail = _split_step_timing_layers(phase_ids, self.num_layers)
        cpu_head_us = gpu_mid_us = cpu_tail_us = 0.0
        if head and tail:
            head_end = self._step_timing_events[(head[-1], bs)].layer_end
            tail_start = self._step_timing_events[(tail[0], bs)].layer_start
            cpu_head_us = float(step_start.elapsed_time(head_end) * 1000.0)
            gpu_mid_us = float(head_end.elapsed_time(tail_start) * 1000.0)
            cpu_tail_us = float(tail_start.elapsed_time(step_end) * 1000.0)
        elif head:
            cpu_head_us = step_us
        elif tail:
            cpu_tail_us = step_us
        else:
            gpu_mid_us = step_us

        overlap_us = 0.0
        for key in self._step_timing_hot_keys:
            if key[1] != bs:
                continue
            timing = self._step_timing_events[key]
            hot_us = float(timing.overlap_start.elapsed_time(timing.hot_done) * 1000.0)
            branch_us = float(
                timing.overlap_start.elapsed_time(timing.wait_done) * 1000.0
            )
            task = self._tasks[key]
            cpu_us = float(self._ext.task_last_run_ns(task)) / 1000.0
            # wait_done is max(CPU completion, hot completion). If the CPU finished
            # last, branch_us - cpu_us estimates its dispatch delay after the common
            # doorbell marker; otherwise its whole native span overlapped hot work.
            cpu_delay_us = max(0.0, branch_us - cpu_us) if branch_us > hot_us else 0.0
            overlap_us += max(0.0, min(cpu_us, hot_us - cpu_delay_us))
        breakdown = self.step_timing_breakdown(bs)
        native = breakdown["total"]
        cpu_layers = len(breakdown["per_layer"])
        result = {
            "cpu_head_us": max(0.0, cpu_head_us),
            "gpu_mid_us": max(0.0, gpu_mid_us),
            "cpu_tail_us": max(0.0, cpu_tail_us),
            "overlap_us": max(0.0, overlap_us),
            "cpu_wake_us": native["wake_us"] / cpu_layers if cpu_layers else 0.0,
            "cpu_compute_us": (
                native["compute_us"] / cpu_layers if cpu_layers else 0.0
            ),
            "cpu_signal_us": (
                native["signal_us"] / cpu_layers if cpu_layers else 0.0
            ),
            "cpu_layers_per_step": cpu_layers,
            "cpu_expert_bytes_per_step": native["total_bytes"],
        }
        if "spin_fallbacks" in breakdown:
            result["spin_fallbacks"] = breakdown["spin_fallbacks"]
        return result

    def register_gpufetch_layer(
        self,
        layer_id: int,
        *,
        capacity: int,
        num_rows_ptr: int,
        row_ids_ptr: int,
        source_ptrs: list[int],
        staging_ptrs: list[int],
        row_bytes: list[int],
    ) -> None:
        """Register one fixed DISK-to-pinned row-fill task with the CPU coordinator."""
        required = (
            "create_gpufetch_task",
            "gpufetch_with_cuda_stream",
            "register_flag_gpufetch_task",
            "gpufetch_stats",
            "gpufetch_error_code",
        )
        if not all(hasattr(self._ext, name) for name in required):
            raise RuntimeError(
                "the CPU MoE extension needs rebuilding for DISK expert staging "
                "(--moe-disk-decode gpufetch or --moe-cold-fetch-max); run "
                "`python setup.py build_ext --inplace` or reinstall the wheel"
            )
        task = self._ext.create_gpufetch_task(
            layer_id,
            capacity,
            num_rows_ptr,
            row_ids_ptr,
            source_ptrs,
            staging_ptrs,
            row_bytes,
        )
        slot = None
        key = ("gpufetch", layer_id)
        if self._flag_sync:
            candidate = len(self._flag_slots)
            if candidate < self._flag_capacity:
                slot = candidate
                self._flag_slots[key] = slot
                self._ext.register_flag_gpufetch_task(slot, task)
        self._gpufetch_tasks[layer_id] = (task, slot)

    def gpufetch(self, layer_id: int) -> None:
        """Run a registered staging fill after its captured D2H control copies."""
        task, slot = self._gpufetch_tasks[layer_id]
        stream = torch.cuda.current_stream().cuda_stream
        if slot is not None:
            self._cpu_moe.memop_submit(
                stream, self._done.data_ptr(), self._ready.data_ptr(), slot,
            )
            self._cpu_moe.memop_sync(stream, self._done.data_ptr(), slot)
        else:
            # Existing portable house fallback: a captured cudaLaunchHostFunc node
            # fills the ring and returns before the following H2D gather can run.
            self._ext.gpufetch_with_cuda_stream(stream, task)

    def _prefill_io_for(self, bs: int) -> dict[str, torch.Tensor]:
        """One reusable prefill buffer, grown only when a larger chunk arrives."""
        validate_cpu_moe_task_tokens(bs, source="CPU MoE prefill batch size")
        if bs > self._prefill_capacity:
            alloc = alloc_pinned_tensor if self.device.type == "cuda" else torch.empty
            self._prefill_io = {
                "x": alloc(bs, self.H, dtype=torch.bfloat16),
                "ids": alloc(bs, self.top_k, dtype=torch.int32),
                "w": alloc(bs, self.top_k, dtype=torch.float32),
                "y": alloc(bs, self.H, dtype=torch.bfloat16),
            }
            self._prefill_capacity = bs
        assert self._prefill_io is not None
        return {name: tensor[:bs] for name, tensor in self._prefill_io.items()}

    def _prefetch_selected(self, layer_id: int, selected: list[int] | tuple[int, ...]) -> int:
        """Issue and account one coalesced prefetch for an already deduped set."""
        banks = self._disk_banks.get(int(layer_id))
        if not banks or not selected:
            return 0
        pages = 0
        paged_banks: set[int] = set()
        by_pager: dict[object, list] = {}
        for bank in banks:
            pager = getattr(bank, "_pager", None)
            if pager is not None:
                by_pager.setdefault(pager, []).append(bank)
                paged_banks.add(id(bank))
        for pager, pager_banks in by_pager.items():
            pages += pager.prefetch(pager_banks, selected)
        pages += sum(
            bank.prefetch_experts(selected) for bank in banks
            if id(bank) not in paged_banks
        )
        self._disk_prefetch_calls[layer_id] += 1
        self._disk_prefetch_pages[layer_id] += pages
        return pages

    def _populate_selected(
        self, layer_id: int, selected: list[int] | tuple[int, ...]
    ) -> int:
        """Populate file-backed rows and retain the UFFD pager's existing path."""
        banks = self._disk_banks.get(int(layer_id))
        if not banks or not selected:
            return 0
        populated = 0
        pages = 0
        pager_groups: dict[object, list] = {}
        file_banks = []
        for bank in banks:
            pager = getattr(bank, "_pager", None)
            if pager is not None:
                pager_groups.setdefault(pager, []).append(bank)
                continue
            if getattr(bank, "_tmpfs_backed", False):
                skipped = bank.selected_rows_nbytes(selected)
                self._prefill_populate_skipped_tmpfs_bytes = getattr(
                    self, "_prefill_populate_skipped_tmpfs_bytes", 0
                ) + skipped
                continue
            file_banks.append(bank)
        scratch = getattr(self, "_prefill_populate_scratch", None)
        if file_banks and scratch is None:
            scratch_bytes = getattr(
                self,
                "_prefill_populate_scratch_bytes",
                _PREFILL_POPULATE_SCRATCH_BYTES,
            )
            if scratch_bytes <= 0:
                raise MemoryError("populate scratch budget is empty")
            scratch = bytearray(scratch_bytes)
            self._prefill_populate_scratch = scratch
        for bank in file_banks:
            populate = getattr(bank, "populate_experts", None)
            if populate is None:
                raise RuntimeError("file-backed bank lacks populate_experts")
            bank_bytes = populate(selected, scratch)
            populated += bank_bytes
            self._prefill_populate_bytes = getattr(
                self, "_prefill_populate_bytes", 0
            ) + bank_bytes
        for pager, pager_banks in pager_groups.items():
            pages += pager.prefetch(pager_banks, selected)
        self._disk_prefetch_calls[layer_id] += 1
        self._disk_prefetch_pages[layer_id] += pages
        return populated

    def _record_prefill_degrade(self, message: str, exc: Exception) -> None:
        self._prefill_coalesce_degrades = getattr(
            self, "_prefill_coalesce_degrades", 0
        ) + 1
        if not getattr(self, "_prefill_coalesce_warned", False):
            logger.warning_rank0(f"CPU MoE prefill {message}: {exc}")
            self._prefill_coalesce_warned = True

    def _degrade_prefill_batch(self, message: str, exc: Exception) -> None:
        self._prefill_batch_enabled = False
        self._prefill_batch_degrades = getattr(
            self, "_prefill_batch_degrades", 0
        ) + 1
        if not getattr(self, "_prefill_batch_warned", False):
            logger.warning_rank0(f"CPU MoE {message}: {exc}")
            self._prefill_batch_warned = True

    def begin_decode_step(self) -> int:
        """Prefetch every DISK layer from its previous decode routing set.

        Called before graph replay or eager model execution. The native per-layer
        callback later compares real routing with this snapshot and advises only the
        delta. A layer without history is deliberately absent from the snapshot, so
        its first decode step keeps the existing reactive behavior.
        """
        if not getattr(self, "_disk_lookahead_enabled", False):
            return 0
        previous = getattr(self, "_disk_previous_experts", {})
        self._disk_predicted_experts = dict(previous)
        return sum(
            self._prefetch_selected(layer_id, self._disk_predicted_experts[layer_id])
            for layer_id in sorted(self._disk_banks)
            if layer_id in self._disk_predicted_experts
        )

    def reset_disk_lookahead(self) -> None:
        """Make the next decode step cold after a prefill or cache reset boundary."""
        self._disk_previous_experts = {}
        self._disk_predicted_experts = {}

    def prefetch_experts(
        self,
        layer_id: int,
        expert_ids,
        is_prefill: bool = False,
        route_pairs: int | None = None,
    ) -> int:
        """Prefetch the union of selected rows for one DISK layer.

        Decode invokes this after routing has reached the pinned host buffer and before
        the native executor wakes its GEMV workers. Prefill invokes it once with the
        whole token block's route union before the synchronous CPU task.
        """
        banks = self._disk_banks.get(int(layer_id))
        if not banks:
            return 0
        selected, counted_pairs = _dedupe_decode_routes(expert_ids, self.num_experts)
        if not is_prefill:
            self._disk_decode_steps += 1
            self._disk_route_pairs += counted_pairs if route_pairs is None else int(route_pairs)
            self._disk_distinct_experts += len(selected)
        if is_prefill:
            return self._prefetch_selected(layer_id, selected)

        predicted_by_layer = getattr(self, "_disk_predicted_experts", {})
        predicted = predicted_by_layer.pop(int(layer_id), None)
        if getattr(self, "_disk_lookahead_enabled", False):
            self._disk_previous_experts[int(layer_id)] = tuple(selected)
        if predicted is None:
            delta = selected
        else:
            predicted_set = set(predicted)
            self._disk_lookahead_routes = getattr(
                self, "_disk_lookahead_routes", 0
            ) + len(selected)
            self._disk_lookahead_hits = getattr(self, "_disk_lookahead_hits", 0) + sum(
                i in predicted_set for i in selected
            )
            delta = [i for i in selected if i not in predicted_set]
        pages = self._prefetch_selected(layer_id, delta)
        self._disk_delta_pages = getattr(self, "_disk_delta_pages", 0) + pages
        return pages

    def _warm_prefill_selected(
        self, layer_id: int, selected: list[int] | tuple[int, ...]
    ) -> None:
        """Apply the configured populate to WILLNEED to fault degrade chain."""
        mode = getattr(self, "_prefill_coalesce_mode", "on")
        if mode == "populate":
            banks = self._disk_banks.get(int(layer_id), ())
            has_file_banks = any(
                getattr(bank, "_pager", None) is None for bank in banks
            )
            if not has_file_banks:
                try:
                    self._prefetch_selected(int(layer_id), selected)
                except Exception as exc:
                    self._record_prefill_degrade(
                        "pager prefetch degraded to demand faults", exc
                    )
                return
            populate_started = time.perf_counter_ns()
            try:
                self._populate_selected(int(layer_id), selected)
            except Exception as exc:
                self._record_prefill_degrade(
                    "populate degraded to WILLNEED", exc
                )
                try:
                    self._prefetch_selected(int(layer_id), selected)
                except Exception as fallback_exc:
                    self._record_prefill_degrade(
                        "WILLNEED degraded to demand faults", fallback_exc
                    )
            finally:
                self._prefill_populate_ns = getattr(
                    self, "_prefill_populate_ns", 0
                ) + time.perf_counter_ns() - populate_started
            return
        try:
            self._prefetch_selected(int(layer_id), selected)
        except Exception as exc:
            self._record_prefill_degrade(
                "WILLNEED degraded to demand faults", exc
            )

    def _prefill_overlap_lock(self) -> threading.Lock:
        lock = getattr(self, "_prefill_populate_overlap_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._prefill_populate_overlap_lock = lock
        return lock

    def _finish_prefill_populate_overlap(
        self, layer_id: int | None = None
    ) -> tuple[int, ...]:
        """Join a predicted populate and account the portion hidden by compute."""
        lock = self._prefill_overlap_lock()
        with lock:
            overlap = getattr(self, "_prefill_populate_overlap", None)
            if overlap is None or (
                layer_id is not None and overlap.layer_id != int(layer_id)
            ):
                return ()
            account = not overlap.joining
            overlap.joining = True
        wait_started = time.perf_counter_ns()
        assert overlap.thread is not None
        overlap.thread.join()
        if not account:
            return ()
        wait_ns = time.perf_counter_ns() - wait_started
        duration_ns = max(0, overlap.finished_ns - overlap.started_ns)
        hidden_ns = max(0, duration_ns - wait_ns)
        self._prefill_populate_overlap_ns = getattr(
            self, "_prefill_populate_overlap_ns", 0
        ) + hidden_ns
        with lock:
            if getattr(self, "_prefill_populate_overlap", None) is overlap:
                self._prefill_populate_overlap = None
        return () if overlap.cancel.is_set() else overlap.experts

    def schedule_prefill_layer_overlap(self, layer_id: int, expert_ids):
        """Predict next-chunk layer-0 rows on one background populate thread.

        The current chunk's layer-0 union is the only route information available
        while its tail layers compute. The next chunk joins this lease before its
        actual layer-0 union is consumed, then synchronously warms only the delta.
        """
        if (
            not getattr(self, "_prefill_coalesce_enabled", True)
            or getattr(self, "_prefill_coalesce_mode", "on") != "populate"
        ):
            return None
        self._finish_prefill_populate_overlap()
        try:
            selected, _route_pairs = _dedupe_decode_routes(
                expert_ids, self.num_experts
            )
            limit = getattr(self, "_prefill_coalesce_limits", {}).get(
                int(layer_id), self.num_experts
            )
            selected = selected[:limit]
        except Exception as exc:
            self._record_prefill_degrade(
                "background sweep degraded to demand faults", exc
            )
            return None
        if not selected:
            return None
        overlap = _PrefillPopulateOverlap(
            int(layer_id), tuple(selected), threading.Event()
        )

        def populate() -> None:
            overlap.started_ns = time.perf_counter_ns()
            try:
                if not overlap.cancel.is_set():
                    self._warm_prefill_selected(overlap.layer_id, overlap.experts)
            except BaseException as exc:
                # No exception crosses the thread boundary. Even unusual failures
                # degrade this prediction only; exact compute can still demand-fault.
                try:
                    self._record_prefill_degrade(
                        "background sweep degraded to demand faults", exc
                    )
                except BaseException:
                    pass
            finally:
                overlap.finished_ns = time.perf_counter_ns()

        overlap.thread = threading.Thread(
            target=populate,
            name="freetoken-prefill-populate",
            daemon=True,
        )
        with self._prefill_overlap_lock():
            self._prefill_populate_overlap = overlap
            overlap.thread.start()
        return overlap

    def cancel_prefill_populate_overlap(self, *, wait: bool = True) -> None:
        """Cancel pending work; optionally drain the in-flight backing read."""
        with self._prefill_overlap_lock():
            overlap = getattr(self, "_prefill_populate_overlap", None)
        if overlap is None:
            return
        overlap.cancel.set()
        if wait:
            self._finish_prefill_populate_overlap()

    def prepare_prefill_layer(self, layer_id: int, expert_ids):
        """Warm one bounded expert union and return its post-compute lease.

        The native executor stores fixed pointers to the original bank mappings, so
        copied staging rows cannot serve it. Populate mode reads exact backing-file
        row ranges into one reusable scratch buffer, making later mmap accesses minor
        faults. A populate failure falls back to WILLNEED, then to demand faults.
        """
        if not getattr(self, "_prefill_coalesce_enabled", True):
            return None
        started = time.perf_counter_ns()
        selected: list[int] = []
        try:
            predicted = set(self._finish_prefill_populate_overlap(int(layer_id)))
            selected, _route_pairs = _dedupe_decode_routes(
                expert_ids, self.num_experts
            )
            limit = getattr(self, "_prefill_coalesce_limits", {}).get(
                int(layer_id), self.num_experts
            )
            had_selected = bool(selected)
            selected = selected[:limit]
            if not selected:
                # A zero ceiling is a deliberate bound, not permission for the
                # layer seam to retry an unbounded advisory sweep.
                return (
                    _PrefillCoalesceLease(int(layer_id), ())
                    if had_selected else None
                )
            delta = [expert for expert in selected if expert not in predicted]
            if delta:
                self._warm_prefill_selected(int(layer_id), delta)
        except Exception as exc:
            self._record_prefill_degrade("sweep degraded to demand faults", exc)
            # A non-None lease suppresses the layer seam's unguarded retry. Empty
            # experts also make the unchanged release path a no-op.
            return _PrefillCoalesceLease(int(layer_id), tuple(selected))
        finally:
            self._prefill_coalesce_ns = getattr(
                self, "_prefill_coalesce_ns", 0
            ) + time.perf_counter_ns() - started
        self._prefill_coalesce_experts = getattr(
            self, "_prefill_coalesce_experts", 0
        ) + len(selected)
        return _PrefillCoalesceLease(int(layer_id), tuple(selected))

    def release_prefill_layer(self, lease: _PrefillCoalesceLease) -> None:
        """Mark successfully swept rows as one-pass after native compute completes.

        Disabled by default: consecutive prefill chunks share most of their
        routed experts (Zipf), so eager per-layer eviction forces the next
        chunk's populate back to disk - measured at 5-11 tok/s vs 56 with the
        release inert. Single-lane serving has no concurrent decode to
        protect; page-cache LRU handles pressure. Opt back in with
        FREETOKEN_PREFILL_EAGER_RELEASE=1 for multi-lane experiments.
        """
        import os as _os

        if _os.environ.get("FREETOKEN_PREFILL_EAGER_RELEASE", "").strip() not in (
            "1", "true", "on",
        ):
            return
        banks = self._disk_banks.get(int(lease.layer_id), ())
        pager_groups: dict[object, list] = {}
        for bank in banks:
            pager = getattr(bank, "_pager", None)
            if pager is not None:
                pager_groups.setdefault(pager, []).append(bank)
                continue
            if getattr(bank, "_tmpfs_backed", False):
                skipped = bank.selected_rows_nbytes(lease.experts)
                self._prefill_release_skipped_tmpfs_bytes = getattr(
                    self, "_prefill_release_skipped_tmpfs_bytes", 0
                ) + skipped
                continue
            release = getattr(bank, "release_rows", None)
            if release is None:
                continue
            try:
                released_pages = release(lease.experts)
                self._prefill_release_pages = getattr(
                    self, "_prefill_release_pages", 0
                ) + int(released_pages or 0)
            except (MemoryError, OSError, RuntimeError) as exc:
                logger.warning_rank0(
                    f"CPU MoE prefill one-pass advice failed for layer "
                    f"{lease.layer_id}: {exc}"
                )
        for pager, pager_banks in pager_groups.items():
            release = getattr(pager, "release_prefill_rows", None)
            if release is None:
                continue
            try:
                release(pager_banks, lease.experts)
            except (MemoryError, OSError, RuntimeError) as exc:
                logger.warning_rank0(
                    f"CPU MoE prefill pager release failed for layer "
                    f"{lease.layer_id}: {exc}"
                )

    def reset_disk_stats(self) -> None:
        self._disk_prefetch_calls = [0] * self.num_layers
        self._disk_prefetch_pages = [0] * self.num_layers
        self._disk_decode_steps = 0
        self._disk_route_pairs = 0
        self._disk_distinct_experts = 0
        self._disk_lookahead_hits = 0
        self._disk_lookahead_routes = 0
        self._disk_delta_pages = 0
        self._prefill_coalesce_experts = 0
        self._prefill_coalesce_ns = 0
        self._prefill_coalesce_degrades = 0
        self._prefill_populate_bytes = 0
        self._prefill_populate_skipped_tmpfs_bytes = 0
        self._prefill_populate_ns = 0
        self._prefill_populate_overlap_ns = 0
        self._prefill_release_pages = 0
        self._prefill_release_skipped_tmpfs_bytes = 0
        self._prefill_batch_rows = 0
        self._prefill_batch_gemms = 0
        self._prefill_batch_degrades = 0
        self._disk_minor_fault_base, self._disk_major_fault_base = _process_faults()
        if getattr(self, "_gpufetch_tasks", None) and hasattr(self._ext, "gpufetch_stats"):
            self._ext.gpufetch_stats(True)
        for pager in getattr(self, "_disk_pagers", ()):
            pager.stats(reset=True)

    def disk_prefetch_stats(self, *, reset: bool = False) -> dict:
        """Aggregate DISK counters, reading procfs only when stats are flushed."""
        per_layer = [
            {
                "layer": layer_id,
                "prefetch_calls": self._disk_prefetch_calls[layer_id],
                "pages_requested": self._disk_prefetch_pages[layer_id],
            }
            for layer_id in sorted(self._disk_banks)
        ]
        minor_now, major_now = _process_faults()
        minor_base = getattr(self, "_disk_minor_fault_base", None)
        minor_faults = None if minor_now is None or minor_base is None else (
            minor_now - minor_base
        )
        major_faults = None if major_now is None or self._disk_major_fault_base is None else (
            major_now - self._disk_major_fault_base
        )
        gpufetch_fills = gpufetch_steps = gpufetch_ns = 0
        if getattr(self, "_gpufetch_tasks", None) and hasattr(self._ext, "gpufetch_stats"):
            gpufetch_fills, gpufetch_steps, gpufetch_ns = self._ext.gpufetch_stats(reset)
        disk_layers = len(self._disk_banks)
        decode_steps = self._disk_decode_steps / disk_layers if disk_layers else 0
        gpufetch_layers = len(getattr(self, "_gpufetch_tasks", ()))
        gpufetch_decode_steps = (
            gpufetch_steps / gpufetch_layers if gpufetch_layers else 0
        )
        if gpufetch_decode_steps:
            decode_steps = gpufetch_decode_steps
        result = {
            "prefetch_calls": sum(self._disk_prefetch_calls),
            "pages_requested": sum(self._disk_prefetch_pages),
            "major_faults": major_faults,
            "major_faults_unit": "kernel_events_4KiB_or_2MiB",
            "major_faults_per_decode_step": (
                major_faults / decode_steps
                if major_faults is not None and decode_steps else 0.0
            ),
            "minor_faults": minor_faults,
            "minor_faults_unit": "kernel_events_4KiB_or_2MiB",
            "minor_faults_per_decode_step": (
                minor_faults / decode_steps
                if minor_faults is not None and decode_steps else 0.0
            ),
            "distinct_experts_per_step": (
                self._disk_distinct_experts / self._disk_decode_steps
                if self._disk_decode_steps else 0.0
            ),
            "dedup_ratio": (
                self._disk_route_pairs / self._disk_distinct_experts
                if self._disk_distinct_experts else 0.0
            ),
            "lookahead_hit_rate": (
                getattr(self, "_disk_lookahead_hits", 0)
                / getattr(self, "_disk_lookahead_routes", 0)
                if getattr(self, "_disk_lookahead_routes", 0) else 0.0
            ),
            "delta_pages_per_step": (
                getattr(self, "_disk_delta_pages", 0) / decode_steps
                if decode_steps else 0.0
            ),
            "gpufetch_fills_per_step": (
                gpufetch_fills / gpufetch_decode_steps
                if gpufetch_decode_steps else 0.0
            ),
            "gpufetch_fill_us": (
                gpufetch_ns / 1_000 / gpufetch_decode_steps
                if gpufetch_decode_steps else 0.0
            ),
            "moe_prefill_coalesce_experts": getattr(
                self, "_prefill_coalesce_experts", 0
            ),
            "moe_prefill_coalesce_ms": getattr(
                self, "_prefill_coalesce_ns", 0
            ) / 1_000_000,
            "moe_prefill_coalesce_degrades": getattr(
                self, "_prefill_coalesce_degrades", 0
            ),
            "moe_prefill_populate_bytes": getattr(
                self, "_prefill_populate_bytes", 0
            ),
            "moe_prefill_populate_skipped_tmpfs_bytes": getattr(
                self, "_prefill_populate_skipped_tmpfs_bytes", 0
            ),
            "moe_prefill_populate_ms": getattr(
                self, "_prefill_populate_ns", 0
            ) / 1_000_000,
            "moe_prefill_populate_overlap_ms": getattr(
                self, "_prefill_populate_overlap_ns", 0
            ) / 1_000_000,
            "moe_prefill_release_pages": getattr(
                self, "_prefill_release_pages", 0
            ),
            "moe_prefill_release_skipped_tmpfs_bytes": getattr(
                self, "_prefill_release_skipped_tmpfs_bytes", 0
            ),
            "moe_prefill_batch_rows": getattr(
                self, "_prefill_batch_rows", 0
            ),
            "moe_prefill_batch_gemms": getattr(
                self, "_prefill_batch_gemms", 0
            ),
            "moe_prefill_batch_degrades": getattr(
                self, "_prefill_batch_degrades", 0
            ),
            "per_layer": per_layer,
        }
        pagers = list(getattr(self, "_disk_pagers", ()))
        if pagers:
            native_stats = [pager.stats(reset=reset) for pager in pagers]
            buckets = native_stats[0]["fill_latency_histogram"]["buckets_us"]
            counts = [0] * len(native_stats[0]["fill_latency_histogram"]["counts"])
            for item in native_stats:
                if item["fill_latency_histogram"]["buckets_us"] != buckets:
                    raise RuntimeError("UFFD pager latency histogram buckets disagree")
                counts = [
                    left + right for left, right in zip(
                        counts, item["fill_latency_histogram"]["counts"]
                    )
                ]
            result.update({
                "pager_backend": "uffd",
                "fills": sum(item["fills"] for item in native_stats),
                "fills_from_prefetch": sum(
                    item["fills_from_prefetch"] for item in native_stats
                ),
                "fault_driven": sum(item["fault_driven"] for item in native_stats),
                "evictions": sum(item["evictions"] for item in native_stats),
                "resident_bytes": sum(item["resident_bytes"] for item in native_stats),
                "pages_installed": sum(
                    item.get("pages_installed", 0) for item in native_stats
                ),
                "rows_spanning_pages": sum(
                    item.get("rows_spanning_pages", 0) for item in native_stats
                ),
                "fill_latency_histogram": {
                    "buckets_us": buckets,
                    "counts": counts,
                },
            })
        else:
            result["pager_backend"] = "madvise"
        if reset:
            self._disk_prefetch_calls = [0] * self.num_layers
            self._disk_prefetch_pages = [0] * self.num_layers
            self._disk_decode_steps = 0
            self._disk_route_pairs = 0
            self._disk_distinct_experts = 0
            self._disk_lookahead_hits = 0
            self._disk_lookahead_routes = 0
            self._disk_delta_pages = 0
            self._prefill_coalesce_experts = 0
            self._prefill_coalesce_ns = 0
            self._prefill_coalesce_degrades = 0
            self._prefill_populate_bytes = 0
            self._prefill_populate_skipped_tmpfs_bytes = 0
            self._prefill_populate_ns = 0
            self._prefill_populate_overlap_ns = 0
            self._prefill_release_pages = 0
            self._prefill_release_skipped_tmpfs_bytes = 0
            self._prefill_batch_rows = 0
            self._prefill_batch_gemms = 0
            self._prefill_batch_degrades = 0
            self._disk_minor_fault_base = minor_now
            self._disk_major_fault_base = major_now
        return result

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One MoE layer of decode on the CPU. Returns a GPU [bs, H] tensor.

        All ops go on the current CUDA stream so the whole thing is captured into
        the active CUDA graph (the two host nodes carry the data dependency on the
        pinned buffers, which hold this step's real routing on replay)."""
        pending = self.decode_submit(layer_id, hidden_states, topk_weights, topk_ids)
        return self.decode_sync(pending)

    def prefill(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run one prefill chunk synchronously on the native CPU worker pool.

        The caller performs the DISK expert-union prefetch before entering here.
        Blocking copies make all task inputs visible before the native task starts, and
        the returned tensor preserves the GPU path's input dtype, shape, and device.
        """
        bs = validate_cpu_moe_task_tokens(
            hidden_states.shape[0], source="CPU MoE prefill batch size"
        )
        io = self._prefill_io_for(bs)
        task_hidden = hidden_states
        if self._gpu_prequant:
            from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip

            task_hidden = act_quant_fp8_roundtrip(hidden_states, block=128)
        io["x"].copy_(task_hidden, non_blocking=False)
        io["ids"].copy_(topk_ids.to(torch.int32), non_blocking=False)
        io["w"].copy_(topk_weights.to(torch.float32), non_blocking=False)
        if not hasattr(self._ext, "run_task_sync"):
            raise RuntimeError(
                "the CPU MoE extension needs rebuilding for DISK CPU prefill; run "
                "`python setup.py build_ext --inplace` or reinstall the wheel"
            )
        if self._prefill_batch_enabled and bs <= self._prefill_batch_capacity:
            try:
                rows, gemms = self._ext.run_prefill_batch_sync(
                    layer_id,
                    bs,
                    io["x"].data_ptr(),
                    io["ids"].data_ptr(),
                    io["w"].data_ptr(),
                    io["y"].data_ptr(),
                )
                self._prefill_batch_rows += int(rows)
                self._prefill_batch_gemms += int(gemms)
            except Exception as exc:
                self._degrade_prefill_batch("batch run degraded to serial", exc)
                self._ext.run_task_sync(
                    layer_id,
                    bs,
                    io["x"].data_ptr(),
                    io["ids"].data_ptr(),
                    io["w"].data_ptr(),
                    io["y"].data_ptr(),
                    False,
                )
        else:
            self._ext.run_task_sync(
                layer_id,
                bs,
                io["x"].data_ptr(),
                io["ids"].data_ptr(),
                io["w"].data_ptr(),
                io["y"].data_ptr(),
                False,
            )
        out = torch.empty_like(hidden_states)
        out.copy_(io["y"], non_blocking=False)
        return out

    def decode_submit(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple:
        """Issue the D2H copies + the CPU-pool submit host node, then return without
        waiting. Lets a caller (the hybrid backend) enqueue GPU work between this and
        :meth:`decode_sync` so the CPU compute overlaps the GPU GEMM / PCIe fetch.

        ``topk_ids`` may carry ``-1`` entries (the C++ kernel skips them), so the CPU
        computes only the routes assigned to it. Returns an opaque handle to pass to
        :meth:`decode_sync`. The output tensor is allocated here so it stays live (and
        distinct from the interleaved GPU work) across the overlap window."""
        bs = hidden_states.shape[0]
        io = self._io_for(bs)
        timing = self._timing_for(layer_id, bs)
        if timing is not None:
            timing.layer_start.record(torch.cuda.current_stream())

        if self._gpu_prequant:
            # DSV4: apply the reference FP8 round-trip on the GPU (the same kernel the
            # GPU W4A8 path uses -> bit-identical grid) so the CPU side reads
            # pre-quantized activations and skips its serial scalar pass.
            from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip

            hidden_states = act_quant_fp8_roundtrip(hidden_states, block=128)

        # D2H: ship this step's activations + routing to pinned host memory.
        if timing is not None:
            timing.d2h_start.record(torch.cuda.current_stream())
        io["x"].copy_(hidden_states, non_blocking=True)
        io["ids"].copy_(topk_ids.to(torch.int32), non_blocking=True)
        io["w"].copy_(topk_weights.to(torch.float32), non_blocking=True)

        task = self._task_for(layer_id, bs)
        out = torch.empty_like(hidden_states)
        slot = self._flag_slots.get((layer_id, bs)) if self._flag_sync else None
        if slot is not None:
            # Front-end memops: done[slot]=0 then ready[slot]=1 (the coordinator's
            # doorbell). No kernel launched; no host-func round trip.
            self._cpu_moe.memop_submit(
                torch.cuda.current_stream().cuda_stream,
                self._done.data_ptr(), self._ready.data_ptr(), slot,
            )
        else:
            stream = torch.cuda.current_stream().cuda_stream
            self._ext.submit_with_cuda_stream(stream, task)
        if timing is not None:
            timing.overlap_start.record(torch.cuda.current_stream())
        return (bs, task, out, slot, timing, int(layer_id))

    def decode_sync(self, pending: tuple, *, hot_partial: bool = False) -> torch.Tensor:
        """Issue the CPU-pool sync + the H2D result copy for a prior :meth:`decode_submit`,
        and return the GPU output tensor. With flag-sync the wait is a front-end stream
        memop on done[slot] (set by the CPU coordinator); otherwise a cudaLaunchHostFunc."""
        bs, task, out, slot, timing, layer_id = pending
        if timing is not None and hot_partial:
            timing.hot_done.record(torch.cuda.current_stream())
            self._step_timing_hot_keys.add((layer_id, bs))
        if slot is not None:
            # Front-end WAIT(done[slot] >= 1): blocks this stream's later nodes without
            # occupying an SM, so GPU utilization stays truthful during the CPU window.
            self._cpu_moe.memop_sync(
                torch.cuda.current_stream().cuda_stream, self._done.data_ptr(), slot,
            )
        else:
            stream = torch.cuda.current_stream().cuda_stream
            self._ext.sync_with_cuda_stream(stream, task)
        if timing is not None:
            timing.wait_done.record(torch.cuda.current_stream())
        io = self._io[bs]
        out.copy_(io["y"], non_blocking=True)
        if timing is not None:
            timing.layer_end.record(torch.cuda.current_stream())
        return out

    def _watchdog_tick(self, suspects: dict) -> None:
        """One watchdog sampling round (called every 2 s by ``_watchdog_main``).

        A slot is only declared dead when THREE things hold across >=10 s: its doorbell
        is still pending (ready==1 && done==0), it was already pending when first
        suspected, and the coordinator has served NOTHING on it since (flag_served_count
        unchanged). The served-count criterion kills the false-positive window: two
        point samples can land on the same slot's (independent, us-scale) pending
        windows under heavy external load, but a coordinator that made progress in
        between is alive by definition. ``suspects`` maps slot -> (first_seen,
        served_at_first_sight) and persists across ticks."""
        stuck = (self._ready == 1) & (self._done == 0)
        if not bool(stuck.any()):
            suspects.clear()
            return
        now = time.monotonic()
        pending = set(stuck.nonzero().flatten().tolist())
        for slot in list(suspects):
            if slot not in pending:
                del suspects[slot]
        dead = []
        for slot in pending:
            served = self._ext.flag_served_count(slot)
            first_seen, served_then = suspects.get(slot, (None, None))
            if first_seen is None or served != served_then:
                suspects[slot] = (now, served)  # new suspect, or alive-but-loaded: rearm
                continue
            if now - first_seen >= 10.0:
                dead.append(slot)
        if not dead:
            return
        logger.error(
            f"cpu-moe flag watchdog: slots {dead} unanswered for >10s with no coordinator "
            "progress (wedged/dead); poisoning done[] and failing the next step"
        )
        for i in dead:
            self._err[i] = 1
        for i in dead:
            self._done[i] = 1  # after err: unblock the stream into a checked failure
            suspects.pop(i, None)

    def raise_if_unhealthy(self) -> None:
        """Raise if the flag watchdog fired (a doorbell stayed unanswered because the
        coordinator never responded). Called by the engine once per forward -- a single
        pinned read -- so a dead coordinator surfaces as a loud error on the next step
        instead of silently shipping stale expert outputs."""
        if self._disk_prefetch_error is not None:
            raise RuntimeError("DISK expert prefetch failed") from self._disk_prefetch_error
        if (
            getattr(self, "_gpufetch_tasks", None)
            and hasattr(self._ext, "gpufetch_error_code")
            and (code := self._ext.gpufetch_error_code())
        ):
            raise RuntimeError(f"DISK GPU-fetch staging failed with error code {code}")
        for pager in getattr(self, "_disk_pagers", ()):
            pager.raise_if_error()
        if self._err is not None and bool((self._err != 0).any()):
            raise RuntimeError(
                "CPU MoE flag-handshake watchdog fired: a decode step's doorbell was "
                "never answered by the coordinator thread (its outputs cannot be "
                "trusted). This indicates a wedged/killed coordinator; restart the "
                "engine, or set FREETOKEN_CPU_MOE_FLAG_SYNC=0 to use the "
                "cudaLaunchHostFunc sync."
            )


def _disk_prefetch_callback(
    executor_ref, layer_id: int, expert_ids, route_pairs: int | None = None,
) -> None:
    """No-throw native pre-run callback without a strong executor reference."""
    executor = executor_ref()
    if executor is None:
        return
    try:
        executor.prefetch_experts(layer_id, expert_ids, route_pairs=route_pairs)
    except BaseException as exc:
        if executor._disk_prefetch_error is None:
            executor._disk_prefetch_error = exc
            logger.error(f"DISK expert prefetch failed: {exc}")


def _watchdog_main(executor_ref) -> None:
    """Watchdog daemon body: weakref-deref per tick so the thread never keeps a dead
    executor alive (see the start site in ``CpuMoeExecutor.__init__``)."""
    suspects: dict = {}
    while True:
        time.sleep(2.0)
        executor = executor_ref()
        if executor is None or executor._watchdog_stop or executor._ready is None:
            return
        try:
            executor._watchdog_tick(suspects)
        finally:
            del executor  # drop the strong ref before the next sleep
