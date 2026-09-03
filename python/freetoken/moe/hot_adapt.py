"""Pure planning and bookkeeping for online HOT expert adaptation."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from freetoken.utils import init_logger


logger = init_logger(__name__)

HOT_PLAN_VERSION = 1
HOT_PLAN_FILENAME = "freetoken_hot_plan.json"
HOT_PLAN_TIER_COMMIT_ENV = "FREETOKEN_TIER_COMMIT"


# Covers the fixed mapped-host mapping/snapshot tensors and one row of rounding
# when max_swap_bytes is smaller than, or not divisible by, an expert row.
HOT_STAGING_HEADROOM_BYTES = 64 << 20
# This target spaces the initial due thresholds. It does not promise a complete
# fill at the first 2,000-token boundary: the per-boundary byte cap intentionally
# spreads an all-cold fill across about two requests at its default fraction.
HOT_ADAPT_TARGET_FILL_TOKENS = 2000
# Keep the established identifier for compatibility; its unit is now routed tokens.
HOT_ADAPT_STEADY_INTERVAL_STEPS = 1000
HOT_ADAPT_MAX_STAGING_FRACTION = 0.25


@dataclass(frozen=True)
class HotPlanSeed:
    """Validated persisted state selected for the current HOT geometry."""

    expert_ids: dict[int, tuple[int, ...]]
    counters: dict[int, tuple[float, ...]]
    seeded_layers: frozenset[int]
    age_seconds: float
    saved_hot_budget_bytes: int
    tier_commit: str
    tier_mismatch: bool


def resolve_tier_commit() -> str:
    """Return the serving code revision without depending on the model directory."""
    env_value = os.environ.get(HOT_PLAN_TIER_COMMIT_ENV, "").strip()
    if env_value:
        logger.info_rank0(
            f"HOT plan tier commit={env_value!r} source={HOT_PLAN_TIER_COMMIT_ENV}"
        )
        return env_value

    from freetoken import version as freetoken_version

    version_file = Path(freetoken_version.__file__ or "").resolve()
    installed_package = any(
        part in {"site-packages", "dist-packages"} for part in version_file.parts
    )
    source_file = Path(__file__).resolve()
    source_root = next(
        (
            parent
            for parent in source_file.parents
            if (parent / "pyproject.toml").is_file()
            and (parent / "python" / "freetoken" / "moe" / "hot_adapt.py").resolve()
            == source_file
        ),
        None,
    )
    if not installed_package and source_root is not None:
        try:
            result = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            git_root = Path(result.stdout.strip()).resolve()
            if git_root == source_root:
                result = subprocess.run(
                    [
                        "git", "-C", str(source_root), "describe", "--always", "--dirty"
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                value = result.stdout.strip()
                if value:
                    logger.info_rank0(
                        f"HOT plan tier commit={value!r} source=git"
                    )
                    return value
        except (OSError, subprocess.SubprocessError):
            pass

    value = f"package-{freetoken_version.__version__}"
    logger.warning_rank0(
        f"HOT plan tier commit={value!r} source=package-version; "
        "git metadata unavailable or package is installed"
    )
    return value


def hot_plan_path(model_path: str, plan_dir: str | None = None) -> str:
    directory = os.path.expanduser(plan_dir or model_path)
    return os.path.join(directory, HOT_PLAN_FILENAME)


def hot_plan_directory_writable(path: str) -> bool:
    """Probe actual create access, including ACL and read-only mount failures."""
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".freetoken-hot-plan-probe-"
        ):
            pass
        return True
    except OSError:
        return False


def checkpoint_identity(model_path: str) -> dict[str, Any]:
    """Build a cheap checkpoint identity from its small index and shard metadata."""
    directory = os.path.realpath(os.path.expanduser(model_path))
    candidates = (
        ("ftw", "freetoken_weight.json"),
        ("safetensors_index", "freetoken_bank_index.json"),
    )
    for kind, filename in candidates:
        index_path = os.path.join(directory, filename)
        if not os.path.isfile(index_path):
            continue
        with open(index_path, "rb") as handle:
            raw = handle.read()
        index = json.loads(raw)
        shards = []
        for entry in index.get("shards", ()):
            shard_name = entry.get("file")
            if not isinstance(shard_name, str):
                raise ValueError(f"checkpoint index {filename!r} has an invalid shard")
            stat = os.stat(os.path.join(directory, shard_name))
            shards.append(
                {
                    "file": shard_name,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        if not shards:
            raise ValueError(f"checkpoint index {filename!r} has no shards")
        return {
            "kind": kind,
            "path": directory,
            "index": filename,
            "index_sha256": hashlib.sha256(raw).hexdigest(),
            "shards": shards,
        }
    raise FileNotFoundError(f"no FTW or expert-bank index found under {directory!r}")


def _validated_counter_row(value: Any, num_experts: int, layer_id: int) -> tuple[float, ...]:
    if not isinstance(value, str):
        raise ValueError(
            f"counter layer {layer_id} must be base64 float32 data"
        )
    try:
        packed = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"counter layer {layer_id} is not valid base64") from exc
    expected_bytes = num_experts * 4
    if len(packed) != expected_bytes:
        raise ValueError(
            f"counter layer {layer_id} has {len(packed)} bytes, expected {expected_bytes}"
        )
    row = struct.unpack(f"<{num_experts}f", packed)
    if any(not math.isfinite(item) or item < 0 for item in row):
        raise ValueError(f"counter layer {layer_id} has an invalid value")
    return row


def load_hot_plan(
    path: str,
    *,
    identity: Mapping[str, Any],
    disk_layer_ids: frozenset[int],
    num_layers: int,
    num_experts: int,
    current_capacity: Mapping[int, int],
    current_hot_budget_bytes: int,
    static_expert_ids: Mapping[int, Sequence[int]],
    tier_commit: str,
    now: float | None = None,
) -> HotPlanSeed:
    """Validate and resize one persisted plan for the current protected slots."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("HOT plan top-level value must be an object")
    if raw.get("version") != HOT_PLAN_VERSION:
        raise ValueError(f"unsupported HOT plan version {raw.get('version')!r}")
    if raw.get("ftw_identity") != dict(identity):
        raise ValueError("FTW identity mismatch")
    if raw.get("num_layers") != num_layers or raw.get("num_experts") != num_experts:
        raise ValueError("expert geometry mismatch")
    saved_disk = raw.get("disk_layer_ids")
    if saved_disk != sorted(disk_layer_ids):
        raise ValueError("DISK layer set mismatch")
    saved_budget = raw.get("hot_budget_bytes")
    if isinstance(saved_budget, bool) or not isinstance(saved_budget, int) or saved_budget <= 0:
        raise ValueError("invalid saved HOT budget")
    if (
        isinstance(current_hot_budget_bytes, bool)
        or not isinstance(current_hot_budget_bytes, int)
        or current_hot_budget_bytes <= 0
    ):
        raise ValueError("invalid current HOT budget")
    budget_grew = current_hot_budget_bytes > saved_budget
    written_at = raw.get("written_at")
    if isinstance(written_at, bool) or not isinstance(written_at, (int, float)):
        raise ValueError("invalid HOT plan timestamp")
    if not math.isfinite(float(written_at)):
        raise ValueError("invalid HOT plan timestamp")
    if raw.get("counter_dtype") != "float32" or raw.get("counter_encoding") != "base64-le":
        raise ValueError("HOT plan counters are not marked float32")
    protected = raw.get("protected_slots")
    counters_raw = raw.get("decayed_counters")
    ranked_raw = raw.get("counter_ranked")
    if not all(isinstance(section, dict) for section in (protected, counters_raw, ranked_raw)):
        raise ValueError("HOT plan layer sections must be objects")

    hot_layer_ids = frozenset(int(layer_id) for layer_id in current_capacity)
    selected = {
        layer_id: tuple(int(expert) for expert in static_expert_ids.get(layer_id, ()))
        for layer_id in hot_layer_ids
    }
    counters: dict[int, tuple[float, ...]] = {}
    seeded_layers: set[int] = set()
    for layer_id in sorted(hot_layer_ids):
        key = str(layer_id)
        if key not in protected:
            continue
        slots = protected[key]
        if not isinstance(slots, list):
            raise ValueError(f"protected slot layer {layer_id} must be a list")
        slot_ids = tuple(int(expert) for expert in slots)
        if (
            len(set(slot_ids)) != len(slot_ids)
            or any(expert < 0 or expert >= num_experts for expert in slot_ids)
        ):
            raise ValueError(f"protected slot layer {layer_id} has invalid expert ids")
        row = _validated_counter_row(counters_raw.get(key), num_experts, layer_id)
        ranked = ranked_raw.get(key)
        if (
            not isinstance(ranked, list)
            or len(ranked) != len(slot_ids)
            or set(ranked) != set(slot_ids)
        ):
            raise ValueError(
                f"counter-ranked layer {layer_id} does not match protected residents"
            )
        capacity = int(current_capacity[layer_id])
        chosen_list = [int(expert) for expert in ranked[:capacity]]
        if budget_grew and capacity > len(chosen_list):
            resident_set = set(slot_ids)
            next_ranked = sorted(
                (expert for expert in range(num_experts) if expert not in resident_set),
                key=lambda expert: (-row[expert], expert),
            )
            chosen_list.extend(next_ranked[: capacity - len(chosen_list)])
        chosen = tuple(chosen_list)
        selected[layer_id] = chosen
        counters[layer_id] = row
        seeded_layers.add(layer_id)
    if not seeded_layers:
        raise ValueError("HOT plan has no protected layers for this process")
    if not any(any(value != 0.0 for value in row) for row in counters.values()):
        raise ValueError("HOT plan counters are all zero")

    saved_tier = raw.get("tier_commit")
    if not isinstance(saved_tier, str) or not saved_tier.strip():
        raise ValueError("HOT plan tier_commit is missing")
    timestamp = float(written_at)
    return HotPlanSeed(
        expert_ids=selected,
        counters=counters,
        seeded_layers=frozenset(seeded_layers),
        age_seconds=max(0.0, (time.time() if now is None else now) - timestamp),
        saved_hot_budget_bytes=saved_budget,
        tier_commit=saved_tier,
        tier_mismatch=saved_tier != tier_commit,
    )


def make_hot_plan_document(
    *,
    identity: Mapping[str, Any],
    disk_layer_ids: Sequence[int],
    num_layers: int,
    num_experts: int,
    hot_budget_bytes: int,
    tier_commit: str,
    protected_slots: Mapping[int, Sequence[int | None]],
    decayed_counters: Mapping[int, Sequence[float]],
    written_at: float | None = None,
) -> dict[str, Any] | None:
    """Create a compact JSON document, or None for an all-zero snapshot."""
    counter_rows = {
        int(layer_id): tuple(float(value) for value in decayed_counters[layer_id])
        for layer_id in sorted(decayed_counters)
    }
    for layer_id, row in counter_rows.items():
        if len(row) != num_experts:
            raise ValueError(
                f"counter layer {layer_id} has {len(row)} values, expected {num_experts}"
            )
        if any(not math.isfinite(value) or value < 0 for value in row):
            raise ValueError(f"counter layer {layer_id} has an invalid value")
    if not any(any(value != 0.0 for value in row) for row in counter_rows.values()):
        return None
    counters = {
        str(layer_id): base64.b64encode(
            struct.pack(f"<{num_experts}f", *row)
        ).decode("ascii")
        for layer_id, row in counter_rows.items()
    }
    protected: dict[str, list[int]] = {}
    ranked: dict[str, list[int]] = {}
    for layer_id in sorted(protected_slots):
        if layer_id not in counter_rows:
            raise ValueError(f"protected layer {layer_id} has no counter row")
        residents = [int(expert) for expert in protected_slots[layer_id] if expert is not None]
        if (
            len(set(residents)) != len(residents)
            or any(expert < 0 or expert >= num_experts for expert in residents)
        ):
            raise ValueError(f"protected layer {layer_id} has invalid expert ids")
        protected[str(layer_id)] = residents
        row = counter_rows[layer_id]
        ranked[str(layer_id)] = sorted(
            residents, key=lambda expert: (-row[expert], expert)
        )
    return {
        "version": HOT_PLAN_VERSION,
        "written_at": time.time() if written_at is None else float(written_at),
        "ftw_identity": dict(identity),
        "disk_layer_ids": sorted(int(layer_id) for layer_id in disk_layer_ids),
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "hot_budget_bytes": int(hot_budget_bytes),
        "tier_commit": str(tier_commit),
        "counter_dtype": "float32",
        "counter_encoding": "base64-le",
        "protected_slots": protected,
        "counter_ranked": ranked,
        "decayed_counters": counters,
    }


def atomic_write_hot_plan(
    path: str,
    document: Mapping[str, Any],
    *,
    publish: Callable[[str, str], bool] | None = None,
) -> bool:
    """Publish one durable plan, or return false when its publish fence closes."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        if publish is not None and not publish(temporary, path):
            return False
        if publish is None:
            os.replace(temporary, path)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(directory, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@dataclass
class HotAdaptIdleTracker:
    """Host-side trigger state for bounded adaptation while serving is idle.

    Preemption bounds host staging to one additional expert row. It does not
    cancel installation of the staged prefix. Up to the configured staging row
    count can still be queued on the scheduler stream before the next forward,
    costing 25 to 50 ms on node-4 at the default 0.5 GiB swap bound.
    """

    idle_seconds: float
    min_interval_seconds: float
    counter_generation: int = 0
    last_tick_generation: int = 0
    idle_started_at: float | None = None
    last_tick_completed_at: float | None = None
    last_tick_swaps: int = 0

    def __post_init__(self) -> None:
        if self.idle_seconds < 0 or self.min_interval_seconds < 0:
            raise ValueError("HOT adaptation idle intervals must be non-negative")

    def note_routed_pairs(self) -> None:
        self.counter_generation += 1

    def begin_idle(self, now: float) -> None:
        if self.idle_started_at is None:
            self.idle_started_at = now

    def end_idle(self) -> None:
        self.idle_started_at = None
        self.last_tick_swaps = 0

    def has_evidence(self) -> bool:
        return (
            self.counter_generation != self.last_tick_generation
            or self.last_tick_swaps > 0
        )

    def due(self, now: float) -> bool:
        if self.idle_started_at is None or not self.has_evidence():
            return False
        if now - self.idle_started_at < self.idle_seconds:
            return False
        return (
            self.last_tick_completed_at is None
            or now - self.last_tick_completed_at >= self.min_interval_seconds
        )

    def seconds_until_due(self, now: float) -> float:
        """Return the bounded wait until the next eligible idle tick."""
        if self.idle_started_at is None or not self.has_evidence():
            return 0.0
        due_at = self.idle_started_at + self.idle_seconds
        if self.last_tick_completed_at is not None:
            due_at = max(
                due_at,
                self.last_tick_completed_at + self.min_interval_seconds,
            )
        return max(0.0, due_at - now)

    def tick_started(self) -> None:
        self.last_tick_generation = self.counter_generation
        self.last_tick_swaps = 0

    def tick_completed(self, now: float, swaps: int) -> None:
        if swaps < 0:
            raise ValueError("HOT adaptation idle swap count must be non-negative")
        self.last_tick_completed_at = now
        self.last_tick_swaps = swaps


@dataclass
class HotAdaptTokenClock:
    """Shared prefill and decode clock for HOT adaptation boundaries.

    Due thresholds accumulate while an earlier rerank or copy is active. The
    next free boundary consumes them together, and each threshold still counts
    as one interval for reporting, including for an explicit fixed interval.
    """

    interval: int
    routed_tokens: int = 0
    next_tick_token: int = 0
    last_tick_token: int = 0

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("HOT adaptation token interval must be positive")
        if self.next_tick_token == 0:
            self.next_tick_token = self.interval

    def advance(self, routed_tokens: int) -> int:
        """Add routed tokens and return the number of interval ticks now due."""
        if routed_tokens < 0:
            raise ValueError("HOT adaptation routed token count must be non-negative")
        self.routed_tokens += routed_tokens
        if self.routed_tokens < self.next_tick_token:
            return 0
        return 1 + (self.routed_tokens - self.next_tick_token) // self.interval

    def consume_tick(self) -> int:
        """Consume one due tick and return its routed-token threshold."""
        if self.routed_tokens < self.next_tick_token:
            raise RuntimeError("HOT adaptation token tick is not due")
        token = self.next_tick_token
        self.last_tick_token = token
        self.next_tick_token += self.interval
        return token

    def consume_forced_tick(self) -> int:
        """Consume a tick now and place the next deadline one interval ahead."""
        self.next_tick_token = self.routed_tokens
        return self.consume_tick()

    def set_interval(self, interval: int) -> None:
        """Apply a new auto interval without moving the clock backwards."""
        if interval <= 0:
            raise ValueError("HOT adaptation token interval must be positive")
        self.interval = interval
        self.next_tick_token = max(
            self.next_tick_token,
            self.last_tick_token + interval,
        )


@dataclass
class HotAdaptIntervalController:
    """Allocation-derived HOT adaptation cadence and its runtime state."""

    auto: bool
    fill_ticks: int
    fill_interval: int
    steady_interval: int
    current_interval: int
    target_fill_tokens: int
    fill_complete: bool = False

    @classmethod
    def create(
        cls,
        interval_steps: str | int,
        *,
        hot_budget_bytes: int,
        max_swap_bytes: int,
        target_fill_tokens: int = HOT_ADAPT_TARGET_FILL_TOKENS,
        steady_interval: int = HOT_ADAPT_STEADY_INTERVAL_STEPS,
    ) -> HotAdaptIntervalController:
        if hot_budget_bytes <= 0 or max_swap_bytes <= 0:
            raise ValueError("HOT adaptation interval geometry must be positive")
        if target_fill_tokens <= 0 or steady_interval <= 0:
            raise ValueError("HOT adaptation interval targets must be positive")
        fill_ticks = (hot_budget_bytes + max_swap_bytes - 1) // max_swap_bytes
        fill_interval = max(1, target_fill_tokens // fill_ticks)
        auto = interval_steps == "auto"
        if not auto and (
            isinstance(interval_steps, bool)
            or not isinstance(interval_steps, int)
            or interval_steps < 0
        ):
            raise ValueError("HOT adaptation interval must be 'auto' or non-negative")
        current = fill_interval if auto else int(interval_steps)
        return cls(
            auto=auto,
            fill_ticks=fill_ticks,
            fill_interval=fill_interval,
            steady_interval=steady_interval,
            current_interval=current,
            target_fill_tokens=target_fill_tokens,
        )

    def complete_tick(
        self,
        *,
        partition_full: bool,
        tick_interval: int,
        staging_seconds: float,
        covered_seconds: float,
    ) -> tuple[bool, bool, int]:
        """Apply a completed tick and return switch, back-off, and back-off floor."""
        if not self.auto:
            return False, False, self.current_interval

        backed_off = (
            covered_seconds > 0
            and staging_seconds
            > HOT_ADAPT_MAX_STAGING_FRACTION * covered_seconds
        )
        backoff_interval = max(self.fill_interval, tick_interval * 2)
        if backed_off:
            self.current_interval = max(self.current_interval, backoff_interval)

        switched = not self.fill_complete and partition_full
        if switched:
            self.fill_complete = True
            self.current_interval = max(
                self.current_interval,
                self.fill_interval,
                self.steady_interval,
            )
        return switched, backed_off, backoff_interval


@dataclass(frozen=True)
class HotSwap:
    """Install ``incoming_expert`` into one fixed HOT bank row."""

    layer_id: int
    row: int
    incoming_expert: int
    outgoing_expert: int | None


def hot_staging_rows(max_swap_bytes: int, expert_bytes: int) -> int:
    """Rows in the reusable host staging bank.

    Adaptation itself remains bounded by ``floor(max_swap / expert_bytes)``.
    One row is retained when that quotient is zero so a profiled initial set and
    a runtime cache rebuild can still be streamed without a full host mirror.
    """
    if max_swap_bytes < 0 or expert_bytes <= 0:
        raise ValueError("HOT staging requires a non-negative swap bound and positive rows")
    return max(1, max_swap_bytes // expert_bytes)


def hot_catchup_swap_bytes(
    max_swap_bytes: int,
    expert_bytes: int,
    tick_count: int,
    *,
    hot_budget_bytes: int,
    boundary_cap_frac: float,
) -> int:
    """Row-aligned planner bound for all ticks sharing one request boundary.

    The boundary cap deliberately prevents one 2,000-token prefill chunk from
    filling an all-cold HOT partition. With the default 0.5 fraction, the
    initial fill normally completes at about the second request boundary.
    """
    if (
        max_swap_bytes < 0
        or expert_bytes <= 0
        or tick_count <= 0
        or hot_budget_bytes <= 0
    ):
        raise ValueError("HOT catch-up staging geometry must be positive")
    if (
        isinstance(boundary_cap_frac, bool)
        or not math.isfinite(boundary_cap_frac)
        or not 0 < boundary_cap_frac <= 1
    ):
        raise ValueError("HOT boundary cap fraction must be finite and in (0, 1]")
    swaps_per_tick = max_swap_bytes // expert_bytes
    tick_bound = swaps_per_tick * tick_count * expert_bytes
    # A valid HOT partition always contains at least one whole row. Preserve
    # progress when the fractional cap is smaller than that single row.
    boundary_bound = max(expert_bytes, int(hot_budget_bytes * boundary_cap_frac))
    boundary_bound -= boundary_bound % expert_bytes
    return min(tick_bound, boundary_bound)


def prefill_run_swap_budget(
    per_boundary_bytes: int,
    expert_bytes: int,
    swapped_bytes: int,
    *,
    hot_budget_bytes: int,
    run_cap_frac: float,
) -> int:
    """Return the row-aligned remaining swap allowance for one prefill run."""
    if (
        per_boundary_bytes < 0
        or expert_bytes <= 0
        or swapped_bytes < 0
        or hot_budget_bytes <= 0
    ):
        raise ValueError("HOT prefill run swap geometry must be positive")
    if (
        isinstance(run_cap_frac, bool)
        or not math.isfinite(run_cap_frac)
        or not 0 <= run_cap_frac <= 1
    ):
        raise ValueError("HOT prefill run cap fraction must be 0 or finite and in (0, 1]")
    if run_cap_frac == 0:
        return per_boundary_bytes
    cap_bytes = int(hot_budget_bytes * run_cap_frac)
    cap_bytes -= cap_bytes % expert_bytes
    remaining_bytes = max(0, cap_bytes - swapped_bytes)
    budget_bytes = min(per_boundary_bytes, remaining_bytes)
    return budget_bytes - budget_bytes % expert_bytes


def hot_boundary_interval_tokens(
    tick_interval: int,
    max_swap_bytes: int,
    staged_bytes: int,
) -> int:
    """Token span whose nominal swap allowance covers actual boundary bytes."""
    if tick_interval <= 0 or max_swap_bytes <= 0 or staged_bytes < 0:
        raise ValueError("HOT boundary bandwidth geometry must be positive")
    staged_intervals = max(
        1,
        (staged_bytes + max_swap_bytes - 1) // max_swap_bytes,
    )
    return tick_interval * staged_intervals


def hot_staging_budget_bytes(max_swap_bytes: int) -> int:
    """Conservative governor charge for staging payload plus fixed control data."""
    if max_swap_bytes < 0:
        raise ValueError("HOT staging swap bound must be non-negative")
    return max_swap_bytes + HOT_STAGING_HEADROOM_BYTES


def decay_multiplier(half_life_steps: int, elapsed_steps: int = 1) -> float:
    """Return the exponential multiplier for the requested number of steps."""
    if half_life_steps <= 0:
        raise ValueError("decay half-life must be positive")
    if elapsed_steps < 0:
        raise ValueError("elapsed decay steps must be non-negative")
    # math.exp2 is Python 3.11+; serving images still run 3.10
    return 2.0 ** (-float(elapsed_steps) / float(half_life_steps))


def update_decayed_counts(
    previous: Sequence[float],
    routed: Sequence[float],
    *,
    half_life_steps: int,
    elapsed_steps: int = 1,
) -> tuple[float, ...]:
    """CPU reference for one exact decay-and-add accumulator update.

    Prefill and decode follow the same rate rule: each routed pair contributes
    one, with no per-step, per-batch, or per-chunk normalization.
    """
    if len(previous) != len(routed):
        raise ValueError("previous and routed counts must have the same length")
    factor = decay_multiplier(half_life_steps, elapsed_steps)
    return tuple(float(old) * factor + float(new) for old, new in zip(previous, routed))


def recompute_hot_partition(
    expert_counts: Mapping[int, Sequence[float]],
    hot_layer_ids: frozenset[int],
    *,
    budget_bytes: int,
    expert_bytes: int,
    num_experts: int,
    capacities: Mapping[int, int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Select each protected layer's top experts under its fixed capacity."""
    if budget_bytes < 0 or expert_bytes <= 0 or num_experts <= 0:
        raise ValueError("HOT planner geometry must be non-negative with positive rows")
    if not hot_layer_ids or budget_bytes == 0:
        return {}
    missing = set(hot_layer_ids) - set(expert_counts)
    if missing:
        raise ValueError(
            f"counts have no entries for protected layers {sorted(missing)}"
        )
    if capacities is None:
        top_n = min(
            num_experts,
            budget_bytes // (expert_bytes * len(hot_layer_ids)),
        )
        capacities = {layer_id: top_n for layer_id in hot_layer_ids}
    elif set(capacities) != set(hot_layer_ids):
        raise ValueError("HOT capacities must cover exactly the protected layers")
    if sum(int(value) for value in capacities.values()) * expert_bytes > budget_bytes:
        raise ValueError("HOT capacities exceed the configured resident-byte budget")
    result = {}
    for layer_id in sorted(hot_layer_ids):
        top_n = min(num_experts, int(capacities[layer_id]))
        if top_n <= 0:
            continue
        counts = expert_counts[layer_id]
        if len(counts) != num_experts:
            raise ValueError(
                f"counts layer {layer_id} has {len(counts)} experts, expected {num_experts}"
            )
        ranked = sorted(
            range(num_experts),
            key=lambda expert_id: (-float(counts[expert_id]), expert_id),
        )
        result[layer_id] = tuple(sorted(ranked[:top_n]))
    return result


def plan_hot_swaps(
    expert_counts: Mapping[int, Sequence[float]],
    slot_owners: Mapping[int, Sequence[int | None]],
    desired: Mapping[int, Sequence[int]],
    *,
    expert_bytes: int,
    max_swap_bytes: int,
) -> tuple[HotSwap, ...]:
    """Plan deterministic highest-gain row replacements within one byte bound."""
    if expert_bytes <= 0 or max_swap_bytes < 0:
        raise ValueError("swap planner requires positive expert bytes and a non-negative bound")
    max_swaps = max_swap_bytes // expert_bytes
    if max_swaps <= 0:
        return ()

    candidates: list[tuple[float, int, int, HotSwap]] = []
    for layer_id in sorted(desired):
        counts = expert_counts[layer_id]
        owners = tuple(slot_owners[layer_id])
        current = {owner for owner in owners if owner is not None}
        target = set(int(expert) for expert in desired[layer_id])
        incoming = sorted(target - current, key=lambda expert: (-counts[expert], expert))
        free_rows = [row for row, owner in enumerate(owners) if owner is None]
        outgoing_rows = sorted(
            (
                (float(counts[owner]), owner, row)
                for row, owner in enumerate(owners)
                if owner is not None and owner not in target
            ),
            key=lambda item: (item[0], -item[1], item[2]),
        )
        available = [(0.0, None, row) for row in free_rows] + outgoing_rows
        for incoming_expert, (out_score, outgoing_expert, row) in zip(incoming, available):
            gain = float(counts[incoming_expert]) - out_score
            swap = HotSwap(layer_id, row, incoming_expert, outgoing_expert)
            candidates.append((-gain, layer_id, incoming_expert, swap))

    candidates.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in candidates[:max_swaps])


def retire_hot_swaps(
    mapping: Sequence[Sequence[int]], swaps: Sequence[HotSwap],
) -> list[list[int]]:
    """Remove outgoing mappings before their rows can be used as staging."""
    retired = [list(layer) for layer in mapping]
    for swap in swaps:
        layer = retired[swap.layer_id]
        if layer[swap.incoming_expert] >= 0:
            raise RuntimeError("incoming HOT expert is already mapped")
        if swap.outgoing_expert is not None:
            if layer[swap.outgoing_expert] != swap.row:
                raise RuntimeError("outgoing HOT row ownership changed before retirement")
            layer[swap.outgoing_expert] = -1
    return retired


def finish_hot_swaps(
    mapping: Sequence[Sequence[int]],
    swaps: Sequence[HotSwap],
    copied_rows: set[tuple[int, int]],
) -> list[list[int]]:
    """Publish only rows whose complete bank copies have been acknowledged."""
    finished = [list(layer) for layer in mapping]
    for swap in swaps:
        key = (swap.layer_id, swap.row)
        if key not in copied_rows:
            raise RuntimeError(
                f"refusing to publish HOT layer {swap.layer_id} row {swap.row} before copy"
            )
        layer = finished[swap.layer_id]
        if layer[swap.incoming_expert] >= 0:
            raise RuntimeError("incoming HOT expert became mapped before copy completion")
        layer[swap.incoming_expert] = swap.row
    return finished
