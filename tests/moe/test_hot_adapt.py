"""GPU-free tests for online HOT expert adaptation policy."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from freetoken.moe.hot_adapt import (
    HOT_ADAPT_MAX_STAGING_FRACTION,
    HOT_STAGING_HEADROOM_BYTES,
    HotAdaptIdleTracker,
    HotAdaptIntervalController,
    HotAdaptTokenClock,
    HotSwap,
    finish_hot_swaps,
    hot_boundary_interval_tokens,
    hot_catchup_swap_bytes,
    prefill_run_swap_budget,
    hot_staging_budget_bytes,
    hot_staging_rows,
    plan_hot_swaps,
    recompute_hot_partition,
    retire_hot_swaps,
    update_decayed_counts,
)


def _real_slot_cache_stats_without_triton(monkeypatch):
    """Load the real slot-cache stat layout while replacing only Triton decorators."""

    def jit(fn=None, **_kwargs):
        return (lambda decorated: decorated) if fn is None else fn

    triton = ModuleType("triton")
    triton.jit = jit
    triton_language = ModuleType("triton.language")
    triton.language = triton_language
    flashlib_spec = importlib.util.find_spec("flashlib")
    assert flashlib_spec is not None and flashlib_spec.origin is not None
    source = (
        Path(flashlib_spec.origin).parent
        / "kernels"
        / "slot_cache"
        / "triton"
        / "lru_ensure.py"
    )
    spec = importlib.util.spec_from_file_location("_real_slot_cache_lru_ensure", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as stats_patch:
        stats_patch.setitem(sys.modules, "triton", triton)
        stats_patch.setitem(sys.modules, "triton.language", triton_language)
        spec.loader.exec_module(module)
    return module.N_STATS, module.Stat


def _offload_cache_class_without_triton(monkeypatch):
    """Import the orchestration class without retaining GPU-kernel stubs."""
    n_stats, stat = _real_slot_cache_stats_without_triton(monkeypatch)
    kernels = ModuleType("flashlib.kernels")
    kernels.__path__ = []
    slot_cache = ModuleType("flashlib.kernels.slot_cache")
    slot_cache.N_STATS = n_stats
    slot_cache.Stat = stat
    monkeypatch.setitem(sys.modules, "flashlib.kernels", kernels)
    monkeypatch.setitem(sys.modules, "flashlib.kernels.slot_cache", slot_cache)
    monkeypatch.delitem(sys.modules, "freetoken.moe.offload_cache", raising=False)
    module = importlib.import_module("freetoken.moe.offload_cache")
    try:
        return module.OffloadMoeCache
    finally:
        sys.modules.pop("freetoken.moe.offload_cache", None)


def _offload_kernels_without_triton(monkeypatch):
    """Import CPU kernel mirrors with inert Triton and flashlib surfaces."""
    triton = ModuleType("triton")
    triton.jit = lambda fn=None, **_kwargs: (
        (lambda decorated: decorated) if fn is None else fn
    )
    triton.next_power_of_2 = lambda value: value
    triton.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
    language = ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    slot_cache = ModuleType("flashlib.kernels.slot_cache")
    slot_cache.lru_ensure = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(sys.modules, "flashlib.kernels.slot_cache", slot_cache)
    monkeypatch.delitem(sys.modules, "freetoken.moe.offload_kernels", raising=False)
    return importlib.import_module("freetoken.moe.offload_kernels")


def _mixed_hot_cache(monkeypatch, *, pinned_capacity=2):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [
            torch.arange(12, dtype=torch.int32).view(4, 3),
            torch.arange(12, dtype=torch.int32).view(4, 3) + 100,
        ],
        "down": [
            torch.arange(8, dtype=torch.int32).view(4, 2),
            torch.arange(8, dtype=torch.int32).view(4, 2) + 100,
        ],
    }
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=12,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    capacities = {0: 2}
    ids = {0: (0, 1)}
    if pinned_capacity:
        capacities[1] = pinned_capacity
        ids[1] = tuple(range(pinned_capacity))
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value, HostResidency.PINNED.value],
        hot_expert_ids=ids,
        hot_expert_capacity=capacities,
    )
    return cache, sources


def test_mixed_hot_carve_and_zero_pinned_budget_geometry(monkeypatch):
    mixed, _ = _mixed_hot_cache(monkeypatch)
    assert mixed._hot_slot_for_row == {0: (8, 9), 1: (10, 11)}
    assert mixed.is_hot_split_layer(0)
    assert not mixed.is_hot_split_layer(1)
    assert mixed.is_pinned_hot_layer(1)
    assert set(mixed._hot_slot_for_row[0]).isdisjoint(mixed._hot_slot_for_row[1])

    disk_only, _ = _mixed_hot_cache(monkeypatch, pinned_capacity=0)
    assert disk_only._hot_slot_for_row == {0: (10, 11)}
    assert min(disk_only._hot_slot_for_row[0]) == 10


def test_hot_gate_rejects_locked_but_accepts_pinned(monkeypatch):
    import torch

    from freetoken.moe.host_banks import HostResidency

    cache, sources = _mixed_hot_cache(monkeypatch)
    assert cache.is_pinned_hot_layer(1)
    rejected = type(cache)(
        num_layers=2,
        num_experts=4,
        cache_size=12,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    rejected.cpu_layer_ids = frozenset({0, 1})
    with pytest.raises(ValueError, match="neither DISK nor PINNED"):
        rejected.set_bank_sources(
            sources,
            layer_residency=[HostResidency.DISK.value, HostResidency.LOCKED.value],
            hot_expert_capacity={1: 1},
        )


def test_copy_plan_keeps_pinned_hot_source_and_zeros_disk_hot_source(monkeypatch):
    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    module_globals = OffloadMoeCache._build_copy_plan.__globals__
    pinned_module = ModuleType("freetoken.kernel.pinned")
    pinned_module.device_ptr = lambda tensor: tensor.data_ptr()
    monkeypatch.setitem(sys.modules, "freetoken.kernel.pinned", pinned_module)
    monkeypatch.setitem(module_globals, "_FUSED_COPY", True)
    real_tensor = torch.tensor

    def cpu_descriptor(data, *args, **kwargs):
        kwargs["device"] = "cpu"
        return real_tensor(data, *args, **kwargs)

    monkeypatch.setattr(module_globals["torch"], "tensor", cpu_descriptor)
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.device = SimpleNamespace(type="cuda")
    cache.num_layers = 2
    cache.hot_expert_capacity = {0: 1, 1: 1}
    cache._pinned_hot_layer_ids = frozenset({1})
    cache.layer_residency = ["disk", "pinned"]
    cache._unpinned_layers = frozenset({0})
    per_layer = [torch.zeros((4, 4), dtype=torch.int32) for _ in range(2)]
    cache.banks = [(per_layer, torch.zeros((8, 4), dtype=torch.int32))]

    cache._build_copy_plan()

    assert cache._copy_src_ptrs_host[0] == [0]
    assert cache._copy_src_ptrs_host[1] == [per_layer[1].data_ptr()]


def test_pinned_hot_decay_counter_and_plain_ensure_orchestration(monkeypatch):
    import torch

    kernels = _offload_kernels_without_triton(monkeypatch)
    cache, _ = _mixed_hot_cache(monkeypatch)
    cache.record_decode_frequency = lambda *_args: None
    cache.hot_adapt_enabled = True
    cache._hot_decay_factor = 0.5
    cache.decayed_decode_freq[1].fill_(2.0)
    cache.ensure_experts(1, torch.tensor([1, 1, 3], dtype=torch.int32))

    assert cache.decayed_decode_freq[0].tolist() == [0.0] * 4
    assert cache.decayed_decode_freq[1].tolist() == pytest.approx(
        [1.0, 3.0, 1.0, 2.0]
    )
    assert cache.usage[10:].tolist() == [torch.iinfo(torch.int64).max] * 2

    row = torch.tensor([4.0, 2.0, 0.0, 8.0])
    kernels._bump_decayed_freq_cpu(
        row, torch.tensor([0, 2, 2]), 0.25, 4
    )
    assert row.tolist() == pytest.approx([2.0, 0.5, 2.0, 2.0])

    before = cache.decayed_decode_freq.clone()
    cache.hot_adapt_enabled = False
    cache.ensure_experts(1, torch.tensor([0], dtype=torch.int32))
    assert torch.equal(cache.decayed_decode_freq, before)


def test_mixed_capacity_planner_uses_each_layers_capacity():
    counts = {0: (1, 9, 2, 8), 1: (7, 3, 6, 5)}
    assert recompute_hot_partition(
        counts,
        frozenset({0, 1}),
        budget_bytes=30,
        expert_bytes=10,
        num_experts=4,
        capacities={0: 1, 1: 2},
    ) == {0: (1,), 1: (0, 2)}


def test_pinned_swap_stages_directly_from_bank_tensor(monkeypatch):
    import builtins
    import torch

    cache, sources = _mixed_hot_cache(monkeypatch)
    cache._hot_staging_rows = 1
    cache._hot_staging = [
        torch.empty_like(sources[name][1][:1]) for name in cache.bank_schema
    ]
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *_args, **_kwargs: pytest.fail("PINNED staging must not open a file"),
    )
    copied, _ = cache._stage_hot_rows(
        None, (HotSwap(1, 0, incoming_expert=3, outgoing_expert=0),)
    )
    assert copied == {(1, 0)}
    assert torch.equal(cache._hot_staging[0][0], sources["gate_up"][1][3])


def test_pinned_lru_stats_report_missing_bytes_and_keep_disk_rate(monkeypatch):
    cache, sources = _mixed_hot_cache(monkeypatch)
    stat = cache.disk_prefetch_stats.__globals__["Stat"]
    cache.collect_stats = True
    cache.lru_stats[1, stat.MISS] = 6
    cache.lru_stats[1, stat.CALLS] = 2
    cache.stat_pinned_hot_pairs.fill_(3)
    cache.stat_pinned_hot_total_pairs.fill_(4)
    cache.stat_hot_pairs.fill_(1)
    cache.stat_hot_total_pairs.fill_(2)

    stats = cache.disk_prefetch_stats(reset=True)
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size()
        for bank in sources.values()
    )
    assert stats["pinned_hot_pair_rate"] == pytest.approx(0.75)
    assert stats["pinned_missing_per_step"] == pytest.approx(3.0)
    assert stats["pinned_h2d_bytes_per_step"] == pytest.approx(3 * expert_bytes)
    assert stats["hot_pair_rate"] == pytest.approx(0.5)
    """Import the CPU kernel mirror with lightweight Triton and flashlib stubs."""

    def jit(fn=None, **_kwargs):
        return (lambda decorated: decorated) if fn is None else fn

    triton = ModuleType("triton")
    triton.jit = jit
    triton_language = ModuleType("triton.language")
    triton.language = triton_language
    slot_cache = ModuleType("flashlib.kernels.slot_cache")
    slot_cache.lru_ensure = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", triton_language)
    monkeypatch.setitem(sys.modules, "flashlib.kernels.slot_cache", slot_cache)
    monkeypatch.delitem(sys.modules, "freetoken.moe.offload_kernels", raising=False)
    module = importlib.import_module("freetoken.moe.offload_kernels")
    try:
        return module
    finally:
        sys.modules.pop("freetoken.moe.offload_kernels", None)


@pytest.mark.parametrize(
    ("hot_gib", "expected_ticks", "expected_interval"),
    [(48, 96, 20), (6, 12, 166)],
)
def test_auto_interval_arithmetic(hot_gib, expected_ticks, expected_interval):
    controller = HotAdaptIntervalController.create(
        "auto",
        hot_budget_bytes=hot_gib << 30,
        max_swap_bytes=512 << 20,
    )

    assert controller.fill_ticks == expected_ticks
    assert controller.fill_interval == expected_interval
    assert controller.current_interval == expected_interval
    assert controller.steady_interval == 1000


def test_auto_interval_switches_to_steady_after_full_rerank():
    controller = HotAdaptIntervalController.create(
        "auto", hot_budget_bytes=48 << 30, max_swap_bytes=512 << 20,
    )

    switched, backed_off, _ = controller.complete_tick(
        partition_full=False,
        tick_interval=controller.current_interval,
        staging_seconds=0.0,
        covered_seconds=1.0,
    )
    assert not switched
    assert not backed_off
    assert controller.current_interval == 20

    switched, backed_off, _ = controller.complete_tick(
        partition_full=True,
        tick_interval=controller.current_interval,
        staging_seconds=0.0,
        covered_seconds=1.0,
    )
    assert switched
    assert not backed_off
    assert controller.current_interval == 1000

    switched, _, _ = controller.complete_tick(
        partition_full=True,
        tick_interval=controller.current_interval,
        staging_seconds=0.0,
        covered_seconds=1.0,
    )
    assert not switched


def test_auto_interval_backs_off_when_staging_exceeds_wall_fraction():
    controller = HotAdaptIntervalController.create(
        "auto", hot_budget_bytes=48 << 30, max_swap_bytes=512 << 20,
    )
    clock = HotAdaptTokenClock(controller.current_interval)
    clock.advance(controller.current_interval)
    clock.consume_tick()

    switched, backed_off, backoff_interval = controller.complete_tick(
        partition_full=False,
        tick_interval=controller.current_interval,
        staging_seconds=HOT_ADAPT_MAX_STAGING_FRACTION + 0.01,
        covered_seconds=1.0,
    )

    assert not switched
    assert backed_off
    assert backoff_interval == 40
    assert controller.current_interval == 40
    assert controller.current_interval >= controller.fill_interval
    clock.set_interval(controller.current_interval)
    assert clock.next_tick_token == 60


def test_boundary_backoff_uses_actual_aggregate_staged_bytes():
    controller = HotAdaptIntervalController.create(
        "auto", hot_budget_bytes=8 << 30, max_swap_bytes=512 << 20,
    )
    boundary_interval = hot_boundary_interval_tokens(
        controller.current_interval,
        512 << 20,
        2 << 30,
    )

    _, backed_off, backoff_interval = controller.complete_tick(
        partition_full=False,
        tick_interval=boundary_interval,
        staging_seconds=0.3,
        covered_seconds=1.0,
    )

    assert boundary_interval == 500
    assert backed_off
    assert backoff_interval == 1000


def test_explicit_interval_bypasses_auto_transition_and_backoff():
    controller = HotAdaptIntervalController.create(
        37, hot_budget_bytes=48 << 30, max_swap_bytes=512 << 20,
    )

    switched, backed_off, _ = controller.complete_tick(
        partition_full=True,
        tick_interval=controller.current_interval,
        staging_seconds=1.0,
        covered_seconds=1.0,
    )

    assert not controller.auto
    assert not switched
    assert not backed_off
    assert controller.current_interval == 37


def test_tick_clock_counts_prefill_tokens_and_decode_batch_members():
    clock = HotAdaptTokenClock(10)

    assert clock.advance(6) == 0
    assert clock.advance(4) == 1
    assert clock.routed_tokens == 10
    assert clock.consume_tick() == 10
    assert clock.advance(10) == 1
    assert clock.routed_tokens == 20


def test_idle_trigger_checks_delay_changed_counters_and_minimum_interval():
    tracker = HotAdaptIdleTracker(
        idle_seconds=0.5,
        min_interval_seconds=2.0,
    )
    tracker.note_routed_pairs()
    tracker.begin_idle(10.0)

    assert tracker.seconds_until_due(10.0) == pytest.approx(0.5)
    assert not tracker.due(10.499)
    assert tracker.due(10.5)

    tracker.tick_started()
    tracker.tick_completed(10.6, swaps=1)
    assert tracker.seconds_until_due(11.0) == pytest.approx(1.6)
    assert not tracker.due(12.599)
    assert tracker.due(12.6)


def test_idle_trigger_stays_off_without_changed_counters_or_prior_swap():
    tracker = HotAdaptIdleTracker(
        idle_seconds=0.5,
        min_interval_seconds=2.0,
    )
    tracker.begin_idle(10.0)

    assert not tracker.due(20.0)

    tracker.note_routed_pairs()
    tracker.tick_started()
    tracker.tick_completed(20.0, swaps=0)
    assert not tracker.due(30.0)


def test_auto_clock_makes_fill_ticks_due_at_one_2000_token_boundary():
    controller = HotAdaptIntervalController.create(
        "auto",
        hot_budget_bytes=8,
        max_swap_bytes=1,
    )
    clock = HotAdaptTokenClock(controller.current_interval)

    assert controller.fill_ticks == 8
    assert controller.fill_interval == 250
    assert clock.advance(2000) == 8
    for tick in range(8):
        assert clock.consume_tick() == (tick + 1) * 250

    assert clock.routed_tokens == 2000
    assert not controller.fill_complete


def test_prefill_boundary_staging_is_capped_to_hot_budget_fraction():
    counts = {0: tuple(range(8, 0, -1))}
    owners = {0: (None,) * 8}
    desired = {0: tuple(range(8))}

    tick_count = 8
    max_swap_bytes = 2
    expert_bytes = 2
    hot_budget_bytes = 16
    boundary_cap_frac = 0.5
    catchup_bytes = hot_catchup_swap_bytes(
        max_swap_bytes, expert_bytes, tick_count,
        hot_budget_bytes=hot_budget_bytes,
        boundary_cap_frac=boundary_cap_frac,
    )
    swaps = plan_hot_swaps(
        counts,
        owners,
        desired,
        expert_bytes=expert_bytes,
        max_swap_bytes=catchup_bytes,
    )

    assert len(swaps) == 4
    assert len(swaps) * expert_bytes <= hot_budget_bytes * boundary_cap_frac
    assert catchup_bytes < tick_count * max_swap_bytes
    assert hot_catchup_swap_bytes(
        1,
        expert_bytes,
        tick_count,
        hot_budget_bytes=hot_budget_bytes,
        boundary_cap_frac=boundary_cap_frac,
    ) == 0


def test_boundary_cap_preserves_one_row_partition_progress():
    assert hot_catchup_swap_bytes(
        4,
        4,
        1,
        hot_budget_bytes=4,
        boundary_cap_frac=0.5,
    ) == 4


def test_prefill_run_swap_budget_is_row_aligned_and_optional():
    assert prefill_run_swap_budget(
        24, 4, 20, hot_budget_bytes=32, run_cap_frac=0.0
    ) == 24
    assert prefill_run_swap_budget(
        24, 4, 4, hot_budget_bytes=32, run_cap_frac=0.5
    ) == 12
    assert prefill_run_swap_budget(
        24, 4, 15, hot_budget_bytes=32, run_cap_frac=0.5
    ) == 0


def test_hot_cpu_counter_route_weight_and_default(monkeypatch):
    import torch

    kernels = _offload_kernels_without_triton(monkeypatch)

    def make_cache():
        return SimpleNamespace(
            num_experts=4,
            cache_size=4,
            hot_row_for_expert=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            slot_for_id=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            id_of_slot=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            usage=torch.zeros(4, dtype=torch.int64),
            step=torch.zeros((), dtype=torch.int64),
            active_mask=torch.zeros(4, dtype=torch.int32),
            evict_slots=torch.empty(4, dtype=torch.int32),
            src_indices=torch.empty(4, dtype=torch.int32),
            num_indices=torch.zeros(1, dtype=torch.int64),
            num_missing_full=torch.zeros(1, dtype=torch.int64),
            stat_hot_pairs=torch.zeros((), dtype=torch.int64),
            stat_hot_total_pairs=torch.zeros((), dtype=torch.int64),
            hot_adapt_enabled=True,
            _hot_decay_factor=0.5,
            decayed_decode_freq=torch.full((1, 4), 2.0),
        )

    weighted = make_cache()
    kernels.ensure_experts_hot(
        weighted,
        0,
        torch.tensor([[0, 1], [1, 3]], dtype=torch.int32),
        route_weight=0.25,
    )
    assert weighted.decayed_decode_freq[0].tolist() == pytest.approx(
        [1.25, 1.5, 1.0, 1.25]
    )

    default = make_cache()
    kernels.ensure_experts_hot(
        default,
        0,
        torch.tensor([[0, 1], [1, 3]], dtype=torch.int32),
    )
    assert default.decayed_decode_freq[0].tolist() == pytest.approx(
        [2.0, 3.0, 1.0, 2.0]
    )


def test_boundary_routing_counts_prefill_decode_and_deferred_ticks(monkeypatch):
    from concurrent.futures import Future

    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    class PendingExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, fn, *args):
            self.calls.append((fn, args))
            return Future()

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.hot_adapt_enabled = True
    cache._hot_adapt_token_clock = HotAdaptTokenClock(4)
    cache._hot_adapt_interval_controller = HotAdaptIntervalController.create(
        4, hot_budget_bytes=16, max_swap_bytes=4,
    )
    cache.hot_adapt_interval_steps = 4
    cache.hot_adapt_ticks = 0
    cache.hot_adapt_ticks_prefill = 0
    cache.hot_adapt_ticks_decode = 0
    cache._hot_adapt_prefill_tokens_counted = 0
    cache._hot_adapt_future = None
    cache._hot_adapt_phase = None
    cache._hot_adapt_deferred_logged = False
    cache._hot_adapt_window_started_at = None
    cache._hot_adapt_snapshot_host = torch.zeros((1, 1))
    cache._hot_adapt_snapshot_device = torch.zeros((1, 1))
    cache.decayed_decode_freq = torch.zeros((1, 1))
    cache._hot_adapt_tick_interval_tokens = 0
    cache._hot_adapt_tick_staged_bytes = 0
    cache.device = torch.device("cpu")
    cache._hot_adapt_executor = PendingExecutor()
    cache._protect_hot_slots = lambda: None
    cache._boost_protected_slots = lambda: None
    cache._poll_hot_adaptation = lambda: None

    # No split counting path ran, so an ordinary prefill boundary is a no-op.
    cache.hot_adapt_prefill_boundary()
    assert cache.hot_adapt_routed_tokens == 0
    assert cache._hot_adapt_executor.calls == []

    cache.record_hot_adapt_prefill_tokens(10)
    cache.hot_adapt_prefill_boundary()
    assert cache.hot_adapt_routed_tokens == 10
    assert cache.hot_adapt_ticks == 2
    assert cache.hot_adapt_ticks_prefill == 2
    assert cache._hot_adapt_executor.calls[0][1][-2:] == ("prefill", 2)

    # One due decode tick is deferred while the prefill plan remains active.
    cache.hot_adapt_step_boundary(4)
    assert cache.hot_adapt_ticks_decode == 0

    # The next free boundary consumes both accumulated fixed-N thresholds.
    cache._hot_adapt_future = None
    cache._hot_adapt_phase = None
    cache.hot_adapt_step_boundary(2)
    assert cache.hot_adapt_routed_tokens == 16
    assert cache.hot_adapt_ticks == 4
    assert cache.hot_adapt_ticks_decode == 2
    assert cache._hot_adapt_executor.calls[1][1][-2:] == ("decode", 2)


def _post_prefill_boundary_cache(monkeypatch, *, enabled):
    from concurrent.futures import Future

    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    class PendingExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, fn, *args):
            self.calls.append((fn, args))
            return Future()

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.hot_adapt_enabled = True
    cache.hot_adapt_post_prefill_tick = enabled
    cache.hot_adapt_prefill_run_cap_frac = 0.0
    cache.hot_adapt_max_swap_bytes = 4
    cache._hot_adapt_token_clock = HotAdaptTokenClock(10)
    cache._hot_adapt_interval_controller = HotAdaptIntervalController.create(
        10, hot_budget_bytes=40, max_swap_bytes=4,
    )
    cache.hot_adapt_interval_steps = 10
    cache.hot_adapt_ticks = 0
    cache.hot_adapt_ticks_prefill = 0
    cache.hot_adapt_ticks_decode = 0
    cache._hot_adapt_prefill_tokens_counted = 0
    cache._hot_adapt_future = None
    cache._hot_adapt_phase = None
    cache._hot_adapt_deferred_logged = False
    cache._hot_adapt_window_started_at = None
    cache._hot_adapt_snapshot_host = torch.zeros((1, 1))
    cache._hot_adapt_snapshot_device = torch.zeros((1, 1))
    cache.decayed_decode_freq = torch.zeros((1, 1))
    cache._hot_adapt_tick_interval_tokens = 0
    cache._hot_adapt_tick_staged_bytes = 0
    cache.device = torch.device("cpu")
    cache._hot_adapt_executor = PendingExecutor()
    cache._protect_hot_slots = lambda: None
    cache._boost_protected_slots = lambda: None
    cache._poll_hot_adaptation = lambda: None
    return cache


def test_post_prefill_tick_starts_immediately_and_consumes_clock(monkeypatch):
    cache = _post_prefill_boundary_cache(monkeypatch, enabled=True)

    cache.record_hot_adapt_prefill_tokens(1)
    cache.hot_adapt_prefill_boundary()
    cache.hot_adapt_step_boundary(1)

    assert len(cache._hot_adapt_executor.calls) == 1
    assert cache._hot_adapt_executor.calls[0][1][-2:] == ("decode", 1)
    assert cache._hot_adapt_executor.calls[0][1][2] == 4
    assert cache.hot_adapt_ticks_decode == 1
    assert cache._hot_adapt_token_clock.next_tick_token == 12
    cache._hot_adapt_future = None
    cache._hot_adapt_phase = None
    cache.hot_adapt_step_boundary(9)
    assert len(cache._hot_adapt_executor.calls) == 1


def test_post_prefill_tick_flag_off_preserves_interval_wait(monkeypatch):
    cache = _post_prefill_boundary_cache(monkeypatch, enabled=False)

    cache.record_hot_adapt_prefill_tokens(1)
    cache.hot_adapt_prefill_boundary()
    cache.hot_adapt_step_boundary(1)

    assert cache._hot_adapt_executor.calls == []
    assert cache.hot_adapt_ticks_decode == 0


def test_post_prefill_tick_defers_while_work_is_in_flight(monkeypatch):
    from concurrent.futures import Future

    cache = _post_prefill_boundary_cache(monkeypatch, enabled=True)
    cache.record_hot_adapt_prefill_tokens(1)
    cache.hot_adapt_prefill_boundary()
    cache._hot_adapt_future = Future()
    cache._hot_adapt_phase = "plan"

    cache.hot_adapt_step_boundary(1)

    assert cache._hot_adapt_executor.calls == []
    assert cache._hot_adapt_after_prefill_pending
    assert cache.hot_adapt_ticks_decode == 0

    cache._hot_adapt_future = None
    cache._hot_adapt_phase = None
    cache.hot_adapt_step_boundary(1)
    assert len(cache._hot_adapt_executor.calls) == 1
    assert not cache._hot_adapt_after_prefill_pending


def test_hot_ensure_returns_token_rows_for_prefill_clock(monkeypatch):
    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    kernels = ModuleType("freetoken.moe.offload_kernels")
    kernels.ensure_experts_hot = lambda cache, layer_id, expert_ids, **kwargs: None
    monkeypatch.setitem(sys.modules, "freetoken.moe.offload_kernels", kernels)

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.hot_expert_capacity = {0: 1}
    cache.hot_adapt_enabled = True
    cache._hot_adapt_prefill_tokens_counted = 0
    cache.record_decode_frequency = lambda layer_id, expert_ids: None
    cache._protect_hot_slots = lambda: None

    counted = cache.ensure_experts_hot(
        0, torch.zeros((2048, 2), dtype=torch.int32)
    )
    cache.record_hot_adapt_prefill_tokens(counted)

    assert counted == 2048
    assert cache._hot_adapt_prefill_tokens_counted == 2048


def test_explicit_interval_remains_fixed_in_shared_token_units():
    controller = HotAdaptIntervalController.create(
        7, hot_budget_bytes=8, max_swap_bytes=1,
    )
    clock = HotAdaptTokenClock(controller.current_interval)

    assert clock.advance(6) == 0
    assert clock.advance(1) == 1
    clock.consume_tick()
    controller.complete_tick(
        partition_full=True,
        tick_interval=clock.interval,
        staging_seconds=1.0,
        covered_seconds=1.0,
    )
    clock.set_interval(controller.current_interval)
    assert clock.advance(6) == 0
    assert clock.advance(1) == 1
    assert controller.current_interval == 7
    assert clock.interval == 7


def test_staging_geometry_is_bounded_by_swap_delta_plus_headroom():
    expert_bytes = 13 << 20
    max_swap_bytes = 512 << 20
    rows = hot_staging_rows(max_swap_bytes, expert_bytes)

    assert rows * expert_bytes <= max_swap_bytes
    assert hot_staging_budget_bytes(max_swap_bytes) == (
        max_swap_bytes + HOT_STAGING_HEADROOM_BYTES
    )
    assert hot_staging_rows(expert_bytes - 1, expert_bytes) == 1


def test_decayed_counter_reaches_one_half_after_one_half_life():
    counts = update_decayed_counts(
        (0.0, 0.0), (1.0, 0.0), half_life_steps=4,
    )
    for _ in range(4):
        counts = update_decayed_counts(
            counts, (0.0, 0.0), half_life_steps=4,
        )
    assert counts == pytest.approx((0.5, 0.0))


def test_decayed_counter_accumulates_new_routes_after_decay():
    assert update_decayed_counts(
        (8.0, 4.0), (1.0, 3.0), half_life_steps=2, elapsed_steps=2,
    ) == pytest.approx((5.0, 5.0))


def test_partition_recompute_uses_same_equal_per_layer_byte_budget():
    counts = {0: (9.0, 1.0, 8.0), 2: (2.0, 7.0, 7.0)}
    assert recompute_hot_partition(
        counts,
        frozenset({0, 2}),
        budget_bytes=499,
        expert_bytes=100,
        num_experts=3,
    ) == {0: (0, 2), 2: (1, 2)}


def test_swap_planner_honors_global_byte_bound_and_prioritizes_gain():
    counts = {0: (1.0, 10.0, 9.0), 1: (1.0, 8.0, 7.0)}
    owners = {0: (0, None), 1: (0, None)}
    desired = {0: (1, 2), 1: (1, 2)}
    swaps = plan_hot_swaps(
        counts, owners, desired, expert_bytes=100, max_swap_bytes=200,
    )
    assert len(swaps) == 2
    assert len(swaps) * 100 <= 200
    assert {(swap.layer_id, swap.incoming_expert) for swap in swaps} == {(0, 1), (0, 2)}
    assert plan_hot_swaps(
        counts, owners, desired, expert_bytes=100, max_swap_bytes=99,
    ) == ()


def test_torn_mapping_guard_requires_copy_ack_before_publish():
    mapping = [[0, -1, -1]]
    swap = HotSwap(layer_id=0, row=0, incoming_expert=1, outgoing_expert=0)
    retired = retire_hot_swaps(mapping, (swap,))
    assert retired == [[-1, -1, -1]]

    with pytest.raises(RuntimeError, match="before copy"):
        finish_hot_swaps(retired, (swap,), copied_rows=set())

    assert finish_hot_swaps(
        retired, (swap,), copied_rows={(0, 0)}
    ) == [[-1, 0, -1]]


def test_worker_installs_into_inference_mode_allocated_bank_cache(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.device = torch.device("cpu")
    cache.bank_schema = ("gate_up",)
    cache._hot_slot_for_row = {0: (1,)}
    cache._hot_staging = [torch.tensor([[7, 8]], dtype=torch.int32)]
    with torch.inference_mode():
        cache.bank_caches = {
            "gate_up": torch.zeros((2, 2), dtype=torch.int32),
        }

    swap = HotSwap(0, 0, incoming_expert=1, outgoing_expert=None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(cache._install_staged_hot_rows, (swap,)).result(timeout=5)

    assert cache.bank_caches["gate_up"].tolist() == [[0, 0], [7, 8]]


def test_synthetic_banks_retire_stage_copy_and_flip_without_host_mirror(monkeypatch):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    assert "freetoken.moe.offload_cache" not in sys.modules
    offload_kernels = ModuleType("freetoken.moe.offload_kernels")

    def reset_cache(cache):
        cache.slot_for_id.fill_(-1)
        cache.id_of_slot.fill_(-1)
        cache.usage.zero_()
        cache.step.zero_()
        cache.active_mask.zero_()
        cache.num_indices.zero_()

    offload_kernels.reset_cache = reset_cache
    monkeypatch.setitem(
        sys.modules, "freetoken.moe.offload_kernels", offload_kernels
    )

    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2) + 100],
    }
    expert_bytes = sum(bank[0][0].numel() * bank[0].element_size() for bank in sources.values())
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"),
        prefill_overlap=False, decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (0, 2)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps=1,
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
    )
    try:
        assert cache._hot_adapt_tick_interval_tokens == 1
        assert not getattr(cache, "hot_bank_sources", {})
        assert sum(t.numel() * t.element_size() for t in cache._hot_staging) <= expert_bytes
        assert getattr(cache, "hot_staging_bytes", 0) <= (
            expert_bytes + HOT_STAGING_HEADROOM_BYTES
        )
        old_slot = cache._hot_slot_for_row[0][0]
        assert torch.equal(cache.bank_caches["gate_up"][old_slot], sources["gate_up"][0][0])

        swap = HotSwap(0, 0, incoming_expert=1, outgoing_expert=0)
        cache._retire_hot_adaptation_swaps((swap,))
        cache._hot_adapt_future.result(timeout=5)
        cache._poll_hot_adaptation()

        assert cache.hot_row_for_expert[0].tolist() == [-1, 0, 1, -1]
        assert cache.slot_for_id[0, 0].item() == -1
        assert cache.slot_for_id[0, 1].item() == old_slot
        assert torch.equal(cache.bank_caches["gate_up"][old_slot], sources["gate_up"][0][1])
        assert torch.equal(cache.bank_caches["down"][old_slot], sources["down"][0][1])

        cache.reset()
        assert cache.slot_for_id[0, 1].item() == old_slot
        assert cache.usage[old_slot].item() == torch.iinfo(torch.int64).max
        assert torch.equal(cache.bank_caches["gate_up"][old_slot], sources["gate_up"][0][1])
    finally:
        cache.shutdown_hot_adaptation()


def _idle_test_cache(
    monkeypatch,
    *,
    max_swap_rows=1,
    seeded=(),
    hot_capacity=2,
    interval_steps="auto",
    idle_ms=1,
    configure_logs=None,
    **configure_kwargs,
):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    if configure_logs is not None:
        logger = OffloadMoeCache.configure_hot_adaptation.__globals__["logger"]
        monkeypatch.setattr(logger, "info_rank0", configure_logs.append)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2) + 100],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size()
        for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4 + hot_capacity,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: seeded},
        hot_expert_capacity={0: hot_capacity},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps=interval_steps,
        max_swap_bytes=max_swap_rows * expert_bytes,
        expert_bytes=expert_bytes,
        boundary_cap_frac=1.0,
        idle_ms=idle_ms,
        idle_min_interval_ms=0,
        **configure_kwargs,
    )
    return cache


def test_prefill_run_cap_limits_consecutive_boundaries_and_decode_resets(
    monkeypatch,
):
    cache = _idle_test_cache(
        monkeypatch,
        hot_capacity=4,
        interval_steps=1,
        idle_ms=0,
        prefill_run_cap_frac=0.5,
    )

    def finish_tick():
        while cache._hot_adapt_future is not None:
            cache._hot_adapt_future.result(timeout=5)
            cache._poll_hot_adaptation()

    try:
        cache.decayed_decode_freq[0].copy_(
            cache.decayed_decode_freq.new_tensor([4.0, 3.0, 2.0, 1.0])
        )
        for expected_swaps in (1, 2):
            cache.record_hot_adapt_prefill_tokens(1)
            cache.hot_adapt_prefill_boundary()
            finish_tick()
            assert cache._hot_adapt_prefill_run_swaps == expected_swaps

        cache.record_hot_adapt_prefill_tokens(1)
        cache.hot_adapt_prefill_boundary()
        assert cache._hot_adapt_future is None
        assert cache.hot_adapt_ticks_prefill == 3
        assert cache._hot_adapt_prefill_run_swaps == 2
        assert cache._hot_adapt_prefill_run_swapped_bytes == (
            2 * cache.hot_adapt_expert_bytes
        )

        cache.hot_adapt_step_boundary(0)
        assert cache._hot_adapt_prefill_run_swapped_bytes == 0
        assert cache._hot_adapt_prefill_run_swaps == 2
    finally:
        cache.shutdown_hot_adaptation()


def test_idle_ticks_fill_partition_without_advancing_token_clock(monkeypatch):
    cache = _idle_test_cache(monkeypatch)
    try:
        cache.decayed_decode_freq[0].copy_(
            cache.decayed_decode_freq.new_tensor([1.0, 2.0, 9.0, 8.0])
        )
        cache._hot_adapt_idle_tracker.note_routed_pairs()
        cache._hot_adapt_interval_controller.steady_interval = 2000
        clock = cache._hot_adapt_token_clock
        before = (
            clock.interval,
            clock.routed_tokens,
            clock.next_tick_token,
            clock.last_tick_token,
        )

        cache.hot_adapt_while_idle(lambda: False)

        after = (
            clock.interval,
            clock.routed_tokens,
            clock.next_tick_token,
            clock.last_tick_token,
        )
        assert after == before
        assert cache._hot_adapt_interval_controller.fill_complete
        assert cache.hot_adapt_interval_steps == 2000
        assert cache.hot_expert_ids == {0: (2, 3)}
        assert cache.hot_adapt_ticks_idle == 3
        assert cache.hot_adapt_ticks == 3
        assert cache.hot_adapt_ticks_prefill == 0
        assert cache.hot_adapt_ticks_decode == 0
        assert cache._hot_plan_last_published_owners == {0: (2, 3)}
    finally:
        cache.shutdown_hot_adaptation()


def test_idle_hook_does_not_tick_when_counters_have_not_changed(monkeypatch):
    cache = _idle_test_cache(monkeypatch)
    try:
        before = cache.hot_adapt_routed_tokens
        checkpoint_before = dict(cache._hot_plan_last_published_owners)

        cache.hot_adapt_while_idle(lambda: False)

        assert cache.hot_adapt_ticks == 0
        assert cache.hot_adapt_ticks_idle == 0
        assert cache.hot_adapt_routed_tokens == before
        assert cache._hot_plan_last_published_owners == checkpoint_before
    finally:
        cache.shutdown_hot_adaptation()


def test_idle_delay_zero_disables_idle_ticks(monkeypatch):
    cache = _idle_test_cache(monkeypatch, idle_ms=0)
    try:
        cache.decayed_decode_freq[0, 3] = 10.0

        cache.hot_adapt_while_idle(lambda: False)

        assert cache._hot_adapt_idle_tracker is None
        assert cache.hot_adapt_ticks == 0
        assert cache.hot_adapt_ticks_idle == 0
        assert cache.hot_expert_ids == {0: ()}
    finally:
        cache.shutdown_hot_adaptation()


def test_tensor_parallelism_disables_idle_ticks_and_logs_staging_bound(monkeypatch):
    logs = []
    cache = _idle_test_cache(
        monkeypatch,
        idle_ms=500,
        tp_size=2,
        configure_logs=logs,
    )
    try:
        assert cache._hot_adapt_idle_tracker is None
        assert any("hot_staging_rows=1" in message for message in logs)
        assert any("idle=off (tensor parallel)" in message for message in logs)
    finally:
        cache.shutdown_hot_adaptation()


def test_zero_swap_idle_tick_checkpoints_published_owners(monkeypatch):
    cache = _idle_test_cache(monkeypatch, seeded=(2, 3))
    checkpoints = []
    try:
        cache.decayed_decode_freq[0].copy_(
            cache.decayed_decode_freq.new_tensor([1.0, 2.0, 9.0, 8.0])
        )
        cache._hot_adapt_idle_tracker.note_routed_pairs()
        monkeypatch.setattr(
            cache,
            "_checkpoint_published_hot_slot_owners",
            lambda: checkpoints.append(tuple(cache._hot_slot_owners[0])),
        )

        cache.hot_adapt_while_idle(lambda: False)

        assert cache.hot_adapt_ticks_idle == 1
        assert cache.hot_adapt_swaps == 0
        assert checkpoints == [(2, 3)]
    finally:
        cache.shutdown_hot_adaptation()


def test_idle_hook_polls_completed_decode_tick_before_evidence_check(monkeypatch):
    from concurrent.futures import Future

    cache = _idle_test_cache(monkeypatch, seeded=(2, 3))
    try:
        completed = Future()
        completed.set_result(((), 1.0, 1))
        cache._hot_adapt_future = completed
        cache._hot_adapt_phase = "plan"
        cache._hot_adapt_tick_boundary = "decode"

        cache.hot_adapt_while_idle(lambda: False)

        assert cache._hot_adapt_future is None
        assert cache._hot_adapt_phase is None
    finally:
        cache.shutdown_hot_adaptation()


def test_cancelled_idle_plan_is_preempted_without_counting_tick(monkeypatch):
    from concurrent.futures import Future

    cache = _idle_test_cache(monkeypatch)
    logs = []
    try:
        cache._hot_adapt_future = Future()
        cache._hot_adapt_phase = "plan"
        cache._hot_adapt_tick_boundary = "idle"
        cache.hot_adapt_ticks = 1
        cache.hot_adapt_ticks_idle = 1
        monkeypatch.setattr(
            "freetoken.moe.offload_cache.logger.info_rank0", logs.append
        )

        cache._finish_preempted_idle_tick()

        assert cache.hot_adapt_ticks == 0
        assert cache.hot_adapt_ticks_idle == 0
        assert cache._hot_adapt_future is None
        assert cache._hot_adapt_phase is None
        assert logs == [
            "MoE HOT adaptation idle tick token=0: preempted before planning"
        ]
    finally:
        cache.shutdown_hot_adaptation()


def test_running_idle_plan_is_preempted_without_counting_tick(monkeypatch):
    from concurrent.futures import Future

    cache = _idle_test_cache(monkeypatch)
    logs = []
    try:
        completed = Future()
        completed.set_result(
            ((HotSwap(0, 0, incoming_expert=1, outgoing_expert=None),), 0.0, 1)
        )
        cache._hot_adapt_future = completed
        cache._hot_adapt_phase = "plan"
        cache._hot_adapt_tick_boundary = "idle"
        cache.hot_adapt_ticks = 1
        cache.hot_adapt_ticks_idle = 1
        monkeypatch.setattr(
            "freetoken.moe.offload_cache.logger.info_rank0", logs.append
        )

        cache._finish_preempted_idle_tick()

        assert cache.hot_adapt_ticks == 0
        assert cache.hot_adapt_ticks_idle == 0
        assert cache.hot_adapt_swaps == 0
        assert cache._hot_adapt_future is None
        assert cache._hot_adapt_phase is None
        assert logs == [
            "MoE HOT adaptation idle tick token=0: preempted after planning"
        ]
    finally:
        cache.shutdown_hot_adaptation()


def test_shutdown_finishes_completed_idle_plan_without_logging_preemption(monkeypatch):
    from concurrent.futures import Future

    cache = _idle_test_cache(monkeypatch)
    logs = []
    try:
        completed = Future()
        completed.set_result(
            ((HotSwap(0, 0, incoming_expert=1, outgoing_expert=None),), 0.0, 1)
        )
        cache._hot_adapt_future = completed
        cache._hot_adapt_phase = "plan"
        cache._hot_adapt_tick_boundary = "idle"
        cache.hot_adapt_ticks = 1
        cache.hot_adapt_ticks_idle = 1
        cache._hot_adapt_stop_event.set()
        monkeypatch.setattr(
            "freetoken.moe.offload_cache.logger.info_rank0", logs.append
        )

        assert cache._drain_hot_adaptation(time.monotonic() + 5.0)

        assert cache.hot_adapt_ticks == 1
        assert cache.hot_adapt_ticks_idle == 1
        assert cache.hot_adapt_idle_swaps == 0
        assert cache._hot_adapt_future is None
        assert not any("preempted" in message for message in logs)
        assert any("executed_swaps=0" in message for message in logs)
    finally:
        cache.shutdown_hot_adaptation()


def test_request_arrival_wakes_bounded_idle_wait(monkeypatch):
    from threading import Event, Thread

    cache = _idle_test_cache(monkeypatch, idle_ms=500)
    request_arrived = Event()
    wait_started = Event()

    class FakeQueue:
        def empty(self):
            return not request_arrived.is_set()

        def wait_for_item(self, timeout_seconds):
            wait_started.set()
            return request_arrived.wait(timeout_seconds)

    queue = FakeQueue()
    cache._hot_adapt_idle_tracker.note_routed_pairs()

    def enqueue_request():
        assert wait_started.wait(timeout=1)
        time.sleep(0.02)
        request_arrived.set()

    producer = Thread(target=enqueue_request)
    producer.start()
    started_at = time.monotonic()
    try:
        cache.hot_adapt_while_idle(
            lambda: not queue.empty(), queue.wait_for_item
        )

        assert time.monotonic() - started_at < 0.25
        assert cache.hot_adapt_ticks_idle == 0
    finally:
        request_arrived.set()
        producer.join(timeout=1)
        cache.shutdown_hot_adaptation()


def test_keyboard_interrupt_stops_inflight_idle_staging_for_shutdown(monkeypatch):
    from threading import Event

    cache = _idle_test_cache(monkeypatch, seeded=(0, 1))
    row_started = Event()
    release_row = Event()

    def controlled_stage(ready, swaps, stop_event=None):
        if ready is not None:
            ready.synchronize()
        row_started.set()
        assert release_row.wait(timeout=5)
        return set(), 0.0

    def interrupt_after_tick_starts(timeout_seconds):
        if cache._hot_adapt_phase != "copy":
            return Event().wait(timeout_seconds)
        assert row_started.wait(timeout=5)
        raise KeyboardInterrupt

    cache._stage_hot_rows = controlled_stage
    cache.decayed_decode_freq[0].copy_(
        cache.decayed_decode_freq.new_tensor([1.0, 2.0, 9.0, 8.0])
    )
    cache._hot_adapt_idle_tracker.note_routed_pairs()
    try:
        with pytest.raises(KeyboardInterrupt):
            cache.hot_adapt_while_idle(
                lambda: False, interrupt_after_tick_starts
            )

        assert cache._hot_adapt_stop_event.is_set()
        assert cache._hot_adapt_future is not None
    finally:
        release_row.set()
        cache.shutdown_hot_adaptation()


def test_idle_swap_queues_interval_gated_plan_snapshot(monkeypatch, tmp_path):
    import json

    path = tmp_path / "freetoken_hot_plan.json"
    cache = _idle_test_cache(
        monkeypatch,
        seeded=(0, 1),
        hot_plan_path=str(path),
        hot_plan_identity={"kind": "ftw", "path": "/model", "shards": []},
        hot_plan_tier_commit="tier-test",
        hot_plan_write_enabled=True,
        hot_plan_interval_seconds=60.0,
    )
    try:
        cache.decayed_decode_freq[0].copy_(
            cache.decayed_decode_freq.new_tensor([8.0, 0.0, 9.0, 0.0])
        )
        cache._hot_plan_last_snapshot -= 61.0
        cache._hot_adapt_idle_tracker.note_routed_pairs()

        cache.hot_adapt_while_idle(lambda: False)

        assert cache.hot_adapt_idle_swaps == 1
        assert cache._hot_plan_future is not None
        assert cache._hot_plan_future.result(timeout=5)
        cache._collect_finished_hot_plan_write()
        document = json.loads(path.read_text())
        assert document["protected_slots"] == {"0": [0, 2]}
    finally:
        cache.shutdown_hot_adaptation()


def test_idle_swap_respects_disabled_plan_persistence(monkeypatch, tmp_path):
    path = tmp_path / "freetoken_hot_plan.json"
    cache = _idle_test_cache(
        monkeypatch,
        seeded=(0, 1),
        hot_plan_path=str(path),
        hot_plan_identity={"kind": "ftw", "path": "/model", "shards": []},
        hot_plan_write_enabled=False,
    )
    try:
        cache.decayed_decode_freq[0].copy_(
            cache.decayed_decode_freq.new_tensor([8.0, 0.0, 9.0, 0.0])
        )
        cache._hot_adapt_idle_tracker.note_routed_pairs()

        cache.hot_adapt_while_idle(lambda: False)

        assert cache.hot_adapt_idle_swaps == 1
        assert cache._hot_plan_future is None
        assert not path.exists()
    finally:
        cache.shutdown_hot_adaptation()


def test_new_request_abandons_idle_tick_at_next_row_boundary(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    cache = _idle_test_cache(monkeypatch, max_swap_rows=2, seeded=(0, 1))
    row_started = Event()
    release_row = Event()
    request_arrived = Event()

    def controlled_stage(ready, swaps, stop_event=None):
        if ready is not None:
            ready.synchronize()
        copied = set()
        started_at = 0.0
        for stage_row, swap in enumerate(swaps):
            if stop_event is not None and stop_event.is_set():
                break
            for bank_id, name in enumerate(cache.bank_schema):
                source = cache.bank_sources[name][swap.layer_id]
                cache._hot_staging[bank_id][stage_row].copy_(
                    source[swap.incoming_expert]
                )
            if stage_row == 0:
                row_started.set()
                assert release_row.wait(timeout=5)
            copied.add((swap.layer_id, swap.row))
        return copied, started_at

    cache._stage_hot_rows = controlled_stage
    cache.decayed_decode_freq[0].copy_(
        cache.decayed_decode_freq.new_tensor([1.0, 2.0, 9.0, 8.0])
    )
    cache._hot_adapt_idle_tracker.note_routed_pairs()
    clock_before = cache.hot_adapt_routed_tokens
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            idle = executor.submit(
                cache.hot_adapt_while_idle, request_arrived.is_set
            )
            assert row_started.wait(timeout=5)
            request_arrived.set()
            assert cache._hot_adapt_stop_event.wait(timeout=5)
            release_row.set()
            idle.result(timeout=5)

        assert cache.hot_adapt_ticks_idle == 1
        assert cache.hot_adapt_swaps == 0
        assert cache.hot_adapt_idle_swaps == 1
        assert cache._hot_slot_owners[0] == [2, 1]
        assert cache.hot_row_for_expert[0].tolist() == [-1, 1, 0, -1]
        assert cache.hot_expert_ids == {0: (1, 2)}
        assert cache._hot_adapt_future is None
        assert cache.hot_adapt_routed_tokens == clock_before
    finally:
        release_row.set()
        cache.shutdown_hot_adaptation()


@pytest.mark.cuda
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_cuda_idle_tick_publishes_without_advancing_token_clock(monkeypatch):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2)],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size()
        for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cuda"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: ()},
        hot_expert_capacity={0: 1},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps=1000,
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
        idle_ms=1,
        idle_min_interval_ms=0,
    )
    try:
        assert cache._hot_adapt_idle_tracker is not None
        cache.decayed_decode_freq[0, 3] = 10.0
        cache._hot_adapt_idle_tracker.note_routed_pairs()
        token_before = cache.hot_adapt_routed_tokens

        cache.hot_adapt_while_idle(lambda: False)

        assert cache.hot_expert_ids == {0: (3,)}
        assert cache.hot_adapt_routed_tokens == token_before
    finally:
        cache.shutdown_hot_adaptation()


def test_rebuild_drains_hot_adaptation_before_replacing_bank_caches(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2) + 100],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size() for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (0, 2)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps=1,
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
    )
    old_bank_caches = cache.bank_caches
    drain_started = Event()
    release = Event()

    class ControlledFuture:
        def cancel(self):
            return False

        def done(self):
            return True

        def result(self, timeout=None):
            drain_started.set()
            assert cache._hot_adapt_stop_event.is_set()
            assert release.wait(timeout=5)
            assert cache.bank_caches is old_bank_caches
            for bank_id, name in enumerate(cache.bank_schema):
                cache._hot_staging[bank_id][0].copy_(sources[name][0][1])
            return {(0, 0)}, 0.0

    cache._hot_adapt_future = ControlledFuture()
    cache._hot_adapt_phase = "copy"
    cache._hot_adapt_swaps_pending = (
        HotSwap(0, 0, incoming_expert=1, outgoing_expert=0),
    )
    cache._hot_adapt_worker_installs = True
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            rebuilt = executor.submit(cache.rebuild, 7)
            assert drain_started.wait(timeout=5)
            assert cache.bank_caches is old_bank_caches
            release.set()
            rebuilt.result(timeout=5)

        assert cache.bank_caches is not old_bank_caches
        assert cache._hot_adapt_future is None
        assert cache._hot_adapt_phase is None
        assert cache._hot_adapt_swaps_pending == ()
        assert not cache._hot_adapt_worker_installs
        for row, expert in enumerate(cache._hot_slot_owners[0]):
            slot = cache._hot_slot_for_row[0][row]
            assert torch.equal(
                cache.bank_caches["gate_up"][slot], sources["gate_up"][0][expert]
            )
    finally:
        release.set()
        cache.shutdown_hot_adaptation()


@pytest.mark.parametrize("already_completed", [False, True])
def test_rebuild_drain_discards_plan_without_starting_copy(
    monkeypatch, already_completed
):
    from concurrent.futures import Future
    from threading import Event

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    future = Future()
    if already_completed:
        future.set_result(((object(),), 0.5, 1))
    cache._hot_adapt_future = future
    cache._hot_adapt_phase = "plan"
    cache._hot_adapt_stop_event = Event()
    cache._hot_adapt_tick_executed_swaps = -1
    cache._hot_adapt_stop_wait_seconds = lambda: 1.0
    submitted_copies = []
    cache._retire_hot_adaptation_swaps = submitted_copies.append

    cache.drain_hot_adaptation_for_rebuild()

    assert future.cancelled() is not already_completed
    assert submitted_copies == []
    assert cache._hot_adapt_future is None
    assert cache._hot_adapt_phase is None
    assert cache._hot_adapt_tick_executed_swaps == 0
    assert not cache._hot_adapt_stop_event.is_set()


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="needs Triton import"
)
def test_engine_rebuild_preserves_hot_state_through_prevalidation(monkeypatch):
    import torch

    from freetoken.engine import engine as engine_module
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.base import CacheRebuildRejected

    events = []
    cache = SimpleNamespace(
        hot_expert_capacity={0: 2},
        _hot_staging=[],
        validate_rebuild=lambda size: events.append(("validate_moe", size)),
        drain_hot_adaptation_for_rebuild=lambda: events.append("drain_hot"),
        rebuild=lambda size, **kwargs: events.append(("rebuild_moe", size, kwargs)),
    )
    kv_cache = SimpleNamespace(
        needs_rebind_on_rebuild=False,
        validate_rebuild=lambda *_args, **_kwargs: events.append("validate_kv"),
        attach_page_table=lambda _table: events.append("attach_page_table"),
    )
    config = SimpleNamespace(
        page_size=1,
        max_seq_len=32,
        max_running_req=1,
        cuda_graph_max_bs=1,
        model_config=SimpleNamespace(vocab_size=16),
    )
    engine = Engine.__new__(Engine)
    engine.config = config
    engine.moe_offload_cache = cache
    engine.kv_cache = kv_cache
    engine.linear_state_pool = None
    engine._baseline_free = 1_000
    engine._weights_bytes = 0
    engine._target_moe_and_expert_bytes = lambda _size: (6, 1)
    engine.device = torch.device("cpu")
    engine.num_pages = 32
    engine.page_table = torch.zeros((2, 32), dtype=torch.int32)
    engine.ctx = SimpleNamespace(page_table=engine.page_table)
    engine.dummy_req = SimpleNamespace(table_idx=0)
    engine.graph_runner = SimpleNamespace(
        graph_bs_list=[1],
        destroy_cuda_graphs=lambda: events.append("destroy_graphs"),
    )
    engine.attn_backend = SimpleNamespace(
        reset_capture=lambda: events.append("reset_capture")
    )
    engine.stream = None
    engine.model = SimpleNamespace()
    engine._sync_get_memory = lambda: (500, 500)
    engine.rebuild_teardown_started = False

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda _device=None: events.append("synchronize")
    )
    monkeypatch.setattr(
        engine_module,
        "GraphRunner",
        lambda **kwargs: SimpleNamespace(graph_bs_list=kwargs["cuda_graph_bs"]),
    )

    with pytest.raises(CacheRebuildRejected, match="pinned staging bank"):
        engine.rebuild_runtime_cache(moe_cache_size=6, preserve_hot_state=True)
    assert not engine.rebuild_teardown_started
    assert events == [("validate_moe", 6)]

    events.clear()
    cache._hot_staging = [object()]
    engine.rebuild_runtime_cache(moe_cache_size=6, preserve_hot_state=True)

    assert engine.rebuild_teardown_started
    assert events == [
        ("validate_moe", 6),
        "validate_kv",
        "drain_hot",
        "synchronize",
        "reset_capture",
        "destroy_graphs",
        ("rebuild_moe", 6, {"preserve_hot_state": True}),
        "attach_page_table",
    ]


def test_ladder_rebuild_preserves_hot_mapping_counters_and_plan(monkeypatch):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2) + 100],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size()
        for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 3)},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps=0,
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
    )
    try:
        cache.stat_hot_pairs.fill_(94)
        cache.stat_hot_total_pairs.fill_(100)
        cache.decayed_decode_freq.copy_(torch.tensor([[1.0, 7.0, 2.0, 9.0]]))
        cache.decode_freq.copy_(
            torch.tensor([[1, 7, 2, 9]], dtype=torch.int64)
        )
        cache._protected_route_baseline = [[0, 3, 0, 4]]
        owners_before = {
            layer_id: tuple(owners)
            for layer_id, owners in cache._hot_slot_owners.items()
        }
        plan_before = dict(cache._hot_plan_last_published_owners)
        decayed_before = cache.decayed_decode_freq.clone()
        decode_before = cache.decode_freq.clone()

        cache.rebuild(6, preserve_hot_state=True)

        assert {
            layer_id: tuple(owners)
            for layer_id, owners in cache._hot_slot_owners.items()
        } == owners_before
        for expert in (1, 3):
            slot = int(cache.slot_for_id[0, expert].item())
            assert slot >= 0
            for name in cache.bank_schema:
                assert torch.equal(
                    cache.bank_caches[name][slot],
                    sources[name][0][expert],
                )
        assert int(cache.stat_hot_pairs.item()) == 94
        assert int(cache.stat_hot_total_pairs.item()) == 100
        assert int(cache.stat_hot_pairs.item()) / int(
            cache.stat_hot_total_pairs.item()
        ) == 0.94
        assert torch.equal(cache.decayed_decode_freq, decayed_before)
        assert torch.equal(cache.decode_freq, decode_before)
        assert cache._protected_route_baseline == [[0, 3, 0, 4]]
        assert cache._hot_plan_last_published_owners == plan_before
    finally:
        cache.shutdown_hot_adaptation()


def test_persisted_counter_seed_survives_reset(monkeypatch):
    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    kernels = ModuleType("freetoken.moe.offload_kernels")
    kernels.reset_cache = lambda _cache: None
    monkeypatch.setitem(sys.modules, "freetoken.moe.offload_kernels", kernels)

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.device = torch.device("cpu")
    cache.decayed_decode_freq = torch.full((2, 3), 99.0)
    cache._hot_plan_counter_seed = {0: (1.0, 2.0, 3.0)}
    cache._restore_hot_slot_metadata = lambda: None
    cache.expert_recency = torch.zeros((2, 3), dtype=torch.int64)
    cache.session_profile_ids = None
    cache.cpu_executor = None

    cache.reset()

    assert cache.decayed_decode_freq[0].tolist() == [1.0, 2.0, 3.0]
    assert cache.decayed_decode_freq[1].tolist() == [0.0, 0.0, 0.0]


def test_fully_persisted_seed_starts_fill_complete(monkeypatch):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2)],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size() for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 3)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps="auto",
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
        persisted_counter_seed={0: (1.0, 9.0, 2.0, 8.0)},
        persisted_seeded_layers=frozenset({0}),
    )
    try:
        assert cache._hot_adapt_interval_controller.fill_complete
        assert cache.hot_adapt_interval_steps == 1000
        assert cache.decayed_decode_freq[0].tolist() == [1.0, 9.0, 2.0, 8.0]
    finally:
        cache.shutdown_hot_adaptation()


def test_missing_persisted_layer_keeps_fill_interval(monkeypatch):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [
            torch.arange(4 * 3, dtype=torch.int32).view(4, 3),
            torch.arange(4 * 3, dtype=torch.int32).view(4, 3) + 100,
        ],
        "down": [
            torch.arange(4 * 2, dtype=torch.int32).view(4, 2),
            torch.arange(4 * 2, dtype=torch.int32).view(4, 2) + 100,
        ],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size() for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0, 1})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value] * 2,
        hot_expert_ids={0: (1,), 1: (2,)},
        hot_expert_capacity={0: 1, 1: 1},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps="auto",
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
        persisted_counter_seed={0: (1.0, 9.0, 2.0, 8.0)},
        persisted_seeded_layers=frozenset({0}),
    )
    try:
        controller = cache._hot_adapt_interval_controller
        assert not controller.fill_complete
        assert cache.hot_adapt_interval_steps == controller.fill_interval
        assert cache.decayed_decode_freq[1].tolist() == [0.0] * 4
    finally:
        cache.shutdown_hot_adaptation()


def test_periodic_plan_snapshot_writes_from_background_worker(monkeypatch, tmp_path):
    import json
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2)],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size() for bank in sources.values()
    )
    identity = {"kind": "ftw", "path": "/model", "shards": []}
    path = tmp_path / "freetoken_hot_plan.json"
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 3)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps="auto",
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
        hot_plan_path=str(path),
        hot_plan_identity=identity,
        hot_plan_tier_commit="tier-test",
        hot_plan_write_enabled=True,
        hot_plan_interval_seconds=0.001,
    )
    try:
        cache.decayed_decode_freq[0].copy_(torch.tensor([1.0, 9.0, 2.0, 8.0]))
        future = cache.snapshot_hot_plan(force=True)
        assert future is not None
        assert future.result(timeout=5)
        cache._collect_finished_hot_plan_write()

        document = json.loads(path.read_text())
        assert document["protected_slots"] == {"0": [1, 3]}
        assert document["counter_dtype"] == "float32"
        assert document["tier_commit"] == "tier-test"
    finally:
        cache.shutdown_hot_adaptation()


def test_shutdown_survives_periodic_and_final_plan_write_oserrors(
    monkeypatch, tmp_path
):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2)],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size() for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 3)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps="auto",
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
        hot_plan_path=str(tmp_path / "freetoken_hot_plan.json"),
        hot_plan_identity={"kind": "ftw", "path": "/model", "shards": []},
        hot_plan_tier_commit="tier-test",
        hot_plan_write_enabled=True,
        hot_plan_interval_seconds=0.001,
    )
    cache.decayed_decode_freq[0].copy_(torch.tensor([1.0, 9.0, 2.0, 8.0]))
    writes = []
    warnings = []

    def fail_write(_path, _document, **_kwargs):
        writes.append(len(writes) + 1)
        raise OSError(f"write failure {writes[-1]}")

    method_globals = cache._write_hot_plan_snapshot.__globals__
    monkeypatch.setattr(method_globals["logger"], "warning_rank0", warnings.append)
    monkeypatch.setattr("freetoken.moe.hot_adapt.atomic_write_hot_plan", fail_write)

    periodic = cache.snapshot_hot_plan(force=True)
    assert periodic is not None
    assert isinstance(periodic.exception(timeout=5), OSError)

    cache.shutdown_hot_adaptation()

    assert writes == [1, 2]
    assert sum("MoE HOT plan write failed" in message for message in warnings) == 2
    assert cache._hot_adapt_executor is None
    assert cache._hot_plan_executor is None


def test_shutdown_fences_final_write_that_misses_deadline(monkeypatch, tmp_path):
    import threading
    import time

    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2)],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size() for bank in sources.values()
    )
    path = tmp_path / "freetoken_hot_plan.json"
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 3)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps="auto",
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
        hot_plan_path=str(path),
        hot_plan_identity={"kind": "ftw", "path": "/model", "shards": []},
        hot_plan_tier_commit="tier-test",
        hot_plan_write_enabled=True,
        hot_plan_interval_seconds=1.0,
    )
    cache.decayed_decode_freq[0].copy_(torch.tensor([1.0, 9.0, 2.0, 8.0]))
    cache._drain_hot_adaptation = lambda _deadline: False
    cache._hot_adapt_future = None
    entered_fsync = threading.Event()
    release_fsync = threading.Event()
    real_fsync = __import__("os").fsync

    def blocked_fsync(fd):
        entered_fsync.set()
        release_fsync.wait(timeout=2.0)
        real_fsync(fd)

    monkeypatch.setattr("freetoken.moe.hot_adapt.os.fsync", blocked_fsync)
    method_globals = cache.shutdown_hot_adaptation.__globals__
    monkeypatch.setitem(method_globals, "_HOT_PLAN_FINAL_WRITE_SECONDS", 0.05)
    warnings = []
    monkeypatch.setattr(method_globals["logger"], "warning_rank0", warnings.append)
    recorded_futures = []
    snapshot = cache.snapshot_hot_plan

    def record_snapshot(**kwargs):
        future = snapshot(**kwargs)
        recorded_futures.append(future)
        return future

    cache.snapshot_hot_plan = record_snapshot

    started = time.monotonic()
    cache.shutdown_hot_adaptation(timeout_seconds=0.01)
    elapsed = time.monotonic() - started
    assert entered_fsync.is_set()
    release_fsync.set()
    assert recorded_futures[-1].result(timeout=2.0) is False

    print(f"bounded final-write shutdown elapsed: {elapsed:.3f} s")
    assert elapsed < 0.25
    assert not path.exists()
    assert any(
        "attempted 2 experts across 1 layers; write did not confirm within 0.05 s"
        in message
        for message in warnings
    )


def test_hot_plan_fence_cancel_times_out_while_publish_holds_lock(monkeypatch):
    from concurrent.futures import Future
    import threading
    import time

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    method_globals = OffloadMoeCache._abandon_hot_plan_write.__globals__
    monkeypatch.setitem(method_globals, "_HOT_PLAN_FENCE_CANCEL_SECONDS", 0.02)
    fence = method_globals["_HotPlanWriteFence"]()
    write_future = Future()
    assert write_future.set_running_or_notify_cancel()
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache._hot_plan_write_fence = fence
    cache._hot_plan_future = write_future
    entered_replace = threading.Event()
    release_replace = threading.Event()
    published = []
    warnings = []

    def blocked_replace(_source, _target):
        entered_replace.set()
        release_replace.wait(timeout=2.0)

    monkeypatch.setattr(method_globals["os"], "replace", blocked_replace)
    monkeypatch.setattr(method_globals["logger"], "warning_rank0", warnings.append)
    publisher = threading.Thread(
        target=lambda: published.append(fence.publish("source", "target")),
        daemon=True,
    )
    publisher.start()
    assert entered_replace.wait(timeout=1.0)

    started = time.monotonic()
    try:
        cache._abandon_hot_plan_write(write_future)
        elapsed = time.monotonic() - started
    finally:
        release_replace.set()
        publisher.join(timeout=1.0)
        write_future.set_result(True)

    print(f"bounded HOT plan fence cancellation elapsed: {elapsed:.3f} s")
    assert elapsed < 0.2
    assert fence._cancelled
    assert published == [True]
    assert cache._hot_plan_write_fence is None
    assert cache._hot_plan_future is None
    assert any(
        "fence lock acquire timed out; abandoning write without confirmation"
        in message
        for message in warnings
    )


def test_ordinary_tick_forwards_stop_event_and_stops_between_rows(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache._hot_staging_rows = 2
    cache.bank_schema = ("gate_up",)
    cache.bank_sources = {"gate_up": [torch.tensor([[1], [2]])]}
    stop = threading.Event()
    cache._hot_adapt_stop_event = stop
    cache._hot_adapt_executor = ThreadPoolExecutor(max_workers=1)
    cache._hot_adapt_tick_interval_tokens = 1
    cache.hot_adapt_max_swap_bytes = 2
    cache.hot_adapt_expert_bytes = 1
    cache._hot_slot_owners = {0: [None, None]}
    cache.hot_expert_ids = {0: ()}
    cache.device = torch.device("cpu")
    cache._hot_mapping_lists = lambda: [[-1, -1]]
    cache._replace_hot_mapping = lambda _mapping: None

    class StagingRow:
        def copy_(self, _source):
            stop.set()

    class StagingBank:
        def __getitem__(self, _row):
            return StagingRow()

    cache._hot_staging = [StagingBank()]
    swaps = (
        HotSwap(0, 0, incoming_expert=0, outgoing_expert=None),
        HotSwap(0, 1, incoming_expert=1, outgoing_expert=None),
    )

    try:
        cache._retire_hot_adaptation_swaps(swaps, tick_count=1)
        copied, _elapsed = cache._hot_adapt_future.result(timeout=5)
    finally:
        cache._hot_adapt_executor.shutdown(wait=True)

    assert copied == {(0, 0)}


def test_shutdown_timeout_persists_last_published_slot_checkpoint(
    monkeypatch, tmp_path
):
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    import json
    import threading

    import torch

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    class FakeExecutor:
        def shutdown(self, **_kwargs):
            pass

    class TimedOutFuture:
        def result(self, timeout=None):
            raise FuturesTimeoutError

        def done(self):
            return False

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.cpu_executor = None
    cache._hot_plan_stop_event = threading.Event()
    cache._hot_adapt_stop_event = threading.Event()
    cache._hot_adapt_future = TimedOutFuture()
    cache._hot_adapt_phase = "copy"
    cache._hot_adapt_executor = FakeExecutor()
    cache._hot_adapt_stop_wait_seconds = lambda: 0.0
    cache._hot_slot_owners = {0: [None, 3]}
    cache._hot_plan_last_published_owners = {0: (1, 3)}
    cache._hot_plan_future = None
    cache._hot_plan_write_enabled = True
    from concurrent.futures import ThreadPoolExecutor

    cache._hot_plan_executor = ThreadPoolExecutor(max_workers=1)
    cache._hot_plan_path = str(tmp_path / "freetoken_hot_plan.json")
    cache._hot_plan_identity = {"kind": "ftw", "path": "/model", "shards": []}
    cache._hot_plan_tier_commit = "tier-test"
    cache._hot_plan_zero_logged = False
    cache._hot_plan_last_snapshot = 0.0
    cache._hot_plan_interval_seconds = 1.0
    cache.hot_expert_capacity = {0: 2}
    cache.num_layers = 1
    cache.num_experts = 4
    cache.hot_adapt_hot_budget_bytes = 100
    cache.decayed_decode_freq = torch.tensor([[1.0, 9.0, 2.0, 8.0]])
    warnings = []
    monkeypatch.setattr(
        cache.shutdown_hot_adaptation.__globals__["logger"],
        "warning_rank0",
        warnings.append,
    )

    cache.shutdown_hot_adaptation(timeout_seconds=0.0)

    document = json.loads((tmp_path / "freetoken_hot_plan.json").read_text())
    assert document["protected_slots"] == {"0": [1, 3]}
    assert any(
        "wrote the last published slot set containing 2 seeded experts from 1 layers"
        in message
        for message in warnings
    )


def test_shutdown_timeout_reports_intended_snapshot_not_stale_file(
    monkeypatch, tmp_path
):
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    import json
    import threading

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    class FakeExecutor:
        def shutdown(self, **_kwargs):
            pass

    class TimedOutFuture:
        def result(self, timeout=None):
            raise FuturesTimeoutError

        def done(self):
            return False

    path = tmp_path / "freetoken_hot_plan.json"
    path.write_text(json.dumps({"protected_slots": {"0": [0]}}))
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.cpu_executor = None
    cache._hot_plan_stop_event = threading.Event()
    cache._hot_adapt_stop_event = threading.Event()
    cache._hot_adapt_future = TimedOutFuture()
    cache._hot_adapt_executor = FakeExecutor()
    cache._hot_adapt_stop_wait_seconds = lambda: 0.0
    cache._hot_plan_future = None
    cache._hot_plan_executor = None
    cache._hot_plan_write_enabled = False
    cache._hot_plan_last_published_owners = {0: (1, 3)}
    cache._hot_plan_path = str(path)
    warnings = []
    monkeypatch.setattr(
        cache.shutdown_hot_adaptation.__globals__["logger"],
        "warning_rank0",
        warnings.append,
    )

    cache.shutdown_hot_adaptation(timeout_seconds=0.0)

    assert any(
        "attempted 2 experts across 1 layers; no plan was published" in message
        for message in warnings
    )


def test_shutdown_reports_written_but_unreadable_fallback(monkeypatch):
    from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
    import threading

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    class FakeExecutor:
        def shutdown(self, **_kwargs):
            pass

    class TimedOutFuture:
        def result(self, timeout=None):
            raise FuturesTimeoutError

        def done(self):
            return False

    written = Future()
    written.set_result(True)
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.cpu_executor = None
    cache._hot_plan_stop_event = threading.Event()
    cache._hot_adapt_stop_event = threading.Event()
    cache._hot_adapt_future = TimedOutFuture()
    cache._hot_adapt_executor = FakeExecutor()
    cache._hot_adapt_stop_wait_seconds = lambda: 0.0
    cache._hot_plan_future = None
    cache._hot_plan_executor = None
    cache._hot_plan_write_enabled = False
    cache._hot_plan_last_published_owners = {0: (1, 3)}
    cache.snapshot_hot_plan = lambda **_kwargs: written
    cache._persisted_hot_plan_counts = lambda: None
    warnings = []
    monkeypatch.setattr(
        cache.shutdown_hot_adaptation.__globals__["logger"],
        "warning_rank0",
        warnings.append,
    )

    cache.shutdown_hot_adaptation(timeout_seconds=0.0)

    assert any(
        "plan written and fsynced; readback failed; "
        "the file on disk is the new plan"
        in message
        for message in warnings
    )


def test_shutdown_bounds_fake_inflight_adaptation_future(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache._hot_plan_stop_event = threading.Event()
    cache._hot_adapt_stop_event = threading.Event()
    cache.cpu_executor = None
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    cache._hot_adapt_future = executor.submit(release.wait)
    cache._hot_adapt_executor = executor
    copy_stream = object()
    cache._hot_adapt_copy_stream = copy_stream
    cache._hot_plan_future = None
    cache._hot_plan_executor = None
    cache._hot_plan_write_enabled = False
    cache._hot_plan_last_published_owners = {}
    cache.hot_adapt_boundary_cap_frac = 1.0
    cache.hot_adapt_hot_budget_bytes = 40 << 30
    method_globals = cache.shutdown_hot_adaptation.__globals__
    monkeypatch.setitem(method_globals, "_HOT_ADAPT_STOP_WAIT_MAX_SECONDS", 0.01)
    monkeypatch.setitem(method_globals, "_HOT_ADAPT_EXECUTOR_JOIN_SECONDS", 0.02)
    monkeypatch.setitem(method_globals, "_HOT_PLAN_FINAL_WRITE_SECONDS", 0.01)
    warnings = []
    infos = []
    monkeypatch.setattr(method_globals["logger"], "warning_rank0", warnings.append)
    monkeypatch.setattr(method_globals["logger"], "info_rank0", infos.append)

    started = time.monotonic()
    try:
        cache.shutdown_hot_adaptation(timeout_seconds=0.01)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        executor.shutdown(wait=True)

    bound = 0.01 + 0.01 + 0.01 + 0.02
    print(f"bounded in-flight adaptation shutdown elapsed: {elapsed:.3f} s")
    assert elapsed <= bound + 0.15
    assert cache._hot_plan_stop_event.is_set()
    assert cache._hot_adapt_stop_event.is_set()
    assert cache._hot_adapt_executor is executor
    assert cache._hot_adapt_copy_stream is copy_stream
    assert any("stop wait clamped from 82 to 0.01 s" in message for message in warnings)
    assert any("executor join exceeded" in message for message in warnings)
    assert sum("MoE HOT shutdown" in message for message in infos) == 4


def test_exit_safe_executor_does_not_hold_process_at_atexit():
    import subprocess
    import time

    script = """
import time
from freetoken.exit_safe_executor import ExitSafeThreadPoolExecutor

executor = ExitSafeThreadPoolExecutor(max_workers=1)
executor.submit(time.sleep, 4.0)
"""
    started = time.monotonic()
    subprocess.run([sys.executable, "-c", script], check=True, timeout=2.0)
    elapsed = time.monotonic() - started

    print(f"bounded executor process exit elapsed: {elapsed:.3f} s")
    assert elapsed < 2.0


def test_shutdown_continues_after_adaptation_worker_oserror(monkeypatch):
    from concurrent.futures import Future
    import threading

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def shutdown(self, **kwargs):
            self.calls.append(kwargs)

    failed = Future()
    failed.set_exception(OSError("staging failed"))
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache._hot_plan_stop_event = threading.Event()
    cache._hot_adapt_stop_event = threading.Event()
    cache.cpu_executor = None
    cache._hot_adapt_future = failed
    cache._hot_adapt_phase = "copy"
    cache._hot_adapt_executor = FakeExecutor()
    cache._hot_plan_future = None
    cache._hot_plan_executor = None
    cache._hot_plan_write_enabled = False
    cache._hot_plan_last_published_owners = {}
    copy_stream = object()
    cache._hot_adapt_copy_stream = copy_stream
    warnings = []
    monkeypatch.setattr(
        cache.shutdown_hot_adaptation.__globals__["logger"],
        "warning_rank0",
        warnings.append,
    )
    executor = cache._hot_adapt_executor

    cache.shutdown_hot_adaptation()

    assert sum("staging failed" in message for message in warnings) == 1
    assert executor.calls == [{"wait": True, "cancel_futures": False}]
    assert cache._hot_adapt_copy_stream is None


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_cuda_seeded_process_has_persisted_coverage_before_first_request(monkeypatch):
    import torch

    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    device = torch.device("cuda")
    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3).pin_memory()],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2).pin_memory()],
    }
    expert_bytes = sum(
        bank[0][0].numel() * bank[0].element_size() for bank in sources.values()
    )
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=device,
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 3)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps="auto",
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
        persisted_counter_seed={0: (1.0, 9.0, 2.0, 8.0)},
        persisted_seeded_layers=frozenset({0}),
    )
    try:
        torch.cuda.synchronize()
        assert cache.decayed_hot_pair_rate() == pytest.approx(17.0 / 20.0)
        assert cache._hot_adapt_interval_controller.fill_complete
    finally:
        cache.shutdown_hot_adaptation()
