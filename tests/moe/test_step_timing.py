from __future__ import annotations

from types import SimpleNamespace

import freetoken.moe.cpu_executor as cpu_executor
from freetoken.moe.cpu_executor import CpuMoeExecutor, _StepTimingEvents


class _Mark:
    def __init__(self, milliseconds: float):
        self.milliseconds = milliseconds

    def elapsed_time(self, other: "_Mark") -> float:
        return other.milliseconds - self.milliseconds


def test_step_timing_breakdown_aggregates_native_rows_and_d2h():
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._step_timing = True
    executor._step_timing_events = {
        (2, 4): _StepTimingEvents(
            _Mark(0), _Mark(1), _Mark(1.25), _Mark(2), _Mark(3), _Mark(4)
        ),
        (7, 4): _StepTimingEvents(
            _Mark(5), _Mark(6), _Mark(6.5), _Mark(7), _Mark(8), _Mark(9)
        ),
    }
    executor._ext = SimpleNamespace(
        step_timing_snapshot_and_reset=lambda: {
            "2": {
                "wake_us": 10,
                "groups_us": 2,
                "gil_us": 3,
                "precb_us": 4,
                "notify_us": 1,
                "coord_pre_us": 2,
                "coord_post_us": 3,
                "last_seen_ns": 0,
                "last_done_stored_ns": 0,
                "compute_us": 100,
                "signal_us": 5,
                "tasks": 1,
                "experts": 4,
                "bytes": 4096,
            },
            7: {
                "wake_us": 20,
                "groups_us": 4,
                "gil_us": 5,
                "precb_us": 6,
                "notify_us": 5,
                "coord_pre_us": 4,
                "coord_post_us": 5,
                "last_seen_ns": 0,
                "last_done_stored_ns": 0,
                "compute_us": 200,
                "signal_us": 6,
                "tasks": 2,
                "experts": 8,
                "bytes": 8192,
            },
        }
    )

    result = executor.step_timing_breakdown(bs=4)

    assert result["per_layer"][2] == {
        "wake_us": 10.0,
        "groups_us": 2.0,
        "gil_us": 3.0,
        "precb_us": 4.0,
        "notify_us": 1.0,
        "coord_pre_us": 2.0,
        "coord_post_us": 3.0,
        "gpu_in_us": 0.0,
        "gpu_out_us": 0.0,
        "h2d_us": 1000.0,
        "compute_us": 100.0,
        "signal_us": 5.0,
        "tasks": 1,
        "experts": 4,
        "bytes": 4096,
    }
    assert result["total"] == {
        "wake_us": 30.0,
        "groups_us": 6.0,
        "gil_us": 8.0,
        "precb_us": 10.0,
        "notify_us": 6.0,
        "coord_pre_us": 6.0,
        "coord_post_us": 8.0,
        "gpu_in_us": 0.0,
        "gpu_out_us": 0.0,
        "h2d_us": 2000.0,
        "compute_us": 300.0,
        "signal_us": 11.0,
        "total_tasks": 3,
        "total_experts": 12,
        "total_bytes": 12_288,
    }
    assert result["submit_d2h_us"] == {
        "per_layer": {2: 250.0, 7: 500.0},
        "total": 750.0,
    }


def test_step_timing_breakdown_preserves_native_wake_decomposition():
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._step_timing = True
    executor._step_timing_events = {}
    executor._ext = SimpleNamespace(
        step_timing_snapshot_and_reset=lambda: {
            3: {
                "wake_us": 17,
                "groups_us": 2,
                "gil_us": 4,
                "precb_us": 8,
                "notify_us": 3,
                "tasks": 1,
            }
        }
    )

    row = executor.step_timing_breakdown()["per_layer"][3]

    assert row["groups_us"] + row["gil_us"] + row["precb_us"] + row["notify_us"] == 17
    assert row["wake_us"] == 17


def test_step_timing_maps_gpu_handoffs_to_the_host_clock():
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 4
    executor._disk_banks = {3: ()}
    executor._step_timing = True
    executor._step_timing_events = {
        (3, 1): _StepTimingEvents(
            _Mark(0), _Mark(1), _Mark(2), _Mark(4), _Mark(7), _Mark(9)
        )
    }
    executor._step_timing_hot_keys = set()
    executor._tasks = {}
    native = {
        3: {
            "last_seen_ns": 992_123_000,
            "last_done_stored_ns": 996_544_000,
            "tasks": 1,
        }
    }
    executor._ext = SimpleNamespace(
        step_timing_snapshot_and_reset=lambda: native
    )

    calibrated = executor.resolve_step_timing(
        1, _Mark(0), _Mark(10), step_end_host_ns=1_000_000_000
    )
    uncalibrated = executor.resolve_step_timing(1, _Mark(0), _Mark(10))

    assert calibrated["cpu_gpu_in_us"] == 123
    assert calibrated["cpu_gpu_out_us"] == 456
    assert uncalibrated["cpu_gpu_in_us"] == 0
    assert uncalibrated["cpu_gpu_out_us"] == 0


def test_pre_run_callback_mode_is_forwarded_to_native_executor():
    calls = []
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._ext = SimpleNamespace(set_pre_run_callback_mode=calls.append)

    executor._configure_pre_run_callback_mode("before")
    executor._configure_pre_run_callback_mode("after")

    assert calls == [1]


def test_step_timing_breakdown_is_zero_and_does_not_call_native_when_off():
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._step_timing = False
    executor._ext = SimpleNamespace(
        step_timing_snapshot_and_reset=lambda: (_ for _ in ()).throw(AssertionError())
    )

    assert executor.step_timing_breakdown() == {
        "per_layer": {},
        "total": {
            "wake_us": 0.0,
            "groups_us": 0.0,
            "gil_us": 0.0,
            "precb_us": 0.0,
            "notify_us": 0.0,
            "coord_pre_us": 0.0,
            "coord_post_us": 0.0,
            "gpu_in_us": 0.0,
            "gpu_out_us": 0.0,
            "h2d_us": 0.0,
            "compute_us": 0.0,
            "signal_us": 0.0,
            "total_tasks": 0,
            "total_experts": 0,
            "total_bytes": 0,
        },
        "submit_d2h_us": {"per_layer": {}, "total": 0.0},
    }


def _willneed_executor(mode: str, *, recent_steps: int = 2, ceiling: float = 2000):
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 2
    executor.num_experts = 8
    executor._disk_banks = {1: [object()]}
    executor._disk_prefetch_calls = [0, 0]
    executor._disk_prefetch_pages = [0, 0]
    executor._disk_decode_steps = 0
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0
    executor._disk_delta_pages = 0
    executor._disk_major_fault_base = 0
    executor._disk_minor_fault_base = 0
    executor._disk_pagers = set()
    executor._gpufetch_tasks = {}
    executor._ext = SimpleNamespace()
    executor._disk_lookahead_enabled = False
    executor._disk_previous_experts = {}
    executor._disk_predicted_experts = {}
    executor._configure_willneed(mode, recent_steps, ceiling)
    advised = []
    executor._prefetch_selected = (
        lambda layer_id, selected: advised.append((layer_id, list(selected)))
        or len(selected)
    )
    return executor, advised


def test_recent_willneed_skips_same_layer_touches_until_window_expires():
    executor, advised = _willneed_executor("recent")

    executor.prefetch_experts(1, [3, 7])
    executor.prefetch_experts(1, [3, 7])
    executor.prefetch_experts(1, [])
    executor.prefetch_experts(1, [3, 7])

    assert advised == [(1, [3, 7]), (1, []), (1, []), (1, [3, 7])]
    assert executor._willneed_skipped_experts == 2
    assert executor._willneed_advised_experts == 4


def test_always_willneed_advises_every_call_without_a_touch_table():
    executor, advised = _willneed_executor("always")

    executor.prefetch_experts(1, [3, 7])
    executor.prefetch_experts(1, [3, 7])

    assert advised == [(1, [3, 7]), (1, [3, 7])]
    assert not hasattr(executor, "_willneed_last_touch")


def test_prefill_willneed_never_consults_or_updates_recent_touches():
    executor, advised = _willneed_executor("recent")
    before = list(executor._willneed_last_touch[1])

    executor.prefetch_experts(1, [3, 7], is_prefill=True)

    assert advised == [(1, [3, 7])]
    assert executor._willneed_layer_steps[1] == 0
    assert executor._willneed_last_touch[1] == before


def test_willneed_fault_guard_forces_256_always_steps_then_resumes(monkeypatch):
    executor, advised = _willneed_executor("recent", recent_steps=256, ceiling=1)
    monkeypatch.setattr(cpu_executor, "_process_faults", lambda: (0, 3000))

    executor.begin_decode_step()
    executor.prefetch_experts(1, [3, 7])
    for _ in range(255):
        executor.begin_decode_step()
        executor.prefetch_experts(1, [3, 7])
    executor.begin_decode_step()
    executor.prefetch_experts(1, [3, 7])

    assert executor._willneed_guard_trips == 1
    assert all(selected == [3, 7] for _, selected in advised[:256])
    assert advised[256] == (1, [])


def test_willneed_stats_reset_counts_but_preserve_guard_trips(monkeypatch):
    executor, _ = _willneed_executor("recent")
    monkeypatch.setattr(cpu_executor, "_process_faults", lambda: (0, 0))
    executor.prefetch_experts(1, [3, 7])
    executor.prefetch_experts(1, [3, 7])
    executor._willneed_guard_trips = 3

    stats = executor.disk_prefetch_stats(reset=True)
    reset = executor.disk_prefetch_stats()

    assert stats["willneed_skipped_experts"] == 2
    assert stats["willneed_advised_experts"] == 2
    assert stats["willneed_guard_trips"] == 3
    assert reset["willneed_skipped_experts"] == 0
    assert reset["willneed_advised_experts"] == 0
    assert reset["willneed_guard_trips"] == 3
