"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _parse_disk_layers_spec as parse_disk
from freetoken.engine.engine import _resolve_cpu_layers as resolve
from freetoken.engine.engine import _resolve_disk_layers as resolve_disk
from freetoken.engine.engine import _auto_cpu_layers as auto_layers
from freetoken.engine.engine import _load_hot_expert_profile as load_hot_profile
from freetoken.engine.engine import _plan_hot_experts as plan_hot
from freetoken.engine.engine import _profiled_hot_pair_rate as profiled_hot_rate
from freetoken.engine.engine import _resolve_hot_expert_sets as resolve_hot
from freetoken.engine.engine import _resolve_hot_expert_setup as resolve_hot_setup
from freetoken.engine.engine import _resolve_persisted_hot_plan as resolve_hot_plan
from freetoken.engine.engine import _validate_disk_prefill_task_size as validate_chunk
from freetoken.moe.cpu_executor import (
    CpuMoeExecutor,
    _StepTimingEvents,
    _split_step_timing_layers,
)

L = 40


def test_step_timing_splits_head_and_tail_at_largest_gap():
    head, tail = _split_step_timing_layers(
        frozenset(range(9)) | frozenset(range(39, 48)), 48
    )
    assert head == tuple(range(9))
    assert tail == tuple(range(39, 48))


@pytest.mark.parametrize(
    ("layers", "expected"),
    [
        ({0, 1, 2}, ((0, 1, 2), ())),
        ({37, 38, 39}, ((), (37, 38, 39))),
        (set(), ((), ())),
    ],
)
def test_step_timing_handles_single_edge_phase(layers, expected):
    assert _split_step_timing_layers(layers, L) == expected


def test_step_timing_resolves_phase_boundaries_and_overlap_without_cuda():
    class Mark:
        def __init__(self, milliseconds):
            self.milliseconds = milliseconds

        def elapsed_time(self, other):
            return other.milliseconds - self.milliseconds

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 48
    executor._disk_banks = {8: (), 39: ()}
    executor._step_timing_events = {
        (8, 1): _StepTimingEvents(
            Mark(1), Mark(1.5), Mark(2), Mark(5), Mark(6), Mark(10)
        ),
        (39, 1): _StepTimingEvents(
            Mark(30), Mark(30.5), Mark(31), Mark(36), Mark(39), Mark(40)
        ),
    }
    executor._step_timing_hot_keys = {(8, 1), (39, 1)}
    executor._tasks = {(8, 1): 108, (39, 1): 139}
    native_ns = {108: 4_000_000, 139: 8_000_000}
    executor._step_timing = True
    executor._ext = SimpleNamespace(
        task_last_run_ns=native_ns.__getitem__,
        step_timing_snapshot_and_reset=lambda: {
            8: {
                "wake_us": 100,
                "groups_us": 20,
                "gil_us": 30,
                "precb_us": 40,
                "notify_us": 10,
                "coord_pre_us": 5,
                "coord_post_us": 7,
                "last_seen_ns": 0,
                "last_done_stored_ns": 0,
                "compute_us": 3900,
                "signal_us": 20,
                "tasks": 1,
                "experts": 4,
                "bytes": 4000,
            },
            39: {
                "wake_us": 300,
                "groups_us": 60,
                "gil_us": 70,
                "precb_us": 80,
                "notify_us": 90,
                "coord_pre_us": 11,
                "coord_post_us": 17,
                "last_seen_ns": 0,
                "last_done_stored_ns": 0,
                "compute_us": 7700,
                "signal_us": 60,
                "tasks": 1,
                "experts": 6,
                "bytes": 6000,
            },
        },
    )

    timing = executor.resolve_step_timing(1, Mark(0), Mark(44))

    assert timing == {
        "cpu_head_us": 10_000,
        "gpu_mid_us": 20_000,
        "cpu_tail_us": 14_000,
        # min(cpu, hot GPU span): min(4ms, 3ms) + min(8ms, 5ms)
        "overlap_us": 8_000,
        "cpu_wake_us": 200,
        "cpu_groups_us": 40,
        "cpu_gil_us": 50,
        "cpu_precb_us": 60,
        "cpu_notify_us": 50,
        "cpu_coord_us": 20,
        "cpu_gpu_in_us": 0,
        "cpu_gpu_out_us": 0,
        "cpu_d2h_us": 1_000,
        "cpu_h2d_us": 5_000,
        "cpu_compute_us": 5800,
        "cpu_signal_us": 40,
        "cpu_layers_per_step": 2,
        "cpu_expert_bytes_per_step": 10_000,
    }


def test_explicit_list():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})  # whitespace + trailing comma
    assert parse("5,5,5", L) == frozenset({5})  # dups collapse


def test_count_evenly_strided():
    assert parse("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse("1", L) == frozenset({0})
    assert len(parse(str(L), L)) == L  # all layers
    assert parse("0", L) == frozenset()


def test_fraction():
    assert len(parse("0.5", L)) == L // 2
    assert len(parse("1.0", L)) == L
    assert parse("0.0", L) == frozenset()


def test_disk_layers_use_the_same_grammar():
    assert parse_disk("3,7,11", L) == frozenset({3, 7, 11})
    assert len(parse_disk("8", L)) == 8
    assert len(parse_disk("0.5", L)) == L // 2


def test_empty():
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5"])
def test_out_of_range_raises(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def _cfg(backend, spec=None, disk=None):
    return SimpleNamespace(
        moe_backend=backend, moe_cpu_layers=spec, moe_disk_layers=disk,
    )


def test_resolve_backend_dispatch():
    # cpu backend -> every layer, ignoring any spec
    assert resolve(_cfg("cpu"), L) == frozenset(range(L))
    assert resolve(_cfg("cpu", "8"), L) == frozenset(range(L))
    # offload + spec -> parsed subset
    assert len(resolve(_cfg("offload", "8"), L)) == 8
    # offload, no spec -> none (plain offload)
    assert resolve(_cfg("offload", None), L) == frozenset()
    # non-offload backend ignores the spec (validation lives in _adjust_config)
    assert resolve(_cfg("fused", "8"), L) == frozenset()


def test_resolve_disk_layers_always_targets_cpu_capable_backends():
    assert resolve_disk(_cfg("offload", disk="3,7"), L) == frozenset({3, 7})
    assert resolve_disk(_cfg("hybrid", disk="2"), L) == frozenset({0, 20})
    assert resolve_disk(_cfg("cpu", disk="1.0"), L) == frozenset(range(L))
    assert resolve_disk(_cfg("fused", disk="3,7"), L) == frozenset()


def _write_ftw_index(path, num_layers):
    index = {
        "format": "freetoken_weight",
        "tensors": [
            {"kind": "experts_bank", "nbytes": 100, "name": f"bank-{i}"}
            for i in range(num_layers)
        ],
    }
    (path / "freetoken_weight.json").write_text(json.dumps(index))


def _auto_config(path, profile=None):
    return SimpleNamespace(
        model_path=str(path),
        model_config=SimpleNamespace(),
        moe_disk_layer_profile=str(profile) if profile is not None else None,
    )


def test_auto_budget_spills_ftw_head_and_tail_layers_to_disk(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 4)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    assert auto_layers(_auto_config(tmp_path), 4) == frozenset({0, 3})


def test_hot_gpu_budget_is_not_reserved_from_host_pin_budget(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 4)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    config = _auto_config(tmp_path)
    config.moe_hot_expert_budget_gib = 100 / 2**30

    # HOT capacity is VRAM-backed. It must not consume 100 bytes from the
    # 201-byte host pin budget and force a third layer onto DISK.
    assert auto_layers(config, 4) == frozenset({0, 3})


def test_disk_and_pinned_hot_budgets_share_the_reserved_slot_pool(tmp_path):
    _write_ftw_index(tmp_path, 4)
    config = SimpleNamespace(
        model_path=str(tmp_path),
        model_config=SimpleNamespace(num_experts=4, num_experts_per_tok=1),
        moe_cache_size=16,
        moe_prefill_overlap=False,
        moe_hot_expert_budget_gib=100 / 2**30,
        moe_pinned_hot_budget_gib=100 / 2**30,
        moe_hot_adapt_interval_steps=1000,
        moe_disk_decode="cpu",
        moe_disk_layer_profile=None,
    )
    plan, capacity, expert_bytes = resolve_hot_setup(
        config,
        frozenset({0, 3}),
        4,
        pinned_layer_ids=frozenset({1, 2}),
    )

    assert expert_bytes == 25
    assert plan == {0: (), 1: (), 2: (), 3: ()}
    assert capacity == {0: 2, 3: 2, 1: 2, 2: 2}
    assert sum(capacity.values()) + 4 <= config.moe_cache_size


def test_auto_budget_uses_lowest_profile_scores_with_stable_ties(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 6)
    profile = tmp_path / "traffic.json"
    profile.write_text(json.dumps({"0": 5, "1": 1, "2": 1, "3": 2, "4": 3, "5": 0}))
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(401 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    logs = []
    monkeypatch.setattr("freetoken.engine.engine.logger.info_rank0", logs.append)

    assert auto_layers(_auto_config(tmp_path, profile), 6) == frozenset({1, 5})
    assert "reserved GPU prefill layers [0, 4, 3, 2] first" in logs[-1]
    assert "([1, 5])" in logs[-1]


def test_auto_budget_accepts_versioned_layer_profile(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 4)
    profile = tmp_path / "traffic-v2.json"
    profile.write_text(json.dumps({
        "version": 2,
        "layers": {"0": 4, "1": 1, "2": 3, "3": 2},
        "expert_hits": {str(layer): [1, 0] for layer in range(4)},
    }))
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )

    assert auto_layers(_auto_config(tmp_path, profile), 4) == frozenset({1, 3})


def test_gpu_prefill_off_sends_all_auto_layers_to_disk(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 4)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(401 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    config = _auto_config(tmp_path)
    config.moe_gpu_prefill_layers = "off"

    assert auto_layers(config, 4) == frozenset(range(4))


def test_hot_partition_uses_equal_top_n_per_disk_layer_and_stable_ties():
    hits = {
        1: (9, 2, 9, 1, 0),
        3: (0, 5, 4, 5, 1),
    }
    # 2 layers * 2 experts/layer * 100 bytes/expert.
    plan = plan_hot(
        hits, frozenset({1, 3}), budget_bytes=499,
        expert_bytes=100, num_experts=5,
    )
    assert plan == {1: (0, 2), 3: (1, 3)}
    assert profiled_hot_rate(hits, plan) == pytest.approx((18 + 10) / 36)


def test_hot_partition_leaves_sub_round_budget_unused():
    hits = {0: (3, 2), 1: (1, 4)}
    assert plan_hot(
        hits, frozenset({0, 1}), budget_bytes=199,
        expert_bytes=100, num_experts=2,
    ) == {}


def test_hot_profile_requires_complete_integer_expert_counts(tmp_path):
    profile = tmp_path / "traffic-v2.json"
    profile.write_text(json.dumps({
        "version": 2,
        "layers": {"0": 1, "1": 2},
        "expert_hits": {"0": [1, 2, 3], "1": [4, -1, 6]},
    }))
    with pytest.raises(ValueError, match="non-negative integers"):
        load_hot_profile(str(profile), 2, 3)


def test_hot_budget_is_independent_of_host_pin_budget(tmp_path, monkeypatch):
    index = {
        "format": "freetoken_weight",
        "tensors": [
            {"kind": "experts_bank", "name": "gate_up#L00000", "nbytes": 400},
            {"kind": "experts_bank", "name": "gate_up#L00001", "nbytes": 400},
            {"kind": "experts_bank", "name": "gate_up_alpha", "nbytes": 20},
        ],
    }
    (tmp_path / "freetoken_weight.json").write_text(json.dumps(index))
    profile = tmp_path / "traffic-v2.json"
    profile.write_text(json.dumps({
        "version": 2,
        "layers": {"0": 1, "1": 1},
        "expert_hits": {"0": [9, 8, 2, 1], "1": [1, 1, 1, 1]},
    }))
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(1 / 2**30))
    config = SimpleNamespace(
        model_path=str(tmp_path),
        model_config=SimpleNamespace(num_experts=4, num_experts_per_tok=1),
        moe_cache_size=20,
        moe_prefill_overlap=False,
        moe_hot_expert_budget_gib=1.0,
        moe_disk_decode="cpu",
        moe_disk_layer_profile=str(profile),
    )

    # The one-byte host pin budget cannot clip the GPU HOT budget. The requested
    # budget and slot capacity both permit all four rows in the DISK layer.
    assert resolve_hot(
        config, frozenset({0}), 2, reserved=10 * 2**30
    ) == {0: (0, 1, 2, 3)}


def test_hot_plan_capacity_is_bounded_by_gpu_slots(tmp_path, monkeypatch):
    num_layers = 4
    num_experts = 8
    index = {
        "format": "freetoken_weight",
        "tensors": [
            {
                "kind": "experts_bank",
                "name": f"gate_up#L{layer_id:05d}",
                "nbytes": 100 * num_experts,
            }
            for layer_id in range(num_layers)
        ],
    }
    (tmp_path / "freetoken_weight.json").write_text(json.dumps(index))
    profile = tmp_path / "traffic-v2.json"
    profile.write_text(json.dumps({
        "version": 2,
        "layers": {str(layer_id): 1 for layer_id in range(num_layers)},
        "expert_hits": {
            str(layer_id): list(range(num_experts, 0, -1))
            for layer_id in range(num_layers)
        },
    }))
    config = SimpleNamespace(
        model_path=str(tmp_path),
        model_config=SimpleNamespace(
            num_experts=num_experts,
            num_experts_per_tok=5,
        ),
        moe_cache_size=28,
        moe_prefill_overlap=True,
        moe_hot_expert_budget_gib=1.0,
        moe_disk_decode="cpu",
        moe_disk_layer_profile=str(profile),
    )
    logs = []
    monkeypatch.setattr("freetoken.engine.engine.logger.info_rank0", logs.append)

    # Reserve max(2E, E + top_k * pinned_layers) = max(16, 8 + 5 * 2)
    # = 18 slots.
    # The remaining 10 slots permit five HOT rows in each of two DISK layers.
    plan = resolve_hot(config, frozenset({0, 3}), num_layers)

    assert plan == {0: (0, 1, 2, 3, 4), 3: (0, 1, 2, 3, 4)}
    assert "bound=slots" in logs[-1]
    assert "slot_limit=5" in logs[-1]


def test_hot_plan_refuses_when_slot_bound_leaves_zero_rows(tmp_path):
    num_layers = 4
    num_experts = 8
    index = {
        "format": "freetoken_weight",
        "tensors": [
            {
                "kind": "experts_bank",
                "name": f"gate_up#L{layer_id:05d}",
                "nbytes": 100 * num_experts,
            }
            for layer_id in range(num_layers)
        ],
    }
    (tmp_path / "freetoken_weight.json").write_text(json.dumps(index))
    profile = tmp_path / "traffic-v2.json"
    profile.write_text(json.dumps({
        "version": 2,
        "layers": {str(layer_id): 1 for layer_id in range(num_layers)},
        "expert_hits": {
            str(layer_id): [1] * num_experts for layer_id in range(num_layers)
        },
    }))
    config = SimpleNamespace(
        model_path=str(tmp_path),
        model_config=SimpleNamespace(
            num_experts=num_experts,
            num_experts_per_tok=5,
        ),
        moe_cache_size=19,
        moe_prefill_overlap=True,
        moe_hot_expert_budget_gib=1.0,
        moe_disk_decode="cpu",
        moe_disk_layer_profile=str(profile),
    )

    config.moe_hot_expert_budget_gib = 50 / 2**30
    assert resolve_hot(config, frozenset({0, 3}), num_layers) == {}
    config.moe_hot_expert_budget_gib = 1.0

    with pytest.raises(
        ValueError,
        match=(
            r"slot bound cannot fit one protected row in each of 2 DISK layers; "
            r"moe_cache_size=19, fetch_reserve=18.*available_slots=1, "
            r"required_hot_slots=2"
        ),
    ):
        resolve_hot(config, frozenset({0, 3}), num_layers)


def test_hot_plan_resolves_hub_id_and_logs_saved_and_current_budgets(
    tmp_path, monkeypatch
):
    import freetoken.moe.hot_adapt as hot_adapt

    plan_path = tmp_path / hot_adapt.HOT_PLAN_FILENAME
    plan_path.write_text("{}")
    identity = {"kind": "ftw", "path": str(tmp_path), "shards": []}
    seed = SimpleNamespace(
        expert_ids={0: (1, 2)},
        counters={0: (1.0, 2.0, 3.0)},
        seeded_layers=frozenset({0}),
        age_seconds=12.0,
        saved_hot_budget_bytes=200,
        tier_commit="tier-test",
        tier_mismatch=False,
    )
    observed = {}
    monkeypatch.setattr(
        "freetoken.utils.hf.download_hf_weight", lambda model_path: str(tmp_path)
    )
    monkeypatch.setattr(hot_adapt, "resolve_tier_commit", lambda: "tier-test")

    def checkpoint_identity(model_path):
        observed["identity_path"] = model_path
        return identity

    monkeypatch.setattr(hot_adapt, "checkpoint_identity", checkpoint_identity)
    monkeypatch.setattr(
        hot_adapt,
        "hot_plan_directory_writable",
        lambda _path: False,
    )

    def load(path, **kwargs):
        observed["load_path"] = path
        observed["current_budget"] = kwargs["current_hot_budget_bytes"]
        return seed

    monkeypatch.setattr(hot_adapt, "load_hot_plan", load)
    logs = []
    monkeypatch.setattr("freetoken.engine.engine.logger.info_rank0", logs.append)
    config = SimpleNamespace(
        model_path="meta-llama/Llama-2-7b-hf",
        moe_hot_plan_persist="auto",
        moe_hot_plan_dir=None,
        moe_hot_plan_interval_minutes=10.0,
        tp_info=SimpleNamespace(is_primary=lambda: True),
        model_config=SimpleNamespace(num_experts=3),
    )

    selected, loaded_seed, runtime = resolve_hot_plan(
        config, frozenset({0}), 1, {0: (0, 1)}, {0: 2}, 150
    )

    assert selected == seed.expert_ids
    assert loaded_seed is seed
    assert runtime["path"] == str(plan_path)
    assert observed == {
        "identity_path": str(tmp_path),
        "load_path": str(plan_path),
        "current_budget": 300,
    }
    assert any(f"resolved_model_path={str(tmp_path)!r}" in message for message in logs)
    assert any(
        "plan was saved at 200 byte budget, current is 300" in message
        for message in logs
    )


def test_hot_plan_resolution_failure_logs_once_and_disables_persistence(monkeypatch):
    import freetoken.moe.hot_adapt as hot_adapt

    def fail_resolution(model_path):
        raise ValueError(f"cannot resolve {model_path}")

    monkeypatch.setattr("freetoken.utils.hf.download_hf_weight", fail_resolution)
    monkeypatch.setattr(hot_adapt, "resolve_tier_commit", lambda: "tier-test")
    warnings = []
    monkeypatch.setattr("freetoken.engine.engine.logger.warning_rank0", warnings.append)
    config = SimpleNamespace(
        model_path="org/missing-model",
        moe_hot_plan_persist="auto",
        moe_hot_plan_dir=None,
        moe_hot_plan_interval_minutes=10.0,
        tp_info=SimpleNamespace(is_primary=lambda: True),
        model_config=SimpleNamespace(num_experts=3),
    )

    selected, seed, runtime = resolve_hot_plan(
        config, frozenset({0}), 1, {0: (0, 1)}, {0: 2}, 150
    )

    assert selected == {0: (0, 1)}
    assert seed is None
    assert runtime["identity"] is None
    assert not runtime["write_enabled"]
    assert len(warnings) == 1
    assert "checkpoint path resolution or identity failed" in warnings[0]


def test_hot_budget_without_profile_starts_all_cold_when_adaptation_is_on(
    tmp_path, monkeypatch,
):
    index = {
        "format": "freetoken_weight",
        "tensors": [
            {"kind": "experts_bank", "name": "gate_up#L00000", "nbytes": 400},
        ],
    }
    (tmp_path / "freetoken_weight.json").write_text(json.dumps(index))
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    config = SimpleNamespace(
        model_path=str(tmp_path),
        model_config=SimpleNamespace(num_experts=4, num_experts_per_tok=1),
        moe_cache_size=8,
        moe_prefill_overlap=False,
        moe_hot_expert_budget_gib=200 / 2**30,
        moe_hot_adapt_interval_steps=1000,
        moe_disk_decode="cpu",
        moe_disk_layer_profile=None,
    )
    assert resolve_hot(config, frozenset({0}), 1) == {0: ()}


def test_hot_budget_without_profile_rejects_static_mode(tmp_path):
    _write_ftw_index(tmp_path, 1)
    config = SimpleNamespace(
        model_path=str(tmp_path),
        model_config=SimpleNamespace(num_experts=1),
        moe_hot_expert_budget_gib=1.0,
        moe_hot_adapt_interval_steps=0,
        moe_disk_decode="cpu",
        moe_disk_layer_profile=None,
    )
    with pytest.raises(ValueError, match="static.*requires.*profile"):
        resolve_hot(config, frozenset({0}), 1)


@pytest.mark.parametrize(
    "contents",
    ["{not-json", json.dumps({"0": 1, "1": 2, "2": 3})],
    ids=["malformed", "incomplete"],
)
def test_bad_profile_warns_and_falls_back(tmp_path, monkeypatch, caplog, contents):
    import logging
    import freetoken.engine.engine as engine

    _write_ftw_index(tmp_path, 4)
    profile = tmp_path / "traffic.json"
    profile.write_text(contents)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: True)
    monkeypatch.setattr(engine.logger, "propagate", True)
    caplog.set_level(logging.WARNING, logger=engine.logger.name)

    assert auto_layers(_auto_config(tmp_path, profile), 4) == frozenset({0, 3})
    assert "falling back to head+tail DISK selection" in caplog.text


def test_external_reservation_refuses_zero_gpu_prefill_layers(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 4)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    config = _auto_config(tmp_path)

    assert auto_layers(config, 4, reserved=0) == frozenset({0, 3})
    with pytest.raises(ValueError, match="cannot fit one MoE layer"):
        auto_layers(config, 4, reserved=200)


@pytest.mark.parametrize("mode", ["cpu", "copy"])
def test_engine_config_accepts_disk_prefill_modes(mode):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_disk_prefill=mode,
    )
    assert config.moe_disk_prefill == mode


@pytest.mark.parametrize("mode", ["cpu", "gpufetch"])
def test_engine_config_accepts_disk_decode_modes(mode):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_disk_decode=mode,
    )
    assert config.moe_disk_decode == mode


def test_engine_config_defaults_disk_prefill_to_cpu():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    assert config.moe_disk_prefill == "cpu"
    assert config.moe_disk_decode == "cpu"
    assert config.moe_gpu_prefill_layers == "auto"
    assert config.moe_hot_expert_budget_gib == 0
    assert config.moe_pinned_hot_budget_gib == 0
    assert config.moe_hot_adapt_halflife_steps == 2000
    assert config.moe_hot_adapt_interval_steps == "auto"
    assert config.moe_hot_adapt_max_swap_gib == 0.5
    assert config.moe_hot_adapt_boundary_cap_frac == 0.5
    assert config.moe_hot_adapt_prefill_weight == 1.0
    assert config.moe_hot_adapt_prefill_run_cap_frac == 0.0
    assert config.moe_hot_adapt_post_prefill_tick is False
    assert config.moe_hot_plan_persist == "auto"
    assert config.moe_hot_plan_dir is None
    assert config.moe_hot_plan_interval_minutes == 10.0
    assert config.moe_hot_adapt_idle_ms == 500
    assert config.moe_hot_adapt_idle_min_interval_ms == 2000


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan")])
def test_engine_config_rejects_invalid_hot_expert_budget(budget):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-expert-budget-gib.*finite non-negative"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_expert_budget_gib=budget,
        )


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan")])
def test_engine_config_rejects_invalid_pinned_hot_expert_budget(budget):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(
        ValueError,
        match="--moe-pinned-hot-budget-gib.*finite non-negative",
    ):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_pinned_hot_budget_gib=budget,
        )


@pytest.mark.parametrize(
    ("backend", "accepted"),
    [("offload", True), ("fused", False), ("cpu", False), ("hybrid", False)],
)
def test_pinned_hot_budget_requires_offload_backend(backend, accepted):
    import torch

    from freetoken.attention import AttnType
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _adjust_config
    from freetoken.models.config import KVCacheGroupSpec

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend="triton",
        moe_backend=backend,
        moe_pinned_hot_budget_gib=1.0,
    )
    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=(0, 1),
        num_kv_heads=1,
        head_dim=64,
        sliding_window=None,
        mla=False,
        index_head_dim=0,
        num_index_layers=0,
        index_ratio=1,
        attn_type=AttnType.FULL,
    )
    object.__setattr__(config, "model_config", SimpleNamespace(
        model_type="test",
        single_stream_only=False,
        is_moe=True,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=False,
        num_layers=2,
        rotary_config=SimpleNamespace(max_position=1024),
        kv_cache_group_specs=lambda: (spec,),
        qwen4_args=None,
        dsv4_args=None,
    ))
    if accepted:
        _adjust_config(config)
    else:
        with pytest.raises(ValueError, match="requires --moe-backend offload"):
            _adjust_config(config)


@pytest.mark.parametrize("half_life", [0, -1, 1.5, True])
def test_engine_config_rejects_invalid_hot_adapt_half_life(half_life):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-halflife-steps"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_halflife_steps=half_life,
        )


@pytest.mark.parametrize("interval", [-1, 1.5, True, "fixed"])
def test_engine_config_rejects_invalid_hot_adapt_interval(interval):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-interval-steps"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_interval_steps=interval,
        )


@pytest.mark.parametrize("max_swap", [0, -1, float("inf"), float("nan")])
def test_engine_config_rejects_invalid_hot_adapt_swap_bound(max_swap):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-max-swap-gib"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_max_swap_gib=max_swap,
        )


@pytest.mark.parametrize(
    "boundary_cap", [0, -1, 1.01, float("inf"), float("nan"), True]
)
def test_engine_config_rejects_invalid_hot_adapt_boundary_cap(boundary_cap):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-boundary-cap-frac"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_boundary_cap_frac=boundary_cap,
        )


@pytest.mark.parametrize("weight", [-0.1, 1.5, float("inf"), float("nan"), True])
def test_engine_config_rejects_invalid_hot_adapt_prefill_weight(weight):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-prefill-weight"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_prefill_weight=weight,
        )


@pytest.mark.parametrize(
    "run_cap", [-0.1, 1.5, float("inf"), float("nan"), True]
)
def test_engine_config_rejects_invalid_hot_adapt_prefill_run_cap(run_cap):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-prefill-run-cap-frac"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_prefill_run_cap_frac=run_cap,
        )


def test_engine_config_rejects_non_bool_hot_adapt_post_prefill_tick():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-post-prefill-tick"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_post_prefill_tick="on",
        )


@pytest.mark.parametrize("mode", ["yes", "", None])
def test_engine_config_rejects_invalid_hot_plan_persist(mode):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-plan-persist"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_plan_persist=mode,
        )


@pytest.mark.parametrize("minutes", [0, -1, float("inf"), float("nan"), True])
def test_engine_config_rejects_invalid_hot_plan_interval(minutes):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-plan-interval-minutes"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_plan_interval_minutes=minutes,
        )


@pytest.mark.parametrize("idle_ms", [-1, 1.5, True])
def test_engine_config_rejects_invalid_hot_adapt_idle_delay(idle_ms):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-hot-adapt-idle-ms"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_idle_ms=idle_ms,
        )


@pytest.mark.parametrize("interval_ms", [-1, 1.5, True])
def test_engine_config_rejects_invalid_hot_adapt_idle_min_interval(interval_ms):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(
        ValueError, match="--moe-hot-adapt-idle-min-interval-ms"
    ):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_hot_adapt_idle_min_interval_ms=interval_ms,
        )


def test_engine_config_rejects_invalid_disk_prefill_mode():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-disk-prefill.*cpu.*copy"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_disk_prefill="gpu",
        )


def test_engine_config_rejects_invalid_disk_decode_mode():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-disk-decode.*cpu.*gpufetch"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_disk_decode="eager",
        )


@pytest.mark.parametrize("lookahead", ["on", "off"])
def test_engine_config_accepts_disk_lookahead_modes(lookahead):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_disk_lookahead=lookahead,
    )
    assert config.moe_disk_lookahead == lookahead


def test_engine_config_rejects_invalid_disk_lookahead_mode():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-disk-lookahead.*on.*off"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_disk_lookahead="auto",
        )


def test_engine_config_validates_cpu_willneed_settings():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    base = {
        "model_path": "/tmp/model",
        "tp_info": DistributedInfo(0, 1),
        "dtype": torch.bfloat16,
    }
    config = EngineConfig(**base, moe_cpu_willneed="recent")
    assert config.moe_cpu_willneed == "recent"
    assert config.moe_cpu_willneed_recent_steps == 256
    assert config.moe_cpu_willneed_fault_ceiling == 2000.0

    with pytest.raises(ValueError, match="--moe-cpu-willneed.*always.*recent"):
        EngineConfig(**base, moe_cpu_willneed="sometimes")
    with pytest.raises(ValueError, match="recent-steps must be positive"):
        EngineConfig(**base, moe_cpu_willneed_recent_steps=0)
    with pytest.raises(ValueError, match="fault-ceiling must be positive"):
        EngineConfig(**base, moe_cpu_willneed_fault_ceiling=0)


@pytest.mark.parametrize("pager", ["madvise", "uffd"])
def test_engine_config_accepts_disk_pagers(pager):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_disk_pager=pager,
        moe_pager_budget_gib=7.5,
    )
    assert config.moe_disk_pager == pager
    assert config.moe_pager_budget_gib == 7.5


def test_engine_config_defaults_to_madvise_disk_pager():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    assert config.moe_disk_pager == "madvise"
    assert config.moe_pager_budget_gib is None
    assert config.host_cache_reserve_gib is None


def test_engine_config_rejects_invalid_uffd_settings():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    base = dict(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="--moe-disk-pager"):
        EngineConfig(**base, moe_disk_pager="kernel")
    with pytest.raises(ValueError, match="--moe-pager-budget-gib"):
        EngineConfig(**base, moe_pager_budget_gib=0)


@pytest.mark.parametrize("backend", ["pinned", "cached", "disk", "uring", "hmm"])
def test_engine_config_accepts_ple_backends(backend):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        ple_backend=backend,
    )
    assert config.ple_backend == backend


def test_engine_config_defaults_ple_backend_to_pinned():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    assert config.ple_backend == "pinned"
    assert config.ple_prefill_gather == "on"


@pytest.mark.parametrize("setting", ["on", "off"])
def test_engine_config_accepts_ple_prefill_gather(setting):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        ple_backend="hmm",
        ple_prefill_gather=setting,
    )
    assert config.ple_prefill_gather == setting


def test_engine_config_rejects_invalid_ple_prefill_gather():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--ple-prefill-gather.*on.*off"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            ple_prefill_gather="auto",
        )


def test_engine_config_rejects_invalid_ple_backend():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(
        ValueError, match="--ple-backend.*pinned.*cached.*disk.*uring.*hmm"
    ):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            ple_backend="ram",
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("ple_uring_staging_mib", 0, "--ple-uring-staging-mib"),
        ("ple_uring_queue_depth", 0, "--ple-uring-queue-depth"),
        ("ple_uring_queue_depth", 4097, "--ple-uring-queue-depth"),
    ],
)
def test_engine_config_rejects_invalid_ple_uring_settings(field, value, match):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    kwargs = {field: value}
    with pytest.raises(ValueError, match=match):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            **kwargs,
        )


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_engine_config_rejects_invalid_ple_cache_budget(budget):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--ple-cache-gib.*finite positive"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            ple_cache_gib=budget,
        )


def _disk_ple_adjust_config(backend="disk"):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        ple_backend=backend,
        attention_backend="triton",
        cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=4,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            single_stream_only=False,
            dsv4_args=None,
            is_moe=False,
            expert_quant="none",
            has_swa_attention=False,
            has_linear_attention=False,
            qwen4_args=SimpleNamespace(ple_layer_ids=(2,)),
        ),
    )
    return config


def test_disk_ple_keeps_cuda_graph_config_enabled(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.delenv("FREETOKEN_PLE_DISK_NO_GRAPHS", raising=False)
    config = _disk_ple_adjust_config()
    _adjust_config(config)
    assert config.cuda_graph_bs == [1, 2, 4]
    assert config.cuda_graph_max_bs == 4


def test_hmm_ple_keeps_cuda_graph_config_enabled(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.setenv("FREETOKEN_PLE_DISK_NO_GRAPHS", "1")
    config = _disk_ple_adjust_config("hmm")
    _adjust_config(config)
    assert config.cuda_graph_bs == [1, 2, 4]
    assert config.cuda_graph_max_bs == 4


def test_cached_ple_keeps_cuda_graph_config_enabled(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.delenv("FREETOKEN_PLE_DISK_NO_GRAPHS", raising=False)
    config = _disk_ple_adjust_config("cached")
    _adjust_config(config)
    assert config.cuda_graph_bs == [1, 2, 4]
    assert config.cuda_graph_max_bs == 4


def test_disk_ple_no_graphs_env_restores_eager_fallback(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.setenv("FREETOKEN_PLE_DISK_NO_GRAPHS", "1")
    config = _disk_ple_adjust_config()
    _adjust_config(config)
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


def test_disk_cpu_prefill_validates_scheduler_chunk_against_task_limit():
    from freetoken.moe.cpu_executor import CPU_MOE_MAX_TASK_TOKENS

    cache = SimpleNamespace(layer_residency=["pinned", "disk"])
    config = SimpleNamespace(moe_disk_prefill="cpu", max_extend_tokens=8192)
    validate_chunk(config, cache)

    for invalid in (0, CPU_MOE_MAX_TASK_TOKENS + 1):
        config.max_extend_tokens = invalid
        with pytest.raises(ValueError, match="max-prefill-length.*token-field range"):
            validate_chunk(config, cache)

    config.moe_disk_prefill = "copy"
    validate_chunk(config, cache)

    config.moe_disk_prefill = "cpu"
    validate_chunk(config, SimpleNamespace(layer_residency=["pinned"]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
