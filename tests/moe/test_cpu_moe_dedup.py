"""GPU-free coverage for batch-aware CPU MoE decode route grouping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _make_bf16_cache(experts: int, hidden: int, intermediate: int):
    gate_up = torch.randn(
        experts, 2 * intermediate, hidden, dtype=torch.bfloat16,
    ).mul_(0.05).contiguous()
    down = torch.randn(
        experts, hidden, intermediate, dtype=torch.bfloat16,
    ).mul_(0.05).contiguous()
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": [gate_up], "down": [down]},
        num_layers=1,
        num_experts=experts,
        decode_target="cpu",
        cpu_executor=None,
    )


def _run_grouped_decode_on_cpu(executor, hidden, weights, ids):
    """Run a persistent task directly, avoiding the CUDA transport wrapper."""
    batch = hidden.shape[0]
    io = executor._io_for(batch)
    io["x"].copy_(hidden)
    io["ids"].copy_(ids)
    io["w"].copy_(weights)
    executor._ext.run_task(executor._task_for(0, batch))
    return io["y"].clone()


@pytest.mark.parametrize("apply_on_input", [False, True])
def test_grouped_decode_matches_ungrouped_reference_with_recurrent_routes(apply_on_input):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(1308 + int(apply_on_input))
    experts, hidden_size, intermediate, top_k, batch = 16, 64, 64, 4, 8
    cache = _make_bf16_cache(experts, hidden_size, intermediate)
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=apply_on_input,
        num_threads=3,
        max_tokens=batch,
        device=torch.device("cpu"),
    )
    hidden = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(batch, top_k, dtype=torch.float32)
    # Experts 0 and 1 recur in every token. The remaining routes cover enough
    # experts to exercise multiple expert-major work items and stable reduction.
    ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [0, 1, 6, 7],
            [0, 1, 8, 9],
            [0, 1, 2, 4],
            [0, 1, 3, 5],
            [0, 1, 6, 8],
            [0, 1, 7, 9],
        ],
        dtype=torch.int32,
    )

    grouped = _run_grouped_decode_on_cpu(executor, hidden, weights, ids)
    ungrouped = executor.prefill(0, hidden, weights, ids)

    # Both schedules evaluate each route identically and reduce in original
    # top-k order, so the native executor's determinism contract is bit exact.
    assert torch.equal(grouped, ungrouped)


def test_grouped_decode_skips_invalid_routes_like_ungrouped_reference():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(1310)
    experts, hidden_size, intermediate, top_k, batch = 8, 64, 64, 4, 4
    cache = _make_bf16_cache(experts, hidden_size, intermediate)
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=2,
        max_tokens=batch,
        device=torch.device("cpu"),
    )
    hidden = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(batch, top_k, dtype=torch.float32)
    ids = torch.tensor(
        [[0, 1, -1, 2], [0, -1, 1, 3], [0, 1, 2, -1], [0, 1, -1, 3]],
        dtype=torch.int32,
    )

    grouped = _run_grouped_decode_on_cpu(executor, hidden, weights, ids)
    ungrouped = executor.prefill(0, hidden, weights, ids)

    assert torch.equal(grouped, ungrouped)


@pytest.mark.parametrize(
    "configure",
    [
        lambda ext: ext.set_decode_threads(2),
        lambda ext: ext.set_barrier_mode(1, 30),
    ],
    ids=["two_decode_threads", "hybrid_barrier"],
)
def test_grouped_decode_worker_policy_matches_all_threads(configure):
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        pytest.skip("Linux CPU MoE extension is not built")
    required = ("set_decode_threads", "set_barrier_mode")
    if not all(hasattr(_cpu_moe.CpuMoeExecutor, name) for name in required):
        pytest.skip("CPU MoE extension needs rebuilding for worker policy")

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(1311)
    experts, hidden_size, intermediate, top_k, batch = 8, 64, 64, 3, 4
    cache = _make_bf16_cache(experts, hidden_size, intermediate)
    common = dict(
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=4,
        max_tokens=batch,
        device=torch.device("cpu"),
    )
    reference = CpuMoeExecutor(cache, **common)
    configured = CpuMoeExecutor(cache, **common)
    configure(configured._ext)
    hidden = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(batch, top_k, dtype=torch.float32)
    ids = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]], dtype=torch.int32
    )

    expected = _run_grouped_decode_on_cpu(reference, hidden, weights, ids)
    actual = _run_grouped_decode_on_cpu(configured, hidden, weights, ids)

    assert torch.equal(actual, expected)
