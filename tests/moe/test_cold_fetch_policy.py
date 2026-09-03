import importlib.util
import sys
from enum import IntEnum
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch._dynamo  # noqa: F401  # load before the local Triton import stub


@pytest.fixture(autouse=True)
def _gpu_kernel_import_stubs(monkeypatch):
    """Keep this policy suite runnable on hosts where Triton is unavailable."""
    if importlib.util.find_spec("triton") is not None:
        return

    def jit(fn=None, **_kwargs):
        return (lambda decorated: decorated) if fn is None else fn

    triton = ModuleType("triton")
    triton.jit = jit
    triton.next_power_of_2 = lambda value: 1 << (int(value) - 1).bit_length()
    language = ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language

    class Stat(IntEnum):
        ACTIVE = 0
        MISS = 1
        CALLS = 2

    flashlib = ModuleType("flashlib")
    flashlib.__path__ = []
    kernels = ModuleType("flashlib.kernels")
    kernels.__path__ = []
    slot_cache = ModuleType("flashlib.kernels.slot_cache")
    slot_cache.N_STATS = len(Stat)
    slot_cache.Stat = Stat
    slot_cache.lru_ensure = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(sys.modules, "flashlib", flashlib)
    monkeypatch.setitem(sys.modules, "flashlib.kernels", kernels)
    monkeypatch.setitem(sys.modules, "flashlib.kernels.slot_cache", slot_cache)


def _policy_cache(*, max_fetch=2, cache_size=6, ring_capacity=None):
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=5,
        cache_size=cache_size,
        device=torch.device("cpu"),
        prefill_overlap=False,
        moe_cold_fetch_max=max_fetch,
    )
    hot_slot = cache_size - 1
    cache.hot_expert_capacity = {0: 1}
    cache.hot_expert_ids = {0: (0,)}
    cache.hot_row_for_expert[0, 0] = 0
    cache._hot_slot_for_row = {0: (hot_slot,)}
    cache._hot_slots_device = torch.tensor([hot_slot], dtype=torch.long)
    cache._hot_slot_owners = {0: [0]}
    cache.slot_for_id[0, 0] = hot_slot
    cache.id_of_slot[hot_slot] = 0
    cache._protect_hot_slots()
    cache._gpufetch_capacity = (
        max_fetch if ring_capacity is None else ring_capacity
    )
    cache.cold_fetch_expert_bytes = 64
    return cache


def _run_policy(cache, raw_ids):
    from freetoken.layers.moe import _split_hot_cold_routes

    routed_slots = raw_ids.clone()
    weights = torch.arange(
        1, raw_ids.numel() + 1, dtype=torch.float32
    ).reshape_as(raw_ids)
    cache.ensure_experts_hot(0, routed_slots)
    cache.ensure_cold_experts_fetch(0, raw_ids, routed_slots)
    return _split_hot_cold_routes(raw_ids, routed_slots, weights)


@pytest.mark.parametrize("decode_target", ["cpu", "hybrid"])
def test_cold_fetch_default_zero_preserves_split_contract(decode_target):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(0, 1)

    def run(with_explicit_zero):
        captured = {}

        class Executor:
            def decode_submit(self, layer_id, hidden, weights, ids):
                captured["cpu_ids"] = ids.clone()
                return object()

            def decode_sync(self, pending, hot_partial=True):
                return torch.zeros((2, 4), dtype=torch.float32)

        cache_fields = {
            "decode_target": decode_target,
            "cpu_executor": Executor(),
            "ensure_experts_hot": lambda layer_id, ids: ids.copy_(
                torch.where(ids == 0, ids.new_full((), 5), ids.new_full((), -1))
            ),
            "copy_missing": lambda: None,
            "bank_views": lambda: (),
            "alphas_for_slots": lambda layer_id: None,
        }
        if with_explicit_zero:
            cache_fields["moe_cold_fetch_max"] = 0
            cache_fields["ensure_cold_experts_fetch"] = lambda *args: pytest.fail(
                "disabled cold fetch was called"
            )
        cache = SimpleNamespace(**cache_fields)
        layer = OffloadMoELayer(
            layer_id=0,
            num_experts=4,
            top_k=2,
            hidden_size=4,
            intermediate_size=4,
        )

        def gemm(_cache, hidden, gpu_weights, gpu_slots, **kwargs):
            captured["gpu_weights"] = gpu_weights.clone()
            captured["gpu_slots"] = gpu_slots.clone()
            return torch.zeros_like(hidden)

        layer._expert_gemm = gemm
        raw = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
        weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
        layer._decode_hot_split(
            cache,
            torch.zeros((2, 4)),
            weights,
            raw.clone(),
        )
        return captured

    baseline = run(False)
    disabled = run(True)
    assert baseline.keys() == disabled.keys()
    for name in baseline:
        assert torch.equal(baseline[name], disabled[name]), name


def test_cold_fetch_within_budget_routes_everything_to_gpu():
    cache = _policy_cache(max_fetch=2)
    raw = torch.tensor([[0, 3], [1, 3]], dtype=torch.int32)

    on_gpu, cpu_ids, gpu_slots, gpu_weights = _run_policy(cache, raw)

    assert on_gpu.all()
    assert (cpu_ids == -1).all()
    assert torch.count_nonzero(gpu_weights) == raw.numel()
    protected = set(cache._hot_slots_device.tolist())
    cold_slots = gpu_slots[raw != 0].tolist()
    assert protected.isdisjoint(cold_slots)
    assert all(cache.id_of_slot[slot].item() % cache.num_experts in {1, 3}
               for slot in cold_slots)
    assert cache.stat_cold_fetched_experts.item() == 2
    assert cache.stat_cold_cpu_experts.item() == 0
    assert cache.stat_cold_fetch_bytes.item() == 128
    assert cache.stat_gpu_all_layers.item() == 1


def test_cold_fetch_over_budget_fetches_first_two_distinct_experts():
    cache = _policy_cache(max_fetch=2)
    raw = torch.tensor([[0, 4, 1, 3, 2]], dtype=torch.int32)

    on_gpu, cpu_ids, gpu_slots, gpu_weights = _run_policy(cache, raw)

    assert on_gpu.tolist() == [[True, True, True, False, False]]
    assert cpu_ids.tolist() == [[-1, -1, -1, 3, 2]]
    assert gpu_weights.tolist() == [[1.0, 2.0, 3.0, 0.0, 0.0]]
    fetched_slots = gpu_slots[0, 1:3].tolist()
    assert all(slot not in cache._hot_slots_device.tolist() for slot in fetched_slots)
    assert [cache.id_of_slot[slot].item() for slot in fetched_slots] == [4, 1]
    assert cache.stat_cold_fetched_experts.item() == 2
    assert cache.stat_cold_cpu_experts.item() == 2
    assert cache.stat_gpu_all_layers.item() == 0


def test_cold_fetch_falls_back_when_no_non_protected_slot_is_available():
    cache = _policy_cache(max_fetch=3, cache_size=5, ring_capacity=3)
    for slot, expert in enumerate((1, 2)):
        cache.slot_for_id[0, expert] = slot
        cache.id_of_slot[slot] = expert
    cache.usage[2:4] = torch.iinfo(torch.int64).max
    raw = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    on_gpu, cpu_ids, _gpu_slots, gpu_weights = _run_policy(cache, raw)

    assert on_gpu.tolist() == [[True, False, False, False]]
    assert cpu_ids.tolist() == [[-1, 1, 2, 3]]
    assert gpu_weights.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert cache.id_of_slot.tolist() == [1, 2, -1, -1, 0]
    assert cache.stat_cold_fetched_experts.item() == 0
    assert cache.stat_cold_cpu_experts.item() == 3


def test_cold_fetch_falls_back_when_ring_capacity_is_exhausted():
    cache = _policy_cache(max_fetch=2, ring_capacity=1)
    raw = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    on_gpu, cpu_ids, _gpu_slots, _gpu_weights = _run_policy(cache, raw)

    assert on_gpu.tolist() == [[True, False, False]]
    assert cpu_ids.tolist() == [[-1, 1, 2]]
    assert cache.stat_cold_fetched_experts.item() == 0


def test_cold_fetch_stats_are_reported_per_step_and_reset():
    cache = _policy_cache(max_fetch=2)
    cache.cpu_executor = SimpleNamespace(
        disk_prefetch_stats=lambda reset=False: {"prefetch_calls": 0}
    )
    _run_policy(cache, torch.tensor([[0, 1, 2]], dtype=torch.int32))

    stats = cache.disk_prefetch_stats(reset=True)

    assert stats["cold_fetched_experts_per_step"] == 2
    assert stats["cold_cpu_experts_per_step"] == 0
    assert stats["cold_fetch_bytes_per_step"] == 128
    assert stats["gpu_all_layers_per_step"] == 1
    assert cache.stat_cold_fetch_layer_calls.item() == 0


def test_cold_fetch_default_is_zero():
    from freetoken.engine.config import EngineConfig

    assert EngineConfig.__dataclass_fields__["moe_cold_fetch_max"].default == 0


def test_cold_fetch_validation_accepts_cpu_decode_with_hot_budget():
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_cold_fetch_max=2,
        moe_disk_decode="cpu",
        moe_hot_expert_budget_gib=1.0,
    )
    assert config.moe_cold_fetch_max == 2


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_cold_fetch_validation_rejects_invalid_cap(value):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(
        ValueError, match=r"--moe-cold-fetch-max.*non-negative integer"
    ):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_cold_fetch_max=value,
        )


@pytest.mark.parametrize(
    ("disk_decode", "hot_budget", "message"),
    [
        ("gpufetch", 0.0, r"requires --moe-disk-decode cpu"),
        ("gpufetch", 1.0, r"requires --moe-disk-decode cpu"),
        ("cpu", 0.0, r"requires a non-zero --moe-hot-expert-budget-gib"),
    ],
)
def test_cold_fetch_validation_rejects_incompatible_modes(
    disk_decode, hot_budget, message
):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match=message):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_cold_fetch_max=1,
            moe_disk_decode=disk_decode,
            moe_hot_expert_budget_gib=hot_budget,
        )
