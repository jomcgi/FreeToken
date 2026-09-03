from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import lru_ensure

# Hybrid backend: which of a step's missing experts to fetch (when capped below the miss
# count). "recency" (default) fetches the experts most-recently active before this step
# (LRU on the expert -> prioritizes recurring misses, lowering the steady miss rate);
# "lowest_id" fetches the smallest expert ids (the original, routing-blind heuristic).
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)


def ensure_experts(
    cache, layer_id: int, expert_ids: torch.Tensor, *, record_stats: bool = True
) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids.

    Delegates to flashlib's slot cache. ``id_base`` maps this layer's expert ids into the
    flat ``layer * num_experts + expert`` space the cache indexes by, and maps
    ``src_indices`` back, so ``copy_missing`` still resolves against this layer's own host
    tensor. ``out_indices`` aliases the input, preserving the in-place rewrite every
    downstream GEMM depends on.
    """
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        cache.id_of_slot,
        cache.usage,
        cache.step,
        expert_ids,
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats and record_stats else None,
        id_base=layer_id * cache.num_experts,
    )


def ensure_experts_hybrid(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, fetch_fraction: float = 0.0
) -> None:
    """Capped-fetch variant of ``ensure_experts`` (hybrid backend).

    Identical LRU bookkeeping, but only the first ``max_fetch`` of this step's missing
    experts are given a slot and scheduled for copy; the overflow misses stay
    non-resident and their ``expert_ids`` positions are rewritten to ``-1`` (compute on
    the CPU). ``fetch_fraction`` > 0 replaces the fixed cap with the bandwidth-matched
    split (fraction = pcie_bw / cpu_bw): fetch ~fraction of the step's misses, rounded to
    the integer that makes the PCIe fetch and the CPU overflow compute finish closest to
    together. ``num_indices`` = capped fetch count (copy_missing); ``num_missing_full`` =
    pre-cap miss count (stats)."""
    # Q16 fixed point so the GPU kernel and the CPU reference cap identically (no float).
    frac_q16 = min(1 << 16, max(0, round(fetch_fraction * (1 << 16))))
    if not expert_ids.is_cuda:
        return _ensure_experts_hybrid_cpu(cache, layer_id, expert_ids, max_fetch, frac_q16)
    _ensure_experts_hybrid_gpu(cache, layer_id, expert_ids, max_fetch, frac_q16)


def ensure_cold_experts_fetch(
    cache,
    layer_id: int,
    raw_expert_ids: torch.Tensor,
    routed_slot_ids: torch.Tensor,
    max_fetch: int,
    ring_capacity: int,
) -> None:
    """Install a bounded prefix of distinct COLD routes in dynamic GPU slots.

    ``routed_slot_ids`` has already passed through ``ensure_experts_hot`` and
    therefore contains protected HOT slots or -1. The selected cold set is
    installed only when every selected miss fits, which makes slot or staging
    exhaustion fall back to the unchanged HOT/COLD split.
    """
    if not raw_expert_ids.is_cuda:
        return _ensure_cold_experts_fetch_cpu(
            cache,
            layer_id,
            raw_expert_ids,
            routed_slot_ids,
            max_fetch,
            ring_capacity,
        )
    _ensure_cold_experts_fetch_gpu(
        cache,
        layer_id,
        raw_expert_ids,
        routed_slot_ids,
        max_fetch,
        ring_capacity,
    )


def ensure_experts_hot(
    cache, layer_id: int, expert_ids: torch.Tensor, *, record_stats: bool = True
) -> None:
    """Current HOT/COLD split for a file-backed DISK layer.

    HOT experts use protected GPU slots. COLD routes are rewritten to -1 for the
    CPU partial. The single fixed-shape launch is graph-safe. Decay is
    intentionally invocation-based: one 2,048-token prefill chunk and one
    1-token decode step each age prior counts by one half-life step. New route
    counts remain unnormalized, so large prefill observations retain their
    proportional traffic weight.
    """
    if not expert_ids.is_cuda:
        return _ensure_experts_hot_cpu(
            cache, layer_id, expert_ids, record_stats=record_stats
        )
    _ensure_experts_hot_gpu(cache, layer_id, expert_ids, record_stats=record_stats)


def prefill_hit_compact(cache, layer_id: int, buffer_id: int) -> None:
    """Compact this layer's cache-resident experts into gather indices, device-side.

    hit = slot_for_id[layer_id][e] >= 2 * num_experts (the double buffer owns the
    slots below, so those bytes are volatile within a prefill chunk and classify
    as miss). Writes fixed-shape ``_prefill_hit_dst``/``_prefill_hit_src`` (buffer
    row / cache slot) and the count into ``_prefill_hit_num``; one launch on the
    current stream, no host sync. Safe against the concurrent buffer invalidation
    on the copy stream: that only rewrites entries already below the threshold."""
    num_experts = cache.num_experts
    _prefill_hit_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        buffer_id * num_experts,
        2 * num_experts,
        num_experts,
        BLOCK=triton.next_power_of_2(num_experts),
    )


def materialize_layer(cache, layer_id: int) -> None:
    _materialize_layer_gpu(cache, layer_id)


def reset_cache(cache) -> None:
    if not cache.slot_for_id.is_cuda:
        cache.slot_for_id.fill_(-1)
        cache.id_of_slot.fill_(-1)
        cache.usage.zero_()
        cache.step.zero_()
        cache.active_mask.zero_()
        cache.num_indices.zero_()
        return
    _reset_cache_gpu(cache)


def update_session_profile(
    cache, layer_id: int, expert_ids: torch.Tensor, table_ids: torch.Tensor
) -> None:
    """One bounded heavy-hitter update per request for this decode layer."""
    batch = int(table_ids.numel())
    if batch == 0 or expert_ids.numel() % batch:
        return
    routes = expert_ids.reshape(batch, -1)
    max_sessions = cache.session_profile_ids.shape[0] - 1
    if not expert_ids.is_cuda:
        from freetoken.moe.session_profile import update_profile_sketch

        valid = table_ids.reshape(-1).long() < max_sessions
        selected = table_ids.reshape(-1).long()[valid]
        if selected.numel() == 0:
            return
        old_ids = cache.session_profile_ids[selected, layer_id]
        old_counts = cache.session_profile_counts[selected, layer_id]
        new_ids, new_counts = update_profile_sketch(
            old_ids, old_counts, routes[valid], decay=cache._session_decay_factor
        )
        cache.session_profile_ids[:, layer_id].index_copy_(0, selected, new_ids)
        cache.session_profile_counts[:, layer_id].index_copy_(0, selected, new_counts)
        return
    _update_session_profile_kernel[(batch,)](
        cache.session_profile_ids,
        cache.session_profile_counts,
        routes,
        table_ids,
        layer_id,
        cache.num_layers,
        max_sessions,
        cache._session_decay_factor,
        PROFILE_K=cache.session_profile_topk,
        ROUTE_K=routes.shape[1],
    )


@triton.jit
def _update_session_profile_kernel(
    ids_ptr,
    counts_ptr,
    routes_ptr,
    table_ids_ptr,
    layer_id,
    num_layers,
    max_sessions,
    decay,
    PROFILE_K: tl.constexpr,
    ROUTE_K: tl.constexpr,
):
    request = tl.program_id(0)
    table = tl.load(table_ids_ptr + request)
    valid_request = table < max_sessions
    offsets = tl.arange(0, PROFILE_K)
    base = (table * num_layers + layer_id) * PROFILE_K
    ids = tl.load(ids_ptr + base + offsets)
    counts = tl.load(counts_ptr + base + offsets).to(tl.float32) * decay
    for route_idx in range(ROUTE_K):
        expert = tl.load(routes_ptr + request * ROUTE_K + route_idx)
        matches = (ids == expert) & (expert >= 0) & valid_request
        present = tl.sum(matches.to(tl.int32), axis=0) > 0
        victim = tl.argmin(counts, axis=0)
        replace = (offsets == victim) & ~present & (expert >= 0) & valid_request
        ids = tl.where(replace, expert, ids)
        counts = counts + matches.to(tl.float32) + replace.to(tl.float32)
    tl.store(ids_ptr + base + offsets, ids, mask=valid_request)
    tl.store(counts_ptr + base + offsets, counts, mask=valid_request)






def _ensure_experts_hybrid_gpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int
) -> None:
    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(cache.cache_size)
    num_warps = 8 if block_c >= 2048 else 4
    _ensure_experts_hybrid_kernel[(1,)](
        expert_ids,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.expert_recency,
        layer_id,
        expert_ids.numel(),
        int(max_fetch),
        int(frac_q16),
        cache.num_experts,
        cache.cache_size,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        BY_RECENCY=_HYBRID_FETCH_BY_RECENCY,
        num_warps=num_warps,
    )


def _ensure_cold_experts_fetch_gpu(
    cache,
    layer_id: int,
    raw_expert_ids: torch.Tensor,
    routed_slot_ids: torch.Tensor,
    max_fetch: int,
    ring_capacity: int,
) -> None:
    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(cache.cache_size)
    _ensure_cold_experts_fetch_kernel[(1,)](
        raw_expert_ids,
        routed_slot_ids,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.stat_cold_fetched_experts,
        cache.stat_cold_cpu_experts,
        cache.stat_cold_fetch_bytes,
        cache.stat_gpu_all_layers,
        cache.stat_cold_fetch_layer_calls,
        layer_id,
        raw_expert_ids.numel(),
        int(max_fetch),
        int(ring_capacity),
        int(cache.cold_fetch_expert_bytes),
        cache.num_experts,
        cache.cache_size,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        num_warps=8 if block_c >= 2048 else 4,
    )


def _ensure_experts_hot_gpu(
    cache, layer_id: int, expert_ids: torch.Tensor, *, record_stats: bool
) -> None:
    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(cache.cache_size)
    _ensure_experts_hot_kernel[(1,)](
        expert_ids,
        cache.hot_row_for_expert,
        cache.decayed_decode_freq,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.stat_hot_pairs,
        cache.stat_hot_total_pairs,
        layer_id,
        expert_ids.numel(),
        cache.num_experts,
        cache.cache_size,
        cache._hot_decay_factor,
        HOT_ADAPT=cache.hot_adapt_enabled,
        RECORD_STATS=record_stats,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        num_warps=8 if block_c >= 2048 else 4,
    )


def _ensure_experts_hybrid_cpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int
) -> None:
    """CPU reference mirror of the hybrid kernel (eviction/fetch decisions bit-identical to
    the GPU path; see tests/test_offload_lru_kernels.py). Fetches at most ``max_fetch`` (or
    the bandwidth-matched ``~frac_q16/2^16 * misses`` when ``frac_q16`` > 0) of the missing
    experts; overflow misses are rewritten to -1. With ``BY_RECENCY`` the fetch set is the
    most-recently-active misses (ties -> lower id); else the lowest ids."""
    seen = []
    for expert in expert_ids.view(-1).tolist():
        if expert not in seen:
            seen.append(expert)

    cache.active_mask.zero_()
    step = int(cache.step.item()) + 1
    cache.step.fill_(step)
    for expert in seen:
        cache.active_mask[expert] = 1

    for expert in seen:
        slot = int(cache.slot_for_id[layer_id, expert].item())
        if slot != -1:
            cache.usage[slot] = step

    missing = [e for e in seen if int(cache.slot_for_id[layer_id, e].item()) == -1]
    if _HYBRID_FETCH_BY_RECENCY:
        rec = cache.expert_recency[layer_id].tolist()
        missing.sort(key=lambda e: (-rec[e], e))
    else:
        missing.sort()
    if frac_q16 > 0:
        m, q = len(missing), 1 << 16
        lo = (m * frac_q16) >> 16
        cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)  # noqa: E731
        max_fetch = lo if cost(lo) <= cost(lo + 1) else lo + 1
    num_fetch = min(len(missing), int(max_fetch))
    cache.num_missing_full.fill_(len(missing))
    cache.num_indices.fill_(num_fetch)

    usage = cache.usage.tolist()
    for idx in range(num_fetch):
        expert = missing[idx]
        victim = min(range(cache.cache_size), key=lambda s: (usage[s], s))
        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1
        cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
        cache.slot_for_id[layer_id, expert] = victim
        cache.usage[victim] = step
        usage[victim] = step
        cache.evict_slots[idx] = victim
        cache.src_indices[idx] = expert  # layer-local row

    if _HYBRID_FETCH_BY_RECENCY:
        for expert in seen:
            cache.expert_recency[layer_id, expert] = step

    # Overflow misses keep slot_for_id == -1, so the rewrite below yields -1 for them.
    flat = expert_ids.view(-1)
    for i in range(flat.numel()):
        flat[i] = int(cache.slot_for_id[layer_id, int(flat[i].item())].item())


def _ensure_cold_experts_fetch_cpu(
    cache,
    layer_id: int,
    raw_expert_ids: torch.Tensor,
    routed_slot_ids: torch.Tensor,
    max_fetch: int,
    ring_capacity: int,
) -> None:
    """CPU reference for the protected-slot cold-fetch planner."""
    raw = [int(expert) for expert in raw_expert_ids.reshape(-1).tolist()]
    hot_slots = [int(slot) for slot in routed_slot_ids.reshape(-1).tolist()]
    cold = []
    for expert, slot in zip(raw, hot_slots):
        if slot < 0 and expert not in cold:
            cold.append(expert)
    selected = cold[: int(max_fetch)]
    selected_ids = {
        layer_id * cache.num_experts + expert for expert in selected
    }
    missing = [
        expert for expert in selected
        if int(cache.slot_for_id[layer_id, expert].item()) < 0
    ]
    max_usage = torch.iinfo(torch.int64).max
    candidates = [
        slot for slot in range(cache.cache_size)
        if int(cache.usage[slot].item()) != max_usage
        and int(cache.id_of_slot[slot].item()) not in selected_ids
    ]
    can_install = len(missing) <= int(ring_capacity) and len(missing) <= len(candidates)
    fetched = 0
    if can_install:
        step = int(cache.step.item())
        usage = cache.usage.tolist()
        for expert in selected:
            slot = int(cache.slot_for_id[layer_id, expert].item())
            if slot >= 0:
                cache.usage[slot] = step
                usage[slot] = step
        for idx, expert in enumerate(missing):
            victim = min(candidates, key=lambda slot: (usage[slot], slot))
            candidates.remove(victim)
            old_id = int(cache.id_of_slot[victim].item())
            if old_id >= 0:
                cache.slot_for_id.view(-1)[old_id] = -1
            cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
            cache.slot_for_id[layer_id, expert] = victim
            cache.usage[victim] = step
            usage[victim] = step
            cache.evict_slots[idx] = victim
            cache.src_indices[idx] = expert
        fetched = len(missing)

    selected_set = set(selected) if can_install else set()
    flat = routed_slot_ids.reshape(-1)
    for idx, (expert, hot_slot) in enumerate(zip(raw, hot_slots)):
        if hot_slot >= 0:
            flat[idx] = hot_slot
        elif expert in selected_set:
            flat[idx] = int(cache.slot_for_id[layer_id, expert].item())
        else:
            flat[idx] = -1
    cold_cpu = len(cold) - len(selected_set)
    cache.num_indices.fill_(fetched)
    cache.stat_cold_fetched_experts += fetched
    cache.stat_cold_cpu_experts += cold_cpu
    cache.stat_cold_fetch_bytes += fetched * int(cache.cold_fetch_expert_bytes)
    cache.stat_gpu_all_layers += int(cold_cpu == 0)
    cache.stat_cold_fetch_layer_calls += 1


def _ensure_experts_hot_cpu(
    cache, layer_id: int, expert_ids: torch.Tensor, *, record_stats: bool = True
) -> None:
    """CPU reference for the invocation-based HOT/COLD decay and routing rule.

    Decay runs once per call, regardless of routed-token count. A 2,048-token
    prefill chunk and a 1-token decode step therefore each advance decay by one
    half-life step, matching the production Triton kernel below.
    """
    raw = [int(expert) for expert in expert_ids.view(-1).tolist()]
    hot_row = cache.hot_row_for_expert[layer_id].tolist()
    seen_hot = []
    for expert in raw:
        if hot_row[expert] >= 0 and expert not in seen_hot:
            seen_hot.append(expert)

    cache.active_mask.zero_()
    step = int(cache.step.item()) + 1
    cache.step.fill_(step)
    for expert in seen_hot:
        cache.active_mask[expert] = 1
        slot = int(cache.slot_for_id[layer_id, expert].item())
        if slot >= 0:
            cache.usage[slot] = step

    missing = sorted(
        expert for expert in seen_hot
        if int(cache.slot_for_id[layer_id, expert].item()) < 0
    )
    cache.num_missing_full.fill_(len(missing))
    cache.num_indices.fill_(len(missing))
    usage = cache.usage.tolist()
    active_ids = {
        layer_id * cache.num_experts + expert for expert in seen_hot
    }
    for idx, expert in enumerate(missing):
        candidates = [
            slot for slot in range(cache.cache_size)
            if int(cache.id_of_slot[slot].item()) not in active_ids
        ]
        victim = min(candidates, key=lambda slot: (usage[slot], slot))
        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1
        cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
        cache.slot_for_id[layer_id, expert] = victim
        cache.usage[victim] = step
        usage[victim] = step
        cache.evict_slots[idx] = victim
        cache.src_indices[idx] = hot_row[expert]

    hot_pairs = sum(hot_row[expert] >= 0 for expert in raw)
    if cache.hot_adapt_enabled and record_stats:
        counts = torch.bincount(
            torch.tensor(raw, dtype=torch.long), minlength=cache.num_experts
        ).to(torch.float32)
        cache.decayed_decode_freq[layer_id].mul_(cache._hot_decay_factor).add_(counts)
    if record_stats:
        cache.stat_hot_pairs += hot_pairs
        cache.stat_hot_total_pairs += len(raw)
    flat = expert_ids.view(-1)
    for idx, expert in enumerate(raw):
        flat[idx] = (
            int(cache.slot_for_id[layer_id, expert].item())
            if hot_row[expert] >= 0 else -1
        )


def _materialize_layer_gpu(cache, layer_id: int) -> None:
    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def _reset_cache_gpu(cache) -> None:
    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


@triton.jit
def _reset_cache_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    num_indices_ptr,
    total_ids: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
    tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
    tl.store(usage_ptr + off, 0, mask=off < cache_size)
    tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
    if tl.program_id(0) == 0:
        tl.store(step_ptr, 0)
        tl.store(num_indices_ptr, 0)


@triton.jit
def _materialize_layer_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    layer_id: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    expert_mask = off < num_experts
    slot_mask = off < cache_size
    slot = off

    base = layer_id * num_experts
    old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
    # Flat ids make "belongs to this layer" a range check instead of a field compare.
    same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
    tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
    tl.store(usage_ptr + slot, 0, mask=same_layer)

    old_valid = expert_mask & (old_id >= 0) & (~same_layer)
    tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
    tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
    tl.store(usage_ptr + slot, step, mask=expert_mask)
    tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
    tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
    tl.store(num_indices_ptr, num_experts)




@triton.jit(do_not_specialize=["layer_id", "num_active", "max_fetch", "fetch_frac_q16"])
def _ensure_experts_hybrid_kernel(
    expert_ids_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    expert_recency_ptr,
    layer_id,
    num_active,
    max_fetch,
    fetch_frac_q16,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BY_RECENCY: tl.constexpr,
):
    """Capped-fetch timestamp-LRU (hybrid backend).

    Same as ``_ensure_experts_lru_v2_kernel`` but only ``min(num_missing, max_fetch)``
    missing experts are evicted-into / scheduled for copy; the overflow misses stay
    non-resident, so Phase 3 rewrites their positions to -1 (the layer computes those on
    the CPU). ``fetch_frac_q16`` > 0 (Q16 fixed point) replaces the fixed cap with the
    bandwidth-matched split ``~frac * num_missing`` (see the Phase-1 comment), computed
    in-kernel because ``num_missing`` only exists device-side (CUDA graph). ``num_indices``
    = the capped fetch count (copy_missing), ``num_missing_full`` = the pre-cap miss count
    (stats).

    Which misses to fetch is the cap policy. ``BY_RECENCY`` (default) fetches the experts
    most-recently active before this step (LRU on the expert, via ``expert_recency``),
    breaking ties toward the lower expert id -- this prioritizes *recurring* misses for
    caching, lowering the steady miss rate. Otherwise the lowest expert ids are fetched
    (``missing_rank``), the original routing-blind heuristic."""
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts

    # ---- Phase 1: active + missing over experts ----
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    is_active = tl.zeros((BLOCK_E,), dtype=tl.int1)
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        is_active = is_active | (off_e == e)
    tl.store(active_mask_ptr + off_e, is_active.to(tl.int32), mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & (slot == -1) & e_mask
    num_missing = tl.sum(is_missing.to(tl.int32))
    # Cap the fetches; the overflow misses are computed on the CPU (left non-resident).
    if fetch_frac_q16 > 0:
        # Bandwidth-matched split (fetch_frac = pcie_bw / cpu_bw): fetch time scales with
        # F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick
        # the integer neighbor that minimizes the slower (max) side of the overlap.
        lo = (num_missing * fetch_frac_q16) >> 16
        cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
        cost_hi = tl.maximum(
            (lo + 1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16
        )
        max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
    num_fetch = tl.minimum(num_missing, max_fetch)
    tl.store(num_missing_full_ptr, num_missing.to(tl.int64))
    tl.store(num_indices_ptr, num_fetch.to(tl.int64))
    is_hit = is_active & (slot >= 0)
    tl.store(usage_ptr + slot, step, mask=is_hit)

    # Fetch-selection priority: encode (recency desc, id asc) into one strictly-ordered
    # score so argmax has no ties (rec deltas are multiples of num_experts; the id term
    # spans only [0, num_experts), so it can only break exact-recency ties).
    if BY_RECENCY:
        rec = tl.load(expert_recency_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
        score = tl.where(
            is_missing, rec * num_experts + (num_experts - 1 - off_e), -1152921504606846976
        ).to(tl.int64)
    else:
        missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1

    # ---- Phase 2: evict victims by argmin(usage), only for the capped fetches ----
    if num_fetch > 0:
        off_c = tl.arange(0, BLOCK_C)
        c_mask = off_c < cache_size
        oid = tl.load(id_of_slot_ptr + off_c, mask=c_mask, other=-1)
        u = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807).to(tl.int64)
        owner_active = c_mask & False
        for i in tl.range(num_active):
            ei = tl.load(expert_ids_ptr + i)
            owner_active = owner_active | (oid == base + ei)
        u = tl.where(owner_active | (~c_mask), 9223372036854775807, u)
        for i in tl.range(num_fetch):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            old_id = tl.sum(tl.where(off_c == victim, oid, 0))
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            if BY_RECENCY:
                e = tl.argmax(score, axis=0).to(tl.int32)
                score = tl.where(off_e == e, -1152921504606846976, score)
            else:
                e = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
            tl.store(id_of_slot_ptr + victim, base + e)
            tl.store(slot_for_id_ptr + base + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, e)  # layer-local row
            u = tl.where(off_c == victim, 9223372036854775807, u)

    # ---- Phase 3: rewrite expert_ids -> slot id (hit/fetched) or -1 (overflow -> CPU) ----
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        s = tl.load(slot_for_id_ptr + base + e)
        tl.store(expert_ids_ptr + i, s)

    # Bump every active expert's recency to this step (LRU on the expert): an overflow miss
    # computed on the CPU now ranks high if it recurs, so it gets fetched next time.
    if BY_RECENCY:
        step_vec = tl.zeros((BLOCK_E,), dtype=tl.int64) + step
        tl.store(expert_recency_ptr + base + off_e, step_vec, mask=is_active & e_mask)


@triton.jit(
    do_not_specialize=[
        "layer_id",
        "num_active",
        "max_fetch",
        "ring_capacity",
        "expert_bytes",
    ]
)
def _ensure_cold_experts_fetch_kernel(
    raw_ids_ptr,
    routed_slots_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    stat_fetched_ptr,
    stat_cpu_ptr,
    stat_bytes_ptr,
    stat_all_gpu_ptr,
    stat_calls_ptr,
    layer_id,
    num_active,
    max_fetch,
    ring_capacity,
    expert_bytes,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Transactionally add the first N distinct COLD routes to dynamic slots."""
    base = layer_id * num_experts
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    cold_active = tl.zeros((BLOCK_E,), dtype=tl.int1)
    selected = tl.zeros((BLOCK_E,), dtype=tl.int1)
    first_pos = tl.zeros((BLOCK_E,), dtype=tl.int32) + num_active

    # Route order, rather than expert id, defines the stable first-N policy.
    for i in tl.range(num_active):
        expert = tl.load(raw_ids_ptr + i)
        is_cold = tl.load(routed_slots_ptr + i) < 0
        matches = (off_e == expert) & e_mask
        unseen = tl.sum((selected & matches).to(tl.int32)) == 0
        room = tl.sum(selected.to(tl.int32)) < max_fetch
        take = is_cold & unseen & room
        cold_active = cold_active | (matches & is_cold)
        selected = selected | (matches & take)
        first_pos = tl.where(matches & take, i, first_pos)

    cold_count = tl.sum(cold_active.to(tl.int32))
    selected_count = tl.sum(selected.to(tl.int32))
    slots = tl.load(
        slot_for_id_ptr + base + off_e, mask=e_mask, other=-1
    )
    missing = selected & (slots < 0) & e_mask
    num_missing = tl.sum(missing.to(tl.int32))

    off_c = tl.arange(0, BLOCK_C)
    c_mask = off_c < cache_size
    owner = tl.load(id_of_slot_ptr + off_c, mask=c_mask, other=-1)
    usage = tl.load(
        usage_ptr + off_c,
        mask=c_mask,
        other=9223372036854775807,
    ).to(tl.int64)
    owner_selected = c_mask & False
    for i in tl.range(num_active):
        expert = tl.load(raw_ids_ptr + i)
        chosen = tl.sum(
            tl.where((off_e == expert) & selected, 1, 0)
        ) > 0
        owner_selected = owner_selected | (chosen & (owner == base + expert))
    candidate = c_mask & (~owner_selected) & (usage < 9223372036854775807)
    available = tl.sum(candidate.to(tl.int32))
    can_install = (num_missing <= ring_capacity) & (num_missing <= available)

    step = tl.load(step_ptr)
    selected_hit = selected & (slots >= 0) & can_install
    tl.store(usage_ptr + slots, step, mask=selected_hit)
    victim_usage = tl.where(
        candidate, usage, 9223372036854775807
    )
    remaining = missing
    if can_install:
        for i in tl.range(num_missing):
            victim = tl.argmin(victim_usage, axis=0).to(tl.int32)
            old_id = tl.sum(tl.where(off_c == victim, owner, 0))
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            expert = tl.argmin(
                tl.where(remaining, first_pos, num_active + 1), axis=0
            ).to(tl.int32)
            tl.store(id_of_slot_ptr + victim, base + expert)
            tl.store(slot_for_id_ptr + base + expert, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, expert)
            victim_usage = tl.where(
                off_c == victim, 9223372036854775807, victim_usage
            )
            remaining = remaining & (off_e != expert)

    for i in tl.range(num_active):
        expert = tl.load(raw_ids_ptr + i)
        hot_slot = tl.load(routed_slots_ptr + i)
        chosen = tl.sum(
            tl.where((off_e == expert) & selected, 1, 0)
        ) > 0
        cold_slot = tl.load(slot_for_id_ptr + base + expert)
        result = tl.where(
            hot_slot >= 0,
            hot_slot,
            tl.where(can_install & chosen, cold_slot, -1),
        )
        tl.store(routed_slots_ptr + i, result)

    fetched = tl.where(can_install, num_missing, 0).to(tl.int64)
    cold_gpu = tl.where(can_install, selected_count, 0)
    cold_cpu = (cold_count - cold_gpu).to(tl.int64)
    tl.store(num_indices_ptr, fetched)
    tl.store(stat_fetched_ptr, tl.load(stat_fetched_ptr) + fetched)
    tl.store(stat_cpu_ptr, tl.load(stat_cpu_ptr) + cold_cpu)
    tl.store(stat_bytes_ptr, tl.load(stat_bytes_ptr) + fetched * expert_bytes)
    tl.store(
        stat_all_gpu_ptr,
        tl.load(stat_all_gpu_ptr) + (cold_cpu == 0).to(tl.int64),
    )
    tl.store(stat_calls_ptr, tl.load(stat_calls_ptr) + 1)


@triton.jit(do_not_specialize=["layer_id", "num_active"])
def _ensure_experts_hot_kernel(
    expert_ids_ptr,
    hot_row_ptr,
    decayed_freq_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    stat_hot_pairs_ptr,
    stat_total_pairs_ptr,
    layer_id,
    num_active,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    decay_factor,
    HOT_ADAPT: tl.constexpr,
    RECORD_STATS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Timestamp-LRU restricted to the currently published HOT set.

    ``hot_row_ptr`` marks published HOT expert ids, or -1 for COLD experts. HOT
    rows are installed in protected slots before publication. COLD pairs are
    rewritten to -1 and therefore run only in the CPU doorbell task.

    The decayed counter update runs once per kernel invocation, not once per
    routed token. Thus a 2,048-token prefill chunk and a 1-token decode step each
    advance decay by one half-life step; this is the intentional production rule.
    """
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    compact_row = tl.load(hot_row_ptr + base + off_e, mask=e_mask, other=-1)
    eligible = compact_row >= 0
    route_count = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for i in tl.range(num_active):
        expert = tl.load(expert_ids_ptr + i)
        route_count += (off_e == expert).to(tl.int32)
    is_active = (route_count > 0) & eligible
    hot_pairs = tl.sum(tl.where(eligible, route_count, 0))
    if HOT_ADAPT and RECORD_STATS:
        decayed = tl.load(decayed_freq_ptr + base + off_e, mask=e_mask, other=0.0)
        tl.store(
            decayed_freq_ptr + base + off_e,
            decayed * decay_factor + route_count.to(tl.float32),
            mask=e_mask,
        )
    tl.store(active_mask_ptr + off_e, is_active.to(tl.int32), mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & eligible & (slot < 0) & e_mask
    missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1
    num_missing = tl.sum(is_missing.to(tl.int32))
    tl.store(num_indices_ptr, num_missing.to(tl.int64))
    tl.store(num_missing_full_ptr, num_missing.to(tl.int64))
    if RECORD_STATS:
        tl.store(stat_hot_pairs_ptr, tl.load(stat_hot_pairs_ptr) + hot_pairs.to(tl.int64))
        tl.store(
            stat_total_pairs_ptr,
            tl.load(stat_total_pairs_ptr) + num_active.to(tl.int64),
        )
    tl.store(usage_ptr + slot, step, mask=is_active & (slot >= 0))

    if num_missing > 0:
        off_c = tl.arange(0, BLOCK_C)
        c_mask = off_c < cache_size
        owner = tl.load(id_of_slot_ptr + off_c, mask=c_mask, other=-1)
        usage = tl.load(
            usage_ptr + off_c,
            mask=c_mask,
            other=9223372036854775807,
        ).to(tl.int64)
        owner_active = c_mask & False
        for i in tl.range(num_active):
            expert = tl.load(expert_ids_ptr + i)
            is_hot = tl.load(hot_row_ptr + base + expert) >= 0
            owner_active = owner_active | ((owner == base + expert) & is_hot)
        usage = tl.where(
            owner_active | (~c_mask), 9223372036854775807, usage
        )
        for i in tl.range(num_missing):
            victim = tl.argmin(usage, axis=0).to(tl.int32)
            old_id = tl.sum(tl.where(off_c == victim, owner, 0))
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            expert = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
            source_row = tl.sum(
                tl.where(off_e == expert, compact_row, 0)
            ).to(tl.int32)
            tl.store(id_of_slot_ptr + victim, base + expert)
            tl.store(slot_for_id_ptr + base + expert, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, source_row)
            usage = tl.where(
                off_c == victim, 9223372036854775807, usage
            )

    for i in tl.range(num_active):
        expert = tl.load(expert_ids_ptr + i)
        is_hot = tl.load(hot_row_ptr + base + expert) >= 0
        result = tl.load(slot_for_id_ptr + base + expert)
        tl.store(expert_ids_ptr + i, tl.where(is_hot, result, -1))


@triton.jit(do_not_specialize=["buffer_base"])
def _prefill_hit_compact_kernel(
    slot_ptr,     # [num_experts] int32: this layer's slot_for_id row
    dst_ptr,      # [num_experts] int32 out: buffer rows, compacted
    src_ptr,      # [num_experts] int32 out: cache slots, compacted
    num_ptr,      # [1] int64 out: hit count
    buffer_base,  # buffer_id * num_experts
    threshold,    # 2 * num_experts
    num_experts,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    is_hit = lane & (slots >= threshold)
    pos = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(dst_ptr + pos, (buffer_base + offs).to(tl.int32), mask=is_hit)
    tl.store(src_ptr + pos, slots, mask=is_hit)
    tl.store(num_ptr, tl.sum(is_hit.to(tl.int64)))
