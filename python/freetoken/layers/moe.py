import os
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_backend
from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl, fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.utils import div_even, init_logger

from .base import BaseOP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` (offload) or
# ``_run_experts`` (dense) with a precomputed routing instead of going through the
# generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"

logger = init_logger(__name__)


def _split_hot_cold_routes(
    raw_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the complementary GPU and CPU route views used by split MoE paths.

    ``slot_ids`` contains protected GPU slot ids for resident routes and ``-1``
    for cold routes. The GPU view clamps skipped ids to a valid row and zeros
    their weights; the CPU view retains raw expert ids only for cold routes.
    """
    on_gpu = slot_ids >= 0
    cpu_ids = torch.where(on_gpu, raw_ids.new_full((), -1), raw_ids).contiguous()
    gpu_ids = slot_ids.clamp_min(0)
    gpu_weights = torch.where(
        on_gpu, topk_weights, torch.zeros_like(topk_weights)
    ).contiguous()
    return on_gpu, cpu_ids, gpu_ids, gpu_weights


def _is_oom_error(exc: BaseException) -> bool:
    """Recognize allocator OOMs across PyTorch versions and test doubles."""
    return isinstance(exc, (MemoryError, torch.OutOfMemoryError)) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _assert_route_ids_in_bank_bounds(
    route_ids: torch.Tensor,
    views: tuple[torch.Tensor, ...],
) -> None:
    """Fail before a routed kernel can index outside any expert bank."""
    assert route_ids.numel(), "routed row ids must not be empty"
    min_id, max_id = torch.aminmax(route_ids)
    lo, hi = int(min_id.item()), int(max_id.item())
    assert lo >= 0, f"routed row id {lo} is negative"
    for bank in views:
        table_row_count = bank.shape[0]
        assert hi < table_row_count, (
            f"routed row id {hi} exceeds bank row count {table_row_count}"
        )


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        weight_format: str = "bf16",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.weight_format = weight_format
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        if allocate_experts:
            self._alloc_resident_experts(intermediate_size_per_partition)

    def _alloc_resident_experts(self, intermediate_size_per_partition: int) -> None:
        """Allocate the resident (in-GPU) expert weights for ``self.weight_format``.

        The resident sibling of the offload bank schemas: each format owns its
        tensor layout here and its kernel branch in ``_resident_gemm``.
        """
        if self.weight_format == "fp8_block":
            # Stacked block-fp8 experts + bf16 per-128x128-block inverse scales.
            # Full (unpartitioned) intermediate size: this layout is TP=1-only.
            from freetoken.kernel.triton.fp8_block_linear import FP8

            blk = 128
            n, i, h = self.num_experts, self.intermediate_size, self.hidden_size
            self.gate_up_proj = torch.empty(n, 2 * i, h, dtype=FP8)
            self.gate_up_scale_inv = torch.empty(
                n, 2 * i // blk, h // blk, dtype=torch.bfloat16
            )
            self.down_proj = torch.empty(n, h, i, dtype=FP8)
            self.down_scale_inv = torch.empty(n, h // blk, i // blk, dtype=torch.bfloat16)
            return
        if self.weight_format == "nvfp4":
            # Native group-16 ModelOpt rows. These are ordinary resident tensors, not
            # OffloadMoeCache slots, and are read directly by the existing inline-dequant
            # NVFP4 kernels.
            n, i, h = self.num_experts, self.intermediate_size, self.hidden_size
            if h % 16 or i % 16:
                raise ValueError(f"resident NVFP4 MoE requires H/I divisible by 16, got {h}/{i}")
            fp8 = torch.float8_e4m3fn
            self.gate_up_packed = torch.empty(n, 2 * i, h // 2, dtype=torch.uint8)
            self.gate_up_scale = torch.empty(n, 2 * i, h // 16, dtype=fp8)
            self.gate_up_global = torch.empty(n, 2 * i, dtype=torch.float16)
            self.down_packed = torch.empty(n, h, i // 2, dtype=torch.uint8)
            self.down_scale = torch.empty(n, h, i // 16, dtype=fp8)
            self.down_global = torch.empty(n, h, dtype=torch.float16)
            return
        assert self.weight_format == "bf16", (
            f"no resident expert allocation for weight_format {self.weight_format!r}"
        )
        self.gate_up_proj = torch.empty(
            self.num_experts,
            2 * intermediate_size_per_partition,
            self.hidden_size,
        )
        self.down_proj = torch.empty(
            self.num_experts,
            self.hidden_size,
            intermediate_size_per_partition,
        )

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _run_experts(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Dense (in-GPU) expert compute for a precomputed routing decision."""
        return fused_experts_impl(
            hidden_states,
            self.gate_up_proj,
            self.down_proj,
            topk_weights,
            topk_ids,
            self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Kernel dispatch on ``self.weight_format`` -- the resident mirror of
        ``OffloadMoELayer._expert_gemm``'s ``cache.quant_format`` dispatch."""
        if self.weight_format == "fp8_block":
            # Prefill dequantizes the layer's experts to bf16 and runs the bf16
            # grouped GEMM; decode dequantizes only the routed rows.
            from freetoken.moe.fused_fp8_block import (
                fused_experts_decode_fp8_block,
                fused_experts_fp8_block,
            )

            # Resident rows need no cache movement semantics, so select the
            # lower-overhead decode GEMV for any one-row call.
            if get_global_ctx().batch.is_prefill and hidden_states.shape[0] > 1:
                return fused_experts_fp8_block(
                    hidden_states, self.gate_up_proj, self.gate_up_scale_inv,
                    self.down_proj, self.down_scale_inv,
                    topk_weights, topk_ids, self.num_experts,
                )
            return fused_experts_decode_fp8_block(
                hidden_states, self.gate_up_proj, self.gate_up_scale_inv,
                self.down_proj, self.down_scale_inv, topk_weights, topk_ids,
            )
        if self.weight_format == "nvfp4":
            views = (
                self.gate_up_packed,
                self.gate_up_scale,
                self.gate_up_global,
                self.down_packed,
                self.down_scale,
                self.down_global,
            )
            if get_global_ctx().batch.is_prefill:
                from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

                return fused_experts_nvfp4(
                    hidden_states,
                    *views,
                    topk_weights,
                    topk_ids,
                    self.num_experts,
                    self.activation,
                    self.apply_router_weight_on_input,
                )
            from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

            return fused_experts_decode_nvfp4_marlin(
                hidden_states,
                *views,
                topk_weights,
                topk_ids,
                self.activation,
                self.apply_router_weight_on_input,
            )
        assert self.weight_format == "bf16", (
            f"no resident expert kernel for weight_format {self.weight_format!r}"
        )
        return self._run_experts(hidden_states, topk_weights, topk_ids)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        if self.weight_format != "bf16":
            # Quantized resident experts: generic softmax router + format kernel.
            # The bf16 path below stays on ctx.moe_backend byte-for-byte.
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            return self._maybe_all_reduce(
                self._resident_gemm(hidden_states, topk_weights, topk_ids)
            )
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        return self._maybe_all_reduce(final_hidden_states)


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
        )
        self.layer_id = layer_id
        self.offload_cache: OffloadMoeCache | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.decode_forward(hidden_states, router_logits)
        return self._maybe_all_reduce(final_hidden_states)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._decode_routed(hidden_states, topk_weights, topk_ids)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here."""
        cache = self.offload_cache
        assert cache is not None
        # Optional-feature probe: custom caches (and the DSV4 fixtures) predate
        # expert-granular residency and expose no hot-split surface.
        is_hot_split = getattr(cache, "is_hot_split_layer", None)
        if is_hot_split is not None and is_hot_split(self.layer_id):
            return self._decode_hot_split(cache, hidden_states, topk_weights, topk_ids)
        if cache.is_gpufetch_layer(self.layer_id):
            cache.ensure_experts(self.layer_id, topk_ids)
            cache.copy_missing()
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.bank_views(),
                n=None,
                alphas=cache.alphas_for_slots(self.layer_id),
                is_prefill=False,
            )
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            cache.record_decode_frequency(self.layer_id, topk_ids)
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split is device-side elementwise and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- the GPU weights are zeroed for CPU-assigned
        routes and the CPU ids are -1 for GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        raw = topk_ids.clone()  # raw expert ids for the CPU partial
        cache.ensure_experts_hybrid(self.layer_id, topk_ids)  # -> slot (hit/fetched) or -1
        if cache.collect_stats:
            cache.record_decode_stats_hybrid(self.layer_id)
        return self._decode_split_partials(
            cache, hidden_states, topk_weights, topk_ids, raw
        )

    def _decode_hot_split(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Current HOT/COLD split for a DISK layer, using the hybrid merge contract."""
        raw = topk_ids.clone()
        cache.ensure_experts_hot(self.layer_id, topk_ids)
        return self._decode_split_partials(
            cache, hidden_states, topk_weights, topk_ids, raw
        )

    def _decode_split_partials(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        gpu_ids: torch.Tensor,
        raw_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Merge GPU-assigned and CPU-assigned weighted route partials.

        Shared by dynamic hybrid overflow and the static HOT/COLD DISK split, so
        both use the same graph-safe doorbell ordering and numerical merge.
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        on_gpu, cpu_ids, gpu_slots, gpu_w = _split_hot_cold_routes(
            raw_ids, gpu_ids, topk_weights
        )
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        cache.copy_missing()
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        cpu_routed = (
            cpu_routed_early
            if not _HYBRID_OVERLAP
            else executor.decode_sync(pending, hot_partial=True)
        )
        return gpu_routed + cpu_routed

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement: DISK layers default to routed CPU compute. Other layers,
        and DISK under the copy escape hatch, stream whole layers to the GPU cache."""
        cache = self.offload_cache
        assert cache is not None
        if self.layer_id == 0 and cache.prefill_overlap:
            cache.begin_prefill()
        residency = getattr(cache, "layer_residency", ())
        if self.layer_id < len(residency) and residency[self.layer_id] == "disk":
            if cache.moe_disk_prefill == "cpu":
                hot_split = getattr(cache, "is_hot_split_layer", None)
                if (
                    getattr(cache, "moe_prefill_hot_split", "on") == "on"
                    and hot_split is not None
                    and hot_split(self.layer_id)
                ):
                    return self._prefill_hot_split(
                        cache, hidden_states, topk_weights, topk_ids
                    )
                self._record_prefill_split(
                    cache,
                    topk_ids,
                    torch.zeros_like(topk_ids, dtype=torch.bool),
                )
                return self._prefill_cpu_routes(
                    cache, hidden_states, topk_weights, topk_ids
                )
            # Preserve the existing advisory sweep for the full-layer copy benchmark.
            cache.prefetch_disk_experts(self.layer_id, topk_ids)
        if cache.prefill_overlap and cache.prefill_overlap_for_layer(self.layer_id):
            views = self._wait_prefill_overlap(cache)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        self._prefetch_next_overlap_layer(cache)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(self.num_experts),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )

    def _prefill_hot_split(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run resident DISK routes on protected slots and cold routes on CPU."""
        raw_ids = topk_ids.clone()
        counted_tokens = cache.ensure_experts_hot(
            self.layer_id,
            topk_ids,
            route_weight=cache.hot_adapt_prefill_weight,
        )
        cache.record_hot_adapt_prefill_tokens(counted_tokens)
        on_gpu, cpu_ids, gpu_slots, gpu_weights = _split_hot_cold_routes(
            raw_ids, topk_ids, topk_weights
        )
        any_hot = bool(on_gpu.any().item())
        if not any_hot:
            self._record_prefill_split(cache, raw_ids, on_gpu)
            return self._prefill_cpu_routes(
                cache, hidden_states, topk_weights, raw_ids
            )

        # Cold routes carry zero weight but the slot-indexed kernels still read
        # the clamped row, and slot 0 may never have been written. Point them at
        # a slot this chunk has resident so unwritten scales cannot leak NaN.
        fill_slot = gpu_slots[on_gpu].reshape(-1)[0]
        gpu_slots = torch.where(on_gpu, gpu_slots, fill_slot).contiguous()

        try:
            # Keep the original activations intact for the cold CPU partial and
            # for the full-CPU OOM fallback.
            original_views = cache.bank_views()
            original_alphas = cache.alphas_for_slots(self.layer_id)
            split_kernel = getattr(cache, "moe_prefill_split_kernel", "grouped")
            use_grouped = (
                split_kernel == "grouped"
                and getattr(cache, "quant_format", "nvfp4") == "nvfp4"
                and not getattr(self, "_prefill_split_grouped_disabled", False)
            )
            if use_grouped:
                try:
                    grouped_views = original_views
                    grouped_alphas = original_alphas
                    slot_tables = getattr(cache, "prefill_slot_tables", None)
                    if slot_tables is not None:
                        grouped_views, grouped_alphas, row_count = slot_tables(
                            self.layer_id
                        )
                    else:
                        assert original_views, "grouped prefill requires expert tables"
                        row_count = original_views[0].shape[0]
                        for table in (
                            *original_views,
                            *(original_alphas or ()),
                        ):
                            assert table.shape[0] == row_count, (
                                "grouped prefill tables must share one row id space"
                            )
                    gpu_routed = self._expert_gemm(
                        cache,
                        hidden_states.clone(),
                        gpu_weights,
                        gpu_slots,
                        views=grouped_views,
                        n=row_count,
                        alphas=grouped_alphas,
                        is_prefill=True,
                    )
                except AssertionError as exc:
                    if not getattr(
                        cache, "_prefill_split_grouped_fallback_warned", False
                    ):
                        logger.warning(
                            "Grouped HOT prefill is unsupported for layer %d (%s); "
                            "using the decode kernel for this layer.",
                            self.layer_id,
                            exc,
                        )
                        cache._prefill_split_grouped_fallback_warned = True
                    self._prefill_split_grouped_disabled = True
                    use_grouped = False
            if not use_grouped:
                _assert_route_ids_in_bank_bounds(gpu_slots, original_views)
                gpu_routed = self._expert_gemm(
                    cache,
                    hidden_states.clone(),
                    gpu_weights,
                    gpu_slots,
                    views=original_views,
                    n=None,
                    alphas=original_alphas,
                    is_prefill=False,
                )
        except (MemoryError, RuntimeError) as exc:
            if not _is_oom_error(exc):
                raise
            # The protected rows remain valid. Only this layer's GPU partial is
            # abandoned, and the original route set is repopulated and run on CPU.
            fallback_mask = torch.zeros_like(on_gpu, dtype=torch.bool)
            self._record_prefill_split(cache, raw_ids, fallback_mask)
            return self._prefill_cpu_routes(
                cache, hidden_states, topk_weights, raw_ids
            )

        self._record_prefill_split(cache, raw_ids, on_gpu)
        if bool(on_gpu.all().item()):
            self._prefetch_next_overlap_layer(cache)
            return gpu_routed
        cpu_routed = self._prefill_cpu_routes(
            cache, hidden_states, topk_weights, cpu_ids
        )
        return gpu_routed + cpu_routed

    def _record_prefill_split(
        self,
        cache: OffloadMoeCache,
        raw_ids: torch.Tensor,
        hot_mask: torch.Tensor,
    ) -> None:
        record = getattr(cache, "record_prefill_hot_split", None)
        if record is not None:
            record(raw_ids, hot_mask)

    def _prefill_cpu_routes(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        cpu_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run exactly the selected routes through the existing CPU prefill seam."""
        executor = cache.cpu_executor
        assert executor is not None, "DISK layer requires the CPU MoE executor"
        prepare = getattr(cache, "prepare_disk_prefill", None)
        lease = prepare(self.layer_id, cpu_ids) if prepare is not None else None
        if lease is None:
            # No coalesce lease (flag off, seam missing, or degraded): keep the
            # pre-coalesce union WILLNEED sweep, now over cold routes only.
            cache.prefetch_disk_experts(self.layer_id, cpu_ids)
        self._prefetch_next_overlap_layer(cache)
        try:
            out = executor.prefill(
                self.layer_id, hidden_states, topk_weights, cpu_ids
            )
        finally:
            release = getattr(cache, "release_disk_prefill", None)
            if release is not None:
                release(lease)
        schedule = getattr(cache, "schedule_next_chunk_disk_prefill", None)
        if schedule is not None:
            schedule(self.layer_id, cpu_ids)
        return out

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        cache.prefetch_prefill_layer(self.layer_id)
        self._prefetch_next_overlap_layer(cache)
        return cache.wait_prefill_layer(self.layer_id)

    def _prefetch_next_overlap_layer(self, cache: OffloadMoeCache) -> None:
        """Start the next physical layer's copy when it has pinned banks."""
        if not cache.prefill_overlap:
            return
        next_layer_id = self.layer_id + 1
        if cache.prefill_overlap_for_layer(next_layer_id):
            cache.prefetch_prefill_layer(next_layer_id)

    # ------------------------------------------------------------------
    # Kernel dispatch -- pure routing on the cache's quant format. ``views``
    # are the bank tensors the movement step produced (in bank registration
    # order) and ``topk_ids`` already index their rows.
    # ------------------------------------------------------------------

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
    ) -> torch.Tensor:
        fmt = cache.quant_format
        if fmt in ("nvfp4_marlin", "nvfp4_b12x"):
            # Borrowed W4A16 fused MoE -- Marlin (vLLM, sm_80-99) or b12x
            # (flashinfer, sm_120) over their pre-tiled banks; one kernel serves
            # prefill and decode, with the movement-matched per-row global scales.
            from freetoken.moe.nvfp4_backends import b12x_fused_experts, marlin_fused_experts

            assert alphas is not None
            gate_up_packed, gate_up_scale, down_packed, down_scale = views
            fused = marlin_fused_experts if fmt == "nvfp4_marlin" else b12x_fused_experts
            return fused(
                hidden_states,
                gate_up_packed,
                gate_up_scale,
                alphas[0],
                down_packed,
                down_scale,
                alphas[1],
                topk_weights,
                topk_ids,
                self.activation,
                self.apply_router_weight_on_input,
            )
        if fmt == "nvfp4":
            # FreeToken's Triton inline-dequant kernels over the native ModelOpt
            # rows: the FP4 banks are read directly in the GEMM, no BF16 copy of
            # the experts is ever materialized. The swigluoai scalars (MiniMax-M3)
            # live on the layer via make_moe_layer's extra_attrs (gpt-oss precedent)
            # and are ignored by the plain *_and_mul activations.
            act_alpha = getattr(self, "hidden_act_alpha", 1.702)
            act_limit = getattr(self, "swiglu_limit", None)
            # None == "no clamp" everywhere else in the repo (mxfp4 maps it to +inf).
            act_limit = float("inf") if act_limit is None else act_limit
            if is_prefill:
                from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

                return fused_experts_nvfp4(
                    hidden_states,
                    *views,
                    topk_weights,
                    topk_ids,
                    n,
                    self.activation,
                    self.apply_router_weight_on_input,
                    act_alpha,
                    act_limit,
                )
            # Marlin-style int32 wide-load GEMV (arithmetic dequant, no HW cvt).
            # Bit-identical to the byte-at-a-time path; lifts gate/up BW ~43%->51%
            # (I=512), ~41%->53% (I=768), 65%->72% (I=1536). CUDA-graph safe (fixed
            # shapes, no host sync).
            from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

            return fused_experts_decode_nvfp4_marlin(
                hidden_states,
                *views,
                topk_weights,
                topk_ids,
                self.activation,
                self.apply_router_weight_on_input,
                act_alpha,
                act_limit,
            )
        if fmt == "fp8_block":
            # Block-fp8 experts: fused inline-dequant grouped GEMM reads the routed fp8 rows
            # directly (fp8 banks halve host/cache bytes; no bf16 materialization).
            from freetoken.moe.fused_fp8_block import (
                fused_experts_decode_fp8_block,
                fused_experts_fp8_block,
            )

            gate_up, gate_up_scale, down, down_scale = views
            if is_prefill:
                return fused_experts_fp8_block(
                    hidden_states, gate_up, gate_up_scale, down, down_scale,
                    topk_weights, topk_ids, n, self.activation,
                    self.apply_router_weight_on_input,
                )
            return fused_experts_decode_fp8_block(
                hidden_states, gate_up, gate_up_scale, down, down_scale,
                topk_weights, topk_ids, self.activation, self.apply_router_weight_on_input,
            )
        if fmt == "q4_0":
            # Native GGUF Q4_0 experts: dequant-in-kernel grouped GEMV (MMVQ) over the
            # streamed packed banks; topk_ids already index the cache slots / layer.
            from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

            gate_up, down = views
            return fused_experts_gguf_q4_0(
                hidden_states, gate_up, down, topk_weights, topk_ids, self.activation
            )
        if fmt == "mxfp4_triton":
            # gpt-oss MXFP4 experts (biased, clamped swiglu): transposed split-K GEMV
            # decode + grouped `_t` prefill. The swiglu scalars live on the layer
            # (set at construction), not in the base signature.
            from freetoken.moe.fused_mxfp4 import (
                run_mxfp4_prefill_experts_t,
                run_mxfp4_splitk_decode_experts,
            )

            gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = views
            run = run_mxfp4_prefill_experts_t if is_prefill else run_mxfp4_splitk_decode_experts
            return run(
                hidden_states, topk_weights, topk_ids,
                gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias,
                top_k=self.top_k,
                hidden_act_alpha=self.hidden_act_alpha,
                swiglu_limit=self.swiglu_limit,
            )
        if fmt == "ds_fp4":
            # DeepSeek-V4 FP4 experts: grouped inline-dequant GEMM for streaming
            # prefill chunks (n = bank rows to sort over); per-route dequant GEMV
            # for decode and the sparse small-chunk slot path (n is None there,
            # and sorting over the full slot cache would drown in padding).
            gate_up_packed, gate_up_scale, down_packed, down_scale = views
            if is_prefill and n is not None:
                from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill

                return routed_experts_fp4_prefill(
                    hidden_states, topk_ids, topk_weights,
                    gate_up_packed, gate_up_scale, down_packed, down_scale,
                    self.swiglu_limit, n,
                )
            from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

            return routed_experts_fp4(
                hidden_states, topk_ids, topk_weights,
                gate_up_packed, gate_up_scale, down_packed, down_scale,
                self.swiglu_limit,
            )
        assert fmt == "bf16", f"unknown quant_format {fmt!r}"
        gate_up, down = views
        impl = fused_experts_impl if is_prefill else fused_experts_decode_impl
        return impl(
            hidden_states,
            gate_up,
            down,
            topk_weights,
            topk_ids,
            self.activation,
            self.apply_router_weight_on_input,
        )


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    weight_format: str = "bf16",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    extra_attrs: dict | None = None,
) -> MoELayer:
    """Build the experts layer for ``config.moe_backend`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid, which
    ignore ``weight_format``: the quant format comes from the offload cache) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``extra_attrs`` become instance attributes --
    the seam for per-format scalars the base signature does not carry (e.g.
    ``hidden_act_alpha``/``swiglu_limit``, read back via ``getattr`` by the engine
    and format kernels). ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_backend(config.moe_backend)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["layer_id"] = layer_id
    else:
        kwargs["weight_format"] = weight_format
    layer = layer_cls(**kwargs)
    for name, value in (extra_attrs or {}).items():
        setattr(layer, name, value)
    return layer
