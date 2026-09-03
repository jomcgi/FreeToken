from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    MoeLayerProfileBackendMsg,
    MoeLayerProfileResultMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager
from .utils import order_pending_requests, priority_queue_stats

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput

    from .config import SchedulerConfig


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


def _moe_oracle_status_fragment(disk: dict) -> str:
    """Format protected-slot oracle coverage when collection is enabled."""
    if "oracle_hit" not in disk:
        return ""
    return (
        f", disk oracle_hit: {disk['oracle_hit']:.2%} "
        f"vs realized: {disk['realized_hit']:.2%}"
    )


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"

_OOM_ERRORS = (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)
_OOM_ERROR_MESSAGE = (
    "server temporarily out of memory for this request size; "
    "shorten prompt / lower max_tokens"
)
_OOM_ERROR_CODE = "server_out_of_memory"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        self.disk_prefix_store = None
        if config.kv_disk_cache_gib > 0:
            if config.cache_type != "hybrid_radix":
                logger.warning_rank0(
                    "Disk prefix cache is enabled but this model is not using hybrid_radix; "
                    "the disk prefix lane is disabled"
                )
            else:
                from freetoken.kvcache.disk_prefix_cache import (
                    DiskPrefixStore,
                    model_cache_identity,
                )

                identity, checkpoint_fingerprint, config_hash = model_cache_identity(config)
                qsa_args = config.model_config.qwen4_args
                hot_blocks = (
                    max(1, qsa_args.index_budget // config.page_size)
                    if qsa_args is not None
                    else 32
                )
                self.disk_prefix_store = DiskPrefixStore(
                    config.kv_disk_cache_dir,
                    int(config.kv_disk_cache_gib * (1 << 30)),
                    identity=identity,
                    checkpoint_fingerprint=checkpoint_fingerprint,
                    config_hash=config_hash,
                    kv_dtype=config.kv_cache_dtype,
                    lazy_restore=config.lazy_restore == "on",
                    hot_blocks=hot_blocks,
                )
                logger.info_rank0(
                    f"Disk prefix cache enabled at {config.kv_disk_cache_dir!r}, "
                    f"budget={config.kv_disk_cache_gib:.2f} GiB, "
                    f"lazy_restore={config.lazy_restore}, hot_blocks={hot_blocks}, "
                    f"kv_dtype={config.kv_cache_dtype}, "
                    f"fingerprint={checkpoint_fingerprint}, config={config_hash[:12]}"
                )

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            kv_cache=self.engine.kv_cache,
            disk_prefix_store=self.disk_prefix_store,
            moe_offload_cache=self.engine.moe_offload_cache,
            expert_prefetch_stream=self.engine.stream,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            priority_aging_seconds=config.priority_aging_seconds,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # Like abort acknowledgements, OOM failures must follow any sampled output from the
        # prior overlapped batch. Keying by uid also prevents duplicate terminal replies.
        self._pending_oom_errors: dict[int, ErrorReplyMsg] = {}
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        # Natural decode completions retained long enough to recognize a late disconnect
        # while the already-produced HTTP body is draining. Values are insertion ordered.
        self._completed_uids: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.engine.sampler.set_guided_tokenizer(self.tokenizer)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self._kv_ladder_waiting: list[UserMsg] = []
        self._kv_ladder_starvation_uid: int | None = None
        self._kv_ladder = self._make_kv_ladder_policy()

        def _status_log(message: str) -> None:
            cache = getattr(self.engine, "moe_offload_cache", None)
            is_decode_status = message.startswith("Decode batch")
            is_prefill_status = message.startswith("Prefill batch")
            is_ple_status = is_decode_status or is_prefill_status
            disk = (
                cache.disk_prefetch_stats(reset=True)
                if cache is not None and (is_decode_status or is_prefill_status) else {}
            )
            ple_stats = (
                self.engine.model.ple_disk_stats(reset=True)
                if is_ple_status and hasattr(self.engine.model, "ple_disk_stats") else {}
            )
            if disk:
                message += (
                    f", disk prefetch calls: {disk['prefetch_calls']}, "
                    f"disk pages requested: {disk['pages_requested']}, "
                    f"disk major faults: {disk['major_faults']}, "
                    f"disk major faults/decode step: "
                    f"{disk['major_faults_per_decode_step']:.2f}, "
                    f"disk minor faults/decode step: "
                    f"{disk.get('minor_faults_per_decode_step', 0.0):.2f}, "
                    f"moe_prefill_coalesce_experts: "
                    f"{disk.get('moe_prefill_coalesce_experts', 0)}, "
                    f"moe_prefill_coalesce_ms: "
                    f"{disk.get('moe_prefill_coalesce_ms', 0.0):.1f}, "
                    f"moe_prefill_populate_bytes: "
                    f"{disk.get('moe_prefill_populate_bytes', 0)}, "
                    f"moe_prefill_populate_skipped_tmpfs_bytes: "
                    f"{disk.get('moe_prefill_populate_skipped_tmpfs_bytes', 0)}, "
                    f"moe_prefill_populate_ms: "
                    f"{disk.get('moe_prefill_populate_ms', 0.0):.1f}, "
                    f"moe_prefill_populate_overlap_ms: "
                    f"{disk.get('moe_prefill_populate_overlap_ms', 0.0):.1f}, "
                    f"moe_prefill_release_pages: "
                    f"{disk.get('moe_prefill_release_pages', 0)}, "
                    f"moe_prefill_release_skipped_tmpfs_bytes: "
                    f"{disk.get('moe_prefill_release_skipped_tmpfs_bytes', 0)}, "
                    f"moe_prefill_batch_rows: "
                    f"{disk.get('moe_prefill_batch_rows', 0)}, "
                    f"moe_prefill_batch_gemms: "
                    f"{disk.get('moe_prefill_batch_gemms', 0)}, "
                    f"prefill_hot_route_frac: "
                    f"{disk.get('prefill_hot_route_frac', 0.0):.2%}, "
                    f"prefill_cpu_experts: "
                    f"{disk.get('prefill_cpu_experts', 0)}, "
                    f"cold_fetched_experts/step: "
                    f"{disk.get('cold_fetched_experts_per_step', 0.0):.2f}, "
                    f"cold_cpu_experts/step: "
                    f"{disk.get('cold_cpu_experts_per_step', 0.0):.2f}, "
                    f"cold_fetch_bytes/step: "
                    f"{disk.get('cold_fetch_bytes_per_step', 0.0):.0f}, "
                    f"gpu_all_layers/step: "
                    f"{disk.get('gpu_all_layers_per_step', 0.0):.2f}, "
                    f"disk distinct_experts/step: "
                    f"{disk['distinct_experts_per_step']:.2f}, "
                    f"disk dedup_ratio: {disk['dedup_ratio']:.2f}, "
                    f"disk hot_pair_rate: {disk.get('hot_pair_rate', 0.0):.2%}, "
                    f"disk hot_swaps/interval: "
                    f"{disk.get('hot_swaps_per_interval', 0.0):.2f}, "
                    f"hot_adapt_idle_swaps/tick: "
                    f"{disk.get('hot_adapt_idle_swaps_per_tick', 0.0):.2f}, "
                    f"disk decayed_hot_pair_rate: "
                    f"{disk.get('decayed_hot_pair_rate', 0.0):.2%}, "
                    f"hot_adapt_interval: "
                    f"{disk.get('hot_adapt_interval', 0)}, "
                    f"hot_adapt_ticks_prefill: "
                    f"{disk.get('hot_adapt_ticks_prefill', 0)}, "
                    f"hot_adapt_ticks_decode: "
                    f"{disk.get('hot_adapt_ticks_decode', 0)}, "
                    f"hot_adapt_ticks_idle: "
                    f"{disk.get('hot_adapt_ticks_idle', 0)}, "
                    f"disk lookahead_hit_rate: "
                    f"{disk['lookahead_hit_rate']:.4f}, "
                    f"disk delta_pages/step: "
                    f"{disk['delta_pages_per_step']:.2f}, "
                    f"disk gpufetch fills/step: "
                    f"{disk['gpufetch_fills_per_step']:.2f}, "
                    f"disk gpufetch fill_us: {disk['gpufetch_fill_us']:.0f}, "
                    f"resume_prefetch_experts: "
                    f"{disk.get('resume_prefetch_experts', 0)}, "
                    f"resume first64 tok/s: "
                    f"{disk.get('resume_first64_tok_s', 0.0):.2f}, "
                    f"resume steady tok/s: "
                    f"{disk.get('resume_steady_tok_s', 0.0):.2f}, "
                    f"resume warmup/steady: "
                    f"{disk.get('resume_warmup_ratio', 0.0):.3f}, "
                    f"protected_experts: {disk.get('protected_experts', 0)}"
                )
                message += _moe_oracle_status_fragment(disk)
                if disk.get("pager_backend") == "uffd":
                    message += (
                        f", uffd fills: {disk['fills']} "
                        f"(prefetch: {disk['fills_from_prefetch']}, "
                        f"fault: {disk['fault_driven']}), "
                        f"uffd pages installed: {disk.get('pages_installed', 0)}, "
                        f"uffd evictions: {disk['evictions']}, "
                        f"uffd resident GiB: {disk['resident_bytes'] / 2**30:.2f}"
                    )
            if ple_stats:
                message += (
                    f", ple_prefetch_pages: {ple_stats['ple_prefetch_pages']}, "
                    f"ple_major_faults: {ple_stats['ple_major_faults']}, "
                    f"ple_staging_us: {ple_stats['ple_staging_us']:.0f}"
                )
                if "ple_prefill_gather_rows" in ple_stats:
                    message += (
                        f", ple_prefill_gather_rows: "
                        f"{ple_stats['ple_prefill_gather_rows']}, "
                        f"ple_prefill_gather_ms: "
                        f"{ple_stats['ple_prefill_gather_ms']:.1f}"
                    )
                if "ple_hits" in ple_stats:
                    message += (
                        f", ple_hits: {ple_stats['ple_hits']}, "
                        f"ple_misses: {ple_stats['ple_misses']}, "
                        f"ple_evictions: {ple_stats['ple_evictions']}, "
                        f"ple_installed_rows: {ple_stats['ple_installed_rows']}, "
                        f"ple_hit_rate: {ple_stats['ple_hit_rate']:.4f}, "
                        f"ple_overflow_fallbacks: {ple_stats['ple_overflow_fallbacks']}"
                    )
                if "ple_rows_per_step" in ple_stats:
                    message += (
                        f", ple_rows_per_step: {ple_stats['ple_rows_per_step']:.2f}, "
                        f"ple_gather_ms_per_decode_step: "
                        f"{ple_stats['ple_gather_ms_per_decode_step']:.2f}, "
                        f"ple_gather_ms_per_prefill_chunk: "
                        f"{ple_stats['ple_gather_ms_per_prefill_chunk']:.2f}, "
                        f"ple_dedup_rate: {ple_stats['ple_dedup_rate']:.4f}"
                    )
            logger.info_rank0(message)

        self.status_reporter = SchedulerStatusReporter(
            log=_status_log,
            decode_log_interval=config.decode_log_interval,
            disk_prefix_store=self.disk_prefix_store,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def _make_kv_ladder_policy(self):
        """Bind the pure ladder policy to this engine's measured pool costs."""
        from .kv_ladder import (
            DEFAULT_KV_LADDER_STEP_TOKENS,
            KVLadderPolicy,
            kv_ladder_eligibility,
        )

        eligibility = kv_ladder_eligibility(self.config)
        if getattr(self.config, "kv_ladder", "off") != "on":
            return None
        if not eligibility.enabled:
            logger.warning_rank0(
                "KV ladder inactive: " + "; ".join(eligibility.inactive_reasons)
            )
            return None
        if not self.cache_manager.supports_runtime_rebuild:
            logger.warning_rank0(
                "KV ladder inactive: this model's cache does not support runtime rebuild"
            )
            return None

        moe = self.engine.moe_offload_cache
        assert moe is not None, "eligible KV ladder requires an offloaded MoE slot cache"

        from freetoken.engine.cache_budget import (
            expert_bytes_per_slot,
            net_cache_budget_bytes,
        )
        from freetoken.kvcache.linear_state_pool import state_pool_bytes

        cache_per_page, fixed_cache_size, page_tokens, min_reserve = (
            type(self.engine.kv_cache).kv_cost(self.config)
        )
        physical_mamba_slots = (
            self.engine.linear_state_pool.num_slots
            if self.engine.linear_state_pool is not None else None
        )
        fixed_cache_size += state_pool_bytes(self.config, physical_mamba_slots)
        protected = tuple(sorted(moe.hot_expert_capacity.items()))
        cap_tokens = getattr(
            self.config, "kv_ladder_cap_tokens", self.config.max_seq_len
        )
        policy = KVLadderPolicy(
            step_tokens=DEFAULT_KV_LADDER_STEP_TOKENS,
            max_context_tokens=cap_tokens,
            page_size=page_tokens,
            pool_budget_bytes=net_cache_budget_bytes(
                self.config.memory_ratio,
                self.engine._baseline_free,
                self.engine._weights_bytes,
                fixed_cache_size,
            ),
            kv_bytes_per_page=cache_per_page,
            moe_bytes_per_slot=expert_bytes_per_slot(moe.bank_sources),
            min_moe_slots=self.config.model_config.num_experts,
            prefill_overlap=moe.prefill_overlap,
            protected_rows_by_layer=protected,
        )
        current_tokens = self.engine.num_pages * policy.page_size
        next_rung_tokens = policy.next_rung_tokens(current_tokens)
        slots_after_first_growth = policy.moe_slots_at_tokens(
            next_rung_tokens, moe.cache_size
        )
        logger.info_rank0(
            "KV ladder startup: floor=%d tokens, cap=%d tokens, current_pool=%d "
            "tokens, expert_slots_at_startup=%d, expert_slots_after_first_growth=%d, "
            "+1 dummy page",
            getattr(
                self.config,
                "kv_ladder_floor_tokens",
                max(self.config.kv_reserve_tokens, min_reserve),
            ),
            policy.max_context_tokens,
            current_tokens,
            moe.cache_size,
            slots_after_first_growth,
        )
        if current_tokens >= policy.max_context_tokens:
            reason = (
                "the configured --num-pages/--num-tokens cap was reached at startup"
                if getattr(self.config, "kv_ladder_explicit_cap", False)
                else "the model context cap was reached at startup"
            )
            logger.warning_rank0(
                "ladder inert: pool already at cap %d; %s",
                policy.max_context_tokens,
                reason,
            )
        return policy

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
        cache = getattr(self.engine, "moe_offload_cache", None)
        idle_hook = getattr(cache, "hot_adapt_while_idle", None)
        if idle_hook is not None:
            if self.config.tp_info.size != 1:
                if not getattr(self, "_hot_adapt_idle_tp_disabled_logged", False):
                    logger.info_rank0(
                        "MoE HOT idle adaptation disabled: tensor parallel size "
                        f"is {self.config.tp_info.size}; synchronized idle ticks "
                        "are not implemented"
                    )
                    self._hot_adapt_idle_tp_disabled_logged = True
                return
            queue = self._recv_from_tokenizer
            idle_hook(lambda: not queue.empty(), queue.wait_for_item)

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
        hot_slot_owners: dict[int, tuple[int | None, ...]] | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        quiesce_lazy_restores = getattr(
            self.cache_manager, "quiesce_lazy_restores", None
        )
        if quiesce_lazy_restores is not None:
            quiesce_lazy_restores()
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages, hot_slot_owners=hot_slot_owners,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
            or getattr(self, "_kv_ladder_waiting", None)
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        if last_data is None and self._pending_rebuild is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._drain_kv_ladder_waiting()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                try:
                    ongoing_data = (forward_input, self._forward(forward_input))
                except _OOM_ERRORS as oom:
                    ongoing_data = self._recover_forward_oom(forward_input, oom)

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_oom_errors()
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
            or getattr(self, "_kv_ladder_waiting", None)
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        if self._pending_rebuild is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._drain_kv_ladder_waiting()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            try:
                ongoing_data = (forward_input, self._forward(forward_input))
            except _OOM_ERRORS as oom:
                ongoing_data = self._recover_forward_oom(forward_input, oom)

        self._process_last_data(ongoing_data)
        self._flush_oom_errors()
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING or self.config.speculative_mtp == "on":
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        if self.disk_prefix_store is not None:
            self.disk_prefix_store.close(wait=True)
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        # getattr probe: overlap-loop test doubles predate the engine attribute.
        engine = getattr(self, "engine", None)
        if batch.is_decode and engine is not None and engine.moe_offload_cache is not None:
            engine.moe_offload_cache.record_resume_decode_batch(batch.reqs)
        if getattr(batch, "mtp_verify", False):
            self.engine.resolve_mtp_timing(batch)
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk. The
                    # harness helper only stages immutable tensors for a disk write. It performs
                    # no radix insertion and changes no KV page or recurrent-slot ownership.
                    if not req.aborted:
                        self.cache_manager.persist_intermediate_cache_anchor(req)
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        if getattr(req, "abort_discard", False):
                            Scheduler._discard_client_req_resources(self, req)
                        else:
                            self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    if getattr(req, "abort_discard", False):
                        Scheduler._discard_client_req_resources(self, req)
                    else:
                        self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                if req.restore_started_at is not None and self.disk_prefix_store is not None:
                    self.disk_prefix_store.note_first_token_after_restore(
                        (time.perf_counter() - req.restore_started_at) * 1000.0
                    )
                    req.restore_started_at = None
                tokens = (
                    next_tokens_cpu
                    if getattr(batch, "mtp_verify", False)
                    else next_tokens_cpu[i : i + 1]
                )
                finished = False
                for next_token_tensor in tokens:
                    req.append_host(next_token_tensor.unsqueeze(0))
                    next_token = int(next_token_tensor.item())
                    # EOS / stop-string -> "stop", output budget exhausted -> "length";
                    # EOS and stop strings win over length. Host length, rather than
                    # req.can_decode, is required when one MTP step has already advanced
                    # device_len by several accepted tokens.
                    hit_length = req.input_ids.numel() >= req.max_device_len
                    hit_eos = (
                        not req.sampling_params.ignore_eos
                        and next_token in self.eos_token_ids
                    )
                    hit_grammar = bool(
                        req.guided_state is not None
                        and req.guided_state.terminated
                    )
                    matched_stop = (
                        self._match_stop_str(req)
                        if (
                            not hit_eos
                            and req.guided_state is None
                            and req.sampling_params.stop_strs
                        )
                        else None
                    )
                    finished = (
                        hit_length or hit_eos or hit_grammar or matched_stop is not None
                    )
                    finish_reason = (
                        (
                            "stop"
                            if (hit_eos or hit_grammar or matched_stop is not None)
                            else "length"
                        )
                        if finished
                        else None
                    )
                    if (
                        next_token == self.toolcall_anchor_id
                        and req.toolcall_anchor_len is None
                        and not finished
                    ):
                        req.toolcall_anchor_len = req.input_ids.numel()
                    reply.append(
                        DetokenizeMsg(
                            uid=req.uid,
                            next_token=next_token,
                            finished=finished,
                            finish_reason=finish_reason,
                            matched_stop=matched_stop,
                            stop_strs=req.sampling_params.stop_strs or None,
                        )
                    )
                    if finished:
                        break

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    Scheduler._remember_completed_uid(self, req.uid)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        queue_reqs, queue_priority_bands, max_wait_seconds = self._queue_stats()
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=queue_reqs,
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            queue_priority_bands=queue_priority_bands,
            max_wait_seconds=max_wait_seconds,
        )
        self.send_result(reply)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _kv_ladder_plan(self, msg: UserMsg):
        policy = getattr(self, "_kv_ladder", None)
        if policy is None:
            return None
        moe = self.engine.moe_offload_cache
        assert moe is not None
        input_tokens = len(msg.input_ids)
        max_output_tokens = min(
            msg.sampling_params.max_tokens,
            max(0, policy.max_context_tokens - input_tokens),
        )
        return policy.plan(
            current_pages=self.engine.num_pages,
            current_moe_slots=moe.cache_size,
            input_tokens=input_tokens,
            max_output_tokens=max_output_tokens,
            protected_rows_by_layer=tuple(sorted(moe.hot_expert_capacity.items())),
            prefill_overlap=moe.prefill_overlap,
        )

    def _queue_for_kv_ladder(self, msg: UserMsg) -> bool:
        """Hold a request until an idle safe point can grow KV before admission."""
        from .kv_ladder import KVLadderCapacityError

        waiting = getattr(self, "_kv_ladder_waiting", None)
        if waiting is None:
            waiting = self._kv_ladder_waiting = []
        clock = getattr(self.prefill_manager, "clock", time.monotonic)
        now = clock()
        aging_seconds = getattr(
            self.prefill_manager, "priority_aging_seconds", 30.0
        )
        starved = self._kv_ladder_starved_waiter(now=now)
        if starved is not None:
            waiting.append(msg)
            waiting[:] = order_pending_requests(
                waiting,
                now=now,
                aging_seconds=aging_seconds,
            )
            logger.info_rank0(
                "KV ladder held request %d behind starvation-bound request %d",
                msg.uid,
                starved.uid,
            )
            return True
        try:
            plan = self._kv_ladder_plan(msg)
        except KVLadderCapacityError as exc:
            logger.warning_rank0("KV ladder cannot grow for request %d: %s", msg.uid, exc)
            return False
        if plan is None:
            return False
        waiting.append(msg)
        waiting[:] = order_pending_requests(
            waiting,
            now=now,
            aging_seconds=aging_seconds,
        )
        logger.info_rank0(
            "KV ladder queued request %d before admission: required=%d tokens, "
            "current=%d tokens, target=%d tokens",
            msg.uid,
            plan.required_tokens,
            plan.current_tokens,
            plan.target_tokens,
        )
        return True

    def _kv_ladder_starved_waiter(
        self, *, now: float | None = None
    ) -> UserMsg | None:
        """Return and log the oldest waiter whose admission grace has expired."""
        waiting = getattr(self, "_kv_ladder_waiting", None)
        aging_seconds = getattr(
            self.prefill_manager, "priority_aging_seconds", 30.0
        )
        if not waiting or aging_seconds <= 0:
            self._kv_ladder_starvation_uid = None
            return None
        if now is None:
            now = getattr(self.prefill_manager, "clock", time.monotonic)()
        starved = [
            req for req in waiting if now - req.arrival_time >= aging_seconds
        ]
        if not starved:
            self._kv_ladder_starvation_uid = None
            return None
        msg = min(starved, key=lambda req: req.arrival_time)
        if getattr(self, "_kv_ladder_starvation_uid", None) != msg.uid:
            self._kv_ladder_starvation_uid = msg.uid
            logger.warning_rank0(
                "KV ladder starvation bound reached: request %d waited %.1fs "
                "(limit %.1fs); pausing new admissions until it is grown and admitted",
                msg.uid,
                max(0.0, now - msg.arrival_time),
                aging_seconds,
            )
        return msg

    def _hot_slot_owners_for_plan(self, plan) -> dict[int, tuple[int | None, ...]] | None:
        if not plan.lost_protected_rows:
            return None
        cache = self.engine.moe_offload_cache
        assert cache is not None
        counts = dict(plan.protected_rows_after)
        selected = {}
        for layer_id in sorted(cache._hot_slot_owners):
            count = counts.get(layer_id, 0)
            if count <= 0:
                continue
            current = cache._hot_slot_owners[layer_id]
            populated = [owner for owner in current if owner is not None]
            selected[layer_id] = tuple(
                (populated + [None] * (count - len(populated)))[:count]
            )
        return selected

    def _drain_kv_ladder_waiting(self) -> None:
        """Grow at idle, admitting a starvation-bound or highest-ranked waiter."""
        waiting = getattr(self, "_kv_ladder_waiting", None)
        if not waiting:
            return
        assert not self.prefill_manager.runnable and not self.decode_manager.runnable
        now = getattr(self.prefill_manager, "clock", time.monotonic)()
        starved = self._kv_ladder_starved_waiter(now=now)
        waiting[:] = order_pending_requests(
            waiting,
            now=now,
            aging_seconds=getattr(self.prefill_manager, "priority_aging_seconds", 30.0),
        )
        if starved is None:
            msg = waiting.pop(0)
        else:
            waiting.remove(starved)
            msg = starved

        from .kv_ladder import KVLadderCapacityError

        try:
            plan = self._kv_ladder_plan(msg)
        except KVLadderCapacityError as exc:
            logger.warning_rank0("KV ladder cannot grow for request %d: %s", msg.uid, exc)
            plan = None
        if plan is not None and not self.cache_manager.supports_runtime_rebuild:
            logger.warning_rank0(
                "KV ladder cannot grow for request %d: this model's cache does not "
                "support runtime rebuild",
                msg.uid,
            )
            plan = None
        if plan is not None:
            self._pending_rebuild = CacheRebuildBackendMsg(
                request_id=f"auto-kv-ladder:{msg.uid}:{plan.target_pages}",
                moe_cache_size=plan.target_moe_slots,
                num_pages=plan.target_pages,
            )
            started_at = time.perf_counter()
            status = self._execute_pending_rebuild(
                hot_slot_owners=self._hot_slot_owners_for_plan(plan),
                send_reply=False,
            )
            rebuild_ms = (time.perf_counter() - started_at) * 1000.0
            if status == "ok":
                logger.info_rank0(
                    "KV ladder growth: tokens %d -> %d, expert slots %d -> %d, "
                    "rebuild_ms=%.1f",
                    plan.current_tokens,
                    plan.target_tokens,
                    plan.current_moe_slots,
                    plan.target_moe_slots,
                    rebuild_ms,
                )
                for layer_id, lost in plan.lost_protected_rows:
                    logger.warning_rank0(
                        "KV ladder growth evicted protected HOT rows: layer=%d, lost_rows=%d",
                        layer_id,
                        lost,
                    )
            else:
                logger.warning_rank0(
                    "KV ladder rebuild for request %d ended with status=%s; "
                    "admitting against retained geometry",
                    msg.uid,
                    status,
                )
        self._admit_user_msg(msg)
        if getattr(self, "_kv_ladder_starvation_uid", None) == msg.uid:
            self._kv_ladder_starvation_uid = None

    def _queue_stats(self) -> tuple[int, dict[str, int], float]:
        """Report prefill and ladder waiters as one scheduler queue."""
        pending = [
            *self.prefill_manager.pending_list,
            *getattr(self, "_kv_ladder_waiting", ()),
        ]
        bands, max_wait = priority_queue_stats(
            pending,
            now=getattr(self.prefill_manager, "clock", time.monotonic)(),
        )
        return len(pending), bands, max_wait

    def _admit_user_msg(self, msg: UserMsg) -> None:
        input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
        max_output_len = max_seq_len - input_len
        if max_output_len <= 0:
            logger.warning_rank0(
                f"Input sequence length {input_len} exceeds {max_seq_len}, "
                f"request {msg.uid} is dropped."
            )
            self.send_result(
                [
                    ErrorReplyMsg(
                        uid=msg.uid,
                        error=(
                            f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                            f"(prompt + generation); shorten the prompt or increase the KV "
                            f"cache budget"
                        ),
                        code="context_length_exceeded",
                    )
                ]
            )
            return
        if msg.sampling_params.max_tokens > max_output_len:
            msg.sampling_params.max_tokens = max_output_len
            logger.warning_rank0(
                f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
            )
        if msg.sampling_params.guided_decoding is not None:
            try:
                self.engine.sampler.validate_guided(msg.sampling_params.guided_decoding)
            except Exception as exc:
                logger.warning_rank0(
                    "Guided decoding setup failed for request %d: %s", msg.uid, exc
                )
                self.send_result([
                    ErrorReplyMsg(
                        uid=msg.uid,
                        error=f"invalid guided decoding request: {exc}",
                        code="invalid_request_error",
                    )
                ])
                return
        self.prefill_manager.add_one_req(msg)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            if not self._queue_for_kv_ladder(msg):
                self._admit_user_msg(msg)
        elif isinstance(msg, AbortBackendMsg):
            # A disconnect can race the terminal sample through the tokenizer queues. Once
            # decode has finished, resources are already released and the output is merely
            # draining, so the successful completion wins and no abort is recorded or acked.
            completed = getattr(self, "_completed_uids", {})
            if msg.uid in completed or any(
                getattr(req, "uid", None) == msg.uid
                for req in getattr(self, "finished_reqs", ())
            ):
                logger.debug_rank0(
                    "Ignoring abort for completed request %d while output drains", msg.uid
                )
                return

            client_disconnected = getattr(msg, "client_disconnected", False)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            if client_disconnected and msg.uid in tombstones:
                return
            phase, tokens_processed = (
                Scheduler._abort_diagnostics(self, msg.uid)
                if client_disconnected else ("queued", 0)
            )
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            waiting = getattr(self, "_kv_ladder_waiting", None)
            if waiting:
                waiting[:] = [req for req in waiting if req.uid != msg.uid]
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                last_data = getattr(self, "_last_data", None)
                inflight = (
                    last_data is not None
                    and req_to_free in last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                    req_to_free.abort_discard = client_disconnected
                elif client_disconnected:
                    # Match OOM recovery's failed cleanup. A disconnect may be observed at
                    # any forward boundary, so do not publish its request-owned tail to the
                    # prefix cache or park a session profile. Shared matched prefixes remain.
                    Scheduler._discard_client_req_resources(self, req_to_free)
                else:
                    self._free_req_resources(req_to_free)
            if client_disconnected:
                Scheduler._record_client_abort(self)
                logger.warning_rank0(
                    "Client abort request_id=%d, phase=%s, tokens_processed=%d",
                    msg.uid,
                    phase,
                    tokens_processed,
                )
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            pending_acks = getattr(self, "_pending_abort_acks", None)
            if pending_acks is None:
                pending_acks = self._pending_abort_acks = set()
            pending_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        elif isinstance(msg, MoeLayerProfileBackendMsg):
            cache = self.engine.moe_offload_cache
            if cache is None:
                self._reply_moe_layer_profile(
                    msg.request_id, "unsupported", error="this model has no MoE offload cache"
                )
            elif not cache.collect_stats:
                self._reply_moe_layer_profile(
                    msg.request_id,
                    "unsupported",
                    error="restart the server with --moe-collect-stats",
                )
            else:
                try:
                    profile = cache.decode_miss_layer_profile()
                except Exception as exc:  # noqa: BLE001
                    self._reply_moe_layer_profile(
                        msg.request_id, "failed", error=f"could not read MoE stats: {exc!r}"
                    )
                else:
                    self._reply_moe_layer_profile(msg.request_id, "ok", profile=profile)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once
            if req.qsa_restore_pending is not None:
                pending = getattr(self.engine.kv_cache, "_pending_ring", None)
                if pending is not None:
                    pending[req.table_idx].copy_(
                        req.qsa_restore_pending.to(device=pending.device)
                    )
                req.qsa_restore_pending = None

    def _free_req_resources(self, req: Req, *, failed: bool = False) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        if failed:
            self.cache_manager.cache_req(req, finished=True, failed=True)
        else:
            self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _abort_diagnostics(self, uid: int) -> tuple[str, int]:
        """Return the scheduler phase and committed token positions for an abort."""
        for msg in getattr(self, "_kv_ladder_waiting", ()):
            if msg.uid == uid:
                return "queued", 0
        last_data = getattr(self, "_last_data", None)
        if last_data is not None:
            batch = last_data[0].batch
            for req in getattr(batch, "reqs", ()):
                if getattr(req, "uid", None) == uid:
                    return batch.phase, max(0, int(getattr(req, "cached_len", 0)))

        prefill = getattr(self, "prefill_manager", None)
        for pending in getattr(prefill, "pending_list", ()):
            if getattr(pending, "uid", None) != uid:
                continue
            chunk = getattr(pending, "chunked_req", None)
            if chunk is not None:
                return "prefill", max(0, int(getattr(chunk, "cached_len", 0)))
            return "queued", 0

        decode = getattr(self, "decode_manager", None)
        for req in getattr(decode, "running_reqs", ()):
            if getattr(req, "uid", None) == uid:
                return "decode", max(0, int(getattr(req, "cached_len", 0)))
        # The abort may beat its UserMsg across tokenizer workers. It is queued from the
        # scheduler's perspective and the tombstone below will drop it before admission.
        return "queued", 0

    def _discard_client_req_resources(self, req: Req) -> None:
        """Release a disconnected request through the OOM guard's no-commit path.

        A successful forward leaves device_len one position ahead of cached_len for the next
        sample. That next position has not been allocated yet. OOM recovery restores this
        length from its write metadata; disconnect cleanup performs the equivalent adjustment
        before calling the same failed cleanup path.
        """
        cached_len = getattr(req, "cached_len", None)
        device_len = getattr(req, "device_len", None)
        if cached_len is not None and device_len is not None and device_len > cached_len:
            req.device_len = cached_len
        self._free_req_resources(req, failed=True)

    def _record_client_abort(self) -> None:
        reporter = getattr(self, "status_reporter", None)
        if reporter is None:
            return
        record = getattr(reporter, "record_client_abort", None)
        if record is not None:
            record()
        else:
            reporter.client_aborts = getattr(reporter, "client_aborts", 0) + 1

    def _remember_completed_uid(self, uid: int) -> None:
        completed = getattr(self, "_completed_uids", None)
        if completed is None:
            completed = self._completed_uids = {}
        completed[uid] = None
        while len(completed) > 65_536:
            completed.pop(next(iter(completed)))

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _reply_moe_layer_profile(
        self,
        request_id: str,
        status: str,
        *,
        profile: dict | None = None,
        error: str | None = None,
    ) -> None:
        self.send_result(
            [
                MoeLayerProfileResultMsg(
                    request_id=request_id,
                    status=status,
                    profile=profile,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(
        self,
        *,
        hot_slot_owners: dict[int, tuple[int | None, ...]] | None = None,
        send_reply: bool = True,
    ) -> str:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        if hot_slot_owners is not None:
            requested["hot_slot_owners"] = hot_slot_owners
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {
            k: snapshot[k]
            for k, v in requested.items()
            if v is not None and k in snapshot
        }
        if hot_slot_owners is not None:
            cache = self.engine.moe_offload_cache
            assert cache is not None
            prior["hot_slot_owners"] = {
                layer_id: tuple(owners)
                for layer_id, owners in cache._hot_slot_owners.items()
            }

        def reply(status: str, error: str | None = None) -> None:
            if send_reply:
                self._reply_rebuild(msg.request_id, status, error=error)
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            reply("rejected", str(e))
            return "rejected"
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                reply("rejected", repr(e))
                return "rejected"
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                reply("failed", repr(e))
                return "failed"
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                reply(
                    "failed",
                    f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return "failed"
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            reply("rejected", f"rebuild failed and was rolled back: {e!r}")
            return "rejected"
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        reply("ok")
        return "ok"

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # Native MTP verifies one greedy request at a time. Reserve seed plus one draft
        # in the paged cache, but keep the batch classified as decode so the target MoE
        # remains on its decode-routed expert path.
        mtp_verify = (
            self.config.speculative_mtp == "on"
            and batch.is_decode
            and len(batch.reqs) == 1
            and batch.reqs[0].sampling_params.is_greedy
            and batch.reqs[0].sampling_params.guided_decoding is None
            and batch.reqs[0].mtp_hidden is not None
        )
        if mtp_verify:
            from freetoken.spec_decode import MTP_DRAFT_STEPS, reserve_mtp_window

            req = batch.reqs[0]
            width = min(MTP_DRAFT_STEPS + 1, req.remain_len)
            if width > 1:
                reserve_mtp_window(batch, width)
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        return self._build_forward_input(batch)

    def _build_forward_input(self, batch: Batch) -> ForwardInput:
        """Build device metadata for a batch whose request resources are already allocated."""
        batch.lazy_restore_pending = batch.lazy_restore_pending or any(
            req.lazy_kv_restore is not None and not req.lazy_kv_restore.complete
            for req in batch.reqs
        )
        if batch.is_prefill:
            self._gather_multimodal(batch)
            chunked_lens = {
                req.uid: req.device_len for req in batch.reqs if isinstance(req, ChunkedReq)
            }
            if chunked_lens:
                batch._oom_chunked_device_lens = chunked_lens
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _prepare_decode_retry(self, reqs: List[Req]) -> ForwardInput:
        """Rebuild decode metadata after shedding one request without allocating pages again."""
        batch = Batch(reqs=reqs, phase="decode")
        self.engine.graph_runner.pad_batch(batch)
        return self._build_forward_input(batch)

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _flush_oom_errors(self) -> None:
        pending = getattr(self, "_pending_oom_errors", None)
        if not pending:
            return
        # A request can finish normally in the prior overlapped batch while the already-launched
        # next decode step OOMs. Its successful terminal result wins over the speculative OOM.
        finished_uids = {req.uid for req in getattr(self, "finished_reqs", ())}
        replies = [msg for uid, msg in pending.items() if uid not in finished_uids]
        pending.clear()
        if replies:
            self.send_result(replies)

    def _record_oom_aborts(self, count: int) -> None:
        reporter = getattr(self, "status_reporter", None)
        if reporter is None:
            return
        record = getattr(reporter, "record_oom_aborts", None)
        if record is not None:
            record(count)
        else:
            reporter.oom_aborts = getattr(reporter, "oom_aborts", 0) + count

    def _abort_oom_requests(self, reqs: List[Req]) -> None:
        pending = getattr(self, "_pending_oom_errors", None)
        if pending is None:
            pending = self._pending_oom_errors = {}
        abort_prefill = getattr(self.prefill_manager, "abort_req", None)
        remove_decode = getattr(self.decode_manager, "remove_req", None)
        for req in reqs:
            if abort_prefill is not None:
                abort_prefill(req.uid)
            if remove_decode is not None:
                remove_decode(req)
            self._free_req_resources(req, failed=True)
            pending[req.uid] = ErrorReplyMsg(
                uid=req.uid,
                error=_OOM_ERROR_MESSAGE,
                code=_OOM_ERROR_CODE,
                status_code=503,
            )
        self._record_oom_aborts(len(reqs))

    def _fatal_cuda_context(self, oom: BaseException, context_error: BaseException) -> NoReturn:
        logger.critical_rank0(
            "CUDA CONTEXT CORRUPTION after scheduler OOM: recovery probe failed; "
            "exiting for supervised restart. original=%r probe=%r",
            oom,
            context_error,
        )
        raise SystemExit(1) from context_error

    def _synchronize_failed_forward(self, oom: BaseException) -> None:
        stream = getattr(getattr(self, "engine", None), "stream", None)
        synchronize = getattr(stream, "synchronize", None)
        if synchronize is None:
            return
        try:
            synchronize()
        except Exception as context_error:  # noqa: BLE001
            self._fatal_cuda_context(oom, context_error)

    def _restore_failed_request_lengths(self, forward_input: ForwardInput) -> None:
        """Restore schedule-time logical lengths if OOM happened after engine bookkeeping."""
        batch = forward_input.batch
        if batch.is_decode and getattr(batch, "mtp_verify", False):
            req = batch.reqs[0]
            req.cached_len = int(batch.mtp_original_cached_len)
            req.device_len = int(batch.mtp_allocated_end)
            return
        if batch.is_decode:
            try:
                allocated_lens = forward_input.write_tuple[1][: len(batch.reqs)].tolist()
            except (AttributeError, IndexError, TypeError):
                return
            for req, allocated_len in zip(batch.reqs, allocated_lens, strict=True):
                allocated_len = int(allocated_len)
                if allocated_len >= 1:
                    req.cached_len = allocated_len - 1
                    req.device_len = allocated_len
            return
        chunked_lens = getattr(batch, "_oom_chunked_device_lens", {})
        try:
            allocated_lens = forward_input.write_tuple[1][: len(batch.reqs)].tolist()
        except (AttributeError, IndexError, TypeError):
            allocated_lens = [-1] * len(batch.reqs)
        for req, allocated_len in zip(batch.reqs, allocated_lens, strict=True):
            allocated_len = int(allocated_len)
            if allocated_len >= 0:
                req.device_len = allocated_len
            elif req.uid in chunked_lens:
                req.device_len = chunked_lens[req.uid]

    def _clear_cache_and_probe_cuda(self, oom: BaseException) -> None:
        probe = None
        try:
            torch.cuda.empty_cache()
            probe = torch.empty(1, dtype=torch.uint8, device=self.device)
        except Exception as context_error:  # noqa: BLE001
            self._fatal_cuda_context(oom, context_error)
        finally:
            del probe

    def _recover_forward_oom(
        self, forward_input: ForwardInput, oom: BaseException
    ) -> ForwardData | None:
        batch = forward_input.batch
        logger.warning_rank0(
            "Scheduler %s forward OOM for request(s) %s; starting request-level recovery: %r",
            batch.phase,
            [req.uid for req in batch.reqs],
            oom,
        )
        self._synchronize_failed_forward(oom)
        try:
            self._restore_failed_request_lengths(forward_input)
        except Exception as context_error:  # noqa: BLE001
            self._fatal_cuda_context(oom, context_error)

        if batch.is_prefill:
            self._abort_oom_requests(batch.reqs)
            self._clear_cache_and_probe_cuda(oom)
            return None

        youngest = max(
            batch.reqs,
            key=lambda req: (
                getattr(req, "admission_order", 0),
                getattr(req, "uid", 0),
            ),
        )
        self._abort_oom_requests([youngest])
        self._clear_cache_and_probe_cuda(oom)
        remaining = [req for req in batch.reqs if req is not youngest]
        if not remaining:
            return None

        retry_input = None
        try:
            retry_input = self._prepare_decode_retry(remaining)
            return retry_input, self._forward(retry_input)
        except _OOM_ERRORS as retry_oom:
            logger.warning_rank0(
                "Scheduler decode forward OOM retry failed for request(s) %s; "
                "aborting the remaining batch: %r",
                [req.uid for req in remaining],
                retry_oom,
            )
            self._synchronize_failed_forward(retry_oom)
            if retry_input is not None:
                try:
                    self._restore_failed_request_lengths(retry_input)
                except Exception as context_error:  # noqa: BLE001
                    self._fatal_cuda_context(retry_oom, context_error)
            self._abort_oom_requests(remaining)
            self._clear_cache_and_probe_cuda(retry_oom)
            return None

    def _forward_with_oom_guard(self, forward_input: ForwardInput) -> ForwardData | None:
        try:
            return forward_input, self._forward(forward_input)
        except _OOM_ERRORS as oom:
            return self._recover_forward_oom(forward_input, oom)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        forward_output = self.engine.forward_batch(batch, sample_args)
        if getattr(batch, "mtp_verify", False):
            req = batch.reqs[0]
            start = int(batch.mtp_original_device_len)
            count = forward_output.next_tokens_gpu.numel()
            self.token_pool[req.table_idx, start : start + count] = (
                forward_output.next_tokens_gpu
            )
            committed = int(batch.mtp_original_cached_len) + count
            self.cache_manager.rollback_paged_tail(
                req, committed, int(batch.mtp_allocated_end)
            )
        else:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        if batch.is_prefill:
            admit_reqs = getattr(self.decode_manager, "admit_reqs", None)
            if admit_reqs is not None:
                admit_reqs(batch.reqs)
            else:
                self.decode_manager.filter_reqs(batch.reqs)
        else:
            self.decode_manager.filter_reqs(batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
