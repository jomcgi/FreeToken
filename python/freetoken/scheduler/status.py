from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from freetoken.core import Batch


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    disk_prefix_store: object | None = None
    _last_prefill_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)
    _decode_drafted_tokens: int = field(default=0, init=False)
    _decode_accepted_tokens: int = field(default=0, init=False)
    _decode_timing_count: int = field(default=0, init=False)
    _decode_timing_totals: dict[str, float] = field(default_factory=dict, init=False)
    _constrained_requests: int = field(default=0, init=False)
    _mask_us: float = field(default=0.0, init=False)
    oom_aborts: int = field(default=0, init=False)
    client_aborts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def report_batch(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        queue_priority_bands: dict[str, int] | None = None,
        max_wait_seconds: float = 0.0,
    ) -> None:
        self._constrained_requests += getattr(batch, "constrained_requests", 0)
        self._mask_us += getattr(batch, "mask_us", 0.0)
        if batch.is_prefill:
            self._report_prefill(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                queue_priority_bands=queue_priority_bands,
                max_wait_seconds=max_wait_seconds,
            )
        elif batch.is_decode:
            self._report_decode(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                queue_priority_bands=queue_priority_bands,
                max_wait_seconds=max_wait_seconds,
            )

    def record_oom_aborts(self, count: int) -> None:
        self.oom_aborts += count

    def record_client_abort(self) -> None:
        self.client_aborts += 1

    def _report_prefill(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        queue_priority_bands: dict[str, int] | None = None,
        max_wait_seconds: float = 0.0,
    ) -> None:
        now = self.clock()
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        # Read the schedule-time snapshot: by report time the forward's complete_one() has
        # advanced each req's cached_len to device_len, so reading the reqs here would log
        # decode-state values (#new-token == #reqs, #cached-token == full prompt).
        new_tokens = batch.log_new_tokens
        cached_tokens = batch.log_cached_tokens
        input_throughput = new_tokens / gap if gap > 0 else 0.0
        if self.disk_prefix_store is not None:
            self.disk_prefix_store.observe_prefill_rate(input_throughput)
        self.log(
            f"Prefill batch, "
            f"#new-seq: {len(batch.reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"client_aborts: {self.client_aborts}, "
            f"{_priority_queue_msg(queue_priority_bands, max_wait_seconds)}"
            f"{', ' if queue_priority_bands is not None else ''}"
            f"input throughput (token/s): {input_throughput:.2f}"
            f"{self._disk_prefix_msg()}"
        )

    def _report_decode(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        queue_priority_bands: dict[str, int] | None = None,
        max_wait_seconds: float = 0.0,
    ) -> None:
        self._decode_forward_count += 1
        self._decode_generated_tokens += getattr(batch, "generated_tokens", 0) or len(
            batch.reqs
        )
        self._decode_drafted_tokens += getattr(batch, "mtp_drafted", 0)
        self._decode_accepted_tokens += getattr(batch, "mtp_accepted", 0)
        timing = getattr(batch, "moe_step_timing", None)
        if timing is not None:
            self._decode_timing_count += 1
            for name in (
                "cpu_head_us",
                "gpu_mid_us",
                "cpu_tail_us",
                "overlap_us",
                "cpu_wake_us",
                "cpu_groups_us",
                "cpu_gil_us",
                "cpu_precb_us",
                "cpu_notify_us",
                "cpu_coord_us",
                "cpu_gpu_in_us",
                "cpu_gpu_out_us",
                "cpu_d2h_us",
                "cpu_h2d_us",
                "cpu_compute_us",
                "cpu_signal_us",
                "cpu_layers_per_step",
                "cpu_expert_bytes_per_step",
            ):
                self._decode_timing_totals[name] = (
                    self._decode_timing_totals.get(name, 0.0)
                    + float(timing.get(name, 0.0))
                )
        if getattr(batch, "mtp_drafted", 0):
            self.log(
                "MTP verify window, route: decode, "
                f"width: {batch.mtp_drafted + 1}, "
                f"accepted: {batch.mtp_accepted}, "
                f"verify_us: {getattr(batch, 'mtp_verify_us', 0.0):.0f}, "
                f"snapshot_us: {getattr(batch, 'mtp_snapshot_us', 0.0):.0f}, "
                f"draft_us: {getattr(batch, 'mtp_draft_us', 0.0):.0f}"
            )
        if self._decode_forward_count % self.decode_log_interval != 0:
            return

        now = self.clock()
        gap = now - self._last_decode_time
        self._last_decode_time = now
        gen_throughput = self._decode_generated_tokens / gap if gap > 0 else 0.0
        tokens_per_step = self._decode_generated_tokens / self.decode_log_interval
        acceptance_rate = (
            self._decode_accepted_tokens / self._decode_drafted_tokens
            if self._decode_drafted_tokens else 0.0
        )
        self._decode_generated_tokens = 0
        drafted = self._decode_drafted_tokens
        accepted = self._decode_accepted_tokens
        self._decode_drafted_tokens = 0
        self._decode_accepted_tokens = 0
        timing_msg = ""
        if self._decode_timing_count:
            count = self._decode_timing_count
            timing_msg = "".join(
                f", {name}: {self._decode_timing_totals.get(name, 0.0) / count:.0f}"
                for name in (
                    "cpu_head_us",
                    "gpu_mid_us",
                    "cpu_tail_us",
                    "overlap_us",
                    "cpu_wake_us",
                    "cpu_groups_us",
                    "cpu_gil_us",
                    "cpu_precb_us",
                    "cpu_notify_us",
                    "cpu_coord_us",
                    "cpu_gpu_in_us",
                    "cpu_gpu_out_us",
                    "cpu_d2h_us",
                    "cpu_h2d_us",
                    "cpu_compute_us",
                    "cpu_signal_us",
                    "cpu_layers_per_step",
                    "cpu_expert_bytes_per_step",
                )
            )
        self._decode_timing_count = 0
        self._decode_timing_totals.clear()
        constrained_requests = self._constrained_requests
        mask_us = self._mask_us
        self._constrained_requests = 0
        self._mask_us = 0.0
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"drafted: {drafted}, accepted: {accepted}, "
            f"acceptance rate: {acceptance_rate:.4f}, "
            f"tokens/step: {tokens_per_step:.2f}, "
            f"oom_aborts: {self.oom_aborts}, "
            f"client_aborts: {self.client_aborts}, "
            f"#queue-req: {queue_reqs}"
            f"{', ' if queue_priority_bands is not None else ''}"
            f"{_priority_queue_msg(queue_priority_bands, max_wait_seconds)}"
            f", constrained_requests: {constrained_requests}, mask_us: {mask_us:.0f}"
            f"{timing_msg}"
            f"{self._disk_prefix_msg()}"
        )

    def _disk_prefix_msg(self) -> str:
        if self.disk_prefix_store is None:
            return ""
        stats = self.disk_prefix_store.stats()
        return (
            f", disk_prefix hits: {stats['hits']}, misses: {stats['misses']}, "
            f"bytes_restored: {stats['bytes_restored']}, "
            f"restore_ms: {stats['restore_ms']:.2f}, "
            f"restore_eager_ms: {stats['restore_eager_ms']:.2f}, "
            f"blocks_faulted: {stats['blocks_faulted']}, "
            f"blocks_streamed: {stats['blocks_streamed']}, "
            f"first_token_after_restore_ms: "
            f"{stats['first_token_after_restore_ms']:.2f}, "
            f"estimated prefill_ms_saved: {stats['prefill_ms_saved']:.2f}, "
            f"write_drops: {stats['write_drops']}, "
            f"stale_format: {stats['stale_format']}, "
            f"corrupt: {stats['corrupt_entries']}, "
            f"fingerprint_mismatch: {stats['fingerprint_mismatches']}, "
            f"harness_anchor_persisted: {stats.get('harness_anchor_persisted', 0)}, "
            f"harness_anchor_skipped_final_chunk: "
            f"{stats.get('harness_anchor_skipped_final_chunk', 0)}, "
            f"harness_anchor_skipped_no_store: "
            f"{stats.get('harness_anchor_skipped_no_store', 0)}, "
            f"harness_anchor_skipped_unaligned: "
            f"{stats.get('harness_anchor_skipped_unaligned', 0)}"
        )


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


def _mamba_msg(mamba_slots: tuple[int, int] | None) -> str:
    """GDN-state (mamba) pool occupancy for hybrid models; empty for the rest."""
    if mamba_slots is None:
        return ""
    used, total = mamba_slots
    return f"#mamba-slot: {used}/{total}, mamba usage: {_usage_ratio(used, total):.2f}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window (swa) pool occupancy for SWA models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"#swa-token: {used}/{total}, swa usage: {_usage_ratio(used, total):.2f}, "


def _priority_queue_msg(bands: dict[str, int] | None, max_wait_seconds: float) -> str:
    if bands is None:
        return ""
    return (
        "#queue-priority: "
        f"negative={bands.get('negative', 0)}/"
        f"zero={bands.get('zero', 0)}/"
        f"positive={bands.get('positive', 0)}, "
        f"max_wait_seconds: {max_wait_seconds:.2f}"
    )
