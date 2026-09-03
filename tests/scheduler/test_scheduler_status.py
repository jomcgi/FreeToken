from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.status import SchedulerStatusReporter, _usage_ratio
from freetoken.scheduler.scheduler import _pinned_hot_status_fragment


def _reporter(interval=40):
    logs: list[str] = []
    clock = {"t": 0.0}
    rep = SchedulerStatusReporter(
        log=logs.append,
        clock=lambda: clock["t"],
        decode_log_interval=interval,
    )
    return rep, logs, clock


def _req(extend, cached):
    return SimpleNamespace(extend_len=extend, cached_len=cached)


def _prefill_batch(new_tokens, cached_tokens, n_seqs):
    # The reporter must read the schedule-time snapshot (log_new_tokens/log_cached_tokens),
    # NOT the live reqs: by report time forward's complete_one() has advanced them to
    # decode state. Live reqs here carry deliberately-wrong values to prove that.
    reqs = [_req(extend=1, cached=10_000) for _ in range(n_seqs)]
    return SimpleNamespace(
        is_prefill=True, is_decode=False, reqs=reqs,
        log_new_tokens=new_tokens, log_cached_tokens=cached_tokens,
    )


def _decode_batch(n):
    return SimpleNamespace(is_prefill=False, is_decode=True, reqs=[_req(1, 0) for _ in range(n)])


def test_pinned_hot_status_fragment_renders_protected_and_refetch_stats():
    fragment = _pinned_hot_status_fragment({
        "pinned_hot_pair_rate": 0.75,
        "pinned_missing_per_step": 12.5,
        "pinned_h2d_bytes_per_step": 3.25 * 2**20,
    })
    assert "pinned_hot_pair_rate: 75.00%" in fragment
    assert "pinned_missing/step: 12.50" in fragment
    assert "pinned_h2d_mb/step: 3.25" in fragment


def test_prefill_line_reports_tokens_and_throughput():
    rep, logs, clock = _reporter()
    clock["t"] = 0.5  # 30 new tokens over 0.5s -> 60 tok/s
    rep.report_batch(
        _prefill_batch(new_tokens=30, cached_tokens=12, n_seqs=2),
        running_reqs=2, queue_reqs=1, kv_used_pages=50, kv_total_pages=200, page_size=16,
    )
    assert len(logs) == 1
    line = logs[0]
    assert "#new-seq: 2" in line
    assert "#new-token: 30" in line  # snapshot, not the live reqs' extend_len (1 each)
    assert "#cached-token: 12" in line  # snapshot, not the live reqs' cached_len (10000 each)
    assert "token usage: 0.25" in line
    assert "#running-req: 2" in line
    assert "#queue-req: 1" in line
    assert "input throughput (token/s): 60.00" in line


def test_status_line_reports_priority_bands_and_max_wait():
    rep, logs, clock = _reporter()
    clock["t"] = 1.0
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1,
        queue_reqs=6,
        kv_used_pages=1,
        kv_total_pages=10,
        page_size=1,
        queue_priority_bands={"negative": 1, "zero": 3, "positive": 2},
        max_wait_seconds=12.345,
    )

    assert "#queue-priority: negative=1/zero=3/positive=2" in logs[-1]
    assert "max_wait_seconds: 12.35" in logs[-1]


def test_mamba_slots_reported_only_when_provided():
    rep, logs, clock = _reporter()
    clock["t"] = 1.0
    # non-hybrid (mamba_slots=None): no mamba field
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
    )
    assert "mamba" not in logs[-1]
    # hybrid: #mamba-slot: used/total and usage ratio
    clock["t"] = 2.0
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        mamba_slots=(37, 256),
    )
    assert "#mamba-slot: 37/256" in logs[-1]
    assert "mamba usage: 0.14" in logs[-1]


def test_swa_tokens_reported_only_when_provided():
    rep, logs, clock = _reporter(interval=1)
    clock["t"] = 1.0
    # non-SWA (swa_tokens=None): no swa field
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
    )
    assert "swa" not in logs[-1]
    # SWA: #swa-token: used/total and usage ratio, on both prefill and decode lines
    clock["t"] = 2.0
    rep.report_batch(
        _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1),
        running_reqs=1, queue_reqs=0, kv_used_pages=1, kv_total_pages=10, page_size=1,
        swa_tokens=(8448, 76800),
    )
    assert "#swa-token: 8448/76800" in logs[-1]
    assert "swa usage: 0.11" in logs[-1]
    clock["t"] = 3.0
    rep.report_batch(_decode_batch(1), running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1,
                     swa_tokens=(8448, 76800))
    assert "#swa-token: 8448/76800" in logs[-1]
    assert "swa usage: 0.11" in logs[-1]


def test_decode_lines_are_throttled_to_every_nth_forward():
    rep, logs, clock = _reporter(interval=3)
    for i, t in enumerate((1.0, 1.5), start=1):
        clock["t"] = t
        rep.report_batch(_decode_batch(2), running_reqs=2, queue_reqs=0,
                         kv_used_pages=60, kv_total_pages=200, page_size=16)
        assert logs == [], f"should not log before the interval (forward {i})"
    clock["t"] = 2.0  # 3rd forward -> log; 6 tokens over 2.0s gap -> 3 tok/s
    rep.report_batch(_decode_batch(2), running_reqs=2, queue_reqs=4,
                     kv_used_pages=62, kv_total_pages=200, page_size=16)
    assert len(logs) == 1
    line = logs[0]
    assert "#running-req: 2" in line
    assert "#queue-req: 4" in line
    assert "#token: 992" in line  # 62 pages * 16
    assert "token usage: 0.31" in line
    assert "gen throughput (token/s): 3.00" in line


def test_status_lines_include_cumulative_client_aborts():
    rep, logs, clock = _reporter(interval=1)
    rep.record_client_abort()
    rep.record_client_abort()

    clock["t"] = 1.0
    rep.report_batch(
        _prefill_batch(new_tokens=1, cached_tokens=0, n_seqs=1),
        running_reqs=1,
        queue_reqs=0,
        kv_used_pages=1,
        kv_total_pages=2,
        page_size=1,
    )
    assert "client_aborts: 2" in logs[-1]

    clock["t"] = 2.0
    rep.report_batch(
        _decode_batch(1),
        running_reqs=1,
        queue_reqs=0,
        kv_used_pages=1,
        kv_total_pages=2,
        page_size=1,
    )
    assert "client_aborts: 2" in logs[-1]


def test_decode_counter_resets_each_interval():
    rep, logs, clock = _reporter(interval=2)
    clock["t"] = 1.0
    rep.report_batch(_decode_batch(5), running_reqs=5, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    clock["t"] = 2.0  # first emission: 10 tokens over 2.0s -> 5 tok/s
    rep.report_batch(_decode_batch(5), running_reqs=5, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    assert "gen throughput (token/s): 5.00" in logs[-1]
    # next window is measured from the previous emission, with a reset token count
    clock["t"] = 3.0
    rep.report_batch(_decode_batch(3), running_reqs=3, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    clock["t"] = 4.0  # 6 tokens over (4.0-2.0)=2.0s -> 3 tok/s
    rep.report_batch(_decode_batch(3), running_reqs=3, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    assert "gen throughput (token/s): 3.00" in logs[-1]


def test_zero_gap_and_zero_total_are_guarded():
    rep, logs, clock = _reporter(interval=1)
    # gap == 0 (clock unchanged since construction) and total == 0 must not raise
    rep.report_batch(_decode_batch(4), running_reqs=4, queue_reqs=0,
                     kv_used_pages=0, kv_total_pages=0, page_size=1)
    line = logs[-1]
    assert "gen throughput (token/s): 0.00" in line
    assert "token usage: 0.00" in line
    assert "#token: 0" in line  # owned-KV (dsv4) reports 0/0 pages


def test_decode_mtp_acceptance_stats():
    rep, logs, clock = _reporter(interval=1)
    batch = _decode_batch(1)
    batch.mtp_drafted = 1
    batch.mtp_accepted = 1
    batch.generated_tokens = 2
    clock["t"] = 2.0
    rep.report_batch(
        batch,
        running_reqs=1,
        queue_reqs=0,
        kv_used_pages=1,
        kv_total_pages=2,
        page_size=1,
    )
    line = logs[-1]
    assert "drafted: 1, accepted: 1" in line
    assert "acceptance rate: 1.0000" in line
    assert "tokens/step: 2.00" in line


def test_decode_mtp_timing_is_logged_per_window():
    rep, logs, clock = _reporter(interval=40)
    batch = _decode_batch(1)
    batch.mtp_drafted = 1
    batch.mtp_accepted = 0
    batch.generated_tokens = 1
    batch.mtp_verify_us = 2300
    batch.mtp_snapshot_us = 180
    batch.mtp_draft_us = 700
    clock["t"] = 1.0

    rep.report_batch(
        batch,
        running_reqs=1,
        queue_reqs=0,
        kv_used_pages=1,
        kv_total_pages=2,
        page_size=1,
    )

    assert logs == [
        "MTP verify window, route: decode, width: 2, accepted: 0, "
        "verify_us: 2300, snapshot_us: 180, draft_us: 700"
    ]


def test_decode_moe_step_timing_is_averaged_on_status_line():
    rep, logs, clock = _reporter(interval=2)
    first = _decode_batch(1)
    first.moe_step_timing = {
        "cpu_head_us": 1000,
        "gpu_mid_us": 2000,
        "cpu_tail_us": 3000,
        "overlap_us": 400,
        "cpu_wake_us": 100,
        "cpu_groups_us": 20,
        "cpu_gil_us": 30,
        "cpu_precb_us": 40,
        "cpu_notify_us": 10,
        "cpu_coord_us": 15,
        "cpu_gpu_in_us": 250,
        "cpu_gpu_out_us": 350,
        "cpu_d2h_us": 500,
        "cpu_h2d_us": 300,
        "cpu_compute_us": 1800,
        "cpu_signal_us": 40,
        "cpu_layers_per_step": 27,
        "cpu_expert_bytes_per_step": 10_000,
    }
    second = _decode_batch(1)
    second.moe_step_timing = {
        "cpu_head_us": 1200,
        "gpu_mid_us": 2400,
        "cpu_tail_us": 3600,
        "overlap_us": 600,
        "cpu_wake_us": 140,
        "cpu_groups_us": 30,
        "cpu_gil_us": 40,
        "cpu_precb_us": 50,
        "cpu_notify_us": 20,
        "cpu_coord_us": 25,
        "cpu_gpu_in_us": 350,
        "cpu_gpu_out_us": 450,
        "cpu_d2h_us": 700,
        "cpu_h2d_us": 500,
        "cpu_compute_us": 2200,
        "cpu_signal_us": 60,
        "cpu_layers_per_step": 29,
        "cpu_expert_bytes_per_step": 14_000,
    }
    clock["t"] = 1.0
    rep.report_batch(first, running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)
    clock["t"] = 2.0
    rep.report_batch(second, running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)

    line = logs[-1]
    assert "cpu_head_us: 1100" in line
    assert "gpu_mid_us: 2200" in line
    assert "cpu_tail_us: 3300" in line
    assert "overlap_us: 500" in line
    assert "cpu_wake_us: 120" in line
    assert "cpu_groups_us: 25" in line
    assert "cpu_gil_us: 35" in line
    assert "cpu_precb_us: 45" in line
    assert "cpu_notify_us: 15" in line
    assert "cpu_coord_us: 20" in line
    assert "cpu_gpu_in_us: 300" in line
    assert "cpu_gpu_out_us: 400" in line
    assert "cpu_d2h_us: 600" in line
    assert "cpu_h2d_us: 400" in line
    assert "cpu_compute_us: 2000" in line
    assert "cpu_signal_us: 50" in line
    assert "cpu_layers_per_step: 28" in line
    assert "cpu_expert_bytes_per_step: 12000" in line


def test_decode_line_omits_moe_step_timing_when_disabled():
    rep, logs, clock = _reporter(interval=1)
    clock["t"] = 1.0
    rep.report_batch(_decode_batch(1), running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)
    line = logs[-1]
    for field in (
        "cpu_head_us",
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
        assert field not in line


def test_guided_decoding_stats_accumulate_and_reset_per_interval():
    rep, logs, clock = _reporter(interval=2)
    prefill = _prefill_batch(new_tokens=10, cached_tokens=0, n_seqs=1)
    prefill.constrained_requests = 1
    prefill.mask_us = 25.0
    rep.report_batch(prefill, running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)
    first = _decode_batch(1)
    first.mask_us = 10.0
    second = _decode_batch(1)
    second.mask_us = 15.0
    clock["t"] = 1.0
    rep.report_batch(first, running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)
    clock["t"] = 2.0
    rep.report_batch(second, running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)

    assert "constrained_requests: 1, mask_us: 50" in logs[-1]

    clock["t"] = 3.0
    rep.report_batch(_decode_batch(1), running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)
    clock["t"] = 4.0
    rep.report_batch(_decode_batch(1), running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=2, page_size=1)
    assert "constrained_requests: 0, mask_us: 0" in logs[-1]


def test_interval_is_clamped_to_at_least_one():
    rep, _, _ = _reporter(interval=0)
    assert rep.decode_log_interval == 1
    rep_neg, _, _ = _reporter(interval=-5)
    assert rep_neg.decode_log_interval == 1


def test_usage_ratio_guard():
    assert _usage_ratio(0, 0) == 0.0
    assert _usage_ratio(5, 0) == 0.0
    assert _usage_ratio(5, 10) == 0.5


def test_disk_status_includes_harness_anchor_counters():
    stats = {
        "hits": 0,
        "misses": 0,
        "bytes_restored": 0,
        "restore_ms": 0.0,
        "restore_eager_ms": 0.0,
        "blocks_faulted": 0,
        "blocks_streamed": 0,
        "first_token_after_restore_ms": 0.0,
        "prefill_ms_saved": 0.0,
        "write_drops": 0,
        "stale_format": 0,
        "corrupt_entries": 0,
        "fingerprint_mismatches": 0,
        "harness_anchor_persisted": 3,
        "harness_anchor_skipped_final_chunk": 4,
        "harness_anchor_skipped_no_store": 5,
        "harness_anchor_skipped_unaligned": 6,
    }
    reporter = SchedulerStatusReporter(
        log=lambda _line: None,
        disk_prefix_store=SimpleNamespace(stats=lambda: stats),
    )

    line = reporter._disk_prefix_msg()

    assert "harness_anchor_persisted: 3" in line
    assert "harness_anchor_skipped_final_chunk: 4" in line
    assert "harness_anchor_skipped_no_store: 5" in line
    assert "harness_anchor_skipped_unaligned: 6" in line
