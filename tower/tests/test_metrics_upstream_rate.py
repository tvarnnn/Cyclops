"""Upstream observation-rate instrumentation.

The 2026-08-21 first physical-glasses remote run delivered 0.8 fps to the
Tower. Diagnosing that required hand arithmetic over log lines: dividing
the last observed `seq` by the session duration to recover the capture
rate, then dividing again by the frame count to recover the sender's
sampling stride. These tests lock in metrics that make the Tower report
both numbers directly, so a future run is diagnosable without inferring
sender behavior from `seq_gap_total`.

Both figures are derived from `source_seq` (the DAT/capture frame index),
which falls back to `seq` for every sender that exists today -- see
tower/frames.py.
"""
from tower.metrics import SessionMetrics


def _fake_clock():
    counter = {"t": 0.0}

    def clock():
        return counter["t"]

    def advance(seconds):
        counter["t"] += seconds

    return clock, advance


def _record(metrics, seq, **overrides):
    kwargs = {
        "seq": seq,
        "byte_count": 100,
        "receive_to_result_ms": 1.0,
        "cv_processing_ms": 1.0,
    }
    kwargs.update(overrides)
    metrics.record_frame(**kwargs)


def test_source_seq_span_reports_the_capture_index_range_observed():
    metrics = SessionMetrics()

    _record(metrics, seq=1)
    _record(metrics, seq=31)
    _record(metrics, seq=61)

    assert metrics.snapshot()["source_seq_span"] == 60


def test_sampling_stride_avg_recovers_a_one_in_thirty_sender_stride():
    """The headline diagnostic: a sender forwarding 1-in-30 capture frames
    should report a stride of ~30, naming the throttle directly instead of
    leaving it to be inferred from a large seq_gap_total.
    """
    metrics = SessionMetrics()

    for seq in (1, 31, 61, 91):
        _record(metrics, seq=seq)

    assert metrics.snapshot()["sampling_stride_avg"] == 30.0


def test_sampling_stride_avg_is_one_when_the_sender_forwards_every_frame():
    metrics = SessionMetrics()

    for seq in (1, 2, 3, 4):
        _record(metrics, seq=seq)

    assert metrics.snapshot()["sampling_stride_avg"] == 1.0


def test_source_fps_estimate_recovers_the_upstream_capture_rate():
    """A 1-in-30 sender delivering 0.8 fps is sampling a ~24 fps capture
    stream. The Tower should say so: source_fps_estimate reads ~24 even
    though effective_fps reads ~0.8.
    """
    clock, advance = _fake_clock()
    metrics = SessionMetrics(clock=clock)

    # 24 capture frames/sec, forwarded 1-in-30 -> one frame every 1.25 s.
    for index in range(5):
        _record(metrics, seq=1 + index * 30)
        if index < 4:
            advance(1.25)

    snapshot = metrics.snapshot()

    assert snapshot["source_fps_estimate"] == 24.0
    assert snapshot["sampling_stride_avg"] == 30.0


def test_source_frame_span_s_makes_the_fps_estimate_auditable():
    """`source_fps_estimate` and `effective_fps` use DIFFERENT windows:
    the former spans first-to-last received frame, the latter spans the
    whole stream_start-bounded session. Without the first window being
    reported, the two numbers cannot be reconciled and look contradictory.
    Reporting it makes source_fps_estimate == span / span_s checkable.
    """
    clock, advance = _fake_clock()
    metrics = SessionMetrics(clock=clock)

    advance(10.0)  # idle time before the first frame -- in the session, not the frame span
    _record(metrics, seq=1)
    advance(2.0)
    _record(metrics, seq=49)
    advance(10.0)  # idle time after the last frame

    snapshot = metrics.snapshot()

    assert snapshot["source_frame_span_s"] == 2.0
    assert snapshot["source_seq_span"] == 48
    # Auditable: 48 / 2.0 == 24.0
    assert snapshot["source_fps_estimate"] == 24.0
    # And demonstrably NOT the same denominator as effective_fps, which
    # divides 2 frames by the full 22 s session.
    assert snapshot["session_duration_s"] == 22.0
    assert snapshot["effective_fps"] == 0.09


def test_source_frame_span_s_is_unavailable_with_fewer_than_two_frames():
    metrics = SessionMetrics()
    _record(metrics, seq=1)

    assert metrics.snapshot()["source_frame_span_s"] is None


def test_upstream_rate_fields_are_unavailable_before_two_frames():
    """Rule 3: a rate needs two samples. One frame must report None, not a
    misleading 0.0 that would read as "the capture stream has stopped".
    """
    metrics = SessionMetrics()
    _record(metrics, seq=1)

    snapshot = metrics.snapshot()

    assert snapshot["source_seq_span"] is None
    assert snapshot["sampling_stride_avg"] is None
    assert snapshot["source_fps_estimate"] is None


def test_upstream_rate_fields_are_unavailable_with_no_frames_at_all():
    snapshot = SessionMetrics().snapshot()

    assert snapshot["source_seq_span"] is None
    assert snapshot["sampling_stride_avg"] is None
    assert snapshot["source_fps_estimate"] is None


def test_source_fps_estimate_is_unavailable_when_no_time_has_elapsed():
    """Two frames in the same clock tick give a span of zero seconds. A
    rate is genuinely unknown there; the stride is still knowable.
    """
    clock, _advance = _fake_clock()
    metrics = SessionMetrics(clock=clock)

    _record(metrics, seq=1)
    _record(metrics, seq=31)

    snapshot = metrics.snapshot()

    assert snapshot["source_fps_estimate"] is None
    assert snapshot["sampling_stride_avg"] == 30.0


def test_upstream_rate_uses_source_seq_when_the_sender_sends_the_split():
    """Post-split senders keep `seq` as-is but carry the capture index in
    `source_seq` and a dense transmit counter in `tx_seq`. The stride must
    come from the capture index, not from the dense counter.
    """
    clock, advance = _fake_clock()
    metrics = SessionMetrics(clock=clock)

    for index in range(3):
        advance(1.0)
        _record(
            metrics,
            seq=index + 1,
            source_seq=1 + index * 15,
            tx_seq=index + 1,
        )

    snapshot = metrics.snapshot()

    assert snapshot["source_seq_span"] == 30
    assert snapshot["sampling_stride_avg"] == 15.0
    assert snapshot["tx_seq_gap_total"] == 0


def test_non_advancing_source_seq_leaves_the_rate_unavailable():
    """A sender restart can make the capture index regress. That is not a
    negative frame rate -- it is an unmeasurable one.
    """
    metrics = SessionMetrics()

    _record(metrics, seq=500)
    _record(metrics, seq=2)

    snapshot = metrics.snapshot()

    assert snapshot["source_fps_estimate"] is None
    assert snapshot["sampling_stride_avg"] is None


def test_replaying_the_first_physical_remote_run_names_its_own_bottleneck():
    """Regression guard tied to real measured hardware data.

    The 2026-08-21 first physical-glasses remote run (Ray-Ban Meta Gen 2 ->
    DAT -> iPhone -> Tailscale -> Tower, ~2h apart) reported:
    session_duration_s 78.999, frames_received 63, effective_fps 0.8,
    seq_gap_total 1797, with the last observed seq around #1860.

    Replaying that shape must surface the cause directly: a ~24 fps
    capture stream sampled 1-in-30. Before these fields existed, reaching
    that conclusion took dividing 1860 by 79 and then by 63 by hand.

    Replayed with an exact stride of 30 (the real run averaged 29.98, so
    its seq_gap_total came out at 1797 rather than the 1798 here) -- the
    diagnostic conclusion is identical either way.
    """
    clock, advance = _fake_clock()
    metrics = SessionMetrics(clock=clock)

    frames_received = 63
    stride = 30
    duration_s = 78.999

    for index in range(frames_received):
        if index:
            advance(duration_s / (frames_received - 1))
        _record(metrics, seq=1 + index * stride)

    snapshot = metrics.snapshot()

    # What the run reported, and what made it look alarming.
    assert snapshot["frames_received"] == 63
    assert snapshot["effective_fps"] == 0.8
    assert snapshot["seq_gap_total"] == 1798

    # What the run could not say for itself, and now can.
    assert snapshot["sampling_stride_avg"] == 30.0
    assert snapshot["source_fps_estimate"] == 23.54

    # The stride fully accounts for the gap total: nothing is left over to
    # attribute to transit loss.
    assert snapshot["seq_gap_total"] == snapshot["source_seq_span"] - (
        snapshot["frames_received"] - 1
    )


def test_source_seq_defaults_to_seq_for_legacy_senders():
    """Backward compatibility: record_frame() callers that predate the
    split pass only `seq`, and must get identical stride accounting.
    """
    metrics = SessionMetrics()

    _record(metrics, seq=1)
    _record(metrics, seq=61)

    assert metrics.snapshot()["sampling_stride_avg"] == 60.0
