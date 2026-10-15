"""Tower-side prep for the source_seq/tx_seq protocol split.

Additive and backward-compatible: a sender that only sends `seq` (every
sender that exists today) must behave exactly as before. See
07-PLATFORM-CONSTRAINTS.md Limitation 9.
"""
import base64
import io

from PIL import Image

from tower.frames import parse_and_decode_frame
from tower.metrics import SessionMetrics


def _jpeg_base64(width: int = 8, height: int = 8) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _message(**overrides) -> dict:
    message = {
        "type": "frame",
        "seq": 1,
        "width": 8,
        "height": 8,
        "format": "jpeg",
        "data": _jpeg_base64(),
    }
    message.update(overrides)
    return message


def test_legacy_message_without_new_fields_still_parses():
    frame = parse_and_decode_frame(_message(seq=7))

    assert frame.seq == 7
    assert frame.source_seq == 7  # falls back to seq
    assert frame.tx_seq is None  # genuinely absent, not zero


def test_source_seq_overrides_seq_when_present():
    frame = parse_and_decode_frame(_message(seq=7, source_seq=210, tx_seq=7))

    assert frame.source_seq == 210
    assert frame.tx_seq == 7


def test_tx_seq_alone_is_accepted():
    frame = parse_and_decode_frame(_message(seq=30, tx_seq=2))

    assert frame.source_seq == 30
    assert frame.tx_seq == 2


def test_tx_seq_gap_is_unavailable_when_sender_does_not_send_tx_seq():
    # Rule 3: unknown values remain unavailable, never a misleading 0.
    # Reporting 0 here would falsely assert "no transit loss observed".
    metrics = SessionMetrics()
    metrics.record_frame(
        seq=1, byte_count=1, receive_to_result_ms=1.0, cv_processing_ms=1.0
    )
    metrics.record_frame(
        seq=30, byte_count=1, receive_to_result_ms=1.0, cv_processing_ms=1.0
    )

    snapshot = metrics.snapshot()

    assert snapshot["seq_gap_total"] == 28
    assert snapshot["tx_seq_gap_total"] is None


def test_tx_seq_gap_is_counted_when_sender_sends_tx_seq():
    metrics = SessionMetrics()
    metrics.record_frame(
        seq=1, byte_count=1, receive_to_result_ms=1.0, cv_processing_ms=1.0, tx_seq=1
    )
    # Sender transmitted tx_seq 2 and 3; the Tower never saw them.
    metrics.record_frame(
        seq=120, byte_count=1, receive_to_result_ms=1.0, cv_processing_ms=1.0, tx_seq=4
    )

    snapshot = metrics.snapshot()

    assert snapshot["tx_seq_gap_total"] == 2


def test_dense_tx_seq_reports_no_transit_loss_despite_sparse_source_seq():
    # The whole point of the split: intentional 1-in-30 source sampling
    # must NOT look like transit loss.
    metrics = SessionMetrics()
    for index, source in enumerate([1, 30, 60, 90], start=1):
        metrics.record_frame(
            seq=source,
            byte_count=1,
            receive_to_result_ms=1.0,
            cv_processing_ms=1.0,
            tx_seq=index,
        )

    snapshot = metrics.snapshot()

    # 28 + 29 + 29 -- all of it intentional source sampling, not loss.
    assert snapshot["seq_gap_total"] == 86
    assert snapshot["tx_seq_gap_total"] == 0  # nothing actually lost
