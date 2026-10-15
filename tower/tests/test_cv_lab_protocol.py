"""The CV Lab wire contract, pinned so changing it is a deliberate act.

Everything a fresh iOS client would have to get right: discovery, the
three surfaces that must agree, the control vocabulary, every refusal
reason, the provenance on every frame, and the document that tells an
engineer who has never read this Python how to implement it.
"""

import json
import pathlib

import pytest

from tests.cv_lab_fixtures import (  # noqa: F401
    _close_cv_lab_clients,
    command,
    drain,
    frame,
    jpeg_message,
    make_client,
    pump,
)
from tower.cv_lab.contracts import (
    CONTROL_CONTRACT,
    FRAME_RESULT_CONTRACT,
    LIFECYCLE_STATES,
    STATUS_CONTRACT,
    TIME_BASIS,
)
from tower.results import registry
from tower.results.contracts import ENVELOPE_CONTRACT

DOCUMENT = pathlib.Path("docs/contracts/EXPERIMENTAL-CV-LAB.md")


# -- discovery ----------------------------------------------------------


def test_the_lab_is_offered_in_the_capability_declaration(monkeypatch):
    """iOS 0.1: a cartridge the Tower says nothing about is "not built
    yet". Until now that was the CV Lab, and it was wrong -- the Lab has
    run on the live frame path since V0.9."""
    client = make_client(monkeypatch)
    declaration = client.get("/cartridges").json()

    offer = next(
        entry
        for entry in declaration["cartridges"]
        if entry["cartridge"] == "experimental_cv"
    )
    assert offer["result_type"] == "status"
    assert offer["contract"] == STATUS_CONTRACT
    assert offer["available"] is True
    assert offer["snapshot_only"] is True
    assert "experimental_cv" not in {
        entry["cartridge"] for entry in declaration["not_offered"]
    }


def test_the_world_builder_offer_is_still_first(monkeypatch):
    """Adding an offer must not renumber the one a shipped client indexes."""
    client = make_client(monkeypatch)
    declaration = client.get("/cartridges").json()
    assert declaration["cartridges"][0]["cartridge"] == "world_builder"


def test_a_tower_with_no_lab_offers_the_contract_and_reports_it_unavailable():
    """The third state in ios/docs/agent-handoffs/IOS-TO-TOWER.md §0.1, which
    must not collapse into the first: "offered, implemented, unreachable ->
    connect" is a different instruction to a person than "not built yet"."""
    declaration = registry.declare(None, cv_lab=None)
    offer = next(
        entry
        for entry in declaration["cartridges"]
        if entry["cartridge"] == "experimental_cv"
    )
    assert offer["contract"] == STATUS_CONTRACT
    assert offer["available"] is False
    assert "without a CV Lab" in offer["unavailable_reason"]


def test_a_lab_that_cannot_report_availability_is_reported_unavailable():
    """A declaration must never fail because a subsystem is unwell."""

    class _Broken:
        def availability(self):
            raise RuntimeError("no")

    offer = next(
        entry
        for entry in registry.declare(None, cv_lab=_Broken())["cartridges"]
        if entry["cartridge"] == "experimental_cv"
    )
    assert offer["available"] is False


def test_the_registry_refuses_every_unoffered_pair(monkeypatch):
    client = make_client(monkeypatch)
    lab = client.app.state.cv_lab
    assert registry.find_offer(None, "experimental_cv", "status", cv_lab=lab) is not None
    assert registry.find_offer(None, "experimental_cv", "metrics", cv_lab=lab) is None
    assert registry.find_offer(None, "experimental_cv", "status", cv_lab=None)[
        "available"
    ] is False


# -- three surfaces, one document ---------------------------------------


def test_http_socket_and_result_channel_serve_the_same_document(monkeypatch):
    """Three surfaces onto one function, never three that agree today.

    A client that read the catalog over HTTP and then subscribed would
    otherwise have to reconcile two pictures of one Lab, and the first
    thing that goes wrong with two pictures is that one of them is older.
    """
    client = make_client(monkeypatch, "edge_detection")

    over_http = client.get("/cv-lab").json()
    with client.websocket_connect("/ws") as ws:
        over_socket = command(ws, {"type": "cv_lab_status"})
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "experimental_cv",
                "result_type": "status",
            }
        )
        drain(ws, "result_subscribed")
        over_channel = drain(ws, "cartridge_result")

    def stable(status):
        # `since` and the source clock advance with wall time and would
        # make any two reads differ for reasons that are not the contract.
        status = json.loads(json.dumps(status))
        status["lifecycle"].pop("since")
        status["source"].pop("last_frame_at")
        # A count that legitimately differs between a read taken before
        # the socket opened and one taken after it. Comparing it would be
        # asserting that the Tower lies about how many clients it has.
        status["source"].pop("clients_connected")
        status["run"].pop("started_at")
        status["run"].pop("elapsed_s")
        # And everything DERIVED from that same clock, which popping
        # `elapsed_s` alone does not cover. This was a real 2-in-10 flake
        # on Windows, pre-existing on `cv-lab/productization-v1` and found
        # during the 2026-08-27 unification.
        #
        # `processed_fps` and `offered_fps` are deliberately `null` while
        # `elapsed == 0` and a number thereafter -- "a rate over a
        # zero-length interval is undefined, not zero". `time.time()` on
        # Windows has ~15.6 ms granularity, so the three surfaces here are
        # read microseconds apart but occasionally straddle a tick: HTTP
        # sees `null`, the socket a moment later sees `0.0`. That is the
        # contract working, not breaking, so the test must exclude it for
        # the same reason it excludes `since` and `started_at`.
        #
        # `capacity_fps` is NOT popped: it comes from measured per-frame
        # cost, not from wall-clock, so it must agree across surfaces.
        status["run"]["throughput"].pop("processed_fps")
        status["run"]["throughput"].pop("offered_fps")
        return status

    assert stable(over_http["status"]) == stable(over_socket["status"])
    assert stable(over_http["status"]) == stable(over_channel["payload"])
    assert over_http["contract"] == STATUS_CONTRACT
    assert over_http["control_contract"] == CONTROL_CONTRACT


def test_the_http_surface_answers_503_when_there_is_no_lab(monkeypatch):
    """404 would say "this Tower has never heard of a CV Lab", which is a
    different claim and the one iOS renders as "not built yet"."""
    client = make_client(monkeypatch)
    client.app.state.cv_lab = None
    response = client.get("/cv-lab")
    assert response.status_code == 503
    assert response.json()["reason"] == "lab_unavailable"


def test_a_subscription_begins_with_a_complete_snapshot(monkeypatch):
    client = make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "experimental_cv",
                "result_type": "status",
                "contract": STATUS_CONTRACT,
            }
        )
        reply = drain(ws, "result_subscribed")
        envelope = drain(ws, "cartridge_result")

    assert reply["contract"] == STATUS_CONTRACT
    assert envelope["seq"] == 1
    assert envelope["snapshot"] is True
    assert envelope["envelope_contract"] == ENVELOPE_CONTRACT
    assert envelope["time_basis"] == TIME_BASIS
    assert envelope["payload"]["lifecycle"]["state"] == "running"


def test_a_contract_mismatch_is_refused_not_served(monkeypatch):
    client = make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "experimental_cv",
                "result_type": "status",
                "contract": "experimental_cv.status/1999-01-01",
            }
        )
        error = drain(ws, "result_error")

    assert error["reason"] == "contract_mismatch"
    assert error["offered_contract"] == STATUS_CONTRACT


def test_the_channel_publishes_a_switch_without_being_asked(monkeypatch):
    """A phone that had to poll to notice the experiment changed would show
    the previous one's numbers under the new one's name for as long as it
    took."""
    client = make_client(monkeypatch, "baseline")
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "experimental_cv",
                "result_type": "status",
            }
        )
        drain(ws, "result_subscribed")
        first = drain(ws, "cartridge_result")

        command(ws, {"type": "cv_lab_start", "experiment_id": "edge_detection"})
        pump(client, times=3)
        # Snapshots are ordered and complete, so a subscriber may still
        # receive one computed before the switch -- it is older, not
        # wrong. What matters is that the switch arrives unasked, and
        # within a bounded number of messages.
        envelopes = [drain(ws, "cartridge_result") for _ in range(3)]

    second = next(e for e in envelopes if e["payload"]["selected"] == "edge_detection")
    assert first["payload"]["selected"] == "baseline"
    assert second["revision"] != first["revision"]
    assert second["revision_changed"] is True
    assert [e["seq"] for e in envelopes] == sorted(e["seq"] for e in envelopes)
    assert second["payload"]["lifecycle"]["run_id"] != first["payload"][
        "lifecycle"
    ]["run_id"]


# -- the control vocabulary ---------------------------------------------


def test_every_accepted_command_replies_with_the_whole_status(monkeypatch):
    client = make_client(monkeypatch, "baseline")
    with client.websocket_connect("/ws") as ws:
        for message, expected_state in (
            ({"type": "cv_lab_pause"}, "paused"),
            ({"type": "cv_lab_resume"}, "running"),
            ({"type": "cv_lab_stop"}, "stopped"),
            (
                {"type": "cv_lab_start", "experiment_id": "edge_detection"},
                "starting",
            ),
        ):
            reply = command(ws, message)
            assert reply["accepted_command"] == message["type"]
            assert reply["contract"] == STATUS_CONTRACT
            assert reply["control_contract"] == CONTROL_CONTRACT
            assert reply["status"]["lifecycle"]["state"] == expected_state
            assert reply["status"]["available"]


def test_a_request_id_is_echoed_and_a_missing_one_is_not_invented(monkeypatch):
    client = make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        with_id = command(ws, {"type": "cv_lab_status", "request_id": "req-9"})
        without = command(ws, {"type": "cv_lab_status"})

    assert with_id["request_id"] == "req-9"
    assert "request_id" not in without


@pytest.mark.parametrize("hostile", [{"x": 1}, 12, ["a"], "x" * 200, "", None])
def test_a_hostile_request_id_is_dropped_rather_than_echoed(monkeypatch, hostile):
    """It goes back onto the wire, so a remote party must not be able to
    put an arbitrary object there."""
    client = make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        reply = command(ws, {"type": "cv_lab_status", "request_id": hostile})
    assert "request_id" not in reply


def test_a_pushed_status_is_distinguishable_from_a_reply(monkeypatch):
    client = make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        asked = command(ws, {"type": "cv_lab_status"})
    assert "accepted_command" not in asked


def test_every_refusal_reason_is_reachable_over_the_wire(monkeypatch):
    """A closed set a client switches on. An unreachable reason is a
    branch nobody can test and a branch nobody can test is a guess."""
    client = make_client(monkeypatch, "baseline")
    seen = set()
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cv_lab_start", "experiment_id": "nope"})
        seen.add(drain(ws, "cv_lab_error")["reason"])

        ws.send_json({"type": "cv_lab_start", "experiment_id": 7})
        seen.add(drain(ws, "cv_lab_error")["reason"])

        ws.send_json({"type": "cv_lab_resume"})
        seen.add(drain(ws, "cv_lab_error")["reason"])

        ws.send_json({"type": "cv_lab_stop", "run_id": "not-a-run"})
        seen.add(drain(ws, "cv_lab_error")["reason"])

        ws.send_json({"type": "cv_lab_stop", "run_id": 7})
        seen.add(drain(ws, "cv_lab_error")["reason"])

    assert seen == {
        "unknown_experiment",
        "malformed_request",
        "invalid_state",
        "stale_run",
    }


def test_a_refusal_carries_the_command_the_contract_and_the_status(monkeypatch):
    client = make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cv_lab_start", "experiment_id": "nope"})
        error = drain(ws, "cv_lab_error")

    assert error["command"] == "cv_lab_start"
    assert error["control_contract"] == CONTROL_CONTRACT
    assert error["status"]["lifecycle"]["state"] == "running"
    assert "available" in error
    assert error["message"]


def test_a_tower_without_a_lab_refuses_commands_rather_than_ignoring_them(
    monkeypatch,
):
    client = make_client(monkeypatch)
    client.app.state.cv_lab = None
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cv_lab_start", "experiment_id": "baseline"})
        error = drain(ws, "cv_lab_error")
    assert error["reason"] == "lab_unavailable"


def test_a_control_message_never_ends_the_connection(monkeypatch):
    """The receive loop is what answers frames. It must not learn that a
    control message had a problem."""
    client = make_client(monkeypatch)
    with client.websocket_connect("/ws") as ws:
        for hostile in (
            {"type": "cv_lab_start"},
            {"type": "cv_lab_start", "experiment_id": ["a"]},
            {"type": "cv_lab_pause", "run_id": {"a": 1}},
        ):
            ws.send_json(hostile)
            drain(ws, "cv_lab_error")
        assert frame(ws, 1)["type"] == "frame_result"


# -- frame_result provenance --------------------------------------------


def test_every_frame_result_names_the_run_that_produced_it(monkeypatch):
    client = make_client(monkeypatch, "edge_detection")
    with client.websocket_connect("/ws") as ws:
        status = command(ws, {"type": "cv_lab_status"})["status"]
        first = frame(ws, 1)
        second = frame(ws, 30)

    provenance = first["cv_lab"]
    assert provenance["contract"] == FRAME_RESULT_CONTRACT
    assert provenance["run_id"] == status["lifecycle"]["run_id"]
    assert provenance["tower_instance_id"] == status["tower_instance_id"]
    assert provenance["experiment_id"] == "edge_detection"
    assert provenance["experiment_name"] == "Edge detection"
    assert provenance["provenance"] == "measured"
    assert provenance["backend"] == "opencv"
    assert provenance["result_label"] == "edge_density"
    assert provenance["time_basis"] == TIME_BASIS
    # Dense within the run, whatever the wire seq does. The current sender
    # forwards one frame in thirty, so `seq` skips by design.
    assert first["cv_lab"]["result_seq"] == 1
    assert second["cv_lab"]["result_seq"] == 2
    assert second["seq"] == 30


def test_the_existing_frame_result_fields_are_untouched(monkeypatch):
    """Additive means additive. Every client that predates this change
    must decode exactly what it decoded before."""
    client = make_client(monkeypatch, "baseline")
    with client.websocket_connect("/ws") as ws:
        result = frame(ws, 1)

    assert result["type"] == "frame_result"
    assert result["seq"] == 1
    assert result["result_label"] == "mean_intensity"
    assert result["result_value"] == result["mean_intensity"]
    assert result["stage_ms"] == {"total": result["processing_ms"]}


def test_a_result_carries_the_experiment_that_produced_it_across_a_switch(
    monkeypatch,
):
    """The staleness defect at the wire. A `frame_result` labelled with the
    previous experiment is the one failure that makes a bench useless and
    looks like nothing."""
    client = make_client(monkeypatch, "baseline")
    with client.websocket_connect("/ws") as ws:
        before = frame(ws, 1)
        command(ws, {"type": "cv_lab_start", "experiment_id": "edge_detection"})
        after = [frame(ws, seq) for seq in range(2, 8)]

    assert before["cv_lab"]["experiment_id"] == "baseline"
    for reply in after:
        if reply["type"] == "frame_error":
            # Refused while arming. Never a result attributed to the
            # experiment that is being replaced.
            assert reply["reason"] == "cv_lab_starting"
            continue
        assert reply["cv_lab"]["experiment_id"] == "edge_detection"
        assert reply["cv_lab"]["run_id"] != before["cv_lab"]["run_id"]
        assert reply["result_label"] == "edge_density"


def test_a_refused_frame_carries_a_reason_and_no_provenance(monkeypatch):
    client = make_client(monkeypatch, "baseline")
    with client.websocket_connect("/ws") as ws:
        command(ws, {"type": "cv_lab_pause"})
        refused = frame(ws, 5)

    assert refused["type"] == "frame_error"
    assert refused["seq"] == 5
    assert refused["reason"] == "cv_lab_paused"
    assert "cv_lab_resume" in refused["message"]
    assert "cv_lab" not in refused


def test_an_undecodable_frame_keeps_the_generic_reason(monkeypatch):
    """A refusal and a bad frame are different facts. Conflating them would
    make "the Lab is paused" indistinguishable from "your JPEG is
    truncated"."""
    client = make_client(monkeypatch, "baseline")
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "frame",
                "seq": 1,
                "width": 16,
                "height": 16,
                "format": "jpeg",
                "data": "////",
            }
        )
        reply = ws.receive_json()
    assert reply["type"] == "frame_error"
    assert reply["reason"] in ("invalid_frame", "frame_skipped")


def test_a_tower_without_a_lab_still_answers_frames(monkeypatch):
    """The provenance block is a diagnostic. It must never be able to cost
    a client the result it is attached to."""
    client = make_client(monkeypatch, "baseline")

    class _Broken:
        def frame_provenance(self):
            raise RuntimeError("no")

    real, client.app.state.cv_lab = client.app.state.cv_lab, _Broken()
    try:
        with client.websocket_connect("/ws") as ws:
            # The module still holds the real Lab, so the frame is still
            # processed; only the attribution read fails.
            reply = frame(ws, 1)
    finally:
        client.app.state.cv_lab = real

    assert reply["type"] == "frame_result"
    assert "cv_lab" not in reply


# -- the document and the code must not drift --------------------------


def test_the_contract_document_matches_the_code():
    """A fresh iOS client is told to implement from this document without
    reading Tower Python. Every identifier it quotes is load-bearing."""
    from tower.cv_lab import contracts
    from tower.routes import cv_lab_ws

    document = DOCUMENT.read_text(encoding="utf-8")

    for value in (
        contracts.STATUS_CONTRACT,
        contracts.CONTROL_CONTRACT,
        contracts.FRAME_RESULT_CONTRACT,
        contracts.CARTRIDGE,
        contracts.RESULT_TYPE_STATUS,
        contracts.TIME_BASIS,
        ENVELOPE_CONTRACT,
    ):
        assert value in document, f"the document never mentions {value!r}"

    for state in LIFECYCLE_STATES:
        assert f"`{state}`" in document, f"undocumented lifecycle state {state!r}"

    for origin in (contracts.ORIGIN_CLIENT_REQUEST, contracts.ORIGIN_STARTUP_DEFAULT):
        assert origin in document, f"undocumented run origin {origin!r}"

    for reason in contracts.REFUSAL_REASONS:
        assert reason in document, f"undocumented refusal reason {reason!r}"

    for reason in contracts.FRAME_REFUSAL_REASONS.values():
        assert reason in document, f"undocumented frame refusal {reason!r}"

    for message_type in sorted(cv_lab_ws.CV_LAB_MESSAGE_TYPES) + [cv_lab_ws.MSG_ERROR]:
        assert message_type in document, f"undocumented message {message_type!r}"

    for label, value in (
        ("metrics per run", contracts.MAX_REPORTED_METRICS),
        ("unclassified names", contracts.MAX_UNCLASSIFIED_REPORTED),
        ("stream-idle threshold", contracts.STREAM_IDLE_AFTER_S),
    ):
        rendered = str(int(value) if float(value).is_integer() else value)
        assert rendered in document, f"the document does not state the {label}"


# Keys whose NAMES are chosen by an experiment rather than by this
# contract: stage timings and the runtime facts an experiment reports
# about itself. The document says both are open maps and that a client
# must not switch on them, so documenting each one would be documenting
# the opposite of what it says.
_OPEN_MAPS = ("stage_ms", "runtime")


def test_every_payload_key_is_documented(monkeypatch):
    """A key on the wire the document never names is a key a consumer has
    to guess at."""
    client = make_client(monkeypatch, "frame_quality")
    with client.websocket_connect("/ws") as ws:
        frame(ws, 1, textured=True)
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "experimental_cv",
                "result_type": "status",
            }
        )
        drain(ws, "result_subscribed")
        envelope = drain(ws, "cartridge_result")

    document = DOCUMENT.read_text(encoding="utf-8")
    missing = []

    def _walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key not in document:
                    missing.append(f"{path}.{key}")
                if key in _OPEN_MAPS:
                    continue
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for item in node:
                _walk(item, path + "[]")

    _walk(envelope["payload"])
    assert missing == [], f"undocumented payload keys: {missing}"


def test_every_frame_provenance_key_is_documented(monkeypatch):
    client = make_client(monkeypatch, "baseline")
    with client.websocket_connect("/ws") as ws:
        result = frame(ws, 1)

    document = DOCUMENT.read_text(encoding="utf-8")
    missing = [key for key in result["cv_lab"] if key not in document]
    assert missing == [], f"undocumented cv_lab keys: {missing}"


def test_the_document_names_every_registered_experiment():
    """A catalog a phone will display, described in a document an engineer
    reads. One of them growing without the other is how a list becomes a
    surprise."""
    from tower.experiments import EXPERIMENTS

    document = DOCUMENT.read_text(encoding="utf-8")
    missing = [name for name in EXPERIMENTS if f"`{name}`" not in document]
    assert missing == [], f"the document never names {missing}"
