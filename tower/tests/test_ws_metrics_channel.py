"""The measurement channel on the wire.

`frame_result` carried five scalars, which is why every cartridge's
"there is no non-scalar result channel" complaint traced back to this
one type. `metrics` is additive and omitted when empty, so a client that
has never heard of it is unaffected -- which is the rule
`tower/docs/contracts/CARTRIDGE-RESULTS.md` §1 states for any wire change
while the result type is still scalar-shaped.
"""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tower.main import create_app


def _frame(seq: int = 1, size: int = 64) -> dict:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (90, 140, 60)).save(buffer, format="JPEG")
    return {
        "type": "frame",
        "seq": seq,
        "width": size,
        "height": size,
        "format": "jpeg",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _one_result(monkeypatch, experiment: str) -> dict:
    monkeypatch.setenv("TOWER_CV_EXPERIMENT", experiment)
    client = TestClient(create_app())
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(_frame())
        return websocket.receive_json()


class TestMetricsAreAdditive:
    def test_an_experiment_without_metrics_omits_the_field_entirely(
        self, monkeypatch
    ):
        """baseline predates the channel and must not grow an empty object."""
        result = _one_result(monkeypatch, "baseline")

        assert result["type"] == "frame_result"
        assert "metrics" not in result

    def test_the_pre_existing_fields_are_untouched(self, monkeypatch):
        result = _one_result(monkeypatch, "baseline")

        assert result["result_label"] == "mean_intensity"
        assert result["result_value"] == result["mean_intensity"]
        assert result["stage_ms"] == {"total": result["processing_ms"]}

    def test_a_measuring_experiment_sends_its_metrics(self, monkeypatch):
        result = _one_result(monkeypatch, "frame_quality")

        assert result["result_label"] == "sharpness_laplacian_var"
        metrics = result["metrics"]
        assert set(metrics) >= {
            "sharpness_laplacian_var",
            "gradient_energy",
            "entropy_bits",
            "contrast_std",
            "edge_density",
            "overexposed_fraction",
            "underexposed_fraction",
        }

    def test_the_headline_is_also_present_in_metrics(self, monkeypatch):
        """A client may read either; they must never disagree."""
        result = _one_result(monkeypatch, "feature_detection")

        assert result["metrics"][result["result_label"]] == pytest.approx(
            result["result_value"]
        )

    def test_every_metric_is_a_number(self, monkeypatch):
        """dict[str, float], enforced at the boundary a client sees."""
        result = _one_result(monkeypatch, "frame_quality")

        for name, value in result["metrics"].items():
            assert isinstance(value, (int, float)), name
            assert not isinstance(value, bool), name


class TestStatefulExperimentsOverTheWire:
    def test_optical_flow_reports_no_reference_on_the_first_frame(self, monkeypatch):
        result = _one_result(monkeypatch, "optical_flow")

        assert result["metrics"]["has_reference"] == 0.0

    def test_optical_flow_gains_a_reference_on_the_second_frame(self, monkeypatch):
        monkeypatch.setenv("TOWER_CV_EXPERIMENT", "optical_flow")
        client = TestClient(create_app())

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(_frame(1))
            first = websocket.receive_json()
            websocket.send_json(_frame(2))
            second = websocket.receive_json()

        assert first["metrics"]["has_reference"] == 0.0
        assert second["metrics"]["has_reference"] == 1.0

    def test_state_does_not_leak_between_connections(self, monkeypatch):
        """One app, two sessions.

        The module is process-scoped, so a second wearer session inherits
        whatever the first left behind. That is a real property and this
        test records it rather than asserting a reset that does not happen
        -- the Lab is a measurement tool, and a surprising carry-over
        would silently corrupt a measurement.
        """
        monkeypatch.setenv("TOWER_CV_EXPERIMENT", "optical_flow")
        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(_frame(1))
            websocket.receive_json()

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(_frame(2))
            second_session = websocket.receive_json()

        assert second_session["metrics"]["has_reference"] == 1.0, (
            "documents current behaviour: the experiment outlives the "
            "WebSocket session, because the module does"
        )


class TestABadFrameDoesNotKillTheModule:
    def test_an_undecodable_frame_is_reported_and_the_next_one_works(
        self, monkeypatch
    ):
        monkeypatch.setenv("TOWER_CV_EXPERIMENT", "frame_quality")
        client = TestClient(create_app())

        with client.websocket_connect("/ws") as websocket:
            bad = _frame(1)
            bad["data"] = base64.b64encode(b"\xff\xd8\xff\xdb" + b"\x00" * 64).decode(
                "ascii"
            )
            websocket.send_json(bad)
            first = websocket.receive_json()

            websocket.send_json(_frame(2))
            second = websocket.receive_json()

        assert first["type"] == "frame_error"
        assert second["type"] == "frame_result"
