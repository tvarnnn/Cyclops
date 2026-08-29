"""What the Lab costs, and the fact that it does not grow.

`handoff.md` 9.3 says a `stream_stop` MAY NEVER ARRIVE, so "for the length
of a run" means "for as long as the Tower is up". Anything that grew per
frame would grow without bound, and this file is the reason to believe
nothing does.

It also pins the arithmetic of the aggregate. A per-frame fraction summed
over 9,199 frames is 768 -- a number that gets printed and believed --
and `MetricKind` exists so the experiment says how its numbers combine.
The tests below check that the Lab ASKS rather than guesses, including for
the metric it has never heard of.
"""

import asyncio
import json

import pytest

from tests.cv_lab_fixtures import (  # noqa: F401
    _close_cv_lab_clients,
    armed_lab,
    drain,
    frame,
    jpeg_bytes,
    make_client,
    make_lab,
)
from tower.cv_lab.contracts import (
    MAX_REPORTED_METRICS,
    MAX_UNCLASSIFIED_REPORTED,
    PREVIEW_PATH,
    TREATMENT_RAW_EPHEMERAL,
)
from tower.cv_lab.run import LabRun
from tower.experiments import ExperimentResult, MetricKind


def _run(experiment_id="frame_quality"):
    return LabRun(
        run_id="r1",
        experiment_id=experiment_id,
        descriptor={"id": experiment_id},
        origin="client_request",
        started_at=0.0,
    )


def _result(**metrics):
    return ExperimentResult(
        result_value=1.0,
        result_label="sharpness_laplacian_var",
        processing_ms=1.0,
        stage_ms={"decode": 0.5, "edges": 0.5},
        metrics=metrics,
    )


# -- constant memory ----------------------------------------------------


def test_a_run_holds_the_same_number_of_objects_after_one_frame_and_after_many():
    """No frame list, no metric history, no sample buffer."""
    run = _run()
    for _ in range(3):
        run.record_result(_result(edge_density=0.1, width=640.0), now=1.0)
    small = (len(run._metrics), len(run.stage_ms), len(run._unclassified))

    for index in range(2000):
        run.record_result(
            _result(edge_density=index / 2000, width=640.0), now=float(index)
        )

    assert (len(run._metrics), len(run.stage_ms), len(run._unclassified)) == small
    assert run.frames_processed == 2003


def test_the_status_document_does_not_grow_with_frames(monkeypatch):
    client = make_client(monkeypatch, "frame_quality")
    lab = client.app.state.cv_lab

    lab.process(jpeg_bytes(textured=True))
    small = len(json.dumps(lab.status()))
    for _ in range(200):
        lab.process(jpeg_bytes(textured=True))
    large = len(json.dumps(lab.status()))

    # Only the digits of the counters may differ.
    assert large - small < 64


def test_the_accumulator_is_bounded_by_the_declared_metric_set():
    """A name flood adds nothing, and the bound is not a number in the code.

    An earlier version of this file carried a `MAX_TRACKED_METRICS = 64`
    cap. It could never fire: a name enters the accumulator only if
    `classify_metric` recognises it, which means only if the experiment
    DECLARED it, and the largest declaration has twelve entries. The cap
    was removed rather than kept as reassurance, and this is the test that
    replaced it.
    """
    from tower.experiments import metric_kinds

    run = _run("frame_quality")
    for index in range(500):
        run.record_result(_result(**{f"m{index}": 1.0}), now=1.0)

    assert run.tracked_metric_count == 0
    assert len(run.unclassified_metrics) == MAX_UNCLASSIFIED_REPORTED

    declared = _run("frame_quality")
    for _ in range(50):
        declared.record_result(
            _result(**{name: 1.0 for name in metric_kinds("frame_quality")}), now=1.0
        )
    assert declared.tracked_metric_count == len(metric_kinds("frame_quality"))


def test_unclassified_metric_names_are_bounded():
    run = _run("frame_quality")
    for index in range(MAX_UNCLASSIFIED_REPORTED * 4):
        run.record_result(_result(**{f"unknown_{index}": 1.0}), now=1.0)

    assert len(run.unclassified_metrics) == MAX_UNCLASSIFIED_REPORTED


def test_the_reported_metric_list_is_capped_and_says_how_many_it_dropped():
    from tower.experiments import experiment_metadata

    run = _run("frame_quality")
    run.record_result(
        _result(**{name: 1.0 for name in ("edge_density", "entropy_bits", "width")}),
        now=1.0,
    )
    # Force more rows than the cap without inventing an experiment.
    for index in range(MAX_REPORTED_METRICS + 5):
        run._metrics[f"forced_{index}"] = run._metrics.setdefault(
            "edge_density", None
        ) or run._metrics["edge_density"]

    rows, omitted = run.metric_rows(experiment_metadata("frame_quality"))
    assert len(rows) == MAX_REPORTED_METRICS
    assert omitted > 0


def test_the_status_payload_stays_under_its_stated_bound(monkeypatch):
    """A BOUND, deliberately looser than the measurement beside it.

    The worst case is `optical_flow` -- fourteen metrics plus the
    eight-experiment catalog -- and it measures 8 852 B today. An earlier
    version of this test asserted < 9 216 B, which is a 364-byte margin:
    the next legitimate field would have tripped it, and a test that fails
    on correct work teaches people to raise the number without reading it.

    16 KB is the bound. It catches the thing worth catching -- a payload
    that grew a category rather than a field -- while the arity itself is
    guarded by `test_the_payload_contains_no_unbounded_list`, and the real
    figure is recorded in the report rather than in an assertion.
    """
    worst = 0
    for experiment_id in (
        "baseline",
        "edge_detection",
        "frame_quality",
        "feature_detection",
        "redaction_impact",
        "optical_flow",
    ):
        lab = asyncio.run(armed_lab(experiment_id))
        for _ in range(3):
            lab.process(jpeg_bytes(640, 360, textured=True))
        worst = max(worst, len(json.dumps(lab.status())))
    assert worst < 16384, worst


def test_the_payload_contains_no_unbounded_list(monkeypatch):
    client = make_client(monkeypatch, "optical_flow")
    lab = client.app.state.cv_lab
    for _ in range(5):
        lab.process(jpeg_bytes(320, 240, textured=True))

    offenders = []

    def _walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            if len(node) > 16:
                offenders.append(f"{path} has {len(node)}")
            for item in node:
                _walk(item, path + "[]")

    _walk(lab.status())
    assert offenders == []


# -- the arithmetic -----------------------------------------------------


def test_a_rate_is_averaged_and_a_count_is_summed():
    run = _run("frame_quality")
    for value in (0.2, 0.4, 0.6):
        run.record_result(_result(edge_density=value, width=640.0), now=1.0)

    from tower.experiments import experiment_metadata

    rows, _ = run.metric_rows(experiment_metadata("frame_quality"))
    by_label = {row["label"]: row for row in rows}

    assert by_label["edge_density"]["aggregation"] == "rate"
    assert by_label["edge_density"]["value"] == pytest.approx(0.4)
    assert by_label["width"]["aggregation"] == "constant"
    assert by_label["width"]["value"] == 640.0


def test_a_constant_that_was_not_constant_reports_no_value_and_says_why():
    """A resolution change mid-run. Reporting whichever width arrived first
    would be a number describing frames that are no longer being sent."""
    run = _run("frame_quality")
    run.record_result(_result(width=640.0), now=1.0)
    run.record_result(_result(width=320.0), now=2.0)

    from tower.experiments import experiment_metadata

    rows, _ = run.metric_rows(experiment_metadata("frame_quality"))
    width = next(row for row in rows if row["label"] == "width")
    assert width["value"] is None
    assert width["varied"] is True


def test_an_unaggregated_metric_publishes_no_value():
    """`dominant_direction_deg` is circular: the mean of 179 and -179 is 0,
    the one direction neither frame was moving in."""
    run = _run("optical_flow")
    run.record_result(
        ExperimentResult(
            result_value=1.0,
            result_label="median_flow_px",
            processing_ms=1.0,
            stage_ms={},
            metrics={"dominant_direction_deg": 179.0},
        ),
        now=1.0,
    )
    run.record_result(
        ExperimentResult(
            result_value=1.0,
            result_label="median_flow_px",
            processing_ms=1.0,
            stage_ms={},
            metrics={"dominant_direction_deg": -179.0},
        ),
        now=2.0,
    )

    from tower.experiments import experiment_metadata

    rows, _ = run.metric_rows(experiment_metadata("optical_flow"))
    direction = next(row for row in rows if row["label"] == "dominant_direction_deg")
    assert direction["aggregation"] == MetricKind.UNAGGREGATED.value
    assert direction["value"] is None
    assert direction["frames"] == 2


def test_an_unclassified_metric_is_excluded_and_named_rather_than_guessed():
    run = _run("frame_quality")
    run.record_result(_result(surprise=5.0, edge_density=0.1), now=1.0)

    from tower.experiments import experiment_metadata

    rows, _ = run.metric_rows(experiment_metadata("frame_quality"))
    assert "surprise" not in {row["label"] for row in rows}
    assert run.unclassified_metrics == ["surprise"]


def test_no_registered_experiment_emits_an_unclassified_metric(monkeypatch):
    """The wire field exists for a bug that must never ship. This is the
    test that keeps it from shipping."""
    for experiment_id in (
        "baseline",
        "edge_detection",
        "frame_quality",
        "feature_detection",
        "redaction_impact",
        "optical_flow",
    ):
        lab = asyncio.run(armed_lab(experiment_id))
        lab.process(jpeg_bytes(320, 240, textured=True))
        assert lab.status()["run"]["unclassified_metrics"] == [], experiment_id


def test_the_headline_is_first_and_carries_the_headline_unit():
    lab = asyncio.run(armed_lab("frame_quality"))
    lab.process(jpeg_bytes(320, 240, textured=True))

    metrics = lab.status()["run"]["metrics"]
    assert metrics[0]["headline"] is True
    assert metrics[0]["label"] == "sharpness_laplacian_var"
    assert metrics[0]["unit"] == "level^2"
    assert sum(1 for row in metrics if row["headline"]) == 1


def test_a_constant_is_measured_even_when_the_experiment_infers():
    """A configured threshold is a fact about how this Tower is set up, not
    a model's opinion."""
    run = _run("object_detection")
    run.record_result(
        ExperimentResult(
            result_value=2.0,
            result_label="detections",
            processing_ms=1.0,
            stage_ms={},
            metrics={"score_threshold": 0.4, "mean_score": 0.7},
        ),
        now=1.0,
    )

    from tower.experiments import experiment_metadata

    rows, _ = run.metric_rows(experiment_metadata("object_detection"))
    by_label = {row["label"]: row for row in rows}
    assert by_label["score_threshold"]["provenance"] == "measured"
    assert by_label["mean_score"]["provenance"] == "inferred"


def test_every_metric_row_carries_provenance_and_no_invented_confidence():
    """iOS makes provenance a REQUIRED field so that whoever decodes the
    reply has to answer it. There is no default here to fall back on."""
    lab = asyncio.run(armed_lab("frame_quality"))
    lab.process(jpeg_bytes(320, 240, textured=True))

    for row in lab.status()["run"]["metrics"]:
        assert row["provenance"] in ("measured", "inferred")
        assert row["confidence"] is None
        assert row["baseline"] is None
        assert row["higher_is_better"] is None


def test_no_result_channel_payload_ever_carries_an_image(monkeypatch):
    """The document says where a picture IS, and never contains one.

    This test used to assert `artifact is None`, because the Tower served
    no imagery at all. It now serves a derived preview, and the invariant
    that actually mattered is unchanged and is the one asserted here:
    **no image bytes ride the result channel.** The descriptor names a
    path, a media type and a treatment; the bytes are a separate fetch a
    client makes only if it wants one. That is the split
    `IOS-to-Tower.md` 5's own `notFetched`/`fetching`/`available` state
    machine was written around, and it is what keeps a 40 KB image off
    the socket the frame path shares a lock with.
    """
    client = make_client(monkeypatch, "feature_detection")
    lab = client.app.state.cv_lab
    lab.process(jpeg_bytes(320, 240, textured=True))

    annotation = lab.status()["run"]["annotation"]
    artifact = annotation["artifact"]
    assert artifact is not None
    # Present means the reason is absent. Never both, never neither.
    assert annotation["artifact_unavailable_reason"] is None
    # The treatment is STATED, and it is the strict value.
    assert artifact["treatment"] == TREATMENT_RAW_EPHEMERAL
    assert artifact["persistence"] == "none"
    # A process claim, never an outcome claim.
    assert artifact["face_filter"] == "none"
    # A path to fetch, not an inline payload.
    assert artifact["path"] == PREVIEW_PATH
    assert "image_bytes" not in artifact
    assert "data" not in artifact

    # And nothing anywhere in the payload is image-shaped.
    encoded = json.dumps(lab.status())
    assert "data:image" not in encoded
    assert "/9j/" not in encoded  # a JPEG in base64 always starts here
    assert "iVBORw0KGgo" not in encoded  # nor a PNG


def test_an_experiment_with_no_picture_says_so_rather_than_going_quiet(monkeypatch):
    """`baseline` is the control and will never have a preview.

    Not because nobody got round to it: a line drawing costs about as
    much as this entire experiment, so giving the cheapest one a picture
    would destroy the figure every other experiment's cost is read
    against. The document says which of the two situations this is,
    because "there is nothing to draw" and "a picture was withheld" are
    different facts about a Tower.
    """
    client = make_client(monkeypatch, "baseline")
    lab = client.app.state.cv_lab
    lab.process(jpeg_bytes(320, 240))

    annotation = lab.status()["run"]["annotation"]
    assert annotation["artifact"] is None
    assert "no visual output" in annotation["artifact_unavailable_reason"]
    assert "withheld" in annotation["artifact_unavailable_reason"]


def test_an_annotation_count_of_zero_is_not_the_same_as_no_count():
    run = _run("object_detection")
    run.record_result(
        ExperimentResult(
            result_value=0.0,
            result_label="detections",
            processing_ms=1.0,
            stage_ms={},
            metrics={"detections": 0.0},
        ),
        now=1.0,
    )
    assert run.metric_total("detections") == 0.0

    empty = _run("baseline")
    assert empty.metric_total("detections") is None


def test_an_experiment_with_no_annotation_metric_reports_null_and_why():
    lab = asyncio.run(armed_lab("baseline"))
    lab.process(jpeg_bytes())
    annotation = lab.status()["run"]["annotation"]
    assert annotation["count"] is None
    assert "reports no annotation count" in annotation["count_unavailable_reason"]


# -- the frame path pays almost nothing ---------------------------------


def test_recording_a_result_never_raises_on_the_frame_path():
    """The frame path answering a client must not end because a
    measurement could not be filed."""

    class _Hostile:
        result_value = float("nan")
        result_label = "x"
        processing_ms = None  # not a number
        stage_ms = {"a": "not a number"}
        metrics = {"b": object()}

    run = _run("frame_quality")
    assert run.record_result(_Hostile(), now=1.0) == 1
    assert run.frames_processed == 1
