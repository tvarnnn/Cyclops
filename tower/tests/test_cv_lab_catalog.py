"""The catalog must describe experiments that exist and say true things.

The failure this file exists to catch is not a crash. It is a Tower that
offers `depth` with a summary describing edge detection, or declares a
headline the experiment never emits, or ships an experiment with no
metadata at all -- none of which anything else would notice, because the
phone displays whatever it is told and matches on none of it.
"""

import asyncio

import pytest

from tests.cv_lab_fixtures import (  # noqa: F401
    _close_cv_lab_clients,
    armed_lab,
    jpeg_bytes,
)
from tower.cv_lab import catalog
from tower.cv_lab.contracts import CARTRIDGE
from tower.experiments import (
    EXPERIMENTS,
    PROVENANCE_INFERRED,
    PROVENANCE_MEASURED,
    experiment_metadata,
    metric_kinds,
)
from tower.results.contracts import (
    CARTRIDGE_EXPERIMENTAL_CV,
    EXPERIMENTAL_CV_STATUS_CONTRACT,
)
from tower.cv_lab.contracts import STATUS_CONTRACT

CHEAP = (
    "baseline",
    "edge_detection",
    "frame_quality",
    "feature_detection",
    "redaction_impact",
    "optical_flow",
)


# -- completeness -------------------------------------------------------


def test_every_registered_experiment_has_catalog_metadata():
    """Registering without describing must be impossible, not merely rare.

    `ExperimentRegistration` takes the metadata positionally for this
    reason -- an experiment added without it is a TypeError at import,
    not an entry a phone renders as a blank row.
    """
    for experiment_id in EXPERIMENTS:
        metadata = experiment_metadata(experiment_id)
        assert metadata.name.strip()
        assert metadata.summary.strip()
        assert metadata.headline_label.strip()
        assert metadata.backend in ("opencv", "torch")
        assert metadata.provenance in (PROVENANCE_MEASURED, PROVENANCE_INFERRED)


def test_the_catalog_covers_the_registry_exactly():
    assert {entry["id"] for entry in catalog.catalog()} == set(EXPERIMENTS)


def test_the_catalog_order_is_stable():
    """A row that moves under a finger, and a revision that changes with
    nothing behind it, are the same bug seen from two ends."""
    first = [entry["id"] for entry in catalog.catalog()]
    assert first == sorted(EXPERIMENTS)
    assert first == [entry["id"] for entry in catalog.catalog()]


def test_ios_reads_three_fields_and_all_three_are_present():
    """ios/docs/agent-handoffs/IOS-TO-TOWER.md §2.1: an opaque id, a name, and
    optionally a summary. Nothing else, and nothing is parsed."""
    for entry in catalog.catalog():
        assert isinstance(entry["id"], str) and entry["id"]
        assert isinstance(entry["name"], str) and entry["name"]
        assert entry["summary"] is None or isinstance(entry["summary"], str)


# -- the metadata must be TRUE ------------------------------------------


@pytest.mark.parametrize("experiment_id", CHEAP)
def test_the_declared_headline_is_the_one_the_experiment_emits(experiment_id):
    """A declared headline nobody checks is a guess with a schema.

    The catalog says what an experiment measures BEFORE it has run a
    frame, which is what makes an experiment picker useful. That claim is
    worth exactly as much as this test.
    """
    lab = asyncio.run(armed_lab(experiment_id))
    result = lab.process(jpeg_bytes(64, 64, textured=True))
    assert result.result_label == experiment_metadata(experiment_id).headline_label


@pytest.mark.parametrize("experiment_id", CHEAP)
def test_the_declared_units_name_metrics_that_exist(experiment_id):
    """A unit for a metric nobody emits is dead weight that reads as care."""
    declared = set(experiment_metadata(experiment_id).metric_units)
    emitted = set(metric_kinds(experiment_id))
    assert declared - emitted == set()


def test_only_the_model_backed_experiments_claim_to_need_a_model():
    requires = {i for i in EXPERIMENTS if experiment_metadata(i).requires_model}
    assert requires == {"depth", "object_detection"}
    for experiment_id in requires:
        assert experiment_metadata(experiment_id).backend == "torch"


def test_every_model_backed_experiment_declares_what_it_needs():
    """Otherwise `available: true` is a claim nothing checked.

    `_missing_extra` returns None for an experiment with no entry in
    `_REQUIRED_MODULES`, so an experiment declaring `requires_model=True`
    and forgetting the map would be advertised as runnable on a Tower
    that cannot run it -- and then fail asynchronously, which is the
    outcome "refused in advance" exists to avoid. Nothing else ties the
    two together, so this does.
    """
    from tower.cv_lab.lab import _REQUIRED_MODULES

    needs_model = {i for i in EXPERIMENTS if experiment_metadata(i).requires_model}
    assert needs_model == set(_REQUIRED_MODULES), (
        "every requires_model experiment needs a _REQUIRED_MODULES entry, "
        "and nothing else may have one"
    )
    for experiment_id, modules in _REQUIRED_MODULES.items():
        assert modules, experiment_id


def test_only_the_model_backed_experiments_report_inference():
    """Rule 16 / Core Principle 2: model output is not measured fact, and
    the two must be distinguishable on the wire rather than in a docstring."""
    inferred = {
        i for i in EXPERIMENTS if experiment_metadata(i).provenance == PROVENANCE_INFERRED
    }
    assert inferred == {"depth", "object_detection"}


def test_the_stateful_flag_matches_the_experiments_that_actually_hold_state():
    stateful = {i for i in EXPERIMENTS if experiment_metadata(i).stateful}
    assert stateful == {"optical_flow", "object_detection", "depth"}


def test_only_object_detection_claims_an_annotation_count():
    """A keypoint is not an annotation and a fraction is not one either.

    iOS renders `annotation.count` as "things found in this frame", where
    0 means "found nothing". Pointing it at `keypoint_count` would put a
    texture measurement in a field that reads as a list of objects.
    """
    annotated = {
        i: experiment_metadata(i).annotation_metric
        for i in EXPERIMENTS
        if experiment_metadata(i).annotation_metric
    }
    assert annotated == {"object_detection": "detections"}


def test_an_annotation_metric_is_a_count_not_a_rate():
    from tower.experiments import MetricKind, classify_metric

    for experiment_id, metric in (("object_detection", "detections"),):
        assert classify_metric(experiment_id, metric) is MetricKind.COUNT


def test_a_unitless_headline_is_a_decision_not_an_omission():
    """`depth` emits relative inverse depth on an arbitrary scale.

    ios/docs/agent-handoffs/IOS-TO-TOWER.md §0.5: iOS renders a figure BARE
    when the Tower names no unit, because a bare number is the honest rendering
    of an unlabelled quantity. A unit string here would be a claim about scale
    that MiDaS-small does not support.
    """
    assert experiment_metadata("depth").headline_unit is None
    for experiment_id in EXPERIMENTS:
        if experiment_id == "depth":
            continue
        assert experiment_metadata(experiment_id).headline_unit


def test_provenance_is_validated_at_construction():
    from tower.experiments import ExperimentMetadata

    with pytest.raises(ValueError, match="provenance must be"):
        ExperimentMetadata(
            name="x",
            summary="x",
            provenance="probably",
            stateful=False,
            requires_model=False,
            headline_label="x",
            backend="opencv",
        )


# -- the two vocabularies must agree ------------------------------------


def test_the_cartridge_name_is_the_one_the_result_channel_offers():
    """Restated in two places, pinned here.

    `tower/results/contracts.py` is the result channel cartridge-blind
    core and must not import the Lab to learn its name. A duplicated
    string a test pins cannot drift; an import would make the shared wire
    surface depend on one cartridge package layout.
    """
    assert CARTRIDGE == CARTRIDGE_EXPERIMENTAL_CV


def test_the_status_contract_identifier_is_restated_identically():
    assert STATUS_CONTRACT == EXPERIMENTAL_CV_STATUS_CONTRACT
