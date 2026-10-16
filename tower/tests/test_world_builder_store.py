"""Persistence invariants for the World Builder store.

These are the tests that matter most in the whole module: everything else
can be rebuilt, but a world that is corrupted, silently mis-read, or
falsely reported as deleted is unrecoverable.
"""

import json
import os

import pytest

from tower.object_memory.records import Confidence
from tower.world_builder.records import (
    CameraIntrinsics,
    Keyframe,
    KeyframeEdge,
    ScaleState,
    Session,
    World,
    format_distance,
    make_keyframe_id,
    new_id,
)
from tower.world_builder.schema import (
    INTRINSICS_SOURCE_SELF_CALIBRATED,
    SCALE_MEASURED,
    SCALE_RELATIVE,
    SCALE_UNKNOWN,
)
from tower.world_builder.store import (
    UnknownPoseConventionError,
    UnsupportedSchemaError,
    WorldLockedError,
    WorldStore,
    compute_input_digest,
    write_json_atomic,
)


def _world(world_id: str = "w1") -> World:
    return World(world_id=world_id, created_at=1.0, updated_at=1.0)


def _session(world_id: str = "w1", session_id: str = "s1") -> Session:
    return Session(session_id=session_id, world_id=world_id, started_at=1.0)


def _keyframe(session_id: str = "s1", source_seq: int = 1) -> Keyframe:
    return Keyframe(
        keyframe_id=make_keyframe_id(session_id, source_seq),
        session_id=session_id,
        source_seq=source_seq,
        received_at=10.0,
        image_relpath=f"images/{source_seq:08d}.jpg",
        width=360,
        height=640,
        byte_count=16000,
    )


def test_world_round_trips_through_disk(tmp_path):
    store = WorldStore(tmp_path)
    world = _world()

    store.write_world(world)

    assert store.read_world("w1") == world


def test_unknown_schema_version_is_refused_not_guessed(tmp_path):
    store = WorldStore(tmp_path)
    store.write_world(_world())
    path = store.world_path("w1")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaError):
        store.read_world("w1")


def test_unknown_pose_convention_is_refused_not_guessed(tmp_path):
    """The failure this prevents is silent, not loud.

    Coordinates read under the wrong handedness or quaternion order come
    back plausible and mirrored, and everything that later references them
    inherits the error.
    """
    store = WorldStore(tmp_path)
    store.write_world(_world())
    path = store.world_path("w1")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pose_convention"]["handedness"] = "left"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnknownPoseConventionError):
        store.read_world("w1")


def test_atomic_write_leaves_no_temp_file_on_success(tmp_path):
    target = tmp_path / "a.json"

    write_json_atomic(target, {"x": 1})

    assert target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_leaves_no_temp_file_when_serialisation_fails(tmp_path):
    """A crashed write must not strand a temp file holding live data.

    This exact defect shipped once in object memory: a .tmp survived
    purge() while purge() still reported success, making the data both
    invisible to queries and unreachable by deletion.
    """
    target = tmp_path / "a.json"

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})

    assert list(tmp_path.glob("*.tmp")) == []


def test_journal_survives_a_torn_final_line(tmp_path):
    """An interrupted append leaves a partial line; it must be skipped."""
    store = WorldStore(tmp_path)
    store.append_keyframe("w1", _keyframe(source_seq=1))
    store.append_keyframe("w1", _keyframe(source_seq=2))
    path = store.keyframes_path("w1", "s1")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": 1, "keyframe_id": "trunc')

    keyframes = store.read_keyframes("w1", "s1")

    assert [kf.source_seq for kf in keyframes] == [1, 2]


def test_schema_mismatch_is_skipped_without_touching_the_file(tmp_path):
    """A record we cannot interpret is not corruption and is not deleted."""
    store = WorldStore(tmp_path)
    store.append_keyframe("w1", _keyframe(source_seq=1))
    path = store.keyframes_path("w1", "s1")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema_version": 42, "unknown": True}) + "\n")
    before = path.read_text(encoding="utf-8")

    keyframes = store.read_keyframes("w1", "s1")

    assert [kf.source_seq for kf in keyframes] == [1]
    assert path.read_text(encoding="utf-8") == before


def test_keyframe_image_is_written_durably(tmp_path):
    store = WorldStore(tmp_path)

    path = store.write_keyframe_image("w1", "s1", "00000001.jpg", b"\xff\xd8jpeg")

    assert path.read_bytes() == b"\xff\xd8jpeg"
    assert list(store.images_dir("w1", "s1").glob("*.tmp")) == []


def test_purge_removes_images_journals_and_temp_files(tmp_path):
    store = WorldStore(tmp_path)
    store.write_world(_world())
    store.write_session(_session())
    store.append_keyframe("w1", _keyframe())
    store.write_keyframe_image("w1", "s1", "00000001.jpg", b"jpeg")
    store.write_derived_manifest("w1", {"schema_version": 1, "input_digest": "x"})
    stray = store.session_dir("w1", "s1") / "keyframes.jsonl.tmp"
    stray.write_text("leftover", encoding="utf-8")

    report = store.purge_world("w1")

    assert report.complete
    assert not store.world_dir("w1").exists()
    assert not stray.exists()


def test_purge_of_a_missing_world_is_a_noop(tmp_path):
    report = WorldStore(tmp_path).purge_world("nope")

    assert report.removed == ()
    assert report.complete


def test_purge_report_does_not_claim_success_when_a_file_survives(
    tmp_path, monkeypatch
):
    """A purge that could not delete everything must say so.

    Reporting success while bytes remain on disk is worse than reporting
    failure: 06-PRIVACY-DATA requires real deletion, and the caller has no
    other way to learn the imagery is still there.
    """
    store = WorldStore(tmp_path)
    store.write_world(_world())
    store.write_keyframe_image("w1", "s1", "00000001.jpg", b"jpeg")

    real_unlink = os.unlink

    def stubborn_unlink(path, *args, **kwargs):
        if str(path).endswith("00000001.jpg"):
            raise PermissionError(32, "in use by another process")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", stubborn_unlink)

    report = store.purge_world("w1")

    assert not report.complete
    assert any("00000001.jpg" in path for path in report.retained)


def test_writer_lock_refuses_a_second_live_writer(tmp_path):
    store = WorldStore(tmp_path)
    write_json_atomic(store.lock_path("w1"), {"pid": os.getppid()})

    with pytest.raises(WorldLockedError):
        store.acquire_writer_lock("w1")


def test_writer_lock_is_reclaimed_from_a_dead_pid(tmp_path):
    """Liveness by pid, not by timeout.

    A staleness timer has to guess how long a legitimate writer may pause.
    Asking whether the pid still exists does not guess.
    """
    store = WorldStore(tmp_path)
    dead_pid = 999_999_999
    write_json_atomic(store.lock_path("w1"), {"pid": dead_pid})

    store.acquire_writer_lock("w1")

    assert json.loads(store.lock_path("w1").read_text())["pid"] == os.getpid()


def test_derived_is_not_current_when_inputs_changed(tmp_path):
    store = WorldStore(tmp_path)
    keyframes = [_keyframe(source_seq=1)]
    store.write_derived_manifest(
        "w1", {"schema_version": 1, "input_digest": compute_input_digest(keyframes)}
    )

    assert store.derived_currency("w1", compute_input_digest(keyframes)) is True
    grown = keyframes + [_keyframe(source_seq=2)]
    assert store.derived_currency("w1", compute_input_digest(grown)) is not True


def test_clearing_derived_leaves_the_world_readable(tmp_path):
    """Every byte under derived/ must be genuinely rebuildable.

    If this invariant ever fails, something authoritative has leaked into
    the derived tier and a mapper change becomes a data migration.
    """
    store = WorldStore(tmp_path)
    store.write_world(_world())
    store.write_session(_session())
    store.append_keyframe("w1", _keyframe())
    store.write_derived_manifest("w1", {"schema_version": 1, "input_digest": "x"})

    store.clear_derived("w1")

    assert store.read_world("w1").world_id == "w1"
    assert len(store.read_keyframes("w1", "s1")) == 1
    assert store.read_derived_manifest("w1") is None


def test_edges_round_trip(tmp_path):
    store = WorldStore(tmp_path)
    edge = KeyframeEdge(
        from_keyframe_id="s1:00000001",
        to_keyframe_id="s1:00000002",
        matches=400,
        inliers=350,
        inlier_ratio=0.875,
        cheirality_fraction=0.997,
        quality=Confidence.HIGH,
    )

    store.append_edge("w1", "s1", edge)

    assert store.read_edges("w1", "s1") == [edge]


def test_session_round_trips_with_unknown_intrinsics(tmp_path):
    store = WorldStore(tmp_path)
    session = _session()

    store.write_session(session)

    reloaded = store.read_session("w1", "s1")
    assert reloaded == session
    assert not reloaded.intrinsics.is_known


def test_self_calibrated_intrinsics_without_a_focal_length_are_refused():
    """A calibration that produced no focal length is not a calibration."""
    payload = CameraIntrinsics.unknown().to_json_dict()
    payload["source"] = INTRINSICS_SOURCE_SELF_CALIBRATED

    from tower.world_builder.records import camera_intrinsics_from_json_dict

    with pytest.raises(ValueError):
        camera_intrinsics_from_json_dict(payload)


def test_new_ids_are_opaque_and_unique():
    assert new_id() != new_id()
    assert len(new_id()) == 32


class TestScaleHonesty:
    """No code path may print metres from an unscaled reconstruction."""

    def test_unknown_scale_reports_world_units(self):
        assert format_distance(12.4, ScaleState()) == "12.400 world units"

    def test_relative_scale_still_reports_world_units(self):
        """"Relative" means internally consistent, NOT metric.

        Monocular scale is fixed arbitrarily by the first solved baseline.
        That number is not a metre and must never be rendered as one.
        """
        scale = ScaleState(state=SCALE_RELATIVE, meters_per_unit=None)

        assert format_distance(12.4, scale) == "12.400 world units"

    def test_measured_scale_is_the_only_state_that_licenses_metres(self):
        scale = ScaleState(
            state=SCALE_MEASURED, meters_per_unit=0.5, confidence=Confidence.HIGH
        )

        assert format_distance(12.4, scale) == "6.200 m"

    def test_measured_state_without_a_factor_still_refuses_metres(self):
        scale = ScaleState(state=SCALE_MEASURED, meters_per_unit=None)

        assert format_distance(12.4, scale) == "12.400 world units"

    def test_default_world_scale_is_unknown(self):
        assert _world().scale.state == SCALE_UNKNOWN


class TestIntrinsicsRescaling:
    """DAT changes resolution mid-stream and we cannot override it."""

    def _known(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            source=INTRINSICS_SOURCE_SELF_CALIBRATED,
            model="pinhole_radtan",
            fx=300.0,
            fy=300.0,
            cx=180.0,
            cy=320.0,
            calibrated_width=360,
            calibrated_height=640,
        )

    def test_rescaling_is_refused_when_linearity_is_unproven(self):
        """Whether DAT resizes or crops between modes is not established.

        Linear scaling holds for a resize and fails for a crop, so the
        default must be refusal rather than a factor that is silently
        wrong by the crop ratio.
        """
        with pytest.raises(ValueError, match="scales_linearly"):
            self._known().scaled_to(504, 896)

    def test_rescaling_is_allowed_once_linearity_is_established(self):
        intrinsics = self._known()
        proven = CameraIntrinsics(
            **{
                **intrinsics.__dict__,
                "scales_linearly_across_resolutions": True,
            }
        )

        rescaled = proven.scaled_to(720, 1280)

        assert rescaled.fx == pytest.approx(600.0)
        assert rescaled.cy == pytest.approx(640.0)

    def test_rescaling_to_the_same_resolution_is_a_noop(self):
        intrinsics = self._known()

        assert intrinsics.scaled_to(360, 640) == intrinsics
