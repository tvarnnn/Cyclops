"""The calibration must reach the builder the Tower starts BY ITSELF.

On 2026-08-25 a 360x640 calibration was measured from real DAT frames,
filed at `<world_root>/intrinsics/360x640.json`, and confirmed by a
direct `IntrinsicsStore.lookup(360, 640)` -- which returned it. The Tower
was then started with that world root and autobuild on, a phone streamed
360x640, and the world came back:

    intrinsics_source  unknown
    backend            unposed  (downgraded_from classical)
    degeneracy         {'no_intrinsics': 69}

Nothing was wrong with the calibration, the path, the store or the
resolution. What was wrong was WHEN the builder asked.

`CaptureWorkerSupervisor` starts the worker from `_start_capture`, at the
instant the capture id is minted -- which is `stream_start`, before the
phone has sent a single frame. The driver then resolved intrinsics
immediately, from a `frames.jsonl` that did not yet have a line in it,
got "resolution not observed", and froze `unknown()` into the session
record. The 360x640 frames arrived ~0.85s later and nothing looked again.

That gap is not a flaky race. Importing the engine, OpenCV included,
takes ~135ms on the Tower host; the first frame landed about a second
after `stream_start`. The worker wins every time.

Every test here therefore attaches the builder to an EMPTY capture and
only then writes frames. A test that writes frames first passes against
the bug -- which is exactly what the existing end-to-end test did.

SYNTHETIC, NOT PHYSICAL: the walk is rendered. The CALIBRATION is the
real measured one, because which record reached the builder is the whole
question here.
"""

import subprocess
import sys
import time

import numpy as np
import pytest

from tower.capture import CaptureRecorder
from tower.world_builder.intrinsics_store import IntrinsicsStore
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.store import WorldStore
from tests import synthetic_scene as ss
from tests.conftest import no_depth_model_env

pytestmark = pytest.mark.slow

# The resolution DAT actually delivered, in the order the frames measure
# at: 360 wide, 640 tall. Portrait -- and the reason the store is keyed
# `<width>x<height>` rather than by a habit of writing landscape.
WIDTH, HEIGHT = 360, 640

# The calibration measured from that physical walk, to the digit. The
# real numbers rather than round ones, so this test is honest about
# which record it is asserting reached the builder.
PHYSICAL_CALIBRATION = CameraIntrinsics(
    source="self_calibrated",
    model="pinhole",
    fx=438.23,
    fy=437.78,
    cx=174.88,
    cy=323.38,
    calibrated_width=WIDTH,
    calibrated_height=HEIGHT,
    reprojection_rms_px=0.2893,
    view_count=511,
)


def _camera_matrix() -> np.ndarray:
    """The calibration as a matrix, so the rendered walk MATCHES it.

    The frames a test renders and the calibration it files have to
    describe one camera. Rendering at a different focal length would
    still exercise the plumbing, but it would bake a lie into the fixture
    and any geometry assertion added here later would fail for a reason
    that has nothing to do with calibration.
    """
    return np.array(
        [
            [PHYSICAL_CALIBRATION.fx, 0.0, PHYSICAL_CALIBRATION.cx],
            [0.0, PHYSICAL_CALIBRATION.fy, PHYSICAL_CALIBRATION.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _walk(count: int) -> list[bytes]:
    """A short synthetic walk at exactly the delivered resolution."""
    scene = ss.furnished_room()
    poses = ss.strafe(count, step=0.09)
    images = ss.render_sequence(scene, poses, _camera_matrix(), WIDTH, HEIGHT)
    return [ss.encode_jpeg(image) for image in images]


def _wait_for(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _only_session(world_root):
    """The one session this driver wrote, read back off disk.

    Read from the STORE rather than from the driver's JSON report,
    because the session record is what `world_inspect` reads and what
    reported `intrinsics_source: unknown` after the physical walk.
    """
    store = WorldStore(world_root)
    world_ids = [path.name for path in (world_root / "worlds").iterdir()]
    assert len(world_ids) == 1, world_ids
    world = store.read_world(world_ids[0])
    assert len(world.session_ids) == 1, world.session_ids
    return store.read_session(world.world_id, world.session_ids[0])


class _Worker:
    """The auto-spawned builder, started the way the supervisor starts it.

    A real subprocess running the real argv `main.py` builds, against a
    capture that is open and empty -- not a function call, and not a
    capture that already has frames in it. Both of those shortcuts are
    what let this bug through.
    """

    def __init__(self, directory, world_root, *extra):
        self.output = ""
        self._process = subprocess.Popen(
            [
                sys.executable,
                "scripts/world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(world_root),
                *extra,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=no_depth_model_env(),
        )
        self._world_root = world_root

    def wait_until_attached(self) -> None:
        """Block until the worker is past the point it used to decide.

        The world directory appearing is the driver's first write, and it
        happens after argument parsing and after the store is
        constructed. Waiting on it means the frames below are written
        strictly AFTER the worker had its chance to look early -- which
        is the failure being reproduced, so the test has to allow it.
        """
        _wait_for(
            lambda: (self._world_root / "worlds").exists(),
            60.0,
            "the worker to create its world",
        )

    def finish(self) -> str:
        self.output = self._process.communicate(timeout=240)[0]
        return self.output

    def kill_if_running(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.communicate()

    @property
    def returncode(self):
        return self._process.returncode


def _run_walk(tmp_path, *, calibrated: bool, frames: int = 14, extra=()):
    """Attach a builder to an empty capture, THEN walk. Returns the session.

    The ordering in this function is the entire regression: recorder
    starts, worker attaches, frames are written. Reversing the last two
    lines makes every test in this file pass against the bug.
    """
    recorder = CaptureRecorder(tmp_path / "capture")
    capture_id = recorder.start()
    directory = recorder.capture_dir(capture_id)

    journal = directory / "frames.jsonl"
    assert not journal.exists() or not journal.read_text(
        encoding="utf-8"
    ).strip(), "the capture must be frameless when the builder attaches"

    world_root = tmp_path / "world"
    if calibrated:
        IntrinsicsStore(world_root).save(PHYSICAL_CALIBRATION)

    worker = _Worker(directory, world_root, *extra)
    try:
        worker.wait_until_attached()
        for index, payload in enumerate(_walk(frames)):
            recorder.write_frame(
                payload,
                source_seq=index,
                wire_seq=index,
                tx_seq=index,
                width=WIDTH,
                height=HEIGHT,
            )
        recorder.stop()
        output = worker.finish()
        assert worker.returncode == 0, output
    finally:
        worker.kill_if_running()

    return _only_session(world_root), output, world_root


class TestBuilderAttachedBeforeTheFirstFrame:
    """The physical ordering: worker first, frames second."""

    def test_the_worker_selects_the_calibration_filed_for_the_frames(self, tmp_path):
        """The bug, reproduced and then refused.

        Fails before the fix with `intrinsics.source == 'unknown'`: the
        driver asked the store about a resolution nobody had measured.
        """
        session, output, _ = _run_walk(tmp_path, calibrated=True)

        assert session.intrinsics.source == "self_calibrated", (
            "the live worker did not find the 360x640 calibration that was on "
            f"disk the whole time; the session says "
            f"{session.intrinsics.source!r}\nworker output:\n{output}"
        )
        assert session.intrinsics.fx == pytest.approx(438.23)
        assert session.intrinsics.fy == pytest.approx(437.78)
        assert session.intrinsics.calibrated_width == WIDTH
        assert session.intrinsics.calibrated_height == HEIGHT

    def test_the_session_does_not_downgrade_for_want_of_intrinsics(self, tmp_path):
        """`backend: unposed, downgraded_from: classical` must not recur."""
        session, output, _ = _run_walk(tmp_path, calibrated=True)

        assert session.backend_id == "classical-sfm", output
        assert session.backend_downgraded_from is None, (
            f"downgraded anyway: {session.backend_downgrade_reason}\n{output}"
        )
        assert session.backend_requires_intrinsics is True

    def test_keyframes_land_and_the_resolution_check_accepts_them(self, tmp_path):
        """Known intrinsics arm `_require_matching_resolution`.

        Unknown intrinsics skip that check entirely, so until this bug
        was fixed nothing ever exercised it on the live path. A build
        that raised there would kill the worker mid-walk.
        """
        session, output, world_root = _run_walk(tmp_path, calibrated=True)

        keyframes = WorldStore(world_root).read_keyframes(
            session.world_id, session.session_id
        )
        assert keyframes, output
        assert {(k.width, k.height) for k in keyframes} == {(WIDTH, HEIGHT)}

    def test_a_mid_walk_rebuild_uses_the_calibration_too(self, tmp_path):
        """The live path rebuilds every N keyframes; those must be posed.

        `--rebuild-every` is how the Tower spawns this worker, and the
        rebuild goes through `_open_live_solve`, which selects a backend
        from the session's intrinsics at session start. If the lookup is
        late, every mid-walk rebuild is unposed as well.
        """
        session, output, _ = _run_walk(
            tmp_path, calibrated=True, extra=("--rebuild-every", "2")
        )

        assert session.backend_id == "classical-sfm", output
        assert "rebuild 1:" in output, output

    def test_the_operator_can_read_why_from_the_log(self, tmp_path):
        """A future physical run must SAY that calibration is active.

        The 2026-08-25 walk produced a worker log from which none of this
        could be read; the only way to learn what had happened was to
        inspect the session record afterwards, hours later.
        """
        _, output, world_root = _run_walk(tmp_path, calibrated=True, frames=8)

        # The absolute world root, so a wrong CWD is visible as one.
        assert str(world_root.resolve()) in output, output
        # The resolution the frames MEASURED at, not one anybody declared.
        assert f"{WIDTH}x{HEIGHT}" in output, output
        # The file that answered, by path.
        assert "360x640.json" in output, output
        # What was selected, on both axes.
        assert "self_calibrated" in output, output
        assert "classical" in output, output


class TestWorldInspectSaysCalibrationIsActive:
    """`world_inspect` is the authority the physical run is judged by.

    It is what reported `intrinsics_source: unknown` and
    `degeneracy: {'no_intrinsics': 69}` after the failed walk, so it is
    what has to report the opposite after a good one -- in terms an
    operator can read without cross-checking a session record.
    """

    def test_the_report_names_the_calibration_and_no_intrinsics_degeneracy(
        self, tmp_path
    ):
        from tower.world_builder.inspect import WorldView
        from tower.world_builder.store import WorldStore

        session, output, world_root = _run_walk(tmp_path, calibrated=True)
        report = WorldView(WorldStore(world_root), session.world_id).to_report()
        row = report["sessions"][0]

        assert row["intrinsics_source"] == "self_calibrated", output
        assert row["backend_id"] == "classical-sfm", output
        # The degeneracy that dominated the failed walk must be absent.
        # Other degeneracies are fine and expected -- low parallax and
        # missing correspondence are properties of the walk, not of the
        # calibration.
        degeneracy = row["degeneracy_counts"]
        assert "no_intrinsics" not in (
            degeneracy if isinstance(degeneracy, dict) else {}
        ), degeneracy

    def test_a_session_that_was_not_downgraded_does_not_read_as_unknown(
        self, tmp_path
    ):
        """"Not downgraded" must not render as "downgraded from unknown".

        An operator verifying a fresh calibration reads this line to
        decide whether it took. `unknown` in that slot is the same word
        the failed walk printed for `intrinsics_source`, and it made a
        clean classical session look like a second downgrade.
        """
        from tower.world_builder.inspect import WorldView, render_text
        from tower.world_builder.store import WorldStore

        session, output, world_root = _run_walk(tmp_path, calibrated=True)
        report = WorldView(WorldStore(world_root), session.world_id).to_report()

        assert report["sessions"][0]["backend_downgraded_from"] == "none", output
        assert "downgraded_from      none" in render_text(report)


class TestAnUncalibratedTowerStillSaysSo:
    """Deferring the lookup must not turn a miss into a hang or a lie."""

    def test_no_calibration_still_downgrades_honestly(self, tmp_path):
        session, output, _ = _run_walk(tmp_path, calibrated=False, frames=8)

        assert session.intrinsics.source == "unknown"
        assert session.backend_id == "unposed"
        assert session.backend_downgraded_from == "classical"
        # And it now names the resolution it wanted, which the old code
        # could not do on this path: it never observed one.
        assert f"{WIDTH}x{HEIGHT}" in output, output

    def test_a_capture_that_never_delivers_a_frame_still_terminates(self, tmp_path):
        """No frame ever arrives: the worker must finish, not wait forever.

        Deferring the lookup until a frame exists means the driver now
        WAITS for one. A capture that opens and closes empty -- a phone
        that connects and immediately drops -- must still end the worker
        rather than leave a process holding a world's writer lock.
        """
        recorder = CaptureRecorder(tmp_path / "capture")
        capture_id = recorder.start()
        directory = recorder.capture_dir(capture_id)
        world_root = tmp_path / "world"

        worker = _Worker(directory, world_root, "--format", "json")
        try:
            worker.wait_until_attached()
            recorder.stop()
            output = worker.finish()
            assert worker.returncode == 0, output
        finally:
            worker.kill_if_running()

        session = _only_session(world_root)
        assert session.frames_observed == 0
        assert session.intrinsics.source == "unknown"
        assert session.ended_at is not None, "the session must be closed"


class TestTheReportDistinguishesNotDowngradedFromNeverBuilt:
    """Both are `backend_downgraded_from: None` on the record. Not in the report."""

    @staticmethod
    def _row(**overrides):
        from tower.world_builder.inspect import _downgrade_report

        class _View:
            backend_id = overrides.get("backend_id", "classical-sfm")
            downgraded_from = overrides.get("downgraded_from")

        return _downgrade_report(_View())

    def test_a_built_session_that_kept_its_backend_reads_none(self):
        assert self._row() == "none"

    def test_a_session_that_was_downgraded_names_what_it_lost(self):
        assert self._row(downgraded_from="classical") == "classical"

    def test_a_session_that_never_built_is_genuinely_unknown(self):
        """No backend_id either: nothing has selected anything yet."""
        assert self._row(backend_id=None) == "unknown"


class TestTheScaleNoteStatesTheRuleNotOneCause:
    """A world can have solved poses and still have no unit.

    `scale: unknown` is set both by "no solved pose" and by "more than
    one segment, each in its own arbitrary unit". The replayed physical
    capture is the second: 29 solved poses across 6 segments, told it had
    no solved pose.
    """

    def test_the_note_does_not_claim_there_are_no_solved_poses(self):
        from tower.world_builder.inspect import render_text

        text = render_text(
            {
                "world_id": "w",
                "display_name": "n",
                "schema_version": 1,
                "frame_revision": 1,
                "images_purged": False,
                "scale": {
                    "state": "unknown",
                    "meters_per_unit": "unknown",
                    "allows_metres": False,
                },
                "storage_bytes": {
                    "total": 0,
                    "images": 0,
                    "journals": 0,
                    "derived": 0,
                },
                "sessions": [],
            }
        )

        assert "no solved pose" not in text
        assert "SINGLE segment" in text
