"""CLI contracts for the World Builder scripts.

These drive the scripts as a user would -- separate process, cold start --
which is also what makes them the real test of "a saved world can be
inspected without the session that produced it".
"""

import json
import subprocess
import sys

from tests.conftest import no_depth_model_env


def _run(script, *args):
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        capture_output=True,
        text=True,
        env=no_depth_model_env(),
    )


class TestWorldBuildSessionCli:
    def test_help_exits_zero(self):
        assert _run("world_build_session.py", "--help").returncode == 0

    def test_requires_a_frame_source(self):
        result = _run("world_build_session.py")

        assert result.returncode != 0
        assert "--frames" in result.stderr

    def test_refuses_both_frame_sources_at_once(self, tmp_path):
        result = _run(
            "world_build_session.py", "--synthetic", "--frames", str(tmp_path)
        )

        assert result.returncode != 0

    def test_a_synthetic_run_reports_an_honest_scale_state(self, tmp_path):
        result = _run(
            "world_build_session.py",
            "--synthetic",
            "--synthetic-frames",
            "10",
            "--root",
            str(tmp_path),
            "--format",
            "json",
        )

        assert result.returncode == 0
        report = json.loads(result.stdout)
        assert report["frames_observed"] == 10
        assert report["keyframes_accepted"] >= 2
        # Relative, never metric: the unit is whatever the first solved
        # baseline happened to be.
        assert report["scale_state"] == "relative"

    def test_an_empty_frame_directory_exits_nonzero(self, tmp_path):
        result = _run("world_build_session.py", "--frames", str(tmp_path))

        assert result.returncode != 0


class TestWorldInspectCli:
    def _built_world(self, tmp_path):
        result = _run(
            "world_build_session.py",
            "--synthetic",
            "--synthetic-frames",
            "10",
            "--root",
            str(tmp_path),
            "--format",
            "json",
        )
        assert result.returncode == 0
        return json.loads(result.stdout)["world_id"]

    def test_help_exits_zero(self):
        assert _run("world_inspect.py", "--help").returncode == 0

    def test_world_is_required_unless_listing(self, tmp_path):
        result = _run("world_inspect.py", "--root", str(tmp_path))

        assert result.returncode != 0
        assert "--world" in result.stderr

    def test_listing_an_empty_root_succeeds(self, tmp_path):
        result = _run("world_inspect.py", "--root", str(tmp_path), "--list")

        assert result.returncode == 0

    def test_an_unknown_world_exits_nonzero(self, tmp_path):
        result = _run(
            "world_inspect.py", "--root", str(tmp_path), "--world", "nope"
        )

        assert result.returncode != 0

    def test_a_saved_world_is_inspectable_from_a_cold_process(self, tmp_path):
        """The definition-of-done item everything else serves."""
        world_id = self._built_world(tmp_path)

        result = _run(
            "world_inspect.py",
            "--root",
            str(tmp_path),
            "--world",
            world_id,
            "--format",
            "json",
        )

        assert result.returncode == 0
        report = json.loads(result.stdout)
        assert report["world_id"] == world_id
        assert report["sessions"]
        assert report["sessions"][0]["keyframes"] >= 2

    def test_the_report_never_claims_metres_for_a_relative_world(self, tmp_path):
        world_id = self._built_world(tmp_path)

        result = _run(
            "world_inspect.py", "--root", str(tmp_path), "--world", world_id
        )

        assert result.returncode == 0
        assert "NOT metric" in result.stdout
        assert " m\n" not in result.stdout

    def test_unavailable_figures_print_as_unknown_not_zero(self, tmp_path):
        """"No poses solved" and "poses never attempted" are different facts."""
        world_id = self._built_world(tmp_path)

        result = _run(
            "world_inspect.py",
            "--root",
            str(tmp_path),
            "--world",
            world_id,
            "--format",
            "json",
        )

        report = json.loads(result.stdout)
        assert report["scale"]["meters_per_unit"] == "unknown"

    def test_verify_passes_on_an_intact_world(self, tmp_path):
        world_id = self._built_world(tmp_path)

        result = _run(
            "world_inspect.py",
            "--root",
            str(tmp_path),
            "--world",
            world_id,
            "--verify",
            "--format",
            "json",
        )

        assert result.returncode == 0
        assert json.loads(result.stdout)["verify"]["ok"] is True

    def test_verify_fails_when_a_referenced_image_is_missing(self, tmp_path):
        """A journal line pointing at a missing image is corruption."""
        world_id = self._built_world(tmp_path)
        images = list((tmp_path / "worlds" / world_id).rglob("*.jpg"))
        assert images
        images[0].unlink()

        result = _run(
            "world_inspect.py",
            "--root",
            str(tmp_path),
            "--world",
            world_id,
            "--verify",
            "--format",
            "json",
        )

        assert result.returncode == 1
        assert json.loads(result.stdout)["verify"]["missing_images"]

    def test_trajectory_output_carries_poses_and_images(self, tmp_path):
        """Everything a recorded-path viewer needs, already persisted.

        No Tower-side replay feature is required: the trajectory is the
        ordered keyframes plus their solved poses plus their image paths.
        """
        world_id = self._built_world(tmp_path)

        result = _run(
            "world_inspect.py",
            "--root",
            str(tmp_path),
            "--world",
            world_id,
            "--trajectory",
            "--format",
            "json",
        )

        assert result.returncode == 0
        trajectory = json.loads(result.stdout)["trajectory"]
        rows = next(iter(trajectory.values()))
        assert len(rows) >= 2
        assert rows[0]["pose"] is not None
        assert rows[0]["image_relpath"].endswith(".jpg")
        assert all("segment_index" in row for row in rows)
