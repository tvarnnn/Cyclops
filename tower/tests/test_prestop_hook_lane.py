"""The replay snapshot barrier must not change an ordinary lane build."""

from pathlib import Path

import scripts.world_build_session as builder
from tests.test_world_builder_finish_stages_identity import _check, _run_builder, rendered


def test_unset_hook_preserves_builder_files_events_and_report(rendered, tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_PRESTOP_SNAPSHOT_DIR", raising=False)
    observed = _run_builder("soft", rendered, tmp_path / "unset", monkeypatch)
    _check("builder:soft", observed)
    assert observed["exit"] == 0
    assert not (tmp_path / "snapshot").exists()


def test_snapshot_follows_solver_wait_and_precedes_final_solve(rendered, tmp_path, monkeypatch):
    from scripts import prestop_snapshot

    barrier = []
    original_wait = builder.BackgroundSolver.wait
    original_final = builder.BackgroundSolver.run_final

    def wait(self, *args, **kwargs):
        result = original_wait(self, *args, **kwargs)
        barrier.append("wait-complete")
        return result

    def snapshot(root, world, session, destination):
        assert Path(destination) == tmp_path / "snapshot"
        assert (Path(root) / "worlds" / world / "sessions" / session / "keyframes.jsonl").is_file()
        assert barrier == ["wait-complete"]
        barrier.append("snapshot")

    def run_final(self, *args, **kwargs):
        assert barrier == ["wait-complete", "snapshot"]
        barrier.append("final")
        return original_final(self, *args, **kwargs)

    monkeypatch.setattr(builder.BackgroundSolver, "wait", wait)
    monkeypatch.setattr(builder.BackgroundSolver, "run_final", run_final)
    monkeypatch.setattr(prestop_snapshot, "snapshot_at_stop", snapshot)
    monkeypatch.setenv("TOWER_PRESTOP_SNAPSHOT_DIR", str(tmp_path / "snapshot"))
    observed = _run_builder("soft", rendered, tmp_path / "enabled", monkeypatch)
    assert observed["exit"] == 0
    assert barrier == ["wait-complete", "snapshot", "final"]
