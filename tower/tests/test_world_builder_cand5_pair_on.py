"""The cand5 DEPLOY PAIR, on together: MERGE_AFTER_STOP + SURFACE_FAR_DEPTH_CAP, with
POSE_SANITY off (CX-CAND5-TESTGAP gap 3).

The two switches live in different places -- the merge in the builder's post-loop path
(`engine.build`, the derived tree), the cap in the surface stage (`surfacify`), which the
builder reaches only through `final_surface_stages` after the final solve. Pinned here:

  * the builder with BOTH on runs exactly the merge-only trace: the merge lands before
    the live surface is closed and before the final solve; the post-Stop merge does not
    poke the live surface; the final surface stage runs after the final build and sees
    the cap on (and pose sanity off);
  * the surface stage with BOTH on publishes exactly what the cap alone publishes
    (record, voxel, faces, digest): the merge switch is invisible to it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts import world_build_session as wbs  # noqa: E402
from tests.test_world_builder_merge_after_stop_main import (  # noqa: E402
    BUILD,
    LOOP_END,
    POLL_AFTER,
    RUN_FINAL,
    _walk,
)
from tests.test_world_builder_surface_far_cap import (  # noqa: E402
    CAP_ENVS,
    _lean,
    _room_blocks,
    _world,
)
from tests.test_world_builder_surface_pipeline import (  # noqa: E402
    SESSION,
    WORLD,
    _manifest,
)
from tower import config as C  # noqa: E402
from tower.world_builder import surface_pipeline as SP  # noqa: E402

POSE_SANITY_ENV = C.WORLD_POSE_SANITY_ENV


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in CAP_ENVS + (POSE_SANITY_ENV, C.WORLD_MERGE_AFTER_STOP_ENV,
                            "TOWER_WORLD_FINAL_IMMEDIATE"):
        monkeypatch.delenv(name, raising=False)


class FakeSurface:
    def __init__(self, events, **_kw):
        self.events = events

    def solve_landed(self, store):
        self.events.append("surface.solve_landed")

    def poll(self, store):
        self.events.append("surface.poll")

    def close(self):
        self.events.append("surface.close")


def _pair_walk(tmp_path, monkeypatch, *, far_cap, final_immediate=False):
    if far_cap:
        monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")
    else:
        monkeypatch.delenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, raising=False)
    events: list = []
    monkeypatch.setattr(wbs, "BackgroundSurface", lambda **kw: FakeSurface(events, **kw))

    def final_surface_stages(store, world_id, session_id, **kwargs):
        events.append(("final_surface", C.world_surface_far_depth_cap_setting(),
                       C.world_pose_sanity_setting()))
        return {}

    monkeypatch.setattr(wbs, "final_surface_stages", final_surface_stages)
    events, code, store, w, s = _walk(tmp_path, monkeypatch, merge=True,
                                      final_immediate=final_immediate, in_loop={2},
                                      after_stop=True, extra_args=("--surface",),
                                      events=events)
    assert code == 0
    return events


@pytest.mark.parametrize("final_immediate", [False, True], ids=["fi-off", "fi-on"])
def test_the_builder_with_both_on_is_the_merge_only_trace(tmp_path, monkeypatch,
                                                          final_immediate):
    both = _pair_walk(tmp_path / "both", monkeypatch, far_cap=True,
                      final_immediate=final_immediate)
    merge_only = _pair_walk(tmp_path / "merge", monkeypatch, far_cap=False,
                            final_immediate=final_immediate)
    strip = [e for e in both if not isinstance(e, tuple)]
    assert strip == [e for e in merge_only if not isinstance(e, tuple)]

    end = both.index(LOOP_END)
    merge_build = both.index(BUILD, end)
    final_at = both.index(RUN_FINAL)
    final_build = both.index(BUILD, final_at)
    (surface_at,) = [i for i, e in enumerate(both) if isinstance(e, tuple)]
    assert both[end + 1] == POLL_AFTER
    assert end < merge_build < both.index("surface.close", end) < final_at < final_build \
        < surface_at
    # the post-Stop merge is a derived-tree build only: it does not launch a live surface
    assert not {"surface.solve_landed", "surface.poll"} & set(strip[strip.index(LOOP_END):])
    # the in-loop landing did reach the live surface, as today
    assert "surface.solve_landed" in both[:end]
    # the final surface stage runs with the cap on and pose sanity off
    assert both[surface_at][1:] == ({"k": 6.0, "max_drop": 0.25}, False)
    assert merge_only[[i for i, e in enumerate(merge_only)
                       if isinstance(e, tuple)][0]][1:] == (None, False)


def test_the_surface_stage_with_both_on_is_the_cap_alone(tmp_path, monkeypatch):
    store = _world(tmp_path, far=(8.0, 8.0))
    budget = int(_room_blocks(store) * 1.2)
    monkeypatch.setenv(C.WORLD_SURFACE_FAR_DEPTH_CAP_ENV, "on")

    def build(merge):
        if merge:
            monkeypatch.setenv(C.WORLD_MERGE_AFTER_STOP_ENV, "on")
        else:
            monkeypatch.delenv(C.WORLD_MERGE_AFTER_STOP_ENV, raising=False)
        result = SP.surfacify(store, WORLD, SESSION, force=True,
                              params=_lean(max_blocks=budget))
        assert result.state == SP.STATE_OK, result.detail
        return _manifest(store)

    cap_alone = build(False)
    both = build(True)
    assert both["outliers"]["far_depth_cap"]["state"] == "applied"
    assert both["outliers"]["far_depth_cap"] == cap_alone["outliers"]["far_depth_cap"]
    for key in ("voxel", "faces", "params_digest", "median_scene_depth"):
        assert both[key] == cap_alone[key], key
    assert both["detail"]["voxel_coarsened_by"] == 1.0
