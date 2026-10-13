"""`TOWER_WORLD_SURFACE_LONGFACE_PRUNE`: decimated faces that leave the archive
surface are removed after the phone level is fitted, and nothing else is.

Off is today byte for byte; on removes a face that bridges a gap, keeps a long
face that lies on the surface, never grows a level past its page, and never
fails a build."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_world_builder_surface_pipeline import (
    SESSION,
    WORLD,
    _manifest,
    _params,
    _synthetic_world,
)
from tower import config
from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

ENV = "TOWER_WORLD_SURFACE_LONGFACE_PRUNE"
VOX = 0.02


def _grid(nx, ny, step, z=0.0, x0=0.0, y0=0.0):
    """A flat triangulated sheet, (V, F), `step` apart."""
    xs = x0 + np.arange(nx) * step
    ys = y0 + np.arange(ny) * step
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1).astype(np.float32)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return V, F.astype(np.int64)


def _two_patches():
    """Level 0: two 0.5 m sheets 1 m apart along x, voxel-scale triangles."""
    Va, Fa = _grid(26, 26, VOX, x0=0.0)
    Vb, Fb = _grid(26, 26, VOX, x0=1.5)
    return np.concatenate([Va, Vb]), np.concatenate([Fa, Fb + len(Va)])


class TestTheSwitch:
    @pytest.mark.parametrize("value,on", [(None, False), ("", False), ("off", False),
                                          ("on", True), ("ON", False), ("1", False),
                                          ("true", False), (" on", False)])
    def test_only_exactly_on_is_on(self, monkeypatch, value, on):
        if value is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, value)
        assert config.world_surface_longface_prune_setting() is on
        assert SP._longface_prune_on() is on


class TestTheTest:
    def test_a_face_bridging_a_gap_is_removed(self):
        V0, F0 = _two_patches()
        # A coarse level: the two sheets' own big triangles, plus one face
        # decimation stretched across the 1 m of nothing between them.
        V = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
                      [1.5, 0, 0], [1.5, 0.5, 0], [2.0, 0, 0], [2.0, 0.5, 0]], np.float32)
        F = np.array([[0, 1, 3], [0, 3, 2],          # sheet a, on the surface
                      [4, 6, 7], [4, 7, 5],          # sheet b, on the surface
                      [1, 4, 5]], np.int64)          # the bridge
        drop, off = S.off_source_faces(V0, V, F, VOX)
        assert drop.tolist() == [False, False, False, False, True]
        assert off[4] > S.LONGFACE_MAX_OFF_VOXELS
        assert off[:4].max() < 1.5, "a big flat face ON the surface is near it everywhere"
        V2, F2, _C2, rec = S.prune_off_source_faces(V0, V, F, None, VOX)
        assert len(F2) == 4 and rec["faces_removed"] == 1
        assert 0 < rec["area_removed_frac"] < 1

    def test_a_long_face_on_the_surface_is_kept(self):
        """The w9b finding: the phone level's > 1 m faces are flat walls, not bridges."""
        V0, F0 = _grid(151, 151, VOX)                      # a 3 m wall
        V = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]], np.float32)
        F = np.array([[0, 1, 3], [0, 3, 2]], np.int64)     # two 4.2 m-edged faces
        drop, off = S.off_source_faces(V0, V, F, VOX)
        assert not drop.any()
        assert (off > 0).all(), "both were long enough to be tested"

    def test_a_normal_mesh_is_untouched(self):
        V0, F0 = _two_patches()
        rng = np.random.default_rng(0)
        V = V0 + rng.normal(0, 0.3 * VOX, V0.shape).astype(np.float32)
        C = rng.integers(0, 255, (len(V), 3), dtype=np.uint8)
        V2, F2, C2, rec = S.prune_off_source_faces(V0, V, F0, C, VOX)
        assert rec["faces_removed"] == 0
        assert V2 is V and F2 is F0 and C2 is C, "nothing removed is nothing rewritten"

    def test_an_empty_level_is_fine(self):
        V0, _F0 = _two_patches()
        drop, off = S.off_source_faces(V0, np.zeros((0, 3)), np.zeros((0, 3), np.int64), VOX)
        assert len(drop) == 0 and len(off) == 0


class TestThePackStage:
    def test_a_pruned_level_never_grows(self):
        V0, _F0 = _two_patches()
        V = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [1.5, 0, 0], [1.5, 0.5, 0]],
                     np.float32)
        F = np.array([[0, 1, 2], [1, 3, 4]], np.int64)
        buf = S.write_mesh_bytes(V, F, None, S.vertex_normals(V, F))
        state = {"levels": []}
        V2, F2, _C2, buf2 = SP._prune_longfaces(state, 1, V0, (V, F, None), buf, VOX)
        assert len(F2) == 1 and len(buf2) < len(buf)
        rec = state["levels"][0]
        assert rec["faces_removed"] == 1 and rec["bytes_after"] == len(buf2)

    def test_an_error_publishes_the_level_unchanged(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("synthetic")

        monkeypatch.setattr(SP, "prune_off_source_faces", boom)
        V0, F0 = _two_patches()
        buf = S.write_mesh_bytes(V0, F0, None)
        state = {"levels": []}
        out = SP._prune_longfaces(state, 2, V0, (V0, F0, None), buf, VOX)
        assert out[0] is V0 and out[1] is F0 and out[3] is buf
        assert "synthetic" in state["levels"][0]["error"]

    def test_a_level_that_would_lose_every_face_is_kept(self, monkeypatch):
        V0, _F0 = _two_patches()
        V = np.array([[1.0, 5, 5], [1.5, 5, 5], [1.0, 5.5, 5]], np.float32)  # nowhere near
        F = np.array([[0, 1, 2]], np.int64)
        buf = S.write_mesh_bytes(V, F, None)
        state = {"levels": []}
        out = SP._prune_longfaces(state, 2, V0, (V, F, None), buf, VOX)
        assert out[3] is buf and len(out[1]) == 1
        assert state["levels"][0]["kept_unchanged"]


class TestThroughTheBuild:
    """`surfacify` on the CPU fixture `test_world_builder_surface_pipeline` uses."""

    def test_off_is_byte_identical_and_on_respects_the_page(self, tmp_path, monkeypatch):
        from tower.world_builder.surface_render import mesh_bytes_for_page, page_overhead_bytes

        def build(sub, value):
            if value is None:
                monkeypatch.delenv(ENV, raising=False)
            else:
                monkeypatch.setenv(ENV, value)
            store = _synthetic_world(tmp_path / sub)
            # Fixture probe at no budget, then a page that forces a fit.
            SP.surfacify(store, WORLD, SESSION, params=_params(mobile_page_bytes=0))
            free = _manifest(store)["levels"][1]
            budget = page_overhead_bytes() + (4 * free["bytes"] // 3) // 2
            result = SP.surfacify(store, WORLD, SESSION, force=True,
                                  params=_params(mobile_page_bytes=budget))
            assert result.state == SP.STATE_OK
            man = _manifest(store)
            blobs = [_find(tmp_path / sub, e["file"]).read_bytes() for e in man["levels"]]
            return man, blobs, budget

        man_unset, blobs_unset, _b = build("unset", None)
        man_off, blobs_off, _b = build("off", "off")
        assert blobs_unset == blobs_off, "off is today, byte for byte"
        assert "longface_prune" not in (man_unset["detail"] or {})
        assert "longface_prune" not in (man_off["detail"] or {})

        man_on, blobs_on, budget = build("on", "on")
        rec = man_on["detail"]["longface_prune"]
        assert rec["switch"] == ENV and len(rec["levels"]) == 1
        assert rec["levels"][0]["level"] == 1 and "error" not in rec["levels"][0]
        phone = man_on["levels"][man_on["mobile_level"]]
        assert phone["bytes"] == len(blobs_on[1]) <= mesh_bytes_for_page(budget)
        # The fixture is a closed box seen from inside: nothing to bridge.
        assert rec["levels"][0]["faces_removed"] == 0
        assert blobs_on == blobs_unset, "a prune that removes nothing changes nothing"


def _find(base, name):
    hits = list(base.rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]
