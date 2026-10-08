"""Exp3 (`PREREG-EXP3-MIRROR-ORACLE-20261008.md`, addendum 1): the oracle mirror-interior
mask module, on synthetic polygons only -- no real walk, no GPU, no live store.

`EXP3_MUTANT` negative controls live in `exp3_mirror_mutator.py`; run e.g.

    EXP3_MUTANT=rasterize_last_only python -m pytest -p exp3_mirror_mutator \\
        tests/test_world_builder_mirror_masks.py
"""

import json

import numpy as np
import pytest

from tower.config import world_mirror_masks_setting, world_mirror_polygons_path_setting
from tower.world_builder import mirror_masks as MM

RECORD = {
    "walk": 7, "keyframe": 926, "image_sha256": "a" * 64,
    "width": 10, "height": 8, "uncertain": False,
    "polygons": [[[2, 2], [6, 2], [6, 5], [2, 5]]],
}


def _write(tmp_path, records):
    path = tmp_path / "mirror-polygons.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# -- load_polygons -----------------------------------------------------------------------

def test_load_polygons_round_trips_a_clean_record(tmp_path):
    path = _write(tmp_path, [RECORD])
    by_key = MM.load_polygons(path)
    assert set(by_key) == {(7, 926)}
    assert by_key[(7, 926)]["polygons"] == [[(2, 2), (6, 2), (6, 5), (2, 5)]]


def test_load_polygons_accepts_an_explicit_empty_record(tmp_path):
    empty = {**RECORD, "keyframe": 846, "polygons": []}
    path = _write(tmp_path, [empty])
    by_key = MM.load_polygons(path)
    assert by_key[(7, 846)]["polygons"] == []


@pytest.mark.parametrize("drop_key", ["walk", "keyframe", "image_sha256", "width",
                                      "height", "uncertain", "polygons"])
def test_load_polygons_rejects_a_record_missing_a_required_key(tmp_path, drop_key):
    bad = {k: v for k, v in RECORD.items() if k != drop_key}
    path = _write(tmp_path, [bad])
    with pytest.raises(MM.MirrorPolygonError):
        MM.load_polygons(path)


def test_load_polygons_rejects_duplicate_walk_keyframe(tmp_path):
    path = _write(tmp_path, [RECORD, RECORD])
    with pytest.raises(MM.MirrorPolygonError):
        MM.load_polygons(path)


def test_load_polygons_rejects_a_vertex_outside_the_frame(tmp_path):
    bad = {**RECORD, "polygons": [[[2, 2], [600, 2], [6, 5]]]}
    path = _write(tmp_path, [bad])
    with pytest.raises(MM.MirrorPolygonError):
        MM.load_polygons(path)


def test_load_polygons_rejects_a_non_integer_vertex(tmp_path):
    bad = {**RECORD, "polygons": [[[2.5, 2], [6, 2], [6, 5]]]}
    path = _write(tmp_path, [bad])
    with pytest.raises(MM.MirrorPolygonError):
        MM.load_polygons(path)


def test_load_polygons_rejects_a_two_vertex_polygon(tmp_path):
    bad = {**RECORD, "polygons": [[[2, 2], [6, 2]]]}
    path = _write(tmp_path, [bad])
    with pytest.raises(MM.MirrorPolygonError):
        MM.load_polygons(path)


def test_file_sha256_matches_the_bytes_on_disk(tmp_path):
    path = _write(tmp_path, [RECORD])
    import hashlib
    assert MM.file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# -- rasterize_polygons -------------------------------------------------------------------

def test_rasterize_polygons_is_all_false_when_empty():
    mask = MM.rasterize_polygons([], 10, 8)
    assert mask.shape == (8, 10)
    assert not mask.any()


def test_rasterize_polygons_marks_the_interior():
    mask = MM.rasterize_polygons([[(2, 2), (6, 2), (6, 5), (2, 5)]], 10, 8)
    assert mask[3, 3]
    assert not mask[0, 0]
    assert not mask[7, 9]


def test_rasterize_polygons_unions_disconnected_regions():
    polys = [[(0, 0), (2, 0), (2, 2), (0, 2)], [(7, 6), (9, 6), (9, 7), (7, 7)]]
    mask = MM.rasterize_polygons(polys, 10, 8)
    assert mask[1, 1]        # inside the first polygon
    assert mask[6, 8]        # inside the second, disconnected polygon
    assert not mask[4, 4]    # between the two: untouched
    assert int(mask.sum()) == int(MM.rasterize_polygons([polys[0]], 10, 8).sum()) + \
        int(MM.rasterize_polygons([polys[1]], 10, 8).sum())


def test_rasterize_polygons_applies_no_dilation():
    # A single pixel polygon marks only that pixel's immediate footprint -- the
    # prereg refuses dilation (section 4), unlike the existing transient mask's
    # 12 px ellipse.
    mask = MM.rasterize_polygons([[(5, 5), (6, 5), (6, 6), (5, 6)]], 10, 8)
    assert mask[5, 5]
    assert not mask[0, 0]
    assert not mask[7, 9]


# -- oracle_mask_in_solver_space / interior_mask_for_frame ---------------------------------

def _identity_maps(width, height):
    m1, m2 = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return m1, m2, (0, 0, width, height)


def test_oracle_mask_in_solver_space_matches_direct_rasterize_under_identity_remap():
    m1, m2, roi = _identity_maps(RECORD["width"], RECORD["height"])
    record = {**RECORD, "polygons": [[(2, 2), (6, 2), (6, 5), (2, 5)]]}
    got = MM.oracle_mask_in_solver_space(record, m1=m1, m2=m2, roi=roi,
                                         out_shape=(RECORD["height"], RECORD["width"]))
    want = MM.rasterize_polygons(record["polygons"], RECORD["width"], RECORD["height"])
    assert np.array_equal(got, want)


def test_oracle_mask_in_solver_space_raises_on_a_shape_mismatch():
    m1, m2, roi = _identity_maps(RECORD["width"], RECORD["height"])
    record = {**RECORD, "polygons": [[(2, 2), (6, 2), (6, 5), (2, 5)]]}
    with pytest.raises(MM.MirrorPolygonError):
        MM.oracle_mask_in_solver_space(record, m1=m1, m2=m2, roi=roi, out_shape=(999, 999))


def test_interior_mask_for_frame_is_none_outside_the_annotated_range():
    m1, m2, roi = _identity_maps(RECORD["width"], RECORD["height"])
    by_key = {(7, 926): {**RECORD, "polygons": [[(2, 2), (6, 2), (6, 5), (2, 5)]]}}
    out_shape = (RECORD["height"], RECORD["width"])
    # a keyframe never annotated: untouched, not an exclusion of anything
    assert MM.interior_mask_for_frame(by_key, 7, 927, m1=m1, m2=m2, roi=roi,
                                      out_shape=out_shape) is None
    assert MM.interior_mask_for_frame(by_key, None, 926, m1=m1, m2=m2, roi=roi,
                                      out_shape=out_shape) is None


def test_interior_mask_for_frame_explicit_empty_is_an_array_not_none():
    m1, m2, roi = _identity_maps(RECORD["width"], RECORD["height"])
    by_key = {(7, 846): {**RECORD, "keyframe": 846, "polygons": []}}
    out_shape = (RECORD["height"], RECORD["width"])
    got = MM.interior_mask_for_frame(by_key, 7, 846, m1=m1, m2=m2, roi=roi, out_shape=out_shape)
    assert got is not None
    assert not got.any()


# -- records --------------------------------------------------------------------------------

def test_off_record_is_not_requested():
    rec = MM.off_record()
    assert rec["requested"] is False
    assert rec["state"] == MM.STATE_OFF


def test_applied_record_shape():
    rec = MM.applied_record(
        polygons_path="x.jsonl", polygons_sha256="f" * 64, walk=7, images=10,
        images_with_record=4, images_touched=2, oracle_pixels_excluded=37,
        oracle_pixel_fraction=0.01)
    assert rec["requested"] is True
    assert rec["state"] == MM.STATE_OK
    assert rec["images_touched"] == 2
    assert rec["oracle_pixels_excluded"] == 37


def test_mirror_extraction_masks_duck_types_rule_id():
    masks = MM.MirrorExtractionMasks(masked={"a.jpg": {"image_sha1": "x", "mask_sha1": "y",
                                                       "masked_frac": 0.1}})
    assert masks.params.rule_id() == "mirror-oracle"
    assert masks.masked["a.jpg"]["masked_frac"] == 0.1


# -- the config switches (independence from TOWER_WORLD_SOLVE_MASKS) -----------------------

def test_mirror_masks_setting_default_off(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_MIRROR_MASKS", raising=False)
    assert world_mirror_masks_setting() is False


def test_mirror_masks_setting_on(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_MIRROR_MASKS", "on")
    assert world_mirror_masks_setting() is True


def test_mirror_masks_setting_independent_of_solve_masks(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_MIRROR_MASKS", "on")
    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS", raising=False)
    from tower.config import world_solve_masks_setting
    assert world_mirror_masks_setting() is True
    assert world_solve_masks_setting() is False


def test_mirror_polygons_path_setting_blank_is_none(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_MIRROR_POLYGONS", "  ")
    assert world_mirror_polygons_path_setting() is None


def test_mirror_polygons_path_setting_strips(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_MIRROR_POLYGONS", "  x.jsonl  ")
    assert world_mirror_polygons_path_setting() == "x.jsonl"
