"""LPbig's solver-only mask rule, on synthetic detector components."""

import json
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tower.world_builder import solve_masks as SM, transients as T


H, W = 100, 359
NAME = "00000010.jpg"


def _part(box=None, *, hand=None):
    hands = np.zeros((H, W), bool)
    phones = np.zeros((H, W), bool)
    if box is not None:
        x0, y0, x1, y1 = box
        phones[y0:y1, x0:x1] = True
    if hand is not None:
        x0, y0, x1, y1 = hand
        hands[y0:y1, x0:x1] = True
    return hands, phones


def _described(*rows, params=None):
    params = params or T.TransientParams(unheld_phone=True)
    return [(name, (H, W), T.phone_components(parts, params)) for name, parts in rows]


def test_default_off_and_rule_id_are_isolated(monkeypatch):
    from tower.config import get_settings, world_solve_masks_unheld_phone_setting

    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS_UNHELD_PHONE", raising=False)
    assert world_solve_masks_unheld_phone_setting() is False
    assert get_settings().world_solve_masks_unheld_phone is False
    assert SM.solver_params().unheld_phone is False
    old = T.TransientParams()
    on = replace(old, unheld_phone=True)
    assert old.rule_id() == T.TransientParams(unheld_phone=False).rule_id()
    assert on.rule_id() == old.rule_id() + "|unheld-phone-low0.75-a0.005-w2"
    assert SM.mask_key(T.COMPONENT_GDSAM, old, NAME, "abc") == SM.mask_key(
        T.COMPONENT_GDSAM, on, NAME, "abc")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_UNHELD_PHONE", "on")
    assert world_solve_masks_unheld_phone_setting() is True
    assert SM.solver_params().unheld_phone is True


def test_lpbig_matches_research_patch_mask_and_cross_detector_neighbour():
    # 20x10 is 200 px (>= 0.5% of 35,900); its bottom is in the lower quarter.
    phone = _part((80, 80, 100, 90))
    blank = _part()
    neighbour = _part((80, 80, 100, 90))
    params = T.TransientParams(unheld_phone=True)
    rows = _described(("00000009.jpg", (blank, neighbour)),
                      (NAME, (phone, blank)), params=params)
    picked = T.unheld_phone_additions(rows, params)
    assert picked[NAME] == [(0, 1)]
    added = phone[1]
    expected = T._ellipse_dilate(added, T.dilation_radius(params, W))
    assert np.array_equal(T.compose((phone, blank), params, (H, W), add=added), expected)
    assert not T.compose((phone, blank), T.TransientParams(), (H, W)).any()


def test_window_uses_sorted_capture_order_with_gaps_and_endpoints():
    phone = _part((80, 80, 100, 90))
    blank = _part()
    params = T.TransientParams(unheld_phone=True)
    # Names 1 and 50 are adjacent keyframes despite the numeric gap. Name 200
    # is three positions away from 1 and must not be its neighbour.
    rows = _described(("00000200.jpg", (phone, blank)),
                      ("00000120.jpg", (blank, blank)),
                      ("00000090.jpg", (blank, blank)),
                      ("00000050.jpg", (phone, blank)),
                      ("00000001.jpg", (phone, blank)), params=params)
    picked = T.unheld_phone_additions(rows, params)
    assert "00000001.jpg" in picked
    assert "00000050.jpg" in picked
    assert "00000200.jpg" not in picked
    # The last keyframe qualifies when a matching phone lies two positions back.
    rows[-1] = ("00000001.jpg", (H, W), [])
    rows[2] = ("00000090.jpg", (H, W), T.phone_components((phone, blank), params))
    assert "00000200.jpg" in T.unheld_phone_additions(rows, params)


@pytest.mark.parametrize("box,expected", [
    ((80, 65, 100, 75), True),     # bottom/H exactly .75
    ((80, 64, 100, 74), False),
    ((80, 80, 98, 90), True),      # 180 px, >= .005 of 35,900
    ((80, 80, 97, 90), False),     # 170 px
])
def test_lower_quarter_and_area_boundaries(box, expected):
    # Area boundary uses 180 pixels: 0.005 * (100 * 359) = 179.5.
    phone = _part(box)
    params = T.TransientParams(unheld_phone=True)
    rows = _described(("00000001.jpg", (phone, _part())),
                      ("00000002.jpg", (phone, _part())), params=params)
    assert ("00000001.jpg" in T.unheld_phone_additions(rows, params)) is expected


def test_held_phone_and_unmatched_phone_do_not_get_added():
    phone = _part((80, 80, 100, 90), hand=(75, 80, 81, 90))
    alone = _part((180, 80, 200, 90))
    params = T.TransientParams(unheld_phone=True)
    rows = _described(("00000001.jpg", (phone, _part())),
                      ("00000002.jpg", (alone, _part())), params=params)
    assert T.unheld_phone_additions(rows, params) == {}
    assert T.compose((phone, _part()), params, (H, W)).any()


@pytest.mark.parametrize("shift,expected", [(40, True), (41, False)])
def test_box_iou_boundary_when_centroids_are_farther_than_30_pixels(shift, expected):
    params = T.TransientParams(unheld_phone=True)
    original = _part((20, 80, 80, 90))
    nearby = _part((20 + shift, 80, 80 + shift, 90))
    rows = _described(("00000001.jpg", (original, _part())),
                      ("00000002.jpg", (nearby, _part())), params=params)
    # 60 px wide, shifted 40: IoU = 20/100 = 0.2. Shift 41: 19/101.
    assert ("00000001.jpg" in T.unheld_phone_additions(rows, params)) is expected


@pytest.mark.parametrize("shift,expected", [(30, True), (31, False)])
def test_centroid_boundary_when_boxes_do_not_overlap(shift, expected):
    params = T.TransientParams(unheld_phone=True)
    original = _part((20, 80, 40, 90))
    nearby = _part((20 + shift, 80, 40 + shift, 90))
    rows = _described(("00000001.jpg", (original, _part())),
                      ("00000002.jpg", (nearby, _part())), params=params)
    assert ("00000001.jpg" in T.unheld_phone_additions(rows, params)) is expected


def test_area_exactly_half_percent_is_inclusive():
    params = T.TransientParams(unheld_phone=True)
    component = {"ci": 0, "label": 1, "bbox": (20, 75, 29, 84), "area": 100,
                 "center": (24.5, 79.5), "held": False}
    rows = [("00000001.jpg", (100, 200), [component]),
            ("00000002.jpg", (100, 200), [component])]
    assert T.unheld_phone_additions(rows, params)["00000001.jpg"] == [(0, 1)]
    smaller = dict(component, area=99)
    rows[0] = ("00000001.jpg", (100, 200), [smaller])
    assert "00000001.jpg" not in T.unheld_phone_additions(rows, params)


def test_on_solver_record_and_png_change_but_surface_components_do_not(tmp_path, monkeypatch):
    from tower.config import world_solve_masks_unheld_phone_setting

    names = ["00000001.jpg", "00000002.jpg"]
    images = tmp_path / "images"
    images.mkdir()
    for name in names:
        ok, jpg = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))
        assert ok
        (images / name).write_bytes(jpg.tobytes())
    workspace = SimpleNamespace(root=tmp_path, images_dir=images)
    phone = _part((80, 80, 100, 90))
    blank = _part()
    for name in names:
        sha = SM.file_sha1(images / name)
        for component, part in zip(T.TransientParams().components, (phone, blank)):
            T.write_component(SM.component_path(SM.cache_dir(workspace), name, component, sha),
                              SM.mask_key(component, T.TransientParams(), name, sha),
                              *part, image_sha1=sha, seconds=0.0)
    monkeypatch.setattr(SM.time, "time", lambda: 100.0)
    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS_UNHELD_PHONE", raising=False)
    off = SM.ensure_solver_masks(workspace, names, shape=(H, W))
    off_png = (SM.masks_dir(workspace) / f"{names[0]}.png").read_bytes()
    off_record = json.dumps(off.record(), sort_keys=True, default=str)
    off_keys = [SM.mask_key(component, off.params, names[0], SM.file_sha1(images / names[0]))
                for component in off.params.components]
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_UNHELD_PHONE", "off")
    explicit_off = SM.ensure_solver_masks(workspace, names, shape=(H, W))
    assert (SM.masks_dir(workspace) / f"{names[0]}.png").read_bytes() == off_png
    assert json.dumps(explicit_off.record(), sort_keys=True, default=str) == off_record
    assert [SM.mask_key(component, explicit_off.params, names[0], SM.file_sha1(images / names[0]))
            for component in explicit_off.params.components] == off_keys
    assert "unheld_phone" not in off.record()
    off_index = (SM.cache_dir(workspace) / SM.INDEX_FILENAME)
    assert not off_index.exists()
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS_UNHELD_PHONE", "on")
    assert world_solve_masks_unheld_phone_setting()
    on = SM.ensure_solver_masks(workspace, names, shape=(H, W))
    on_png = (SM.masks_dir(workspace) / f"{names[0]}.png").read_bytes()
    assert on_png != off_png
    assert on.record()["unheld_phone"] == {"images": 2, "components": 2}
    assert on.record()["rule"] != off.record()["rule"]
    assert on.masked[names[0]]["mask_sha1"] != off.masked[names[0]]["mask_sha1"]
    assert json.dumps(off.record(), sort_keys=True, default=str) == off_record
    # The cached components from which surface and appearance compose did not move.
    sha = SM.file_sha1(images / names[0])
    for component in T.TransientParams().components:
        path = SM.component_path(SM.cache_dir(workspace), names[0], component, sha)
        assert T.read_component(path, SM.mask_key(component, T.TransientParams(), names[0], sha),
                                (H, W)) is not None
    assert not T.compose((phone, blank), T.TransientParams(), (H, W)).any()
