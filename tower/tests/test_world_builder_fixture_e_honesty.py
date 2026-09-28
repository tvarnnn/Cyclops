"""FX1-E CPU honesty check on the local, image-bearing Walk 4 world copy.

The copy stays outside the repository.  This module reads its manifests and
proxies, composes pages in memory, and never reads or publishes image chunks.
The producing Walk 4 integration was f58d890; the world/session identifiers
below bind the characterization to that run, not to a synthetic world.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from tests.test_world_builder_appearance_page import _BANNED_VIEWER_CLAIMS, _config, _meta

WORLD = "c81766a3bd3a4eeb9739aa4271fdbb75"
SESSION = "9b03ac81387b471bb90645f6c74ddc03"
CAPTURE = "61f2fdb5d88a4452953e90dce019597a"
DEFAULT_WORLD = Path(
    "C:/Users/tvllo/Projects/Glasses-scratch/wb-coherence-run-2026-09-23/"
    "physical-test/scored/c81766a3/worlds"
) / WORLD

# CHARACTERIZATION: SHA-256 of each frozen W4 appearance manifest.  The mesh
# filename is content addressed by the manifest and is read from this copy.
PAGES = {
    "room": (None, 434, 0, "897af3df92ad293723b49afcd8b7c3cd64e1deb8278cadf6aafc368a8fe0a260"),
    "bath": ("7be3085db326d0de", 481, 0, "c0f578c4374068e04a6f458ed29bd4aaf2ee5a727015a148b2dfa4ecd5dc023f"),
    "closet64": ("df2784b4c3805b6e", 317, 0, "ce2d6c254efb09eb215881077918a79c56ed123ebe9289012e65c543f3ee24a6"),
    "closet21": ("a70f7df590072dba", 389, 0, "428623316bf164af430b9851ac3c68d079b0cc3979bc2355899fa4c5c1c06515"),
}
ROOM_COVERED_CONTROL = (610, 8836)
CONTRACT_BANNED = ("not photographed", "not captured", "never looked", "nobody photographed")
# The contract's four plus T-B's whole-page list, so the literal W4 sentence
# "Nothing was photographed this way" fails here too (review LOW-2).  T-B holds
# only its own list absent from the whole page, comments included; the
# contract's bare "not captured" is not in that list and does occur in two
# viewer comments, which is why the union is held here against the
# comment-stripped text of the W4 pages and of the research page.
BANNED = CONTRACT_BANNED + tuple(
    claim for claim in _BANNED_VIEWER_CLAIMS if claim not in CONTRACT_BANNED)


@pytest.fixture(scope="module")
def frozen_world():
    path = Path(os.environ.get("WB_FIXTURE_E_WORLD", str(DEFAULT_WORLD)))
    if not path.is_dir():
        pytest.skip(f"local-only W4 world copy absent: {path}")
    assert path.name == WORLD, "fixture must name the W4 world, not another capture"
    # The pages are built through WorldStore(path.parent.parent), which needs
    # the store layout <root>\worlds\<id>; a bare copy of the world directory
    # would otherwise fail later with an unexplained "no world ..." error.
    assert path.parent.name == "worlds", (
        f"WB_FIXTURE_E_WORLD must point at <root>\\worlds\\{WORLD} inside a "
        f"world-store root, not a bare copy of the world directory; got {path}")
    # CHARACTERIZATION: bind this local copy to the original receipt lineage.
    session = json.loads((path / "sessions" / SESSION / "session.json")
                         .read_text(encoding="utf-8"))
    assert session["session_id"] == SESSION
    assert session["capture_id"] == CAPTURE
    return path


def _manifest(frozen_world: Path, area_id: str | None, expected_sha: str):
    directory = (frozen_world / "appearance" / SESSION if area_id is None else
                 frozen_world / "areas" / area_id / "appearance" / SESSION)
    data = (directory / "manifest.json").read_bytes()
    # CHARACTERIZATION: fail on a changed local snapshot instead of silently
    # comparing a new artifact with W4's old vertex counts.
    assert hashlib.sha256(data).hexdigest() == expected_sha
    return directory, json.loads(data)


def _shown_source(page: str) -> str:
    """Remove comments before scanning HTML text, labels, and JS strings.

    The kept C1 comment itself says 'not captured'. A whole-source search
    falsely rejects that harmless comment and cannot guard the actual copy.
    """
    page = re.sub(r"<!--.*?-->", "", page, flags=re.DOTALL)
    parts = re.split(r"(<script\b[^>]*>.*?</script>)", page,
                     flags=re.IGNORECASE | re.DOTALL)
    shown = []
    for part in parts:
        if part.lower().startswith("<script"):
            start = part.index(">") + 1
            end = part.lower().rfind("</script>")
            shown.append(part[:start] + _without_js_comments(part[start:end]) + part[end:])
        else:
            shown.append(re.sub(r"/\*.*?\*/", "", part, flags=re.DOTALL))
    return "".join(shown)


def _without_js_comments(source: str) -> str:
    result = []
    quote = None
    i = 0
    while i < len(source):
        if quote is not None:
            char = source[i]
            result.append(char)
            if char == "\\" and i + 1 < len(source):
                i += 1
                result.append(source[i])
            elif char == quote:
                quote = None
            i += 1
            continue
        if source[i] in "\"'`":
            quote = source[i]
            result.append(source[i])
            i += 1
            continue
        if source.startswith("/*", i):
            end = "*/"
            stop = source.find(end, i + len(end))
            assert stop >= 0, "unterminated page comment"
            result.append("\n" * source[i:stop + len(end)].count("\n"))
            i = stop + len(end)
            continue
        if source.startswith("//", i):
            stop = source.find("\n", i)
            i = len(source) if stop < 0 else stop
            continue
        result.append(source[i])
        i += 1
    return "".join(result)


def _reverse_vertices(directory: Path, manifest: dict, keyframe_id: int) -> int:
    """DIAG4 Q2's no-occlusion frustum metric, from this page's own proxy."""
    from tower.world_builder.surface import read_mesh_bytes

    proxy = directory / f"p.{manifest['proxy']['digest']}.bin"
    raw = proxy.read_bytes()
    # CHARACTERIZATION: the bytes counted are the bytes named by the frozen
    # manifest, not a similarly named mesh left beside it.
    assert hashlib.sha256(raw).hexdigest().startswith(manifest["proxy"]["digest"])
    vertices, _faces, _colors, _normals = read_mesh_bytes(raw)
    vertices = np.asarray(vertices, dtype=np.float64)
    camera = next(k for k in manifest["keyframes"] if k["ki"] == keyframe_id)
    rotation = np.asarray(camera["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(camera["translation"], dtype=np.float64)
    centre = -rotation.T @ translation
    reverse = np.diag([-1.0, 1.0, -1.0]) @ rotation
    points = vertices @ reverse.T - reverse @ centre
    z = points[:, 2]
    ahead = z > 0.05
    intrinsics = manifest["camera"]
    u = intrinsics["fx"] * points[ahead, 0] / z[ahead] + intrinsics["cx"]
    v = intrinsics["fy"] * points[ahead, 1] / z[ahead] + intrinsics["cy"]
    return int(((u >= 0) & (u < intrinsics["width"]) &
                (v >= 0) & (v < intrinsics["height"])).sum())


@pytest.mark.parametrize("name", PAGES)
def test_w4_pages_tell_the_truth_at_reverse_poses(frozen_world, name):
    from tower.results.world_builder_render import build_area_render, build_world_render
    from tower.world_builder.store import WorldStore

    area_id, keyframe_id, expected_count, digest = PAGES[name]
    directory, manifest = _manifest(frozen_world, area_id, digest)
    store = WorldStore(frozen_world.parent.parent)
    page = (build_world_render(store, WORLD, SESSION, representation="appearance")
            if area_id is None else
            build_area_render(store, WORLD, SESSION, area_id, representation="appearance"))
    config = _config(page)
    # ORACLE: this really is the page with the frozen proxy and provenance.
    assert _meta(page, "wb-representation") == "appearance"
    assert manifest["format"] == "wb-appearance-keyframes/1"
    assert config["imagery_source"] == manifest["imagery_source"]
    # ORACLE: the contract rule (WORLD-BUILDER-APPEARANCE.md §6.6, :947):
    # privacy_safe is false exactly when imagery_source is not "redacted".
    assert config["privacy_safe"] is (config["imagery_source"] == "redacted")
    if area_id is None:
        assert "area" not in config
    else:
        assert config.get("area", {}).get("id") == area_id
    # CHARACTERIZATION: a fact about this W4 copy, not a rule of the page: its
    # appearance was built from redacted session keyframes.
    assert manifest["imagery_source"] == "redacted"

    shown = _shown_source(page)
    # ORACLE: no shown text, aria label, or JS-assigned text claims a capture
    # absence that the page cannot know. Comments are deliberately excluded.
    for phrase in BANNED:
        assert phrase not in shown.casefold(), (name, phrase)
    # ORACLE (copy): the page's dark treatment and the contract's explanation
    # are present in the page's uncommented source.  Searched in the
    # comment-stripped text, so a commented-out line does not satisfy them.
    # Present is not reachable: `const want = false && ...` passes here.  This
    # does not depend on the reverse count below and does not run the dark
    # trigger on W4 data.  That the count ~0 at these poses drives
    # dark > DARK_SAY_ON is inferred here, not tested; T-D tests the trigger on
    # its synthetic page (test_world_builder_appearance_page.py::
    # test_captured_look_back_outside_the_shown_component_is_called_unreconstructed).
    # OPEN: `dark` is darkness(support.s) and support reads only the field's
    # coverage, never image detail.  A no-occlusion field from this proxy and
    # every manifest keyframe over-states coverage, so it under-states dark;
    # dark > DARK_SAY_ON on it at these poses would test the link on W4 with
    # no GPU and no image chunks.
    assert 'const HINT_DARK = "Not reconstructed from here";' in shown
    assert 'const HINT_DARK_TAP = "Tap to turn back";' in shown
    assert 'b.textContent = HINT_DARK;' in shown
    assert 's.textContent = HINT_DARK_TAP;' in shown
    assert 'el.classList.add("on");' in shown
    assert 'const DARK_SAY_ON = 0.90, DARK_SAY_OFF = 0.45, DARK_SAY_MS = 320;' in shown
    assert 'const on = darkShown ? dark > DARK_SAY_OFF : dark > DARK_SAY_ON;' in shown
    assert 'if (on && !want) requestDraw();' in shown

    count = _reverse_vertices(directory, manifest, keyframe_id)
    # CHARACTERIZATION: pins W4's shipped, hash-pinned proxy files, the product
    # decoder (`surface.read_mesh_bytes`) and this module's metric (DIAG4 Q2),
    # not the pipeline.  Nothing here rebuilds anything, so a pipeline fix to
    # component loss or reverse coverage, better or worse, cannot move these
    # numbers.  A rebuilt copy trips the SHA-256 pin in _manifest before this
    # line is reached.
    # OPEN: seeing a real fix needs a rebuild path (frozen inputs ->
    # area/surface/appearance into tmp_path -> this same metric), or DIAG4's
    # metric run on other corpora (W3, acc2).
    assert count == expected_count, (name, keyframe_id, count)

    if name == "room":
        # CHARACTERIZATION: a room reverse control with substantial geometry;
        # prevents a test that always reports zero from passing.
        control_id, control_count = ROOM_COVERED_CONTROL
        assert _reverse_vertices(directory, manifest, control_id) == control_count


def test_research_page_reports_its_actual_imagery_source(tmp_path, monkeypatch):
    """C1 LOW-2: the marker's JS exists on every page; inspect served CONFIG."""
    from tests.test_world_builder_appearance import (
        SESSION as SYNTH_SESSION, WORLD as SYNTH_WORLD, World, _paint_forbidden_sources,
    )
    from tests.test_world_builder_raw_imagery import _raw_params
    from tower.world_builder.appearance_render import build_appearance_page
    from tower.world_builder.raw_imagery import RAW_IMAGERY_ENV

    research = World(tmp_path)
    _paint_forbidden_sources(research)
    assert research.build(params=_raw_params()).state == "ok"
    monkeypatch.setenv(RAW_IMAGERY_ENV, "1")
    page = build_appearance_page(research.store, SYNTH_WORLD, SYNTH_SESSION)
    config = _config(page)
    # ORACLE: the served research artifact's provenance, not a marker string.
    assert _meta(page, "wb-representation") == "appearance"
    assert config["imagery_source"] == research.manifest()["imagery_source"]
    assert config["imagery_source"] != "redacted"
    assert config["privacy_safe"] is False
    for phrase in BANNED:
        assert phrase not in _shown_source(page).casefold(), phrase
