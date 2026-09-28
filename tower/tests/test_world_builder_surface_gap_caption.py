"""The surface page's gap caption (WORLD-BUILDER-SURFACE.md §8, v2 `a034eec`: its
IMPLEMENTATION SPEC, shipped under the copy-only exception; manager 109/114).

The caption says only what the page knows. After either `CONFIG.evidence_filter`
branch, on the room page and on an area page, the note reads *A gap is where this page
has no reconstructed surface. It is not proof that nothing is there.* No string the
page shows says a place was not photographed or not captured, was never looked at,
that nothing or nobody looked there, or that nobody photographed it.

§8's verification list, in order:

1. a composed room page and a composed area page, each with `evidence_filter` true and
   false: the note has the new suffix directly after the branch sentence, and on the
   area page the area caption precedes the branch;
2. no composed page contains "nothing looked";
3. a shown-string scan with comments stripped (JS `//` and `/* */`, HTML `<!-- -->`)
   finds none of the six phrases. It has to strip comments: the template comment above
   `let note` says "nobody looked". That comment is not shown and may stay.

The pages are composed by the product (`build_world_render` / `build_area_render`,
representation "surface") from the synthetic solved, surfaced world of
`test_world_builder_appearance.World`. `evidence_filter` reaches CONFIG as it does in
production, read off the surface manifest by `surface_render.evidence_filter_ran`.
Where node is on the host, the page's own caption code runs over a stand-in DOM, so
(1) asserts the text the wearer reads. A node-free check of the note statement's
string literals backs it.

ORACLE marks contract text; CHARACTERIZATION marks a fact of this fixture.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.test_world_builder_appearance import SESSION, WORLD, World, _never_redact
from tests.test_world_builder_component_captions import _DOM, _run
from tests.test_world_builder_components_areas import (
    AREA1,
    _config,
    _entries,
    _finalize,
    _meta,
    build_area,
    write_components,
)
from tests.test_world_builder_fixture_e_honesty import _shown_source

# ORACLE (SURFACE v2 §8): the two evidence-filter branches stay as they were.
BRANCH = {
    True: ("Surfaces appear only where at least two camera views measured them and did "
           "not contradict each other."),
    False: ("Surfaces appear only where the cameras' fused measurements reached the "
            "evidence threshold."),
}
# ORACLE (SURFACE v2 §8, the implementation spec's table and its two literals).
SUFFIX_LITERALS = ["A gap is where this page has no reconstructed surface. ",
                   "It is not proof that nothing is there."]
SUFFIX = "".join(SUFFIX_LITERALS)
# ORACLE (SURFACE v2 §8): the scale note remains as it is.
SCALE = " Scale is unknown, so distances here are relative."
# ORACLE (COMPONENTS §5.4, surface rung): the area caption, for AREA1's capture spans
# (86.7 s .. 109.6 s, the contract's own example).
AREA1_CAPTION = (
    "Surfaces the Tower reconstructed from the walk, from 1:26 to 1:49 of this walk. "
    "This area could not be placed relative to the room, so it is shown on its own: "
    "its position, direction and size are not comparable with the room's. Not to scale.")
AREA1_HEAD = "Area 1 of 2 — not placed in the room"
# ORACLE (SURFACE v2 §8): no shown string, room or area, may say any of these.
BANNED = ("not photographed", "not captured", "never looked", "nothing looked",
          "nobody looked", "nobody photographed")

# "room": a session with no components record, which is every saved world today.
# "room-with-areas": the room page of a split walk (it gains " · 2 more areas").
# "area": AREA1's own page.
KINDS = ("room", "room-with-areas", "area")
PAGE_KEYS = [(kind, ran) for kind in KINDS for ran in (True, False)]


def _page_id(key):
    kind, ran = key
    return f"{kind}-{'evidence_filter' if ran else 'older_surface'}"


def _set_evidence_filter(manifest_path, ran: bool) -> None:
    """Record in the surface manifest whether the per-face evidence filter ran:
    `detail.evidence_filter` is its stats when it ran and null when it did not, the key
    `evidence_filter_ran` believes first."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["detail"] = {"evidence_filter": {"fixture": True} if ran else None}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    """{(kind, evidence_filter): the composed surface page}, built once."""
    from tower.results.world_builder_render import build_area_render, build_world_render

    world = World(tmp_path_factory.mktemp("gapcap"))
    world.build(redactor_factory=_never_redact)
    _finalize(world)
    room_manifest = world.store.world_dir(WORLD) / "surface" / SESSION / "manifest.json"
    out = {}
    for ran in (True, False):
        _set_evidence_filter(room_manifest, ran)
        out[("room", ran)] = build_world_render(world.store, WORLD, SESSION,
                                                representation="surface")
    record = write_components(world.store, _entries(world.kids))
    view = build_area(world, AREA1, record, appearance=False)
    area_manifest = view.area_dir / "surface" / SESSION / "manifest.json"
    for ran in (True, False):
        _set_evidence_filter(room_manifest, ran)
        _set_evidence_filter(area_manifest, ran)
        out[("room-with-areas", ran)] = build_world_render(world.store, WORLD, SESSION,
                                                           representation="surface")
        out[("area", ran)] = build_area_render(world.store, WORLD, SESSION, AREA1,
                                               representation="surface")
    return out


def _caption(page: str, tmp_path) -> dict:
    """The page's own caption code, run under node over a stand-in DOM with the page's
    own CONFIG: the header, the whole caption, and `note`."""
    start = page.index("  const used = CONFIG.frames_used")
    end = page.index("\n", page.index("  cap.append(head", start))
    program = (_DOM
               + "document.getElementById = () => CAP;\n"
               + "const CONFIG = " + json.dumps(_config(page)) + ";\n"
               + "const mesh = {nI: 0};\n"
               + page[start:end] + "\n"
               + "console.log(JSON.stringify({head: CAP.children[0].textContent, "
               + "all: text(CAP), note}));\n")
    return _run(program, tmp_path)


def _note_literals(page: str) -> list:
    """The JavaScript string literals of the `let note = ...;` statement, in order."""
    start = page.index("  let note = ")
    end = page.index(";\n", start)
    return [json.loads(lit) for lit in re.findall(r'"(?:[^"\\]|\\.)*"', page[start:end])]


# ---------------------------------------------------------------------------
# 1. the note: the branch sentence, then directly the new suffix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PAGE_KEYS, ids=_page_id)
def test_the_note_follows_the_branch_with_what_the_page_knows(pages, key, tmp_path):
    kind, ran = key
    page = pages[key]
    config = _config(page)
    # CHARACTERIZATION: this is the surface rung, composed with the filter state this
    # key names (read off the manifest by the product), at the fixture's unknown scale.
    assert _meta(page, "wb-representation") == "surface"
    assert config["evidence_filter"] is ran
    assert config["scale_state"] == "unknown"
    if kind == "area":
        assert _meta(page, "wb-area") == AREA1

    got = _caption(page, tmp_path)
    prefix = AREA1_CAPTION + " " if kind == "area" else ""
    # ORACLE (§8 verification 1): the branch sentence, then directly the new suffix,
    # then the scale note. On the area page the area caption precedes the branch.
    assert got["note"] == prefix + BRANCH[ran] + " " + SUFFIX + SCALE
    assert (BRANCH[ran] + " " + SUFFIX) in got["all"]
    assert got["all"].endswith(got["note"])
    # ORACLE (COMPONENTS §5.4): the header the rung carries.
    assert got["head"] == (AREA1_HEAD if kind == "area" else "Reconstructed surface")
    # ORACLE (§8): the text the wearer reads makes no claim about where nobody looked,
    # however the literals are split.
    for phrase in BANNED:
        assert phrase not in got["all"].casefold(), (key, phrase)
    assert "place the views" not in got["all"]


@pytest.mark.parametrize("kind", KINDS)
def test_the_note_statement_carries_the_specs_two_literals(pages, kind):
    """Node-free: the statement's string literals, in source order."""
    prefix = AREA1_CAPTION + " " if kind == "area" else ""
    for ran in (True, False):
        literals = _note_literals(pages[(kind, ran)])
        # ORACLE (§8 implementation spec): the suffix is the two named literals,
        # concatenated, and it is the last thing in the statement.
        assert literals[-2:] == SUFFIX_LITERALS
        # ORACLE: area caption (area page only), the true branch, the false branch,
        # then the suffix, with nothing between them.
        assert "".join(literals) == (prefix + BRANCH[True] + " " + BRANCH[False] + " "
                                     + SUFFIX)


# ---------------------------------------------------------------------------
# 2. no composed page says "nothing looked", comments included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PAGE_KEYS, ids=_page_id)
def test_no_composed_page_contains_nothing_looked(pages, key):
    # ORACLE (§8 verification 2): the whole composed page, comments and all.
    assert "nothing looked" not in pages[key].casefold()


# ---------------------------------------------------------------------------
# 3. no shown string claims where nobody looked (comments stripped)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PAGE_KEYS, ids=_page_id)
def test_no_shown_string_claims_where_nobody_looked(pages, key):
    shown = _shown_source(pages[key])
    # ORACLE (§8 verification 3): HTML text, aria-labels, labels and JS strings, with
    # comments stripped.
    for phrase in BANNED:
        assert phrase not in shown.casefold(), (key, phrase)
    # CHARACTERIZATION (the scan is not vacuous): the stripped text still holds the
    # caption's shown literals and the buttons' HTML text.
    assert '"' + SUFFIX_LITERALS[0] + '"' in shown
    assert '"Surfaces appear only where at least two camera views measured "' in shown
    assert '<button id="bReset">Reset</button>' in shown


@pytest.mark.parametrize("phrase", BANNED)
def test_the_scan_reads_shown_strings_and_skips_comments(phrase):
    """The stripper behind verification 3: a comment passes, a shown string does not."""
    commented = (f"<!-- {phrase} -->\n<div>ok</div>\n<script>\n// {phrase}\n"
                 f"/* {phrase} */\nconst a = \"ok\"; // {phrase}\n</script>")
    assert phrase not in _shown_source(commented)
    for shown in (f"<div>{phrase}</div>",
                  f'<button aria-label="{phrase}">x</button>',
                  f'<script>const a = "{phrase}"; // ok\n</script>',
                  f"<script>b.textContent = '{phrase}';</script>",
                  f"<script>const c = `{phrase}`;</script>"):
        assert phrase in _shown_source(shown), shown
