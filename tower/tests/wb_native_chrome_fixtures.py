"""The request matrix and the fixture worlds of U1.1's byte-identity golden.

`tests/test_world_builder_native_chrome.py` holds the native-chrome switch
(`TOWER_WORLD_NATIVE_CHROME`) to "off is today, byte for byte" against
`golden/world_builder_native_chrome_dc35d51.json`. That golden was recorded by
THIS FILE, copied unchanged into a detached worktree of dc35d51 -- the deployed
`world-builder/ux-tower-v1` before native chrome existed -- and run there:

    PYTHONPATH=<dc35d51 worktree>/tower python tests/wb_native_chrome_fixtures.py <out.json>

So it must import nothing that dc35d51 does not have. It builds the fixture
worlds (the synthetic solved, surfaced world of `test_world_builder_appearance`)
and sends every request of the matrix through the real routes (`TestClient`).

WHAT IS MASKED, AND WHY ONLY THAT. An appearance build names itself
`f"{time.time_ns():x}{os.getpid():x}"` (`appearance_pipeline.build_id`), and a
build's id becomes its `epoch` and so its revision. A surface manifest's
`built_at` (the surface rung's revision) is a wall-clock float, and its mesh file
is named after the build's clock. Those strings -- read back from the manifests
the builds wrote (`volatile`), never guessed by pattern -- are replaced by
`<VOLATILE>` before hashing. Nothing else is: every other byte of every body, and
every header, is kept (`content-length` as "the body's own length" when it is).

The appearance manifests themselves are NOT in the golden (`data_requests`): they
carry their build's own stage timings and parameter digest. The test compares them
in-suite instead, with and without `wb-chrome`, against one built world.

KEEP IN STEP: add a request here only together with a new golden recorded on a
base commit, never from a tree whose page output is under review.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:      # run as a script from a base worktree
    sys.path.insert(0, str(HERE.parent))

from tests.test_world_builder_appearance import (  # noqa: E402
    SESSION,
    WORLD,
    World,
    _app,
    _never_redact,
    _paint_forbidden_sources,
)

SWITCH_ENV = "TOWER_WORLD_NATIVE_CHROME"
RAW_ENV = "TOWER_WORLD_RAW_IMAGERY"
V = "appearance-1"
CHROME_VALUES = (None, "web", "native", "NATIVE", "", "x")
TRANSPORTS = ("app", "tower")
AREA = "a1a1a1a1a1a1a1a1"

# kind of each request: an appearance page native chrome may change, a page of a
# lower rung it must not, or a JSON/binary route it must not.
APPEARANCE, LOWER, DATA = "appearance", "lower", "data"


def build_worlds(tmp: Path) -> dict:
    """The fixture worlds, each under its own root: name -> root."""
    from tests.test_world_builder_components_areas import (
        _entries,
        _finalize,
        build_area,
        write_components,
    )
    from tower.world_builder import appearance as A

    roots = {}
    room = World(tmp / "room")
    room.build(redactor_factory=_never_redact)
    roots["room"] = room.root

    areas = World(tmp / "areas")
    areas.build(redactor_factory=_never_redact)
    _finalize(areas)
    record = write_components(areas.store, _entries(areas.kids))
    build_area(areas, AREA, record)
    roots["areas"] = areas.root

    research = World(tmp / "research")
    _paint_forbidden_sources(research)
    from tower.world_builder import raw_imagery as RAWIMG

    research.build(params=A.AppearanceParams(selection_samples=4000,
                                             imagery_source=RAWIMG.IMAGERY_RAW))
    roots["research"] = research.root

    # no appearance at all: `auto` falls to the surface rung
    roots["surface"] = World(tmp / "surface").root

    # the sparse rung: conftest's two-segment derived tree (world w0, session s0),
    # the only fixture here with points -- the synthetic world above has none
    import tests.conftest as conftest

    derived = getattr(conftest.derived_world, "__pytest_wrapped__", None)
    derived = derived.obj if derived is not None else conftest.derived_world.__wrapped__
    sparse_root = tmp / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)
    store, _w, _s = derived(sparse_root)
    roots["sparse"] = store.root
    return roots


SPARSE_WORLD, SPARSE_SESSION = "w0", "s0"
SPARSE_ROOM = f"/worlds/{SPARSE_WORLD}/render"


def _sparse(transport, **extra):
    return {"session_id": SPARSE_SESSION, "viewer": V, "transport": transport, **extra}


def _room(transport, **extra):
    return {"session_id": SESSION, "viewer": V, "transport": transport, **extra}


def requests() -> list:
    """(key, world, path, params, kind, env): every request the golden pins, with no
    `wb-chrome`. `with_chrome` adds each value of CHROME_VALUES."""
    room = f"/worlds/{WORLD}/render"
    area = f"/worlds/{WORLD}/areas/{SESSION}/{AREA}/render"
    out = []
    for t in TRANSPORTS:
        out += [
            (f"room/{t}", "room", room, _room(t), APPEARANCE, {}),
            (f"room-with-areas/{t}", "areas", room, _room(t), APPEARANCE, {}),
            (f"area/{t}", "areas", area, {"transport": t}, APPEARANCE, {}),
            (f"research/{t}", "research", room, _room(t), APPEARANCE, {RAW_ENV: "1"}),
            (f"room-pinned-appearance/{t}", "room", room,
             _room(t, representation="appearance"), APPEARANCE, {}),
            (f"area-pinned-appearance/{t}", "areas", area,
             {"transport": t, "representation": "appearance"}, APPEARANCE, {}),
            # the lower rungs (N7): pinned surface, a session with no appearance,
            # the diagnostics view (sparse), a client that cannot draw appearance
            (f"room-pinned-surface/{t}", "room", room, _room(t, representation="surface"),
             LOWER, {}),
            (f"area-pinned-surface/{t}", "areas", area,
             {"transport": t, "representation": "surface"}, LOWER, {}),
            (f"no-appearance/{t}", "surface", room, _room(t), LOWER, {}),
            (f"sparse/{t}", "sparse", SPARSE_ROOM, _sparse(t), LOWER, {}),
            (f"diagnostics/{t}", "sparse", SPARSE_ROOM, _sparse(t, view="diagnostics"), LOWER, {}),
            (f"room-no-viewer/{t}", "room", room, {"session_id": SESSION, "transport": t},
             LOWER, {}),
        ]
    out += [
        ("room-revision", "room", f"{room}/revision", {"session_id": SESSION, "viewer": V},
         DATA, {}),
        ("room-with-areas-revision", "areas", f"{room}/revision",
         {"session_id": SESSION, "viewer": V}, DATA, {}),
        ("area-revision", "areas", f"{area}/revision", {}, DATA, {}),
        ("research-revision", "research", f"{room}/revision",
         {"session_id": SESSION, "viewer": V}, DATA, {RAW_ENV: "1"}),
    ]
    return out


def data_requests() -> list:
    """The appearance data routes. Not in the golden -- a manifest carries its
    build's own timings and parameter digest -- but compared in-suite, with and
    without `wb-chrome`, against the same built world."""
    return [
        ("room-manifest", "room", f"/worlds/{WORLD}/appearance/{SESSION}/manifest", {}, DATA, {}),
        ("area-manifest", "areas", f"/worlds/{WORLD}/areas/{SESSION}/{AREA}/appearance/manifest",
         {}, DATA, {}),
        ("research-manifest", "research", f"/worlds/{WORLD}/appearance/{SESSION}/manifest", {},
         DATA, {RAW_ENV: "1"}),
    ]


def with_chrome(params: dict, chrome) -> dict:
    return dict(params) if chrome is None else {**params, "wb-chrome": chrome}


def chrome_key(key: str, chrome) -> str:
    return f"{key}?wb-chrome={'<absent>' if chrome is None else repr(chrome)}"


_VOLATILE_KEYS = ("build_id", "epoch", "built_at", "surface_built_at", "file")


def _volatile_values(node, found: set) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _VOLATILE_KEYS:
                if isinstance(value, str) and len(value) >= 8:
                    found.add(value)
                elif (isinstance(value, (int, float)) and not isinstance(value, bool)
                      and value > 1e9):          # a wall-clock stamp
                    found.add(repr(value))
                    found.add(json.dumps(value))
            _volatile_values(value, found)
    elif isinstance(node, list):
        for value in node:
            _volatile_values(value, found)


def volatile(roots: dict) -> list:
    """The per-build strings to mask, read from what the builds wrote: in every
    manifest, each `build_id`, `epoch`, `built_at`, `surface_built_at` and level
    `file` (a surface mesh is named after its build's clock). Longest first."""
    found: set = set()
    for root in roots.values():
        for path in sorted(Path(root).rglob("manifest.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            _volatile_values(data, found)
    return sorted(found, key=lambda s: (-len(s), s))


VOLATILE = b"<VOLATILE>"


def mask(body: bytes, tokens: list) -> bytes:
    for token in tokens:
        body = body.replace(token.encode("utf-8"), VOLATILE)
    return body


def _header(name: str, value: str, raw: bytes) -> list:
    # `content-length` is the served body's length, which a masked stamp changes
    # (a float's repr is not fixed-width): kept as "the body's own length" when it
    # is exactly that, and verbatim -- so a mismatch shows -- when it is not.
    if name == "content-length" and value == str(len(raw)):
        return [name, "<len(body)>"]
    return [name, value]


def describe(response, tokens: list, body: bytes | None = None) -> dict:
    """What the golden keeps of one response: status, every header (lower-cased,
    sorted), and the masked body's length and SHA-256."""
    raw = response.content if body is None else body
    masked = mask(raw, tokens)
    return {
        "status": response.status_code,
        "headers": sorted(_header(k.lower(), v, raw) for k, v in response.headers.items()),
        "length": len(masked),
        "sha256": hashlib.sha256(masked).hexdigest(),
    }


class Session:
    """One TestClient per fixture world, and the request runner."""

    def __init__(self, roots: dict):
        from fastapi.testclient import TestClient

        self.roots = roots
        self.clients = {name: TestClient(_app(root)) for name, root in roots.items()}
        self.tokens = volatile(roots)

    def get(self, world: str, path: str, params: dict, env: dict, switch):
        saved = {k: os.environ.get(k) for k in (SWITCH_ENV, RAW_ENV)}
        try:
            for k in (SWITCH_ENV, RAW_ENV):
                os.environ.pop(k, None)
            if switch is not None:
                os.environ[SWITCH_ENV] = switch
            os.environ.update(env)
            return self.clients[world].get(path, params=params)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def golden_outputs(tmp: Path, switch=None) -> dict:
    """Every response of the matrix, described, with the switch set to `switch`."""
    session = Session(build_worlds(tmp))
    out = {}
    for key, world, path, params, _kind, env in requests():
        for chrome in CHROME_VALUES:
            r = session.get(world, path, with_chrome(params, chrome), env, switch)
            out[chrome_key(key, chrome)] = describe(r, session.tokens)
    return out


def record(target: Path) -> None:
    import tempfile

    import fastapi
    import numpy

    # what tests/conftest.py's autouse fixture does for every test: the transient
    # detector would otherwise load its checkpoints (~2 GB) for the research build
    os.environ["TOWER_WORLD_TRANSIENTS"] = "off"
    with tempfile.TemporaryDirectory(prefix="wbnc-golden-") as tmp:
        outputs = golden_outputs(Path(tmp))
    target.write_text(json.dumps({
        "recorded_by": "tests/wb_native_chrome_fixtures.py",
        "numpy": numpy.__version__,
        "fastapi": fastapi.__version__,
        "outputs": outputs,
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    record(Path(sys.argv[1]))
