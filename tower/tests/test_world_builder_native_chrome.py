"""U1.1 native chrome, the Tower side: the switch, the five insertions, the bridge.

Contract: docs/contracts/WORLD-BUILDER-WORLDS.md §4c (v5 draft), COMPONENTS
§5.1/§5.4 (v12 draft). Spec: RUN/lead/specs/U11-NATIVE-CHROME-SPEC-20261004.md §2.

NO BIG BANG. `TOWER_WORLD_NATIVE_CHROME` is off by default, and off -- and every
request that is not exactly `wb-chrome=native` on an appearance page -- is held
to dc35d51's bytes by `golden/world_builder_native_chrome_dc35d51.json`, recorded
on dc35d51 itself by `tests/wb_native_chrome_fixtures.py` (see its docstring).
That golden is PERMANENT: re-record it only on a base commit whose page output
legitimately changed, never from a tree whose chrome handling is under review.

The tests are numbered as the spec numbers them (N1-N16). Node tests honour
`WB_NODE_REQUIRED=1`; the browser test (N14) skips without Chrome unless
`WB_CHROME_BROWSER_REQUIRED=1`.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import os
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tests import wb_native_chrome_fixtures as F

GOLDEN = Path(__file__).parent / "golden" / "world_builder_native_chrome_dc35d51.json"
BROWSER_REQUIRED_ENV = "WB_CHROME_BROWSER_REQUIRED"
WORLDS_CONTRACT = (Path(__file__).resolve().parents[2] / "docs" / "contracts"
                   / "WORLD-BUILDER-WORLDS.md")


def _nc():
    from tower.world_builder import native_chrome

    return native_chrome


def _template():
    from tower.world_builder.appearance_render import viewer_template_path

    return viewer_template_path().read_text(encoding="utf-8")


def _bridge_source():
    return _nc().bridge_path().read_text(encoding="utf-8")


def _meta(html, name):
    m = re.search(rf'<meta name="{name}" content="([^"]*)">', html[:4096])
    return m.group(1) if m else None


def _golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["outputs"]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """The fixture worlds, built once for the module (a module-scoped fixture runs
    before conftest's per-test autouse ones, so the transient detector is switched
    off here too: it would load ~2 GB of checkpoints for the research build)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TOWER_WORLD_TRANSIENTS", "off")
        roots = F.build_worlds(tmp_path_factory.mktemp("native-chrome"))
    return F.Session(roots)


def _reverse(native: bytes) -> bytes:
    """The native body with each of the five inserted texts deleted once -- each
    must occur exactly once."""
    text = native.decode("utf-8")
    for name, inserted in _nc().INSERTED.items():
        assert text.count(inserted) == 1, f"insertion {name} occurs {text.count(inserted)} times"
        text = text.replace(inserted, "", 1)
    return text.encode("utf-8")


def _same_but_length(a: dict, b: dict) -> bool:
    strip = lambda d: {**d, "headers": [h for h in d["headers"] if h[0] != "content-length"]}  # noqa: E731
    return strip(a) == strip(b)


def _matrix(session, switch, *, native_on: bool):
    """Every request of the golden matrix at `switch`: each response against the
    golden, and against the same request without `wb-chrome`, in-suite."""
    golden = _golden()
    seen = set()
    changed = []
    for key, world, path, params, kind, env in F.requests():
        plain = session.get(world, path, params, env, switch)
        for chrome in F.CHROME_VALUES:
            k = F.chrome_key(key, chrome)
            seen.add(k)
            r = session.get(world, path, F.with_chrome(params, chrome), env, switch)
            if native_on and chrome == "native" and kind == F.APPEARANCE:
                assert r.content != plain.content, k
                web = _reverse(r.content)
                assert web == plain.content, k
                assert _same_but_length(F.describe(r, session.tokens, web), golden[k]), k
                assert sorted((h, v) for h, v in r.headers.items() if h != "content-length") \
                    == sorted((h, v) for h, v in plain.headers.items() if h != "content-length"), k
                changed.append(k)
                continue
            assert F.describe(r, session.tokens) == golden[k], k
            assert r.content == plain.content, k
            assert sorted(r.headers.items()) == sorted(plain.headers.items()), k
    assert seen == set(golden), "the golden and the matrix name the same requests"
    for key, world, path, params, _kind, env in F.data_requests():
        plain = session.get(world, path, params, env, switch)
        assert plain.status_code == 200, key
        for chrome in F.CHROME_VALUES:
            r = session.get(world, path, F.with_chrome(params, chrome), env, switch)
            assert (r.status_code, r.content, sorted(r.headers.items())) == \
                (plain.status_code, plain.content, sorted(plain.headers.items())), (key, chrome)
    return changed


# ---------------------------------------------------------------------------
# N1 -- the switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,on", [
    (None, False), ("", False), ("  ", False), ("0", False), ("false", False), ("no", False),
    ("off", False), ("native", False), ("2", False),
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True), (" On ", True),
])
def test_the_switch_is_off_unless_set(monkeypatch, value, on):
    from tower import config
    from tower.results.world_builder_render import wants_native_chrome

    assert config.WORLD_NATIVE_CHROME_ENV == F.SWITCH_ENV == "TOWER_WORLD_NATIVE_CHROME"
    if value is None:
        monkeypatch.delenv(F.SWITCH_ENV, raising=False)
    else:
        monkeypatch.setenv(F.SWITCH_ENV, value)
    assert config.world_native_chrome_setting() is on
    # exactly `native`, and only with the switch on
    assert wants_native_chrome("native") is on
    for other in (None, "", "web", "NATIVE", "Native", " native", "native ", "x"):
        assert wants_native_chrome(other) is False, other


# ---------------------------------------------------------------------------
# N2 / N3 / N7 -- §2.7 layer 1, against dc35d51's golden
# ---------------------------------------------------------------------------


def test_the_golden_was_recorded_by_the_fixtures_on_the_base():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["recorded_by"] == "tests/wb_native_chrome_fixtures.py"
    out = data["outputs"]
    # it pins what it claims: appearance pages, lower rungs and revisions, every
    # chrome value, every transport -- and no page in it carries the echo
    assert len(out) == len(F.requests()) * len(F.CHROME_VALUES)
    assert all(v["status"] == 200 for v in out.values())
    pages = [k for k in out if "/app?" in k or "/tower?" in k]
    assert len(pages) == 2 * 12 * len(F.CHROME_VALUES)


@pytest.mark.parametrize("switch", [None, "0"])
def test_off_every_chrome_request_is_byte_identical(session, switch):
    assert _matrix(session, switch, native_on=False) == []


def test_on_only_exactly_native_changes_the_page(session):
    changed = _matrix(session, "1", native_on=True)
    # the appearance pages, and nothing else: 6 kinds x 2 transports
    assert len(changed) == 12
    assert all(k.endswith("?wb-chrome='native'") for k in changed)


def test_lower_rungs_ignore_the_chrome_request(session):
    """LOW-7: pinned surface, a session with no appearance, the diagnostics view,
    a client without `appearance-1` -- `native` with the switch on is the page
    without it, and none carries the echo."""
    lower = [r for r in F.requests() if r[4] == F.LOWER]
    assert len(lower) == 12
    for key, world, path, params, _kind, env in lower:
        plain = session.get(world, path, params, env, "1")
        native = session.get(world, path, F.with_chrome(params, "native"), env, "1")
        assert plain.status_code == 200 and native.content == plain.content, key
        assert _meta(native.text, "wb-representation") != "appearance", key
        assert _nc().ECHO not in native.text, key


# ---------------------------------------------------------------------------
# N4 / N5 / N8 / N9 -- the native page
# ---------------------------------------------------------------------------


def _appearance_requests():
    return [r for r in F.requests() if r[4] == F.APPEARANCE]


def test_native_is_the_web_page_plus_exactly_five_insertions(session):
    nc = _nc()
    names = [name for name, _a, _r in nc.INSERTIONS]
    assert names == ["E", "C", "K", "F", "B"] and set(nc.INSERTED) == set(names)
    for name, anchor, replacement in nc.INSERTIONS:
        # deleting the inserted text from the replacement gives the anchor back
        assert replacement.count(anchor) <= 1
        assert replacement.replace(nc.INSERTED[name], "", 1) == anchor, name
    for key, world, path, params, _kind, env in _appearance_requests():
        web = session.get(world, path, params, env, "1")
        native = session.get(world, path, F.with_chrome(params, "native"), env, "1")
        assert web.status_code == native.status_code == 200, key
        assert _reverse(native.content) == web.content, key
        assert len(native.content) - len(web.content) == sum(
            len(t.encode("utf-8")) for t in nc.INSERTED.values()), key


def test_the_echo_is_in_the_first_4096_characters_once(session):
    nc = _nc()
    for key, world, path, params, _kind, env in _appearance_requests():
        page = session.get(world, path, F.with_chrome(params, "native"), env, "1").text
        assert page.count(nc.ECHO) == 1, key
        at = page.index(nc.ECHO)
        assert at + len(nc.ECHO) <= 4096 and _meta(page, "wb-chrome") == "native", key
        # representation, revision, (area), chrome
        head = page[:at]
        rep = head.index('<meta name="wb-representation" content="appearance">')
        rev = head.index('<meta name="wb-revision" ')
        assert rep < rev, key
        if key.startswith("area"):
            assert rev < head.index('<meta name="wb-area" '), key
        web = session.get(world, path, params, env, "1").text
        assert "wb-chrome" not in web and _meta(web, "wb-chrome") is None, key


def test_one_revision_for_both_variants(session):
    for page_key, rev_key in (("room/app", "room-revision"), ("area/app", "area-revision"),
                              ("room-with-areas/app", "room-with-areas-revision"),
                              ("research/app", "research-revision")):
        _k, world, path, params, _kind, env = next(r for r in F.requests() if r[0] == page_key)
        _k, rworld, rpath, rparams, _kind, renv = next(r for r in F.requests() if r[0] == rev_key)
        web = session.get(world, path, params, env, "1").text
        native = session.get(world, path, F.with_chrome(params, "native"), env, "1").text
        rev = session.get(rworld, rpath, rparams, renv, "1")
        rev_native = session.get(rworld, rpath, F.with_chrome(rparams, "native"), renv, "1")
        assert rev.status_code == 200 and rev.content == rev_native.content, rev_key
        assert _meta(web, "wb-revision") == _meta(native, "wb-revision") \
            == rev.json()["revision"] is not None, page_key


def test_the_headers_are_identical_across_variants(session):
    for key, world, path, params, _kind, env in _appearance_requests():
        web = session.get(world, path, params, env, "1")
        native = session.get(world, path, F.with_chrome(params, "native"), env, "1")
        for header in ("cache-control", "content-security-policy", "content-type"):
            assert web.headers[header] == native.headers[header], (key, header)
        meta = r'<meta http-equiv="Content-Security-Policy" content="([^"]*)">'
        assert re.search(meta, web.text[:4096]).group(1) \
            == re.search(meta, native.text[:4096]).group(1) == web.headers["content-security-policy"]


def test_the_template_carries_every_anchor_once_and_composes():
    nc = _nc()
    template = _template()
    for name, anchor, _r in nc.INSERTIONS:
        assert template.count(anchor) == 1, name
    composed = nc.compose(template)
    assert composed is not None and composed.count(nc.ECHO) == 1
    # K is inside main(), after the last name it reads; F ends frame()
    main = composed.index("async function main(){")
    k = composed.index(nc.HOOKS)
    assert main < k < composed.index("main().catch(")
    f = composed.index(nc.FRAME_CALL)
    assert composed[f + len(nc.FRAME_CALL):].startswith("\n  }\n")


# ---------------------------------------------------------------------------
# N6 -- all or nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["E", "C", "K", "F", "B"])
def test_an_anchor_miss_costs_the_echo_and_nothing_else(session, monkeypatch, caplog, missing):
    nc = _nc()
    patched = tuple((n, "\x00no such anchor\x00" if n == missing else a, r)
                    for n, a, r in nc.INSERTIONS)
    monkeypatch.setattr(nc, "INSERTIONS", patched)
    for key, world, path, params, _kind, env in _appearance_requests():
        if not key.endswith("/app"):
            continue
        web = session.get(world, path, params, env, "1")
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="tower.world_builder.native_chrome"):
            native = session.get(world, path, F.with_chrome(params, "native"), env, "1")
        assert native.content == web.content and nc.ECHO not in native.text, (key, missing)
        assert sorted(native.headers.items()) == sorted(web.headers.items()), key
        errors = [r for r in caplog.records
                  if r.name == "tower.world_builder.native_chrome" and r.levelno == logging.ERROR]
        assert len(errors) == 1, (key, [r.getMessage() for r in errors])
        assert f"anchor {missing} occurs 0 times" in errors[0].getMessage()


@pytest.mark.parametrize("repeated", ["E", "C", "K", "F", "B"])
def test_a_repeated_anchor_composes_nothing(caplog, repeated):
    nc = _nc()
    anchor = dict((n, a) for n, a, _r in nc.INSERTIONS)[repeated]
    page = _template().replace(anchor, anchor + "\n" + anchor, 1)
    with caplog.at_level(logging.ERROR, logger="tower.world_builder.native_chrome"):
        assert nc.compose(page) is None
    assert any(f"anchor {repeated} occurs 2 times" in r.getMessage() for r in caplog.records)


def test_a_chrome_module_that_fails_serves_the_web_page(session, monkeypatch):
    """Not a lower rung: the appearance page, as the web serves it."""
    nc = _nc()

    def boom(page):
        raise RuntimeError("chrome bug")

    monkeypatch.setattr(nc, "compose", boom)
    _k, world, path, params, _kind, env = next(r for r in F.requests() if r[0] == "room/app")
    web = session.get(world, path, params, env, "1")
    native = session.get(world, path, F.with_chrome(params, "native"), env, "1")
    assert native.status_code == 200 and native.content == web.content


def test_the_bridge_is_never_able_to_end_its_script_early(tmp_path, monkeypatch):
    nc = _nc()
    assert "</script" not in _bridge_source().lower()
    bad = tmp_path / nc.BRIDGE_NAME
    bad.write_text("const x = '</script>';", encoding="utf-8")
    monkeypatch.setattr(nc, "bridge_path", lambda: bad)
    with pytest.raises(ValueError):
        nc._bridge_block()


# ---------------------------------------------------------------------------
# N10 / N11 -- the inventory and the literals
# ---------------------------------------------------------------------------


class _BodyChildren(HTMLParser):
    VOID = {"meta", "br", "img", "input", "link", "hr"}

    def __init__(self):
        super().__init__()
        self.depth = None
        self.children = []

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.depth = 0
            return
        if self.depth is None:
            return
        if self.depth == 0:
            self.children.append((tag, dict(attrs).get("id")))
        if tag not in self.VOID:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag == "body":
            self.depth = None
        elif self.depth:
            self.depth -= 1


def test_every_overlay_is_in_the_native_inventory():
    parser = _BodyChildren()
    parser.feed(_template())
    kids = [(t, i) for t, i in parser.children if t != "script"]
    assert ("div", "wrap") in kids
    assert tuple(i for _t, i in kids if i != "wrap") == _nc().INVENTORY
    # the CSS block hides exactly the inventory
    hidden = re.findall(r"body\.wbnative #(\w+)", _nc().CSS)
    assert tuple(hidden) == _nc().INVENTORY
    assert _nc().CSS.rstrip().endswith("{display:none!important}")


def test_the_bridge_literals_match_the_template():
    template = _template()
    bridge = _bridge_source()
    assert template.count('g.fillText("YOU", c, c);') == 1
    assert template.count('toggle.textContent = captionOpen ? "Less" : "About";') == 1
    assert bridge.count('const RING_CENTER = "YOU";') == 1
    assert bridge.count('const ABOUT_OPEN = "About", ABOUT_CLOSE = "Less";') == 1
    main = template[template.index("async function main(){"):template.index("main().catch(")]
    for declaration in ("let shownAt = null;", "let restoring = false;",
                        "let ringProf = null, ringAt = null, ringFor = null;",
                        "function yawSign(){", "function shownPose(){",
                        "function currentCamera(aspectOverride){",
                        'const HINT_EDGE = "Movement stops here";',
                        'const HINT_DARK = "Not reconstructed from here";',
                        'const HINT_DARK_TAP = "Tap to turn back";',
                        "NAV.smooth(0.08, NAV.T_HI, ringProf[i])"):
        assert main.count(declaration) == 1, declaration
    # the hooks read exactly what drawCompass draws with
    hooks = _nc().HOOKS
    for name in ("shownAt", "restoring", "ringProf", "yawSign()", "shownPose().yaw",
                 "currentCamera()", "HINT_DARK", "HINT_DARK_TAP", "HINT_EDGE",
                 "NAV.smooth(0.08, NAV.T_HI, s)"):
        assert name in hooks, name
    # the page has no name the bridge declares at its top level
    for name in ("WBCHROME",):
        assert name not in template


# ---------------------------------------------------------------------------
# node: N12, N13, N15, N16
# ---------------------------------------------------------------------------


def _node():
    from tests.wb_node import node_or_skip

    return node_or_skip("run the native-chrome bridge")


def _run_node(program: str, tmp_path: Path, tag: str) -> dict:
    script = tmp_path / f"{tag}.js"
    script.write_text(program, encoding="utf-8")
    r = subprocess.run([_node(), str(script)], capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


class _Elements(HTMLParser):
    """Every element of a page's <body> with an id: tag, attributes, its direct
    text nodes, and its text -- what the stub DOM is built from."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside = False
        self.stack = []
        self.found = {}

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.inside = True
            return
        if not self.inside or tag == "script":
            if tag == "script":
                self.inside = False
            return
        a = dict(attrs)
        el = {"tag": tag, "attrs": {k: (v if v is not None else "") for k, v in a.items()},
              "texts": [], "parent": self.stack[-1]["attrs"].get("id") if self.stack else None}
        if "id" in a:
            self.found[a["id"]] = el
        if tag not in ("br", "meta", "img", "input"):
            self.stack.append(el)

    def handle_endtag(self, tag):
        if self.inside and self.stack and tag == self.stack[-1]["tag"]:
            self.stack.pop()

    def handle_data(self, data):
        if self.inside and self.stack and data.strip():
            self.stack[-1]["texts"].append(data.strip())


def _elements(page: str) -> dict:
    p = _Elements()
    p.feed(page)
    return p.found


_STUB = r"""
"use strict";
const assert = require("assert");
let moCallbacks = [];
const MUT = [];
let queued = false;
function notify(target){
  MUT.push(target);
  if (queued) return;
  queued = true;
  queueMicrotask(() => { queued = false; const recs = MUT.splice(0).map(t => ({target: t}));
    moCallbacks.slice().forEach(cb => cb(recs)); });
}
class MutationObserver {
  constructor(cb){ this.cb = cb; }
  observe(target, opts){ this.opts = opts; moCallbacks.push(this.cb); }
  disconnect(){ moCallbacks = moCallbacks.filter(c => c !== this.cb); }
}
function mk(tag, id){
  const el = {tagName: tag.toUpperCase(), id: id || "", parent: null, kids: [], texts: [], own: "",
    attrs: {}, cls: new Set(), _disabled: false, _hidden: false, clicks: 0, style: {}, onclick: null,
    get children(){ return this.kids; },
    get childNodes(){ return this.texts.map(t => ({nodeType: 3, textContent: t})); },
    get textContent(){ return this.own + this.kids.map(k => k.textContent).join(""); },
    set textContent(v){ this.own = String(v); this.kids = []; this.texts = String(v) ? [String(v)] : [];
      notify(this); },
    get className(){ return [...this.cls].join(" "); },
    set className(v){ this.cls = new Set(String(v).split(/\s+/).filter(Boolean)); notify(this); },
    get disabled(){ return this._disabled; }, set disabled(v){ this._disabled = !!v; notify(this); },
    get hidden(){ return this._hidden; }, set hidden(v){ this._hidden = !!v; notify(this); },
    getAttribute(k){ return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; },
    setAttribute(k, v){ this.attrs[k] = String(v); notify(this); },
    append(...c){ for (const x of c){ x.parent = this; this.kids.push(x); } notify(this); },
    contains(n){ for (let x = n; x; x = x.parent) if (x === this) return true; return false; },
    click(){ if (!this._disabled){ this.clicks++; if (this.onclick) this.onclick(); } }};
  el.classList = {add: c => { el.cls.add(c); notify(el); }, remove: c => { el.cls.delete(c); notify(el); },
                  contains: c => el.cls.has(c)};
  return el;
}
const EL = {};
const body = mk("body", "");
for (const [id, spec] of Object.entries(MARKUP)){
  const el = mk(spec.tag, id);
  for (const [k, v] of Object.entries(spec.attrs)){
    if (k === "class") el.cls = new Set(v.split(/\s+/).filter(Boolean));
    else if (k === "hidden") el._hidden = true;
    else if (k === "disabled") el._disabled = true;
    else el.attrs[k] = v;
  }
  el.texts = spec.texts.slice(); el.own = spec.texts.join("");
  EL[id] = el;
}
for (const [id, spec] of Object.entries(MARKUP)){
  const parent = spec.parent ? EL[spec.parent] : body;
  EL[id].parent = parent; parent.kids.push(EL[id]);
}
const document = {body, getElementById: id => EL[id] || null, createElement: tag => mk(tag)};
const settle = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(r => setTimeout(r, 0)); };
const ascii = s => s.replace(/[^\x00-\x7e]/g, c => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
"""

_HANDLER = r"""
const posts = [];
const pending = {state: [], view: [], await: []};
let helloReply = {v: 1, type: "welcome", protocol: 1, nonce: "nonce-1"};
const handler = {postMessage(msg){
  const m = JSON.parse(JSON.stringify(msg));
  posts.push(m);
  if (m.type === "hello") return Promise.resolve(helloReply);
  return new Promise((res, rej) => pending[m.type].push({res, rej, msg: m}));
}};
const answer = (type, value) => { const p = pending[type].shift(); assert(p, "nothing pending: " + type);
  p.res(value === undefined ? null : value); };
const of = type => posts.filter(p => p.type === type);
const lastOf = type => { const a = of(type); return a[a.length - 1]; };
const listeners = {};
"""


def _load_bridge(*, handler: bool, config: dict, raw: bool, s_obj: str) -> str:
    """JS that runs the bridge file verbatim, as the page would, over the stub."""
    win = ("{webkit: {messageHandlers: {wbChromeV1: handler}}, __wbAppearance: S, "
           "addEventListener: (t, f) => { listeners[t] = f; }}" if handler
           else "{__wbAppearance: S, addEventListener: (t, f) => { listeners[t] = f; }}")
    return (f"const S = {s_obj};\n"
            f"const window = {win};\n"
            f"const WB = (new Function('window', 'document', 'CONFIG', 'RAW_IMAGERY', "
            f"'MutationObserver', 'crypto', {json.dumps(_bridge_source() + chr(10) + 'return WBCHROME;')}))"
            f"(window, document, {json.dumps(config)}, {json.dumps(raw)}, MutationObserver, "
            f"globalThis.crypto);\n")


_S_OBJ = r"""{phase: "loading", holding: false,
  chromeView: () => VIEW,
  chromeWords: () => ({dark: "Not reconstructed from here", darkTap: "Tap to turn back",
                       edge: "Movement stops here"})}"""

_VIEW = "let VIEW = {drawn: false, restoring: false, lit: null, sense: 1, headingRad: 0.5, halfFovRad: 0.6};\n"


def _program(page: str, body: str, *, handler=True, config=None, raw=False, s_obj=_S_OBJ,
             prelude: str = "") -> str:
    """`body` runs after the bridge has loaded over the stub DOM built from `page`'s
    own markup; `prelude` runs before it loads (the bridge posts `hello` at once)."""
    markup = {k: v for k, v in _elements(page).items()}
    return ("const MARKUP = " + json.dumps(markup) + ";\n" + _STUB + _HANDLER + _VIEW
            + prelude + "\n"
            + _load_bridge(handler=handler, config=config or {}, raw=raw, s_obj=s_obj)
            + "(async () => {\n" + body + "\n})().then(() => process.exit(0), e => { "
            "console.error(e && e.stack || e); process.exit(1); });\n")


@pytest.fixture(scope="module")
def pages(session):
    """The served native pages, by kind."""
    out = {}
    for key in ("room/app", "area/app", "research/app", "room-with-areas/app"):
        _k, world, path, params, _kind, env = next(r for r in F.requests() if r[0] == key)
        r = session.get(world, path, F.with_chrome(params, "native"), env, "1")
        assert _nc().ECHO in r.text
        out[key.split("/")[0]] = r.text
    return out


def test_the_bridge_script_parses_and_posts_nothing_without_a_handler(pages, tmp_path):
    node = _node()
    path = tmp_path / "bridge.js"
    path.write_text(_bridge_source(), encoding="utf-8")
    r = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = _run_node(_program(pages["room"], r"""
await settle();
console.log(JSON.stringify({posts: posts.length, bridge: S.chromeBridge(),
  frame: typeof S.chromeFrame, mo: moCallbacks.length, wbnative: body.classList.contains("wbnative")}));
""", handler=False), tmp_path, "nohandler")
    assert out == {"posts": 0, "bridge": {"handler": False}, "frame": "undefined", "mo": 0,
                   "wbnative": False}
    # a handler without replies (an app that registered a plain one): one hello, then nothing
    out = _run_node(_program(pages["room"], r"""
await settle();
console.log(JSON.stringify({types: posts.map(p => p.type), bridge: S.chromeBridge(),
  frame: typeof S.chromeFrame}));
""", prelude="handler.postMessage = msg => { posts.push(msg); return undefined; };"),
        tmp_path, "noreply")
    assert out["types"] == ["hello"] and out["frame"] == "undefined"
    assert out["bridge"]["dead"] is True and out["bridge"]["welcomed"] is False


LABELS_ROOM = {
    "best": {"text": "Best view", "name": "Best view: fly to the clearest vantage"},
    "face": {"text": "Face the room",
             "name": "Face the room: turn to the nearest reconstructed direction"},
    "previous": {"text": "\u2190", "name": "Previous recorded view"},
    "next": {"text": "\u2192", "name": "Next recorded view"},
    "reset": {"text": "Reset", "name": "Reset: return to the opening view"},
    "ring": {"name": "Which directions are reconstructed from here",
             "label": "reconstructed\nfrom here", "center": "YOU"},
    "about": {"open": "About", "close": "Less"},
    "dark": {"title": "Not reconstructed from here", "tap": "Tap to turn back",
             "name": "Not reconstructed from here. Tap to turn back"},
    "edge": "Movement stops here",
}


def _labels_area():
    out = json.loads(json.dumps(LABELS_ROOM))
    out["face"] = {"text": "Face the area",
                   "name": "Face the area: turn to the nearest reconstructed direction"}
    return out


_LIFECYCLE = r"""
const LIM = WB.LIMITS;
await settle();
const hello = posts[0];
assert.strictEqual(hello.type, "hello"); assert.strictEqual(hello.seq, 0); assert.strictEqual(hello.v, 1);
assert(/^[a-z0-9]{16}$/.test(hello.pageId), hello.pageId);
assert(!("nonce" in hello));
// welcome -> a first state, and one await outstanding
assert.strictEqual(of("state").length, 1, "one state");
assert.strictEqual(of("await").length, 1, "one await");
assert.strictEqual(typeof S.chromeFrame, "function");
const first = of("state")[0];
for (const m of posts.slice(1)){ assert.strictEqual(m.nonce, "nonce-1"); assert.strictEqual(m.pageId, hello.pageId); }
posts.forEach((m, i) => assert.strictEqual(m.seq, i, "seq rises by one"));
const steps = {};
async function change(name, f){
  f(); await settle();
  // one state in flight: answer the one pending, then read what the change posted
  while (pending.state.length){ answer("state"); await settle(); }
  steps[name] = lastOf("state");
}
await change("status", () => { EL.status.textContent = "Placing images 3 / 8"; });
await change("dark", () => { EL.dark.textContent = ""; EL.dark.classList.add("on"); });
await change("edge", () => { EL.back.className = "on left"; });
await change("hint", () => { EL.hint.textContent = "One moment \u2014 you can look around as soon as it draws";
  EL.hint.style.opacity = "0.9"; EL.hint.classList.add("on"); });
await change("disabled", () => { EL.bPrev.disabled = true; });
await change("pos", () => { EL.pos.textContent = "3 / 8"; });
await change("msg", () => { EL.msg.textContent = "The graphics context was taken away"; EL.msg.classList.add("on"); });
await change("long", () => { EL.status.textContent = "x".repeat(500); });
// an unchanged state is not posted again
const before = of("state").length;
EL.status.textContent = "x".repeat(500); await settle();
const unchanged = of("state").length - before;
// at most one state in flight: three changes while one is out -> one more, the latest
EL.status.textContent = "a"; await settle();
const inflight1 = pending.state.length;
EL.status.textContent = "b"; await settle(); EL.status.textContent = "c"; await settle();
const inflight2 = pending.state.length;
const postedBefore = of("state").length;
answer("state"); await settle();
const coalesced = {posted: of("state").length - postedBefore, status: lastOf("state").status};
answer("state"); await settle();
// view: per frame, more than 1e-3, one in flight, the latest delivered
VIEW = {...VIEW, drawn: true, headingRad: 1.0};
S.chromeFrame(); await settle();
const v1 = of("view").length;
VIEW = {...VIEW, headingRad: 1.0005}; S.chromeFrame(); await settle();
const vSmall = of("view").length - v1;
VIEW = {...VIEW, headingRad: 1.2}; S.chromeFrame(); await settle();
VIEW = {...VIEW, headingRad: 1.4}; S.chromeFrame(); await settle();
const vInFlight = of("view").length - v1;
answer("view"); await settle();
const vAfter = {n: of("view").length - v1, heading: lastOf("view").headingRad};
answer("view"); await settle();
while (pending.state.length){ answer("state"); await settle(); }
const drawnState = lastOf("state");
// the ring: a new profile (and its sense) is a new state, from the frame hook alone
VIEW = {...VIEW, lit: Array.from({length: 36}, (_, i) => i / 35), sense: -1};
EL.compass.cls.add("on");                      // no mutation: the frame must carry it
S.chromeFrame(); await settle();
while (pending.state.length){ answer("state"); await settle(); }
const ringState = lastOf("state").ring;
// a field the page changes with no DOM change and no frame: the sweep posts it
S.holding = true;
await new Promise(r => setTimeout(r, 1200));
while (pending.state.length){ answer("state"); await settle(); }
const holdingState = lastOf("state").holding;
// commands
answer("await", {v: 1, type: "activate"}); await settle();
const activated = body.classList.contains("wbnative");
while (pending.state.length){ answer("state"); await settle(); }
const activeState = lastOf("state").active;
const awaits1 = of("await").length;
answer("await", {v: 1, type: "action", name: "best"}); await settle();
answer("await", {v: 1, type: "action", name: "previous"}); await settle();   // disabled
answer("await", {v: 1, type: "action", name: "toString"}); await settle();
const clicks = {best: EL.bOverview.clicks, previous: EL.bPrev.clicks, face: EL.bBack.clicks,
  next: EL.bNext.clicks, reset: EL.bReset.clicks};
answer("await", {v: 1, type: "frobnicate"}); await new Promise(r => setTimeout(r, 400));
const awaits2 = of("await").length;
while (pending.state.length){ answer("state"); await settle(); }
answer("await", {v: 1, type: "deactivate"}); await settle();
while (pending.state.length){ answer("state"); await settle(); }
const deactivated = {wbnative: body.classList.contains("wbnative"), active: lastOf("state").active,
  awaitsPending: pending.await.length, frame: typeof S.chromeFrame, mo: moCallbacks.length,
  bridge: S.chromeBridge()};
const afterStop = posts.length;
EL.status.textContent = "after"; await settle();
const sizes = {};
for (const m of posts){ const n = Buffer.byteLength(JSON.stringify(m), "utf8");
  sizes[m.type] = Math.max(sizes[m.type] || 0, n); }
console.log(ascii(JSON.stringify({hello, first, steps, unchanged, inflight1, inflight2, coalesced,
  vSmall, vInFlight, vAfter, drawnState, ringState, holdingState, activated, activeState, clicks,
  awaitsAfterActions: awaits2 - awaits1, deactivated, postsAfterStop: posts.length - afterStop,
  sizes, LIM})));
"""


def _check_strings(state, lim):
    for field, bound in (("status", "status"), ("message", "message"), ("walk", "walk")):
        if state[field] is not None:
            assert len(state[field]) <= lim[bound], field
    if state["hint"]:
        assert len(state["hint"]["text"]) <= lim["hint"] and 0 <= state["hint"]["opacity"] <= 1
    cap = state["caption"]
    if cap:
        assert len(cap["head"]) <= lim["head"] and 1 <= len(cap["sections"]) <= lim["sections"]


@pytest.mark.parametrize("kind", ["room", "area"])
def test_the_bridge_under_a_stub_dom(pages, tmp_path, kind):
    config = {"area": {"id": F.AREA, "levelled": True}} if kind == "area" else {}
    out = _run_node(_program(pages[kind], _LIFECYCLE, config=config), tmp_path, f"life-{kind}")
    hello = out["hello"]
    assert hello["kind"] == kind and hello["protocol"] == [1]
    assert hello["labels"] == (_labels_area() if kind == "area" else LABELS_ROOM)
    assert hello["research"] == {"raw": False, "marker": None}
    first = out["first"]
    assert set(first) == {"v", "type", "pageId", "seq", "nonce", "active", "phase", "drawn",
                          "holding", "restoring", "status", "message", "hint", "dark", "edge",
                          "buttons", "walk", "ring", "caption", "research"}
    assert first["active"] is False and first["phase"] == "loading" and first["drawn"] is False
    assert first["buttons"] == dict.fromkeys(("best", "face", "previous", "next", "reset"), True)
    assert first["ring"] == {"shown": False, "lit": None, "sense": 1}
    assert first["status"] is None and first["message"] is None and first["hint"] is None
    assert first["dark"] is False and first["edge"] is None and first["walk"] is None
    s = out["steps"]
    assert s["status"]["status"] == "Placing images 3 / 8"
    assert s["dark"]["dark"] is True
    assert s["edge"]["edge"] == "left"
    assert s["hint"]["hint"] == {"text": "One moment \u2014 you can look around as soon as it draws",
                                 "opacity": 0.9}
    assert s["disabled"]["buttons"]["previous"] is False and s["disabled"]["buttons"]["next"] is True
    assert s["pos"]["walk"] == "3 / 8"
    assert s["msg"]["message"] == "The graphics context was taken away"
    assert s["long"]["status"] == "x" * out["LIM"]["status"]
    assert out["unchanged"] == 0
    assert out["inflight1"] == 1 and out["inflight2"] == 1
    assert out["coalesced"] == {"posted": 1, "status": "c"}
    # the first view went at once; a move under 1e-3 is not one; two moves while it
    # was in flight post nothing, and its reply posts the latest
    assert out["vSmall"] == 0 and out["vInFlight"] == 0
    assert out["vAfter"] == {"n": 1, "heading": 1.4}
    assert out["drawnState"]["drawn"] is True
    assert out["ringState"] == {"shown": True, "lit": [i / 35 for i in range(36)], "sense": -1}
    assert out["holdingState"] is True
    assert out["activated"] is True and out["activeState"] is True
    assert out["clicks"] == {"best": 1, "previous": 0, "face": 0, "next": 0, "reset": 0}
    assert out["awaitsAfterActions"] == 4
    d = out["deactivated"]
    assert d["wbnative"] is False and d["active"] is False and d["awaitsPending"] == 0
    assert d["frame"] == "undefined" and d["mo"] == 0 and d["bridge"]["dead"] is True
    assert out["postsAfterStop"] == 0
    for state in [out["first"], *out["steps"].values(), out["drawnState"]]:
        _check_strings(state, out["LIM"])
    sizes = out["sizes"]
    assert sizes["hello"] <= 32 * 1024 and sizes["state"] <= 24 * 1024
    assert sizes["view"] <= 512 and sizes["await"] <= 256


_DECLINES = r"""
await settle();
VIEW = {...VIEW, headingRad: 2}; if (S.chromeFrame) S.chromeFrame();
EL.status.textContent = "changed"; await settle();
console.log(JSON.stringify({types: posts.map(p => p.type), frame: typeof S.chromeFrame,
  wbnative: body.classList.contains("wbnative"), bridge: S.chromeBridge()}));
"""


@pytest.mark.parametrize("reply", [
    {"v": 1, "type": "decline"}, None, {"v": 1, "type": "welcome", "protocol": 2, "nonce": "n"},
    {"v": 1, "type": "welcome", "protocol": 1, "nonce": ""}, {"v": 2, "type": "welcome",
                                                              "protocol": 1, "nonce": "n"},
    {"v": 1, "type": "welcome", "protocol": 1},
])
def test_anything_but_a_welcome_stops_the_bridge_for_good(pages, tmp_path, reply):
    out = _run_node(_program(pages["room"], _DECLINES, prelude=f"helloReply = {json.dumps(reply)};"),
                    tmp_path, "decline")
    assert out["types"] == ["hello"] and out["frame"] == "undefined"
    assert out["wbnative"] is False and out["bridge"]["welcomed"] is False


_REJECTED = r"""
await settle();
answer("await", {v: 1, type: "activate"}); await settle();
while (pending.state.length){ answer("state"); await settle(); }
const active = body.classList.contains("wbnative");
pending.await.shift().rej(new Error("the app is gone")); await settle();
console.log(JSON.stringify({active, after: body.classList.contains("wbnative"),
  bridge: S.chromeBridge()}));
"""


_CLOSED = r"""
await settle();
answer("await", {v: 1, type: "activate"}); await settle();
while (pending.state.length){ answer("state"); await settle(); }
EL.status.textContent = "in flight"; await settle();          // one state out when close lands
answer("await", {v: 1, type: "close"}); await settle();
const n = posts.length;
answer("state"); await settle();
EL.status.textContent = "after close"; await settle();
if (S.chromeFrame) S.chromeFrame();
await new Promise(r => setTimeout(r, 1200));                   // past the sweep
console.log(JSON.stringify({after: posts.length - n, awaits: pending.await.length,
  frame: typeof S.chromeFrame, wbnative: body.classList.contains("wbnative"), bridge: S.chromeBridge()}));
"""


def test_close_stops_and_posts_nothing_more(pages, tmp_path):
    out = _run_node(_program(pages["room"], _CLOSED), tmp_path, "closed")
    # the phone is leaving: nothing more is posted, and the page is left as it is
    assert out["after"] == 0 and out["awaits"] == 0 and out["frame"] == "undefined"
    assert out["wbnative"] is True and out["bridge"]["dead"] is True


def test_a_failed_reply_puts_the_page_chrome_back(pages, tmp_path):
    out = _run_node(_program(pages["room"], _REJECTED), tmp_path, "rejected")
    assert out["active"] is True and out["after"] is False and out["bridge"]["dead"] is True


_MISSING_HOOKS = r"""
await settle();
console.log(JSON.stringify({hello: posts[0]}));
"""


def test_missing_hooks_leave_dark_and_edge_out_of_hello(pages, tmp_path):
    out = _run_node(_program(pages["room"], _MISSING_HOOKS, s_obj='{phase: "failed"}'),
                    tmp_path, "nohooks")
    labels = out["hello"]["labels"]
    assert "dark" not in labels and "edge" not in labels and labels["best"]["text"] == "Best view"


_RESEARCH_PRELUDE = r"""
EL.rawmark.textContent = "Research build \u2014 unredacted local capture \u2014 not privacy-safe";
EL.rawmark.hidden = false;
"""
_RESEARCH = r"""
await settle();
console.log(ascii(JSON.stringify({hello: posts[0], state: of("state")[0]})));
"""


def test_the_research_marker_reaches_the_phone(pages, tmp_path):
    out = _run_node(_program(pages["research"], _RESEARCH, raw=True, prelude=_RESEARCH_PRELUDE),
                    tmp_path, "research")
    marker = "Research build \u2014 unredacted local capture \u2014 not privacy-safe"
    assert out["hello"]["research"] == {"raw": True, "marker": marker}
    assert out["state"]["research"] == {"raw": True, "marker": marker}


def test_hello_labels_are_the_closed_name_table(pages, tmp_path):
    """N15: the seven names of WORLDS §4 *Controls*, from the contract itself."""
    import itertools

    if not WORLDS_CONTRACT.exists():
        pytest.skip("the WORLDS contract is not beside this Tower checkout")
    text = WORLDS_CONTRACT.read_text(encoding="utf-8")
    lines = [line.strip() for line in
             text[text.index("| Control | Shows | Accessible name |"):].splitlines()[2:]]
    rows = list(itertools.takewhile(lambda line: line.startswith("|"), lines))
    assert len(rows) == 7
    table = {}
    for row in rows:
        control, _shows, name = (c.strip() for c in row.strip().strip("|").split("|"))
        table[re.search(r"`#(\w+)`", control).group(1)] = re.search(r"\*([^*]+)\*", name).group(1)
    slot = {"bOverview": ("best", "name"), "bBack": ("face", "name"),
            "bPrev": ("previous", "name"), "bNext": ("next", "name"),
            "bReset": ("reset", "name"), "compass": ("ring", "name"), "dark": ("dark", "name")}
    assert set(slot) == set(table)
    for kind in ("room", "area"):
        config = {"area": {"id": F.AREA}} if kind == "area" else {}
        hello = _run_node(_program(pages[kind], "await settle(); console.log(JSON.stringify(posts[0]));",
                                   config=config), tmp_path, f"names-{kind}")
        for control, (key, field) in slot.items():
            want = table[control]
            if kind == "area" and control == "bBack":
                want = want.replace("Face the room", "Face the area")
            assert hello["labels"][key][field] == want, (kind, control)


# N16: the caption, as `updateCaption` builds it, through the bridge's reader

_CAPTION = r"""
const CAP = EL.caption;
const NAV = {MAX_SKIP: 4}; let captionOpen = false;
let manifest = {quality: "final"}; let encoding = "astc-6x6-rgba";
const $ = id => EL[id];
const RAW = RAW_FLAG;
(function(){ const RAW_IMAGERY = RAW; const CONFIG = {scale_state: "unknown", current: true};
  const S = {layers: 7, keyframesInManifest: 8, holding: false};
  UPDATE_CAPTION
  updateCaption();
})();
const got = WB.captionOf(CAP);
// what updateCaption built, read off the tree directly
const [head, toggle, more] = CAP.children;
const want = {head: head.textContent, line: null, titles: [], bodies: [], tail: null};
more.children.forEach((c, i) => {
  if (c.tagName === "I") want.line = c.textContent;
  if (c.tagName === "EM"){ want.titles.push(c.textContent); want.bodies.push(more.children[i + 1].textContent); }
  if (c.cls.has("tail")) want.tail = c.textContent;
});
console.log(ascii(JSON.stringify({got, want, toggle: toggle.textContent})));
"""


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("kind", ["room", "area"])
def test_the_caption_reaches_the_bridge_verbatim(tmp_path, kind, raw):
    from tower.world_builder import appearance_render as AR

    captions = ({"area": {"number": 1, "of": 2, "from_s": 86.7, "to_s": 109.6, "levelled": False}}
                if kind == "area" else None)
    page = _nc().compose(AR.apply_captions(_template(), captions))
    assert page is not None
    source = page[page.index("  function updateCaption(){"):
                  page.index("  /* -------- verification hooks")]
    body = (_CAPTION.replace("RAW_FLAG", json.dumps(raw))
            .replace("UPDATE_CAPTION", source))
    out = _run_node(_program(page, body), tmp_path, f"caption-{kind}-{raw}")
    got, want = out["got"], out["want"]
    assert got["head"] == want["head"] and got["line"] == want["line"] and got["tail"] == want["tail"]
    assert [s["title"] for s in got["sections"]] == want["titles"]
    assert [s["body"] for s in got["sections"]] == want["bodies"]
    assert want["titles"] == ["What you are looking at", "The flat grey patches", "Looking and moving",
                              "Finding your way", "The walk"]
    assert got["line"] == ("original local capture: faces and screens are NOT redacted · " if raw else "") \
        + "7 of 8 images loaded."
    assert got["tail"] == "Scale is unknown, so distances are relative."
    if kind == "area":
        assert "Area 1 of 2 \u2014 not placed in the room" in got["head"]
        assert any("may look tilted" in s["body"] for s in got["sections"])
    else:
        assert got["head"].endswith("captured images on reconstructed geometry") \
            or got["head"] == "Captured images on reconstructed geometry"
    if raw:
        assert got["head"].startswith("RESEARCH BUILD")
    assert out["toggle"] == "About"


# ---------------------------------------------------------------------------
# N14 -- in a browser (optional; required with WB_CHROME_BROWSER_REQUIRED=1)
# ---------------------------------------------------------------------------


def _chrome_binary():
    import shutil

    for candidate in (os.environ.get("WB_CHROME"), shutil.which("chrome"),
                      shutil.which("google-chrome"), shutil.which("chromium"),
                      r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                      r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if candidate and os.path.isfile(candidate):
            return candidate
    if os.environ.get(BROWSER_REQUIRED_ENV) == "1":
        pytest.fail(f"{BROWSER_REQUIRED_ENV}=1 but there is no Chrome on this host")
    pytest.skip("no Chrome on this host for the optional browser test")


# A handler shim stands in for WebKit's (the test's only addition to the page, put
# before the page's own script): it welcomes, then answers the page's `await` with
# `activate`, then `deactivate`, recording the computed display and accessibility
# role of every inventory element at each step.
_SHIM = r"""<script>
(function(){
"use strict";
const out = {posts: [], steps: {}};
const waits = [];
const INV = INVENTORY;
const snap = () => { const r = {};
  for (const id of INV){ const e = document.getElementById(id); const s = getComputedStyle(e);
    r[id] = {display: s.display, visibility: s.visibility,
             role: "computedRole" in e ? e.computedRole : null}; }
  return r; };
if (WITH_HANDLER) window.webkit = {messageHandlers: {wbChromeV1: {postMessage(msg){
  out.posts.push(JSON.parse(JSON.stringify(msg)));
  if (msg.type === "hello") return Promise.resolve({v: 1, type: "welcome", protocol: 1, nonce: "probe"});
  if (msg.type === "await") return new Promise(res => waits.push(res));
  return Promise.resolve(null);
}}}};
const done = () => { const pre = document.createElement("pre"); pre.id = "wbout";
  pre.textContent = JSON.stringify(out); document.documentElement.appendChild(pre); };
addEventListener("load", () => setTimeout(() => {
  const S = window.__wbAppearance;
  out.hooks = {view: typeof S.chromeView, words: typeof S.chromeWords, phase: S.phase};
  try { out.view = S.chromeView(); out.words = S.chromeWords(); } catch (e){ out.hookError = String(e); }
  out.steps.before = snap();
  if (!WITH_HANDLER){ done(); return; }
  waits.shift()({v: 1, type: "activate"});
  setTimeout(() => {
    out.steps.active = snap();
    out.bridgeActive = S.chromeBridge();
    waits.shift()({v: 1, type: "deactivate"});
    setTimeout(() => { out.steps.after = snap(); out.bridgeAfter = S.chromeBridge(); done(); }, 300);
  }, 300);
}, 1500));
})();
</script>"""


def _in_browser(chrome, page: str, where: Path, *, handler: bool) -> dict:
    shim = (_SHIM.replace("INVENTORY", json.dumps(list(_nc().INVENTORY)))
            .replace("WITH_HANDLER", "true" if handler else "false"))
    # before the page's own script: right after <title>, inside <head>
    page = page.replace("<title>", shim + "<title>", 1)
    where.mkdir(parents=True, exist_ok=True)
    target = where / "page.html"
    target.write_text(page, encoding="utf-8")
    run = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--enable-unsafe-swiftshader", "--use-gl=angle",
         "--use-angle=swiftshader", "--no-first-run", "--no-default-browser-check",
         "--disable-extensions", "--enable-blink-features=ComputedAccessibilityInfo",
         f"--user-data-dir={where / 'profile'}", "--window-size=390,844",
         "--virtual-time-budget=15000", "--dump-dom", target.as_uri()],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    m = re.search(r'<pre id="wbout">(.*?)</pre>', run.stdout, re.S)
    assert m, f"the browser reported nothing: {run.stderr[-2000:]}"
    return json.loads(html_lib.unescape(m.group(1)))


def test_the_bridge_in_a_browser(pages, session, tmp_path):
    chrome = _chrome_binary()
    _k, world, path, params, _kind, env = next(r for r in F.requests() if r[0] == "research/app")
    web_page = session.get(world, path, params, env, "1").text
    for kind in ("room", "research"):
        native = _in_browser(chrome, pages[kind], tmp_path / kind, handler=True)
        assert native["hooks"]["view"] == "function" and native["hooks"]["words"] == "function", native
        assert native["words"] == {"dark": "Not reconstructed from here", "darkTap": "Tap to turn back",
                                   "edge": "Movement stops here"}
        assert 0 < native["view"]["halfFovRad"] < 1.5708
        hello = native["posts"][0]
        assert hello["type"] == "hello" and "dark" in hello["labels"] and "edge" in hello["labels"]
        before, active, after = (native["steps"][k] for k in ("before", "active", "after"))
        for el in _nc().INVENTORY:
            assert active[el]["display"] == "none", (kind, el)
            # CHARACTERIZATION: Chrome's `computedRole` reports an element's nominal
            # role even under `display:none` ("generic" for #caption), so it cannot
            # say whether a node is in the accessibility tree. Computed
            # `display:none` is the proof: it takes a node out of rendering,
            # hit-testing and the accessibility tree at once (CSS Display 3 §2.7).
            assert after[el] == before[el], (kind, el)
        assert native["bridgeActive"]["active"] is True and native["bridgeAfter"]["active"] is False
        states = [p for p in native["posts"] if p["type"] == "state"]
        assert states[-1]["active"] is False
        if kind == "research":
            # the marker is on screen before activate, and only activate hides it
            assert before["rawmark"]["display"] != "none" and before["rawmark"]["visibility"] == "visible"
            assert hello["research"]["raw"] is True and hello["research"]["marker"]
        # before `activate` the inventory renders exactly as the web page's
        web = _in_browser(chrome, web_page if kind == "research" else
                          session.get(*_room_web(session)).text, tmp_path / f"{kind}-web",
                          handler=False)
        assert web["steps"]["before"] == before, kind


def _room_web(session):
    _k, world, path, params, _kind, env = next(r for r in F.requests() if r[0] == "room/app")
    return world, path, params, env, "1"
