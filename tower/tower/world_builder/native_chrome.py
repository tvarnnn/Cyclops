"""U1.1 native chrome: the appearance page, plus five insertions, for a phone
that draws the page's chrome itself.

Contract: `docs/contracts/WORLD-BUILDER-WORLDS.md` §4c (and COMPONENTS §5.4 for
an area page). Spec: RUN/lead/specs/U11-NATIVE-CHROME-SPEC-20261004.md §2.3.

THE TEMPLATE IS NOT EDITED. The native variant is the composed web page --
today's bytes -- with five insertions applied after composition, each at an
anchor that must occur EXACTLY ONCE in that page:

    E  the echo, `<meta name="wb-chrome" content="native">`, right after the
       page's `wb-representation` meta;
    C  one style block that hides the page's chrome while `body.wbnative` is set;
    K  two read-only hooks in `main()` (`S.chromeView`, `S.chromeWords`);
    F  one line at the end of `frame()` that calls the bridge's per-frame hook;
    B  one inline `<script>`: `appearance_chrome_bridge.js`.

ALL FIVE OR NONE. A miss -- an anchor a future template edit moved, deleted or
repeated -- logs one ERROR naming it and returns None, and the caller serves the
web page. So the echo, which is what tells the phone to draw, can never be
present unless every hook it relies on is. That is `surface_render.replace_anchors`'
all-or-nothing rule, with the result made visible.

The two variants share one revision (§4a rule 3): the caller stamps the composed
page with the same `wb-revision` as the web page.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROTOCOL = 1
BRIDGE_NAME = "appearance_chrome_bridge.js"

# What the page stops drawing while `body.wbnative` is set: every element child
# of `<body>` but `#wrap` (the canvas) and the scripts. A test parses the
# template and fails on any overlay this list does not name.
INVENTORY = ("caption", "status", "bar", "compass", "clabel", "back", "hint", "dark",
             "rawmark", "msg")

ECHO = '<meta name="wb-chrome" content="native">'

ANCHOR_ECHO = '<meta name="wb-representation" content="appearance">'
ANCHOR_CSS = "</style>\n</head>"
ANCHOR_HOOKS = "  S.faceTheRoom = () => faceTheRoom();"
ANCHOR_FRAME = ("    savePose();                      "
                "// where he is, for a reload of this tab")
ANCHOR_BRIDGE = "</script>\n</body>"

CSS = (
    "/* U1.1 native chrome (WORLDS §4c): while the phone draws the chrome, the page "
    "draws only its canvas. */\n"
    "body.wbnative #caption,body.wbnative #status,body.wbnative #bar,body.wbnative #compass,\n"
    "body.wbnative #clabel,body.wbnative #back,body.wbnative #hint,body.wbnative #dark,\n"
    "body.wbnative #rawmark,body.wbnative #msg{display:none!important}\n"
)

HOOKS = (
    "\n"
    "\n"
    "  // U1.1 native chrome (WORLDS §4c): what the bridge reads. Nothing here draws "
    "or moves anything.\n"
    "  S.chromeView = () => ({drawn: shownAt !== null, restoring,\n"
    "    lit: ringProf ? Array.from(ringProf, s => NAV.smooth(0.08, NAV.T_HI, s)) : null,\n"
    "    sense: ringProf ? yawSign() : 1, headingRad: shownPose().yaw,\n"
    "    halfFovRad: (c => Math.atan(Math.tan(c.fy / 2) * c.aspect))(currentCamera())});\n"
    "  S.chromeWords = () => ({dark: HINT_DARK, darkTap: HINT_DARK_TAP, edge: HINT_EDGE});"
)

FRAME_CALL = '\n    if (typeof S.chromeFrame === "function") S.chromeFrame();'


def bridge_path() -> Path:
    return Path(__file__).with_name(BRIDGE_NAME)


def _bridge_block() -> str:
    # Read like the template is (`read_text`, universal newlines), so a CRLF
    # checkout serves the same bytes as an LF one.
    source = bridge_path().read_text(encoding="utf-8")
    if "</script" in source.lower():
        # it is inlined in a <script>: this would end it early
        raise ValueError(f"{BRIDGE_NAME} must not contain a closing script tag")
    return "\n<script>\n" + source.rstrip("\n") + "\n</script>"


BRIDGE_BLOCK = _bridge_block()

# (name, anchor, replacement). Each replacement contains its anchor once, and
# deleting the inserted text (`INSERTED[name]`) from the replacement gives the
# anchor back -- which is how the tests prove the native page is the web page
# plus these five things and nothing else.
INSERTIONS: tuple[tuple[str, str, str], ...] = (
    ("E", ANCHOR_ECHO, ANCHOR_ECHO + ECHO),
    ("C", ANCHOR_CSS, CSS + ANCHOR_CSS),
    ("K", ANCHOR_HOOKS, ANCHOR_HOOKS + HOOKS),
    ("F", ANCHOR_FRAME, ANCHOR_FRAME + FRAME_CALL),
    ("B", ANCHOR_BRIDGE, "</script>" + BRIDGE_BLOCK + "\n</body>"),
)
INSERTED: dict[str, str] = {"E": ECHO, "C": CSS, "K": HOOKS, "F": FRAME_CALL, "B": BRIDGE_BLOCK}


def compose(page: str) -> str | None:
    """`page` with all five insertions, or None (logged at ERROR) when any anchor
    does not occur exactly once in it."""
    for name, anchor, _replacement in INSERTIONS:
        n = page.count(anchor)
        if n != 1:
            logger.error("[Tower][WorldBuilder][chrome] native chrome not composed: "
                         "anchor %s occurs %d times", name, n)
            return None
    for _name, anchor, replacement in INSERTIONS:
        page = page.replace(anchor, replacement, 1)
    return page
