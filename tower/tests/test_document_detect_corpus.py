"""The glyph gate, judged against REAL frames instead of rendered ones.

PHYSICAL, NOT SYNTHETIC. Every negative in this file is a frame the
glasses actually recorded.

This file exists because the old suite could not have caught the bug it
guards. `test_document_detect.py` and `test_document_memory_hostile.py`
draw their negatives from the same renderer as their positives, so a
threshold tuned on that renderer passed both. The module docstring in
`detect.py` claimed venetian blinds, brick, tiles, stripes and a keyboard
all measured 0 row transitions against a threshold of 8 -- "an order of
magnitude below the text floor". Run over `data/captures/`, a real
venetian blind measures 8.0 and a real backlit keyboard 19-26, and the
detector fired six times in 9,199 frames, every one of them wrong.

The corpus is machine-local (`data/` is gitignored), so these skip where
it is absent. That is a real limit and it is the reason the derivation is
also summarised beside `MIN_ROW_TRANSITIONS` in
`tower/document_memory/detect.py`.

What is NOT here: a page. No capture in the corpus contains a sheet of
paper, so nothing in this file measures recall or proves the detector
finds documents. It pins the negative side only.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from tower.document_memory.detect import (
    MIN_ROW_TRANSITIONS,
    detect_page,
    measure_text_likeness,
    order_corners,
    warp_page,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "captures"

pytestmark = pytest.mark.skipif(
    not CORPUS.is_dir(),
    reason="real capture corpus absent; data/ is machine-local and gitignored",
)


# The eight quads that cleared every gate in 9,199 frames on 2026-08-26,
# with the corners the detector itself produced. Hard-coded rather than
# re-detected, so the measurement stays available even after a gate change
# stops the detector from proposing them.
FALSE_POSITIVES = [
    (
        "venetian blind over a kitchen window",
        "22e9d4289cb440fbb3f14e6da369a136",
        569,
        [[78.0, 398.0], [194.0, 421.0], [200.0, 639.0], [58.0, 639.0]],
        8.0,
    ),
    (
        "backlit laptop keyboard",
        "b5a0d654182548f5a8695ff40b772829",
        1109,
        [[57.0, 463.0], [244.0, 455.0], [282.0, 595.0], [61.0, 634.0]],
        19.0,
    ),
    (
        "backlit laptop keyboard",
        "b5a0d654182548f5a8695ff40b772829",
        1111,
        [[60.0, 459.0], [243.0, 455.0], [277.0, 589.0], [61.0, 621.0]],
        25.0,
    ),
    (
        "backlit laptop keyboard",
        "b5a0d654182548f5a8695ff40b772829",
        1113,
        [[61.0, 455.0], [244.0, 453.0], [275.0, 586.0], [60.0, 614.0]],
        26.0,
    ),
    (
        "backlit laptop keyboard",
        "b5a0d654182548f5a8695ff40b772829",
        1116,
        [[65.0, 454.0], [247.0, 452.0], [277.0, 586.0], [65.0, 611.0]],
        19.0,
    ),
    (
        "backlit laptop keyboard",
        "b5a0d654182548f5a8695ff40b772829",
        1117,
        [[68.0, 455.0], [250.0, 454.0], [282.0, 585.0], [68.0, 612.0]],
        21.0,
    ),
]

# Screens the wearer was actually reading, cropped by hand off a
# coordinate grid and checked by eye. The ONLY real text in the corpus.
# Not ground truth for OCR -- there is no transcript -- but ground truth
# for "a human confirms this crop is text".
REAL_SCREENS = [
    ("64f481147ec04674a0d857ca4f1964f3", 723, [[68, 305], [270, 285], [277, 430], [80, 455]]),
    ("69030fba28c54ed5a31ba0bf3677130f", 803, [[48, 273], [256, 267], [248, 378], [53, 373]]),
    ("2e6cffa275b24b7d87d68ec1d6a6cfdf", 733, [[51, 238], [251, 235], [248, 373], [48, 368]]),
    ("2e6cffa275b24b7d87d68ec1d6a6cfdf", 1212, [[115, 248], [309, 249], [307, 360], [116, 363]]),
    ("b35d8ab85c364b9da44499d2a7f00638", 1, [[19, 281], [216, 265], [224, 406], [21, 425]]),
    ("341b0fdac88a4b6f9d6ff720d4341690", 251, [[146, 378], [222, 350], [276, 570], [190, 592]]),
    ("68a7c7ba6cb0443886137422ac7cf336", 1, [[69, 210], [317, 206], [312, 349], [67, 355]]),
    ("0f0c55b662fe4df189fa275bf3dd506d", 1821, [[180, 383], [272, 378], [276, 578], [184, 585]]),
    ("b1ab1d413c0544f0971d27038818fa44", 869, [[88, 306], [291, 300], [288, 440], [90, 445]]),
    # Not hand-drawn. These are the corners detect_page's own contour
    # stage proposed for a laptop screen dense with text -- it found the
    # screen, warped it, and the glyph gate scored it 0.0, in the same
    # corpus where a keyboard scored 26.
    ("b901bc7fce0c4f5fbd1e1282a28e8c38", 3055, [[88, 297], [263, 307], [257, 418], [84, 410]]),
]


def _frame(capture_id: str, seq: int) -> np.ndarray:
    """One real frame as grayscale. Read, never copied into the repo."""
    path = CORPUS / capture_id / "frames" / f"{seq:06d}.jpg"
    if not path.exists():
        matches = [
            p
            for p in (CORPUS / capture_id / "frames").glob("*.jpg")
            if p.stem.isdigit() and int(p.stem) == seq
        ]
        if not matches:
            pytest.skip(f"{capture_id}/{seq} absent from this corpus")
        path = matches[0]
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert gray is not None, path
    return gray


class TestTheRealNegatives:
    """Blinds and keyboards, at the values the corpus actually produces."""

    @pytest.mark.parametrize(
        "what,capture_id,seq,corners,expected",
        FALSE_POSITIVES,
        ids=[f"{f[0]}-{f[2]}" for f in FALSE_POSITIVES],
    )
    def test_a_real_structure_scores_below_the_glyph_gate(
        self, what, capture_id, seq, corners, expected
    ):
        """The published table said 0 for both of these. It said so from
        a renderer. Here is what the surface itself measures."""
        gray = _frame(capture_id, seq)
        quad = order_corners(np.array(corners, np.float32))

        _, _, transitions = measure_text_likeness(warp_page(gray, quad))

        assert transitions == pytest.approx(expected, abs=1.0), (
            f"{what} measured {transitions}, was {expected} on 2026-08-26"
        )
        assert transitions < MIN_ROW_TRANSITIONS, (
            f"{what} at {transitions} transitions still clears the glyph gate"
        )

    @pytest.mark.parametrize(
        "what,capture_id,seq,corners,expected",
        FALSE_POSITIVES,
        ids=[f"{f[0]}-{f[2]}" for f in FALSE_POSITIVES],
    )
    def test_the_frame_it_came_from_is_no_longer_called_a_document(
        self, what, capture_id, seq, corners, expected
    ):
        """End of the cheap path, on the exact frame that fired."""
        assert detect_page(_frame(capture_id, seq)) is None, (
            f"{what} in {capture_id}/{seq} is still detected as a page"
        )


class TestTheWholeCorpus:
    """9,199 frames of ordinary indoor life, none of them a document."""

    def test_no_frame_in_the_corpus_is_called_a_document(self):
        """The precision measurement, run over everything.

        A visual review of all 18 captures found laptop screens, phone
        screens, ChArUco boards, walls, doors, ceilings and carpet, and
        not one sheet of paper. So every firing here is a false positive
        by construction, and the only correct count is zero.

        Sampling would not do: at stride 20 the old detector fired on
        nothing, because six frames in 9,199 is not something a sample
        finds. Reading the whole corpus costs about eleven seconds.
        """
        frames = fired = 0
        offenders = []
        for capture_dir in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
            for path in sorted((capture_dir / "frames").glob("*.jpg")):
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                frames += 1
                if detect_page(gray) is not None:
                    fired += 1
                    if len(offenders) < 10:
                        offenders.append(f"{capture_dir.name[:8]}/{path.stem}")

        assert frames > 1000, f"corpus looks truncated: {frames} frames"
        assert fired == 0, f"{fired}/{frames} frames detected as documents: {offenders}"


class TestTheOnlyRealTextThereIs:
    """Screens, and why they cannot rescue the gate.

    The corpus is full of laptop and phone displays dense with text. They
    are the only real positives available. They do not save the metric --
    they condemn it. Pinned so that the next person to touch the threshold
    sees the overlap rather than rediscovering it.
    """

    @pytest.mark.parametrize(
        "capture_id,seq,corners", REAL_SCREENS, ids=[f"{s[0][:6]}-{s[1]}" for s in REAL_SCREENS]
    )
    def test_a_real_screen_of_text_scores_no_higher_than_a_wall(
        self, capture_id, seq, corners
    ):
        """At 360x640 the glyphs are about two pixels tall.

        `measure_text_likeness` thresholds THRESH_BINARY_INV, and every
        screen here is dark-mode, so its "ink" is the background: the crop
        binarises to one blob per line, not to glyphs. Inverting the
        polarity does not rescue it either (1-8 transitions, measured);
        the detail is gone before polarity gets a say.
        """
        gray = _frame(capture_id, seq)
        quad = order_corners(np.array(corners, np.float32))
        page = warp_page(gray, quad)

        _, _, transitions = measure_text_likeness(page)
        _, _, inverted = measure_text_likeness(255 - page)

        assert transitions <= 2.0, f"{capture_id}/{seq} scored {transitions}"
        assert inverted <= 8.0, f"{capture_id}/{seq} inverted scored {inverted}"

    def test_real_text_scores_below_the_hardest_real_negative(self):
        """The finding, in one assertion.

        A backlit keyboard out-scores every genuine line of text in this
        corpus. `row_transitions` is not separating glyphs from structure;
        it is separating big crops from small ones. Any threshold that
        keeps the keyboard out keeps all of this text out too, and no
        threshold anywhere reverses the order.
        """
        keyboard = max(f[4] for f in FALSE_POSITIVES)
        best_real_text = 0.0
        for capture_id, seq, corners in REAL_SCREENS:
            gray = _frame(capture_id, seq)
            page = warp_page(gray, order_corners(np.array(corners, np.float32)))
            for probe in (page, 255 - page):
                best_real_text = max(best_real_text, measure_text_likeness(probe)[2])

        assert best_real_text < keyboard, (
            "the order flipped: real text now out-scores the keyboard, so the "
            "derivation in detect.py rests on evidence that has changed"
        )
