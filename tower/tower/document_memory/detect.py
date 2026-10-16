"""Is there a page-like region in this frame, and does it hold text?

Cheap enough to run on every frame, because that is the only way the
expensive path can be rare. Classical, not learned: a page is a bright,
roughly rectangular, roughly convex region, and a contour finder answers
that in a couple of milliseconds without a model, a download or a GPU.

**Three gates, not one.** A closed laptop lid, a picture frame, a monitor
bezel and a blank whiteboard are all page-shaped. Rectangle detection
alone would call every one of them a document.

The second gate asks whether the interior has rows of ink. That is
necessary and, on its own, badly insufficient: **venetian blinds, brick
courses, floor tiles and a striped shirt all produce rows of dark pixels
just as reliably as text does**, so a row-ink gate alone is insufficient.

The third gate counts dark/light transitions along each inky row, on the
theory that a line of text is many short runs while a slat or a mortar
course is one run spanning the width.

**That theory is only half true.** Against 9,199 real frames the gate
separates page-shaped crops by SIZE, not by glyphiness: a real backlit
keyboard scores 19-26, while all thirteen crops of real on-screen text
found in that corpus -- laptop and phone displays, the only real text
there -- score 0-8, BELOW the keyboard. At 360x640 the glyphs are about
two pixels tall and the keycaps are fifteen, so the keyboard is the more
glyph-like of the two.

The gate still stands between a brick wall and OCR, on a 10-wide window
derived in `MIN_ROW_TRANSITIONS` below, not on an enormous margin.
Guarded by `tests/test_document_detect_corpus.py`.

Retained, not on the live path: `gate.PageFinder` is the live per-frame
stage. This module is kept for `warp_page` and the corpus tests.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# A page should occupy a real portion of the view. Below this it is
# either far away or not the thing the wearer is looking at, and warping
# it would produce an unreadable crop.
MIN_AREA_FRACTION = 0.06
MAX_AREA_FRACTION = 0.98

# How far from a true quadrilateral a contour may be. approxPolyDP's
# epsilon as a fraction of perimeter.
POLY_EPSILON_FRACTION = 0.02

# A page is convex. A hand across it, or two overlapping sheets, is not
# something V1 tries to read.
MIN_SOLIDITY = 0.85

# Aspect bounds, generous: A4 portrait is 1.41, landscape 0.71, and a
# perspective view of either stretches that further.
MIN_ASPECT = 0.25
MAX_ASPECT = 4.0

# Text-likeness. Measured inside the detected quad after warping to a
# canonical size, so the threshold does not depend on how big the page
# happened to be in frame.
TEXT_PROBE_SIZE = (480, 360)
MIN_TEXT_ROW_FRACTION = 0.08
MIN_INK_FRACTION = 0.004
MAX_INK_FRACTION = 0.60

# The glyph gate, derived against 9,199 real frames. Synthetic negatives
# are not a basis for it: rendered text had scored 43-86 with
# blinds/bricks/tiles/stripes/keyboard all 0 and a threshold of 8, but
# over `data/captures/` a real venetian blind measures 8.0 and a real
# backlit keyboard 19-26. A rendered slat is one clean run, while a real
# slat has a rail, a bracket, a gap and a highlight, and a real backlit
# keyboard is a grid of small light patches separated by dark gaps, which
# is a tolerable description of a line of glyphs.
#
# READ THIS BEFORE CHANGING THE NUMBER. `row_transitions` does not
# separate text from structure. It separates LARGE crops from SMALL ones:
# on the same rendered page, shrinking it from 59% of a 360x640 frame to
# 14% takes a receipt from 40 transitions to 24, straight through the
# keyboard's 19-26. The two populations genuinely overlap, and no
# threshold on this statistic tells a page from a keyboard by shape. What
# makes a threshold derivable at all is a second fact that happens to
# coincide at this resolution, and both edges are measured, not chosen:

# Ceiling of the real negatives. The worst case in 1,395 page-shaped
# quads across 9,199 frames of ordinary indoor life -- one backlit laptop
# keyboard. The median over those quads is 0 and p99 is 5; this is the
# tail, and it rests on a single keyboard in a single capture.
CORPUS_STRUCTURE_CEILING = 26.0

# Floor of the pages worth detecting. The smallest rendered placement that
# still yielded usable text at 360x640: a receipt at 38% of the frame,
# word_recall 0.714. Every placement below it measured 0.000 recall -- so
# the pages this gate now rejects are pages OCR could not have read.
READABLE_PAGE_FLOOR = 36.0

# Halfway, because neither edge is soft. Sitting at the ceiling would ship
# with no margin against the next keyboard; sitting at the floor would
# reject the last page that still reads. This buys 5 transitions of
# headroom on each side, and that is the whole budget -- a 10-wide
# window, not an order of magnitude.
#
# THIS IS PINNED TO 360x640. Both edges are functions of how many pixels
# the crop has: raise the stream or move to higher-resolution stills (the
# recommended route, since a higher-resolution stream would hurt World
# Builder's tracking), and the structure ceiling and the readable floor
# both move. Re-derive against `tests/test_document_detect_corpus.py`'s
# real negatives, do not scale.
MIN_ROW_TRANSITIONS = round((CORPUS_STRUCTURE_CEILING + READABLE_PAGE_FLOOR) / 2)


@dataclass(frozen=True)
class PageCandidate:
    """A page-like region, and the evidence for calling it one."""

    corners: np.ndarray  # (4, 2) float32, clockwise from top-left
    area_fraction: float
    aspect: float
    solidity: float
    text_row_fraction: float
    ink_fraction: float
    row_transitions: float
    sharpness: float
    squareness: float

    @property
    def centre(self) -> tuple[float, float]:
        return (
            float(self.corners[:, 0].mean()),
            float(self.corners[:, 1].mean()),
        )


def order_corners(points: np.ndarray) -> np.ndarray:
    """Clockwise from top-left, so a warp is never mirrored or rotated.

    By coordinate sums and differences rather than by angle: it is exact
    for a convex quad and needs no trigonometry.
    """
    points = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(total)]  # top-left
    ordered[2] = points[np.argmax(total)]  # bottom-right
    ordered[1] = points[np.argmin(diff)]  # top-right
    ordered[3] = points[np.argmax(diff)]  # bottom-left
    return ordered


def _squareness(corners: np.ndarray) -> float:
    """How fronto-parallel the quad is: 1.0 is a perfect rectangle.

    The ratio of the shorter to the longer of each opposing side pair. A
    page seen at a steep angle has one pair much shorter than the other,
    and that is exactly what makes it a poor OCR candidate.
    """
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    horizontal = min(top, bottom) / max(top, bottom) if max(top, bottom) else 0.0
    vertical = min(left, right) / max(left, right) if max(left, right) else 0.0
    return float(min(horizontal, vertical))


def warp_page(gray: np.ndarray, corners: np.ndarray, size=None) -> np.ndarray:
    """Flatten the quad into a rectangle.

    This is what lets the wearer read normally instead of holding a page
    flat to the camera -- the passive-operation premise, applied to
    documents.
    """
    if size is None:
        width = int(
            max(
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[3]),
            )
        )
        height = int(
            max(
                np.linalg.norm(corners[3] - corners[0]),
                np.linalg.norm(corners[2] - corners[1]),
            )
        )
        size = (max(width, 1), max(height, 1))
    target = np.array(
        [[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    return cv2.warpPerspective(gray, matrix, size)


def measure_text_likeness(page_gray: np.ndarray) -> tuple[float, float, float]:
    """(text_row_fraction, ink_fraction, row_transitions) for a warped page.

    Lines of text produce rows with far more dark pixels than the margins
    between them. A blank sheet has no such rows; a photograph has dark
    pixels everywhere and no row structure. Measuring the FRACTION OF ROWS
    that stand out separates those two, where a plain darkness count could
    not.

    It does NOT separate text from blinds, bricks, tiles or stripes, all
    of which have exactly that row structure. `row_transitions` is the
    median number of dark/light crossings ALONG an inky row; on real
    frames it does not separate text from structure either (real
    on-screen text scores 0-8, a real slat about 8, a keyboard 19-26);
    see `MIN_ROW_TRANSITIONS`.
    """
    if page_gray.size == 0:
        return 0.0, 0.0, 0.0
    probe = cv2.resize(page_gray, TEXT_PROBE_SIZE, interpolation=cv2.INTER_AREA)
    # Otsu rather than a fixed level: page brightness varies enormously
    # with lighting, and a fixed threshold would measure the lamp.
    _, binary = cv2.threshold(
        probe, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    ink_fraction = float(np.count_nonzero(binary)) / binary.size
    row_ink = binary.sum(axis=1) / (255.0 * binary.shape[1])

    # A row carries text if it is meaningfully inkier than the median row.
    threshold = max(float(np.median(row_ink)) * 1.5, 0.02)
    inky_rows = np.flatnonzero(row_ink > threshold)
    text_row_fraction = float(len(inky_rows)) / row_ink.size
    if not len(inky_rows):
        return text_row_fraction, ink_fraction, 0.0

    # Median, not mean: one row of solid rule or a table border should not
    # drag the figure down for a page that is otherwise glyphs.
    mask = binary > 0
    transitions = [
        int(np.count_nonzero(np.diff(mask[row].astype(np.int8))))
        for row in inky_rows
    ]
    return text_row_fraction, ink_fraction, float(np.median(transitions))


def detect_page(gray: np.ndarray) -> PageCandidate | None:
    """The cheap per-frame gate. None means "no page-like region here".

    Returns the single best candidate rather than a list: V1 reads the
    document the wearer is facing, and a frame with two competing pages is
    a frame worth skipping rather than guessing at.
    """
    if gray is None or gray.size == 0:
        return None
    height, width = gray.shape[:2]
    if min(height, width) < 32:
        return None

    frame_area = float(height * width)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Close small gaps so a broken page border still forms one contour.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area / frame_area < MIN_AREA_FRACTION:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, POLY_EPSILON_FRACTION * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        corners = order_corners(approx)
        quad_area = cv2.contourArea(corners)
        area_fraction = quad_area / frame_area
        if not MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = float(quad_area / hull_area) if hull_area else 0.0
        if solidity < MIN_SOLIDITY:
            continue

        top = np.linalg.norm(corners[1] - corners[0])
        left = np.linalg.norm(corners[3] - corners[0])
        aspect = float(top / left) if left else 0.0
        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            continue

        page = warp_page(gray, corners)
        text_row_fraction, ink_fraction, row_transitions = measure_text_likeness(
            page
        )
        if text_row_fraction < MIN_TEXT_ROW_FRACTION:
            continue
        if not MIN_INK_FRACTION <= ink_fraction <= MAX_INK_FRACTION:
            continue
        # The glyph gate. Without it a brick wall reaches OCR.
        if row_transitions < MIN_ROW_TRANSITIONS:
            continue

        candidate = PageCandidate(
            corners=corners,
            area_fraction=float(area_fraction),
            aspect=aspect,
            solidity=solidity,
            text_row_fraction=text_row_fraction,
            ink_fraction=ink_fraction,
            row_transitions=row_transitions,
            sharpness=float(cv2.Laplacian(page, cv2.CV_64F).var()),
            squareness=_squareness(corners),
        )
        if best is None or candidate.area_fraction > best.area_fraction:
            best = candidate

    return best
