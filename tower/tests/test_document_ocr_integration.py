"""Real OCR, against text we know because we rendered it.

Opt-in: set TOWER_RUN_MODEL_TESTS=1. Constructing the reader costs ~5 s
and reading a page ~1.2 s, and the models download on first run. A
default suite that paid either is a suite people stop running.

This is the only place OCR ACCURACY is measured. Everything else in the
Document Memory suite uses `FixedTextRecogniser`, which tests the
pipeline rather than the recogniser -- and would happily pass while OCR
returned nonsense. That gap is what this file closes.
"""

import os
import re

import cv2
import pytest

from tower.confidence import Confidence
from tower.document_memory.detect import detect_page, warp_page
from tower.document_memory.dwell import DwellPolicy
from tower.document_memory.engine import DocumentMemoryEngine
from tower.document_memory.ocr import EasyOcrRecogniser
from tower.document_memory.retrieval import DocumentMemory
from tower.document_memory.store import DocumentStore
from tests import document_fixtures as fx

pytestmark = pytest.mark.skipif(
    os.environ.get("TOWER_RUN_MODEL_TESTS") != "1",
    reason="opt-in: downloads EasyOCR models on first run and costs ~1.2s "
    "per page; set TOWER_RUN_MODEL_TESTS=1 to run",
)


@pytest.fixture(scope="module")
def recogniser():
    engine = EasyOcrRecogniser()
    engine.load()
    yield engine
    engine.release()


_TOKEN = re.compile(r"[a-z0-9]+")


def word_recall(expected: str, actual: str) -> float:
    """Fraction of the rendered words that OCR actually captured.

    Word recall rather than sequence similarity, and the difference is not
    cosmetic. A receipt's columns come back with their whitespace runs
    collapsed and occasionally with a word reordered -- measured: "days"
    migrating to the end of the line. Character-sequence similarity scores
    that as a failure; retrieval does not care, because a document is
    findable if its distinctive WORDS were captured.

    Recall, not F1: a spurious extra word costs a little precision and
    almost nothing in a lexical search, while a missing word makes a
    document unfindable.
    """
    expected_words = _TOKEN.findall(expected.lower())
    actual_words = set(_TOKEN.findall(actual.lower()))
    if not expected_words:
        return 1.0
    return sum(1 for word in expected_words if word in actual_words) / len(
        expected_words
    )


class TestAccuracyAgainstRenderedText:
    @pytest.mark.parametrize(
        "lines", [fx.TRANSFORMER_PAPER, fx.DEPTH_NOTES, fx.RECEIPT]
    )
    def test_a_clean_page_is_read_almost_exactly(self, recogniser, lines):
        """The ground truth is the string the fixture rendered.

        Measured at 1.000 recall on all three fixtures at full page
        resolution. Bounded at 0.95 so an OCR upgrade does not have to
        chase an exact number.
        """
        page = cv2.cvtColor(fx.render_page(lines), cv2.COLOR_BGR2GRAY)

        result = recogniser.read(page)

        assert word_recall(fx.page_text(lines), result.text) > 0.95

    @pytest.mark.parametrize("tilt", [0.0, 0.5, 1.0])
    def test_tilt_barely_matters_once_the_page_is_warped(self, recogniser, tilt):
        """Perspective correction is what lets the wearer read normally.

        The measured spread across a full range of tilt at 640x480 is
        small -- 0.905 to 1.000 recall -- which is the evidence that the
        warp is doing its job. Resolution, not angle, is what hurts; see
        the class below.
        """
        frame, _ = fx.place_page(
            fx.render_page(fx.DEPTH_NOTES), frame_size=(640, 480), tilt=tilt
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidate = detect_page(gray)
        assert candidate is not None

        result = recogniser.read(warp_page(gray, candidate.corners))

        assert word_recall(fx.page_text(fx.DEPTH_NOTES), result.text) > 0.85

    def test_confidence_survives_as_a_label(self, recogniser):
        page = cv2.cvtColor(fx.render_page(fx.RECEIPT), cv2.COLOR_BGR2GRAY)

        result = recogniser.read(page)

        assert result.region_count > 3
        assert 0.0 < result.mean_confidence <= 1.0
        assert result.confidence_label in (Confidence.MEDIUM, Confidence.HIGH)

    def test_a_blank_page_yields_no_text_rather_than_invented_text(
        self, recogniser
    ):
        """"We looked and found nothing" is a real answer.

        It is a different fact from "we never looked", and an OCR engine
        that hallucinated a word here would poison retrieval.
        """
        import numpy as np

        result = recogniser.read(np.full((600, 450), 250, dtype=np.uint8))

        assert result.text.strip() == ""
        assert result.region_count == 0
        assert result.confidence_label is Confidence.UNKNOWN

    def test_regions_carry_boxes_so_a_title_can_be_reassembled(self, recogniser):
        """A heading arrives as several regions, not one."""
        page = cv2.cvtColor(fx.render_page(fx.TRANSFORMER_PAPER), cv2.COLOR_BGR2GRAY)

        result = recogniser.read(page)

        assert all(region.box is not None for region in result.regions)
        first = result.regions[0]
        assert first.height and first.height > 0


class TestResolutionIsTheBindingConstraint:
    """The most consequential measurement this cartridge produced.

    The glasses deliver 640x360 today. A page inside that frame warps to
    roughly 500x320, which puts a 34px rendered font at about 10px --
    below what the recogniser needs. Measured word recall:

        1280x720   0.957 - 1.000
         640x480   0.905 - 1.000
         640x360   0.429 - 0.810     <- what we actually receive

    This is not a Tower problem and no amount of Tower work fixes it.
    `CARTRIDGE-GROUNDWORK.md` predicted it -- "Text/Document ... Missing:
    resolution negotiation" -- and this is the measurement that turns the
    prediction into a requirement on the iOS/DAT side.
    """

    @staticmethod
    def _recall_at(recogniser, frame_size) -> float:
        frame, _ = fx.place_page(
            fx.render_page(fx.TRANSFORMER_PAPER), frame_size=frame_size
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidate = detect_page(gray)
        assert candidate is not None, f"page not detected at {frame_size}"
        result = recogniser.read(warp_page(gray, candidate.corners))
        return word_recall(fx.page_text(fx.TRANSFORMER_PAPER), result.text)

    def test_a_high_resolution_frame_reads_well(self, recogniser):
        assert self._recall_at(recogniser, (1280, 720)) > 0.90

    def test_the_delivered_resolution_reads_materially_worse(self, recogniser):
        """Asserted as a COMPARISON, so it survives an OCR upgrade.

        If this test ever fails because the delivered resolution caught
        up, that is the good news it looks like -- delete it and record
        the new measurement.
        """
        high = self._recall_at(recogniser, (1280, 720))
        delivered = self._recall_at(recogniser, (640, 360))

        assert delivered < high - 0.2, (
            f"delivered={delivered:.3f} high={high:.3f}; if the gap has "
            "closed, re-measure and update the resolution requirement in "
            "tower/docs/contracts/IOS-TO-TOWER-RECONCILIATION.md §3.2"
        )

    def test_the_page_is_still_DETECTED_at_the_delivered_resolution(
        self, recogniser
    ):
        """Detection is not the bottleneck. Recognition is.

        Worth pinning separately: it means the dwell machinery works at
        the delivered rate and only the OCR stage is starved.
        """
        frame, _ = fx.place_page(
            fx.render_page(fx.TRANSFORMER_PAPER), frame_size=(640, 360)
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        assert detect_page(gray) is not None


class TestEndToEndWithRealOcr:
    def test_a_rendered_reading_session_becomes_a_searchable_memory(
        self, tmp_path, recogniser
    ):
        """The whole product claim, on synthetic footage.

        Observe two documents, then ask a question whose answer is known
        because we chose which page contains which words.
        """
        store = DocumentStore(tmp_path)
        clock = [1000.0]
        engine = DocumentMemoryEngine(
            store,
            recogniser,
            policy=DwellPolicy(min_frames=3, min_seconds=0.6),
            clock=lambda: clock[0],
        )

        frames = (
            fx.document_frames(fx.TRANSFORMER_PAPER, 8)
            + [fx.encode(fx.no_page_frame())] * 6
            + fx.document_frames(fx.RECEIPT, 8)
        )
        for index, frame in enumerate(frames):
            clock[0] += 0.3
            engine.observe(frame, received_at=clock[0], source_seq=index)
        engine.flush()

        assert store.count() == 2

        memory = DocumentMemory(store)

        paper = memory.search_text("transformer attention")
        assert paper.sufficient_evidence
        assert "Attention" in (paper.matches[0].document.title or "")

        policy = memory.search_text("return policy")
        assert policy.sufficient_evidence
        assert "HARDWARE" in (policy.matches[0].document.title or "").upper()

        # And the refusal, which matters as much as the matches.
        absent = memory.search_text("mitochondrial dna sequencing")
        assert absent.sufficient_evidence is False

    def test_the_title_is_the_whole_first_line_not_the_first_word(
        self, tmp_path, recogniser
    ):
        """A recogniser splits a heading across regions.

        Taking regions[0] alone yielded "Attention" where the page says
        "Attention Is All You Need".
        """
        store = DocumentStore(tmp_path)
        clock = [1000.0]
        engine = DocumentMemoryEngine(
            store,
            recogniser,
            policy=DwellPolicy(min_frames=3, min_seconds=0.6),
            clock=lambda: clock[0],
        )
        for index, frame in enumerate(fx.document_frames(fx.TRANSFORMER_PAPER, 8)):
            clock[0] += 0.3
            engine.observe(frame, received_at=clock[0], source_seq=index)
        engine.flush()

        title = store.read_all()[0].title

        assert title is not None
        assert len(title.split()) >= 4, title
