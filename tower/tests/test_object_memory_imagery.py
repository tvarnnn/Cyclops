"""Serving the picture behind a record, and refusing to when it cannot.

This is the first surface in this cartridge that serves PIXELS. Until now
it exposed a pointer -- "capture 22e9d428..., frame 3410" -- which is a
correct thing to show a developer and close to useless to a wearer.

Three properties matter more than the rest, and each one has a test that
fails loudly without it:

  1. NO PATH SERVES AN UNFILTERED FRAME. A Tower whose face-detection
     weights are missing refuses. There is no lenient default, because a
     lenient default here means a raw first-person frame on a LAN-local
     origin.
  2. EXPIRED IMAGERY IS A TRUTHFUL ANSWER, not an empty one. Capture-side
     retention is not this cartridge's to set, so a record outliving its
     picture is the ORDINARY case, and "the memory is kept and the
     picture is not" is the sentence it has to be able to produce.
  3. THE LABEL DOES NOT OVERCLAIM. This runs on read; the stored frame is
     unchanged; YuNet has measured blind spots. The payload says
     `display-filter/...` and never "redacted" or "privacy-safe".
"""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tower.object_memory.imagery import (
    FILTER_UNAVAILABLE,
    IMAGERY_EXPIRED,
    NO_CAPTURE_ROOT,
    NOT_FOUND,
    FaceFilter,
)
from tower.object_memory.records import ObjectObservation, observation_id_for
from tower.object_memory.store import ObservationStore
from tower.results.object_memory import build_face_filter
from tower.routes import observations as observations_route
from tower.confidence import Confidence

WIDTH, HEIGHT = 360, 640
CAPTURE_ID = "cap-1"
RELPATH = "frames/00000042.jpg"


def _image(colour=120):
    return np.full((HEIGHT, WIDTH, 3), colour, np.uint8)


def _observation(**kwargs):
    fields = dict(
        object_class="laptop",
        detector_score=0.81,
        confidence=Confidence.HIGH,
        observed_at=1000.0,
        time_basis="tower-receipt",
        recorded_at=1000.0,
        source="glasses-camera",
        module_id="object-memory",
        session_id=CAPTURE_ID,
        frame_seq=40,
        bounding_box=(0.1, 0.1, 0.5, 0.5),
        retention_tag="default",
        privacy_tags=("derived-only", "frame-referenced"),
        spatial_ref=None,
        external_refs=(),
        best_score=0.95,
        last_seen_at=1004.0,
        frame_count=29,
        best_frame_seq=42,
        best_relpath=RELPATH,
        best_bounding_box=(0.2, 0.2, 0.6, 0.6),
        tier="remembered",
        verification=None,
    )
    fields.update(kwargs)
    return ObjectObservation(**fields)


@pytest.fixture
def world(tmp_path):
    """A capture with one frame, and a store with one record pointing at it."""
    capture_root = tmp_path / "data"
    frames = capture_root / "captures" / CAPTURE_ID / "frames"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "00000042.jpg"), _image())

    store_root = tmp_path / "memory"
    store = ObservationStore(store_root, retention_seconds=None)
    observation = _observation()
    store.append(observation)
    return capture_root, store_root, observation.observation_id


def _client(world, monkeypatch, *, capture_root=True, face_filter=None):
    """An app wired at the environment, through monkeypatch.

    `os.environ` directly would leak a capture root into every later
    test in the session, which is exactly the kind of cross-test coupling
    that produces a suite that passes alone and fails in order.
    """
    from tower.main import create_app

    captures, store_root, _ = world
    monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(store_root))
    if capture_root:
        monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(captures))
    else:
        monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
    app = create_app()
    if face_filter is not None:
        app.state.object_memory_face_filter = face_filter
    return TestClient(app)


# -- the picture -------------------------------------------------------


class TestServingAPicture:
    def test_the_frame_route_returns_a_jpeg(self, world, monkeypatch):
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert cv2.imdecode(
            np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
        ) is not None

    def test_the_crop_route_returns_a_smaller_picture_than_the_frame(
        self, world, monkeypatch
    ):
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        frame = client.get(f"/object-memory/observations/{observation_id}/frame")
        crop = client.get(f"/object-memory/observations/{observation_id}/crop")

        decoded_frame = cv2.imdecode(
            np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR
        )
        decoded_crop = cv2.imdecode(
            np.frombuffer(crop.content, np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded_crop.shape[0] < decoded_frame.shape[0]
        assert decoded_crop.shape[1] < decoded_frame.shape[1]

    def test_imagery_is_never_cached(self, world, monkeypatch):
        """Sensitive first-person imagery on a LAN-local origin.

        A proxy or a browser holding a copy is a second store nobody
        chose, governed by nobody's retention.
        """
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        for route in ("frame", "crop"):
            response = client.get(
                f"/object-memory/observations/{observation_id}/{route}"
            )
            assert response.headers["cache-control"] == "no-store", route

    def test_the_crop_uses_the_strongest_looks_box(self, world, monkeypatch):
        """`bounding_box` describes the first frame and `best_bounding_box`
        the strongest, and the two go with different frames.

        Cropping the strongest frame with the first frame's box would cut
        the wrong part of the right picture.
        """
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        view = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert view["bounding_box_normalized"] == [0.2, 0.2, 0.6, 0.6]


# -- what the payload may claim ----------------------------------------


class TestWhatItClaims:
    def test_the_filter_is_named_as_a_display_filter(self, world, monkeypatch):
        """Not "redacted", not "anonymised", not "privacy-safe".

        This runs on read. The stored frame is unchanged, and YuNet has
        measured blind spots -- a face occluded past about 60%, a face
        rotated about 90 degrees in plane. The label says what RAN.
        """
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        view = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert view["filter"].startswith("display-filter/")
        assert "yunet" in view["filter"]
        assert view["filter_means"] == (
            "applied-on-read-the-stored-frame-is-unchanged"
        )
        rendered = json.dumps(view).lower()
        for forbidden in ("anonymis", "anonymiz", "privacy-safe", "faces removed"):
            assert forbidden not in rendered, forbidden

    def test_zero_regions_filled_is_not_a_claim_that_there_were_no_faces(self, world, monkeypatch):
        """The count is what the DETECTOR found, and it says so by being a
        count rather than a boolean."""
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        view = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert view["regions_filled"] == 0
        assert "faces_present" not in view

    def test_the_payload_keeps_imagery_retention_capture_side(self, world, monkeypatch):
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        view = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert view["imagery_retention"] == "capture-side"

    def test_the_listing_advertises_the_routes_and_the_handle(self, world, monkeypatch):
        """A client cannot build a URL without an id, and had none."""
        client = _client(world, monkeypatch)

        body = client.get("/object-memory/observations").json()

        assert body["imagery"]["view"].endswith("/imagery")
        assert body["observations"][0]["observation_id"]


# -- refusing ----------------------------------------------------------


class TestRefusals:
    def test_a_missing_frame_says_the_memory_is_kept(self, world, monkeypatch):
        """The case the whole shape exists for.

        Capture-side retention is not this cartridge's to set, so a
        record outliving its picture is ordinary. An empty response would
        be indistinguishable from a broken Tower.
        """
        captures, store_root, observation_id = world
        (captures / "captures" / CAPTURE_ID / "frames" / "00000042.jpg").unlink()
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 410
        detail = response.json()["detail"]
        assert detail["reason"] == IMAGERY_EXPIRED
        assert detail["memory_retained"] is True
        assert detail["available"] is False

    def test_the_record_itself_survives_its_imagery(self, world, monkeypatch):
        captures, _, observation_id = world
        (captures / "captures" / CAPTURE_ID / "frames" / "00000042.jpg").unlink()
        client = _client(world, monkeypatch)

        body = client.get("/object-memory/observations").json()

        assert body["observation_count"] == 1

    def test_a_tower_with_no_face_model_serves_nothing(self, world, monkeypatch):
        """No lenient default. A missing model means a refusal, never a
        raw first-person frame with an apologetic header on it."""
        _, _, observation_id = world
        client = _client(world, monkeypatch, face_filter=FaceFilter(path="no-such-model.onnx"))

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 503
        assert response.json()["detail"]["reason"] == FILTER_UNAVAILABLE

    def test_a_filter_that_raises_serves_nothing(self, world, monkeypatch):
        """A filter that failed has said nothing about this frame, and an
        unfiltered frame is not the fallback."""

        class ExplodingFilter:
            available = True
            label = "display-filter/exploding"

            def apply(self, image):
                raise RuntimeError("no")

        _, _, observation_id = world
        client = _client(world, monkeypatch, face_filter=ExplodingFilter())

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 503
        assert response.json()["detail"]["reason"] == FILTER_UNAVAILABLE

    def test_an_app_built_without_a_filter_refuses_rather_than_raising(self, world, monkeypatch):
        """Most of this repository's tests build an app and set nothing.

        The failure has to be a refusal, never an AttributeError, and
        certainly never a raw frame.
        """
        _, _, observation_id = world
        client = _client(world, monkeypatch)
        client.app.state.object_memory_face_filter = None

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 503

    def test_a_tower_with_no_capture_root_has_nowhere_to_look(self, world, monkeypatch):
        _, _, observation_id = world
        client = _client(world, monkeypatch, capture_root=False)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 503
        assert response.json()["detail"]["reason"] == NO_CAPTURE_ROOT

    def test_an_unknown_handle_is_a_404(self, world, monkeypatch):
        client = _client(world, monkeypatch)

        response = client.get(
            "/object-memory/observations/0000000000000000/imagery"
        )

        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == NOT_FOUND

    def test_an_expired_record_is_unreachable_by_its_handle(
        self, tmp_path, monkeypatch
    ):
        """Retention is not bypassed by knowing an id.

        A handle to a record the listing will not serve must not be a
        back door to its picture -- otherwise retention is a default
        rather than a promise.
        """
        captures = tmp_path / "data"
        frames = captures / "captures" / CAPTURE_ID / "frames"
        frames.mkdir(parents=True)
        cv2.imwrite(str(frames / "00000042.jpg"), _image())

        store_root = tmp_path / "memory"
        store = ObservationStore(store_root, retention_seconds=1.0)
        old = _observation(recorded_at=1.0, observed_at=1.0)
        store.append(old)

        monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(store_root))
        monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(captures))
        from tower.main import create_app

        client = TestClient(create_app())

        handle = observation_id_for(CAPTURE_ID, "laptop", 1.0)
        response = client.get(
            f"/object-memory/observations/{handle}/frame"
        )

        assert response.status_code == 404


# -- the filter itself -------------------------------------------------


class TestTheFilter:
    def test_it_fills_a_real_face(self):
        """Not a synthetic rectangle: an actual photograph of a person,
        resized to the resolution the glasses actually deliver."""
        from skimage import data

        image = cv2.resize(
            cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (WIDTH, HEIGHT)
        )
        before = image.copy()

        filtered, filled = build_face_filter().apply(image)

        assert len(filled) == 1
        assert not np.array_equal(before, filtered)

    def test_it_fills_nothing_in_a_flat_frame(self):
        _, filled = build_face_filter().apply(_image())

        assert filled == []

    def test_it_reports_where_it_filled_not_only_how_often(self):
        """A count cannot say that the fill landed on the subject.

        On frame 2708 of the validated capture -- a desk with no person
        in it -- this filter fired twice and one fill covered the mouse a
        record was about. The crop served for that record is a black
        rectangle, and the payload has to be able to say so.
        """
        from skimage import data

        image = cv2.resize(
            cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (WIDTH, HEIGHT)
        )

        _, filled = build_face_filter().apply(image)

        (x0, y0, x1, y1), = filled
        assert x1 > x0 and y1 > y0

    def test_a_filter_with_no_weights_reports_why(self):
        face_filter = FaceFilter(path="no-such-model.onnx")

        assert face_filter.available is False
        assert "face-detection model" in face_filter.unavailable_reason
        assert face_filter.label.endswith("/none")

    def test_the_route_module_reaches_only_its_adapter(self):
        """Asserted here as well as in the boundary suite, because this
        file is where the temptation lives: the reason constants are
        cartridge-owned and the route needs them."""
        import ast
        import pathlib

        source = pathlib.Path(observations_route.__file__).read_text(
            encoding="utf-8"
        )
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "tower.object_memory" not in node.module


class TestPathContainment:
    """A record builds a path, so a record must not be able to leave the tree.

    `session_id` and `best_relpath` both come off a JSONL record and both
    go into `Path(...)`. Nothing that writes them today can produce a
    `..`: `CaptureRecorder` mints the id and names every frame
    `frames/<seq:08d>.jpg`. But a store file is a plain text file on
    disk, and an operator restoring one from a backup, a future producer,
    or a merge of two stores could all introduce a path this route would
    otherwise follow out of the capture tree and serve over HTTP.
    """

    def _secret(self, tmp_path):
        secret = tmp_path / "secret.jpg"
        cv2.imwrite(str(secret), np.full((32, 32, 3), 7, np.uint8))
        return secret

    def test_a_relpath_that_climbs_out_resolves_to_nothing(
        self, world, tmp_path
    ):
        """Asserted against `frame_path` rather than the route.

        The route would fall back to the frame-sequence convention and
        serve the RIGHT frame, which is correct behaviour and hides the
        thing under test. What matters is that the escaping path itself
        never resolves.
        """
        from tower.object_memory.imagery import frame_path

        captures, _, _ = world
        self._secret(tmp_path)
        escaping = _observation(
            best_relpath="../../../secret.jpg",
            best_frame_seq=None,
            frame_seq=None,
        )

        assert frame_path(captures, escaping) is None

    def test_a_session_id_that_climbs_out_serves_nothing(
        self, world, monkeypatch, tmp_path
    ):
        _, store_root, _ = world
        secret = self._secret(tmp_path)
        store = ObservationStore(store_root, retention_seconds=None)
        escaping = _observation(
            observed_at=3000.0,
            session_id="../..",
            best_relpath="secret.jpg",
            best_frame_seq=None,
            frame_seq=None,
        )
        store.append(escaping)
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{escaping.observation_id}/frame"
        )

        assert response.status_code in (404, 410), response.status_code
        assert response.content != secret.read_bytes()

    def test_an_absolute_relpath_serves_nothing(self, world, tmp_path):
        """`Path("a") / "/etc/passwd"` is `/etc/passwd`. Pathlib joins that way."""
        from tower.object_memory.imagery import frame_path

        captures, _, _ = world
        secret = self._secret(tmp_path)
        escaping = _observation(
            best_relpath=str(secret),
            best_frame_seq=None,
            frame_seq=None,
        )

        assert frame_path(captures, escaping) is None

    def test_the_containment_check_still_admits_a_real_frame(self, world):
        """A guard that refuses everything guards nothing."""
        from tower.object_memory.imagery import frame_path

        captures, store_root, observation_id = world
        store = ObservationStore(store_root, retention_seconds=None)
        (observation,) = store.all_observations()

        assert frame_path(captures, observation) is not None


class TestTheFilterIsBounded:
    """The lock is shared, so the work under it must not be unbounded.

    `UPSCALE = 2` was measured at 640x360. Applied blindly it is
    quadratic: a reviewer measured a 4000x4000 frame holding the shared
    lock for 2.18 seconds while every other request for a picture waited.
    Raising capture resolution is the one change this cartridge's own
    roadmap entry recommends, so this is not hypothetical.
    """

    def test_the_corpus_resolution_is_upscaled_exactly_as_before(self):
        """Nothing measured on 360x640 may move."""
        from tower.object_memory.imagery import TARGET_LONG_SIDE, UPSCALE

        assert TARGET_LONG_SIDE / 640 == UPSCALE

    def test_a_frame_larger_than_the_target_is_scaled_down(self):
        """Asserted on the decision, not on a stopwatch.

        A timing assertion on a machine that carries several agent lanes
        would be a flake generator, and the invariant is not "it is fast"
        -- it is "the detector is never handed more than the target".
        """
        from tower.object_memory.imagery import TARGET_LONG_SIDE, detector_scale

        for height, width in ((4000, 4000), (2400, 1800), (1440, 2560)):
            scale = detector_scale(height, width)
            assert max(height, width) * scale <= TARGET_LONG_SIDE + 1, (
                height,
                width,
                scale,
            )

    def test_the_corpus_frame_is_still_upscaled_by_exactly_two(self):
        from tower.object_memory.imagery import UPSCALE, detector_scale

        assert detector_scale(640, 360) == UPSCALE

    def test_it_still_finds_a_face_in_a_large_frame(self):
        """A bound that stopped the filter working would be worse than
        the unbounded version."""
        from skimage import data

        large = cv2.resize(
            cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (1600, 1600)
        )

        _, filled = build_face_filter().apply(large)

        # At least one, not exactly one. Above the target long side the
        # upscale is 1.0 rather than 2, and YuNet at native scale finds
        # extra regions in this image -- which is the filter erring
        # towards covering more, the direction it should err in.
        assert len(filled) >= 1


# -- the reconnect lineage ---------------------------------------------


def _capture_manifest(directory, capture_id, *, continues=None, end_reason=None):
    """The half of a capture manifest this resolution actually reads."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capture.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capture_id": capture_id,
                "started_at": 1000.0,
                "ended_at": 1100.0,
                "end_reason": end_reason,
                "continues_capture": continues,
            }
        ),
        encoding="utf-8",
    )


class TestTheReconnectLineage:
    """A record's frame may be in the capture AFTER the one it names.

    This is not hypothetical and it is not rare. iOS reconnects about half
    a second after a WiFi hiccup, the recorder mints a new capture
    directory naming the old one in `continues_capture`, and
    `CaptureFollower` walks the producer into it. The producer's
    `session_id` was fixed at spawn and does not move, so a sighting that
    opened before the hiccup and peaked after it names the OLD capture and
    the NEW capture's frame.

    Measured on the development host's own store on 2026-08-29: 74
    records, 70 resolved, 4 did not -- and all 4 of those frames were on
    disk in the successor directory. Nothing had expired; this repository
    has no capture pruner at all. The route was answering "the memory is
    kept, the picture is gone" about pictures that were one directory
    away.
    """

    def _split_walk(self, world):
        """Move the frame into a successor capture, as a reconnect does."""
        captures, store_root, observation_id = world
        root = captures / "captures"
        _capture_manifest(root / CAPTURE_ID, CAPTURE_ID, end_reason="disconnect")

        successor = root / "cap-2"
        (successor / "frames").mkdir(parents=True)
        (root / CAPTURE_ID / "frames" / "00000042.jpg").rename(
            successor / "frames" / "00000042.jpg"
        )
        _capture_manifest(successor, "cap-2", continues=CAPTURE_ID)
        return observation_id

    def test_a_frame_in_the_successor_capture_is_still_served(
        self, world, monkeypatch
    ):
        observation_id = self._split_walk(world)
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0

    def test_the_crop_route_follows_the_lineage_too(self, world, monkeypatch):
        observation_id = self._split_walk(world)
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/crop"
        )

        assert response.status_code == 200, response.text

    def test_a_chain_of_reconnects_is_followed_to_the_end(
        self, world, monkeypatch
    ):
        """TWENTY-FIVE, and the number is not arbitrary.

        This case first used five, and the implementation carried a
        matching `MAX_LINEAGE_STEPS = 8` justified as "chains of five are
        in this corpus". Both were guesses. Measured against the
        development host on 2026-08-29: 32 lineage roots, and one walk
        **eighteen captures long** -- one sitting, seventeen reconnects,
        source frames 1 to 8,690. Four records in that walk were still
        answering 410 about frames sitting on disk, truncated by the
        bound that was supposed to protect the walk.

        The fix was to delete the constant rather than raise it: `seen`
        is what terminates the walk, and it cannot run longer than there
        are captures. Twenty-five is comfortably past the longest thing
        anyone has recorded and still finishes instantly.
        """
        captures, _, observation_id = world
        root = captures / "captures"
        _capture_manifest(root / CAPTURE_ID, CAPTURE_ID, end_reason="disconnect")

        previous = CAPTURE_ID
        for index in range(2, 26):
            name = f"cap-{index}"
            _capture_manifest(root / name, name, continues=previous)
            (root / name / "frames").mkdir(parents=True)
            previous = name

        (root / CAPTURE_ID / "frames" / "00000042.jpg").rename(
            root / previous / "frames" / "00000042.jpg"
        )
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 200, response.text

    def test_a_manifest_that_names_itself_does_not_hang(
        self, world, monkeypatch
    ):
        """A `continues_capture` pointing at its own capture is not a shape
        the recorder can produce. It is a shape a restore, a merge or a
        future writer can, and an unbounded walk over it would be an
        infinite loop inside an HTTP handler."""
        captures, _, observation_id = world
        root = captures / "captures"
        (root / CAPTURE_ID / "frames" / "00000042.jpg").unlink()
        _capture_manifest(root / CAPTURE_ID, CAPTURE_ID, continues=CAPTURE_ID)
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 410

    def test_two_captures_naming_one_predecessor_do_not_hang(
        self, world, monkeypatch
    ):
        captures, _, observation_id = world
        root = captures / "captures"
        (root / CAPTURE_ID / "frames" / "00000042.jpg").unlink()
        _capture_manifest(root / CAPTURE_ID, CAPTURE_ID, end_reason="disconnect")
        for name in ("cap-2", "cap-3"):
            _capture_manifest(root / name, name, continues=CAPTURE_ID)
            (root / name / "frames").mkdir(parents=True)
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 410

    def test_the_lineage_cannot_be_used_to_leave_the_capture_root(
        self, world, monkeypatch
    ):
        """`continues_capture` is read off a file on disk and is used to
        BUILD A PATH. `_contained` is what stops it, and it is asserted
        here rather than assumed because this is a second, newer way for
        an attacker-controlled string to reach path construction."""
        captures, store_root, _ = world
        root = captures / "captures"
        (root / CAPTURE_ID / "frames" / "00000042.jpg").unlink()
        _capture_manifest(root / CAPTURE_ID, CAPTURE_ID, end_reason="disconnect")

        outside = captures.parent / "secret.jpg"
        cv2.imwrite(str(outside), _image())
        escaping = root / "cap-escape"
        _capture_manifest(escaping, "cap-escape", continues=CAPTURE_ID)

        store = ObservationStore(store_root, retention_seconds=None)
        record = _observation(best_relpath="../../secret.jpg", best_frame_seq=None)
        store.append(record)
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{record.observation_id}/frame"
        )

        assert response.status_code == 410

    def test_a_walk_whose_directories_are_all_gone_still_reports_expiry(
        self, world, monkeypatch
    ):
        """The honest 410. The reason a client sees does not change --
        there is nothing a wearer can do differently about the two -- but
        `capture_lineage_present` must say which one happened."""
        from tower.object_memory.imagery import capture_lineage_present

        captures, store_root, observation_id = world
        root = captures / "captures"
        record = _observation()
        assert capture_lineage_present(captures, record) is True

        for path in sorted(
            (root / CAPTURE_ID).rglob("*"), key=lambda p: len(p.parts), reverse=True
        ):
            path.rmdir() if path.is_dir() else path.unlink()
        (root / CAPTURE_ID).rmdir()

        assert capture_lineage_present(captures, record) is False

        client = _client(world, monkeypatch)
        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )
        assert response.status_code == 410
        assert response.json()["detail"]["reason"] == IMAGERY_EXPIRED

    def test_an_intact_capture_missing_one_frame_is_a_pointer_defect(
        self, world, monkeypatch
    ):
        """The shape that was misreported: the recording is right there
        and the frame it names is not."""
        from tower.object_memory.imagery import capture_lineage_present

        captures, _, observation_id = world
        (captures / "captures" / CAPTURE_ID / "frames" / "00000042.jpg").unlink()

        assert capture_lineage_present(captures, _observation()) is True

        client = _client(world, monkeypatch)
        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )
        assert response.status_code == 410

    def test_an_uninterrupted_capture_never_scans_the_captures_root(
        self, world, monkeypatch
    ):
        """The ordinary case must not pay for the rare one. `data/captures`
        is never pruned in this repository, so a directory listing per
        picture would grow without bound for the life of the host."""
        from tower.object_memory import imagery as imagery_module

        calls = []
        original = imagery_module._successors

        def counting(root):
            calls.append(root)
            return original(root)

        monkeypatch.setattr(imagery_module, "_successors", counting)
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        response = client.get(
            f"/object-memory/observations/{observation_id}/frame"
        )

        assert response.status_code == 200
        assert calls == []
