"""The one picture Object Memory owns, and everything that must refuse.

WHY THIS FILE EXISTS AT ALL, which is the first test in it.

A record has 30-day retention. The picture it points at lives in
`data/captures/<session_id>/frames/`, owned by capture-side lifecycle,
and every record has always said so -- `privacy_tags: ["derived-only",
"frame-referenced"]`, `imagery_retention: "capture-side"`. Nothing prunes
captures today, so nothing has gone wrong yet; the first thing that does
takes the picture out of every memory at once. A durable record pointing
into an ephemeral store is a race, not a retention policy.

`test_a_crop_survives_deleting_the_whole_capture_tree` is the test that
proves the change did anything, and it was written first.

The rest of the file is about the cost of having done it, because this
is the first place in this cartridge where a pixel reaches disk:

  * FAIL CLOSED. No model, a filter that raises, an encode that fails, a
    write that fails -- all four leave the directory exactly as they
    found it. There is no branch that writes an unfiltered crop, and a
    partial write is never served.
  * THE SIDECAR IS REQUIRED. A `.jpg` with nothing beside it is not
    evidence that a filter ran on it, so it is ignored rather than
    served.
  * RETENTION REALLY GOVERNS IT. Prune deletes the pictures of expired
    records; purge deletes all of them and REPORTS what it could not.
  * AN ID CANNOT ESCAPE. `observation_id` reaches the store as a string
    off a JSONL file and is used to build a path.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tower.confidence import Confidence
from tower.object_memory import keyframes as keyframes_module
from tower.object_memory.detector import Detection, FixedDetector
from tower.object_memory.engine import ObjectMemoryEngine
from tower.object_memory.imagery import (
    IMAGERY_EXPIRED,
    SOURCE_CAPTURE,
    SOURCE_KEYFRAME,
    FaceFilter,
    render,
)
from tower.object_memory.keyframes import (
    FILTER_FAILED,
    FILTER_UNAVAILABLE,
    MAX_LONG_SIDE,
    NO_IMAGERY,
    SCHEMA_VERSION,
    UNUSABLE_ID,
    KeyframeStore,
)
from tower.object_memory.records import ObjectObservation, observation_id_for
from tower.object_memory.relevance import RelevancePolicy
from tower.object_memory.store import ObservationStore
from tower.results.object_memory import (
    IMAGERY_CONTRACT,
    build_imagery_view,
)

WIDTH, HEIGHT = 360, 640
CAPTURE_ID = "cap-1"
RELPATH = "frames/00000042.jpg"
# What `records.observation_id_for` actually produces: 16 lowercase hex
# characters. Spelled out here so the containment cases below are
# obviously comparing against a real one rather than a plausible one.
GOOD_ID = "0123456789abcdef"


def _image(colour=120, height=HEIGHT, width=WIDTH):
    return np.full((height, width, 3), colour, np.uint8)


class _StubFilter:
    """A filter that fills exactly what it is told to, and nothing else.

    Used instead of the real YuNet everywhere the test is about the
    STORE rather than about detection. A test that asserted "the model
    found a face" while meaning "the store wrote a file" would fail for
    the wrong reason on a host whose weights are missing, and would pass
    while writing unfiltered bytes on a host whose weights are present.
    """

    label = "display-filter/stub@0.30"

    def __init__(self, fills=(), *, available=True, raises=False):
        self.available = available
        self._fills = [tuple(fill) for fill in fills]
        self._raises = raises
        self.calls = 0

    def apply(self, image):
        self.calls += 1
        if self._raises:
            raise RuntimeError("the filter failed")
        for x0, y0, x1, y1 in self._fills:
            image[y0:y1, x0:x1] = 0
        return image, list(self._fills)


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
    """A capture with one frame, and a store with one record naming it."""
    capture_root = tmp_path / "data"
    frames = capture_root / "captures" / CAPTURE_ID / "frames"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "00000042.jpg"), _image())

    store_root = tmp_path / "memory"
    store = ObservationStore(store_root, retention_seconds=None)
    observation = _observation()
    store.append(observation)
    return capture_root, store_root, observation.observation_id


def _client(world, monkeypatch, *, capture_root=True):
    """An app wired at the environment, exactly as the imagery suite does."""
    from tower.main import create_app

    captures, store_root, _ = world
    monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(store_root))
    if capture_root:
        monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(captures))
    else:
        monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
    return TestClient(create_app())


def _delete_tree(root: Path) -> None:
    """rm -rf, so a test can say "the recording is gone" and mean it."""
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    root.rmdir()


def _write_keyframe(store_root, observation_id, *, crop=None, fills=()):
    keyframes = KeyframeStore(store_root)
    result = keyframes.write(
        observation_id,
        _image(200, height=120, width=90) if crop is None else crop,
        _StubFilter(fills),
        source_capture=CAPTURE_ID,
        source_relpath=RELPATH,
        written_at=1005.0,
    )
    assert result.written, result.reason
    return keyframes, result


# -- the point of the whole change -------------------------------------


class TestTheKeyframeOutlivesTheCapture:
    """The test that proves the change did anything. Written first.

    Before this, deleting `data/captures/` took the picture out of every
    memory in the store at once. The records survived and were useless:
    a class name, a timestamp, and a 410 where the image had been.
    """

    def test_a_crop_survives_deleting_the_whole_capture_tree(
        self, world, monkeypatch
    ):
        captures, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch)

        _delete_tree(captures / "captures")

        response = client.get(f"/object-memory/observations/{observation_id}/crop")

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/jpeg"
        assert (
            cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            is not None
        )

    def test_the_record_still_lists_after_its_recording_is_gone(
        self, world, monkeypatch
    ):
        """The half that already worked, asserted beside the half that did
        not, so a regression in either is visible here."""
        captures, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch)
        _delete_tree(captures / "captures")

        body = client.get("/object-memory/observations").json()

        assert body["observation_count"] == 1

    def test_the_whole_frame_is_still_gone_when_the_capture_is(
        self, world, monkeypatch
    ):
        """A keyframe is a CROP. It is not a frame, and it must not be
        served as one.

        The full-frame view is the context -- where the wearer was and
        what else was around -- and there is nothing to synthesise that
        out of. When the recording is deleted, 410 is the honest answer
        and stays the honest answer.
        """
        captures, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch)
        _delete_tree(captures / "captures")

        response = client.get(f"/object-memory/observations/{observation_id}/frame")

        assert response.status_code == 410
        assert response.json()["detail"]["reason"] == IMAGERY_EXPIRED

    def test_the_owned_crop_does_not_need_a_capture_root_at_all(
        self, world, monkeypatch
    ):
        """A Tower with no capture root configured has nowhere to look for
        a frame, and still owns its keyframes."""
        _, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch, capture_root=False)

        assert (
            client.get(
                f"/object-memory/observations/{observation_id}/crop"
            ).status_code
            == 200
        )

    def test_a_record_with_no_keyframe_still_crops_the_capture_frame(
        self, world, monkeypatch
    ):
        """Every record written before this existed has no keyframe.

        None of them may lose the picture they had. The fallback is the
        behaviour that shipped: crop the capture frame, filtered on read.
        """
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        assert (
            client.get(
                f"/object-memory/observations/{observation_id}/crop"
            ).status_code
            == 200
        )


# -- fail closed -------------------------------------------------------


class TestFailClosed:
    """No configuration, and no failure, writes an unfiltered crop.

    This is the property the whole module is arranged around, and it is
    worth more here than in `imagery.render`: a byte served wrongly is
    one response, and a byte written wrongly is thirty days on disk, in
    a backup, past every label that travelled with it.
    """

    def _files(self, store_root):
        directory = store_root / "keyframes"
        return sorted(p.name for p in directory.iterdir()) if directory.exists() else []

    def test_a_tower_with_no_face_model_writes_nothing(self, tmp_path):
        keyframes = KeyframeStore(tmp_path)

        result = keyframes.write(
            GOOD_ID, _image(), FaceFilter(path="no-such-model.onnx")
        )

        assert result.written is False
        assert result.reason == FILTER_UNAVAILABLE
        assert self._files(tmp_path) == []

    def test_no_filter_at_all_writes_nothing(self, tmp_path):
        """`None` is not "skip the filter". It is the same refusal."""
        result = KeyframeStore(tmp_path).write(GOOD_ID, _image(), None)

        assert result.reason == FILTER_UNAVAILABLE
        assert self._files(tmp_path) == []

    def test_a_filter_that_raises_writes_nothing(self, tmp_path):
        """The case that would have to fail OPEN for this to be dangerous.

        A filter that raised has said nothing about these pixels. If any
        branch fell through to writing the input crop, this is where it
        would show.
        """
        result = KeyframeStore(tmp_path).write(
            GOOD_ID, _image(), _StubFilter(raises=True)
        )

        assert result.written is False
        assert result.reason == FILTER_FAILED
        assert self._files(tmp_path) == []

    def test_an_encode_failure_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, None))

        result = KeyframeStore(tmp_path).write(GOOD_ID, _image(), _StubFilter())

        assert result.written is False
        assert self._files(tmp_path) == []

    def test_a_sidecar_that_cannot_be_written_takes_the_image_with_it(
        self, tmp_path, monkeypatch
    ):
        """An image with no sidecar would be ignored by `read` forever and
        carried by `prune` for thirty days. Undo the half that worked."""
        original = Path.write_text

        def refuse(self, *args, **kwargs):
            if self.suffix == ".json":
                raise OSError("no")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", refuse)

        result = KeyframeStore(tmp_path).write(GOOD_ID, _image(), _StubFilter())

        assert result.written is False
        assert self._files(tmp_path) == []

    def test_a_sighting_that_held_no_crop_is_not_an_error(self, tmp_path):
        """A zero-area box holds nothing to write. Counted, not logged as
        a failure, and certainly not written as an empty file."""
        result = KeyframeStore(tmp_path).write(None, None, _StubFilter())

        assert result.reason == UNUSABLE_ID

        result = KeyframeStore(tmp_path).write(GOOD_ID, None, _StubFilter())

        assert result.reason == NO_IMAGERY
        assert self._files(tmp_path) == []

    def test_nothing_is_served_after_a_refused_write(self, world, monkeypatch):
        """A refusal must leave the record where it started -- served from
        the capture, or 410 -- and never half-served from a partial file."""
        captures, store_root, observation_id = world
        refused = KeyframeStore(store_root).write(
            observation_id, _image(), _StubFilter(raises=True)
        )
        assert refused.written is False

        assert KeyframeStore(store_root).read(observation_id) is None

        client = _client(world, monkeypatch)
        _delete_tree(captures / "captures")

        assert (
            client.get(
                f"/object-memory/observations/{observation_id}/crop"
            ).status_code
            == 410
        )

    def test_the_bytes_written_are_the_filters_output_not_its_input(
        self, tmp_path
    ):
        """Asserted on the PIXELS, not on the code path.

        The crop is a uniform bright image and the filter blacks out its
        whole area. If the input had reached `imencode` the decoded
        keyframe would be bright; it is black.
        """
        keyframes = KeyframeStore(tmp_path)
        crop = _image(240, height=120, width=90)

        keyframes.write(
            GOOD_ID, crop, _StubFilter(fills=[(0, 0, 90, 120)])
        )

        image_bytes, _ = keyframes.read(GOOD_ID)
        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.max() < 16, decoded.max()


# -- the sidecar -------------------------------------------------------


class TestTheSidecar:
    def test_it_records_what_ran_and_where_it_came_from(self, tmp_path):
        keyframes, result = _write_keyframe(tmp_path, GOOD_ID, fills=[(0, 0, 20, 20)])

        _, sidecar = keyframes.read(GOOD_ID)

        assert sidecar["schema_version"] == SCHEMA_VERSION
        assert sidecar["filter_label"] == "display-filter/stub@0.30"
        assert sidecar["regions_filled"] == 1
        assert sidecar["subject_obscured"] > 0.0
        assert sidecar["written_at"] == 1005.0
        assert sidecar["source_capture"] == CAPTURE_ID
        assert sidecar["source_relpath"] == RELPATH
        assert result.bytes_written > 0

    def test_it_never_claims_the_picture_is_safe(self, tmp_path):
        """The same wording rule `imagery.py` holds itself to. YuNet has
        measured blind spots; the label names the detector and its
        threshold and asserts no outcome."""
        keyframes, _ = _write_keyframe(tmp_path, GOOD_ID)

        _, sidecar = keyframes.read(GOOD_ID)

        rendered = json.dumps(sidecar).lower()
        for forbidden in (
            "redact",
            "anonymis",
            "anonymiz",
            "privacy-safe",
            "faces removed",
        ):
            assert forbidden not in rendered, forbidden

    def test_an_image_with_no_sidecar_is_not_served(self, tmp_path):
        """A `.jpg` alone is not evidence that a filter ran on it.

        It could have come from a backup taken before this existed, been
        copied in by hand, or been left by a writer that did not exist
        when it was made. Serving it would be serving an unfiltered
        first-person crop on the strength of its filename.
        """
        keyframes, result = _write_keyframe(tmp_path, GOOD_ID)
        result.path.with_suffix(".json").unlink()

        assert result.path.exists()
        assert keyframes.read(GOOD_ID) is None

    def test_an_unreadable_sidecar_is_not_served_either(self, tmp_path):
        keyframes, result = _write_keyframe(tmp_path, GOOD_ID)
        result.path.with_suffix(".json").write_text("{not json", encoding="utf-8")

        assert keyframes.read(GOOD_ID) is None

    def test_a_sidecarless_image_is_not_served_over_http_either(
        self, world, monkeypatch
    ):
        captures, store_root, observation_id = world
        _, result = _write_keyframe(store_root, observation_id)
        result.path.with_suffix(".json").unlink()
        client = _client(world, monkeypatch)
        _delete_tree(captures / "captures")

        assert (
            client.get(
                f"/object-memory/observations/{observation_id}/crop"
            ).status_code
            == 410
        )


# -- how big it is -----------------------------------------------------


class TestTheSizeBound:
    """A picture kept for thirty days has to have a ceiling.

    384 px on the long side at JPEG quality 80 measured 11.7 KB mean over
    all 116 of this host's records -- about 4.3 MB an hour of walking against
    the ~2.1 GB an hour the recording costs.
    """

    def _decoded(self, tmp_path, crop):
        keyframes = KeyframeStore(tmp_path)
        assert keyframes.write(GOOD_ID, crop, _StubFilter()).written
        image_bytes, _ = keyframes.read(GOOD_ID)
        return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    def test_a_large_crop_is_downscaled(self, tmp_path):
        decoded = self._decoded(tmp_path, _image(height=1500, width=1000))

        assert max(decoded.shape[:2]) == MAX_LONG_SIDE
        # Aspect ratio preserved, not squashed into a square.
        assert decoded.shape[0] > decoded.shape[1]

    def test_a_small_crop_is_never_upscaled(self, tmp_path):
        """Upscaling would invent detail, cost bytes, and make a
        3%-of-frame object look like evidence it is not."""
        decoded = self._decoded(tmp_path, _image(height=90, width=60))

        assert decoded.shape[:2] == (90, 60)

    def test_a_crop_exactly_at_the_bound_is_left_alone(self, tmp_path):
        decoded = self._decoded(
            tmp_path, _image(height=MAX_LONG_SIDE, width=MAX_LONG_SIDE // 2)
        )

        assert decoded.shape[:2] == (MAX_LONG_SIDE, MAX_LONG_SIDE // 2)


# -- retention really governs it ---------------------------------------


class TestRetention:
    """`imagery_retention: "object-memory"` is a promise, and this is it.

    A keyframe that outlived the record it belongs to would be
    first-person imagery under a retention nobody enforces -- exactly the
    problem this whole change exists to fix, moved one directory over.
    """

    def _store(self, tmp_path, **kwargs):
        return ObservationStore(tmp_path, **kwargs)

    def test_prune_removes_the_keyframes_of_expired_records_only(self, tmp_path):
        store = self._store(tmp_path, retention_seconds=100.0)
        fresh = _observation(observed_at=950.0, recorded_at=950.0)
        stale = _observation(observed_at=10.0, recorded_at=10.0)
        store.append(fresh)
        store.append(stale)
        keyframes = KeyframeStore(tmp_path)
        for observation in (fresh, stale):
            assert keyframes.write(
                observation.observation_id, _image(), _StubFilter()
            ).written

        removed = store.prune_expired(now=1000.0)

        assert removed == 1
        assert keyframes.read(fresh.observation_id) is not None
        assert keyframes.read(stale.observation_id) is None

    def test_prune_removes_an_orphan_no_record_claims(self, tmp_path):
        """A keep list rather than a delete list, so a picture whose record
        vanished by a route this cartridge does not model still goes."""
        store = self._store(tmp_path, retention_seconds=100.0)
        kept = _observation(observed_at=950.0, recorded_at=950.0)
        store.append(kept)
        keyframes = KeyframeStore(tmp_path)
        keyframes.write(kept.observation_id, _image(), _StubFilter())
        keyframes.write(GOOD_ID, _image(), _StubFilter())

        store.prune_expired(now=1000.0)

        assert keyframes.read(kept.observation_id) is not None
        assert keyframes.read(GOOD_ID) is None

    def test_an_expired_record_cannot_be_served_its_keyframe(
        self, tmp_path, monkeypatch
    ):
        """Retention is not bypassed by knowing an id, and the picture is
        no more reachable than the record."""
        captures = tmp_path / "data"
        (captures / "captures" / CAPTURE_ID / "frames").mkdir(parents=True)
        store_root = tmp_path / "memory"
        store = ObservationStore(store_root, retention_seconds=1.0)
        old = _observation(observed_at=1.0, recorded_at=1.0)
        store.append(old)
        _write_keyframe(store_root, old.observation_id)

        monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(store_root))
        monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(captures))
        from tower.main import create_app

        client = TestClient(create_app())

        response = client.get(
            f"/object-memory/observations/{old.observation_id}/crop"
        )

        assert response.status_code == 404

    def test_purge_deletes_every_keyframe_and_the_directory(self, tmp_path):
        store = self._store(tmp_path, retention_seconds=None)
        observation = _observation()
        store.append(observation)
        _write_keyframe(tmp_path, observation.observation_id)

        assert store.purge() == 1

        assert store.last_keyframe_purge == (1, ())
        # The same assertion the store's own purge test makes: every file
        # artifact the store owns is gone, keyframes directory included.
        assert not any(tmp_path.iterdir())

    def test_purge_reports_what_it_could_not_delete(self, tmp_path, monkeypatch):
        """A false claim of deletion is worse than an honest failure.

        On Windows an open handle is enough to make `unlink` raise, and a
        wearer asking for erasure has to be told that pictures of their
        home are still on disk rather than handed a count.
        """
        store = self._store(tmp_path, retention_seconds=None)
        observation = _observation()
        store.append(observation)
        _write_keyframe(tmp_path, observation.observation_id)
        original = Path.unlink

        def refuse(self, *args, **kwargs):
            if self.suffix == ".jpg":
                raise OSError("in use")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", refuse)

        store.purge()

        removed, retained = store.last_keyframe_purge
        assert removed == 0
        assert retained == (observation.observation_id,)

    def test_the_purge_cli_says_so_and_exits_non_zero(self, tmp_path, monkeypatch):
        import scripts.object_query as object_query

        store = ObservationStore(tmp_path, retention_seconds=None)
        observation = _observation()
        store.append(observation)
        _write_keyframe(tmp_path, observation.observation_id)
        original = Path.unlink

        def refuse(self, *args, **kwargs):
            if self.suffix == ".jpg":
                raise OSError("in use")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", refuse)

        code = object_query.main(
            ["--root", str(tmp_path), "--purge-all", "--format", "json"]
        )

        assert code == 1


# -- an id cannot escape -----------------------------------------------


class TestContainment:
    """`observation_id` arrives as a string off a JSONL file and BUILDS A PATH.

    Nothing that writes one today can produce anything but 16 lowercase
    hex characters -- `records.observation_id_for` is a blake2b digest.
    But a store file is a plain text file on disk, and a restored backup,
    a merge of two stores or a future producer could all introduce one
    this store would otherwise follow out of its own directory.
    """

    HOSTILE = (
        "..",
        "../../secret",
        "..\\..\\secret",
        "/etc/passwd",
        "C:\\windows\\system32",
        "keyframes/../../secret",
        "a/b",
        "",
        "0123456789ABCDEF",  # uppercase is not what ids look like
        "0123",  # too short to be one
        "0123456789abcdeg",  # not hex
        "0123456789abcdef.jpg",
        None,
        1234,
    )

    @pytest.mark.parametrize("observation_id", HOSTILE)
    def test_a_hostile_id_addresses_nothing(self, tmp_path, observation_id):
        assert KeyframeStore(tmp_path).path_for(observation_id) is None

    @pytest.mark.parametrize("observation_id", HOSTILE)
    def test_a_hostile_id_writes_nothing_anywhere(self, tmp_path, observation_id):
        root = tmp_path / "memory"
        root.mkdir()
        before = sorted(p.name for p in tmp_path.rglob("*"))

        result = KeyframeStore(root).write(observation_id, _image(), _StubFilter())

        assert result.written is False
        assert result.reason == UNUSABLE_ID
        assert sorted(p.name for p in tmp_path.rglob("*")) == before

    @pytest.mark.parametrize("observation_id", HOSTILE)
    def test_a_hostile_id_reads_nothing(self, tmp_path, observation_id):
        assert KeyframeStore(tmp_path).read(observation_id) is None

    def test_a_guard_that_refused_everything_would_guard_nothing(self, tmp_path):
        """A real id, produced the way the pipeline produces one."""
        real = observation_id_for(CAPTURE_ID, "laptop", 1000.0)

        path = KeyframeStore(tmp_path).path_for(real)

        assert path is not None
        assert path.parent == (tmp_path / "keyframes").resolve()

    def test_prune_ignores_a_file_it_did_not_write(self, tmp_path):
        """This directory is ours; a file we cannot explain is still not
        ours to delete."""
        keyframes = KeyframeStore(tmp_path)
        keyframes.write(GOOD_ID, _image(), _StubFilter())
        stranger = tmp_path / "keyframes" / "notes.txt"
        stranger.write_text("hello", encoding="utf-8")

        keyframes.prune(keep_ids=())

        assert keyframes.read(GOOD_ID) is None
        assert stranger.exists()


# -- the engine writes exactly one, for the classes that exist ---------


def _frame() -> bytes:
    return cv2.imencode(".jpg", _image())[1].tobytes()


def _engine(tmp_path, *, keyframes=None, face_filter=None, label="laptop"):
    store = ObservationStore(tmp_path, retention_seconds=None)
    engine = ObjectMemoryEngine(
        store,
        FixedDetector([[Detection(label=label, score=0.81, box=(36.0, 64.0, 180.0, 320.0))]]),
        policy=RelevancePolicy(),
        session_id=CAPTURE_ID,
        clock=lambda: 1000.0,
        keyframes=keyframes,
        face_filter=face_filter,
    )
    engine.load()
    return store, engine


def _walk(engine, count=5, *, start=900.0, step=0.1):
    for index in range(count):
        engine.observe(
            _frame(),
            received_at=start + index * step,
            source_seq=index,
            relpath=f"frames/{index:08d}.jpg",
        )


class TestTheEngineWritesOne:
    def test_a_remembered_class_gets_a_keyframe_with_no_verifier_at_all(
        self, tmp_path
    ):
        """The regression this change is really about.

        The crop used to be made only when `self._verification is not
        None and needs_verification(label)`, so `laptop` and `cell phone`
        -- the two REMEMBERED-tier classes, the only two a Tower with no
        verifier writes, and the two the physical walk produced -- never
        held a crop at all. A keyframe store hung off that condition
        would have had nothing to write for exactly the records that
        exist.
        """
        keyframes = KeyframeStore(tmp_path)
        store, engine = _engine(
            tmp_path, keyframes=keyframes, face_filter=_StubFilter()
        )

        _walk(engine)
        engine.finish()

        (observation,) = store.all_observations()
        assert keyframes.read(observation.observation_id) is not None
        assert engine.counters()["keyframes_written"] == 1
        assert engine.counters()["keyframes_refused"] == {}

    def test_the_keyframe_is_addressed_by_the_records_own_handle(self, tmp_path):
        """There is no second identifier to keep in step: the file is named
        by the same derivation the HTTP handle uses."""
        keyframes = KeyframeStore(tmp_path)
        store, engine = _engine(
            tmp_path, keyframes=keyframes, face_filter=_StubFilter()
        )

        _walk(engine)
        engine.finish()

        (observation,) = store.all_observations()
        expected = observation_id_for(CAPTURE_ID, "laptop", observation.observed_at)
        assert (tmp_path / "keyframes" / f"{expected}.jpg").exists()

    def test_one_keyframe_per_sighting_not_one_per_frame(self, tmp_path):
        keyframes = KeyframeStore(tmp_path)
        _, engine = _engine(tmp_path, keyframes=keyframes, face_filter=_StubFilter())

        _walk(engine, 30)
        engine.finish()

        assert engine.counters()["keyframes_written"] == 1
        assert len(list((tmp_path / "keyframes").glob("*.jpg"))) == 1

    def test_an_engine_with_no_keyframe_store_persists_no_pixels(self, tmp_path):
        """The default, and it is the behaviour that shipped. Every engine
        test written before this change constructs one this way."""
        store, engine = _engine(tmp_path)

        _walk(engine)
        engine.finish()

        assert store.all_observations()
        assert not (tmp_path / "keyframes").exists()
        assert "keyframes_written" not in engine.counters()

    def test_a_refusal_is_counted_by_reason_rather_than_swallowed(self, tmp_path):
        """A walk that wrote 11 records and no pictures must not look like
        one that wrote 11 of each."""
        keyframes = KeyframeStore(tmp_path)
        store, engine = _engine(
            tmp_path,
            keyframes=keyframes,
            face_filter=_StubFilter(available=False),
        )

        _walk(engine)
        engine.finish()

        assert store.all_observations()
        assert engine.counters()["keyframes_written"] == 0
        assert engine.counters()["keyframes_refused"] == {FILTER_UNAVAILABLE: 1}
        assert not (tmp_path / "keyframes").exists()

    def test_the_sighting_still_releases_its_crop_afterwards(self, tmp_path):
        """Writing the keyframe must not keep the pixels alive. The bound
        is still one crop per OPEN sighting."""
        keyframes = KeyframeStore(tmp_path)
        _, engine = _engine(tmp_path, keyframes=keyframes, face_filter=_StubFilter())

        _walk(engine)
        held = list(engine._tracker.open_sightings.values())
        assert held and held[0].best_crop is not None
        engine.finish()

        assert held[0].best_crop is None


# -- what the payload may claim ----------------------------------------


class TestThePayloadReportsRetentionTruthfully:
    def test_an_owned_crop_reports_object_memory_retention(self, world):
        captures, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        observation = _observation()

        image = render(
            captures,
            observation,
            FaceFilter(path=""),
            crop=True,
            keyframes=KeyframeStore(store_root),
        )
        view = build_imagery_view(
            observation, image, observation_id=observation_id
        )

        assert image.source == SOURCE_KEYFRAME
        assert view["imagery_retention"] == "object-memory"
        assert "object-memory" in view["imagery_retention_means"]
        assert view["imagery_source"] == SOURCE_KEYFRAME

    def test_a_capture_frame_still_reports_capture_side(self, world):
        captures, _, observation_id = world
        observation = _observation()

        image = render(captures, observation, _StubFilter(), crop=False)
        view = build_imagery_view(
            observation, image, observation_id=observation_id
        )

        assert image.source == SOURCE_CAPTURE
        assert view["imagery_retention"] == "capture-side"
        assert "data-captures" in view["imagery_retention_means"]

    def test_the_contract_identifier_does_not_move(self, world):
        """A shipped iOS build compares this for equality and refuses a
        payload that does not carry it. Every field added here is
        additive for exactly that reason."""
        _, _, observation_id = world
        image = render(None, _observation(), _StubFilter(), crop=True)

        view = build_imagery_view(
            _observation(), image, observation_id=observation_id
        )

        assert view["contract"] == IMAGERY_CONTRACT == (
            "object_memory.imagery/2026-08-27"
        )

    def test_the_owned_crop_is_not_filtered_again_on_read(self, world):
        """The bytes were filtered before they were written, which is a
        stronger guarantee than a read-time filter. Re-running the
        detector on an image whose faces are already filled would find
        nothing and report a label describing this read rather than what
        actually protected the file."""
        captures, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        reading_filter = _StubFilter()

        image = render(
            captures,
            _observation(),
            reading_filter,
            crop=True,
            keyframes=KeyframeStore(store_root),
        )

        assert image.available
        assert reading_filter.calls == 0
        assert image.filter_label == "display-filter/stub@0.30"

    def test_a_keyframe_store_that_misbehaves_falls_back_to_the_capture(
        self, world
    ):
        """Another FILTERED picture, never a raw one. Degrading here costs
        the ownership and nothing else."""

        class _Broken:
            def read(self, observation_id):
                raise RuntimeError("no")

        captures, _, observation_id = world

        image = render(
            captures, _observation(), _StubFilter(), crop=True, keyframes=_Broken()
        )

        assert image.available
        assert image.source == SOURCE_CAPTURE


def test_the_keyframe_module_imports_no_other_cartridge():
    """Asserted here as well as in the boundary suite, because this module
    is new and is the kind that grows a "just for a constant" import."""
    import ast

    source = Path(keyframes_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("tower.world_builder")
            assert not node.module.startswith("tower.document_memory")
            assert not node.module.startswith("tower.scene")



class TestTheViewRouteDescribesWhatCanBeServed:
    """`/imagery` must not answer for `/frame` while `/crop` can serve.

    THE DEFECT THIS EXISTS FOR, AND IT MADE THE WHOLE FEATURE INVISIBLE.

    The view route rendered a FRAME and reported its availability. A
    record whose recording had been deleted but whose owned crop was
    sitting on disk therefore answered `available: false, reason:
    imagery-no-longer-available` -- and the shipped iOS loader gates on
    exactly that boolean:

        guard description.available else { self.phase = .noPicture(...) }

    So it would never have asked for the crop, and the wearer would have
    been told the picture was gone while it was being held for them, in
    precisely the case keyframes were built for. The keyframes would have
    been written, pruned, purged and served correctly by every test in
    this file, and no client would ever have seen one.
    """

    def test_the_view_says_available_when_only_the_owned_crop_survives(
        self, world, monkeypatch
    ):
        captures, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch)
        _delete_tree(captures / "captures")

        body = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert body["available"] is True, (
            "the crop can be served, so a client must not be told there is "
            "no picture"
        )
        assert body["reason"] is None
        assert body["memory_retained"] is True
        assert body["imagery_source"] == SOURCE_KEYFRAME
        assert body["imagery_retention"] == "object-memory"

    def test_the_view_says_the_context_frame_is_gone_at_the_same_time(
        self, world, monkeypatch
    ):
        """Both facts, separately, because they no longer stand or fall
        together: the owned crop outlives the recording and the context
        view does not."""
        captures, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch)
        _delete_tree(captures / "captures")

        body = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert body["available"] is True
        assert body["frame_available"] is False

    def test_both_are_available_while_the_recording_is_still_there(
        self, world, monkeypatch
    ):
        _, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch)

        body = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert body["available"] is True
        assert body["frame_available"] is True

    def test_a_record_with_no_keyframe_still_describes_the_capture_crop(
        self, world, monkeypatch
    ):
        """The path every record written before keyframes existed takes.

        The view is still built from a crop render; with no owned
        keyframe that render crops the capture frame, so the answer is
        the one it always was.
        """
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        body = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert body["available"] is True
        assert body["imagery_source"] == SOURCE_CAPTURE
        assert body["imagery_retention"] == "capture-side"
        assert body["frame_available"] is True

    def test_nothing_anywhere_is_still_the_honest_410(
        self, world, monkeypatch
    ):
        """No keyframe and no recording. `available: false` is now a much
        stronger statement than it used to be, and it must still be
        reachable."""
        captures, _, observation_id = world
        client = _client(world, monkeypatch)
        _delete_tree(captures / "captures")

        body = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert body["available"] is False
        assert body["reason"] == IMAGERY_EXPIRED
        assert body["memory_retained"] is True
        assert body["frame_available"] is False

    def test_the_contract_identifier_did_not_move(self, world, monkeypatch):
        """Every field added here is additive. A shipped iOS build
        compares this identifier for equality and refuses a payload that
        does not carry it."""
        _, _, observation_id = world
        client = _client(world, monkeypatch)

        body = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        ).json()

        assert body["contract"] == "object_memory.imagery/2026-08-27"

    def test_the_view_costs_no_second_filter_pass(self, world, monkeypatch):
        """`frame_available` is an existence check, not a render.

        The view route is asked once per row on a screen. Paying a YuNet
        pass per row to answer a boolean would make a metadata route cost
        more than the picture it describes.
        """
        from tower.object_memory import imagery as imagery_module

        _, store_root, observation_id = world
        _write_keyframe(store_root, observation_id)
        client = _client(world, monkeypatch)

        applies = []
        original = imagery_module.FaceFilter.apply

        def counting(self, image):
            applies.append(image)
            return original(self, image)

        monkeypatch.setattr(imagery_module.FaceFilter, "apply", counting)

        response = client.get(
            f"/object-memory/observations/{observation_id}/imagery"
        )

        assert response.status_code == 200
        assert applies == [], (
            "an owned keyframe is served from bytes filtered at write time; "
            "the view route must not run the filter at all"
        )



def _settings():
    """`Settings` as this process's environment currently describes it.

    `get_settings` reads the environment on every call and caches
    nothing, which is what lets a parametrised case here set one variable
    and read the answer back. Wrapped anyway so the import stays local to
    the cases that need it, exactly as `test_capture_arming.py` does.
    """
    from tower.config import get_settings

    return get_settings()


class TestTheWriteIsExceptionTight:
    """`write()` must not raise, whatever the filter hands back.

    `_write_keyframe` promises in its docstring that it never raises, and
    the caller is a producer inside its frame loop -- so an escape here
    ends the walk and takes `engine.release()` with it. One unwritable
    keyframe would cost every sighting still open, which is the exact
    failure the graceful-stop work exists to prevent, arriving by a
    different door.

    A reviewer reproduced three escapes: a filter returning a generator
    instead of a sequence of boxes, an int, and a box of the wrong arity.
    `len(filled)`, `encoded.tobytes()` and `_obscured_fraction` all sat
    outside the guard.
    """

    def _write(self, tmp_path, filled):
        store = KeyframeStore(tmp_path)
        image = _image()

        class Filter:
            available = True
            label = "display-filter/yunet-2023mar@0.30"

            def apply(self, frame):
                return frame, filled

        return store, store.write(
            "0" * 16, image, Filter(), source_capture="cap", source_relpath="f.jpg"
        )

    @pytest.mark.parametrize(
        "filled",
        [
            (row for row in []),
            5,
            [(0, 0, "a", 1)],
            [(0, 0)],
            object(),
        ],
        ids=["generator", "int", "string-in-a-box", "short-box", "object"],
    )
    def test_a_filter_of_the_wrong_shape_refuses_rather_than_raising(
        self, tmp_path, filled
    ):
        store, result = self._write(tmp_path, filled)

        assert result.written is False
        assert list((tmp_path / "keyframes").glob("*")) == [] or not (
            tmp_path / "keyframes"
        ).exists()
        assert store.read("0" * 16) is None

    def test_a_sidecar_that_cannot_be_serialised_leaves_no_image(self, tmp_path):
        """The partial pair a reviewer found, and the invariant it broke.

        `json.dumps` used to run inside a `try` that caught `OSError`
        only, AFTER the image had already been written -- so an
        unserialisable `filter_label` raised `TypeError` out of `write()`
        and left an unattributable first-person crop on disk for the
        retention window. `read()` refused to serve it, which is why
        nothing leaked; the file was there anyway, and this module's
        docstring said in bold that it could not be.
        """
        store = KeyframeStore(tmp_path)

        class Filter:
            available = True
            label = object()  # not JSON-serialisable

            def apply(self, frame):
                return frame, []

        result = store.write(
            "0" * 16, _image(), Filter(), source_capture="cap", source_relpath="f.jpg"
        )

        assert result.written is False
        assert not list((tmp_path / "keyframes").glob("*")), (
            "the image must not survive a sidecar that could not be written"
        )
        assert store.read("0" * 16) is None

    def test_the_engine_survives_a_keyframe_store_that_raises(self, tmp_path):
        """The second wall. `write()` is tight now; this proves the caller
        does not depend on that being true forever."""

        class Exploding:
            def write(self, *args, **kwargs):
                raise RuntimeError("no")

        _, engine = _engine(tmp_path, keyframes=Exploding(), face_filter=_StubFilter())
        _walk(engine, count=6)
        engine.release()

        assert engine.observations_recorded >= 1, (
            "the record must still be written when its picture cannot be"
        )
        assert engine.keyframes_refused.get("store-raised", 0) >= 1


class TestTheSwitch:
    """`TOWER_OBSERVATION_KEEP_IMAGERY` had no test of any kind.

    It is the single setting governing whether this cartridge persists
    first-person pixels, and neither its default, its parsing, nor its
    journey into the producer's argv was covered anywhere. The repo has
    the precedent for pinning exactly this shape --
    `test_the_settings_and_the_producer_agree_about_verifier_names`.
    """

    def test_it_defaults_to_on(self, monkeypatch):
        monkeypatch.delenv("TOWER_OBSERVATION_KEEP_IMAGERY", raising=False)

        assert _settings().observation_keep_imagery is True

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("", True),
            ("treu", False),
        ],
        ids=[
            "true", "TRUE", "1", "yes", "on",
            "false", "0", "no", "blank", "a-typo",
        ],
    )
    def test_only_the_shared_spellings_of_true_mean_true(
        self, monkeypatch, value, expected
    ):
        """Pinned because the last case is a trap worth knowing about.

        `config._flag` is one helper for every on/off variable, and its
        docstring says why: so a fourth flag cannot arrive with a fifth
        spelling of "true". Two consequences are worth pinning. A BLANK
        value means "unset, use the default" -- this file's convention
        everywhere, and the opposite of `TOWER_DEV_MODE`'s, which that
        docstring calls out. And anything else outside the accepted list,
        including a typo, reads as **false** -- which for this variable
        means the cartridge stops keeping pictures.

        That is not softened here. Growing a private spelling for one
        flag is exactly what the shared helper exists to prevent, and the
        failure is at least loud in behaviour: no keyframe is written,
        the startup line says `off`, and the producer's report carries
        `keep_imagery: false`. What was missing was anybody having
        checked, which is what this case is.
        """
        monkeypatch.setenv("TOWER_OBSERVATION_KEEP_IMAGERY", value)

        assert _settings().observation_keep_imagery is expected

    def test_the_producer_is_told_which_way_it_is_set(self, monkeypatch, tmp_path):
        """Both halves of the agreement, in the argv the Tower builds."""
        from tower.main import OBJECT_MEMORY_WORKER, create_app

        for setting, flag in (("true", "--keep-imagery"), ("false", "--no-keep-imagery")):
            monkeypatch.setenv("TOWER_OBSERVATION_KEEP_IMAGERY", setting)
            monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(tmp_path / "memory"))
            monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
            app = create_app()
            argv = list(app.state.capture_workers.spec_for(OBJECT_MEMORY_WORKER).argv)
            assert flag in argv, f"{setting!r} should pass {flag}"
            other = "--no-keep-imagery" if flag == "--keep-imagery" else "--keep-imagery"
            assert other not in argv

    def test_the_producer_accepts_both_flags(self):
        """Run as a user runs it, so a flag the Tower passes cannot be one
        the producer refuses -- which would kill it at spawn with only a
        warning in a log nobody is reading."""
        for flag in ("--keep-imagery", "--no-keep-imagery"):
            result = subprocess.run(
                [sys.executable, "scripts/object_memory_session.py", "--help"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert flag in result.stdout

    def test_switching_it_off_writes_no_keyframe(self, tmp_path):
        _, engine = _engine(tmp_path, keyframes=None, face_filter=_StubFilter())
        _walk(engine, count=6)
        engine.release()

        assert engine.observations_recorded >= 1
        assert engine.keyframes_written == 0
        assert not (tmp_path / "keyframes").exists()


class TestTheUnboundedStore:
    def test_a_keyframe_is_written_when_nothing_ever_expires(self, tmp_path):
        """`retention is None` short-circuits `prune_expired`, keyframes
        included. That must mean "never deleted", not "never written"."""
        store = KeyframeStore(tmp_path)
        result = store.write(
            "a" * 16, _image(), _StubFilter(), source_capture="cap",
            source_relpath="f.jpg",
        )

        assert result.written is True

        observations = ObservationStore(tmp_path, retention_seconds=None)
        observations.prune_expired()

        assert store.read("a" * 16) is not None


class TestTheLineagePresenceIsContained:
    @pytest.mark.parametrize(
        "session_id",
        ["../..", "..\\..", "/etc", "C:\\Windows", "a/../../b"],
    )
    def test_a_traversal_shaped_session_id_reports_nothing(
        self, tmp_path, session_id
    ):
        """It only steers a log line, and it is still built into a path.

        `_frame_in` runs `session_id` through `_contained` and this did
        not, which is the kind of inconsistency that is free to fix and
        expensive to discover.
        """
        from tower.object_memory.imagery import capture_lineage_present

        record = _observation(session_id=session_id)

        assert capture_lineage_present(tmp_path, record) is False
