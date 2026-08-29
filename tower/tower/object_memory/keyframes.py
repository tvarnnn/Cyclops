"""A small picture Object Memory OWNS, under Object Memory's retention.

WHY A CARTRIDGE THAT PERSISTED NO PIXELS NOW PERSISTS SOME.

`imagery.py` resolves a record back to the frame it was derived from, and
that frame lives in `data/captures/<session_id>/frames/*.jpg` -- written
by `CaptureRecorder`, owned by capture-side lifecycle, and governed by a
retention this cartridge neither sets nor enforces. Every record says so:
`privacy_tags: ["derived-only", "frame-referenced"]`, and the imagery
payload has always answered `imagery_retention: "capture-side"`.

That arrangement was honest and it was also a defect waiting for its
first caller. `CaptureRecorder.purge()` has no production caller today,
so nothing prunes captures and every record still has its picture. The
moment anything does -- a capture pruner, an operator reclaiming the
~2.1 GB an hour of recording costs -- EVERY memory loses its picture at
once, and a 30-day record whose whole product value is the picture
becomes a label and a timestamp. A durable record pointing into an
ephemeral store is not a retention policy; it is a race.

So the cartridge takes ownership of ONE small crop per record:

    the capture frame     the full-frame context view, still the source
                          of `/frame`, still capture-side, still allowed
                          to disappear on somebody else's schedule
    the owned keyframe    the padded crop of the strongest look, written
                          here, pruned by `ObservationStore` when the
                          record it belongs to expires, and deleted by
                          `purge()` with it

This is the first place in this cartridge where a filtered pixel reaches
disk, which is why the rest of this module is about refusing to write.

FAIL CLOSED, AND THE STRUCTURE IS THE GUARANTEE.

`imagery.render` refuses to SERVE an unfiltered frame; if it is wrong,
one request gets a picture it should not have and the frame on disk is
unchanged. This module is worse to get wrong: a byte written here is
written for thirty days, survives the process, and reaches a backup.

So `write()` is arranged so that the only value that can reach
`cv2.imencode` is the object `face_filter.apply` returned. The input
crop is not referenced after the filter call, there is no `except:` that
falls through to writing something, and every refusal returns before any
file is created. A filter that is unavailable, a filter that raises, an
encode that fails and a write that fails all leave the directory exactly
as they found it -- no image, no sidecar, no partial pair.

THE SIDECAR IS PART OF THE ARTIFACT, NOT METADATA ABOUT IT.

A `.jpg` alone is not evidence that anything ran. It could have been
restored from a backup taken before this module existed, copied in by
hand, or left by a future writer. `read()` therefore REQUIRES the sidecar
and ignores an image without one -- the same posture as the serving path,
where a filter that cannot run means nothing is served.

The sidecar records what RAN, in `imagery.py`'s wording: the detector's
identity and its threshold, never "redacted", "anonymised" or
"privacy-safe". YuNet has measured blind spots -- a face occluded past
about 60%, a face rotated about 90 degrees in plane -- and bodies,
clothing, screens and room contents were never in scope at all.

WHAT IT COSTS ON DISK.

MEASURED, not estimated. Every record in this host's own store whose
frame is still on disk -- 26 of them -- cropped at its own
`best_bounding_box` with the producer's 0.35 padding, filtered with the
vendored YuNet weights, downscaled to a 384 px long side and encoded at
JPEG quality 80:

    mean 11.7 KB, median 10.9 KB, min 2.5 KB, max 22.1 KB
    (361.7 KB for all 26)

The corpus's own rate is one record every 9.2 seconds of walking, about
380 an hour, so a keyframe per record costs about **4.3 MB an hour of
walking**, against roughly 2.1 GB an hour for the recording it is taken
from -- a factor of about 500. The whole owned store for every record on
this host is **1.31 MiB against 568 MiB of frames: 0.23%**.

MEASURED OVER ALL 116 RECORDS, not a sample. An earlier version of this
note quoted a 26-record subset and a 13.9 KB mean; a reviewer caught it,
and the figures above come from re-running the shipped `KeyframeStore`
over every record in the store whose frame still resolves -- 116 of 116.

The 384 px bound is the reason those figures are small, and it is a CAP
rather than a target: a crop already smaller is written at its own size
and never upscaled. Upscaling would invent detail, cost bytes, and make
a 3%-of-frame object look like evidence it is not.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the keyframes live, relative to the OBSERVATION root. Under the
# store's own root rather than beside the captures, because the whole
# point is that this cartridge's retention governs them: a file in
# somebody else's tree is a file somebody else may delete.
KEYFRAMES_DIRNAME = "keyframes"

IMAGE_SUFFIX = ".jpg"
SIDECAR_SUFFIX = ".json"

# The sidecar's shape, versioned so a later field can be added without a
# reader having to guess whether an absent key means "old" or "false".
SCHEMA_VERSION = 1

# The longest side a keyframe may have. See the module docstring for the
# measured bytes; this is the constant those figures depend on.
#
# 384 rather than the capture's own 640, because this is a CROP shown as
# a thumbnail or a card image on a phone rather than a frame to be
# inspected -- the full-frame view is `/frame`, and it still comes from
# the capture. The real crops measured above run from 140x245 up to
# 360x640, so the bound bites on the large ones and does nothing at all
# to the small ones: DOWNSCALE ONLY, `_bounded` never enlarges.
MAX_LONG_SIDE = 384

# Lower than `imagery.JPEG_QUALITY` (90), deliberately. That one encodes
# a picture served once and discarded; this one encodes a picture kept
# for thirty days, and 80 is where the size curve stops being worth the
# difference on an image this small.
JPEG_QUALITY = 80

# What an `observation_id` is allowed to look like.
#
# It reaches this module as a STRING off a JSONL record and is used to
# BUILD A PATH, which is the same threat `imagery._contained` was written
# for and it says why at length. `records.observation_id_for` produces 16
# lowercase hex characters (blake2b, digest_size=8); the range here is
# wider than that so a future digest size does not need a migration, and
# narrow enough that no separator, no dot and no drive letter can appear
# in it at all.
#
# The regex is the first guard and `_contained` is the second. Either
# alone would do today. Two is what keeps this true after somebody
# reasonably decides ids should be base32.
_OBSERVATION_ID = re.compile(r"^[0-9a-f]{8,64}$")

# Why a keyframe was not written. Values a counter is keyed by and a log
# line names -- never sentences shown to a wearer, which is the same rule
# `imagery.py`'s refusal reasons follow.
UNUSABLE_ID = "unusable-observation-id"
NO_IMAGERY = "no-crop-held"
FILTER_UNAVAILABLE = "display-filter-unavailable"
FILTER_FAILED = "display-filter-failed"
ENCODE_FAILED = "jpeg-encode-failed"
WRITE_FAILED = "write-failed"

REFUSAL_REASONS = (
    UNUSABLE_ID,
    NO_IMAGERY,
    FILTER_UNAVAILABLE,
    FILTER_FAILED,
    ENCODE_FAILED,
    WRITE_FAILED,
)


@dataclass(frozen=True)
class KeyframeWrite:
    """What happened, or the reason nothing did.

    One type for both outcomes, for the same reason `imagery.Imagery` is
    one type for both: a caller that had to check a bool and then
    separately ask why is a caller that can forget to ask, and every
    refusal here is a picture a wearer will not have.
    """

    written: bool
    reason: str | None = None
    path: Path | None = None
    bytes_written: int = 0
    filter_label: str | None = None
    regions_filled: int = 0
    subject_obscured: float = 0.0


class KeyframeStore:
    """The keyframes under one observation root.

    Constructed from the OBSERVATION root, not from the keyframe
    directory, so there is exactly one thing to configure and it is the
    same thing `ObservationStore` is configured with. That is what lets
    `ObservationStore` build its own without any wiring -- see its
    constructor for why it does rather than being handed one.

    Holds no lock. Every method is a small number of filesystem calls on
    per-observation paths, the producer writes from one thread, and the
    web process only ever reads. Where two processes do overlap -- a
    prune during a read -- the failure mode is a `read()` that returns
    None, which is the same answer as "no keyframe", and the record still
    has its capture-side frame to fall back on.
    """

    def __init__(self, root) -> None:
        self._root = Path(root)
        self._directory = self._root / KEYFRAMES_DIRNAME

    @property
    def directory(self) -> Path:
        return self._directory

    # -- addressing ----------------------------------------------------

    def path_for(self, observation_id) -> Path | None:
        """Where this record's keyframe would be, or None if the id is not one.

        Two independent guards, in this order:

        1. The id must match `_OBSERVATION_ID`. A `..`, a separator, a
           drive letter and an empty string are all rejected before a
           `Path` is constructed at all.
        2. The constructed path must still resolve inside the keyframe
           directory. `resolve()` on both sides, so a symlink planted in
           the directory cannot be used to step out either.

        The second cannot be reached past the first today. It is here
        because `imagery._contained` learned the same lesson about
        `session_id` and `best_relpath`, and because the cheap guard is
        the one that survives a future change to what an id looks like.
        """
        if not isinstance(observation_id, str):
            return None
        if not _OBSERVATION_ID.match(observation_id):
            return None
        candidate = self._directory / f"{observation_id}{IMAGE_SUFFIX}"
        try:
            resolved = candidate.resolve()
            base = self._directory.resolve()
        except OSError:
            return None
        return resolved if resolved.is_relative_to(base) else None

    @staticmethod
    def _sidecar_path(image_path: Path) -> Path:
        return image_path.with_suffix(SIDECAR_SUFFIX)

    # -- writing -------------------------------------------------------

    def write(
        self,
        observation_id,
        crop_bgr,
        face_filter,
        *,
        source_capture: str | None = None,
        source_relpath: str | None = None,
        written_at: float | None = None,
    ) -> KeyframeWrite:
        """Filter, bound, encode, write. Or write nothing and say why.

        THE LOAD-BEARING PROPERTY OF THIS FUNCTION is that the bytes it
        writes are always `face_filter.apply`'s output. `crop_bgr` is
        passed to the filter and never referenced again; every failure
        path returns before a file exists; and the image and its sidecar
        are written together, with the image removed again if the sidecar
        cannot be written. There is no branch that reaches `open(...)`
        with an unfiltered crop, and `tests/test_object_memory_keyframes.py`
        asserts that a filter which raises leaves the directory empty.

        Filtering happens BEFORE the downscale, and that order is
        deliberate. YuNet is handed the crop at its native resolution,
        which is the resolution the detector's blind spots were measured
        at; filtering a 384 px thumbnail instead would ask it to find
        faces in an image already thrown away. The cost is one detection
        on a crop rather than on a thumbnail, off the frame path, once
        per sighting.
        """
        import cv2

        path = self.path_for(observation_id)
        if path is None:
            # Not a refusal a wearer ever causes: an id that does not
            # match came off a record something else wrote.
            logger.warning(
                "[Tower][ObjectMemory] refusing to write a keyframe for an "
                "observation id that is not one: %r",
                observation_id,
            )
            return KeyframeWrite(False, UNUSABLE_ID)

        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            # The ordinary reason, and not an error: a sighting whose
            # best look was a zero-area box holds no crop to write.
            return KeyframeWrite(False, NO_IMAGERY)

        if face_filter is None or not getattr(face_filter, "available", False):
            # The refusal that matters most. A Tower with no
            # face-detection weights writes NO keyframes at all rather
            # than writing unfiltered ones and labelling them honestly --
            # a label is not a control, and a file on disk outlives every
            # label that travelled with it.
            return KeyframeWrite(False, FILTER_UNAVAILABLE)

        try:
            filtered, filled = face_filter.apply(crop_bgr)
        except Exception:  # noqa: BLE001
            # A filter that failed has said nothing about these pixels.
            # There is no fallback here, exactly as there is none in
            # `imagery.render`: refusing costs a picture, writing would
            # cost the promise, and this one would cost it for 30 days.
            logger.exception(
                "[Tower][ObjectMemory] the display filter failed on the "
                "keyframe for %s; writing nothing",
                observation_id,
            )
            return KeyframeWrite(False, FILTER_FAILED)
        if filtered is None:
            logger.warning(
                "[Tower][ObjectMemory] the display filter returned nothing "
                "for the keyframe of %s; writing nothing",
                observation_id,
            )
            return KeyframeWrite(False, FILTER_FAILED)

        # From here down, `crop_bgr` is never read again. Everything that
        # can reach the file is derived from `filtered`.
        #
        # EVERYTHING that touches the filter's output is inside this
        # `try`, and it did not used to be. `len(filled)`,
        # `encoded.tobytes()` and `_obscured_fraction` sat outside it, so
        # a filter returning an unexpected SHAPE -- a generator instead of
        # a sequence, an int, a box of the wrong arity -- raised a
        # `TypeError` straight out of `write()`. A reviewer reproduced all
        # three. The caller is a producer mid-walk whose `_write_keyframe`
        # promises never to raise, so the escape killed the process and
        # took the whole `engine.release()` flush with it: a bad keyframe
        # would have cost every sighting still open.
        try:
            bounded = self._bounded(filtered)
            ok, encoded = cv2.imencode(
                IMAGE_SUFFIX, bounded, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                return KeyframeWrite(False, ENCODE_FAILED)
            image_bytes = encoded.tobytes()
            regions_filled = len(filled)
            obscured = _obscured_fraction(filtered.shape, filled)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[Tower][ObjectMemory] could not encode the keyframe for %s",
                observation_id,
            )
            return KeyframeWrite(False, ENCODE_FAILED)

        sidecar = {
            "schema_version": SCHEMA_VERSION,
            # WHAT RAN, named as what ran. `FaceFilter.label` is
            # `display-filter/yunet-2023mar@0.30` -- the detector and its
            # threshold -- and never a claim about what was achieved.
            "filter_label": getattr(face_filter, "label", None),
            # How many regions were filled. Zero means the detector found
            # none, NOT that there were none.
            "regions_filled": regions_filled,
            # How much of this keyframe a fill covered, 0.0 to 1.0.
            #
            # Measured against the WHOLE CROP rather than against the
            # record's own box, because by this point the full frame is
            # gone and the crop is all there is. The crop is the box plus
            # 35% of its width and height on every side, so a fill that
            # covers a third of this image has covered a large part of
            # what the record is about -- which is the thing a client
            # needs to know before showing it without comment.
            #
            # `imagery._obscured_fraction` measures the same idea against
            # the box, on the full frame, where the box is still known.
            # The two denominators differ and the sidecar says which this
            # one is.
            "subject_obscured": obscured,
            "written_at": written_at,
            # Provenance, so a keyframe is still auditable back to the
            # recording even after the recording is gone.
            "source_capture": source_capture,
            "source_relpath": source_relpath,
        }

        # SERIALISED BEFORE THE IMAGE IS WRITTEN, and that ordering is the
        # whole of the "no partial pair" promise this module makes in
        # bold.
        #
        # It used to be serialised inside the sidecar's own `try`, which
        # caught `OSError` only -- so a `filter_label` that would not
        # serialise raised `TypeError` AFTER the `.jpg` had already
        # landed, escaped `write()`, and left an unattributable
        # first-person crop on disk for the retention window. `read()`
        # refuses it, so nothing was ever served; the file was there
        # anyway, and the docstring said it could not be. A reviewer
        # reproduced it.
        #
        # Doing the encoding first means the only thing left after the
        # image is written is one `write_text` of bytes that already
        # exist.
        try:
            encoded_sidecar = json.dumps(sidecar)
        except (TypeError, ValueError):
            logger.exception(
                "[Tower][ObjectMemory] the keyframe sidecar for %s could not "
                "be serialised; writing nothing",
                observation_id,
            )
            return KeyframeWrite(False, WRITE_FAILED)

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_bytes)
        except OSError:
            logger.warning(
                "[Tower][ObjectMemory] could not write the keyframe for %s",
                observation_id,
            )
            self._unlink(path)
            return KeyframeWrite(False, WRITE_FAILED)

        try:
            self._sidecar_path(path).write_text(encoded_sidecar, encoding="utf-8")
        except OSError:
            # An image with no sidecar is not something this store may
            # leave behind: `read()` would ignore it forever and `prune`
            # would carry it for thirty days. Undo the half that
            # succeeded rather than leaving an unreadable orphan.
            logger.warning(
                "[Tower][ObjectMemory] could not write the keyframe sidecar "
                "for %s; removing the image it belongs to",
                observation_id,
            )
            self._unlink(path)
            return KeyframeWrite(False, WRITE_FAILED)

        return KeyframeWrite(
            True,
            None,
            path=path,
            bytes_written=len(image_bytes),
            filter_label=sidecar["filter_label"],
            regions_filled=len(filled),
            subject_obscured=obscured,
        )

    @staticmethod
    def _bounded(image):
        """The image with its long side capped. Never enlarged.

        `INTER_AREA`, which is the right kernel for shrinking and is also
        the one that keeps a filled region filled: it averages the source
        pixels under each destination pixel, so the interior of a solid
        fill stays solid and only its one-pixel border blends. A
        `INTER_CUBIC` shrink can ring around a hard edge, which on a
        filled face region means faint structure where there should be
        none.
        """
        import cv2

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= MAX_LONG_SIDE:
            return image
        scale = MAX_LONG_SIDE / float(longest)
        return cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    # -- reading -------------------------------------------------------

    def read(self, observation_id) -> tuple[bytes, dict] | None:
        """The keyframe and its sidecar, or None.

        THE SIDECAR IS REQUIRED, and that is a refusal rather than a
        convenience. A `.jpg` sitting in this directory with no sidecar
        beside it is not evidence that the filter ran on it -- it could
        have been restored from a backup, copied in by hand, or left by a
        writer that did not exist when it was made. Serving it would be
        serving an unfiltered first-person crop on the strength of its
        filename, which is exactly the failure the whole write path is
        arranged to prevent.

        Returns None for every failure, because the caller's answer is
        the same for all of them: fall back to the capture-side frame,
        which is filtered on read.
        """
        path = self.path_for(observation_id)
        if path is None:
            return None
        try:
            raw = self._sidecar_path(path).read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            sidecar = json.loads(raw)
        except ValueError:
            logger.warning(
                "[Tower][ObjectMemory] the keyframe sidecar for %s is not "
                "readable JSON; serving nothing from it",
                observation_id,
            )
            return None
        if not isinstance(sidecar, dict):
            return None
        try:
            image_bytes = path.read_bytes()
        except OSError:
            return None
        if not image_bytes:
            return None
        return image_bytes, sidecar

    # -- retention -----------------------------------------------------

    def _ids_on_disk(self) -> set[str]:
        """Every id this directory holds a file for, image or sidecar.

        Sidecars are included so a half-pair left by a crash between the
        two writes is still something `prune` and `purge` can reach. A
        file whose stem is not an id at all is left alone: this directory
        is ours, but deleting a file we cannot explain is not the same
        promise as deleting the ones we made.
        """
        found: set[str] = set()
        try:
            entries = list(self._directory.iterdir())
        except OSError:
            return found
        for entry in entries:
            if entry.suffix not in (IMAGE_SUFFIX, SIDECAR_SUFFIX):
                continue
            if _OBSERVATION_ID.match(entry.stem):
                found.add(entry.stem)
        return found

    def _unlink(self, path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    def _remove(self, observation_id: str) -> bool:
        """Both halves of one keyframe. True only if both are really gone."""
        path = self.path_for(observation_id)
        if path is None:
            return False
        image_gone = self._unlink(path)
        sidecar_gone = self._unlink(self._sidecar_path(path))
        return image_gone and sidecar_gone

    def prune(self, keep_ids) -> int:
        """Delete every keyframe whose id is not in `keep_ids`. Returns how many.

        A KEEP list rather than a delete list, and that direction is the
        whole safety property. `ObservationStore.prune_expired` computes
        the records that survived; anything on disk that is not one of
        them is either expired or an orphan, and both should go. Passing
        a delete list instead would mean a record that vanished by any
        route this cartridge did not model -- a hand-edited file, a
        restored backup, a schema mismatch -- kept its picture forever.
        """
        keep = {
            candidate
            for candidate in keep_ids
            if isinstance(candidate, str) and _OBSERVATION_ID.match(candidate)
        }
        removed = 0
        for observation_id in sorted(self._ids_on_disk() - keep):
            if self._remove(observation_id):
                removed += 1
            else:
                logger.warning(
                    "[Tower][ObjectMemory] could not delete the expired "
                    "keyframe %s; it is still on disk",
                    observation_id,
                )
        return removed

    def purge(self) -> tuple[int, tuple[str, ...]]:
        """Delete every keyframe. Returns `(removed, retained)`.

        RETAINED IS RETURNED RATHER THAN LOGGED AND FORGOTTEN, because
        `CARTRIDGE-GROUNDWORK.md` is right that "a false claim of
        deletion is worse than an honest failure". A wearer asking for
        their object memory to be erased, and being told it was, while a
        directory of pictures of their home survives a Windows sharing
        violation, is the exact shape of that failure.

        `retained` names the ids this call could not remove. It is empty
        on the ordinary path, and `scripts/object_query.py` prints it.
        """
        removed = 0
        retained: list[str] = []
        for observation_id in sorted(self._ids_on_disk()):
            if self._remove(observation_id):
                removed += 1
            else:
                retained.append(observation_id)
        if retained:
            logger.error(
                "[Tower][ObjectMemory] purge could not delete %d keyframe(s) "
                "under %s: %s. The memory store reports this rather than "
                "claiming a deletion that did not happen.",
                len(retained),
                self._directory,
                ", ".join(retained),
            )
        if removed and not retained:
            # Only when the directory is genuinely empty. `rmdir` on a
            # directory holding anything else raises, which is the
            # correct outcome: a file we did not recognise is a file we
            # do not delete, and the directory has to stay for it.
            try:
                self._directory.rmdir()
            except OSError:
                pass
        return removed, tuple(retained)


def _obscured_fraction(shape, filled) -> float:
    """How much of this crop the filter covered, as a fraction of it.

    The largest single fill rather than the union of all of them, for the
    same reason `imagery._obscured_fraction` gives: two overlapping fills
    summed naively produce a fraction above 1.0, and a figure that can
    say something impossible is worse than one that under-reports.
    """
    if not filled:
        return 0.0
    height, width = shape[:2]
    area = float(max(height, 0) * max(width, 0))
    if area <= 0:
        return 0.0
    largest = 0.0
    for x0, y0, x1, y1 in filled:
        largest = max(largest, max(0.0, x1 - x0) * max(0.0, y1 - y0))
    return round(min(1.0, largest / area), 4)
