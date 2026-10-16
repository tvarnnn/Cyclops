"""A second opinion on a label, bought once per sighting and never per frame.

WHY THIS EXISTS AT ALL.

Reading the crops the shipped detector produced over the real corpus
found that its labels are unreliable for exactly the classes this
cartridge is FOR. A ceiling fan is `airplane` at 0.99 and `scissors` at
0.93. A laptop keyboard is `remote` at 0.87 -- the three highest-scoring
`remote` sightings in 18,821 frames are all keyboards, and the three that
appear to be a real remote all score below 0.68, so the score orders them
exactly wrong. Widening the class whitelist over that detector alone
would fill a wearer's memory with ceiling fans.

So a class the tables consider worth finding, but the detector cannot be
trusted to name, is only written if something else agrees. That is the
whole of this module.

THE RATE, WHICH IS THE POINT.

The validated capture is 186 seconds long and produced 4,287
detections. Nothing here runs on any of them. Verification is asked ONCE
PER SIGHTING, when the sighting matures -- and across the whole
18,821-frame corpus there are 499 sightings of at least three frames, of
which the `verify` tier accounts for **53**. That is one call per 355
frames, or one every 36.6 seconds of recording, against a detector that
runs about ten times a second.

Measured end to end rather than projected, one run at a time on an idle
host, replaying that capture:

    verifier none    8 observations    87.7 s   39.810 ms/frame
    verifier owlv2  12 observations   102.7 s   46.621 ms/frame
                                      5 calls in 1.08 s of model time

**1.08 seconds of inference across a 186-second capture** -- 0.6% of the
recording -- for four more memories, plus a one-off model load of about
seven seconds. The per-frame difference between the two runs is INSIDE
the run-to-run spread of the same configuration (39.8 to 46.9 across five
replays), so it is not attributed to the verifier here. What is
attributable is the 1.08 seconds, and it is counted rather than
estimated.

The queue never queued: peak depth 0, zero backlog drops.

Every one of those numbers is counted at runtime rather than assumed:
`VerificationQueue` reports submissions, completions, refusals, drops and
the deepest the backlog ever got, and the producer prints them. A funnel
whose narrowing is not measured is a funnel that has quietly stopped
narrowing.

WHAT A VERIFIER MAY AND MAY NOT DO.

It may say "no". It may say "yes, and here is what I think it is". It may
not introduce a class: `RelevanceFilter` consults it only for classes the
deterministic tables have already admitted, and a verdict naming anything
else is recorded as evidence and changes nothing. The privacy policy is
not a model's to overturn, and `person` is not reachable from here by any
path.

ASYNCHRONOUS, AND BOUNDED.

The queue holds a fixed number of pending crops and DROPS THE OLDEST when
it is full. Dropping is the correct failure: an unverified sighting is
simply not written, which is the same outcome as a Tower with no verifier
at all, whereas an unbounded queue on a fifteen-minute walk is a memory
leak with a semantic model attached to it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# How many sightings may be waiting for a verdict at once.
#
# Small on purpose. At the measured rate -- roughly one `verify`-tier
# sighting every 36.6 seconds of recording -- a backlog of eight means
# the verifier has fallen nearly five minutes behind, and at that point
# the honest thing is to drop rather than to keep a queue that will never
# catch up. If this is ever seen to fill in the field, the finding is
# that the model is too slow for this host, not that the queue is too
# small.
DEFAULT_MAX_PENDING = 8

# How long a worker waits for a job before checking whether it should
# stop. Short enough that shutdown is prompt, long enough that an idle
# verifier is not a busy loop.
_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class Verdict:
    """What a verifier concluded about one proposed label.

    `agrees` is the only field the policy reads. Everything else is
    provenance, and it is carried onto the persisted record so that a
    memory admitted by a model says WHICH model and HOW STRONGLY -- a
    record that merely said "verified" would be a claim with no way to
    audit it and no way to re-evaluate it when the model changes.

    `label` is what the verifier would have called it, which may differ
    from `proposed` even when it agrees closely enough. It is recorded
    and NOT used to relabel the observation: relabelling would let a
    model move a record between classes the tables gate separately.
    """

    agrees: bool
    proposed: str
    label: str | None
    score: float | None
    model: str
    reason: str

    def to_json_dict(self) -> dict:
        return {
            "agrees": bool(self.agrees),
            "proposed": self.proposed,
            "label": self.label,
            "score": self.score,
            "model": self.model,
            "reason": self.reason,
        }


@runtime_checkable
class Verifier(Protocol):
    """Anything that can second-guess a label on a crop.

    The same load/act/release shape as `tower/detection.py`'s `Detector`,
    for the same reason: whatever holds a model has to be loadable late
    and releasable early, and a protocol says so without anybody
    inheriting anything.
    """

    name: str

    def load(self) -> None: ...

    def verify(self, crop_bgr, proposed_class: str) -> Verdict: ...

    def release(self) -> None: ...


class RefusingVerifier:
    """Agrees with nothing. The default, and not a stub.

    A Tower with no semantic model genuinely cannot tell a remote from a
    laptop keyboard, and this is what saying so looks like. It makes the
    no-verifier configuration a real, tested path rather than a `None`
    check scattered through the producer -- and it means the behaviour of
    a Tower without a model is exercised by the same tests as one with.
    """

    name = "none"

    def load(self) -> None:
        return None

    def verify(self, crop_bgr, proposed_class: str) -> Verdict:
        return Verdict(
            agrees=False,
            proposed=proposed_class,
            label=None,
            score=None,
            model=self.name,
            reason="no-verifier-configured",
        )

    def release(self) -> None:
        return None


class ScriptedVerifier:
    """Returns verdicts a test wrote down. Not a mock of any real model.

    The same argument `FixedDetector` makes: it lets a producer test
    assert against INDEPENDENT truth, because the answers are ones the
    test chose, and it keeps the default suite from downloading 692 MB
    of weights.
    """

    name = "scripted"

    def __init__(self, agrees_with=(), *, delay_seconds: float = 0.0) -> None:
        self._agrees = frozenset(agrees_with)
        self._delay_seconds = delay_seconds
        self.calls: list[str] = []

    def load(self) -> None:
        return None

    def verify(self, crop_bgr, proposed_class: str) -> Verdict:
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        self.calls.append(proposed_class)
        agrees = proposed_class in self._agrees
        return Verdict(
            agrees=agrees,
            proposed=proposed_class,
            label=proposed_class if agrees else None,
            score=1.0 if agrees else 0.0,
            model=self.name,
            reason="scripted",
        )

    def release(self) -> None:
        return None


@dataclass
class _Job:
    sighting: object
    crop: object
    submitted_at: float


class VerificationQueue:
    """Runs a verifier off the frame path, with a bounded backlog.

    One worker thread, not a pool. The models this is for hold weights,
    and two threads through one model on one device is contention rather
    than throughput; if the rate ever justifies more, the finding will be
    a measured backlog, and the backlog is counted here.

    A synchronous mode exists (`workers=0`) and it is not a debug flag:
    a replay of a recorded capture has no reason to be asynchronous, and
    a test that had to wait for a thread would be a test that sometimes
    does not.
    """

    def __init__(
        self,
        verifier,
        *,
        max_pending: int = DEFAULT_MAX_PENDING,
        workers: int = 1,
        clock=time.monotonic,
    ) -> None:
        self._verifier = verifier
        self._max_pending = max_pending
        self._clock = clock
        self._synchronous = workers < 1

        self._pending: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        # Jobs submitted whose verdict has not yet been PUBLISHED.
        #
        # One counter rather than "the queue is empty and nothing is in
        # flight", because those are two observations and there were two
        # windows between them. It was `_in_flight`, incremented after
        # `_pending.get()` returned -- so a job could be off the queue and
        # not yet counted, and `wait_idle` would say idle while a verdict
        # was still coming. Moving the decrement past `_done.put` closed
        # the tail of that window and left the head open; a single counter
        # incremented BEFORE the put and decremented AFTER the publish has
        # no window at either end.
        self._outstanding = 0
        self._lock = threading.Lock()

        self.submitted = 0
        self.completed = 0
        self.agreed = 0
        self.refused = 0
        self.dropped_backlog = 0
        self.failed = 0
        self.peak_pending = 0
        self.verify_seconds = 0.0

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        self._verifier.load()
        if self._synchronous:
            return
        self._thread = threading.Thread(
            target=self._run, name="object-memory-verify", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop accepting work and wait for what is in flight.

        Waits rather than abandoning, because a verdict that arrives
        after the producer has exited is a sighting silently not written.
        Bounded, because a model that has hung must not hold a session
        open forever -- and the bound is generous relative to any
        per-crop cost this is meant to run.
        """
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                with self._lock:
                    outstanding = self._outstanding
                logger.warning(
                    "[Tower][ObjectMemory] the verifier did not finish within "
                    "%.0fs; %s sightings will not be verified",
                    timeout,
                    outstanding,
                )
        self._verifier.release()

    # -- work ---------------------------------------------------------

    def submit(self, sighting, crop) -> bool:
        """Queue one sighting for a verdict. False if it was dropped.

        Dropping the OLDEST is deliberate. A backlog means the verifier
        is behind the walk, and the sighting most likely to still be
        worth verifying is the newest one -- the wearer is looking at it
        now. Dropping the newest would make the queue a device for
        remembering only the beginning of a walk.
        """
        if self._stopping.is_set():
            return False
        if self._synchronous:
            self.submitted += 1
            # Counted here too, and it must be: `_run_one` decrements in
            # a `finally`, so a synchronous submit that skipped the
            # increment drove the counter negative and `wait_idle` --
            # which asks for exactly zero -- never returned again. Seven
            # engine tests sat out a 30-second timeout apiece before this
            # line existed.
            with self._lock:
                self._outstanding += 1
            self._run_one(_Job(sighting, crop, self._clock()))
            return True

        while self._pending.qsize() >= self._max_pending:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                break
            self.dropped_backlog += 1
            # A dropped job will never publish a verdict, so it must stop
            # being outstanding here or `wait_idle` waits forever for one.
            with self._lock:
                self._outstanding -= 1
            logger.warning(
                "[Tower][ObjectMemory] verification backlog full (%s); the "
                "oldest waiting sighting was dropped and will not be "
                "remembered",
                self._max_pending,
            )
        self.submitted += 1
        # Counted BEFORE it reaches the queue, so there is never a moment
        # when a job exists and nothing knows about it.
        with self._lock:
            self._outstanding += 1
        self._pending.put(_Job(sighting, crop, self._clock()))
        self.peak_pending = max(self.peak_pending, self._pending.qsize())
        return True

    def drain(self) -> list:
        """Every sighting whose verdict has arrived since the last call.

        Pulled by the producer on the frame path rather than pushed by
        the worker, so nothing mutates a sighting from another thread.
        The producer owns its sightings; the queue owns only the crop it
        was handed and the verdict it produces.
        """
        finished = []
        while True:
            try:
                finished.append(self._done.get_nowait())
            except queue.Empty:
                return finished

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until every submitted job has PUBLISHED its verdict.

        For the end of a session and for tests. A producer that exited
        without waiting would discard verdicts it had already paid for.

        Reads one counter rather than two conditions. "The queue is empty
        and nothing is in flight" is two observations with a gap at each
        end of the worker's handling of a job, and both gaps were real.
        """
        deadline = self._clock() + timeout
        while self._clock() < deadline:
            with self._lock:
                if self._outstanding == 0:
                    return True
            time.sleep(0.01)
        with self._lock:
            return self._outstanding == 0

    def counters(self) -> dict:
        return {
            "verifier": self._verifier.name,
            "submitted": self.submitted,
            "completed": self.completed,
            "agreed": self.agreed,
            "refused": self.refused,
            "failed": self.failed,
            # The measurement that says whether the funnel is holding.
            # Non-zero means the semantic stage could not keep up with a
            # walk, and sightings went unremembered because of it.
            "dropped_backlog": self.dropped_backlog,
            "peak_pending": self.peak_pending,
            "seconds": round(self.verify_seconds, 3),
            "ms_per_call": (
                round(self.verify_seconds * 1000 / self.completed, 1)
                if self.completed
                else None
            ),
        }

    # -- internals ----------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                job = self._pending.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue
            self._run_one(job)

    def _run_one(self, job: _Job) -> None:
        """Verify one job, publish the verdict, then stop counting it.

        The `finally` is not optional. An earlier version dropped it to
        get the publish before the decrement, and that leaked the counter
        forever on any `BaseException` -- a `KeyboardInterrupt` inside the
        model would leave `wait_idle` unable to return True again, and in
        threaded mode the worker would die silently holding the count.
        Publishing inside the `try` and decrementing in the `finally`
        gets both.
        """
        started = self._clock()
        try:
            try:
                verdict = self._verifier.verify(
                    job.crop, job.sighting.object_class
                )
            except Exception:
                # Counted, not swallowed, and NOT treated as agreement. A
                # verifier that raised has said nothing, and "said
                # nothing" must resolve the same way as "said no" --
                # otherwise a broken model becomes a way to widen the
                # policy.
                self.failed += 1
                logger.exception(
                    "[Tower][ObjectMemory] the verifier raised on a %s "
                    "sighting; it will not be remembered",
                    job.sighting.object_class,
                )
                verdict = Verdict(
                    agrees=False,
                    proposed=job.sighting.object_class,
                    label=None,
                    score=None,
                    model=getattr(self._verifier, "name", "unknown"),
                    reason="verifier-failed",
                )
            self.verify_seconds += self._clock() - started
            self.completed += 1
            if verdict.agrees:
                self.agreed += 1
            else:
                self.refused += 1
            self._done.put((job.sighting, verdict))
        finally:
            with self._lock:
                self._outstanding -= 1


# --- the verifier this build actually offers ----------------------------

# `google/owlv2-base-patch16-ensemble`. Apache-2.0, 155 M parameters,
# ~600 MB of weights downloaded on first use, and reachable through plain
# `transformers` with no compiled operator, no MMCV and no compiler --
# which matters on Windows and matters more on a Blackwell card, where
# prebuilt CUDA extensions for older architectures do not load at all.
#
# CHOSEN ON MEASURED FITNESS, NOT ON A LEADERBOARD.
#
# Benchmarked against `iSEE-Laboratory/llmdet_tiny` -- the stronger model
# on LVIS rare-class AP, and the one a survey of the literature picks --
# over 94 human-labelled crops from this corpus
# (`scripts/research/open_vocab_verifier_bench.py`):
#
#     model         accept correct  reject wrong  median ms  peak VRAM
#     owlv2-base            0.949         0.857        126       842 MB
#     llmdet-tiny           0.407         1.000      3,091     1,643 MB
#
# Both over the SHIPPED vocabulary. An earlier run of this benchmark used
# a hand-copied 34-word list while `verifier_vocabulary()` returned 31,
# so its figures described a configuration that does not ship; the bench
# now imports the list. Re-running changed nothing but llmdet's latency.
#
# The gap is architectural rather than incidental. OWLv2 embeds each
# prompt SEPARATELY and scores it against the image's object queries, so
# a vocabulary of thirty-one names produces thirty-one scores and the
# question "did the proposed label come first" has an answer. LLMDet is
# phrase grounding: the vocabulary is joined into one sentence and what
# comes back is TEXT SPANS -- "a set", "a pair", "a" -- which have to be
# mapped back onto class names by string matching. It also pays for that
# sentence, at 3.1 seconds a crop against 126 milliseconds, because
# cross-attention scales with text length.
#
# A model that answers a different question well is not the better model
# for this question.
#
# WHAT IT DOES NOT FIX. Every one of its false rejects is a crop of 5.3%
# of the frame or smaller -- three real remotes at 3.7-3.9%, called
# `computer mouse` and `cell phone`. The size floor the shipped detector
# has is not removed by a second opinion; it moves one stage later. On
# 360x640 source imagery that is a property of the pixels, and the fixes
# are upstream: a higher capture resolution, or tiled detection on the
# async path.
OWLV2_REPO = "google/owlv2-base-patch16-ensemble"

# Accept only if the proposed label ranks FIRST and scores at least this.
#
# Swept over the 94 labelled crops. 0.40 to 0.50 is a shoulder with its
# peak at 0.45: balanced 0.938, accepting ~93% of correct labels and
# rejecting ~94% of wrong ones, with two false accepts. Above 0.55
# acceptance collapses -- 0.576 at 0.60 -- because the small crops score
# low even when they are right.
#
# ~93% and ~94% rather than 93.2% and 94.3%, deliberately. 81% of the
# positives in that set are two block assertions -- `[True] * 24` for
# laptop and the same for cell phone -- so three significant figures
# would be more precision than the labels carry. An audit found the
# conclusion robust to relabelling: any single flip moves balanced
# accuracy by at most 0.015, and 0.45 stays optimal under every group
# relabelling tried. Two audits disagreed about how many adversarial
# flips it takes to push balanced accuracy below 0.90 -- one said seven,
# one said three -- so neither number is quoted here.
#
# It is still a threshold fitted to 94 crops from ONE home, and it should
# be re-measured against any corpus with a different camera, a different
# room, or a bystander in it.
OWLV2_MIN_SCORE = 0.45


class OwlV2Verifier:
    """Ranks a proposed label against a fixed vocabulary, on one crop.

    Holds a model, so it follows the same load/act/release shape as
    `tower/detection.py`'s detector, and for the same reason: whatever
    holds weights must be loadable late and releasable early.

    Deliberately answers the NARROWEST question that serves the need. It
    is not asked what is in the picture; the funnel has already produced
    a crop and a proposed name, and all that is left is whether the name
    survives against the alternatives. That is a much easier question,
    and asking it is why one model call per three hundred frames is
    enough.
    """

    name = "owlv2-base-patch16-ensemble"

    def __init__(
        self,
        *,
        device: str = "cuda",
        min_score: float = OWLV2_MIN_SCORE,
        vocabulary=None,
        prompt_for=None,
        repo: str = OWLV2_REPO,
    ) -> None:
        self._device = device
        self._min_score = min_score
        self._repo = repo
        # Injected rather than imported, so this class stays testable
        # without the policy, and so the policy stays where it is.
        self._vocabulary = tuple(vocabulary) if vocabulary else ()
        self._prompt_for = prompt_for or (lambda name: name)
        self._processor = None
        self._model = None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        # Local imports: `transformers` is an optional extra, and nothing
        # outside a verification path may require it. A Tower without it
        # must still start, still record, and still serve.
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        device = self._device
        if device.startswith("cuda") and not torch.cuda.is_available():
            # Reported, not silently honoured. A CUDA verifier that
            # quietly became a CPU one would turn a 128 ms call into a
            # 2.5-second one on a host whose report still says GPU.
            logger.warning(
                "[Tower][ObjectMemory] %s was asked for CUDA and this host "
                "has none; running on CPU, which measured 2.5 s a crop "
                "against 128 ms",
                self.name,
            )
            device = "cpu"
        self._device = device
        self._processor = AutoProcessor.from_pretrained(self._repo)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self._repo)
            .eval()
            .to(device)
        )
        logger.info(
            "[Tower][ObjectMemory] verifier %s loaded on %s over a "
            "%s-word vocabulary (accept at rank 1 and score >= %.2f)",
            self.name,
            device,
            len(self._vocabulary),
            self._min_score,
        )

    def verify(self, crop_bgr, proposed_class: str) -> Verdict:
        import cv2
        import torch
        from PIL import Image

        if self._model is None:
            self.load()
        prompt = self._prompt_for(proposed_class)
        if prompt not in self._vocabulary:
            # A class the vocabulary cannot express cannot be confirmed
            # by it. Refusing is the only honest answer and also the safe
            # one: this is the shape a policy change would take if it
            # added a class and forgot the prompt.
            return Verdict(
                agrees=False,
                proposed=proposed_class,
                label=None,
                score=None,
                model=self.name,
                reason="not-in-verifier-vocabulary",
            )

        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(
            text=[list(self._vocabulary)], images=image, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, target_sizes=[(image.height, image.width)]
        )[0]

        best: dict[str, float] = {}
        for index, score in zip(
            results["labels"].tolist(), results["scores"].tolist()
        ):
            name = self._vocabulary[int(index)]
            best[name] = max(best.get(name, 0.0), float(score))
        if not best:
            return Verdict(
                agrees=False,
                proposed=proposed_class,
                label=None,
                score=None,
                model=self.name,
                reason="nothing-matched",
            )

        top_label, top_score = max(best.items(), key=lambda kv: kv[1])
        agrees = top_label == prompt and top_score >= self._min_score
        return Verdict(
            agrees=agrees,
            proposed=proposed_class,
            # What the verifier would have called it. Recorded and NOT
            # used to relabel the observation: relabelling would let a
            # model move a record between classes the tables gate
            # separately.
            label=top_label,
            score=round(top_score, 4),
            model=self.name,
            reason=(
                "ranked-first"
                if agrees
                else ("below-threshold" if top_label == prompt else "outranked")
            ),
        )

    def release(self) -> None:
        was_cuda = str(self._device).startswith("cuda")
        self._model = None
        self._processor = None
        if was_cuda:
            import torch

            torch.cuda.empty_cache()
