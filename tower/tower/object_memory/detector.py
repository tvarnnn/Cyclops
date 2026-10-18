"""This cartridge's detector: the shared one, configured for a memory.

The `Detector` protocol, `Detection` and `FixedDetector` are platform code
in `tower/detection.py`; this file binds the SSDLite detector to Object
Memory's choices.

**The seam exists** because the real detector costs a model download that
a default test suite must not pay for, and the fake lets the producer be
tested against detections a test wrote down rather than against whatever
torchvision happened to find.

**Still not the Experimental CV Lab.** `experiments/object_detection.py`
loads these exact weights and measured them (38.6 ms median on CUDA
across the real 9,199-frame corpus), but it reports COUNTS and a memory
needs the individual detection. Sharing a platform module with Scene
Understanding is not the same as importing a sandbox: shared code is
code the platform maintains, a sandbox is code that may be thrown away,
and nothing that may be thrown away belongs upstream of a persistent
store.

**Nothing here reaches another cartridge.** The shared detector holds no
state and no model, so a detector that cannot load fails for THIS
producer and no other -- which is what lets the engine keep reporting its
refusals honestly instead of inheriting somebody else's failure.

Everything here is **model inference, not measured fact**
(`07-PLATFORM-CONSTRAINTS.md` Core Principle 2). A detection is evidence
that something scored above a threshold, and nothing more.
"""

from tower.detection import (
    SCORE_THRESHOLD,
    Detection,
    Detector,
    FixedDetector,
    SSDLite320Detector,
)

# Re-exported so this cartridge's callers keep importing from this
# cartridge.
__all__ = [
    "SCORE_THRESHOLD",
    "Detection",
    "Detector",
    "FixedDetector",
    "TorchvisionDetector",
]


class TorchvisionDetector(SSDLite320Detector):
    """The shared SSDLite detector, with this cartridge's two choices.

    **No class filter.** A memory of what was around cannot pre-judge
    which categories will turn out to matter, so every COCO class the
    model reports is offered to the relevance filter, which applies its
    own higher `min_score` and its own `PERSISTED_CLASSES` downstream.
    Scene Understanding narrows at the detector instead, because a live
    state cluttered with ninety-one classes buries the answer. The
    difference is the reason the class list is a parameter rather than a
    constant in shared code.

    **Its own weights.** Subclassing binds defaults; it does not share a
    model with anyone. Two cartridges running at once load 13.4 MB twice,
    which is cheaper than the failure mode a shared residency would
    introduce.
    """

    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            score_threshold=score_threshold,
            classes=None,
            device=device,
            owner="ObjectMemory",
        )
