# Object detection: the device that was right on an idle box, and the people who were the wearer

**2026-08-29.** Two findings from the first physical CV Lab session, both
investigated because the numbers alone could not distinguish a bug from a fact.

---

## 1. `auto` resolved to CPU, and 29 ms became 199 ms

### What was reported

Physical run, Ray-Ban Meta → iPhone → Tower:

```
backend            torch
device_requested   auto
device             cpu
processing mean    199 ms/frame
capacity           5.03 fps
measured           3.67 fps
worst frame        4 483 ms
```

On the same Tower, `depth` resolves `auto` to `cuda` and runs at 9.4 ms.

### Why it was CPU

Deliberately. Commit `0a755d7` (2026-08-27) pinned `auto` to CPU for this one
experiment, and its measurement was good:

> Measured at the delivered 360x640, same-process interleaved A/B, 480 timed
> frames per device, 30 warm-up frames each so CUDA context creation is
> excluded, block order alternated:
>
> ```
> object_detection   cpu 29.41 ms   cuda 38.17 ms   CPU faster
> depth              cpu 20.03 ms   cuda 10.41 ms   CUDA faster
> ```
>
> CUDA lost every one of eight blocks for object_detection and won every one
> for depth.

The explanation was sound: MobileNetV3 at an internal 320 px is bound by
kernel-launch overhead rather than by arithmetic, so a GPU has nothing to win.
It also reclaimed 196 MB of VRAM.

### Why 29 ms became 199 ms

**The idle case does not happen.** That measurement was taken with nothing else
on the machine. A Tower ships with `scene_autostart` on and `scene_device` set
to `"cpu"`, so the box this experiment actually runs on always has a second
CPU-resident detector on it — and `config.py`'s own `scene_torch_threads`
comment already records that the torch thread pool is process-global and
shared.

Reproduced on this machine under CPU contention (10-18 concurrent Python
processes, which is itself an honest description of a working Tower):

| | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| CPU, contended | 818 ms | 327 ms | 3 488 ms | 4 247 ms | 4 544-7 803 ms |
| CPU, high priority | 128 ms | 85 ms | — | — | 479 ms |
| **CUDA, same contention** | **43.9 ms** | **44.1 ms** | ~50 ms | ~50 ms | **50.3 ms** |

CUDA was measured under the same contention that was producing multi-second CPU
stalls, and did not notice it.

### The 4.48 second frame was not a warm-up

Forty consecutive CPU inferences on an already-loaded model:

```
0 4388   1 4916   2 6137   3 5832   4 7077   5 4104  ...  29 3326
30 1027  31  436  32  818  33  882  34  211  ...  39  385
```

Elevated latency persisted for **thirty consecutive frames** and then decayed as
other work on the machine eased. A warm-up is paid once. This is a recurring,
load-correlated stall — which matters more here than almost anywhere, because
`CVLab.process()` runs synchronously on the event loop, so a 4.5-second frame is
4.5 seconds in which the Tower answers no socket at all.

CUDA showed the opposite shape: one 528 ms first inference (context creation
plus cuDNN algorithm selection), then 39-50 ms for every frame after.

### What changed

`auto` prefers CUDA again, and a throwaway inference now runs at load time on
CUDA only, on the worker thread the module's load timeout already bounds. Half a
second added to an arm that already downloads weights is invisible; half a second
added to the wearer's first frame is a stall they feel and a figure that poisons
`processing_ms_max` for the life of the run.

The trade is: give up about 9 ms in a condition that does not occur, to avoid
losing 150 ms and multi-second event-loop stalls in the one that does. Output is
equivalent across devices — same labels, boxes agreeing to 0.018 px, scores to
0.0004 — so nothing is bought with correctness. `TOWER_CV_DEVICE=cpu` still
forces CPU.

**Caveat, recorded rather than hidden.** The contended CPU figures above were
taken on a machine that this lane's own subagents were loading. They are
therefore an example of contention rather than a measurement of the Tower's
contention. The decisive evidence is the *physical* run — 199 ms mean, 4 483 ms
worst, on real hardware doing real work — and the CUDA figures, which were
stable under whatever the machine was doing.

`config.py`'s `scene_device` comment still says CUDA is 30.4 ms against CPU's
32.9 for this model. That is a third figure, on a third harness, and it is left
standing rather than edited. A successor re-measuring `scene_device` should know
all three exist, and should measure under load.

---

## 2. 160 `person` detections in a room with nobody in it

### What was reported

`count_person: 160` over 479 frames, and later 281 over 716, in scenes the
wearer described as empty. Cumulative across frames, not 281 distinct people —
but still a detector repeatedly calling something a person.

### What it is not

Four hypotheses were tested directly and all four are ruled out.

**Off-by-one in the class map.** `_COCO_CATEGORIES` has `__background__` at index
0 and `person` at index 1, exactly as suspected. But `self._categories` is
`weights.meta["categories"]` unmodified — no slice, no offset — and both the
model's `labels` output and `categories.index("person")` index into that same
list. Self-consistent by construction. Verified by instantiation:
`categories[0:3] == ['__background__', 'person', 'bicycle']`,
`categories.index("person") == 1`.

**BGR fed as RGB.** The code does `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`.
Tested the bug anyway by skipping it: total kept detections **fell** 215 → 160
and `person` **fell** 61 → 17. The wrong channel order degrades confidence; it
does not manufacture people.

**Missing 0-1 normalisation.** `weights.transforms()` is the official
`ObjectDetection` preset, which is exactly `convert_image_dtype(img, float)` —
no resize, no normalisation, both of which happen inside the model. Tested the
bug: feeding float `[0, 255]` explodes detections 215 → 1 445 and `person`
61 → 947, and on a **blank grey wall** produces 14 person detections at up to
0.994. That is the shape of the reported symptom — and the code does not contain
it.

**Weak-detector noise near the threshold.** Only 7 of 61 kept `person` boxes
(11%) score below 0.5. These are not near-misses.

### What it is

The wearer.

| class | n | median score | median area | median bottom edge | touching bottom |
|---|---|---|---|---|---|
| person | 61 | 0.655 | 47.7% | `y2/H = 0.990` | **87%** |
| laptop | 56 | 0.929 | 28.8% | — | — |
| cell phone | 35 | 0.803 | 8.1% | — | — |

Median horizontal centre `cx = 0.497`. Bottom-anchored, dead-centred, half the
frame, confidently scored: the signature of a head-mounted camera looking at its
own wearer's hands, forearms and torso. The highest-scoring example (0.980) was
opened directly and is the wearer's two hands holding a phone in an otherwise
empty bedroom.

Two of this repository's own earlier research documents independently found the
same thing on far larger samples —
`2026-08-26-real-corpus-first-measurement.md` (223 person boxes, median bottom
edge 0.981, 59% bottom-touching) and
`2026-08-26-detector-oracle-and-the-size-floor.md`, where **81.3% of shipped
person boxes were confirmed by `fasterrcnn_resnet50_fpn_v2` at IoU ≥ 0.5**. A
second, larger model agrees they are really there.

"No people present" meant "no bystanders", and it was true. The wearer's own
limbs are unavoidably `person`-class content to a COCO detector, and this
pipeline has no wearer-versus-bystander distinction — the oracle document is
explicit that there is no natural boundary and no confirmed bystander in the
corpus to validate one against.

### What changed

Nothing in the detector. No threshold was moved, no class was suppressed, no
preprocessing was touched — there was no defect to fix.

What changed is that you can now see it. `object_detection` declares
`preview_kind = "detections"` and the live view draws every box with its class
and score, over a line drawing of the room, with below-threshold detections
faded rather than hidden. A `person` box wrapped around a forearm is a fact
somebody can act on in one glance. `count_person: 160` never was.

**On synthetic content, the detector hallucinates nothing:** a blank grey wall,
uniform noise, a rectangles-as-furniture scene and the ArUco board produce zero
person detections between them under the real pipeline.

---

## Files

- `tower/experiments/object_detection.py` — the device comment now carries both
  measurements and the reason they do not contradict each other.
- `tower/cv_lab/preview.py` — `_encode_detections`.
- `docs/agent-handoffs/CV-LAB-LIVE-VIEW-MAC-HANDOFF.md` §7 — what to confirm
  physically.
