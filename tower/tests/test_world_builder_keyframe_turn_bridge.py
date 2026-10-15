"""KF-TURNS: the keyframe blur floor and the turn bridge (W9-DIAG.md cause 1).

Walk 9's 9 component breaks all sit on a turn where the blur gate rejected
every frame between two keyframes 52-391 deg apart, with a ~45 deg FOV. The
World Builder engine always runs the turn bridge: a blurred frame is accepted
when the tracker says the view has turned far enough since the last keyframe.
A bare `KeyframePolicy()` keeps it off, so these tests pin the mechanism by
comparing the two.
"""

import random

import pytest

from tower.world_builder.frontend import FrameQuality, MotionSummary
from tower.world_builder.keyframes import (
    ACCEPT,
    REASON_BLURRED,
    REASON_TURN_BRIDGE,
    REJECT,
    KeyframePolicy,
    KeyframeSelector,
)


def _q(sharpness: float) -> FrameQuality:
    return FrameQuality(width=360, height=640, sharpness=sharpness)


def _m(overlap: float, *, disp: float = 20.0, tracked: int | None = None) -> MotionSummary:
    seeded = 200
    tracked = int(seeded * overlap) if tracked is None else tracked
    return MotionSummary(
        seeded_count=seeded,
        tracked_count=tracked,
        survival_ratio=tracked / seeded,
        overlap_ratio=overlap,
        median_displacement_px=disp if tracked else None,
        homography_residual_px=0.5 if tracked >= 12 else None,
    )


def _run(selector, frames):
    """Drive the selector as the engine does; returns the decisions."""
    out = []
    for quality, motion in frames:
        selector.note_frame(quality)
        decision = selector.evaluate(quality, motion)
        if decision.lost:
            selector.note_lost()
        elif decision.accepted:
            selector.note_accepted()
        out.append((decision.outcome, decision.reason))
    return out


def _seeded(policy):
    selector = KeyframeSelector(policy)
    _run(selector, [(_q(300.0), None)] + [(_q(300.0), _m(0.95, disp=1.0))] * 8)
    return selector


# -- configuration ------------------------------------------------------------


def test_policy_defaults():
    assert KeyframePolicy().min_sharpness == 25.0
    assert KeyframePolicy().turn_bridge is False


def test_engine_runs_the_turn_bridge(monkeypatch, tmp_path):
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.store import WorldStore

    class _NoRedactor:
        available = True
        unavailable_reason = None

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store, redactor_factory=_NoRedactor, frame_quality_log=False,
                                relocalizer="off")
    world = engine.create_world()
    engine.start_session(world)
    assert engine._selector.policy == KeyframePolicy(turn_bridge=True)
    engine.stop_session()

    # Nothing in the environment tunes the policy; a leftover demo line is harmless.
    for value in ("on", "off"):
        monkeypatch.setenv("TOWER_WORLD_KEYFRAME_TURN_BRIDGE", value)
        monkeypatch.setenv("TOWER_WORLD_KEYFRAME_MIN_SHARPNESS", "7")
        engine.start_session(world)
        assert engine._selector.policy == KeyframePolicy(turn_bridge=True)
        engine.stop_session()

    # An injected policy is used as given.
    injected = KeyframePolicy(min_sharpness=40.0)
    engine = WorldBuilderEngine(store, policy=injected, redactor_factory=_NoRedactor,
                                frame_quality_log=False, relocalizer="off")
    engine.start_session(engine.create_world())
    assert engine._selector.policy is injected
    engine.stop_session()


# -- default identity ---------------------------------------------------------


def _random_frames(seed: int, n: int = 3000):
    rng = random.Random(seed)
    frames = []
    for _ in range(n):
        sharp = rng.choice([0.5, 2.0, 5.0, 8.0, 15.0, 24.0, 26.0, 60.0, 150.0, 400.0])
        if rng.random() < 0.03:
            motion = None
        else:
            overlap = rng.choice([0.0, 0.03, 0.1, 0.3, 0.45, 0.6, 0.74, 0.8, 0.95, 1.0])
            motion = _m(overlap, disp=rng.choice([0.5, 3.0, 8.0, 20.0, 60.0, 150.0]),
                        tracked=rng.choice([None, 5, 0]) if rng.random() < 0.1 else None)
        frames.append((_q(sharp), motion))
    return frames


@pytest.mark.parametrize("seed", range(5))
def test_bridge_only_turns_blur_rejects_into_accepts(seed):
    """On, a frame the gate passes gets the same treatment as off; the only
    decisions that differ start at a blur reject turned into a bridge."""
    frames = _random_frames(seed)
    off = _run(KeyframeSelector(KeyframePolicy()), frames)
    on = _run(KeyframeSelector(KeyframePolicy(turn_bridge=True)), frames)
    first = next((i for i, (a, b) in enumerate(zip(off, on)) if a != b), None)
    if first is not None:
        assert off[first] == (REJECT, REASON_BLURRED)
        assert on[first] == (ACCEPT, REASON_TURN_BRIDGE)


# -- the floor ----------------------------------------------------------------


def test_floor_is_respected():
    low = KeyframeSelector(KeyframePolicy(min_sharpness=7.0))
    assert _run(low, [(_q(10.0), None)]) == [(ACCEPT, "session_seed")]
    low = KeyframeSelector(KeyframePolicy(min_sharpness=7.0))
    assert _run(low, [(_q(6.9), None)]) == [(REJECT, REASON_BLURRED)]
    today = KeyframeSelector(KeyframePolicy())
    assert _run(today, [(_q(10.0), None)]) == [(REJECT, REASON_BLURRED)]


# -- the bridge ---------------------------------------------------------------


def test_bridge_triggers_on_a_synthetic_turn():
    """A head turn: every frame after the keyframe is below the floor while
    the view slides off the reference. Off, nothing is taken until a sharp
    frame arrives -- by then the overlap is gone. On, the bridge takes a frame
    once overlap falls below its bound, and the chain keeps a link."""
    turn = [(_q(12.0), _m(o, disp=40.0)) for o in (0.9, 0.8, 0.65, 0.45, 0.3, 0.1, 0.0)]

    off = _run(_seeded(KeyframePolicy()), turn)
    assert off == [(REJECT, REASON_BLURRED)] * len(turn)

    on = _run(_seeded(KeyframePolicy(turn_bridge=True)), turn)
    assert on[:3] == [(REJECT, REASON_BLURRED)] * 3
    assert on[3] == (ACCEPT, REASON_TURN_BRIDGE)


def test_bridge_displacement_bound():
    policy = KeyframePolicy(turn_bridge=True)
    big = 0.16 * _q(12.0).diagonal_px
    out = _run(_seeded(policy), [(_q(12.0), _m(0.9, disp=big))] * 3)
    assert out[-1] == (ACCEPT, REASON_TURN_BRIDGE)


def test_bridge_never_takes_smear():
    policy = KeyframePolicy(turn_bridge=True)
    turn = [(_q(2.0), _m(0.1, disp=80.0))] * 30
    assert _run(_seeded(policy), turn) == [(REJECT, REASON_BLURRED)] * 30


def test_bridge_time_bound_when_motion_is_unmeasurable():
    policy = KeyframePolicy(turn_bridge=True)
    # Moving (displacement over the minimum) but not past either bound.
    frames = [(_q(12.0), _m(0.7, disp=20.0))] * 8
    out = _run(_seeded(policy), frames)
    first = out.index((ACCEPT, REASON_TURN_BRIDGE))
    assert first == policy.bridge_max_frames - 1


def test_no_flood_on_a_static_scene():
    """A still camera on a view below the floor (white wall, dim corner):
    the tracker says nothing moved, so nothing is bridged -- for minutes."""
    policy = KeyframePolicy(min_sharpness=25.0, turn_bridge=True)
    still = [(_q(12.0), _m(0.97, disp=1.0))] * 600
    assert _run(_seeded(policy), still) == [(REJECT, REASON_BLURRED)] * 600
    # Even on a texture-poor view with too few tracks for motion evidence.
    few = [(_q(12.0), _m(1.0, disp=1.0, tracked=6))] * 600
    assert _run(_seeded(policy), few) == [(REJECT, REASON_BLURRED)] * 600


def test_bridge_rate_is_bounded_in_a_sustained_blur():
    """A long blurred turn, every frame reading low overlap: at most one
    bridge per `bridge_min_frames` frames."""
    policy = KeyframePolicy(turn_bridge=True)
    turn = [(_q(12.0), _m(0.2, disp=60.0))] * 120
    out = _run(_seeded(policy), turn)
    accepts = sum(1 for o in out if o == (ACCEPT, REASON_TURN_BRIDGE))
    assert 0 < accepts <= len(turn) // policy.bridge_min_frames


def test_bridge_seeds_a_segment_only_after_a_sustained_run():
    policy = KeyframePolicy(turn_bridge=True)
    out = _run(KeyframeSelector(policy), [(_q(12.0), None)] * policy.bridge_max_frames)
    assert out[:-1] == [(REJECT, REASON_BLURRED)] * (policy.bridge_max_frames - 1)
    assert out[-1] == (ACCEPT, REASON_TURN_BRIDGE)


def test_a_raised_bridge_floor_refuses_a_sharpness_4_bridge():
    # The e2e9 fly-away at 295 s was a bridge frame of sharpness 4: floor 8 refuses it, floor 3 takes it.
    low = KeyframePolicy(turn_bridge=True)
    high = KeyframePolicy(turn_bridge=True, bridge_min_sharpness=8.0)
    frames = [(_q(4.0), None)] * (low.bridge_max_frames + 2)
    assert any(d == REASON_TURN_BRIDGE for _, d in _run(KeyframeSelector(low), frames)[1:])
    assert not any(d == REASON_TURN_BRIDGE for _, d in _run(KeyframeSelector(high), frames)[1:])
