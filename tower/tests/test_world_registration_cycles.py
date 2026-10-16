"""Cycle consistency: the first independent check registration can run.

Every gate in `admit()` judges ONE pair using evidence from that pair.
None of them can see an error that only shows up going round a loop --
which is exactly the failure the research note calls the dangerous one,
because a wrong Sim3 fits well and reads as a slightly odd floor plan.

`compose_tree` places segments along a spanning tree and its docstring
says so: "A spanning tree is used rather than a pose graph because on the
real walk the admitted subgraph has no cycle at all... When cycles do
appear, the right next step is cycle-consistency CHECKING, which is free
and independent." It is the first thing to add when a cycle appears.

A cycle has appeared. After the solve-chain segmentation change, capture
`2e6cffa2` admits (12,16), (12,19) and (16,19) -- a triangle. Measured on
it, composing 12->16->19 against the direct 12->19 disagrees by 6.0% in
scale, 5.899 degrees in rotation, and 5.13 in translation, while every
one of those three edges passed reciprocity on its own.
"""

import math

import numpy as np
import pytest

from scripts.world_registration import (
    MAX_CYCLE_ROTATION_DEG,
    MAX_CYCLE_SCALE_RATIO,
    Sim3,
    cycle_residuals,
)


def _rot_z(degrees):
    t = math.radians(degrees)
    return np.array(
        [[math.cos(t), -math.sin(t), 0.0],
         [math.sin(t), math.cos(t), 0.0],
         [0.0, 0.0, 1.0]]
    )


def _sim3(scale=1.0, degrees=0.0, translation=(0.0, 0.0, 0.0)):
    return Sim3(scale, _rot_z(degrees), np.asarray(translation, dtype=float))


def _triangle(closure):
    """Edges a->b, b->c and the closing a->c, with `closure` as the
    direct a->c transform. Placements are composed along a->b->c, so a
    consistent closure is exactly the composition."""
    ab = _sim3(scale=2.0, degrees=10.0, translation=(1.0, 0.0, 0.0))
    bc = _sim3(scale=1.5, degrees=20.0, translation=(0.0, 2.0, 0.0))
    edges = [
        (0, 1, ab.scale, ab.rotation, ab.translation),
        (1, 2, bc.scale, bc.rotation, bc.translation),
        (0, 2, closure.scale, closure.rotation, closure.translation),
    ]
    placements = {0: _sim3(), 1: ab, 2: bc.compose(ab)}
    tree = {(0, 1), (1, 2)}
    return edges, placements, tree


def test_a_consistent_cycle_has_no_residual():
    """The control. If a closing edge agrees with the path around the
    loop, the check must say nothing -- otherwise it refuses honest
    clusters."""
    ab = _sim3(scale=2.0, degrees=10.0, translation=(1.0, 0.0, 0.0))
    bc = _sim3(scale=1.5, degrees=20.0, translation=(0.0, 2.0, 0.0))
    edges, placements, tree = _triangle(bc.compose(ab))

    residuals = cycle_residuals(edges, placements, tree)
    assert len(residuals) == 1, "the closing edge is the only cycle here"
    residual = residuals[0]
    assert residual["scale_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert residual["rotation_deg"] == pytest.approx(0.0, abs=1e-6)
    assert residual["translation"] == pytest.approx(0.0, abs=1e-6)


def test_a_rotated_closure_is_caught():
    """Rotation error around a loop is invisible to every pairwise gate:
    each edge can agree with its own reverse solve while the loop does not
    close."""
    ab = _sim3(scale=2.0, degrees=10.0, translation=(1.0, 0.0, 0.0))
    bc = _sim3(scale=1.5, degrees=20.0, translation=(0.0, 2.0, 0.0))
    good = bc.compose(ab)
    bad = Sim3(good.scale, _rot_z(45.0) @ good.rotation, good.translation)

    residual = cycle_residuals(*_triangle(bad))[0]
    assert residual["rotation_deg"] == pytest.approx(45.0, abs=1e-6)


def test_a_rescaled_closure_is_caught():
    ab = _sim3(scale=2.0, degrees=10.0, translation=(1.0, 0.0, 0.0))
    bc = _sim3(scale=1.5, degrees=20.0, translation=(0.0, 2.0, 0.0))
    good = bc.compose(ab)
    bad = Sim3(good.scale * 3.0, good.rotation, good.translation)

    residual = cycle_residuals(*_triangle(bad))[0]
    assert residual["scale_ratio"] == pytest.approx(3.0, rel=1e-6)


def test_a_ratio_is_reported_the_same_way_up_whichever_way_it_is_wrong():
    """0.31x and 3.2x are the same disagreement, and a bar applied to the
    raw ratio would catch one and miss the other."""
    ab = _sim3(scale=2.0, degrees=10.0, translation=(1.0, 0.0, 0.0))
    bc = _sim3(scale=1.5, degrees=20.0, translation=(0.0, 2.0, 0.0))
    good = bc.compose(ab)

    over = cycle_residuals(*_triangle(Sim3(good.scale * 4.0, good.rotation, good.translation)))[0]
    under = cycle_residuals(*_triangle(Sim3(good.scale / 4.0, good.rotation, good.translation)))[0]
    assert over["scale_ratio"] == pytest.approx(under["scale_ratio"], rel=1e-6)


def test_edges_inside_the_spanning_tree_are_not_cycles():
    """Only the CLOSING edges are checkable. A tree edge compared against
    the placements built from it would trivially agree, and reporting that
    as a passing check would be a lie about how much evidence exists."""
    ab = _sim3(scale=2.0, degrees=10.0, translation=(1.0, 0.0, 0.0))
    edges = [(0, 1, ab.scale, ab.rotation, ab.translation)]
    placements = {0: _sim3(), 1: ab}
    assert cycle_residuals(edges, placements, {(0, 1)}) == []


def test_an_edge_touching_an_unplaced_segment_is_skipped():
    """Nothing to compose against, so nothing to check -- and inventing a
    residual for it would be fiction."""
    ab = _sim3(scale=2.0, degrees=10.0, translation=(1.0, 0.0, 0.0))
    edges = [(0, 7, ab.scale, ab.rotation, ab.translation)]
    placements = {0: _sim3()}
    assert cycle_residuals(edges, placements) == []


def test_the_bars_sit_between_measured_honest_and_documented_broken():
    """Set from evidence, like every other threshold on this branch.

    Measured honest, on the only real cycle that exists: 5.899 degrees and
    a 1.06x scale ratio. Documented broken, from the registration research:
    wrong rotations of 31.9 to 166.0 degrees and a scale 3.2x out.
    """
    assert 5.899 < MAX_CYCLE_ROTATION_DEG < 31.9
    assert 1.06 < MAX_CYCLE_SCALE_RATIO < 3.2


def test_a_cluster_with_a_broken_loop_is_refused_whole():
    """Refusal is on the CLUSTER, not the closing edge.

    The closure is not in the spanning tree, so dropping it would change
    no placement and leave the bad edge in place. A cycle proves an
    inconsistency exists without saying which edge carries it.
    """
    from scripts.world_registration import cycle_refusal_for

    assert cycle_refusal_for([]) is None
    assert cycle_refusal_for([
        {"edge": (0, 2), "rotation_deg": 3.0, "scale_ratio": 1.1,
         "translation": 0.2},
    ]) is None

    reason = cycle_refusal_for([
        {"edge": (0, 2), "rotation_deg": 45.0, "scale_ratio": 1.1,
         "translation": 0.2},
    ])
    assert reason is not None and "does not close" in reason
    assert "(0, 2)" in reason

    assert cycle_refusal_for([
        {"edge": (1, 4), "rotation_deg": 2.0, "scale_ratio": 3.2,
         "translation": 0.2},
    ]) is not None


def test_the_real_measured_residual_would_not_be_refused():
    """The one honest cycle in the corpus must survive the bars, or the
    check refuses the only cluster it has ever been able to examine."""
    from scripts.world_registration import cycle_refusal_for

    assert cycle_refusal_for([
        {"edge": (12, 16), "rotation_deg": 5.899, "scale_ratio": 1.0599,
         "translation": 22.427},
    ]) is None
