"""Safety gates for the offline local-chain candidate."""

import numpy as np

from scripts.world_live_placement_v11b import chain_pose


def _anchor(key, x, at, index):
    pose = (np.array([float(x), 0., 0.]), np.eye(3))
    return key, pose, pose, at, index, 0


def test_chain_requires_two_observable_same_segment_anchors_and_reanchors():
    anchors = [_anchor("a", 0., 0., 0), _anchor("b", 1., 1., 1)]
    query = (np.array([1.2, 0., 0.]), np.eye(3))
    predicted, reason = chain_pose(anchors, query, 1.2, 2)
    assert reason is None
    assert np.allclose(predicted[0], [1.2, 0., 0.])
    assert chain_pose(anchors[:1], query, 1.2, 2)[1] == "fewer-than-two-anchors"
    # The latest anchor, not the first one, is the origin of propagation.
    shifted = [anchors[0], ("b", anchors[1][1],
                            (np.array([1.2, 0., 0.]), np.eye(3)), 1., 1, 0)]
    assert np.allclose(chain_pose(shifted, query, 1.2, 2)[0][0], [1.44, 0., 0.])


def test_chain_expires_on_time_steps_travel_or_unobservable_scale():
    anchors = [_anchor("a", 0., 0., 0), _anchor("b", 1., 1., 1)]
    near = (np.array([1.2, 0., 0.]), np.eye(3))
    assert chain_pose(anchors, near, 3.01, 2)[1] == "drift-budget-expired"
    assert chain_pose(anchors, near, 1.2, 8)[1] == "drift-budget-expired"
    assert chain_pose(anchors, (np.array([2., 0., 0.]), np.eye(3)), 1.2, 2)[1] == "drift-budget-expired"
    tiny = [_anchor("a", 0., 0., 0), _anchor("b", .01, 1., 1)]
    assert chain_pose(tiny, near, 1.2, 2)[1] == "unobservable-scale"
