"""Frozen on-disk conventions for World Builder.

Everything here is a contract, not a tuning knob. A value in this module
changes what already-persisted data *means*, so changing one is a schema
migration, not an edit.

The single most important rule this module encodes: a reader that meets a
convention it does not recognise **refuses**, it never guesses. A bare
``[x, y, z]`` interpreted under the wrong pose convention does not fail
loudly -- it produces a plausible, mirrored, completely wrong answer, and
it stays wrong forever once other records reference it.
"""

# Bump when the meaning of any persisted field changes. An integer, not
# semver: there are no compatibility ranges here worth reasoning about.
# Readers refuse a version they do not know (see store.require_schema).
SCHEMA_VERSION = 1

# There is no capture timestamp anywhere on the wire (tower/frames.py has
# no time field), so every timestamp World Builder writes is Tower-receipt
# time. Naming the basis on every record keeps it from being read as
# capture time once persisted.
TIME_BASIS = "tower-receipt"

# The pose contract, written into every world.json so the artifact is
# self-describing rather than depending on this file still existing in the
# form the writer had.
#
# T_world_camera: its translation IS the camera position in world
# coordinates, so no consumer ever computes -R.T @ t. Getting that inversion
# wrong mirrors every camera through the origin, a plausible-looking map.
#
# OpenCV camera axes because every cv2 calibration, solvePnP and projection
# call already assumes them; anything else means converting at the place
# that is easiest to get wrong.
POSE_CONVENTION = {
    "pose_type": "T_world_camera",
    "quaternion_order": "wxyz",
    "handedness": "right",
    "camera_axes": "opencv_x_right_y_down_z_forward",
    "translation_units": "world",
    "world_axes_origin": "first_keyframe_camera",
    # No IMU is exposed by DAT, so there is no gravity observation and
    # "up" is genuinely unknown; declaring y-up would be fabricated.
    # Upgraded only by an actual floor-plane estimate.
    "up_axis": "unknown",
    "pose_dtype": "float64",
    "point_dtype": "float32",
}

# Pose availability. A backend that cannot justify a pose returns
# UNAVAILABLE plus a degeneracy reason -- that is a first-class answer,
# not an error, and it is what lets the whole engine run honestly on
# uncalibrated footage.
POSE_STATUS_UNAVAILABLE = "unavailable"
POSE_STATUS_SOLVED = "solved"
POSE_STATUS_ROTATION_ONLY = "rotation_only"
POSE_STATUS_ANCHOR = "anchor"

# Why a pose was refused. Recorded per keyframe so a thin reconstruction
# is explainable rather than merely empty.
DEGENERACY_NONE = ""
DEGENERACY_PURE_ROTATION = "pure_rotation"
DEGENERACY_LOW_PARALLAX = "low_parallax"
DEGENERACY_NO_CORRESPONDENCE = "no_correspondence"
DEGENERACY_NO_INTRINSICS = "no_intrinsics"

# Scale states. "relative" means the reconstruction is internally
# consistent but its unit is arbitrary -- fixed by whatever baseline the
# first solved pair happened to have. It is NOT metric and must never be
# rendered in metres. "measured" is the only state that licenses a metre.
SCALE_UNKNOWN = "unknown"
SCALE_RELATIVE = "relative"
SCALE_ESTIMATED = "estimated"
SCALE_MEASURED = "measured"

# Only a measured scale licenses printing a physical unit. Checked in one
# place (records.format_distance) and pinned by a test, because "we will
# remember not to print metres" is not a mechanism.
SCALE_STATES_ALLOWING_METRES = (SCALE_MEASURED,)

# Where intrinsics came from. "unknown" is what CameraIntrinsics.unknown()
# produces. There is deliberately no value meaning "guessed": the published
# 100-degree Ray-Ban FOV describes a 3:4 still, while the stream is 9:16
# through an undocumented crop, so no legitimate FOV conversion exists.
INTRINSICS_SOURCE_UNKNOWN = "unknown"
INTRINSICS_SOURCE_SELF_CALIBRATED = "self_calibrated"
INTRINSICS_SOURCE_DECLARED = "declared"

# Gauge entries. Nothing advances frame_revision past 1 yet, so no gauge
# entry is written. The rule any future writer must follow: a coordinate
# stamped revision R can be brought current only if EVERY entry from R to
# HEAD is a global similarity. A loop closure that re-anchors one submap
# moves part of the world, not all of it, so otherwise the coordinate is
# re-resolved from its anchor keyframe or reported as unknown, never
# composed forward.

# Session end reasons. "interrupted" is what recovery stamps on a session
# whose process died; its ended_at stays null because we genuinely do not
# know when it stopped, and unknown stays unknown.
END_REASON_STOP = "stop"
END_REASON_DISCONNECT = "disconnect"
END_REASON_INTERRUPTED = "interrupted"
END_REASON_ERROR = "error"
END_REASON_BOUNDED_LIMIT = "bounded_limit"

