"""Negative-control mutants for Exp3's oracle mirror-mask module (`mirror_masks.py`).

    EXP3_MUTANT=rasterize_last_only python -m pytest -p exp3_mirror_mutator \
        tests/test_world_builder_mirror_masks.py
    EXP3_MUTANT=untouched_is_zero python -m pytest -p exp3_mirror_mutator \
        tests/test_world_builder_mirror_masks.py
    EXP3_MUTANT=no_shape_check python -m pytest -p exp3_mirror_mutator \
        tests/test_world_builder_mirror_masks.py

Each must turn at least one green test in `test_world_builder_mirror_masks.py` red:
that is what "mutants killed" means here, not a count of assertions.
"""

import os

import numpy as np


def pytest_configure(config):
    from tower.world_builder import mirror_masks as MM

    mutant = os.environ["EXP3_MUTANT"]
    if mutant == "rasterize_last_only":
        # The union of disconnected visible regions becomes "only the LAST polygon
        # drawn": kills `test_rasterize_polygons_unions_disconnected_regions`.
        original = MM.rasterize_polygons

        def broken(polygons, width, height):
            return original(polygons[-1:], width, height) if polygons else \
                original(polygons, width, height)

        MM.rasterize_polygons = broken
    elif mutant == "untouched_is_zero":
        # A keyframe outside the annotated range becomes an exclusion of nothing
        # (indistinguishable from an explicit empty record) instead of None: kills
        # `test_interior_mask_for_frame_is_none_outside_the_annotated_range`.
        def broken(polygons_by_key, walk, source_seq, *, m1, m2, roi, out_shape):
            if walk is None or source_seq is None:
                return None
            record = polygons_by_key.get((int(walk), int(source_seq)))
            if record is None:
                return np.zeros(out_shape, bool)
            return MM.oracle_mask_in_solver_space(record, m1=m1, m2=m2, roi=roi,
                                                  out_shape=out_shape)

        MM.interior_mask_for_frame = broken
    elif mutant == "no_shape_check":
        # The remapped mask is returned uncropped-checked: a solver/dense camera
        # shape that disagrees with the polygon's frame is never caught. Kills
        # `test_oracle_mask_in_solver_space_raises_on_a_shape_mismatch`.
        import cv2

        def broken(record, *, m1, m2, roi, out_shape):
            raw = MM.rasterize_polygons(record["polygons"], record["width"], record["height"])
            x, y, rw, rh = roi
            remapped = cv2.remap(raw.astype(np.uint8) * 255, m1, m2, cv2.INTER_NEAREST)
            return remapped[y:y + rh, x:x + rw] > 0

        MM.oracle_mask_in_solver_space = broken
    else:
        raise ValueError(f"unknown EXP3_MUTANT {mutant!r}")
