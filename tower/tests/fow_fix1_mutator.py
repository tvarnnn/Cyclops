"""Negative-control mutants for the FOW fix-round verification."""

import copy
import os


def pytest_configure(config):
    from tower.results import world_builder as status
    from tower.world_builder import guidance_coverage as coverage

    mutant = os.environ["FOW_MUTANT"]
    if mutant == "poll_stat":
        original = status.WorldBuilderStatusProducer._coverage

        def broken(self, *args):
            os.stat(self._root)
            return original(self, *args)

        status.WorldBuilderStatusProducer._coverage = broken
    elif mutant == "late_toggle":
        original = status.WorldBuilderStatusProducer.__init__

        def broken(self, *args, **kwargs):
            kwargs["coverage_enabled"] = None
            return original(self, *args, **kwargs)

        status.WorldBuilderStatusProducer.__init__ = broken
    elif mutant == "discovery_exits":
        original = coverage.CoverageWorker._run

        def broken(self):
            self._discover()
            return original(self)

        coverage.CoverageWorker._run = broken
    elif mutant == "missing_placement":
        original = coverage.compute_coverage

        def broken(**kwargs):
            refs = {c["reference_segment"] for c in kwargs["manifest"]["global_solve"]["components"]}
            placed = {p["segment_index"] for p in kwargs["placements"]}
            if refs - placed:
                raise ValueError("missing component reference placement")
            return original(**kwargs)

        coverage.compute_coverage = broken
    elif mutant == "unreliable_placement":
        original = coverage.compute_coverage

        def broken(**kwargs):
            kwargs["placements"] = copy.deepcopy(kwargs["placements"])
            for placement in kwargs["placements"]:
                placement["evidence"].pop("placement_reliable", None)
                placement["evidence"].pop("scale_reliable", None)
            return original(**kwargs)

        coverage.compute_coverage = broken
    elif mutant == "nonfinite_component":
        coverage._finite_component = lambda component: True
    elif mutant == "stale_revision":
        original = status.geometry_revision_from_manifest
        cached = []

        def broken(manifest):
            if not cached:
                cached.append(original(manifest))
            return cached[0]

        status.geometry_revision_from_manifest = broken
    elif mutant == "oversize_status":
        status._coverage_status_fits = lambda payload: True
    elif mutant == "boolean_clock":
        original = coverage.compute_coverage

        def broken(**kwargs):
            kwargs["computed_at"] = float(kwargs["computed_at"])
            return original(**kwargs)

        coverage.compute_coverage = broken
    else:
        raise ValueError(mutant)
