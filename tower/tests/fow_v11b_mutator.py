"""Negative controls for the four FOW v1.1 B acceptance guards.

Run with FOW_V11B_MUTANT set to one of: b1_missing, route_off,
raw_source, b2_missing, b2_redate, cap_ignored, byte_cap_ignored,
next_push_no_wait.
Each selected fixture fails.
"""

import os


def pytest_configure(config):
    from tower.world_builder import guidance_imagery as imagery
    from tower.world_builder import guidance_coverage as coverage

    mutant = os.environ["FOW_V11B_MUTANT"]
    if mutant == "b1_missing":
        imagery.ImageryState.live = lambda self, root, world, session: None
    elif mutant == "route_off":
        from tower import main
        from tower.routes import guidance_imagery as route
        original = main.create_app

        def broken():
            app = original()
            app.include_router(route.router)
            return app

        main.create_app = broken
    elif mutant == "raw_source":
        imagery.IMAGERY_REDACTED = "raw-local-research"
    elif mutant == "b2_missing":
        original = coverage.CoverageWorker.__init__

        def broken(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self._imagery_landed = False

        coverage.CoverageWorker.__init__ = broken
    elif mutant == "b2_redate":
        original = coverage.CoverageWorker.offer

        def broken(self, world_id, session_id, solved_at, horizon,
                   solution_stat, geometry_revision, tree_tag):
            self._receipts.pop((world_id, session_id, solved_at,
                                horizon, solution_stat), None)
            return original(self, world_id, session_id, solved_at, horizon,
                            solution_stat, geometry_revision, tree_tag)

        coverage.CoverageWorker.offer = broken
    elif mutant == "cap_ignored":
        imagery.MAX_TILES = 1000
    elif mutant == "byte_cap_ignored":
        imagery.MAX_BYTES = 1 << 20
    elif mutant == "next_push_no_wait":
        original = imagery.ImageryState.landed
        imagery.ImageryState.landed = lambda self, world, session, *, wait=False: original(
            self, world, session, wait=False)
    else:
        raise ValueError(mutant)
