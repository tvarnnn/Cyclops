"""Negative-control mutants for FOW fix round 2 (items 1 and 6).

Run one at a time, e.g.:
    FOW_MUTANT=read_ignores_budget python -m pytest -p fow_fix2_mutator \
        tests/test_world_builder_guidance_coverage.py -k slow_os_read
Each mutant must make its targeted test fail.
"""

import os


def pytest_configure(config):
    from tower.results import world_builder as status
    from tower.results import publisher
    import tower.results as results
    from tower.world_builder import guidance_coverage as coverage

    mutant = os.environ["FOW_MUTANT"]
    if mutant == "read_ignores_budget":
        # Bounded by size, but the budget is checked only before the read.
        def broken(path, budget):
            limit = coverage._FILE_LIMITS[path.name]
            budget.check()
            if path.stat().st_size > limit:
                raise ValueError(f"guidance file exceeds size limit: {path.name}")
            chunks = []
            with path.open("rb") as stream:
                while True:
                    chunk = stream.read(coverage._READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return b"".join(chunks).decode("utf-8")

        coverage._read_bounded = broken
    elif mutant == "no_size_cap":
        for name in list(coverage._FILE_LIMITS):
            coverage._FILE_LIMITS[name] = 1 << 40
    elif mutant == "caps_16mib":
        coverage._FILE_LIMITS["solution.json"] = 16 * 1024 * 1024
        coverage._FILE_LIMITS["poses.json"] = 16 * 1024 * 1024
    elif mutant == "shutdown_noop":
        coverage.CoverageWorker.shutdown = lambda self: None
    elif mutant == "hub_unwired":
        original = results.make_snapshot_for

        def broken(*args, **kwargs):
            snapshot_for = original(*args, **kwargs)
            del snapshot_for.shutdown_guidance
            return snapshot_for

        results.make_snapshot_for = broken
    elif mutant == "hub_shutdown_propagates":
        original = publisher.ResultHub.shutdown

        async def broken(self):
            hook = getattr(self._snapshot_for, "shutdown_guidance", None)
            if hook is not None:
                hook()
            return await original(self)

        publisher.ResultHub.shutdown = broken
    elif mutant == "budget_noop":
        coverage._Budget.check = lambda self: None
    elif mutant == "allowance_9k":
        status._MAX_STATUS_WITH_COVERAGE_BYTES = 9 * 1024
    else:
        raise ValueError(mutant)
