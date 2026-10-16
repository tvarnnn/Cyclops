"""The Experimental CV Lab as a product surface rather than a variable.

The experiment is chosen at runtime by a client command, and every
result names the run that produced it; nothing about the choice lives in
the process environment.

Six modules, in dependency order:

    contracts.py   the strings a phone switches on -- identifiers,
                   lifecycle states, refusal reasons, bounds
    catalog.py     the registry, rendered as the list iOS displays
    run.py         one run's identity and its constant-memory measurements
    loader.py      one reusable daemon thread for blocking experiment loads
    preview.py     the throttled, in-memory live preview and its renderers
    lab.py         the slot: lifecycle, selection, the frame path, and the
                   one status document every surface serves

Nothing here persists anything, writes a file, or retains raw imagery. The
module descriptor this Lab lives behind declares `persists_data=False`
and `retains_raw_imagery=False`, and that declaration is what the privacy
policy is enforced against -- so it has to stay true of this package too.

Nothing here imports a cartridge, and a test says so.
"""

from tower.cv_lab.lab import CommandOutcome, CVLab

__all__ = ["CVLab", "CommandOutcome"]
