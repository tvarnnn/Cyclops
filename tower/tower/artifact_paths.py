"""Where Tower is allowed to put the artifacts it writes.

This exists because of a specific, repeated, measured failure -- not as a
general-purpose path utility.

Two independent mechanisms have put Glasses data outside every Glasses
directory on this host:

1.  **A relative root resolved from the wrong place.** Every session CLI
    below takes ``--root`` and defaults it to a *relative* path
    (``data/world_builder``). Relative to what is decided by whatever
    directory the caller happened to be standing in. An agent working
    from ``C:\\`` with ``--root wbr3`` gets ``C:\\wbr3``, silently, with no
    error -- and that is exactly the shape of the run roots that were
    found at the drive root (``C:\\wbr3``, ``C:\\wbr4``, ``C:\\wbrev`` ...).

2.  **MAX_PATH pressure.** Windows' 260-character path limit makes deep
    project paths genuinely fail: a clone has died with ``Filename too
    long`` and been worked around by cloning to ``C:\\m3s``. The
    workaround is real. The drive root is not the right place to put it.

The fix is not to guess a directory for the caller. It is to refuse the two
placements that are never correct and say why, at the CLI boundary, where
the mistake is actually made and can still be corrected by hand.

Refused:

* a filesystem root itself (``C:\\``, ``/``)
* a direct child of a filesystem root (``C:\\wbmut``, ``C:\\wb-stage0``)
* the user's home directory itself
* a direct child of the user's home directory (``~/wbscratch``)

Everything else is allowed. In particular a temp directory
(``%LOCALAPPDATA%\\Temp\\<something>\\...``) is deep inside the home
directory, not a direct child of it, so ordinary ``tempfile`` use is
untouched -- tests and smoke runs keep working.

See ``CLAUDE.md`` ("Glasses filesystem / worktree policy") for where
artifacts are supposed to go instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = [
    "ArtifactRootError",
    "resolve_artifact_root",
    "artifact_root_arg",
]

_SCRATCH_HINT = (
    "Disposable Glasses artifacts belong under "
    "C:\\Users\\<you>\\Projects\\Glasses-scratch\\ ; durable ones belong "
    "inside the repository or a worktree under "
    "C:\\Users\\<you>\\Projects\\."
)


class ArtifactRootError(ValueError):
    """A chosen artifact root is one Tower must never write to."""


def _is_filesystem_root(path: Path) -> bool:
    return path.parent == path


def resolve_artifact_root(root: Path | str) -> Path:
    """Return *root* as an absolute path, refusing unwritable placements.

    The returned path is resolved, so a relative ``--root`` is pinned to
    the caller's current directory *here*, once, instead of being
    re-interpreted by every later ``open()``.

    Raises:
        ArtifactRootError: if *root* resolves to a filesystem root, a
            direct child of one, the user's home directory, or a direct
            child of the home directory.
    """
    resolved = Path(root).expanduser().resolve()

    if _is_filesystem_root(resolved):
        raise ArtifactRootError(
            f"refusing to use the filesystem root {resolved} as an artifact "
            f"root. {_SCRATCH_HINT}"
        )

    if _is_filesystem_root(resolved.parent):
        raise ArtifactRootError(
            f"refusing to write artifacts to {resolved}: it sits directly at "
            f"the drive root {resolved.parent}. This is usually a relative "
            f"--root resolved from the wrong working directory. "
            f"{_SCRATCH_HINT}"
        )

    home = Path.home().resolve()
    if resolved == home:
        raise ArtifactRootError(
            f"refusing to use the home directory {resolved} as an artifact "
            f"root. {_SCRATCH_HINT}"
        )
    if resolved.parent == home:
        raise ArtifactRootError(
            f"refusing to write artifacts to {resolved}: it sits directly in "
            f"the home directory {home}. {_SCRATCH_HINT}"
        )

    return resolved


def artifact_root_arg(value: str) -> Path:
    """``argparse`` ``type=`` converter wrapping :func:`resolve_artifact_root`.

    Converting here means argparse reports the refusal as a normal usage
    error against the offending flag, before the script has created
    anything.
    """
    try:
        return resolve_artifact_root(value)
    except ArtifactRootError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
