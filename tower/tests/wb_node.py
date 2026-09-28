"""The node binary for the World Builder tests that run a page's own JavaScript.

Those tests skip on a host without node, which on a full-suite run turns a page's
only executed-copy check into a silent skip (C13 review LOW-2; T-UX0b LOW-1).
`WB_NODE_REQUIRED=1` turns that skip into a failure, in the style of
`WB_FROZEN_FIXTURES_REQUIRED`: opt-in, for the lead's integration run on a host
that has node. Unset, a node-less host still skips.
"""

from __future__ import annotations

import os
import shutil

import pytest

NODE_REQUIRED_ENV = "WB_NODE_REQUIRED"


def node_or_skip(purpose: str = "run the page's script") -> str:
    """The path of `node`; a skip without it, or a failure under WB_NODE_REQUIRED=1."""
    node = shutil.which("node")
    if node is None:
        reason = f"no node on this host to {purpose}"
        if os.environ.get(NODE_REQUIRED_ENV) == "1":
            pytest.fail(f"{NODE_REQUIRED_ENV}=1 but {reason}")
        pytest.skip(reason)
    return node
