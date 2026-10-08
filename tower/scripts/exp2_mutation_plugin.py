"""Pytest-only in-memory Exp2 mutants. Set EXP2_MUTATION to one name below."""

import os
from pathlib import Path


MUTANTS = {
    "cadence_11": ("CADENCE, K, MIN_DISTANCE = 10, 50, 0",
                   "CADENCE, K, MIN_DISTANCE = 11, 50, 0"),
    "reverse_tie": ("candidates.sort(key=lambda i: (-float(scores[q, i]), i))",
                    "candidates.sort(key=lambda i: (-float(scores[q, i]), -i))"),
    "image_hash_ignored": ('"images": [{"name": p.name, "sha256": sha256(p)} for p in images],\n                "height": HEIGHT',
                           '"images": [{"name": p.name, "sha256": "0" * 64} for p in images],\n                "height": HEIGHT'),
    "copy_digest_ignored": ("if not file.is_file() or sha256(file) != expected:",
                            "if not file.is_file():"),
    "gem_p_2": ("HEIGHT, WIDTH, GEM_P = 224, 392, 3",
                "HEIGHT, WIDTH, GEM_P = 224, 392, 2"),
}


def pytest_configure(config):
    name = os.environ.get("EXP2_MUTATION")
    if not name:
        return
    if name not in MUTANTS:
        raise ValueError(name)
    from tower.world_builder import exp2_retrieval as module
    source = Path(module.__file__).read_text(encoding="utf-8")
    old, new = MUTANTS[name]
    if source.count(old) != 1:
        raise AssertionError(f"mutation anchor {name} appears {source.count(old)} times")
    exec(compile(source.replace(old, new), module.__file__, "exec"), module.__dict__)
