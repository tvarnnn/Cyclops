#!/usr/bin/env python3
"""Compare the contract identifiers this iOS build implements against the
identifiers a live Tower actually serves.

Why this exists
---------------
Every contract fixture in `GlassesTests` was captured by hand, off a Tower
that was running at the time. Nothing re-checks them. If a Windows lane
publishes `world_builder.status/2026-09-01`, this build keeps passing 388
green tests against the old bytes and only finds out on a device, in a room,
with the glasses on.

This script is the check that was missing. It reads the identifiers out of
the Swift source (so it cannot drift from the source the way a second list
would) and asks a live Tower what it serves.

Exit codes: 0 agreement, 1 drift, 2 Tower unreachable.

    ./scripts/contract-drift-check.py [--tower http://host:port]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

SWIFT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "Glasses"
IDENTIFIER = re.compile(r'"([a-z_]+\.[a-z_]+/20\d{2}-\d{2}-\d{2})"')
ENVELOPE = re.compile(r'"(cartridge_results\.envelope/20\d{2}-\d{2}-\d{2})"')
# Tower cartridge names as this build spells them. Deliberately a sweep for the
# literal rather than a parse of `towerCartridgeNames`: that map's values are
# constant references (`WorldBuilderResultContract.towerCartridge`), not string
# literals, so parsing the map alone finds nothing and reports every cartridge
# as unmapped. A name the Swift never spells cannot be recognised by any of it.
TOWER_CARTRIDGE_LITERAL = re.compile(r'towerCartridge\s*=\s*"([a-z][a-z_]*)"')


def implemented() -> dict[str, str]:
    """Identifier -> the Swift file:line that declares it."""
    found: dict[str, str] = {}
    for path in sorted(SWIFT_ROOT.rglob("*.swift")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in list(IDENTIFIER.finditer(line)) + list(ENVELOPE.finditer(line)):
                found.setdefault(
                    match.group(1), f"{path.relative_to(SWIFT_ROOT.parent)}:{number}"
                )
    return found


def mapped_tower_names() -> set[str]:
    """The Tower cartridge names this build can recognise in a declaration.

    `TowerCapabilities.towerCartridgeNames` has exactly one row. If the Tower
    promotes a cartridge out of `not_offered` and starts declaring it, iOS
    resolves `.noContract` and says nothing — no error, no log, and no test
    fails. That is the silence this check exists to break.
    """
    names: set[str] = set()
    for path in SWIFT_ROOT.rglob("*.swift"):
        names.update(TOWER_CARTRIDGE_LITERAL.findall(path.read_text(encoding="utf-8")))
    return names


def fetch(tower: str, path: str) -> tuple[int, object]:
    request = urllib.request.Request(tower.rstrip("/") + path)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read())
        except Exception:
            return error.code, None


def served(tower: str) -> tuple[dict[str, str], list[dict], set[str]]:
    """Identifier -> where the Tower stated it, plus the not_offered list."""
    status, body = fetch(tower, "/cartridges")
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"/cartridges answered {status}")

    live: dict[str, str] = {}
    offered_names: set[str] = set()
    if envelope := body.get("envelope_contract"):
        live[envelope] = "/cartridges envelope_contract"
    for offer in body.get("cartridges", []):
        if name := offer.get("cartridge"):
            offered_names.add(name)
        if contract := offer.get("contract"):
            live[contract] = f"/cartridges {offer.get('cartridge')}/{offer.get('result_type')}"

    # `http_contracts` -- capabilities the Tower serves over HTTP rather than by
    # subscription. Added 2026-08-27 with the Tower unification, and this check
    # was blind to the whole block until then: it read `cartridges` only, so a
    # Tower could move `document_memory.library` to a new identifier and this
    # gate would still print AGREEMENT.
    #
    # That is exactly the failure this script exists to catch, so the blind spot
    # mattered more than the missing row: a drift check that cannot see a
    # surface is worse than no check for that surface, because it reports
    # confidence about it.
    #
    # These are counted as SERVED, not as a separate class. From this script's
    # point of view an identifier the Tower states is an identifier the build
    # must implement, and how it travels is not this comparison's business.
    for offer in body.get("http_contracts", []):
        if name := offer.get("cartridge"):
            offered_names.add(name)
        if contract := offer.get("contract"):
            route = offer.get("entry_route", "?")
            live[contract] = f"/cartridges http_contracts {offer.get('cartridge')} -> {route}"

    # Object Memory is never declared over the socket: its contract travels in
    # the body of an answer, so the only way to learn it is to ask.
    not_offered = body.get("not_offered", []) if isinstance(body, dict) else []

    status, body = fetch(tower, "/object-memory/observations")
    if status == 200 and isinstance(body, dict) and (contract := body.get("contract")):
        live[contract] = "/object-memory/observations"
        # The imagery contract travels NESTED inside this same body, under
        # `imagery.contract`. This function already held those bytes and threw
        # them away -- the same class of blind spot as the `http_contracts` one
        # fixed above, and worse in consequence: iOS compares
        # `object_memory.imagery/...` for equality and REFUSES the payload on a
        # mismatch, so drift there means every frame and crop hard-refuses and
        # the picture view goes permanently blank, silently, with this gate
        # still printing AGREEMENT.
        imagery = body.get("imagery")
        if isinstance(imagery, dict) and (imagery_contract := imagery.get("contract")):
            live[imagery_contract] = "/object-memory/observations imagery.contract"
    elif status == 404:
        live["<object-memory: no root configured>"] = f"/object-memory/observations 404"

    return live, not_offered, offered_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", default="http://100.110.156.55:8000")
    args = parser.parse_args()

    build = implemented()
    print(f"Tower: {args.tower}\n")

    print("Implemented by this build:")
    for identifier, where in sorted(build.items()):
        print(f"  {identifier:<45} {where}")

    try:
        live, not_offered, offered_names = served(args.tower)
    except Exception as error:
        print(f"\nTOWER UNREACHABLE: {error}", file=sys.stderr)
        return 2

    print("\nServed by the live Tower:")
    for identifier, where in sorted(live.items()):
        print(f"  {identifier:<45} {where}")

    # Only identifiers the Tower actually stated can drift. A contract this
    # build implements but the Tower does not currently offer is not drift:
    # geometry is fetched per-world over HTTP and is absent until a world
    # exists, and Object Memory's is absent until a root is configured.
    # The envelope contract is deliberately NOT hardcoded on the iOS side, and
    # reporting its absence as drift would make this tool cry wolf on every run.
    # Gating on it was proposed, reviewed and deferred: it has no defined user
    # state, no precedent at this layer, the largest possible blast radius (one
    # string comparison failing every cartridge at once), and the field is
    # absent from three Tower->client messages today, so an "absent means
    # mismatch" rule would fire against a conforming Tower. It is surfaced below as
    # information.
    envelope = {i: w for i, w in live.items() if i.startswith("cartridge_results.envelope/")}
    unreadable = {
        i: w for i, w in live.items()
        if not i.startswith("<") and i not in build and i not in envelope
    }

    print()
    for identifier, where in sorted(envelope.items()):
        note = "matches this build's fixtures" if identifier in build else "not pinned in Swift, by decision"
        print(f"Envelope: {identifier}  ({note})")
    print()

    if not_offered:
        print("Tower is deliberately not offering:")
        for entry in not_offered:
            print(f"  {entry.get('cartridge')}: {entry.get('reason', '')[:100]}")
        print()

    unmapped = sorted(offered_names - mapped_tower_names())

    if unmapped:
        print("DRIFT — the Tower declares cartridges this build cannot even see:")
        for name in unmapped:
            print(f"  {name:<45} no row in TowerCapabilities.towerCartridgeNames")
        print(
            "  (iOS resolves these to .noContract silently — no error, no log,\n"
            "   and no test fails. Add the mapping and a client that reads it.)"
        )

    if unreadable:
        print("DRIFT — the Tower serves contracts this build cannot read:")
        for identifier, where in sorted(unreadable.items()):
            print(f"  {identifier:<45} {where}")

    if unreadable or unmapped:
        return 1

    print("AGREEMENT — every contract the Tower stated is implemented by this build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
