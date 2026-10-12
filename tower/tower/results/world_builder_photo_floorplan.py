"""FOW v1.2 photo floor plan transport (wire: mail 303, tower/docs/contracts/
WORLD-BUILDER-GUIDANCE-IMAGERY.md "Photo floor plan v1.2").

Registered by `main.py` only when TOWER_WORLD_PHOTO_FLOORPLAN is exactly `on` (with
TOWER_WORLD_GUIDANCE_COVERAGE=on); every handler re-checks the switch and answers 404
when it is off. Both URL identifiers are untrusted and must name one contained world
and one of its sessions. The manifest is <= 8 KiB; a tile is served only while its
digest belongs to the CURRENT landing (a superseded or unknown digest is 404).
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from tower.config import world_photo_floorplan_setting
from tower.results.world_builder_appearance import NO_STORE_HEADERS
from tower.results.world_builder_geometry import contained_world_id, store_from_root
from tower.world_builder.guidance_imagery import contained_session_id
from tower.world_builder.photo_floorplan_worker import (
    MANIFEST_MAX_BYTES,
    manifest_bytes,
    state_for,
)
from tower.world_builder.photo_floorplan import TILE_CAP_BYTES
from tower.world_builder.store import WorldStoreError

router = APIRouter()
_HEX = frozenset("0123456789abcdef")
# A manifest request (never a status poll) arriving while its landing's job is in
# flight waits for it at most 1.6 s, its own constant: decoupled from the job's wall
# cap (manager 323), so the shared threadpool is never held longer. A landing that
# publishes later is picked up by the phone's bounded re-fetch.
_MANIFEST_WAIT_S = 1.6


def _root(request):
    root = getattr(request.app.state, "world_root", None)
    if root is None or not world_photo_floorplan_setting():
        raise HTTPException(404, "photo floor plan is off or unavailable",
                            headers=NO_STORE_HEADERS)
    return root


def _contained(root, world_id, session_id):
    store = store_from_root(root)
    try:
        if (contained_world_id(store, world_id) != world_id or
                contained_session_id(store, world_id, session_id) != session_id):
            raise ValueError("world mismatch")
        if store.read_session(world_id, session_id).world_id != world_id:
            raise ValueError("session mismatch")
    except (OSError, ValueError, KeyError, TypeError, WorldStoreError):
        raise HTTPException(404, "no such world or session",
                            headers=NO_STORE_HEADERS) from None


@router.get("/worlds/{world_id}/guidance/{session_id}/photo-floorplan/manifest")
def manifest(world_id: str, session_id: str, request: Request):
    root = _root(request)
    _contained(root, world_id, session_id)
    landed = state_for(root).landed(world_id, session_id, wait_s=_MANIFEST_WAIT_S)
    body = manifest_bytes(landed)
    if len(body) > MANIFEST_MAX_BYTES:  # the worker never publishes one; belt and braces
        body = manifest_bytes(None)
    return Response(body, media_type="application/json", headers=NO_STORE_HEADERS)


@router.get("/worlds/{world_id}/guidance/{session_id}/photo-floorplan/tile/{digest}")
def tile(world_id: str, session_id: str, digest: str, request: Request):
    root = _root(request)
    if len(digest) != 64 or not set(digest) <= _HEX:
        raise HTTPException(404, "unknown tile", headers=NO_STORE_HEADERS)
    _contained(root, world_id, session_id)
    hit = state_for(root).tile(world_id, session_id, digest)
    if hit is None or len(hit[0]) > TILE_CAP_BYTES:
        raise HTTPException(404, "unknown or superseded tile", headers=NO_STORE_HEADERS)
    return Response(hit[0], media_type=hit[1], headers=NO_STORE_HEADERS)


__all__ = ["router"]
