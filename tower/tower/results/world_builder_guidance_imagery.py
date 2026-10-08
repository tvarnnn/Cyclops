"""Optional FOW v1.1 B imagery transport."""

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from tower.results.world_builder_appearance import NO_STORE_HEADERS
from tower.world_builder.guidance_imagery import contained_session_id, state_for
from tower.world_builder.appearance import label_is_trusted
from tower.results.world_builder_geometry import contained_world_id, store_from_root
from tower.world_builder.store import WorldStoreError

router = APIRouter()


def _configuration(request):
    root = getattr(request.app.state, "world_root", None)
    live = os.environ.get("TOWER_WORLD_GUIDANCE_IMAGERY_LIVE") == "on"
    landed = (os.environ.get("TOWER_WORLD_GUIDANCE_IMAGERY_LANDED") == "on" and
              os.environ.get("TOWER_WORLD_GUIDANCE_COVERAGE") == "on")
    if root is None or not (live or landed):
        raise HTTPException(404, "guidance imagery is off or unavailable",
                            headers=NO_STORE_HEADERS)
    return root, live, landed


@router.get("/worlds/{world_id}/guidance/{session_id}/imagery/manifest")
def manifest(world_id: str, session_id: str, request: Request):
    root, live, landed = _configuration(request)
    store = store_from_root(root)
    try:
        if (contained_world_id(store, world_id) != world_id or
                contained_session_id(store, world_id, session_id) != session_id):
            raise ValueError("world mismatch")
        if store.read_session(world_id, session_id).world_id != world_id:
            raise ValueError("session mismatch")
        if not label_is_trusted(store.keyframe_image_set(world_id, session_id).redaction):
            raise ValueError("redaction is not trusted")
    except (OSError, ValueError, KeyError, TypeError, WorldStoreError):
        raise HTTPException(404, "imagery is absent or untrusted",
                            headers=NO_STORE_HEADERS) from None
    state = state_for(root)
    payload = {"version": 1}
    if live:
        block = state.live(root, world_id, session_id)
        if block is None:
            raise HTTPException(404, "no imagery for this world or session",
                                headers=NO_STORE_HEADERS)
        payload["live"] = block
    if landed:
        payload["landed"] = state.landed(world_id, session_id, wait=True)
    return JSONResponse(payload, headers=NO_STORE_HEADERS)


@router.get("/worlds/{world_id}/guidance/{session_id}/imagery/tile/{digest}")
def tile(world_id: str, session_id: str, digest: str, request: Request):
    root, live, landed = _configuration(request)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise HTTPException(404, "unknown tile", headers=NO_STORE_HEADERS)
    state = state_for(root)
    store = store_from_root(root)
    try:
        if (contained_world_id(store, world_id) != world_id or
                contained_session_id(store, world_id, session_id) != session_id):
            raise ValueError("world mismatch")
        if store.read_session(world_id, session_id).world_id != world_id:
            raise ValueError("session mismatch")
        if not label_is_trusted(store.keyframe_image_set(world_id, session_id).redaction):
            raise ValueError("redaction is not trusted")
        if live:
            state.live(root, world_id, session_id)  # re-check current segment
    except (OSError, ValueError, KeyError, TypeError, WorldStoreError):
        raise HTTPException(404, "imagery is absent or untrusted",
                            headers=NO_STORE_HEADERS) from None
    blob = state.tile(world_id, session_id, digest, live, landed)
    if blob is None:
        raise HTTPException(404, "unknown or superseded tile", headers=NO_STORE_HEADERS)
    return Response(blob, media_type="image/jpeg", headers=NO_STORE_HEADERS)
