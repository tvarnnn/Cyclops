"""The live scene, and the switch that turns it on. Over HTTP.

Named after the QUESTION rather than the cartridge, exactly as
`geometry.py` and `observations.py` are: a route file named after the
cartridge would put its name into `main.py`'s import list, which is the
boundary two tests exist to hold.

WHY A CONTROL SURFACE EXISTS AT ALL, GIVEN THE PHONE SENDS NOTHING

`ios/docs/agent-handoffs/IOS-TO-TOWER.md` §6.2 is explicit that opening a
cartridge on the phone sends **nothing** -- "a test asserts the wire stays
silent" -- so the phone cannot be the thing that starts a session. Two other
consumers can, and both are real:

  * a Mac operator, or curl, driving a physical test. Somebody has to be
    able to start a session, walk into a room and read the counts back,
    and that has to be possible before any Swift exists.
  * this Tower's own result channel, on a subscription. A client that
    subscribes wants a scene; starting on demand is the same rule the
    result hub already applies to itself -- "a Tower nobody is watching
    does no polling and no disk IO on its behalf".

Both drive the same session object through the same verbs, so there is
one state machine and not two.

WHY GET /scene DUPLICATES THE SOCKET PAYLOAD

It is the identical function -- `live_payload` -- so the two cannot
disagree, which is the same arrangement `/cartridges` has with
`{"type": "cartridges"}`. Having it on HTTP is what makes the cartridge
physically testable by a person with a browser and no WebSocket client.

Every handler is `def` rather than `async def` on purpose. FastAPI runs a
sync endpoint in its threadpool, so `stop()`'s bounded join on a worker
thread happens there and not on the event loop -- the same reason
`geometry.py` and `observations.py` give for the same choice.

NOTHING HERE OBSERVES A FRAME.

The session is fed by `tower/routes/ws.py`, from the connection that
receives frames. This module can start it, pause it, stop it and read it.
It cannot hand it a frame, and there is no code path here that could.
"""

from fastapi import APIRouter, HTTPException, Request

from tower.results.contracts import SCENE_LIVE_CONTRACT
from tower.results.registry import SCENE_DISABLED_REASON
from tower.results.scene_understanding import live_payload

router = APIRouter()


def _session(request: Request):
    live = getattr(request.app.state, "live_cartridges", None)
    session = None if live is None else live.scene
    if session is None:
        # A claim about configuration, never about what is in the room.
        # The same distinction `observations.py` draws between a 404 and
        # `observed: false`: this Tower serves no live scene at all, which
        # is different from a scene that happens to be empty.
        #
        # The specific reason when there is one, so this route and
        # `/cartridges` cannot give an operator two different explanations
        # of one configuration -- which is the drift
        # `SCENE_DISABLED_REASON` was made a module constant to prevent.
        # Falls back to the configured-off wording, unchanged.
        # The SHARED constant for the fallback, not a second copy of the
        # sentence. This route used to carry its own shorter wording, so
        # the configured-off case already answered differently here than
        # at `/cartridges` -- the drift the constant exists to prevent,
        # sitting next to a comment claiming it could not happen.
        reason = getattr(request.app.state, "scene_unavailable_reason", None)
        raise HTTPException(
            status_code=404, detail=reason or SCENE_DISABLED_REASON
        )
    return session


def _view(session) -> dict:
    """Status and scene, through the one function the socket also uses."""
    state, _observed_at, _computed_at = session.latest()
    payload = live_payload(session.status(), state)
    payload["contract"] = SCENE_LIVE_CONTRACT
    return payload


@router.get("/scene")
def scene(request: Request) -> dict:
    """The live scene, or an explicit account of why there is not one."""
    return _view(_session(request))


@router.post("/scene/start")
def start(request: Request) -> dict:
    """Begin observing. Idempotent; resumes a paused session.

    Returns the full view rather than an acknowledgement, because the
    interesting answer is not "accepted" -- it is `state: "starting"` and
    a `loading_seconds` that a caller can poll. A bare 204 would make the
    caller guess how long a model load takes.
    """
    session = _session(request)
    session.start()
    return _view(session)


@router.post("/scene/pause")
def pause(request: Request) -> dict:
    """Stop observing; keep the last scene, and say it is not current.

    The models stay loaded. Pausing to release a detector would make
    Pause cost more than Stop, which is backwards.
    """
    session = _session(request)
    session.pause()
    return _view(session)


@router.post("/scene/resume")
def resume(request: Request) -> dict:
    session = _session(request)
    session.resume()
    return _view(session)


@router.post("/scene/stop")
def stop(request: Request) -> dict:
    """End the session, and forget the scene.

    The response will carry `scene_available: false`. That is the point
    of calling it: a scene retained past the end of a session is a claim
    about a room the wearer has left, and no staleness number is large
    enough to make that safe.
    """
    session = _session(request)
    session.stop()
    return _view(session)
