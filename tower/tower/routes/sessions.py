"""Start, Pause and Stop for a cartridge, over HTTP.

**The control surface for a cartridge session: Start, Pause and Stop.**
Without it a producer is started by a human typing a command against a
capture id copied out of a directory listing, which is not a product: a
physical run that remembered 64 observations needed a second terminal for
every one of them.

**Cartridge-blind.** This module imports no cartridge and names none. It
looks a session up by the cartridge id in the path and asks it to apply
an action; `main.py` decides which cartridges have sessions at all. A
Tower that runs none answers 404 here for every id, which is a claim
about configuration and not about the request.

**Why `POST` and not a socket message.** The socket carries frames, and
its send lock is shared with the frame path -- the same reason
`tower/routes/geometry.py` exists as HTTP. A Pause that had to queue
behind a frame send would be a Pause that arrives late, and the whole
point of the control is that it is obeyed when pressed.

**Why sync `def`.** `detach` waits up to `DETACH_GRACE_SECONDS` for a
producer to finish its current record before terminating it. On the
event loop that would stall every frame on the Tower for five seconds.
FastAPI runs a sync endpoint in its threadpool, which is where a call
that can block belongs, and
`test_the_session_handlers_are_sync_so_a_pause_cannot_stall_the_frame_path`
pins it because it is the whole mechanism.

**Refusals are 409, not 200.** `GET /object-memory/last-seen/teapot`
answers 200 with `observed: false` because a 404 there would be a claim
about the world. This is the opposite situation: an action that could not
be honoured is not a fact about anything, and a client that ignored the
body would read 200 as "paused". A status code a client cannot ignore is
the right shape for a control surface, and the body still carries the
reason and the state that was actually reached.
"""

from fastapi import APIRouter, HTTPException, Request

from tower.cartridge_session import ACTIONS, STATES, SessionRefused

router = APIRouter()

# Opaque and dated, in the style of every other contract identifier here.
# Compared for equality only: never parsed, never ordered, never used to
# infer that one contract is newer than another.
SESSION_CONTRACT = "cartridge_session.control/2026-08-27"

# What this surface does NOT claim, carried in the payload rather than
# only in a document -- the same reason the observation envelope carries
# `claim` and `absence_means` as values a decoder can switch on.
#
# A session is an INTENT. `state: "active"` means a person asked this
# cartridge to run; it does not mean a producer is alive, and the two
# come apart exactly when it matters most (a worker that died in the
# first ten seconds of a walk). `following` is the observed fact, and a
# client that draws "remembering" from `state` alone will draw it for the
# rest of a walk that remembered nothing.
STATE_MEANS = "intent-not-liveness"


def _sessions(request: Request) -> dict:
    return getattr(request.app.state, "cartridge_sessions", None) or {}


def _session(request: Request, cartridge: str):
    session = _sessions(request).get(cartridge)
    if session is None:
        # 404 on the RESOURCE, not on the action: this Tower has no
        # session for that cartridge, which is a configuration answer.
        # It is never the answer to "may I start", which is 409.
        raise HTTPException(
            status_code=404,
            detail=(
                f"this Tower has no controllable session for {cartridge!r}; "
                f"it has {sorted(_sessions(request)) or 'none'}"
            ),
        )
    return session


def _envelope(payload: dict) -> dict:
    payload = dict(payload)
    payload["contract"] = SESSION_CONTRACT
    payload["state_means"] = STATE_MEANS
    payload["states"] = list(STATES)
    payload["actions"] = list(ACTIONS)
    return payload


@router.get("/cartridges/{cartridge}/session")
def read_session(cartridge: str, request: Request) -> dict:
    return _envelope(_session(request, cartridge).snapshot())


@router.post("/cartridges/{cartridge}/session/{action}")
def apply_session_action(cartridge: str, action: str, request: Request) -> dict:
    session = _session(request, cartridge)
    try:
        result = session.apply(action)
    except SessionRefused as refusal:
        raise HTTPException(
            status_code=409,
            detail=_envelope(
                {
                    "accepted": False,
                    "reason": refusal.reason,
                    "message": refusal.message,
                    # The state it is ACTUALLY in, not the one that was
                    # asked for. A client that refreshes after a refusal
                    # should not need a second request to find out.
                    **session.snapshot(),
                }
            ),
        ) from refusal
    result["accepted"] = True
    return _envelope(result)
