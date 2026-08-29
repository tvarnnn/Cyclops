"""`GET /cv-lab/preview` -- the newest derived picture, and nothing else.

Why HTTP, when every CV Lab command is a socket message
-------------------------------------------------------
`ws.py` gives the frame path and the result sender ONE shared
`asyncio.Lock`. Every bulky or slow thing in this Tower is deliberately
an HTTP route for that reason -- `geometry.py` says so about a megabyte
of points, `observations.py` about a JPEG, `documents.py` about a JSONL
read -- because anything large on that socket queues behind, or in front
of, a `frame_result`. A preview is 5-40 KB arriving several times a
second. It belongs here, beside `GET /object-memory/.../frame`, which is
the same shape of thing served the same way.

It is also the shape that cannot build a backlog. An HTTP GET is a pull:
a client that falls behind asks again and gets the NEWEST picture, and
the ones it missed were dropped at the moment they were replaced rather
than queued against its return. There is no server-side per-client state
at all.

Why it is `def` and not `async def`
------------------------------------
Starlette runs a synchronous route in its thread pool. That is what keeps
the resize, the colourising and the PNG encoder off the event loop the CV
Lab's `process()` runs on -- so the answer to "can the viewer slow the
pipeline down" is structural rather than a promise. Making this
`async def` would put the encoder on the loop and silently undo the
entire design.

Two routes, and the second one is the interesting one
------------------------------------------------------
`GET /cv-lab/preview` is the bytes. `GET /cv-lab/preview/status` is the
descriptor without them -- the same "answer before you download"
split `observations.py` makes between `/imagery` and `/frame`, and the
same one `geometry.py` makes between a manifest and a segment. A client
deciding whether to draw a viewer at all should not have to fetch a
picture to find out there isn't one.

The descriptor also rides the status document (`run.annotation.artifact`)
on the result channel, so a phone already subscribed learns about the
preview without asking. This route is for `curl`, and for a client that
wants to check without opening a socket.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from tower.cv_lab.contracts import (
    ERR_LAB_UNAVAILABLE,
    PREVIEW_CONTRACT,
    PREVIEW_DISABLED,
    PREVIEW_NONE_YET,
    PREVIEW_NOT_VISUAL,
    PREVIEW_RENDER_FAILED,
    PREVIEW_RUN_CHANGED,
    PREVIEW_STALE,
)
from tower.cv_lab.preview import PreviewNotModified, PreviewRefusal

router = APIRouter()


# Which refusal is which kind of "no".
#
# 404 -- there is no such picture and asking again will not help. The
#        experiment has nothing to draw.
# 409 -- you asked about a run that is not the current one. The
#        conflict IS the answer: the reply names the run that is.
# 503 -- this Tower is willing and has nothing right now. Previews are
#        off, or no frame has arrived yet, or the newest one aged out, or
#        the encoder failed. All four are "try again", and all four say
#        which one they are on the body.
#
# The reason VALUE is on the body in every case and a client should switch
# on that rather than on the code -- the rule `observations.py` states for
# its own imagery routes, for the same reason: a status code is a class of
# answer and these are five different answers.
_PREVIEW_STATUS = {
    PREVIEW_NOT_VISUAL: 404,
    PREVIEW_RUN_CHANGED: 409,
    PREVIEW_DISABLED: 503,
    PREVIEW_NONE_YET: 503,
    PREVIEW_STALE: 503,
    PREVIEW_RENDER_FAILED: 503,
}

# Never cached, and this one is not a formality. The treatment on every
# preview is `raw_ephemeral`, whose whole definition is "live view only,
# never for anything persisted". A proxy or a URL cache holding a copy
# would be the persistence that treatment says does not happen, created
# by a party that never agreed to it.
_NO_STORE = {"Cache-Control": "no-store"}


def _lab(request: Request):
    return getattr(request.app.state, "cv_lab", None)


def _no_lab():
    # 503, not 404, and the same wording `GET /cv-lab` uses: the route
    # exists on every build of this Tower, and a 404 would say "this Tower
    # has never heard of a CV Lab", which is a different claim.
    return JSONResponse(
        status_code=503,
        content={
            "reason": ERR_LAB_UNAVAILABLE,
            "message": "this Tower runs no CV Lab",
            "contract": PREVIEW_CONTRACT,
        },
    )


def _refusal_response(refusal: PreviewRefusal):
    return JSONResponse(
        status_code=_PREVIEW_STATUS.get(refusal.reason, 503),
        content={
            "reason": refusal.reason,
            "message": refusal.message,
            "contract": PREVIEW_CONTRACT,
            # So a client that asked about a stale run can correct itself
            # in one round trip instead of re-reading the status document.
            "current_run_id": refusal.current_run_id,
        },
        headers=_NO_STORE,
    )


@router.get("/cv-lab/preview/status")
def preview_status(request: Request):
    """Whether there is a picture, and what may be said about it.

    Answerable without downloading anything, which is the point.
    """
    lab = _lab(request)
    if lab is None:
        return _no_lab()
    descriptor = lab.preview_descriptor()
    return JSONResponse(
        content={
            "contract": PREVIEW_CONTRACT,
            "artifact": descriptor,
            "artifact_unavailable_reason": (
                None if descriptor is not None else lab.preview_unavailable_reason()
            ),
        },
        headers=_NO_STORE,
    )


@router.get("/cv-lab/preview")
def preview(request: Request, run_id: str | None = None):
    """The newest derived picture for the running experiment.

    `run_id` is optional and should always be sent. It is the staleness
    guard, enforced at the Tower rather than trusted to the phone: stop
    Edge Detection, start Depth, and a request still naming Edge's run is
    refused with 409 and the current run's id, instead of being answered
    with a picture that would be drawn under Depth's name. The identity
    also travels on the response headers, so a client that forgot to ask
    can still check what it got.

    `If-None-Match` is honoured and worth sending. A phone polling faster
    than the Tower is producing gets a 304 with no body and no encode,
    which is what makes over-polling cost a round trip rather than CPU.
    """
    lab = _lab(request)
    if lab is None:
        return _no_lab()

    rendered = lab.render_preview(
        run_id=run_id, if_none_match=request.headers.get("if-none-match")
    )
    if isinstance(rendered, PreviewRefusal):
        return _refusal_response(rendered)
    if isinstance(rendered, PreviewNotModified):
        return Response(
            status_code=304,
            headers={
                **_NO_STORE,
                "ETag": rendered.etag,
                "X-CV-Preview-Run": rendered.run_id or "",
                "X-CV-Preview-Seq": str(rendered.result_seq),
            },
        )

    return Response(
        content=rendered.image_bytes,
        media_type=rendered.media_type,
        headers={
            **_NO_STORE,
            "ETag": rendered.etag,
            # Identity, on the bytes themselves. The status document
            # carries the DESCRIPTOR -- what a preview is and how to fetch
            # one -- and deliberately not this, because `result_seq`
            # changes every frame and would make every result-channel poll
            # report news about a Lab that had merely gone on running.
            "X-CV-Preview-Run": rendered.run_id or "",
            "X-CV-Preview-Seq": str(rendered.result_seq),
            "X-CV-Preview-Kind": rendered.kind,
            # Seconds since the Tower produced this, so a client can show
            # a picture and say how old it is instead of implying it is
            # now.
            "X-CV-Preview-Age": f"{rendered.age_s:.3f}",
            # The privacy treatment, on the bytes. A client that fetched
            # the image without ever reading the status document still
            # learns that this is untreated, live-view-only imagery it
            # must not persist.
            "X-CV-Preview-Treatment": rendered.treatment,
            "X-CV-Preview-Contract": PREVIEW_CONTRACT,
        },
    )
