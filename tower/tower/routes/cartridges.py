"""`GET /cartridges` -- the capability declaration, over HTTP.

HTTP rather than only a WebSocket message, because of the third state in
`ios/docs/agent-handoffs/IOS-TO-TOWER.md` §0.1: "the Tower offers a contract
this build implements, but is **unreachable** -> 'connect'". A client that can
only learn the contract set by opening the socket cannot distinguish that
state from "not built yet" -- it would have to be connected to find out
whether it should be connected.

The declaration is also the thing iOS CACHES ("iOS caches a declaration
rather than fetching a registry"), and a cacheable thing wants a plain
GET.

The same declaration is available on the socket as `{"type":
"cartridges"}`. Both call `registry.declare()`, and a test asserts the
two are byte-identical -- two surfaces onto one function, never two
functions that agree today.
"""

from fastapi import APIRouter, Request

from tower.results import registry

router = APIRouter()


@router.get("/cartridges")
def cartridges(request: Request) -> dict:
    return registry.declare(**registry.declaration_inputs(request.app.state))
