"""Contract identifiers, and why they are opaque strings.

`ios/docs/agent-handoffs/IOS-TO-TOWER.md` section 6 item 6 is explicit that iOS
compares contract identifiers **for equality only**:

    "Contract versioning semantics -- identifiers are compared for
     equality only, so ordering and compatibility remain the Tower's to
     define."

and section 0.1 that the identifier is "an **opaque identifier compared
for equality** (`CartridgeContract`)" which "deliberately does *not*
assume integer versions, ordering, or backward compatibility".

So these are strings, not integers, and nothing may parse them. That is a
deliberate departure from `tower/world_builder/schema.py: SCHEMA_VERSION`,
which is an int -- and the two are not in tension, because they answer
different questions. SCHEMA_VERSION governs whether THIS BUILD can read a
file THIS PROJECT wrote, where a monotonic int and a refuse-on-unknown
rule are right. A contract identifier governs whether ANOTHER MACHINE,
built and shipped separately, implements the agreement we are offering --
and there the only sound operation is equality, because a phone in the
App Store cannot be assumed to know anything about an identifier minted
after it shipped.

Dated rather than numbered, for the same reason: a date makes it obvious
that two identifiers differ without inviting anyone to compute which is
"greater".

Changing a value here is a WIRE BREAK. iOS will report "update the app"
and stop decoding, which is the intended behaviour -- see
`ios/docs/agent-handoffs/IOS-TO-TOWER.md` section 0.1. Change one only when the
payload's meaning changes in a way an existing decoder would get WRONG. Adding
an optional field that an older decoder ignores is not that; removing a field,
renaming one, or changing what a field means is.
"""

# The cartridge names iOS keys on. Stable identifiers, not display names.
CARTRIDGE_WORLD_BUILDER = "world_builder"
CARTRIDGE_EXPERIMENTAL_CV = "experimental_cv"
CARTRIDGE_DOCUMENT_MEMORY = "document_memory"
# Present here before it is present in `registry.declare()`, and that is
# not an oversight. The name is what a session URL and a `/health` row
# are keyed on, and both exist now; the socket DECLARATION is a separate
# decision that breaks a pinned iOS test the moment it lands, so it waits
# for the iOS lane to take both halves at once.
CARTRIDGE_OBJECT_MEMORY = "object_memory"
CARTRIDGE_SCENE_UNDERSTANDING = "scene_understanding"

# Result types within a cartridge. A cartridge may eventually offer more
# than one; the pair (cartridge, result_type) is what a subscription
# names.
RESULT_TYPE_STATUS = "status"

# Scene Understanding's result type is `live`, not `status`, and the
# difference is the payload rather than the cadence. World Builder's
# `status` describes a BUILD -- how far it has got, what it has accepted.
# Scene Understanding's payload IS the answer: the counts are the product,
# not progress towards one. Naming it `status` would have invited a client
# to render it in the place it renders "building...".
RESULT_TYPE_LIVE = "live"

# Document Memory's result type is `status` -- and there it is the right
# word, because the payload really is progress: how long this Tower has
# been watching for documents, how many it has recorded, whether it is
# mid-dwell. The documents themselves do not travel here. They are bulk,
# they are text, and a list of them belongs on HTTP for the same reason
# World Builder's geometry does -- see `tower/routes/documents.py`.

# World Builder's session status.
#
# Bump the date when the MEANING of an existing field changes (or a
# `model_state` word is added), not merely when an optional field is added:
# a consumer pinned to the old identifier is refused rather than quietly
# served a number that means something different. Compare for equality
# only; a mismatch means we are not talking about the same agreement.
#
# One line per bump:
#   /2026-08-25  `trajectory.pose_count` changed meaning. It had been
#                `keyframes - poses_refused`, which counts a segment
#                anchor -- an identity rotation at the origin, by
#                construction -- as a camera position. One physical walk
#                reported "36 camera poses" from a build whose manifest
#                read `poses_solved: 0, points: 0`.
#   /2026-09-06  `model_state` gained the word `interrupted`, and the
#                payload gained `selection` and `lifecycle.finalization`.
#                A phone on the older agreement refuses an unknown
#                `model_state` outright (an unknown word is a contract
#                disagreement, not an empty world), so it must be told to
#                update rather than shown a decode failure.
#   /2026-09-10  A `lifecycle.state: "stopped_unbuilt"` with no geometry
#                now projects to `interrupted` rather than `finalizing`.
#                No word was added and an older phone decodes both, but a
#                client that switched on `lifecycle.state` and assumed the
#                old projection would draw the wrong screen, silently.
WORLD_BUILDER_STATUS_CONTRACT = "world_builder.status/2026-09-10"

# Scene Understanding's live scene.
#
# The date is when the agreement reached a wire: `where` carries per-label
# SIDE COUNTS rather than one side per label (one side cannot describe a
# chair on the left and a chair on the right), and a `lifecycle` block
# exists because this cartridge has a Start and a Stop that World
# Builder's file-reading status does not.
SCENE_LIVE_CONTRACT = "scene_understanding.live/2026-08-27"

# Document Memory's session status. The library itself is not here; see
# `DOCUMENT_MEMORY_LIBRARY_CONTRACT` below, which governs the HTTP surface.
#
# Moves together with the library identifier, for the same reason:
# `identity` changed meaning. A record may carry SIGHTINGS (later
# observations merged onto it), the session reports them, and a decoder
# written against "no identity across sightings" would render a merged
# record as one observation. A shipped decoder exists, so the identifier
# moves and that decoder says "needs update".
DOCUMENT_MEMORY_STATUS_CONTRACT = "document_memory.status/2026-09-07"

# Document Memory's library, which travels over HTTP rather than on this
# channel. Declared all the same -- see `registry.declare`'s
# `http_contracts` -- because iOS CACHES a declaration, and a contract it
# can only discover by making a call is a contract it cannot plan around.
DOCUMENT_MEMORY_LIBRARY_CONTRACT = "document_memory.library/2026-09-07"

# The Experimental CV Lab status document. Restated here rather than
# imported from `tower/cv_lab/contracts.py`, and a test asserts the two
# are equal: this module is the result channel cartridge-blind core, and
# importing a cartridge to learn its identifier is exactly the coupling
# `test_the_result_channel_core_is_cartridge_blind` forbids. A duplicated
# string that a test pins cannot drift; an import would make the shared
# surface depend on one cartridge package layout.
EXPERIMENTAL_CV_STATUS_CONTRACT = "experimental_cv.status/2026-08-27"

# The channel's own envelope contract, distinct from any cartridge's.
# A change here affects every cartridge at once, which is exactly why it
# is versioned separately from the payloads it carries.
ENVELOPE_CONTRACT = "cartridge_results.envelope/2026-08-23"

# Every timestamp this channel emits. There is no capture timestamp
# anywhere on the wire (tower/frames.py carries no time field), so a
# Tower timestamp is when the TOWER saw something, never when the glasses
# did. `ios/docs/agent-handoffs/IOS-TO-TOWER.md` §0.3 holds observedAt and
# receivedAt separately and will never substitute one for the other; naming the
# basis on every envelope is how that survives the hop.
TIME_BASIS = "tower-receipt"
