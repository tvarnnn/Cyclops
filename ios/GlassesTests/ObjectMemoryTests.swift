//
//  ObjectMemoryTests.swift
//  GlassesTests
//
//  Object Memory's first product surface, tested at the three seams that can
//  actually be wrong on this side of the wire:
//
//  1. **Decoding** — against payloads shaped like the ones the live route
//     produces over the real 55-observation corpus, including the two the
//     contract insists on: an explicit `null` `spatial_ref`, and an empty
//     result set.
//  2. **Transport** — a stubbed `URLProtocol`, so a 404, an unreachable Tower
//     and a contract mismatch are exercised as the three different states they
//     are rather than as one "error".
//  3. **Copy** — the one that matters most. This cartridge knows far less than
//     a reader will assume, and every natural phrasing of what it holds is a
//     claim it cannot support. `ObjectMemoryCopyTests` runs the whole output of
//     `ObjectMemoryCopy` through the sentences it is forbidden to say.
//

import Combine
import Foundation
import XCTest
@testable import Glasses

// MARK: - Fixtures

/// Payloads shaped exactly like the route's.
///
/// Taken from a live dump of `build_observations` / `build_last_seen` against
/// `tower/data/object_memory` — including the values the contract document does
/// not mention, such as `retention_tag: "default"` and the fourth `confidence`
/// word — rather than from the field tables alone.
enum ObjectMemoryFixtures {

    static let contract = "object_memory.observations/2026-08-26"

    static func retention(
        requested: Any = NSNull(), effective: Any = 30.0, clamped: Bool = false
    ) -> [String: Any] {
        [
            "requested_days": requested,
            "effective_days": effective,
            "clamped": clamped,
            "policy": "min(persisted, requested): a reader may narrow this window and can never widen it",
        ]
    }

    /// The `imagery` block, byte for byte as the live route emits it.
    ///
    /// **Templates, not URLs.** The Tower does not know what host a phone
    /// reached it on, so it hands out paths with `{observation_id}` in them and
    /// the client resolves them against the base URL it already holds.
    static func imageryRoutes(
        contract: Any = "object_memory.imagery/2026-08-27",
        claim: Any = "frame-from-the-recording-this-record-was-derived-from",
        filterMeans: Any = "applied-on-read-the-stored-frame-is-unchanged",
        view: Any = "/object-memory/observations/{observation_id}/imagery",
        frame: Any = "/object-memory/observations/{observation_id}/frame",
        crop: Any = "/object-memory/observations/{observation_id}/crop"
    ) -> [String: Any] {
        [
            "contract": contract,
            "claim": claim,
            "filter_means": filterMeans,
            "view": view,
            "frame": frame,
            "crop": crop,
        ]
    }

    static func envelope(
        objectClass: Any = NSNull(),
        retention: [String: Any] = ObjectMemoryFixtures.retention(),
        imagery: Any = ObjectMemoryFixtures.imageryRoutes()
    ) -> [String: Any] {
        [
            "contract": contract,
            "claim": "category-was-visible-once",
            "identity": "category-not-instance",
            "absence_means": "not-observed-by-this-cartridge",
            // Explicit null at the envelope, not omitted. The decoder requires
            // the key to be *present and null*, because an absent key looks
            // like version skew and invites a client to go looking for the
            // value somewhere else.
            "spatial_ref": NSNull(),
            "recorded_classes": ["laptop", "cell phone"],
            // Present on every real payload since 2026-08-27, and **dropped on
            // decode** until this build — which is why iOS showed a wearer a
            // capture id and a frame number and never the frame.
            "imagery": imagery,
            "retention": retention,
            "object_class": objectClass,
        ]
    }

    static func frame(
        sessionID: Any = "22e9d4289cb440fbb3f14e6da369a136",
        frameSeq: Any = 3410,
        box: Any = [0.33483524322509767, 0.5125146865844726, 0.551875008477105, 0.7734769821166992]
    ) -> [String: Any] {
        [
            "kind": "frame-reference",
            "spatial_ref": NSNull(),
            "session_id": sessionID,
            "frame_seq": frameSeq,
            "camera": "glasses-camera",
            "bounding_box_normalized": box,
            "imagery_retention": "capture-side",
        ]
    }

    /// One record, with every field the route actually sends.
    ///
    /// `observation_id` is the handle the three imagery routes take: 16 hex
    /// characters, **derived** from `session_id`, `object_class` and
    /// `observed_at` rather than minted, so a record written before the field
    /// existed has one too.
    static func observation(
        objectClass: String = "cell phone",
        confidence: String = "high",
        detectorScore: Any = 0.9320967793464661,
        bestScore: Any = 0.9950999617576599,
        frame: [String: Any] = ObjectMemoryFixtures.frame(),
        observationID: Any = "9f2c41b7ad0e6538",
        lastSeenAt: Any = 1787695291.118,
        frameCount: Any = 47,
        tier: Any = "remembered",
        verification: Any = NSNull()
    ) -> [String: Any] {
        [
            "object_class": objectClass,
            "claim": "category-was-visible-once",
            "identity": "category-not-instance",
            "confidence": confidence,
            "detector_score": detectorScore,
            "best_score": bestScore,
            "observed_at": 1787695282.4820995,
            "time_basis": "tower-receipt",
            "recorded_at": 1787730242.226801,
            "module_id": "object-memory",
            "retention_tag": "default",
            "privacy_tags": ["derived-only", "frame-referenced"],
            "where": frame,
            "observation_id": observationID,
            "last_seen_at": lastSeenAt,
            "frame_count": frameCount,
            "tier": tier,
            "verification": verification,
        ]
    }

    /// What a second opinion said. Absent on a default Tower — the verifier
    /// ships **off** — so this is only ever used deliberately.
    static func verification(
        agrees: Bool = true,
        label: Any = "mouse",
        score: Any = 0.2871,
        model: Any = "google/owlv2-base-patch16-ensemble"
    ) -> [String: Any] {
        [
            "agrees": agrees,
            "proposed": "mouse",
            "label": label,
            "score": score,
            "model": model,
            "reason": "above-threshold",
        ]
    }

    static func listing(
        observations: [[String: Any]] = [ObjectMemoryFixtures.observation()],
        objectClass: Any = NSNull(),
        count: Int? = nil
    ) -> [String: Any] {
        var payload = envelope(objectClass: objectClass)
        payload["observation_count"] = count ?? observations.count
        payload["observations"] = observations
        return payload
    }

    /// An empty listing.
    ///
    /// `[[String: Any]]()` rather than `[]`. An empty collection literal in an
    /// `Any` slot has no element type to infer and does not compile — a trap
    /// this codebase has already been bitten by once.
    static func emptyListing(objectClass: Any = NSNull()) -> [String: Any] {
        listing(observations: [[String: Any]](), objectClass: objectClass)
    }

    static func lastSeen(
        objectClass: String = "cell phone",
        recordable: Bool = true,
        observed: Bool = true,
        observation: Any? = nil
    ) -> [String: Any] {
        var payload = envelope(objectClass: objectClass)
        payload["recordable"] = recordable
        payload["observed"] = observed
        payload["observation"] = observation ?? (observed ? self.observation() : NSNull())
        payload["where"] = observed ? frame() : NSNull()
        return payload
    }

    /// The silence for a class the cartridge never writes.
    static func neverLookedFor(objectClass: String = "teapot") -> [String: Any] {
        var payload = envelope(objectClass: objectClass)
        payload["recordable"] = false
        payload["observed"] = false
        payload["observation"] = NSNull()
        payload["where"] = NSNull()
        return payload
    }

    // MARK: The session

    /// A session reading, byte for byte as the live route emits it — including
    /// `cartridge` and `worker`, which are on the wire and absent from the
    /// contract document's field table.
    static func session(
        state: Any = "stopped",
        supported: Bool = true,
        sessionID: Any = NSNull(),
        startedAt: Any = NSNull(),
        following: Any = [String](),
        // `nil` OMITS the key, which is what a Tower older than 2026-08-29
        // sends and is a different thing from sending `[]`. Omission is the
        // default so that every case written before the field existed keeps
        // exercising the fallback path, which is what those cases are about.
        followingThisSession: [String]? = nil,
        captures: Any = [String](),
        contract: Any = "cartridge_session.control/2026-08-27",
        stateMeans: Any = "intent-not-liveness",
        actions: Any = ["start", "pause", "resume", "stop"]
    ) -> [String: Any] {
        var payload: [String: Any] = [
            "cartridge": "object_memory",
            "worker": "object-memory-session",
            "supported": supported,
            "state": state,
            "session_id": sessionID,
            "started_at": startedAt,
            "changed_at": 1787832967.312166,
            "following": following,
            "captures": captures,
            "contract": contract,
            "state_means": stateMeans,
            "states": ["stopped", "active", "paused"],
            "actions": actions,
        ]
        if let followingThisSession {
            payload["following_this_session"] = followingThisSession
        }
        return payload
    }

    /// The same, with the three fields a `POST` adds.
    static func sessionAction(
        state: Any = "active",
        changed: Bool = true,
        attachedCaptureID: Any = NSNull(),
        sessionID: Any = "61a78a9b32284cd2a89583d9a8cc8702",
        startedAt: Any = 1787833925.6521091,
        following: Any = [String]()
    ) -> [String: Any] {
        var payload = session(
            state: state, sessionID: sessionID, startedAt: startedAt, following: following
        )
        payload["accepted"] = true
        payload["changed"] = changed
        payload["attached_capture_id"] = attachedCaptureID
        return payload
    }

    /// A 409 body, **inside FastAPI's `detail` wrapper** exactly as it arrives.
    ///
    /// `reason` is a parameter with no default worth calling safe, because the
    /// contract document and the running Tower disagree about which word a
    /// `resume` from `stopped` carries — the document says `not-active`, the
    /// wire says `not-paused` — and every test here exercises both.
    static func sessionRefusal(
        reason: String,
        message: String = "there is no paused session to resume; this cartridge is stopped",
        state: Any = "stopped"
    ) -> [String: Any] {
        var body = session(state: state)
        body["accepted"] = false
        body["reason"] = reason
        body["message"] = message
        return ["detail": body]
    }

    // MARK: The pictures

    /// An `/imagery` description. Defaults to a served picture.
    static func imagery(
        observationID: Any = "9f2c41b7ad0e6538",
        objectClass: Any = "laptop",
        available: Bool = true,
        reason: Any = NSNull(),
        memoryRetained: Bool = true,
        filter: Any = "display-filter/yunet-2023mar@0.30",
        regionsFilled: Any = 0,
        subjectObscured: Any = 0.0,
        box: Any = [0.1120081901550293, 0.6517109870910645, 0.44075003729926215, 0.9019964218139649],
        contract: Any = "object_memory.imagery/2026-08-27",
        claim: Any = "frame-from-the-recording-this-record-was-derived-from",
        filterMeans: Any = "applied-on-read-the-stored-frame-is-unchanged",
        // `nil` OMITS the key, exactly as `followingThisSession` does on the
        // session fixture and for the same reason: a Tower older than these
        // two fields sends neither, and "it did not say" is a different thing
        // from "it said no". Omission is the default so that every case
        // written before the fields existed keeps exercising the older-Tower
        // path, which is what those cases are about.
        frameAvailable: Bool? = nil,
        imagerySource: String? = nil,
        imageryRetention: Any = "capture-side"
    ) -> [String: Any] {
        var payload: [String: Any] = [
            "contract": contract,
            "observation_id": observationID,
            "object_class": objectClass,
            "claim": claim,
            "available": available,
            "reason": reason,
            "memory_retained": memoryRetained,
            "filter": filter,
            "filter_means": filterMeans,
            "regions_filled": regionsFilled,
            "subject_obscured": subjectObscured,
            "bounding_box_normalized": box,
            "imagery_retention": imageryRetention,
        ]
        if let frameAvailable { payload["frame_available"] = frameAvailable }
        if let imagerySource { payload["imagery_source"] = imagerySource }
        return payload
    }

    /// **The 410.** The pointer is intact and the picture is gone.
    ///
    /// `memory_retained: true` with `available: false` is the entire reason the
    /// payload has this shape, and `filter` is null because nothing was served
    /// and therefore nothing was filtered.
    static func imageryGone() -> [String: Any] {
        [
            "detail": imagery(
                available: false,
                reason: "imagery-no-longer-available",
                memoryRetained: true,
                filter: NSNull()
            )
        ]
    }

    /// A 503: this Tower serves no picture at all. **There is no lenient
    /// default here, because the lenient default is a raw first-person frame.**
    static func imageryRefused(reason: String) -> [String: Any] {
        [
            "detail": imagery(
                available: false, reason: reason, memoryRetained: true, filter: NSNull()
            )
        ]
    }

    static func data(_ json: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: json)
    }
}

// MARK: - Decoding

@MainActor
final class ObjectMemoryDecodingTests: XCTestCase {

    func testAListingDecodesFieldForField() throws {
        let listing = try ObjectMemoryDecoder.listing(from: ObjectMemoryFixtures.listing())

        XCTAssertEqual(listing.envelope.contract, ObjectMemoryContract.identifier)
        XCTAssertEqual(listing.envelope.recordedClasses, ["laptop", "cell phone"])
        XCTAssertNil(listing.envelope.objectClass, "an unfiltered listing narrowed to no class")
        XCTAssertEqual(listing.observations.count, 1)

        let observation = try XCTUnwrap(listing.observations.first)
        XCTAssertEqual(observation.objectClass, "cell phone")
        XCTAssertEqual(observation.confidence, .high)
        XCTAssertEqual(observation.timeBasis, "tower-receipt")
        XCTAssertEqual(observation.moduleID, "object-memory")
        XCTAssertEqual(observation.retentionTag, "default")
        XCTAssertEqual(observation.privacyTags, ["derived-only", "frame-referenced"])
        XCTAssertEqual(observation.frame.sessionID, "22e9d4289cb440fbb3f14e6da369a136")
        XCTAssertEqual(observation.frame.frameSeq, 3410)
        XCTAssertEqual(observation.frame.camera, "glasses-camera")
        XCTAssertEqual(observation.frame.imageryRetention, "capture-side")
        XCTAssertEqual(observation.frame.boundingBoxNormalized?.count, 4)
    }

    /// The retention view is the sharpest constraint in the contract, and the
    /// clamp is *reported* rather than merely applied.
    func testTheRetentionWindowDecodesIncludingTheClamp() throws {
        let payload = ObjectMemoryFixtures.listing()
        let listing = try ObjectMemoryDecoder.listing(from: payload)
        XCTAssertNil(listing.envelope.retention.requestedDays, "nothing was asked for")
        XCTAssertEqual(listing.envelope.retention.effectiveDays, 30.0)
        XCTAssertFalse(listing.envelope.retention.clamped)

        let clampedPayload = ObjectMemoryFixtures.listing()
        var clamped = clampedPayload
        clamped["retention"] = ObjectMemoryFixtures.retention(
            requested: 3650.0, effective: 30.0, clamped: true
        )
        let refused = try ObjectMemoryDecoder.listing(from: clamped)
        XCTAssertEqual(refused.envelope.retention.requestedDays, 3650.0)
        XCTAssertEqual(refused.envelope.retention.effectiveDays, 30.0)
        XCTAssertTrue(refused.envelope.retention.clamped, "a refused widening must be visible")
    }

    /// `effective_days == nil` is **unbounded**, and is never `0` — zero days
    /// would be the opposite claim.
    func testAnUnboundedWindowStaysNilRatherThanBecomingZero() throws {
        var payload = ObjectMemoryFixtures.listing()
        payload["retention"] = ObjectMemoryFixtures.retention(effective: NSNull())
        let listing = try ObjectMemoryDecoder.listing(from: payload)
        XCTAssertNil(listing.envelope.retention.effectiveDays)
    }

    /// The whole point of the cartridge, asserted on the type: `spatial_ref` is
    /// carried as an explicit null and there is nothing to read out of it.
    func testAnExplicitNullSpatialRefIsRequiredAndCarriesNothing() throws {
        let payload = ObjectMemoryFixtures.listing()
        XCTAssertTrue(
            ObjectMemoryDecoder.isExplicitlyNull(payload, key: "spatial_ref"),
            "the fixture must actually carry the null this test is about"
        )
        XCTAssertNoThrow(try ObjectMemoryDecoder.listing(from: payload))
    }

    /// An omitted key is not the same as a null one, and the contract sends the
    /// null deliberately.
    func testAnOmittedSpatialRefIsRefused() {
        var payload = ObjectMemoryFixtures.listing()
        payload.removeValue(forKey: "spatial_ref")
        XCTAssertNil(ObjectMemoryDecoder.envelope(from: payload))
    }

    /// A **populated** `spatial_ref` is refused rather than ignored. It would be
    /// a claim about where something is, and this app has no honest way to show
    /// one — decoding around it is how a reserved field becomes a map pin.
    func testAPopulatedSpatialRefIsRefusedRatherThanIgnored() {
        var frame = ObjectMemoryFixtures.frame()
        frame["spatial_ref"] = ["x": 1.0, "y": 2.0, "z": 3.0]
        XCTAssertNil(ObjectMemoryDecoder.frame(from: frame))

        var payload = ObjectMemoryFixtures.listing()
        payload["spatial_ref"] = ["frame": "world"]
        XCTAssertNil(ObjectMemoryDecoder.envelope(from: payload))
    }

    /// `null` means absent. Never `0`, never `""`.
    func testNullScoresStayNilRatherThanBecomingZero() throws {
        let payload = ObjectMemoryFixtures.listing(observations: [
            ObjectMemoryFixtures.observation(
                confidence: "unknown", detectorScore: NSNull(), bestScore: NSNull()
            )
        ])
        let listing = try ObjectMemoryDecoder.listing(from: payload)
        let observation = try XCTUnwrap(listing.observations.first)

        XCTAssertNil(observation.detectorScore)
        XCTAssertNil(observation.bestScore)
        XCTAssertNotEqual(observation.detectorScore, 0)
        XCTAssertNotEqual(observation.bestScore, 0)
    }

    /// `unknown` is a real fourth value on this wire.
    ///
    /// `docs/contracts/OBJECT-MEMORY.md` §4.4 lists three; `tower/confidence.py`
    /// defines four, and `Confidence.from_score(None)` returns the fourth. A
    /// decoder written from the document alone refuses a real record.
    func testTheFourthConfidenceValueDecodes() throws {
        for word in ["unknown", "low", "medium", "high"] {
            let payload = ObjectMemoryFixtures.listing(observations: [
                ObjectMemoryFixtures.observation(confidence: word)
            ])
            let listing = try ObjectMemoryDecoder.listing(from: payload)
            XCTAssertEqual(listing.observations.first?.confidence.rawValue, word)
        }
    }

    /// An empty result set is an **answer**, decoded like any other.
    func testAnEmptyListingDecodesAsAnAnswerRatherThanAFailure() throws {
        let listing = try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.emptyListing(objectClass: "laptop")
        )
        XCTAssertTrue(listing.isEmpty)
        XCTAssertEqual(listing.observations.count, 0)
        XCTAssertEqual(listing.envelope.objectClass, "laptop")
        // The universe of classes still arrives, which is what lets an empty
        // answer distinguish "looked for and not seen" from "never looked for".
        XCTAssertEqual(listing.envelope.recordedClasses, ["laptop", "cell phone"])
    }

    /// A count that disagrees with the array is a broken payload, and the safe
    /// direction from a broken payload is a failure — never a smaller number of
    /// records, which reads as "you saw fewer things than you did".
    func testACountThatDisagreesWithTheArrayIsRefused() {
        let payload = ObjectMemoryFixtures.listing(count: 7)
        XCTAssertThrowsError(try ObjectMemoryDecoder.listing(from: payload)) { error in
            XCTAssertEqual((error as? CartridgeFailure)?.kind, .undecodableResponse)
        }
    }

    func testADifferentContractIsNotDecoded() {
        var payload = ObjectMemoryFixtures.listing()
        payload["contract"] = "object_memory.observations/2027-01-01"
        XCTAssertNil(ObjectMemoryDecoder.envelope(from: payload))
        XCTAssertEqual(
            ObjectMemoryDecoder.contractIdentifier(from: payload),
            "object_memory.observations/2027-01-01",
            "the identifier is still readable, so the app can say which one it saw"
        )
    }

    /// A change to any of the three claim values is the most breaking change
    /// this contract can carry: it changes what the data *means*. Rendering
    /// such a payload under the old wording is the worst failure available.
    func testAChangedClaimIsRefused() {
        for key in ["claim", "identity", "absence_means"] {
            var payload = ObjectMemoryFixtures.listing()
            payload[key] = "something-else"
            XCTAssertNil(
                ObjectMemoryDecoder.envelope(from: payload),
                "\(key) changed meaning and the payload still decoded"
            )
        }
        // Per record too, because a record travels out of the context of its
        // envelope the moment it is put in a list row.
        var record = ObjectMemoryFixtures.observation()
        record["identity"] = "instance"
        XCTAssertNil(ObjectMemoryDecoder.observation(from: record))
    }

    func testAWhereThatIsNotAFrameReferenceIsRefused() {
        var frame = ObjectMemoryFixtures.frame()
        frame["kind"] = "world-position"
        XCTAssertNil(ObjectMemoryDecoder.frame(from: frame))
    }

    /// A short box is a broken box, and half a rectangle is worse than none.
    func testAMalformedBoundingBoxIsRefused() {
        let frame = ObjectMemoryFixtures.frame(box: [0.1, 0.2])
        XCTAssertNil(ObjectMemoryDecoder.frame(from: frame))
    }

    /// A record with no capture provenance keeps `nil`, not a zeroth session.
    func testARecordWithoutCaptureProvenanceKeepsNils() throws {
        let frame = ObjectMemoryFixtures.frame(
            sessionID: NSNull(), frameSeq: NSNull(), box: NSNull()
        )
        let decoded = try XCTUnwrap(ObjectMemoryDecoder.frame(from: frame))
        XCTAssertNil(decoded.sessionID)
        XCTAssertNil(decoded.frameSeq)
        XCTAssertNil(decoded.boundingBoxNormalized)
        XCTAssertFalse(decoded.pointsAtACapture)
    }

    func testLastSeenDecodesASighting() throws {
        let answer = try ObjectMemoryDecoder.lastSeen(from: ObjectMemoryFixtures.lastSeen())
        XCTAssertTrue(answer.observed)
        XCTAssertTrue(answer.recordable)
        XCTAssertEqual(answer.objectClass, "cell phone")
        XCTAssertNotNil(answer.observation)
        XCTAssertEqual(answer.frame?.frameSeq, 3410)
    }

    /// The silence, decoded as a silence. 200, not 404 — a 404 would read as
    /// "there is no laptop", which is a claim about the world.
    func testLastSeenDecodesAnHonestSilence() throws {
        let answer = try ObjectMemoryDecoder.lastSeen(
            from: ObjectMemoryFixtures.lastSeen(objectClass: "laptop", observed: false)
        )
        XCTAssertFalse(answer.observed)
        XCTAssertTrue(answer.recordable, "laptop is a class this cartridge does write")
        XCTAssertNil(answer.observation)
        XCTAssertNil(answer.frame)
    }

    /// A class the cartridge never writes is a *weaker* silence, and the
    /// payload distinguishes it.
    func testAClassThatWasNeverLookedForSaysSo() throws {
        let answer = try ObjectMemoryDecoder.lastSeen(from: ObjectMemoryFixtures.neverLookedFor())
        XCTAssertFalse(answer.recordable)
        XCTAssertFalse(answer.observed)
    }

    /// `observed: true` with no record is not a coherent answer. Quietly
    /// rewriting it to "nothing observed" would manufacture a negative
    /// statement about the wearer's memory out of a decode failure.
    func testASightingWithNoRecordIsRefusedRatherThanEmptied() {
        var payload = ObjectMemoryFixtures.lastSeen()
        payload["observation"] = NSNull()
        XCTAssertThrowsError(try ObjectMemoryDecoder.lastSeen(from: payload)) { error in
            XCTAssertEqual((error as? CartridgeFailure)?.kind, .undecodableResponse)
        }
    }

    func testARecordWithNoSightingIsRefused() {
        var payload = ObjectMemoryFixtures.lastSeen()
        payload["observed"] = false
        XCTAssertThrowsError(try ObjectMemoryDecoder.lastSeen(from: payload))
    }
}

// MARK: - Transport

/// Answers HTTP requests without a network.
///
/// Deliberately holds only Foundation types, so it stays outside the app
/// target's default `MainActor` isolation and can be driven from the URL
/// loading system's own threads.
final class ObjectMemoryStubProtocol: URLProtocol {
    /// What to answer, and the requests that were made. Test-only global state,
    /// reset in `setUp`.
    static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?
    static var requestedURLs: [URL] = []
    /// The whole request, not just its URL.
    ///
    /// A URL alone cannot answer "did the request that actually ran carry the
    /// deadline the client claims to set?", and a test that reads the client's
    /// own `timeout` property back instead is reading the literal it was
    /// initialised with — it stays green when the `timeoutInterval` is dropped
    /// from the `URLRequest` and the call silently inherits `URLSession`'s
    /// 60-second default.
    static var requestedRequests: [URLRequest] = []

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        if let url = request.url { ObjectMemoryStubProtocol.requestedURLs.append(url) }
        ObjectMemoryStubProtocol.requestedRequests.append(request)
        guard let handler = ObjectMemoryStubProtocol.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

@MainActor
final class ObjectMemoryTransportTests: XCTestCase {

    private let baseURL = URL(string: "http://tower.test")!

    private func makeClient() -> ObjectMemoryHTTPClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ObjectMemoryStubProtocol.self]
        return ObjectMemoryHTTPClient(
            baseURL: baseURL, session: URLSession(configuration: configuration)
        )
    }

    private func respond(_ status: Int, _ json: [String: Any]) {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, try JSONSerialization.data(withJSONObject: json))
        }
    }

    override func setUp() {
        super.setUp()
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
    }

    override func tearDown() {
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
        super.tearDown()
    }

    func testAListingIsFetchedAndDecoded() async throws {
        respond(200, ObjectMemoryFixtures.listing())
        let listing = try await makeClient().listing()
        XCTAssertEqual(listing.observations.count, 1)
        XCTAssertEqual(
            ObjectMemoryStubProtocol.requestedURLs.first?.path, "/object-memory/observations"
        )
    }

    func testNarrowingByClassAndWindowTravelsAsQuery() async throws {
        respond(200, ObjectMemoryFixtures.listing(objectClass: "laptop"))
        _ = try await makeClient().listing(objectClass: "laptop", retentionDays: 0.5)

        let url = try XCTUnwrap(ObjectMemoryStubProtocol.requestedURLs.first)
        let components = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let items = components.queryItems ?? []
        XCTAssertTrue(items.contains(URLQueryItem(name: "object_class", value: "laptop")))
        XCTAssertEqual(items.first { $0.name == "retention_days" }?.value, "0.5")
    }

    /// A class with a space in it is one of the two real ones.
    func testAClassWithASpaceIsPercentEncodedIntoThePath() async throws {
        respond(200, ObjectMemoryFixtures.lastSeen())
        _ = try await makeClient().lastSeen(objectClass: "cell phone")

        let url = try XCTUnwrap(ObjectMemoryStubProtocol.requestedURLs.first)
        XCTAssertEqual(url.path, "/object-memory/last-seen/cell phone")
        XCTAssertTrue(
            url.absoluteString.contains("cell%20phone"),
            "the space must be encoded rather than splitting the path: \(url.absoluteString)"
        )
    }

    /// 404 means **this Tower serves no object memory**. It is a statement
    /// about configuration and must never be reported as an answer about a
    /// class.
    func testAConfigurationFourOhFourIsItsOwnCase() async {
        respond(404, ["detail": "no object memory root is configured"])
        do {
            _ = try await makeClient().listing()
            XCTFail("a 404 decoded as an answer")
        } catch let error as ObjectMemoryFetchError {
            XCTAssertEqual(error, .noObjectMemoryConfigured)
        } catch {
            XCTFail("expected an ObjectMemoryFetchError, got \(error)")
        }
    }

    func testADifferentContractIsReportedRatherThanDecoded() async {
        var payload = ObjectMemoryFixtures.listing()
        payload["contract"] = "object_memory.observations/2027-01-01"
        respond(200, payload)
        do {
            _ = try await makeClient().listing()
            XCTFail("a foreign contract was decoded anyway")
        } catch let error as ObjectMemoryFetchError {
            XCTAssertEqual(
                error, .unsupportedContract(identifier: "object_memory.observations/2027-01-01")
            )
        } catch {
            XCTFail("expected an ObjectMemoryFetchError, got \(error)")
        }
    }

    /// The Tower being unreachable is a first-class state, not a crash and not
    /// an empty answer.
    func testAnUnreachableTowerBecomesATransportFailure() async {
        ObjectMemoryStubProtocol.handler = { _ in throw URLError(.cannotConnectToHost) }
        do {
            _ = try await makeClient().listing()
            XCTFail("an unreachable Tower produced an answer")
        } catch let error as ObjectMemoryFetchError {
            guard case .transport = error else {
                return XCTFail("expected .transport, got \(error)")
            }
        } catch {
            XCTFail("expected an ObjectMemoryFetchError, got \(error)")
        }
    }

    func testAnUnreadableBodyIsUndecodableRatherThanEmpty() async {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil
            )!
            return (response, Data("not json".utf8))
        }
        do {
            _ = try await makeClient().listing()
            XCTFail("garbage decoded as an answer")
        } catch let error as ObjectMemoryFetchError {
            XCTAssertEqual(error, .undecodable)
        } catch {
            XCTFail("expected an ObjectMemoryFetchError, got \(error)")
        }
    }
}

// MARK: - Client state

@MainActor
final class ObjectMemoryClientStateTests: XCTestCase {

    private let baseURL = URL(string: "http://tower.test")!

    private func makeClient() -> TowerObjectMemoryClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ObjectMemoryStubProtocol.self]
        return TowerObjectMemoryClient(
            http: ObjectMemoryHTTPClient(
                baseURL: baseURL, session: URLSession(configuration: configuration)
            )
        )
    }

    override func setUp() {
        super.setUp()
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
    }

    override func tearDown() {
        ObjectMemoryStubProtocol.handler = nil
        super.tearDown()
    }

    private func respond(_ status: Int, _ json: [String: Any]) {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, try JSONSerialization.data(withJSONObject: json))
        }
    }

    /// Construction touches nothing. No request until a person asks one.
    func testAFreshClientHasAskedNothing() {
        let client = makeClient()
        XCTAssertEqual(client.state, .idle)
        XCTAssertEqual(client.service, .unprobed)
        XCTAssertTrue(ObjectMemoryStubProtocol.requestedURLs.isEmpty)
    }

    /// The one case the shell's four-word availability vocabulary cannot say:
    /// nothing has been *asked*, which is not the same as nothing having been
    /// *declared*. Hiding the control that would find out is a loop that cannot
    /// terminate, so the workspace reads `knownAvailability` and gets `nil`.
    func testAvailabilityIsUnknownUntilSomethingHasBeenAsked() {
        let client = makeClient()
        XCTAssertNil(client.knownAvailability(isTowerReachable: true))
        // Projected conservatively for anything that needs a total answer.
        XCTAssertEqual(client.availability(isTowerReachable: true), .noContract)
        // With the socket down there is something true to say without asking.
        XCTAssertEqual(client.knownAvailability(isTowerReachable: false), .towerUnreachable)
    }

    func testAnAnswerTeachesTheClientWhichContractTheTowerSpeaks() async {
        respond(200, ObjectMemoryFixtures.listing())
        let client = makeClient()
        await client.ask(.listing(objectClass: nil))

        XCTAssertEqual(client.state.phase, .settled)
        XCTAssertNotNil(client.state.answer)
        XCTAssertEqual(
            client.service,
            .serving(
                CartridgeContract(
                    cartridgeID: "object-memory", identifier: ObjectMemoryContract.identifier
                )
            )
        )
        XCTAssertEqual(
            client.availability(isTowerReachable: true),
            .available(
                CartridgeContract(
                    cartridgeID: "object-memory", identifier: ObjectMemoryContract.identifier
                )
            )
        )
    }

    /// An empty answer is an answer: settled, not failed, and not empty-because
    /// -something-went-wrong.
    func testAnEmptyAnswerSettlesRatherThanFailing() async throws {
        respond(200, ObjectMemoryFixtures.emptyListing(objectClass: "laptop"))
        let client = makeClient()
        await client.ask(.listing(objectClass: "laptop"))

        XCTAssertEqual(client.state.phase, .settled)
        let answer = try XCTUnwrap(client.state.answer)
        XCTAssertFalse(answer.hasRecord, "an empty answer is still an answer")
    }

    /// A Tower with no object memory configured gets its own state, and it is
    /// never reported as a failure or as an empty memory.
    func testATowerWithNoObjectMemoryIsItsOwnState() async {
        respond(404, ["detail": "no object memory root is configured"])
        let client = makeClient()
        await client.ask(.listing(objectClass: nil))

        XCTAssertEqual(client.state, .noObjectMemory)
        XCTAssertEqual(client.state.phase, .unsupported)
        XCTAssertEqual(client.service, .notConfigured)
        XCTAssertEqual(client.availability(isTowerReachable: true), .noContract)
    }

    /// An unreachable Tower is drawn as `.disconnected`, not `.failed`: the
    /// capability exists and could not be reached, which calls for a different
    /// headline and glyph than a broken answer.
    func testAnUnreachableTowerIsADisconnectedStateNotAnError() async {
        ObjectMemoryStubProtocol.handler = { _ in throw URLError(.cannotConnectToHost) }
        let client = makeClient()
        await client.ask(.listing(objectClass: nil))

        XCTAssertEqual(client.state.phase, .disconnected)
        XCTAssertEqual(client.availability(isTowerReachable: true), .towerUnreachable)
        guard case .failed(let failure) = client.state else {
            return XCTFail("expected a failed state carrying the reason")
        }
        XCTAssertEqual(failure.kind, .transport)
        XCTAssertFalse(failure.message.isEmpty)
        XCTAssertNil(client.state.answer, "a phase that may not carry data carried some")
    }

    /// A phase that may not carry data must not carry any. The invariant
    /// `CartridgePhase.mayCarryData` exists to be checked, not assumed.
    func testNoStateCarriesDataItsPhaseForbids() async {
        respond(404, ["detail": "no object memory root is configured"])
        let client = makeClient()
        await client.ask(.listing(objectClass: nil))
        XCTAssertFalse(client.state.phase.mayCarryData)
        XCTAssertNil(client.state.answer)
    }

    /// An unreadable answer says nothing about whether the Tower serves object
    /// memory, so what was already learned is not thrown away.
    func testAnUnreadableAnswerDoesNotForgetTheTower() async {
        respond(200, ObjectMemoryFixtures.listing())
        let client = makeClient()
        await client.ask(.listing(objectClass: nil))

        respond(200, ObjectMemoryFixtures.listing(count: 9))
        await client.ask(.listing(objectClass: nil))

        XCTAssertEqual(client.state.phase, .failed)
        XCTAssertEqual(
            client.service,
            .serving(
                CartridgeContract(
                    cartridgeID: "object-memory", identifier: ObjectMemoryContract.identifier
                )
            ),
            "one bad payload reported the Tower as absent"
        )
    }

    func testTheClientAnswersForItsOwnCartridgeOnly() {
        XCTAssertEqual(makeClient().cartridgeID, "object-memory")
        XCTAssertTrue(Cartridge.catalog.contains { $0.id == "object-memory" })
    }
}

// MARK: - View model

/// A client that reports whatever a test needs, without a network.
@MainActor
final class StubObjectMemoryClient: ObjectMemoryClient {
    let cartridgeID = "object-memory"
    var state: ObjectMemoryState
    var service: ObjectMemoryService
    /// `nil` means "nothing has been asked", exactly as the real client reports
    /// it before a probe.
    var stubbedAvailability: CartridgeAvailability?
    private(set) var asked: [ObjectMemoryQuestion] = []

    /// The session half. Same discipline: whatever a test needs, with the
    /// verbs it was sent recorded so a test can assert what the workspace
    /// actually did rather than what it displayed afterwards.
    var session: ObjectMemorySessionState
    private(set) var applied: [CartridgeSessionAction] = []
    private(set) var sessionReads = 0
    /// What the next `readSession` should report. Lets a test model the
    /// producer that ignored `SIGTERM`: a Pause that answers `paused` and a
    /// read-back that still names the capture.
    var sessionAfterRead: ObjectMemorySessionState?

    /// Successive readings, consumed in order by `readSession` and `apply`.
    ///
    /// `sessionAfterRead` answers the same reading every time, which cannot
    /// model a *sequence* — a Start that answers `active` with nothing
    /// following, then a read a moment later that names a capture. Empty by
    /// default, and when it is empty `sessionAfterRead` behaves exactly as it
    /// did, so no existing test changes meaning.
    var sessionScript: [ObjectMemorySessionState] = []

    /// A shared ordering log. `nil` unless a test is asserting the *order* two
    /// collaborators were called in — which, for the composed lifecycle, is the
    /// single most important thing about it.
    var journal: RecordingJournal?

    /// The imagery half, and what was asked for.
    var stubbedImagery: ObjectMemoryImageryAnswer = .routesUnknown
    var stubbedPicture: ObjectMemoryPictureAnswer = .routesUnknown
    private(set) var imageryAsked: [String] = []
    private(set) var picturesAsked: [(String, ObjectMemoryImageryKind)] = []

    private let subject = PassthroughSubject<ObjectMemoryState, Never>()
    var stateUpdates: AnyPublisher<ObjectMemoryState, Never> { subject.eraseToAnyPublisher() }

    private let sessionSubject = PassthroughSubject<ObjectMemorySessionState, Never>()
    var sessionUpdates: AnyPublisher<ObjectMemorySessionState, Never> {
        sessionSubject.eraseToAnyPublisher()
    }

    init(
        state: ObjectMemoryState = .idle,
        service: ObjectMemoryService = .unprobed,
        availability: CartridgeAvailability? = nil,
        session: ObjectMemorySessionState = .unread
    ) {
        self.state = state
        self.service = service
        self.stubbedAvailability = availability
        self.session = session
    }

    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        stubbedAvailability ?? .noContract
    }

    func knownAvailability(isTowerReachable: Bool) -> CartridgeAvailability? {
        stubbedAvailability
    }

    func ask(_ question: ObjectMemoryQuestion) async {
        asked.append(question)
    }

    func readSession() async {
        sessionReads += 1
        journal?.record("tower:read")
        answerFromTheScript()
    }

    func apply(_ action: CartridgeSessionAction) async {
        applied.append(action)
        journal?.record("tower:\(action.rawValue)")
        answerFromTheScript()
    }

    /// The scripted reading if there is one, otherwise the standing one.
    private func answerFromTheScript() {
        if !sessionScript.isEmpty {
            publish(sessionScript.removeFirst())
        } else if let next = sessionAfterRead {
            publish(next)
        }
    }

    func imageryDescription(for observationID: String) async -> ObjectMemoryImageryAnswer {
        imageryAsked.append(observationID)
        return stubbedImagery
    }

    func picture(
        for observationID: String, kind: ObjectMemoryImageryKind
    ) async -> ObjectMemoryPictureAnswer {
        picturesAsked.append((observationID, kind))
        return stubbedPicture
    }

    func publish(_ state: ObjectMemoryState) {
        self.state = state
        subject.send(state)
    }

    func publish(_ session: ObjectMemorySessionState) {
        self.session = session
        sessionSubject.send(session)
    }
}

@MainActor
final class ObjectMemoryViewModelTests: XCTestCase {

    /// Lets one main-queue turn run.
    ///
    /// The view model subscribes with `.receive(on: DispatchQueue.main)` — which
    /// `TowerWorldBuilderClient` documents as load-bearing rather than
    /// stylistic — so a state published in a test is delivered on the *next*
    /// turn, and a synchronous assertion after `publish` would run first and
    /// see the old value.
    private func settle() async {
        try? await Task.sleep(nanoseconds: 50_000_000)
    }

    func testAnUnaskedWorkspaceIsIdleRatherThanUnsupported() {
        let memory = ObjectMemoryViewModel(client: StubObjectMemoryClient())
        XCTAssertEqual(memory.phase(isTowerReachable: true), .idle)
        XCTAssertTrue(memory.recordableClasses.isEmpty, "the app must not assert the Tower's list")
    }

    /// Availability outranks the client's own state once it is *known*.
    func testAKnownUnavailabilityOutranksTheState() {
        let client = StubObjectMemoryClient(
            state: .idle, service: .notConfigured, availability: .noContract
        )
        let memory = ObjectMemoryViewModel(client: client)
        XCTAssertEqual(memory.phase(isTowerReachable: true), .unsupported)
        XCTAssertEqual(
            memory.unavailableExplanation(isTowerReachable: true),
            ObjectMemoryCopy.noObjectMemoryConfigured,
            "the shared 'has not declared a contract' sentence is wrong for a 404"
        )
    }

    func testTheCategoryListComesFromTheAnswerNotFromThisApp() async throws {
        let listing = try ObjectMemoryDecoder.listing(from: ObjectMemoryFixtures.listing())
        let client = StubObjectMemoryClient()
        let memory = ObjectMemoryViewModel(client: client)
        XCTAssertTrue(memory.recordableClasses.isEmpty)

        client.publish(
            .answered(question: .listing(objectClass: nil), answer: .listing(listing))
        )
        await settle()
        XCTAssertEqual(memory.recordableClasses, ["laptop", "cell phone"])
    }

    func testAskingForOneCategoryCarriesTheSelection() async {
        let client = StubObjectMemoryClient()
        let memory = ObjectMemoryViewModel(client: client)
        memory.selectedClass = "laptop"
        memory.askWhenLastInView()

        // `ask` hops through a `Task`, so the question reaches the client on a
        // later turn rather than inside the call.
        var turns = 0
        while client.asked.isEmpty && turns < 100 {
            await Task.yield()
            turns += 1
        }
        XCTAssertEqual(client.asked, [.lastSeen(objectClass: "laptop")])
    }

    /// "When was it last in view" needs a category. With none selected there is
    /// no question to ask, and asking one anyway would answer about a class
    /// nobody named.
    func testLastInViewWithoutACategoryAsksNothing() {
        let client = StubObjectMemoryClient()
        let memory = ObjectMemoryViewModel(client: client)
        memory.selectedClass = nil
        memory.askWhenLastInView()
        XCTAssertTrue(client.asked.isEmpty)
    }
}

// MARK: - Copy

/// **The test this cartridge exists to pass.**
///
/// Object Memory knows three things: that a category was visible, once, and
/// when. It does not know whose object it was, where it is, or whether it is
/// still anywhere. Every fluent English sentence about "the laptop you saw"
/// claims at least one of those, so the copy is checked rather than trusted.
///
/// It asserts over `ObjectMemoryCopy`'s entire output, which is also the only
/// source the workspace view renders from.
@MainActor
final class ObjectMemoryCopyTests: XCTestCase {

    private func listing() throws -> ObservationListing {
        try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.listing(observations: [
                ObjectMemoryFixtures.observation(objectClass: "laptop"),
                ObjectMemoryFixtures.observation(
                    objectClass: "cell phone",
                    confidence: "unknown",
                    detectorScore: NSNull(),
                    bestScore: NSNull(),
                    frame: ObjectMemoryFixtures.frame(
                        sessionID: NSNull(), frameSeq: NSNull(), box: NSNull()
                    )
                ),
            ])
        )
    }

    /// Sentences no string may contain, whatever it is describing.
    ///
    /// Two kinds. The class-independent ones are present-tense location and
    /// possession claims; the generated ones close the same door around each
    /// category the Tower can record, because "the laptop is on the desk" is
    /// the exact sentence a reader wants and the exact one that is false.
    private func forbiddenPhrases(classes: [String]) -> [String] {
        var phrases = [
            "still there",
            "is still",
            "right now",
            "at the moment",
            "on the map",
            "last known location",
            "location of",
            "we know where",
            "you left",
            "where you left",
            "go and get",
            "does not exist",
            "there is no ",
            "you have no ",
            "you do not have",
            "found your",
            "no results",
            "last seen in session",
            "seen in session",
        ]
        for objectClass in classes {
            phrases.append(contentsOf: [
                "your \(objectClass)",
                "my \(objectClass)",
                "the \(objectClass) is",
                "\(objectClass) is in",
                "\(objectClass) is at",
                "\(objectClass) is on",
                "\(objectClass) is still",
                "\(objectClass) exists",
            ])
        }
        return phrases
    }

    private func assertNoOverclaim(
        _ strings: [String], classes: [String], file: StaticString = #filePath, line: UInt = #line
    ) {
        let forbidden = forbiddenPhrases(classes: classes)
        for string in strings {
            let lowered = string.lowercased()
            XCTAssertFalse(lowered.isEmpty, "an empty string reached the screen", file: file, line: line)
            for phrase in forbidden {
                XCTAssertFalse(
                    lowered.contains(phrase),
                    "copy claims more than a record supports (\"\(phrase)\") in: \(string)",
                    file: file, line: line
                )
            }
        }
    }

    private let classes = ["laptop", "cell phone"]

    /// Every string shown for an answer that *found* something.
    func testCopyForAFoundRecordClaimsNoPossessionAndNoPlace() throws {
        let strings = ObjectMemoryCopy.everyString(
            for: .listing(try listing()), question: .listing(objectClass: nil)
        )
        XCTAssertFalse(strings.isEmpty)
        assertNoOverclaim(strings, classes: classes)
    }

    /// Every string shown for an answer that found nothing.
    func testCopyForAnEmptyAnswerClaimsNothingAboutTheWorld() throws {
        let empty = try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.emptyListing(objectClass: "laptop")
        )
        let silence = try ObjectMemoryDecoder.lastSeen(
            from: ObjectMemoryFixtures.lastSeen(objectClass: "laptop", observed: false)
        )
        let neverLookedFor = try ObjectMemoryDecoder.lastSeen(
            from: ObjectMemoryFixtures.neverLookedFor()
        )

        assertNoOverclaim(
            ObjectMemoryCopy.everyString(for: .listing(empty)), classes: classes + ["teapot"]
        )
        assertNoOverclaim(
            ObjectMemoryCopy.everyString(for: .lastSeen(silence)), classes: classes + ["teapot"]
        )
        assertNoOverclaim(
            ObjectMemoryCopy.everyString(for: .lastSeen(neverLookedFor)),
            classes: classes + ["teapot"]
        )
    }

    /// Chrome is copy too. "Find my laptop" on a button would walk straight
    /// past a test that only read record rows.
    func testTheChromeMakesNoClaimEither() {
        assertNoOverclaim(ObjectMemoryCopy.everyStaticString, classes: classes)
        assertNoOverclaim(
            [
                ObjectMemoryCopy.lastInViewButton("laptop"),
                ObjectMemoryCopy.lastInViewButton("cell phone"),
                ObjectMemoryCopy.questionLine(.listing(objectClass: "laptop")),
                ObjectMemoryCopy.questionLine(.lastSeen(objectClass: "cell phone")),
            ],
            classes: classes
        )
    }

    /// A category, not an instance: an indefinite article and the past tense,
    /// every time.
    func testASightingHeadlineIsACategoryInThePastTense() throws {
        let listing = try self.listing()
        for observation in listing.observations {
            let headline = ObjectMemoryCopy.sightingHeadline(observation)
            XCTAssertTrue(
                headline.hasPrefix("A ") || headline.hasPrefix("An "),
                "not an indefinite article: \(headline)"
            )
            XCTAssertTrue(headline.hasSuffix(" was visible"), "not past tense: \(headline)")
            XCTAssertFalse(headline.lowercased().contains("your"))
            XCTAssertFalse(headline.lowercased().contains(" is "))
        }
    }

    /// The failure this whole screen is built against is
    /// *"Your laptop: last seen in session 22e9d4…"*. A capture id may never
    /// appear without the words that say what it is.
    func testACaptureIdentifierNeverAppearsWithoutBeingCalledAFrameReference() throws {
        let listing = try self.listing()
        let observation = try XCTUnwrap(listing.observations.first)
        let sessionID = try XCTUnwrap(observation.frame.sessionID)
        let shortened = ObjectMemoryCopy.shortened(sessionID)

        var sawTheIdentifier = false
        for string in ObjectMemoryCopy.everyString(for: observation) {
            guard string.contains(shortened) || string.contains(sessionID) else { continue }
            sawTheIdentifier = true
            XCTAssertTrue(
                string.contains("Frame reference"),
                "a capture id was shown without saying what it is: \(string)"
            )
        }
        XCTAssertTrue(sawTheIdentifier, "the fixture must actually render its capture id")
    }

    /// The two caveats that carry the contract's claims are not optional
    /// decoration: they must be in what a record renders.
    func testEveryRecordCarriesTheClaimAndTheNotAPlaceCaveat() throws {
        let listing = try self.listing()
        for observation in listing.observations {
            let strings = ObjectMemoryCopy.everyString(for: observation)
            XCTAssertTrue(
                strings.contains { $0.contains("not a place") },
                "a record was rendered without saying a frame reference is not a place"
            )
            XCTAssertTrue(
                strings.contains { $0.contains("does not say anything about now") },
                "a record was rendered without the once-visible claim"
            )
        }
    }

    /// An empty answer must say whose silence it is. "No results" would let a
    /// reader supply "so it is not there" for free.
    func testAnEmptyAnswerSaysWhoseSilenceItIs() {
        let explanation = ObjectMemoryCopy.nothingObservedExplanation(
            objectClass: "laptop", recordable: true
        )
        XCTAssertTrue(explanation.contains("not about what exists"))
        XCTAssertTrue(explanation.contains("Absence of a record is not absence of the thing"))

        let neverLookedFor = ObjectMemoryCopy.nothingObservedExplanation(
            objectClass: "teapot", recordable: false
        )
        XCTAssertTrue(neverLookedFor.contains("never been looked for"))
        XCTAssertTrue(neverLookedFor.contains("no information at all"))
        XCTAssertNotEqual(
            explanation, neverLookedFor,
            "the two silences must not share a sentence"
        )
    }

    /// A missing score is said out loud. Never `0%`, which would be a claim of
    /// no evidence rather than of no measurement.
    func testAMissingScoreIsNamedRatherThanZeroed() throws {
        let listing = try self.listing()
        let untracked = try XCTUnwrap(listing.observations.last)
        XCTAssertNil(untracked.bestScore)

        let line = ObjectMemoryCopy.confidenceLine(untracked)
        XCTAssertTrue(line.contains("not tracked"))
        XCTAssertTrue(line.contains("not recorded"))
        XCTAssertFalse(line.contains("0%"))
    }

    /// A record with no capture provenance says so, rather than pointing at
    /// nothing.
    func testARecordWithNoCaptureSaysThereIsNothingToPointAt() throws {
        let listing = try self.listing()
        let unpointed = try XCTUnwrap(listing.observations.last)
        XCTAssertFalse(unpointed.frame.pointsAtACapture)

        let line = ObjectMemoryCopy.frameLine(unpointed.frame)
        XCTAssertTrue(line.contains("Frame reference: none"))
        XCTAssertNil(ObjectMemoryCopy.boxLine(unpointed.frame))
        XCTAssertNil(
            ObjectMemoryCopy.imageryLine(unpointed.frame),
            "there is no imagery to warn about when nothing is pointed at"
        )
    }

    /// The time is the Tower's receipt time and is never presented as a shutter
    /// time — `Rule 16` forbids conflating them.
    func testTheTimestampIsQualifiedRatherThanPresentedAsACaptureTime() throws {
        let listing = try self.listing()
        let observation = try XCTUnwrap(listing.observations.first)
        let line = ObjectMemoryCopy.timeLine(observation)
        XCTAssertTrue(line.contains("receipt time"))
        XCTAssertTrue(line.contains("not the moment the shutter fired"))
    }

    /// A basis this build does not recognise is shown uninterpreted rather than
    /// being read as the one it does know.
    func testAnUnknownTimeBasisIsNotInterpreted() throws {
        var raw = ObjectMemoryFixtures.observation()
        raw["time_basis"] = "on-glasses-capture"
        let observation = try XCTUnwrap(ObjectMemoryDecoder.observation(from: raw))
        let line = ObjectMemoryCopy.timeLine(observation)
        XCTAssertTrue(line.contains("does not recognise"))
        XCTAssertTrue(line.contains("on-glasses-capture"))
        XCTAssertFalse(line.contains("receipt time"))
    }

    /// The window is stated, and a refused widening is stated separately —
    /// never silently.
    func testTheRetentionWindowIsStatedAndARefusalIsVisible() throws {
        let unclamped = ObjectMemoryRetention(
            requestedDays: nil, effectiveDays: 30, clamped: false, policy: "…"
        )
        XCTAssertTrue(ObjectMemoryCopy.retentionLine(unclamped).contains("30 days"))
        XCTAssertNil(
            ObjectMemoryCopy.clampLine(unclamped),
            "a caller that asked for nothing has been refused nothing"
        )

        let clamped = ObjectMemoryRetention(
            requestedDays: 3650, effectiveDays: 30, clamped: true, policy: "…"
        )
        let line = try XCTUnwrap(ObjectMemoryCopy.clampLine(clamped))
        XCTAssertTrue(line.contains("3650 days"))
        XCTAssertTrue(line.contains("refused"))
    }

    func testTheArticleAgreesWithTheCategory() {
        XCTAssertEqual(ObjectMemoryCopy.indefiniteArticle(for: "laptop"), "a")
        XCTAssertEqual(ObjectMemoryCopy.indefiniteArticle(for: "cell phone"), "a")
        XCTAssertEqual(ObjectMemoryCopy.indefiniteArticle(for: "apple"), "an")
        XCTAssertEqual(ObjectMemoryCopy.indefiniteArticle(for: ""), "a")
    }

    /// The drawer row is the first sentence a person reads about this module,
    /// and it used to promise the one thing the cartridge cannot do.
    func testTheCatalogSummaryDoesNotPromiseALocation() throws {
        let cartridge = try XCTUnwrap(Cartridge.catalog.first { $0.id == "object-memory" })
        assertNoOverclaim([cartridge.summary], classes: classes)
        XCTAssertFalse(cartridge.summary.lowercased().contains("where"))
        XCTAssertEqual(cartridge.workspace, .objectMemory)
        // Was `.planned`, asserted as "a screen does not promote a roadmap
        // position" — true while the badge answered a roadmap question. It no
        // longer does. `CartridgeStatus` now answers what a person can do in
        // this build, and Object Memory has two live HTTP routes and a decoder
        // pinned against a real Tower's bytes further down this very file.
        //
        // The live Tower answering 404 (`no object memory root is configured`)
        // does not change this: that is one Tower's configuration, and the
        // workspace renders it truthfully — see the tests above.
        XCTAssertEqual(cartridge.status, .readyToTest)
    }
}

// MARK: - Pinned against a real Tower

/// The three branches, decoded from **bytes taken verbatim off a running
/// Tower** rather than from a reading of its contract.
///
/// The composed fixtures above prove the decoder reads what this build
/// expects. These prove the Tower sends it — and in particular that all three
/// arrive as HTTP 200 with the same envelope, which is the single most likely
/// functional bug here: absence is not a 404, and the two silences are not the
/// same silence.
final class ObjectMemoryRealTowerTests: XCTestCase {

    private func answer(_ text: String) throws -> LastSeenAnswer {
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        )
        return try ObjectMemoryDecoder.lastSeen(from: json)
    }

    /// Recordable, and seen inside the window.
    func testARealSeenAnswerDecodesFieldForField() throws {
        let seen = try answer(Self.seenFromTower)

        XCTAssertEqual(seen.envelope.contract, "object_memory.observations/2026-08-26")
        XCTAssertEqual(seen.envelope.claim, "category-was-visible-once")
        XCTAssertEqual(seen.envelope.identity, "category-not-instance")
        XCTAssertEqual(seen.envelope.absenceMeans, "not-observed-by-this-cartridge")
        XCTAssertEqual(seen.envelope.recordedClasses, ["laptop", "cell phone"])

        XCTAssertTrue(seen.recordable)
        XCTAssertTrue(seen.observed)
        let observation = try XCTUnwrap(seen.observation)

        // "Where" is a pointer back into a recording, never a place. The Tower
        // nulls `spatial_ref` on read, and this build must carry that through
        // rather than inventing a location from the frame reference.
        XCTAssertEqual(observation.frame.camera, "glasses-camera")
        XCTAssertEqual(observation.frame.imageryRetention, "capture-side")
        XCTAssertEqual(observation.frame.frameSeq, 3214)
        XCTAssertEqual(observation.privacyTags, ["derived-only", "frame-referenced"])
    }

    /// Recordable, and **nothing inside the window** — a real statement about
    /// what the camera captured.
    func testARealUnobservedAnswerIsARecordableClassWithNoRecord() throws {
        let narrowed = try answer(Self.narrowedFromTower)

        XCTAssertTrue(narrowed.recordable, "the class is one the cartridge writes")
        XCTAssertFalse(narrowed.observed)
        XCTAssertNil(narrowed.observation, "absent, not an empty observation")
    }

    /// **Never looked for.** Carries no information at all, and must not read
    /// to a person the way the case above does.
    func testARealNeverLookedForAnswerIsTheWeakerSilence() throws {
        let never = try answer(Self.neverFromTower)

        XCTAssertFalse(never.recordable, "person is not in the whitelist")
        XCTAssertFalse(never.observed)
        XCTAssertNil(never.observation)

        // The distinction the whole screen turns on: both say "no record", and
        // only one of them is a statement about the world.
        let narrowed = try answer(Self.narrowedFromTower)
        XCTAssertNotEqual(
            never.recordable, narrowed.recordable,
            "the two silences must stay tellable apart after decoding"
        )
    }

    /// `GET /object-memory/last-seen/laptop`, verbatim.
    private static let seenFromTower = """
        {"contract":"object_memory.observations/2026-08-26","claim":"category-was-visible-once","identity":"category-not-instance","absence_means":"not-observed-by-this-cartridge","spatial_ref":null,"recorded_classes":["laptop","cell phone"],"retention":{"requested_days":null,"effective_days":30.0,"clamped":false,"policy":"min(persisted, requested): a reader may narrow this window and can never widen it"},"object_class":"laptop","recordable":true,"observed":true,"observation":{"object_class":"laptop","claim":"category-was-visible-once","identity":"category-not-instance","confidence":"high","detector_score":0.5121034979820251,"best_score":0.9853731989860535,"observed_at":1787695274.3187459,"time_basis":"tower-receipt","recorded_at":1787730238.1640532,"module_id":"object-memory","retention_tag":"default","privacy_tags":["derived-only","frame-referenced"],"where":{"kind":"frame-reference","spatial_ref":null,"session_id":"22e9d4289cb440fbb3f14e6da369a136","frame_seq":3214,"camera":"glasses-camera","bounding_box_normalized":[0.1120081901550293,0.6517109870910645,0.44075003729926215,0.9019964218139649],"imagery_retention":"capture-side"}},"where":{"kind":"frame-reference","spatial_ref":null,"session_id":"22e9d4289cb440fbb3f14e6da369a136","frame_seq":3214,"camera":"glasses-camera","bounding_box_normalized":[0.1120081901550293,0.6517109870910645,0.44075003729926215,0.9019964218139649],"imagery_retention":"capture-side"}}
        """

    /// `GET /object-memory/last-seen/laptop?retention_days=1e-07`, verbatim.
    /// A window narrowed past every record. `clamped` is false because the
    /// reader narrowed rather than tried to widen.
    private static let narrowedFromTower = """
        {"contract":"object_memory.observations/2026-08-26","claim":"category-was-visible-once","identity":"category-not-instance","absence_means":"not-observed-by-this-cartridge","spatial_ref":null,"recorded_classes":["laptop","cell phone"],"retention":{"requested_days":1e-7,"effective_days":1e-7,"clamped":false,"policy":"min(persisted, requested): a reader may narrow this window and can never widen it"},"object_class":"laptop","recordable":true,"observed":false,"observation":null,"where":null}
        """

    /// `GET /object-memory/last-seen/person`, verbatim. The corpus holds zero
    /// `person` observations and `person` is not a recordable class, so this is
    /// the third branch and it is still HTTP 200.
    private static let neverFromTower = """
        {"contract":"object_memory.observations/2026-08-26","claim":"category-was-visible-once","identity":"category-not-instance","absence_means":"not-observed-by-this-cartridge","spatial_ref":null,"recorded_classes":["laptop","cell phone"],"retention":{"requested_days":null,"effective_days":30.0,"clamped":false,"policy":"min(persisted, requested): a reader may narrow this window and can never widen it"},"object_class":"person","recordable":false,"observed":false,"observation":null,"where":null}
        """
}

// MARK: - Session decoding

/// The generic Start/Pause/Resume/Stop surface, decoded from the shapes the
/// running Tower emits.
///
/// The payload carries two fields that describe the same session and mean
/// different things, and getting them the wrong way round is the one failure
/// this whole surface exists to prevent. Every test here that touches
/// `following` is about that.
final class CartridgeSessionDecodingTests: XCTestCase {

    func testASessionReadingDecodesFieldForField() throws {
        let snapshot = try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(from: ObjectMemoryFixtures.session())
        )

        XCTAssertEqual(snapshot.contract, "cartridge_session.control/2026-08-27")
        XCTAssertEqual(snapshot.cartridge, "object_memory")
        XCTAssertEqual(snapshot.worker, "object-memory-session")
        XCTAssertTrue(snapshot.supported)
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.stateMeans, "intent-not-liveness")
        XCTAssertEqual(snapshot.states, [.stopped, .active, .paused])
        XCTAssertEqual(snapshot.offeredActions, [.start, .pause, .resume, .stop])
        XCTAssertTrue(snapshot.unofferedActions.isEmpty)
        XCTAssertNil(snapshot.sessionID)
        XCTAssertNil(snapshot.startedAt)
        XCTAssertTrue(snapshot.following.isEmpty)

        // A read, not an action. `nil` rather than `false`, because "this was a
        // refresh" and "this action changed nothing" are different facts and a
        // `false` here would report every refresh as a double tap.
        XCTAssertNil(snapshot.accepted)
        XCTAssertNil(snapshot.changed)
        XCTAssertNil(snapshot.wasAnIdempotentNoOp)
    }

    /// The contract identifier is opaque and compared for equality. A different
    /// date is a different agreement, not a newer one.
    func testADifferentSessionContractIsNotDecoded() {
        XCTAssertNil(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(
                    contract: "cartridge_session.control/2027-01-01"
                )
            )
        )
    }

    /// `state_means` is the Tower saying that `state` is not liveness. A
    /// payload that stopped saying it means something this build's entire
    /// rendering assumes, so it is refused rather than decoded on the old
    /// assumption.
    func testAChangedStateMeaningIsRefusedRatherThanIgnored() {
        XCTAssertNil(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(stateMeans: "liveness")
            )
        )
    }

    /// A fourth state must arrive **as itself**. Failing the decode would take
    /// the controls off screen; folding it into `stopped` would tell somebody
    /// nothing is running when the Tower did not say that.
    func testAnUnrecognisedStateSurvivesRatherThanBecomingStopped() throws {
        let snapshot = try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(state: "draining")
            )
        )
        XCTAssertEqual(snapshot.state.rawValue, "draining")
        XCTAssertFalse(snapshot.state.isRecognised)
        XCTAssertNotEqual(snapshot.state, .stopped)
    }

    /// A verb the Tower offers and this build has no button for is reported
    /// rather than dropped: a control surface that silently hides half its
    /// vocabulary looks complete and is not.
    func testAVerbThisBuildCannotSendIsReportedRatherThanHidden() throws {
        let snapshot = try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(
                    actions: ["start", "pause", "resume", "stop", "flush"]
                )
            )
        )
        XCTAssertEqual(snapshot.offeredActions, [.start, .pause, .resume, .stop])
        XCTAssertEqual(snapshot.unofferedActions, ["flush"])
    }

    /// A `POST` adds three fields, and `changed: false` is **not an error**.
    func testAnIdempotentNoOpDecodesAsHonouredAndUnmoved() throws {
        let snapshot = try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.sessionAction(changed: false)
            )
        )
        XCTAssertEqual(snapshot.accepted, true)
        XCTAssertEqual(snapshot.changed, false)
        XCTAssertEqual(snapshot.wasAnIdempotentNoOp, true)
    }

    /// Starting before the camera is running is legal, and the payload says so
    /// with an `active` state and a null capture. A decoder that treated the
    /// null as a failure would refuse the normal case.
    func testStartingBeforeTheCameraIsLegalAndCarriesNoCapture() throws {
        let snapshot = try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(from: ObjectMemoryFixtures.sessionAction())
        )
        XCTAssertEqual(snapshot.state, .active)
        XCTAssertNil(snapshot.attachedCaptureID)
        XCTAssertTrue(snapshot.following.isEmpty)
    }

    /// Both words the two Towers use for the same refusal decode, and both keep
    /// the state that was actually reached.
    func testBothRefusalWordsDecodeAndCarryTheStateActuallyReached() throws {
        for reason in CartridgeSessionRefusalReason.interchangeableStateRefusals {
            let body = try XCTUnwrap(
                ObjectMemoryFixtures.sessionRefusal(reason: reason.rawValue)["detail"]
                    as? [String: Any]
            )
            let refusal = try XCTUnwrap(
                CartridgeSessionDecoder.refusal(from: body, action: .resume)
            )
            XCTAssertEqual(refusal.reason, reason)
            XCTAssertEqual(refusal.action, .resume)
            XCTAssertEqual(refusal.snapshot.state, .stopped)
        }
    }
}

// MARK: - Liveness

/// **The invariant this cartridge is most likely to get wrong, pinned.**
///
/// `state` is intent. `following` is fact. A Pause whose producer ignores
/// `SIGTERM` answers 200 with `state: "paused"` and `changed: true` — a
/// positive claim that the action took effect — while `following` still names
/// the capture and the process is still recording. A Pause button keyed on
/// `state` tells a person they stopped being recorded when they did not.
///
/// These tests exist so that a future refactor which "simplifies" liveness into
/// `state == .active` fails here rather than on somebody's head.
final class ObjectMemoryLivenessTests: XCTestCase {

    private func snapshot(
        state: String, following: [String]
    ) throws -> CartridgeSessionSnapshot {
        try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(state: state, following: following)
            )
        )
    }

    /// **The reproduced failure.** Paused, and still recording.
    func testAPausedSessionStillFollowingACaptureIsStillRecording() throws {
        let paused = try snapshot(state: "paused", following: ["22e9d4289cb440fb"])

        XCTAssertEqual(paused.state, .paused, "intent says the pause was honoured")
        XCTAssertTrue(
            paused.isFollowingACapture,
            """
            liveness comes from `following`, and a producer that ignored SIGTERM \
            is still writing — reading `state` here is what tells a person they \
            stopped being recorded when they did not
            """
        )
        XCTAssertTrue(paused.intentContradictsLiveness)
    }

    /// A Stop that did not stop is the same failure one step further on, and
    /// must be reported the same way rather than trusted because Stop is never
    /// refused.
    func testAStoppedSessionStillFollowingACaptureIsStillRecording() throws {
        let stopped = try snapshot(state: "stopped", following: ["22e9d4289cb440fb"])
        XCTAssertTrue(stopped.isFollowingACapture)
        XCTAssertTrue(stopped.intentContradictsLiveness)
    }

    /// **Active with nothing followed is legal, and must not be reported as a
    /// contradiction.** Start before the camera is running looks exactly like
    /// this, and so does a producer that died — one payload cannot tell them
    /// apart, so this app says what it knows instead of guessing which.
    func testAnActiveSessionFollowingNothingIsNotClaimedAsAFailure() throws {
        let active = try snapshot(state: "active", following: [])
        XCTAssertFalse(active.isFollowingACapture, "nothing is being written into")
        XCTAssertFalse(
            active.intentContradictsLiveness,
            "starting before the camera is legal and is not a producer that died"
        )
    }

    /// An unrecognised state makes no contradiction claim either way, and
    /// liveness still reads correctly off `following`.
    func testAnUnrecognisedStateStillYieldsLivenessFromFollowing() throws {
        let odd = try snapshot(state: "draining", following: ["22e9d4289cb440fb"])
        XCTAssertTrue(odd.isFollowingACapture)
        XCTAssertFalse(
            odd.intentContradictsLiveness,
            "this app cannot say a state it does not understand contradicts anything"
        )
    }

    /// `captures` is a history and `following` is not. A recording the producer
    /// has finished with must never light the live indicator.
    func testAFinishedCaptureInTheHistoryIsNotLiveness() throws {
        let snapshot = try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(
                    state: "active", following: [], captures: ["22e9d4289cb440fb"]
                )
            )
        )
        XCTAssertFalse(snapshot.isFollowingACapture)
        XCTAssertEqual(snapshot.captures.count, 1)
    }

    /// The copy obeys the same rule as the model, because the model being right
    /// buys nothing if the sentence is written from the other field.
    func testTheLivenessSentenceIsWrittenFromFollowingNotFromState() throws {
        let pausedAndRecording = try snapshot(
            state: "paused", following: ["22e9d4289cb440fb"]
        )
        let activeAndIdle = try snapshot(state: "active", following: [])

        XCTAssertNotEqual(
            ObjectMemoryCopy.livenessLine(pausedAndRecording),
            ObjectMemoryCopy.livenessLine(activeAndIdle),
            "two sessions with opposite `following` must not read the same"
        )
        // And the pairing is the *opposite* of what `state` would give: the
        // paused one is the one being written into.
        XCTAssertEqual(
            ObjectMemoryCopy.livenessLine(pausedAndRecording),
            ObjectMemoryCopy.livenessLine(
                try snapshot(state: "active", following: ["22e9d4289cb440fb"])
            ),
            "the sentence must depend on `following` alone, not on `state`"
        )
        XCTAssertNotNil(
            ObjectMemoryCopy.livenessContradictsIntentLine(pausedAndRecording),
            "the harmful contradiction has to be said out loud"
        )
        XCTAssertNil(ObjectMemoryCopy.livenessContradictsIntentLine(activeAndIdle))
    }

    /// The view model's own liveness accessor, which is what the screen binds
    /// to. `nil` before anything is read — not `false`, which would be a claim.
    @MainActor
    func testTheWorkspaceHasNoLivenessClaimBeforeItHasRead() async throws {
        let client = StubObjectMemoryClient()
        let memory = ObjectMemoryViewModel(client: client)
        XCTAssertNil(memory.liveness, "\"we have not asked\" is not \"nothing is recording\"")

        client.publish(
            ObjectMemorySessionState.known(
                try snapshot(state: "paused", following: ["22e9d4289cb440fb"])
            )
        )
        try await Task.sleep(nanoseconds: 30_000_000)
        XCTAssertEqual(memory.liveness, true)
        XCTAssertTrue(memory.intentContradictsLiveness)
    }
}

// MARK: - Session transport

@MainActor
final class CartridgeSessionTransportTests: XCTestCase {

    private let baseURL = URL(string: "http://tower.test")!

    private func makeClient() -> CartridgeSessionHTTPClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ObjectMemoryStubProtocol.self]
        return CartridgeSessionHTTPClient(
            baseURL: baseURL, session: URLSession(configuration: configuration)
        )
    }

    private func respond(_ status: Int, _ json: Any) {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, try JSONSerialization.data(withJSONObject: json))
        }
    }

    override func setUp() {
        super.setUp()
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
    }

    override func tearDown() {
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
        super.tearDown()
    }

    func testTheSessionIsReadFromTheCartridgeKeyedPath() async throws {
        respond(200, ObjectMemoryFixtures.session())
        _ = try await makeClient().read()

        let url = try XCTUnwrap(ObjectMemoryStubProtocol.requestedURLs.first)
        XCTAssertEqual(url.path, "/cartridges/object_memory/session")
        XCTAssertEqual(ObjectMemoryStubProtocol.requestedRequests.first?.httpMethod, "GET")
    }

    func testEachVerbPostsToItsOwnPath() async throws {
        for action in CartridgeSessionAction.allCases {
            ObjectMemoryStubProtocol.requestedURLs = []
            ObjectMemoryStubProtocol.requestedRequests = []
            respond(200, ObjectMemoryFixtures.sessionAction())
            _ = try await makeClient().apply(action)

            let url = try XCTUnwrap(ObjectMemoryStubProtocol.requestedURLs.first)
            XCTAssertEqual(url.path, "/cartridges/object_memory/session/\(action.rawValue)")
            XCTAssertEqual(ObjectMemoryStubProtocol.requestedRequests.first?.httpMethod, "POST")
        }
    }

    /// **A second start is 200 with `changed: false`, and that is not an
    /// error.** It reaches the caller as `.honoured`, not as a refusal, and
    /// nothing downstream may draw it under an error glyph.
    func testASecondStartIsHonouredAndNotAnError() async throws {
        respond(200, ObjectMemoryFixtures.sessionAction(changed: false))
        let outcome = try await makeClient().apply(.start)

        guard case .honoured(let snapshot) = outcome else {
            return XCTFail("an idempotent start decoded as a refusal: \(outcome)")
        }
        XCTAssertEqual(snapshot.wasAnIdempotentNoOp, true)
        XCTAssertEqual(snapshot.state, .active)
    }

    /// **Stop is never refused from any state**, and a Stop from `stopped`
    /// arrives the same way a second start does.
    func testStopFromStoppedIsHonouredAndUnmoved() async throws {
        respond(
            200,
            ObjectMemoryFixtures.sessionAction(
                state: "stopped", changed: false, sessionID: NSNull(), startedAt: NSNull()
            )
        )
        let outcome = try await makeClient().apply(.stop)
        guard case .honoured(let snapshot) = outcome else {
            return XCTFail("stop was refused, which this surface never does: \(outcome)")
        }
        XCTAssertEqual(snapshot.wasAnIdempotentNoOp, true)
    }

    /// **A 409 is an outcome, not a throw.** "Resume continues a paused
    /// session; this one is stopped" is a true and actionable sentence, and
    /// pushing it through the same channel as an unreachable Tower would put it
    /// under the wrong glyph.
    ///
    /// Runs over **both** words, because the contract document says
    /// `not-active` and the running Tower says `not-paused`.
    func testAResumeFromStoppedIsARefusalUnderEitherWord() async throws {
        for reason in CartridgeSessionRefusalReason.interchangeableStateRefusals {
            respond(409, ObjectMemoryFixtures.sessionRefusal(reason: reason.rawValue))
            let outcome = try await makeClient().apply(.resume)

            guard case .refused(let refusal) = outcome else {
                return XCTFail("a 409 did not arrive as a refusal: \(outcome)")
            }
            XCTAssertEqual(refusal.reason, reason)
            XCTAssertEqual(refusal.snapshot.state, .stopped)
            // The sentence is written from the action and the reached state, so
            // it must not vary with which of the two words arrived.
            XCTAssertEqual(
                ObjectMemoryCopy.refusalLine(refusal),
                ObjectMemoryCopy.refusalLine(
                    CartridgeSessionRefusal(
                        action: .resume,
                        reason: .notActive,
                        message: refusal.message,
                        snapshot: refusal.snapshot
                    )
                ),
                "the wire and the document disagree about this word; the copy must not"
            )
        }
    }

    /// A 404 is a **configuration** answer — this Tower has no controllable
    /// session — and is never the answer to "may I start", which is a 409.
    func testAFourOhFourIsAConfigurationAnswerAndNotARefusal() async {
        respond(404, ["detail": "this Tower has no controllable session for 'object_memory'"])
        do {
            _ = try await makeClient().read()
            XCTFail("a 404 decoded as a reading")
        } catch let error as CartridgeSessionFetchError {
            XCTAssertEqual(error, .noSuchCartridgeSession)
        } catch {
            XCTFail("expected a CartridgeSessionFetchError, got \(error)")
        }
    }

    /// A status this surface does not define must not become a transport
    /// failure: the Tower was reached, and telling somebody to check a network
    /// about a machine that answered is the failure this build removed.
    func testAnUndefinedStatusIsNotReportedAsAnUnreachableTower() async {
        respond(500, ["detail": "boom"])
        do {
            _ = try await makeClient().read()
            XCTFail("a 500 decoded as a reading")
        } catch let error as CartridgeSessionFetchError {
            XCTAssertEqual(error, .undecodable)
            if case .transport = error {
                XCTFail("a Tower that answered was reported as unreachable")
            }
        } catch {
            XCTFail("expected a CartridgeSessionFetchError, got \(error)")
        }
    }

    /// Nothing here may be answered out of a URL cache: a session state read
    /// from a stale copy is a claim about a running producer.
    func testSessionRequestsRefuseCachedAnswers() async throws {
        respond(200, ObjectMemoryFixtures.session())
        _ = try await makeClient().read()
        let request = try XCTUnwrap(ObjectMemoryStubProtocol.requestedRequests.first)
        XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalAndRemoteCacheData)
        XCTAssertEqual(request.timeoutInterval, 10)
    }
}

// MARK: - The client's session state

@MainActor
final class ObjectMemorySessionClientTests: XCTestCase {

    private func makeClient() -> TowerObjectMemoryClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ObjectMemoryStubProtocol.self]
        let session = URLSession(configuration: configuration)
        return TowerObjectMemoryClient(
            http: ObjectMemoryHTTPClient(
                baseURL: URL(string: "http://tower.test")!, session: session
            ),
            control: CartridgeSessionHTTPClient(
                baseURL: URL(string: "http://tower.test")!, session: session
            ),
            pictures: ObjectMemoryImageryHTTPClient(
                baseURL: URL(string: "http://tower.test")!, session: session
            )
        )
    }

    private func respond(_ status: Int, _ json: Any) {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, try JSONSerialization.data(withJSONObject: json))
        }
    }

    override func setUp() {
        super.setUp()
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
    }

    override func tearDown() {
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
        super.tearDown()
    }

    /// Nothing is claimed about a session until one has been read. **Not
    /// `stopped`** — a Tower that has not been asked has not said it is
    /// stopped, and a Stopped badge drawn from silence is the same error as a
    /// recording badge drawn from `state`.
    func testAFreshClientClaimsNothingAboutTheSession() {
        let client = makeClient()
        XCTAssertEqual(client.session, .unread)
        XCTAssertNil(client.session.isFollowingACapture)
    }

    /// **An action is followed by a read**, because a `POST` answers with
    /// intent and only a read reports `following`. A Stop that did not stop is
    /// exactly the case that a trusted `POST` body would hide.
    func testAnActionIsFollowedByAReadRatherThanTrusted() async throws {
        var responses: [(Int, Any)] = [
            // The POST: paused, honoured, changed.
            (200, ObjectMemoryFixtures.sessionAction(state: "paused")),
            // The read-back: still following. The producer ignored SIGTERM.
            (
                200,
                ObjectMemoryFixtures.session(
                    state: "paused",
                    sessionID: "61a78a9b32284cd2a89583d9a8cc8702",
                    following: ["22e9d4289cb440fb"]
                )
            ),
        ]
        ObjectMemoryStubProtocol.handler = { request in
            let (status, json) = responses.isEmpty ? (200, [:] as Any) : responses.removeFirst()
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, try JSONSerialization.data(withJSONObject: json))
        }

        let client = makeClient()
        await client.apply(.pause)

        XCTAssertEqual(
            ObjectMemoryStubProtocol.requestedRequests.count, 2,
            "the POST alone is intent; the read-back is what reports liveness"
        )
        XCTAssertEqual(
            client.session.isFollowingACapture, true,
            "the read-back must win over the POST's claim that the pause took effect"
        )
        XCTAssertTrue(client.session.snapshot?.intentContradictsLiveness == true)
    }

    /// A 409 is a state, not a failure, and its explanation **survives the
    /// read-back** — otherwise the sentence that says which control would have
    /// worked flashes and vanishes behind the refresh.
    func testARefusalSurvivesTheReadBackThatFollowsIt() async {
        // What the real Tower does: 409 to the POST, 200 to the GET after it.
        var responses: [(Int, Any)] = [
            (409, ObjectMemoryFixtures.sessionRefusal(reason: "not-paused")),
            (200, ObjectMemoryFixtures.session(state: "stopped")),
        ]
        ObjectMemoryStubProtocol.handler = { request in
            let (status, json) = responses.isEmpty ? (200, [:] as Any) : responses.removeFirst()
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, try JSONSerialization.data(withJSONObject: json))
        }

        let client = makeClient()
        await client.apply(.resume)

        guard case .refused(let refusal) = client.session else {
            return XCTFail("a refusal became something else: \(client.session)")
        }
        XCTAssertEqual(refusal.action, .resume)
        XCTAssertEqual(refusal.reason, .notPaused)
        XCTAssertEqual(client.session.snapshot?.state, .stopped)
        XCTAssertEqual(client.session.isFollowingACapture, false)
    }

    /// **A 409 on a read is not an unreachable Tower.**
    ///
    /// Nothing was asked for on a `GET`, so nothing can have been refused — but
    /// the refusal that a `POST` legitimately produces travels as its own error
    /// type, and letting it escape a read would wrap it as `.transport` and
    /// report a Tower that answered as one that could not be reached. That is
    /// the same defect class this change removed from the imagery path, and it
    /// was found here by a test rather than by a wearer.
    func testARefusalOnAReadIsNotReportedAsAnUnreachableTower() async {
        respond(409, ObjectMemoryFixtures.sessionRefusal(reason: "not-paused"))
        let client = makeClient()
        await client.readSession()

        guard case .failed(let failure) = client.session else {
            return XCTFail("a 409 on a read became something else: \(client.session)")
        }
        XCTAssertEqual(failure.kind, .undecodableResponse)
        XCTAssertNotEqual(
            failure.kind, .transport,
            "the Tower answered; calling that unreachable is a false claim about it"
        )
    }

    /// A Tower with no controllable session is a configuration state, not an
    /// error: it can still be asked what it already recorded.
    func testNoControllableSessionIsItsOwnState() async {
        respond(404, ["detail": "this Tower has no controllable session for 'object_memory'"])
        let client = makeClient()
        await client.readSession()
        XCTAssertEqual(client.session, .noSessionControl)
    }
}

// MARK: - Imagery

/// The three imagery routes, and the four things that can come back.
///
/// **The 410 is the one to watch.** It means the pointer is intact and the
/// picture is gone, `memory_retained: true` is in the body, and it must never
/// render as a broken image, an empty row, or a connection error. Before this
/// build it rendered as the third of those.
final class ObjectMemoryImageryDecodingTests: XCTestCase {

    private let baseURL = URL(string: "http://tower.test")!

    private func routes() throws -> ObjectMemoryImageryRoutes {
        try XCTUnwrap(
            ObjectMemoryImageryDecoder.routes(from: ObjectMemoryFixtures.imageryRoutes())
        )
    }

    /// The envelope's `imagery` block was **dropped on decode** until this
    /// build, which is why a wearer saw a capture id and never a frame.
    func testTheEnvelopeCarriesTheImageryRoutes() throws {
        let listing = try ObjectMemoryDecoder.listing(from: ObjectMemoryFixtures.listing())
        let imagery = try XCTUnwrap(
            listing.envelope.imagery, "the imagery block must survive the decode"
        )
        XCTAssertEqual(imagery.contract, "object_memory.imagery/2026-08-27")
        XCTAssertEqual(
            imagery.frameTemplate, "/object-memory/observations/{observation_id}/frame"
        )
    }

    /// A Tower that predates the imagery routes sends no block. That is a
    /// missing feature, not a broken payload, and taking the whole listing off
    /// screen over it would show a wearer nothing to protect them from a
    /// missing thumbnail.
    func testAnEnvelopeWithoutAnImageryBlockStillDecodes() throws {
        var payload = ObjectMemoryFixtures.listing()
        payload.removeValue(forKey: "imagery")
        let listing = try ObjectMemoryDecoder.listing(from: payload)
        XCTAssertNil(listing.envelope.imagery)
        XCTAssertEqual(listing.observations.count, 1)
    }

    /// A changed `filter_means` would mean the filter no longer runs on read
    /// and the stored frame is no longer unchanged — which is what every
    /// sentence this app writes about the filter is licensed by.
    func testAChangedFilterMeaningIsRefused() {
        XCTAssertNil(
            ObjectMemoryImageryDecoder.routes(
                from: ObjectMemoryFixtures.imageryRoutes(filterMeans: "applied-before-storage")
            )
        )
    }

    /// A template that lost its placeholder would resolve to one shared URL,
    /// and every row would show the same picture — a photograph attributed to
    /// the wrong record. Refusing is the only safe direction.
    func testATemplateWithoutItsPlaceholderIsRefusedRatherThanShared() throws {
        let broken = try XCTUnwrap(
            ObjectMemoryImageryDecoder.routes(
                from: ObjectMemoryFixtures.imageryRoutes(
                    frame: "/object-memory/observations/frame"
                )
            )
        )
        XCTAssertNil(broken.path(for: .frame, observationID: "9f2c41b7ad0e6538"))
        XCTAssertNotNil(broken.path(for: .crop, observationID: "9f2c41b7ad0e6538"))
    }

    func testAHandleIsSubstitutedIntoTheTemplate() throws {
        let url = try XCTUnwrap(
            routes().url(
                for: .crop, observationID: "9f2c41b7ad0e6538", relativeTo: baseURL
            )
        )
        XCTAssertEqual(
            url.absoluteString,
            "http://tower.test/object-memory/observations/9f2c41b7ad0e6538/crop"
        )
    }

    /// A served picture.
    func testAServedPictureDescribesItselfAndTheFilterThatRan() throws {
        let description = try XCTUnwrap(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(), statusCode: 200
            )
        )
        XCTAssertEqual(description.situation, .aPicture)
        XCTAssertTrue(description.available)
        XCTAssertNil(description.reason)
        XCTAssertEqual(description.filter, "display-filter/yunet-2023mar@0.30")
        XCTAssertEqual(
            description.filterMeans, "applied-on-read-the-stored-frame-is-unchanged"
        )
        XCTAssertFalse(description.subjectIsBehindAFill)
        XCTAssertEqual(description.preferredKind, .crop)
    }

    /// **The 410**, decoded out of FastAPI's `detail` wrapper.
    func testTheMemoryIsKeptAndThePictureIsGone() throws {
        let description = try XCTUnwrap(
            ObjectMemoryImageryDecoder.descriptionUnwrapping(
                ObjectMemoryFixtures.imageryGone(), statusCode: 410
            )
        )
        XCTAssertEqual(description.situation, .thePictureIsGone)
        XCTAssertFalse(description.available)
        XCTAssertTrue(
            description.memoryRetained,
            "the record is still here; only the picture is gone"
        )
        XCTAssertEqual(description.reason, .imageryNoLongerAvailable)
        XCTAssertEqual(description.statusCode, 410)
    }

    /// The three 503 reasons are a claim about the **Tower**, not about the
    /// record, and are grouped so the copy cannot tell somebody their memory
    /// lost a picture when in fact no weights are installed.
    func testTheFiveOhThreeFamilyIsAboutTheTowerAndNotTheRecord() throws {
        for reason in ObjectMemoryImageryReason.towerCannotServeAny {
            let description = try XCTUnwrap(
                ObjectMemoryImageryDecoder.descriptionUnwrapping(
                    ObjectMemoryFixtures.imageryRefused(reason: reason.rawValue),
                    statusCode: 503
                )
            )
            XCTAssertEqual(description.situation, .theTowerServesNoPictures(reason))
            XCTAssertTrue(
                description.memoryRetained, "a Tower that cannot serve has not lost a record"
            )
        }
    }

    /// The two 404 reasons stay apart: "nothing under this handle" and "this
    /// record never had a pointer" are different facts.
    func testTheTwoFourOhFourReasonsStayApart() throws {
        let unknown = try XCTUnwrap(
            ObjectMemoryImageryDecoder.descriptionUnwrapping(
                ["detail": ObjectMemoryFixtures.imagery(
                    objectClass: NSNull(),
                    available: false,
                    reason: "no-such-observation",
                    memoryRetained: false,
                    filter: NSNull()
                )],
                statusCode: 404
            )
        )
        XCTAssertEqual(unknown.situation, .noSuchRecord)
        XCTAssertFalse(unknown.memoryRetained)

        let noPointer = try XCTUnwrap(
            ObjectMemoryImageryDecoder.descriptionUnwrapping(
                ["detail": ObjectMemoryFixtures.imagery(
                    available: false,
                    reason: "record-has-no-frame-reference",
                    memoryRetained: true,
                    filter: NSNull()
                )],
                statusCode: 404
            )
        )
        XCTAssertEqual(noPointer.situation, .theRecordNeverPointedAtAFrame)
        XCTAssertTrue(noPointer.memoryRetained)
    }

    /// A reason this build does not implement arrives as itself rather than
    /// failing the parse — which is what would send it back down the
    /// connection-failure path this build removed.
    func testAnUnrecognisedReasonSurvivesRatherThanFailingTheParse() throws {
        let description = try XCTUnwrap(
            ObjectMemoryImageryDecoder.descriptionUnwrapping(
                ObjectMemoryFixtures.imageryRefused(reason: "weather"), statusCode: 503
            )
        )
        guard case .anUnrecognisedRefusal(let reason) = description.situation else {
            return XCTFail("an unknown reason was folded into a known one: \(description.situation)")
        }
        XCTAssertEqual(reason.rawValue, "weather")
    }

    /// **`regions_filled: 0` means nothing was detected**, not that there was
    /// nothing there, and the sentence has to carry that on its own.
    func testZeroFilledRegionsIsNotAnAllClear() throws {
        let description = try XCTUnwrap(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(regionsFilled: 0), statusCode: 200
            )
        )
        let line = ObjectMemoryCopy.regionsFilledLine(description).lowercased()
        XCTAssertTrue(
            line.contains("detected nothing"),
            "zero must be said as \"detected nothing\": \(line)"
        )
        XCTAssertFalse(line.contains("no faces were present"))
    }

    /// **`subject_obscured > 0` falls back to the whole frame**, because the
    /// crop is the picture the fill is sitting on. The filter is not weakened
    /// to get a nicer thumbnail — a face-detection threshold is not a
    /// picture-quality knob.
    func testAnObscuredSubjectFallsBackToTheWholeFrame() throws {
        let description = try XCTUnwrap(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(regionsFilled: 2, subjectObscured: 0.1266),
                statusCode: 200
            )
        )
        XCTAssertTrue(description.subjectIsBehindAFill)
        XCTAssertEqual(description.preferredKind, .frame)
        XCTAssertNotNil(
            ObjectMemoryCopy.subjectObscuredLine(description, kind: .frame),
            "the fill has to be said as well as worked around"
        )
        XCTAssertNil(
            ObjectMemoryCopy.subjectObscuredLine(
                try XCTUnwrap(
                    ObjectMemoryImageryDecoder.description(
                        from: ObjectMemoryFixtures.imagery(), statusCode: 200
                    )
                ),
                kind: .crop
            ),
            "nothing overlapping means nothing to say"
        )
    }

    /// A malformed box is refused rather than half-drawn over a photograph.
    func testAMalformedBoxIsRefused() {
        XCTAssertNil(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(box: [0.1, 0.2]), statusCode: 200
            )
        )
    }
}

// MARK: - Imagery transport

@MainActor
final class ObjectMemoryImageryTransportTests: XCTestCase {

    private let baseURL = URL(string: "http://tower.test")!

    private func makeClient() -> ObjectMemoryImageryHTTPClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ObjectMemoryStubProtocol.self]
        return ObjectMemoryImageryHTTPClient(
            baseURL: baseURL, session: URLSession(configuration: configuration)
        )
    }

    private func routes() throws -> ObjectMemoryImageryRoutes {
        try XCTUnwrap(
            ObjectMemoryImageryDecoder.routes(from: ObjectMemoryFixtures.imageryRoutes())
        )
    }

    private func respond(_ status: Int, json: Any) {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil
            )!
            return (response, try JSONSerialization.data(withJSONObject: json))
        }
    }

    private func respond(_ status: Int, bytes: Data) {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: status,
                httpVersion: nil,
                // The header the Tower actually sends, so a test that ever
                // starts asserting on it has it to assert on.
                headerFields: ["Cache-Control": "no-store", "Content-Type": "image/jpeg"]
            )!
            return (response, bytes)
        }
    }

    override func setUp() {
        super.setUp()
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
    }

    override func tearDown() {
        ObjectMemoryStubProtocol.handler = nil
        ObjectMemoryStubProtocol.requestedURLs = []
        ObjectMemoryStubProtocol.requestedRequests = []
        super.tearDown()
    }

    func testAServedPictureComesBackAsBytes() async throws {
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])
        respond(200, bytes: jpeg)
        let answer = await makeClient().picture(
            for: "9f2c41b7ad0e6538", kind: .crop, routes: try routes()
        )
        XCTAssertEqual(answer, .picture(jpeg))
    }

    /// **The defect this file exists to prove is gone.**
    ///
    /// A 410 used to become `.transport("The Tower answered 410.")`, which the
    /// workspace maps to `.disconnected` — so a wearer was told their Tower was
    /// unreachable when it had answered clearly that capture-side retention had
    /// taken the picture and kept the memory. Worse than a broken image,
    /// because it sends someone to check a network cable.
    func testAGoneImageIsARenderedSentenceAndNotAConnectionFailure() async throws {
        respond(410, json: ObjectMemoryFixtures.imageryGone())
        let answer = await makeClient().picture(
            for: "9f2c41b7ad0e6538", kind: .crop, routes: try routes()
        )

        guard case .refused(let description) = answer else {
            return XCTFail("a 410 did not arrive as a described refusal: \(answer)")
        }
        if case .unreachable = answer { XCTFail("a 410 was reported as an unreachable Tower") }
        XCTAssertEqual(description.situation, .thePictureIsGone)
        XCTAssertTrue(description.memoryRetained)

        // And the sentence a person reads says both halves.
        let headline = ObjectMemoryCopy.noPictureHeadline(description).lowercased()
        XCTAssertTrue(headline.contains("memory is kept"), headline)
        XCTAssertTrue(headline.contains("picture is gone"), headline)
    }

    /// A 503 is a refusal to serve, described. **This app does not work around
    /// it**, because the lenient default here is a raw first-person frame.
    func testAFilterlessTowerRefusesAndThisAppDoesNotRetry() async throws {
        respond(503, json: ObjectMemoryFixtures.imageryRefused(reason: "display-filter-unavailable"))
        let answer = await makeClient().picture(
            for: "9f2c41b7ad0e6538", kind: .frame, routes: try routes()
        )

        guard case .refused(let description) = answer else {
            return XCTFail("a 503 did not arrive as a described refusal: \(answer)")
        }
        XCTAssertEqual(
            description.situation, .theTowerServesNoPictures(.displayFilterUnavailable)
        )
        XCTAssertEqual(
            ObjectMemoryStubProtocol.requestedRequests.count, 1,
            "a 503 must not be retried into a second request for the same bytes"
        )
    }

    /// `/imagery` answers **200 even when there is no picture**; only an
    /// unknown handle is a real 404 there.
    func testTheDescriptionRouteAnswersEvenWhenThereIsNoPicture() async throws {
        respond(
            200,
            json: ObjectMemoryFixtures.imagery(
                available: false,
                reason: "record-has-no-frame-reference",
                memoryRetained: true,
                filter: NSNull()
            )
        )
        let answer = await makeClient().description(
            for: "9f2c41b7ad0e6538", routes: try routes()
        )
        guard case .described(let description) = answer else {
            return XCTFail("a 200 with no picture did not arrive as a description: \(answer)")
        }
        XCTAssertFalse(description.available)
        XCTAssertEqual(description.statusCode, 200)
    }

    /// **The bytes are never answered out of a cache.** Both binary routes send
    /// `Cache-Control: no-store`, and a copy anybody holds is a second store
    /// nobody chose and nobody's retention governs.
    func testPictureRequestsRefuseEveryCache() async throws {
        respond(200, bytes: Data([0xFF, 0xD8]))
        _ = await makeClient().picture(
            for: "9f2c41b7ad0e6538", kind: .crop, routes: try routes()
        )
        let request = try XCTUnwrap(ObjectMemoryStubProtocol.requestedRequests.first)
        XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalAndRemoteCacheData)
    }

    /// And when no session is injected, the one this client builds for itself
    /// has no URL cache at all — so honouring `no-store` does not depend on
    /// which session a caller happened to pass in.
    func testTheDefaultImagerySessionKeepsNoCache() {
        let session = ObjectMemoryImageryHTTPClient.uncachedSession()
        XCTAssertNil(session.configuration.urlCache)
        XCTAssertEqual(
            session.configuration.requestCachePolicy, .reloadIgnoringLocalAndRemoteCacheData
        )
    }

    /// A 200 with an empty body is not a picture. Handing `UIImage` bytes it
    /// will silently fail on leaves a blank rectangle, which on this screen
    /// reads as "there was nothing there".
    func testEmptyBytesAreRefusedRatherThanDrawnAsABlank() async throws {
        respond(200, bytes: Data())
        let answer = await makeClient().picture(
            for: "9f2c41b7ad0e6538", kind: .crop, routes: try routes()
        )
        XCTAssertEqual(answer, .undecodable)
    }

    /// The `view` route is JSON and has no bytes to hand back; asking it for a
    /// picture is a programming error and is refused rather than fetched.
    func testTheDescriptionRouteIsNeverFetchedAsAPicture() async throws {
        let answer = await makeClient().picture(
            for: "9f2c41b7ad0e6538", kind: .view, routes: try routes()
        )
        XCTAssertEqual(answer, .routesUnknown)
        XCTAssertTrue(ObjectMemoryStubProtocol.requestedRequests.isEmpty)
    }
}

// MARK: - The picture loader

@MainActor
final class ObjectMemoryPictureLoaderTests: XCTestCase {

    private func description(
        available: Bool = true, reason: Any = NSNull(), subjectObscured: Any = 0.0
    ) throws -> ObjectMemoryImageryDescription {
        try XCTUnwrap(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(
                    available: available, reason: reason, subjectObscured: subjectObscured
                ),
                statusCode: available ? 200 : 410
            )
        )
    }

    private func settle() async throws {
        try await Task.sleep(nanoseconds: 50_000_000)
    }

    /// The description is fetched **before** the bytes, which is what lets a
    /// 410 be a sentence rather than a failed image load.
    func testTheDescriptionIsFetchedBeforeAnyBytes() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(try description())
        client.stubbedPicture = .picture(Data([0xFF, 0xD8]))

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()

        XCTAssertEqual(client.imageryAsked, ["9f2c41b7ad0e6538"])
        XCTAssertEqual(client.picturesAsked.count, 1)
        guard case .picture = loader.phase else {
            return XCTFail("a served picture did not arrive: \(loader.phase)")
        }
    }

    /// **A 410 becomes a rendered sentence, never a failure.**
    func testAGoneImageDoesNotReachTheFailedPhase() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(
            try description(available: false, reason: "imagery-no-longer-available")
        )

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()

        guard case .noPicture(let description) = loader.phase else {
            return XCTFail("a 410 reached the wrong phase: \(loader.phase)")
        }
        XCTAssertEqual(description.situation, .thePictureIsGone)
        XCTAssertTrue(
            client.picturesAsked.isEmpty,
            "there is no reason to ask for bytes the Tower already said are gone"
        )
    }

    /// The 503 family too: nothing is fetched, and nothing renders as an error.
    func testAFilterlessTowerNeverReachesTheBytes() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(
            try description(available: false, reason: "display-filter-unavailable")
        )

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()

        guard case .noPicture = loader.phase else {
            return XCTFail("a 503 reached the wrong phase: \(loader.phase)")
        }
        XCTAssertTrue(client.picturesAsked.isEmpty)
    }

    /// An obscured subject fetches the **whole frame** rather than the crop the
    /// fill is sitting on.
    func testAnObscuredSubjectFetchesTheFrameInsteadOfTheCrop() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(try description(subjectObscured: 0.42))
        client.stubbedPicture = .picture(Data([0xFF, 0xD8]))

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()

        XCTAssertEqual(client.picturesAsked.first?.1, .frame)
        XCTAssertEqual(loader.kind, .frame)
    }

    /// The race: the description said available and the bytes came back gone,
    /// because capture-side retention closed between the two requests. It ends
    /// in the same sentence rather than a broken image.
    func testARetentionRaceEndsInTheSameSentence() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(try description())
        client.stubbedPicture = .refused(
            try description(available: false, reason: "imagery-no-longer-available")
        )

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()

        guard case .noPicture(let description) = loader.phase else {
            return XCTFail("a mid-flight expiry reached the wrong phase: \(loader.phase)")
        }
        XCTAssertEqual(description.situation, .thePictureIsGone)
    }

    /// **Nothing is kept.** `forget()` drops the bytes rather than waiting for
    /// deallocation, because a row leaving the screen is exactly when this app
    /// should stop holding a photograph of somebody's home.
    func testForgettingDropsTheBytes() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(try description())
        client.stubbedPicture = .picture(Data([0xFF, 0xD8]))

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()
        guard case .picture = loader.phase else { return XCTFail("no picture to forget") }

        loader.forget()
        XCTAssertEqual(loader.phase, .unasked, "the bytes must not survive the row")
    }

    /// A Tower with no imagery block offers no pictures, and that is a fact
    /// about the Tower rather than a failure.
    func testATowerThatOffersNoPicturesSaysSoRatherThanFailing() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .routesUnknown

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()
        XCTAssertEqual(loader.phase, .noPicturesOffered)
    }
}

// MARK: - Copy for the session and the pictures

/// The same forbidden-phrase discipline `ObjectMemoryCopyTests` applies to
/// records, applied to the two surfaces added on 2026-08-27.
///
/// The picture half matters most. **A picture is a much stronger location cue
/// than a sentence, and no string test can catch it** — every rule the record
/// copy enforces can be undone by a photograph of a desk with a weak line under
/// it. What a test *can* do is hold the line that the caption exists, says what
/// the picture is and is not, and never uses the words this cartridge is
/// forbidden to use about its filter.
@MainActor
final class ObjectMemorySessionAndPictureCopyTests: XCTestCase {

    private let classes = ["laptop", "cell phone"]

    /// The same list `ObjectMemoryCopyTests` uses, kept in one place by being
    /// re-derived rather than by being shared: these two suites test different
    /// surfaces and a change to one list should be a deliberate change to both.
    private func forbiddenPhrases() -> [String] {
        var phrases = [
            "still there", "is still", "right now", "at the moment", "on the map",
            "last known location", "location of", "we know where", "you left",
            "where you left", "go and get", "does not exist", "there is no ",
            "you have no ", "you do not have", "found your", "no results",
            "last seen in session", "seen in session",
        ]
        for objectClass in classes {
            phrases.append(contentsOf: [
                "your \(objectClass)", "my \(objectClass)", "the \(objectClass) is",
                "\(objectClass) is in", "\(objectClass) is at", "\(objectClass) is on",
                "\(objectClass) is still", "\(objectClass) exists",
            ])
        }
        return phrases
    }

    /// **The words this cartridge may never use about its display filter.**
    ///
    /// The Tower's own privacy transformation runs before persistence, at the
    /// one choke point every persisted pixel passes, and earns those names.
    /// This one runs on **read**: the raw frame stays exactly where it was and
    /// the capture manifests record `redaction: "none"`. Calling it the other
    /// thing would tell a wearer their recordings are altered when they are
    /// not.
    private let forbiddenFilterWords = [
        "redact", "anonymis", "anonymiz", "privacy-safe", "privacy safe",
        "de-identif", "deidentif", "scrubbed", "sanitis", "sanitiz",
    ]

    private func assertNoOverclaim(
        _ strings: [String], file: StaticString = #filePath, line: UInt = #line
    ) {
        for string in strings {
            let lowered = string.lowercased()
            XCTAssertFalse(lowered.isEmpty, "an empty string reached the screen", file: file, line: line)
            for phrase in forbiddenPhrases() {
                XCTAssertFalse(
                    lowered.contains(phrase),
                    "copy claims more than the sensor supports (\"\(phrase)\") in: \(string)",
                    file: file, line: line
                )
            }
            for word in forbiddenFilterWords {
                XCTAssertFalse(
                    lowered.contains(word),
                    "the display filter must never be called \"\(word)\" in: \(string)",
                    file: file, line: line
                )
            }
        }
    }

    /// `followingThisSession` defaults to `nil`, which **omits** the key. That
    /// default is also how the leftover-producer sentence went unread — see
    /// `testEverySessionSentenceMakesNoClaimItCannotSupport`.
    private func snapshot(
        state: String = "active",
        following: [String] = [],
        followingThisSession: [String]? = nil
    ) throws -> CartridgeSessionSnapshot {
        try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(
                    state: state,
                    sessionID: "61a78a9b32284cd2a89583d9a8cc8702",
                    startedAt: 1787833925.6521091,
                    following: following,
                    followingThisSession: followingThisSession,
                    captures: following
                )
            )
        )
    }

    /// `frameAvailable` and `imagerySource` default to `nil`, which omits the
    /// keys — the older-Tower shape, and the shape every case written before
    /// the two stores existed is about.
    private func description(
        available: Bool = true, reason: Any = NSNull(), subjectObscured: Any = 0.0,
        regionsFilled: Any = 0,
        frameAvailable: Bool? = nil,
        imagerySource: String? = nil,
        filterMeans: Any = "applied-on-read-the-stored-frame-is-unchanged",
        imageryRetention: Any = "capture-side"
    ) throws -> ObjectMemoryImageryDescription {
        try XCTUnwrap(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(
                    available: available,
                    reason: reason,
                    regionsFilled: regionsFilled,
                    subjectObscured: subjectObscured,
                    filterMeans: filterMeans,
                    frameAvailable: frameAvailable,
                    imagerySource: imagerySource,
                    imageryRetention: imageryRetention
                ),
                statusCode: 200
            )
        )
    }

    /// **A sentence this test could not see for months.**
    ///
    /// `ObjectMemoryCopy.leftoverProducerLine` contained the substring
    /// `"is still"`, which is on the forbidden list at the top of this file and
    /// on the identical list in `ObjectMemoryCopyTests` — and it passed both.
    /// The reason is the loop below: `snapshot(...)` leaves
    /// `following_this_session` **omitted**, `recordingsThisControlDidNotStart`
    /// returns `[]` for an unscoped Tower, and the line therefore returned
    /// `nil` in every case this test generated. The forbidden-phrase check ran
    /// over a sentence that was never produced.
    ///
    /// So the third loop is the fix that matters, and the reworded sentence is
    /// only the fix that was visible. It sets `followingThisSession: []` beside
    /// a **non-empty** `following` — the one combination that produces the
    /// line, and the one the default could never reach.
    ///
    /// The two are kept apart deliberately. `nil` and `[]` are different
    /// claims: an omitted field means the Tower cannot scope the question and
    /// must draw no warning, an empty list means it scoped it and this control
    /// started nothing that is running. The first loop covers the first, the
    /// third loop covers the second, and folding them together would lose the
    /// distinction the field exists for.
    func testEverySessionSentenceMakesNoClaimItCannotSupport() throws {
        // An older Tower: the key is absent, liveness falls back to
        // `following`, and no leftover warning is drawn at all.
        for state in ["stopped", "active", "paused", "draining"] {
            for following in [[], ["22e9d4289cb440fb"]] {
                assertNoOverclaim(
                    ObjectMemoryCopy.everyString(
                        for: try snapshot(state: state, following: following)
                    )
                )
            }
        }

        // A scoping Tower that says this control started what is running.
        for state in ["stopped", "active", "paused", "draining"] {
            assertNoOverclaim(
                ObjectMemoryCopy.everyString(
                    for: try snapshot(
                        state: state,
                        following: ["22e9d4289cb440fb"],
                        followingThisSession: ["22e9d4289cb440fb"]
                    )
                )
            )
        }

        // **The case the leftover sentence lives in**, and the one nothing here
        // used to generate: something is recording and this control did not
        // start it.
        for state in ["stopped", "active", "paused", "draining"] {
            for following in [["22e9d4289cb440fb"], ["22e9d4289cb440fb", "8c1f0a7b44de9021"]] {
                let scoped = try snapshot(
                    state: state, following: following, followingThisSession: []
                )
                XCTAssertNotNil(
                    ObjectMemoryCopy.leftoverProducerLine(scoped),
                    """
                    if this is ever nil again, the forbidden-phrase check below \
                    is running over a sentence nobody produced
                    """
                )
                assertNoOverclaim(ObjectMemoryCopy.everyString(for: scoped))
            }
        }
    }

    /// The leftover sentence claims only what the field knows.
    ///
    /// It does **not** claim session ownership, and it must not: the Tower
    /// scopes `following_this_session` by "started at or after this session
    /// last went active" and re-takes that mark on every Resume, so a producer
    /// that survived a Pause is genuinely this session's and is still correctly
    /// outside the scoped list. It also makes no claim about what would stop
    /// it, having previously ended "Restarting the Tower will" about a process
    /// that had already ignored `terminate()`.
    func testTheLeftoverSentenceClaimsNeitherOwnershipNorARemedy() throws {
        let line = try XCTUnwrap(
            ObjectMemoryCopy.leftoverProducerLine(
                try snapshot(
                    state: "active",
                    following: ["22e9d4289cb440fb"],
                    followingThisSession: []
                )
            )
        )
        let lowered = line.lowercased()
        XCTAssertFalse(
            lowered.contains("this session did not start"),
            "a producer that survived a Pause is this session's, and this list cannot tell"
        )
        XCTAssertFalse(
            lowered.contains("restart"),
            "nothing establishes that a restart reaches a producer that ignored terminate()"
        )
        XCTAssertTrue(
            lowered.contains("stopping here does not reach it"),
            "the one thing a wearer needs from this sentence is that Stop will not reach it"
        )
    }

    /// A Tower that does not scope the question draws no warning at all.
    ///
    /// `nil` is not `[]`. An unscoped Tower has said nothing about who started
    /// what, and turning that into "something is recording that this control
    /// did not start" would put a warning over every ordinary recording.
    func testAnUnscopedTowerDrawsNoLeftoverWarning() throws {
        XCTAssertNil(
            ObjectMemoryCopy.leftoverProducerLine(
                try snapshot(state: "active", following: ["22e9d4289cb440fb"])
            )
        )
    }

    func testEveryRefusalSentenceMakesNoClaimEither() throws {
        let stopped = try snapshot(state: "stopped")
        for action in CartridgeSessionAction.allCases {
            for reason in [
                CartridgeSessionRefusalReason.notActive, .notPaused, .unsupported,
                .unknownAction, CartridgeSessionRefusalReason(rawValue: "weather"),
            ] {
                assertNoOverclaim(
                    ObjectMemoryCopy.everyString(
                        for: CartridgeSessionRefusal(
                            action: action,
                            reason: reason,
                            message: "there is nothing to pause; this cartridge is stopped",
                            snapshot: stopped
                        )
                    )
                )
            }
        }
    }

    func testEveryPictureSentenceMakesNoClaimEither() throws {
        let cases: [(Bool, Any, Any, Any)] = [
            (true, NSNull(), 0.0, 0),
            (true, NSNull(), 0.42, 2),
            (false, "imagery-no-longer-available", 0.0, 0),
            (false, "record-has-no-frame-reference", 0.0, 0),
            (false, "no-such-observation", 0.0, 0),
            (false, "display-filter-unavailable", 0.0, 0),
            (false, "no-capture-root-configured", 0.0, 0),
            (false, "frame-unreadable", 0.0, 0),
            (false, "weather", 0.0, 0),
        ]
        for (available, reason, obscured, filled) in cases {
            let description = try self.description(
                available: available,
                reason: reason,
                subjectObscured: obscured,
                regionsFilled: filled
            )
            for kind in ObjectMemoryImageryKind.allCases {
                assertNoOverclaim(
                    ObjectMemoryCopy.everyString(for: description, kind: kind)
                )
            }
        }
    }

    /// **The caption carries the whole burden**, so it has to say all three
    /// things: what was observed, what the picture is, and what it is not.
    func testTheCaptionSaysWhatThePictureIsAndWhatItIsNot() {
        for kind in ObjectMemoryImageryKind.allCases {
            let caption = ObjectMemoryCopy
                .pictureCaption(objectClass: "laptop", kind: kind).lowercased()
            XCTAssertTrue(caption.contains("a laptop was visible"), caption)
            XCTAssertTrue(caption.contains("recording"), caption)
            XCTAssertTrue(caption.contains("not a place"), caption)
            XCTAssertTrue(caption.contains("anything about now"), caption)
            XCTAssertFalse(caption.contains("your laptop"), caption)
        }
    }

    /// The filter is described by what **ran**, with its blind spots named. A
    /// filter described only by what it catches reads as a guarantee.
    func testTheFilterIsDescribedByWhatRanAndByWhatIsStillInThePicture() throws {
        let line = ObjectMemoryCopy.filterLine(try description()).lowercased()
        XCTAssertTrue(line.contains("display filter"), line)
        XCTAssertTrue(line.contains("display-filter/yunet-2023mar@0.30"), line)
        XCTAssertTrue(line.contains("stored frame is unchanged"), line)
        XCTAssertTrue(
            line.contains("still in the picture"),
            "bodies, clothing, screens and undetected faces are all still there: \(line)"
        )
    }

    /// The 410's two halves have to arrive in the same breath. Either one alone
    /// is misleading.
    func testTheGoneSentenceSaysBothHalves() throws {
        let gone = try XCTUnwrap(
            ObjectMemoryImageryDecoder.descriptionUnwrapping(
                ObjectMemoryFixtures.imageryGone(), statusCode: 410
            )
        )
        let headline = ObjectMemoryCopy.noPictureHeadline(gone).lowercased()
        let explanation = ObjectMemoryCopy.noPictureExplanation(gone).lowercased()
        XCTAssertTrue(headline.contains("memory is kept"), headline)
        XCTAssertTrue(headline.contains("picture is gone"), headline)
        XCTAssertTrue(explanation.contains("untouched"), explanation)
        // Worded as the correct outcome it is, rather than as an error. The
        // sentence contains the word "failed" on purpose — "nothing failed" —
        // so the check is for the phrasings that would make it *read* as a
        // failure, not for the substring.
        XCTAssertTrue(explanation.contains("nothing failed"), explanation)
        for misreading in [
            "something went wrong", "could not be reached", "try again",
            "unavailable", "unreachable", "we were unable",
        ] {
            XCTAssertFalse(
                explanation.contains(misreading),
                "a 410 is capture-side retention working, not a failure: \(explanation)"
            )
        }
    }

    /// A 503 is a claim about the **Tower**, not about the record, and the
    /// sentence must say so — otherwise somebody is told their memory lost a
    /// picture when in fact no weights are installed.
    func testAFilterlessTowerIsDescribedAsTheTowerAndNotAsTheRecord() throws {
        let refused = try XCTUnwrap(
            ObjectMemoryImageryDecoder.descriptionUnwrapping(
                ObjectMemoryFixtures.imageryRefused(reason: "display-filter-unavailable"),
                statusCode: 503
            )
        )
        let explanation = ObjectMemoryCopy.noPictureExplanation(refused).lowercased()
        XCTAssertTrue(explanation.contains("this tower"), explanation)
        XCTAssertTrue(
            explanation.contains("records themselves are unaffected"),
            explanation
        )
        XCTAssertTrue(
            explanation.contains("refuses") || explanation.contains("unfiltered"),
            "the refusal is the correct behaviour and is stated as such: \(explanation)"
        )
    }

    /// The idempotent no-op is worded as success, because it is one.
    func testAnIdempotentNoOpReadsAsSuccessRatherThanAsAnError() {
        for action in CartridgeSessionAction.allCases {
            let line = ObjectMemoryCopy.idempotentNoOpLine(action).lowercased()
            XCTAssertFalse(line.contains("error"), line)
            XCTAssertFalse(line.contains("failed"), line)
            XCTAssertFalse(line.contains("could not"), line)
            XCTAssertTrue(line.contains("nothing needed to change"), line)
        }
    }

    /// The session id is not a capture id and is never shown as one — the same
    /// discipline `frameLine` applies for the same reason.
    func testTheSessionIdentifierIsNeverShownAsARecording() throws {
        let line = ObjectMemoryCopy.sessionProvenanceLine(try snapshot()).lowercased()
        XCTAssertTrue(line.contains("not a recording"), line)
        XCTAssertTrue(line.contains("not a place"), line)
    }

    /// Late attachment is a consent decision, and it is stated where the button
    /// is: a wearer who starts remembering at 15:03 has not asked for the 15:00
    /// part of the walk.
    func testStartSaysItDoesNotReachBackwards() {
        let line = ObjectMemoryCopy.startMeaningLine.lowercased()
        XCTAssertTrue(line.contains("does not reach back"), line)
    }

    /// What this app was told, including what it was **not** told. A screen
    /// that stayed quiet about the unknown detector would let a reader assume
    /// the app knows and is choosing not to say.
    func testTheConfigSummaryNamesOnlyWhatThePayloadCarried() throws {
        let listing = try ObjectMemoryDecoder.listing(from: ObjectMemoryFixtures.listing())
        let summary = ObjectMemoryCopy.whatThisTowerReports(.listing(listing))
        assertNoOverclaim([summary])
        XCTAssertTrue(summary.lowercased().contains("laptop and cell phone"), summary)
        XCTAssertTrue(
            summary.lowercased().contains("does not report which detector"),
            "the app must say it was not told, rather than imply it knows: \(summary)"
        )

        let verified = try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.listing(observations: [
                ObjectMemoryFixtures.observation(
                    tier: "verify", verification: ObjectMemoryFixtures.verification()
                )
            ])
        )
        let verifiedSummary = ObjectMemoryCopy.whatThisTowerReports(.listing(verified))
        XCTAssertTrue(
            verifiedSummary.contains("google/owlv2-base-patch16-ensemble"),
            "a verifier that ran is named from the payload: \(verifiedSummary)"
        )
    }

    /// A verifier's similarity is not a calibrated probability and is never
    /// presented as one.
    func testAVerificationScoreIsNotPresentedAsAProbability() throws {
        let listing = try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.listing(observations: [
                ObjectMemoryFixtures.observation(
                    tier: "verify", verification: ObjectMemoryFixtures.verification()
                )
            ])
        )
        let observation = try XCTUnwrap(listing.observations.first)
        let line = try XCTUnwrap(ObjectMemoryCopy.verificationLine(observation))
        assertNoOverclaim([line])
        XCTAssertTrue(line.contains("google/owlv2-base-patch16-ensemble"), line)
        XCTAssertTrue(line.lowercased().contains("not a calibrated probability"), line)
        XCTAssertTrue(
            line.lowercased().contains("94 crops"),
            "the threshold's provenance travels with the number: \(line)"
        )
    }
}

// MARK: - The additive record fields

final class ObjectMemoryAdditiveFieldTests: XCTestCase {

    /// The handle the imagery routes take, off the payload.
    func testARecordCarriesItsImageryHandle() throws {
        let listing = try ObjectMemoryDecoder.listing(from: ObjectMemoryFixtures.listing())
        let observation = try XCTUnwrap(listing.observations.first)
        XCTAssertEqual(observation.observationID, "9f2c41b7ad0e6538")
        XCTAssertEqual(
            observation.id, "9f2c41b7ad0e6538",
            "the row identity and the imagery handle are the same string when there is one"
        )
    }

    /// A record without a handle keeps its composed row identity and simply
    /// gets no picture. Refusing the whole listing over one such record would
    /// take a wearer's memory off screen to protect them from a missing
    /// thumbnail.
    func testARecordWithoutAHandleStillDecodesAndKeepsARowIdentity() throws {
        let listing = try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.listing(observations: [
                ObjectMemoryFixtures.observation(observationID: NSNull())
            ])
        )
        let observation = try XCTUnwrap(listing.observations.first)
        XCTAssertNil(observation.observationID)
        XCTAssertTrue(observation.id.contains("cell phone"))
    }

    /// `null` stays `nil` and never becomes a number: a `frame_count` of 0
    /// would claim a sighting that spanned no frames.
    func testMissingSightingFieldsStayNilRatherThanBecomingZero() throws {
        let listing = try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.listing(observations: [
                ObjectMemoryFixtures.observation(
                    lastSeenAt: NSNull(), frameCount: NSNull(), tier: NSNull()
                )
            ])
        )
        let observation = try XCTUnwrap(listing.observations.first)
        XCTAssertNil(observation.lastSeenAt)
        XCTAssertNil(observation.frameCount)
        XCTAssertNil(observation.tier)
        XCTAssertNil(ObjectMemoryCopy.durationLine(observation))
        XCTAssertNil(ObjectMemoryCopy.tierLine(observation))
    }

    /// A verification block that cannot be read reports **nothing was asked**,
    /// which is the weaker and safer of the two things it could say.
    func testAnUnreadableVerificationReportsNothingRatherThanGuessing() throws {
        let listing = try ObjectMemoryDecoder.listing(
            from: ObjectMemoryFixtures.listing(observations: [
                ObjectMemoryFixtures.observation(verification: ["model": "something"])
            ])
        )
        let observation = try XCTUnwrap(listing.observations.first)
        XCTAssertNil(observation.verification)
        XCTAssertNil(ObjectMemoryCopy.verificationLine(observation))
    }
}

// MARK: - The status-code defect

@MainActor
final class ObjectMemoryStatusCodeTests: XCTestCase {

    private func makeClient() -> TowerObjectMemoryClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ObjectMemoryStubProtocol.self]
        return TowerObjectMemoryClient(
            http: ObjectMemoryHTTPClient(
                baseURL: URL(string: "http://tower.test")!,
                session: URLSession(configuration: configuration)
            )
        )
    }

    override func setUp() {
        super.setUp()
        ObjectMemoryStubProtocol.handler = nil
    }

    override func tearDown() {
        ObjectMemoryStubProtocol.handler = nil
        super.tearDown()
    }

    /// **A Tower that answered must never be reported as unreachable.**
    ///
    /// Every non-404 non-2xx used to become `.transport("The Tower answered
    /// N.")`, which `ObjectMemoryState.phase` maps to `.disconnected` — so a
    /// clear answer from a working machine was presented as a connection
    /// failure, sending a person to check a network cable.
    func testAnUnusableStatusIsAFailureAndNotADisconnection() async {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 410, httpVersion: nil, headerFields: nil
            )!
            return (response, Data("{}".utf8))
        }

        let client = makeClient()
        await client.ask(.listing(objectClass: nil))

        guard case .failed(let failure) = client.state else {
            return XCTFail("an unusable status did not fail: \(client.state)")
        }
        XCTAssertEqual(failure.kind, .towerReportedFailure)
        XCTAssertNotEqual(
            failure.kind, .transport,
            "the Tower answered; calling that a transport failure is a false claim about it"
        )
        XCTAssertEqual(
            client.state.phase, .failed,
            "and it must not reach the screen under the disconnected glyph"
        )
    }

    /// A 404 keeps meaning what it meant: this Tower serves no object memory.
    func testTheConfigurationFourOhFourIsUnaffected() async {
        ObjectMemoryStubProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 404, httpVersion: nil, headerFields: nil
            )!
            return (response, Data("{\"detail\":\"no object memory root is configured\"}".utf8))
        }
        let client = makeClient()
        await client.ask(.listing(objectClass: nil))
        XCTAssertEqual(client.state, .noObjectMemory)
    }
}

// MARK: - Session, pinned against a real Tower

/// Session payloads taken **verbatim off the running Tower** on 2026-08-27,
/// rather than from a reading of §9 of its contract.
///
/// This is where the wire/document disagreement was found: §4.1 of
/// `TOWER-UNIFIED-CARTRIDGES.md` records `resume` from `stopped` as
/// `not-active`; the bytes below say `not-paused`. Both are in §10's
/// vocabulary and both are truthful, so neither is a bug — but a client that
/// switched on one of them would be wrong against the other Tower. Hence the
/// copy is written from the action and the state, and both words are pinned
/// here so a change to either is a test failure rather than a surprise.
final class CartridgeSessionRealTowerTests: XCTestCase {

    private func json(_ text: String) throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        )
    }

    private func snapshot(_ text: String) throws -> CartridgeSessionSnapshot {
        try XCTUnwrap(CartridgeSessionDecoder.snapshot(from: try json(text)))
    }

    func testARealStoppedSessionDecodes() throws {
        let stopped = try snapshot(Self.stoppedFromTower)
        XCTAssertEqual(stopped.contract, "cartridge_session.control/2026-08-27")
        XCTAssertEqual(stopped.state, .stopped)
        XCTAssertEqual(stopped.stateMeans, "intent-not-liveness")
        XCTAssertTrue(stopped.supported)
        XCTAssertNil(stopped.sessionID)
        XCTAssertFalse(stopped.isFollowingACapture)
        XCTAssertEqual(stopped.offeredActions, [.start, .pause, .resume, .stop])
    }

    /// **Start before the camera is running is legal**, and this is what it
    /// looks like: `active`, a session id, and `attached_capture_id: null`.
    func testARealStartBeforeTheCameraIsActiveWithNothingAttached() throws {
        let started = try snapshot(Self.startFromTower)
        XCTAssertEqual(started.state, .active)
        XCTAssertEqual(started.sessionID, "61a78a9b32284cd2a89583d9a8cc8702")
        XCTAssertNil(started.attachedCaptureID)
        XCTAssertEqual(started.changed, true)
        XCTAssertFalse(
            started.isFollowingACapture,
            "an intent to remember is not a producer writing anything"
        )
    }

    /// **A second start is 200 with `changed: false`** — honoured, nothing
    /// moved. The session id does not move either.
    func testARealSecondStartIsHonouredAndUnmoved() throws {
        let first = try snapshot(Self.startFromTower)
        let second = try snapshot(Self.secondStartFromTower)
        XCTAssertEqual(second.changed, false)
        XCTAssertEqual(second.wasAnIdempotentNoOp, true)
        XCTAssertEqual(second.state, .active)
        XCTAssertEqual(second.sessionID, first.sessionID)
    }

    /// **Stop is never refused.** From `stopped` it is 200 with
    /// `changed: false`, and the session id is already cleared.
    func testARealStopFromStoppedIsHonoured() throws {
        let stopped = try snapshot(Self.stopFromStoppedFromTower)
        XCTAssertEqual(stopped.changed, false)
        XCTAssertEqual(stopped.accepted, true)
        XCTAssertNil(stopped.sessionID)
    }

    /// **The disagreement, pinned.** The wire says `not-paused` where the
    /// contract document says `not-active`.
    func testARealResumeFromStoppedIsRefusedWithTheWireWord() throws {
        let body = try XCTUnwrap(
            try json(Self.resumeRefusedFromTower)["detail"] as? [String: Any]
        )
        let refusal = try XCTUnwrap(
            CartridgeSessionDecoder.refusal(from: body, action: .resume)
        )
        XCTAssertEqual(
            refusal.reason, .notPaused,
            """
            the running Tower answers `not-paused` here; §4.1 of the unified \
            contract says `not-active`, and this build must handle both
            """
        )
        XCTAssertTrue(
            CartridgeSessionRefusalReason.interchangeableStateRefusals.contains(refusal.reason)
        )
        XCTAssertEqual(refusal.snapshot.state, .stopped)
        XCTAssertEqual(refusal.snapshot.accepted, false)
    }

    /// And `pause` from `stopped` is the **other** word, on the same Tower, for
    /// the neighbouring situation — which is why neither word can be treated as
    /// the one that means "wrong state".
    func testARealPauseFromStoppedUsesTheOtherWord() throws {
        let body = try XCTUnwrap(
            try json(Self.pauseRefusedFromTower)["detail"] as? [String: Any]
        )
        let refusal = try XCTUnwrap(
            CartridgeSessionDecoder.refusal(from: body, action: .pause)
        )
        XCTAssertEqual(refusal.reason, .notActive)
    }

    func testARealUnknownVerbIsRefusedWithoutChangingAnything() throws {
        let body = try XCTUnwrap(
            try json(Self.unknownActionFromTower)["detail"] as? [String: Any]
        )
        let refusal = try XCTUnwrap(
            CartridgeSessionDecoder.refusal(from: body, action: .start)
        )
        XCTAssertEqual(refusal.reason, .unknownAction)
        XCTAssertEqual(refusal.snapshot.state, .stopped)
    }

    /// The `imagery` block, and the 404 body, taken off the same Tower.
    func testTheRealEnvelopeCarriesTheImageryTemplates() throws {
        let listing = try ObjectMemoryDecoder.listing(from: try json(Self.envelopeFromTower))
        let imagery = try XCTUnwrap(listing.envelope.imagery)
        XCTAssertEqual(imagery.contract, "object_memory.imagery/2026-08-27")
        XCTAssertEqual(
            imagery.viewTemplate, "/object-memory/observations/{observation_id}/imagery"
        )
        XCTAssertEqual(
            imagery.url(
                for: .crop,
                observationID: "9f2c41b7ad0e6538",
                relativeTo: URL(string: "http://127.0.0.1:8765")!
            )?.absoluteString,
            "http://127.0.0.1:8765/object-memory/observations/9f2c41b7ad0e6538/crop"
        )
    }

    /// **Only an unknown handle is a real 404 on `/imagery`**, and this is what
    /// one looks like: `filter` is null because nothing was served, and
    /// `memory_retained` is false because there is no record to retain.
    func testARealUnknownHandleIsTheOneRealFourOhFour() throws {
        let description = try XCTUnwrap(
            ObjectMemoryImageryDecoder.descriptionUnwrapping(
                try json(Self.imageryNotFoundFromTower), statusCode: 404
            )
        )
        XCTAssertEqual(description.situation, .noSuchRecord)
        XCTAssertFalse(description.memoryRetained)
        XCTAssertNil(description.filter)
        XCTAssertNil(description.objectClass)
        XCTAssertEqual(description.regionsFilled, 0)
        XCTAssertEqual(description.subjectObscured, 0.0)
    }

    /// `GET /cartridges/object_memory/session`, verbatim.
    private static let stoppedFromTower = """
        {"cartridge":"object_memory","worker":"object-memory-session","supported":true,"state":"stopped","session_id":null,"started_at":null,"changed_at":1787832967.312166,"following":[],"captures":[],"contract":"cartridge_session.control/2026-08-27","state_means":"intent-not-liveness","states":["stopped","active","paused"],"actions":["start","pause","resume","stop"]}
        """

    /// `POST …/start` from `stopped`, with no camera running.
    private static let startFromTower = """
        {"cartridge":"object_memory","worker":"object-memory-session","supported":true,"state":"active","session_id":"61a78a9b32284cd2a89583d9a8cc8702","started_at":1787833925.6521091,"changed_at":1787833925.6521118,"following":[],"captures":[],"changed":true,"attached_capture_id":null,"accepted":true,"contract":"cartridge_session.control/2026-08-27","state_means":"intent-not-liveness","states":["stopped","active","paused"],"actions":["start","pause","resume","stop"]}
        """

    /// `POST …/start` again. **200, `changed: false`. Not an error.**
    private static let secondStartFromTower = """
        {"cartridge":"object_memory","worker":"object-memory-session","supported":true,"state":"active","session_id":"61a78a9b32284cd2a89583d9a8cc8702","started_at":1787833925.6521091,"changed_at":1787833925.6521118,"following":[],"captures":[],"changed":false,"attached_capture_id":null,"accepted":true,"contract":"cartridge_session.control/2026-08-27","state_means":"intent-not-liveness","states":["stopped","active","paused"],"actions":["start","pause","resume","stop"]}
        """

    /// `POST …/stop` from `stopped`. Never refused.
    private static let stopFromStoppedFromTower = """
        {"cartridge":"object_memory","worker":"object-memory-session","supported":true,"state":"stopped","session_id":null,"started_at":null,"changed_at":1787832967.312166,"following":[],"captures":[],"changed":false,"attached_capture_id":null,"accepted":true,"contract":"cartridge_session.control/2026-08-27","state_means":"intent-not-liveness","states":["stopped","active","paused"],"actions":["start","pause","resume","stop"]}
        """

    /// `POST …/resume` from `stopped`. **409, and the word is `not-paused`.**
    private static let resumeRefusedFromTower = """
        {"detail":{"accepted":false,"reason":"not-paused","message":"there is no paused session to resume; this cartridge is stopped","cartridge":"object_memory","worker":"object-memory-session","supported":true,"state":"stopped","session_id":null,"started_at":null,"changed_at":1787832967.312166,"following":[],"captures":[],"contract":"cartridge_session.control/2026-08-27","state_means":"intent-not-liveness","states":["stopped","active","paused"],"actions":["start","pause","resume","stop"]}}
        """

    /// `POST …/pause` from `stopped`. 409, and the word is `not-active`.
    private static let pauseRefusedFromTower = """
        {"detail":{"accepted":false,"reason":"not-active","message":"there is nothing to pause; this cartridge is stopped","cartridge":"object_memory","worker":"object-memory-session","supported":true,"state":"stopped","session_id":null,"started_at":null,"changed_at":1787832967.312166,"following":[],"captures":[],"contract":"cartridge_session.control/2026-08-27","state_means":"intent-not-liveness","states":["stopped","active","paused"],"actions":["start","pause","resume","stop"]}}
        """

    /// `POST …/wiggle`.
    private static let unknownActionFromTower = """
        {"detail":{"accepted":false,"reason":"unknown-action","message":"'wiggle' is not an action; expected one of start, pause, resume, stop","cartridge":"object_memory","worker":"object-memory-session","supported":true,"state":"stopped","session_id":null,"started_at":null,"changed_at":1787833925.690068,"following":[],"captures":[],"contract":"cartridge_session.control/2026-08-27","state_means":"intent-not-liveness","states":["stopped","active","paused"],"actions":["start","pause","resume","stop"]}}
        """

    /// `GET /object-memory/observations`, verbatim, on a Tower whose store is
    /// empty and whose window was written unbounded — which is why
    /// `effective_days` is null rather than 30.
    private static let envelopeFromTower = """
        {"contract":"object_memory.observations/2026-08-26","claim":"category-was-visible-once","identity":"category-not-instance","absence_means":"not-observed-by-this-cartridge","spatial_ref":null,"recorded_classes":["laptop","cell phone"],"imagery":{"contract":"object_memory.imagery/2026-08-27","claim":"frame-from-the-recording-this-record-was-derived-from","filter_means":"applied-on-read-the-stored-frame-is-unchanged","view":"/object-memory/observations/{observation_id}/imagery","frame":"/object-memory/observations/{observation_id}/frame","crop":"/object-memory/observations/{observation_id}/crop"},"retention":{"requested_days":null,"effective_days":null,"clamped":false,"policy":"min(persisted, requested): a reader may narrow this window and can never widen it"},"object_class":null,"observation_count":0,"observations":[]}
        """

    /// `GET /object-memory/observations/0123456789abcdef/imagery`, verbatim.
    private static let imageryNotFoundFromTower = """
        {"detail":{"contract":"object_memory.imagery/2026-08-27","observation_id":"0123456789abcdef","object_class":null,"claim":"frame-from-the-recording-this-record-was-derived-from","available":false,"reason":"no-such-observation","memory_retained":false,"filter":null,"filter_means":"applied-on-read-the-stored-frame-is-unchanged","regions_filled":0,"subject_obscured":0.0,"bounding_box_normalized":null,"imagery_retention":"capture-side"}}
        """
}


// MARK: - The composed lifecycle

/// A shared ordering log for the two collaborators a Start sequences.
///
/// The sequencing *is* the behaviour here: "the Tower is asked before the
/// camera is" and "the camera is asked before the Tower is" produce identical
/// call counts and completely different products. Counting calls cannot tell
/// them apart, so both fakes append to one list instead.
@MainActor
final class RecordingJournal {
    private(set) var steps: [String] = []
    func record(_ step: String) { steps.append(step) }
}

/// A counter that `refreshRecords` can move.
///
/// **A reference type rather than a captured local `var`, and that is a
/// compiler requirement rather than a preference.**
/// `ObjectMemoryRecordingCoordinator.refreshRecords` is
/// `(@MainActor () -> Void)?`, and a global-actor-isolated function type is
/// implicitly `@Sendable` under this project's concurrency settings — so
/// `var n = 0; recording.refreshRecords = { n += 1 }` mutates captured state
/// from a `@Sendable` closure, which is a concurrency diagnostic. A
/// main-actor-isolated class is `Sendable` by its isolation, and what the
/// closure captures is the reference; the mutation happens on the actor the
/// closure is already isolated to.
@MainActor
final class RefreshCounter {
    private(set) var count = 0
    func bump() { count += 1 }
}

/// A camera owner with no DAT, no device and no permission.
///
/// `GlassesConnection`'s capture surface cannot be driven from a test at all —
/// `createSession` throws without an eligible device, `deviceSessionState` is
/// `private(set)` and moved only by DAT callbacks, and the whole thing is
/// `#if DEBUG`. That is exactly why `ObjectMemoryCaptureOwner` exists, and this
/// is the other implementation of it.
@MainActor
final class FakeCaptureOwner: ObjectMemoryCaptureOwner {
    /// What this app's claim on the camera currently is.
    var claim: CaptureClaim = .unclaimed
    /// What a start moves the claim to. `.running` unless a test says otherwise.
    var claimAfterStart: CaptureClaim = .running
    /// What `startCameraSession()` reports afterwards. `nil` means it proceeded.
    var refusal: CaptureStartRefusal?

    private(set) var starts = 0
    private(set) var stops = 0
    var journal: RecordingJournal?

    private let subject = PassthroughSubject<CaptureClaim, Never>()

    var captureClaim: CaptureClaim { claim }
    var captureClaimUpdates: AnyPublisher<CaptureClaim, Never> { subject.eraseToAnyPublisher() }
    var lastCaptureStartRefusal: CaptureStartRefusal? { refusal }

    func startCameraSession() {
        starts += 1
        journal?.record("camera:start")
        if refusal == nil { claim = claimAfterStart }
    }

    func stopCameraSession() {
        stops += 1
        journal?.record("camera:stop")
        claim = .unclaimed
    }

    /// A claim that changed for a reason nobody on the Object Memory screen
    /// asked for: glasses folded, Bluetooth dropped, Stop pressed on Home, a
    /// cap-touch pause. Published the way `GlassesConnection` publishes it.
    func publish(_ claim: CaptureClaim) {
        self.claim = claim
        subject.send(claim)
    }
}

/// **What the one Start button actually does, in what order, and what it
/// refuses to do.**
///
/// Every test here is about behaviour rather than layout: which collaborator
/// was called, in which order, what the phase became, and — the case this whole
/// cartridge exists for — whether a phase that merely *asked* for something is
/// ever rendered as one that observed it.
@MainActor
final class ObjectMemoryRecordingTests: XCTestCase {

    // MARK: Building the pieces

    private func snapshot(
        state: String = "active", following: [String] = [], supported: Bool = true
    ) throws -> CartridgeSessionSnapshot {
        try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(
                    state: state, supported: supported, following: following
                )
            )
        )
    }

    private func refusal(
        action: CartridgeSessionAction = .resume, reason: String = "not-paused"
    ) throws -> CartridgeSessionRefusal {
        let body = try XCTUnwrap(
            ObjectMemoryFixtures.sessionRefusal(reason: reason)["detail"] as? [String: Any]
        )
        return try XCTUnwrap(CartridgeSessionDecoder.refusal(from: body, action: action))
    }

    /// A coordinator whose waits are measured in milliseconds.
    ///
    /// The production cadence is three seconds with a twelve-second budget, and
    /// a suite that waited those out would take a minute per convergence test.
    /// The *shape* of the wait is what is under test — a bounded poll with a
    /// deadline — and that is identical at either scale.
    private func makeCoordinator(
        camera: FakeCaptureOwner?, client: StubObjectMemoryClient
    ) -> ObjectMemoryRecordingCoordinator {
        ObjectMemoryRecordingCoordinator(
            camera: camera,
            client: client,
            interval: .milliseconds(1),
            budget: .milliseconds(30)
        )
    }

    /// A coordinator whose convergence is long enough to do something *during*.
    ///
    /// The one above settles in about thirty milliseconds, which is what makes
    /// the outcome tests fast and what makes them useless for the window this
    /// screen's worst defect lived in: the twelve seconds after a Start, while
    /// the poll is running. A test that wants to press Stop mid-convergence, or
    /// to let a camera refusal arrive late, needs the wait to still be there
    /// when it looks. Five seconds is a ceiling rather than a duration — every
    /// test using it ends the wait itself and none of them runs it out.
    private func makeConvergingCoordinator(
        camera: FakeCaptureOwner?, client: StubObjectMemoryClient
    ) -> ObjectMemoryRecordingCoordinator {
        ObjectMemoryRecordingCoordinator(
            camera: camera,
            client: client,
            interval: .milliseconds(10),
            budget: .seconds(5)
        )
    }

    /// Waits for the in-flight sequence to finish, and for the main-queue hop
    /// the two Combine subscriptions take.
    ///
    /// `isSequenceRunning` and **not** `isActing`. The two used to be one flag
    /// and are now deliberately not: `isActing` covers the `POST` and the
    /// camera call, and goes false the moment convergence begins, because that
    /// is the flag both `.disabled` gates read and leaving it up through a
    /// twelve-second poll disabled every control on the screen including Stop.
    /// Waiting on it here would return before a Start had converged and would
    /// assert against a phase that was still moving. Same semantics as before,
    /// read off the flag that still has them.
    private func settle(_ recording: ObjectMemoryRecordingCoordinator) async {
        var turns = 0
        while recording.isSequenceRunning && turns < 500 {
            try? await Task.sleep(nanoseconds: 2_000_000)
            turns += 1
        }
        // One more turn, so a reading published during the sequence has been
        // delivered before anything is asserted.
        try? await Task.sleep(nanoseconds: 20_000_000)
    }

    /// Waits until a Start has got as far as its convergence poll.
    ///
    /// The point in the sequence where the `POST` and the camera call are done,
    /// the gate is down, and the screen is sitting on "the Tower accepted,
    /// waiting for a producer" — which is where a wearer presses Stop.
    private func waitUntilConverging(_ recording: ObjectMemoryRecordingCoordinator) async {
        var turns = 0
        while recording.phase != .waitingToBeFollowed && turns < 500 {
            try? await Task.sleep(nanoseconds: 2_000_000)
            turns += 1
        }
    }

    // MARK: The order

    /// **The Tower first.** Starting the camera first would open a capture
    /// against a gate that is not open yet, and a producer that attaches
    /// afterwards runs `--attach-mode from-now` and never reads back the frames
    /// that were lost in between.
    func testStartAsksTheTowerBeforeItAsksForACapture() async throws {
        let journal = RecordingJournal()
        let client = StubObjectMemoryClient()
        client.journal = journal
        client.sessionScript = [.known(try snapshot(following: ["22e9d4289cb440fb"]))]
        let camera = FakeCaptureOwner()
        camera.journal = journal

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(
            Array(journal.steps.prefix(2)), ["tower:start", "camera:start"],
            "the camera must not open a capture before the Tower's gate is open"
        )
        XCTAssertEqual(client.applied, [.start])
        XCTAssertEqual(camera.starts, 1)
    }

    /// A capture Home or World Builder started is left exactly as it is.
    /// `startCameraSession()` refuses when a session exists, so calling it would
    /// be asking for a refusal and then having to explain one.
    func testStartDoesNotOpenASecondCaptureOverOneAlreadyRunning() async throws {
        let client = StubObjectMemoryClient()
        client.sessionScript = [.known(try snapshot(following: ["22e9d4289cb440fb"]))]
        let camera = FakeCaptureOwner()
        camera.claim = .running

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(camera.starts, 0, "a capture was already running")
        XCTAssertEqual(client.applied, [.start], "the Tower half must still happen")
        XCTAssertFalse(
            recording.reading.cameraStartedHere,
            "a capture this screen did not start must not be claimed as its own"
        )
    }

    /// **The Tower first on the way out too.** `stop` is never refused and it is
    /// what detaches the producer so it can finalise the record it is holding;
    /// killing the frames first would leave the producer to notice the stream
    /// ending on its own.
    func testStopAsksTheTowerBeforeItEndsTheCapture() async throws {
        let journal = RecordingJournal()
        let client = StubObjectMemoryClient()
        client.journal = journal
        client.sessionScript = [.known(try snapshot(following: ["22e9d4289cb440fb"]))]
        let camera = FakeCaptureOwner()
        camera.journal = journal

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        client.sessionScript = [.known(try snapshot(state: "stopped"))]
        recording.stop()
        await settle(recording)

        let stopSteps = journal.steps.drop { $0 != "tower:stop" }
        XCTAssertEqual(
            Array(stopSteps.prefix(2)), ["tower:stop", "camera:stop"],
            "the producer must be detached before its frames stop arriving"
        )
        XCTAssertEqual(camera.stops, 1)
    }

    /// Ownership, in the direction that matters: Stop ends a capture this screen
    /// started, and leaves one it did not.
    func testStopOnlyEndsACaptureThisScreenStarted() async throws {
        let client = StubObjectMemoryClient()
        client.sessionScript = [.known(try snapshot(following: ["22e9d4289cb440fb"]))]
        let camera = FakeCaptureOwner()
        camera.claim = .running

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        client.sessionScript = [.known(try snapshot(state: "stopped"))]
        recording.stop()
        await settle(recording)

        XCTAssertEqual(client.applied, [.start, .stop], "the session must still end")
        XCTAssertEqual(
            camera.stops, 0,
            "ending a capture Home started would reach across two other screens"
        )
        XCTAssertEqual(camera.claim, .running)
    }

    /// A capture that ends underneath this screen is no longer this screen's to
    /// end. Otherwise a later Stop would call `stopCameraSession()` against
    /// whatever session exists by then — possibly one it never started.
    func testACaptureThatEndsElsewhereIsNoLongerOwnedHere() async throws {
        let client = StubObjectMemoryClient()
        client.sessionScript = [.known(try snapshot(following: ["22e9d4289cb440fb"]))]
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)
        XCTAssertTrue(recording.reading.cameraStartedHere)

        // Glasses folded, or Stop pressed on Home.
        camera.publish(.unclaimed)
        try? await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertFalse(recording.reading.cameraStartedHere)

        client.sessionScript = [.known(try snapshot(state: "stopped"))]
        recording.stop()
        await settle(recording)
        XCTAssertEqual(camera.stops, 0)
    }

    // MARK: Double taps

    /// A second tap while a sequence is in flight is dropped entirely, so two
    /// overlapping sequences cannot race over who owns the capture.
    func testASecondTapWhileASequenceIsInFlightIsDropped() async throws {
        let client = StubObjectMemoryClient()
        client.sessionScript = [.known(try snapshot(following: ["22e9d4289cb440fb"]))]
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        recording.start()
        recording.start()
        await settle(recording)

        XCTAssertEqual(client.applied, [.start])
        XCTAssertEqual(camera.starts, 1)
    }

    /// **A repeated verb is not an error.** The Tower answers a second `start`,
    /// a second `pause` and a `stop` from stopped with 200 and
    /// `changed: false` — "you already have what you asked for" — and every one
    /// of those must land in an ordinary reading rather than a failure.
    func testARepeatedVerbConvergesAndIsNotShownAsAnError() async throws {
        let following = ObjectMemorySessionState.known(
            try snapshot(following: ["22e9d4289cb440fb"])
        )
        let paused = ObjectMemorySessionState.known(try snapshot(state: "paused"))
        let stopped = ObjectMemorySessionState.known(try snapshot(state: "stopped"))

        let client = StubObjectMemoryClient()
        let camera = FakeCaptureOwner()
        let recording = makeCoordinator(camera: camera, client: client)

        client.sessionAfterRead = following
        recording.start()
        await settle(recording)
        recording.start()
        await settle(recording)
        XCTAssertEqual(recording.phase, .remembering)

        client.sessionAfterRead = paused
        recording.pause()
        await settle(recording)
        recording.pause()
        await settle(recording)
        XCTAssertEqual(recording.phase, .paused)

        client.sessionAfterRead = following
        recording.resume()
        await settle(recording)
        recording.resume()
        await settle(recording)
        XCTAssertEqual(recording.phase, .remembering)

        client.sessionAfterRead = stopped
        recording.stop()
        await settle(recording)
        recording.stop()
        await settle(recording)
        XCTAssertEqual(recording.phase, .stopped)

        XCTAssertEqual(
            client.applied,
            [.start, .start, .pause, .pause, .resume, .resume, .stop, .stop],
            "a repeated verb is sent; it is the Tower's job to answer changed:false"
        )
    }

    // MARK: Asked for, and not observed

    /// **The case this whole type exists for.**
    ///
    /// A Start answers 200 `active` with `attached_capture_id: null`, which is
    /// documented, honest and successful — and means nothing is being recorded.
    /// Past the deadline that has to read as *asked for, and not observed*, and
    /// it must not be drawn as recording.
    func testAnActiveSessionFollowingNothingIsNotRenderedAsRemembering() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(recording.phase, .notObserved)
        XCTAssertNotEqual(recording.phase, .remembering)
        XCTAssertFalse(
            recording.phase.isFollowingACapture,
            "a successful Start is not a recording"
        )
        XCTAssertGreaterThan(
            client.sessionReads, 0, "the wait must actually re-read `following`"
        )
        XCTAssertNotEqual(
            ObjectMemoryCopy.recordingHeadline(recording.reading),
            ObjectMemoryCopy.recordingHeadline(
                ObjectMemoryRecordingReading(
                    phase: .remembering,
                    camera: recording.reading.camera,
                    cameraStartedHere: recording.reading.cameraStartedHere,
                    cameraIsReachable: true
                )
            ),
            "asked-for and observed must not read as the same sentence"
        )
    }

    /// A producer that attaches during the wait settles the phase, and settles
    /// it from `following` rather than from `state`.
    func testAProducerThatAttachesDuringTheWaitSettlesTheClaim() async throws {
        let client = StubObjectMemoryClient()
        // The Start's own read-back: accepted, nothing attached yet.
        client.sessionScript = [
            .known(try snapshot(state: "active", following: [])),
            .known(try snapshot(state: "active", following: [])),
            .known(try snapshot(state: "active", following: ["22e9d4289cb440fb"])),
        ]
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(recording.phase, .remembering)
    }

    // MARK: Losing the ability to tell

    /// **A read that fails clears the liveness claim.** A screen that keeps
    /// saying "remembering" after it stopped being able to check is asserting
    /// something it no longer knows, and this is the exact shape a Tower going
    /// away mid-session takes.
    func testATransportFailureDuringTheWatchLoopClearsTheLivenessClaim() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)
        XCTAssertEqual(recording.phase, .remembering)

        // The workspace's own watch loop reads again and the Tower is gone.
        let failure = CartridgeFailure(
            kind: .transport, message: "The Tower did not answer: connection lost"
        )
        client.publish(ObjectMemorySessionState.failed(failure))
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(recording.phase, .cannotTell(failure))
        XCTAssertFalse(
            recording.phase.isFollowingACapture,
            "a stale 'remembering' after the reading stopped arriving is a false claim"
        )
    }

    /// A producer that dies with the Tower still answering empties `following`,
    /// and the phase must follow it down rather than staying where the last
    /// action put it.
    func testAProducerThatDiesStopsTheRememberingClaim() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)
        XCTAssertEqual(recording.phase, .remembering)

        client.publish(ObjectMemorySessionState.known(try snapshot(state: "active", following: [])))
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(recording.phase, .notObserved)
    }

    // MARK: Refusals, from both halves

    /// **A 409 is an answer, not a failure.** `resume` from `stopped` says which
    /// control would have worked, and that sentence is exactly what a person
    /// needs.
    func testResumeFromStoppedIsReportedAsAnAnswer() async throws {
        let refused = try refusal(action: .resume, reason: "not-paused")
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .refused(refused)
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.resume()
        await settle(recording)

        XCTAssertEqual(recording.phase, .refused(refused))
        XCTAssertEqual(
            ObjectMemoryCopy.recordingHeadline(recording.reading),
            ObjectMemoryCopy.refusalLine(refused),
            "a refusal must be worded as the answer it is"
        )
        XCTAssertEqual(camera.starts, 0)
        XCTAssertEqual(camera.stops, 0)
    }

    /// A Tower with no producer disables the control and says why — and the
    /// camera is never asked, because a capture running for a memory that
    /// cannot be written into is a camera running for nothing.
    func testAnUnsupportedSessionDisablesTheControlAndSaysWhy() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "stopped", supported: false))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(recording.phase, .unsupported)
        XCTAssertEqual(camera.starts, 0)
        XCTAssertEqual(
            ObjectMemoryCopy.recordingHeadline(recording.reading),
            ObjectMemoryCopy.sessionUnsupported
        )
    }

    /// A Tower that offers no controllable session at all is the same
    /// configuration answer, reached by a 404 rather than by `supported: false`.
    func testNoControllableSessionIsAlsoUnsupported() async {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .noSessionControl
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(recording.phase, .unsupported)
        XCTAssertEqual(camera.starts, 0)
    }

    /// An unreachable Tower stops the sequence before it opens a capture.
    func testAnUnreachableTowerNeverOpensACapture() async {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .failed(
            CartridgeFailure(kind: .transport, message: "The Tower did not answer.")
        )
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(camera.starts, 0, "a camera running for a gate that is shut records nothing")
        if case .failed = recording.phase {} else {
            XCTFail("an unreachable Tower became something else: \(recording.phase)")
        }
    }

    /// **A hardware pause is reported, not overridden.** The glasses paused
    /// delivery themselves and resume on their own; nothing in this app can
    /// force it, and the Tower's half is deliberately left standing so the next
    /// capture to open finds the gate ready.
    func testADevicePausedCaptureIsReportedRatherThanRestarted() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()
        camera.claim = .devicePaused

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(camera.starts, 0, "there is no way to restart a device-paused session")
        XCTAssertEqual(recording.phase, .cameraRefused(.deviceHasPausedCapture))
        XCTAssertEqual(client.applied, [.start], "the Tower's gate stays open")
    }

    /// A camera that refuses for its own reason carries that reason through to
    /// the sentence, instead of arriving as a generic failure.
    func testACameraRefusalCarriesItsReason() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()
        camera.refusal = .noActiveDevice

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(recording.phase, .cameraRefused(.noActiveDevice))
        XCTAssertFalse(
            recording.reading.cameraStartedHere,
            "a refused start must not be recorded as ownership"
        )
        XCTAssertEqual(
            ObjectMemoryCopy.recordingHeadline(recording.reading),
            ObjectMemoryCopy.cameraRefusalLine(.noActiveDevice)
        )
    }

    /// **A capture that is shutting down belongs to nobody.**
    ///
    /// `.ending` used to share a branch with `.running` and be recorded as
    /// "somebody else owns it", which was false in both halves: the previous
    /// owner has already let go, and `startCameraSession()` refuses in this
    /// window anyway. The wearer got no camera, no refusal and no sentence —
    /// a Start that reported nothing at all.
    func testStartWhileACaptureIsShuttingDownIsRefusedRatherThanSilent() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()
        camera.claim = .ending

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(
            camera.starts, 0, "startCameraSession() refuses while a session is being torn down"
        )
        XCTAssertEqual(recording.phase, .cameraRefused(.captureIsShuttingDown))
        XCTAssertFalse(
            recording.reading.cameraStartedHere,
            "nobody owns a capture that is dying, least of all the screen that did not start it"
        )
        XCTAssertEqual(client.applied, [.start], "the Tower's gate stays open")
        XCTAssertNotEqual(
            ObjectMemoryCopy.recordingHeadline(recording.reading),
            ObjectMemoryCopy.cameraRefusalLine(.alreadyRunning),
            """
            "a capture was already under way, so this screen left it alone" \
            sends a wearer to look for a stream that is on its way out
            """
        )
    }

    /// A teardown running underneath this screen does **not** give away a
    /// capture this screen started. Ownership is dropped when the claim reaches
    /// `.unclaimed` and not before — a capture that is stopping is still ours
    /// to stop, and `stopCameraSession()` on an already-stopping session is the
    /// same idempotent no-op it always was.
    func testACaptureThatIsShuttingDownIsStillThisScreensToStop() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)
        XCTAssertTrue(recording.reading.cameraStartedHere)

        camera.publish(.ending)
        try? await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(recording.reading.camera, .ending)
        XCTAssertTrue(
            recording.reading.cameraStartedHere,
            "a capture that is stopping has not yet ended, and is still this screen's"
        )

        client.sessionAfterRead = .known(try snapshot(state: "stopped"))
        recording.stop()
        await settle(recording)

        XCTAssertEqual(camera.stops, 1)
        XCTAssertFalse(recording.reading.cameraStartedHere)
    }

    /// **A device pause that arrives mid-run reaches the reading.**
    ///
    /// The glasses can pause themselves at any moment — a temple press, or
    /// heat — including while a Start is converging. The subscription is what
    /// carries that, and it is deliberately **not** gated on a sequence being
    /// in flight: a screen that swallowed a camera claim because it was busy
    /// would keep saying the camera is open while the glasses had stopped
    /// delivering.
    func testADevicePauseArrivingDuringAStartReachesTheReading() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()

        let recording = makeConvergingCoordinator(camera: camera, client: client)
        recording.start()
        await waitUntilConverging(recording)
        XCTAssertTrue(recording.reading.cameraStartedHere)

        camera.publish(.devicePaused)
        try? await Task.sleep(nanoseconds: 30_000_000)

        XCTAssertEqual(recording.reading.camera, .devicePaused)
        XCTAssertTrue(
            recording.reading.cameraStartedHere,
            "a device pause holds the session open; it is not the capture ending"
        )
        XCTAssertEqual(
            ObjectMemoryCopy.recordingCameraLine(recording.reading),
            ObjectMemoryCopy.recordingCameraLine(
                ObjectMemoryRecordingReading(
                    phase: recording.phase,
                    camera: .devicePaused,
                    cameraStartedHere: true,
                    cameraIsReachable: true
                )
            )
        )

        // Leave nothing polling behind the test.
        client.sessionAfterRead = .known(try snapshot(state: "stopped"))
        recording.stop()
        await settle(recording)
    }

    /// **The refusal that could never be shown.**
    ///
    /// `startCameraSession()` sets `lastCaptureStartRefusal` synchronously for
    /// four of its five reasons. Camera permission is the fifth and arrives
    /// later: the device session starts, DAT's state observer fires,
    /// `beginCameraStream` finds the permission ungranted, and
    /// `abandonSessionAfterFailedStart` writes the refusal — all after the call
    /// this sequence made had already returned `nil`. So this screen claimed
    /// ownership, converged, and reported `notObserved` twelve seconds later
    /// while the true answer was known, specific, actionable, and had a written
    /// sentence that could never reach a person.
    func testACameraPermissionRefusalThatArrivesLateIsSurfaced() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()

        let recording = makeConvergingCoordinator(camera: camera, client: client)
        recording.start()
        await waitUntilConverging(recording)
        XCTAssertTrue(
            recording.reading.cameraStartedHere,
            "the synchronous read said the start proceeded, which is what it knew"
        )

        // The teardown reaches the publisher first and clears ownership. The
        // convergence loop must still make the check, which is why it captured
        // "this run started a camera" rather than re-reading the flag.
        camera.publish(.unclaimed)
        try? await Task.sleep(nanoseconds: 30_000_000)
        XCTAssertFalse(recording.reading.cameraStartedHere)

        camera.refusal = .cameraPermissionNotGranted
        await settle(recording)

        XCTAssertEqual(recording.phase, .cameraRefused(.cameraPermissionNotGranted))
        XCTAssertNotEqual(
            recording.phase, .notObserved,
            "\"asked for, and not observed\" is not the answer when the answer is known"
        )
        XCTAssertEqual(
            ObjectMemoryCopy.recordingHeadline(recording.reading),
            ObjectMemoryCopy.cameraRefusalLine(.cameraPermissionNotGranted)
        )
        XCTAssertEqual(client.applied, [.start], "the Tower's gate stays open")
    }

    /// A refusal left over from somebody else's Start is not this run's answer.
    ///
    /// `lastCaptureStartRefusal` is cleared at the top of every
    /// `startCameraSession()` and survives until the next one, so on a run that
    /// started no camera it may still hold Home's answer to Home's question.
    /// The convergence re-check is scoped to a capture this run started for
    /// exactly that reason.
    func testACameraRefusalLeftOverFromAnotherScreenIsNotAdopted() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()
        // Home started a capture, and before that Home's own start was refused
        // once. Both facts are still readable here.
        camera.claim = .running
        camera.refusal = .noActiveDevice

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(camera.starts, 0)
        XCTAssertEqual(
            recording.phase, .notObserved,
            "a stale refusal from another screen must not become this run's verdict"
        )
    }

    /// **A 409 on Start**, which until now only `resume` was covered for. Start
    /// is normally idempotent, so a refusal here means a Tower that has said
    /// something this app must report rather than reinterpret — and the camera
    /// is never asked, because a capture running for a gate that will not open
    /// records nothing.
    func testARefusedStartIsReportedAndNeverOpensACapture() async throws {
        let refused = try refusal(action: .start, reason: "not-active")
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .refused(refused)
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(recording.phase, .refused(refused))
        XCTAssertEqual(camera.starts, 0)
        XCTAssertEqual(
            ObjectMemoryCopy.recordingHeadline(recording.reading),
            ObjectMemoryCopy.refusalLine(refused),
            "a refusal must be worded as the answer it is, on start as on resume"
        )
        XCTAssertEqual(
            recording.primaryAction, .start,
            "a Start the Tower refused opened no session, so Stop has nothing to close"
        )
    }

    // MARK: Stop must never be out of reach

    /// **The blocker this whole flag was re-cut for.**
    ///
    /// `isActing` used to cover the entire Start sequence, which ends with a
    /// convergence poll that runs to twelve seconds. Both `.disabled` gates in
    /// `ObjectMemoryWorkspaceView` read it, so for those twelve seconds every
    /// control on the screen — the primary Stop included — was dead, while the
    /// Tower session was open and the glasses camera this screen had just
    /// started was streaming. The only way to stop being recorded was to leave
    /// the screen.
    ///
    /// This asserts the two expressions those gates are written from, rather
    /// than rendering a view: `reading.phase == .unsupported || isActing` for
    /// the primary control, and `!snapshot.supported || isActing` for the
    /// session panel's row.
    func testTheControlsAreNotDisabledWhileConverging() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()

        let recording = makeConvergingCoordinator(camera: camera, client: client)
        recording.start()
        await waitUntilConverging(recording)

        XCTAssertEqual(recording.phase, .waitingToBeFollowed)
        XCTAssertTrue(recording.isSequenceRunning, "the poll really is still running")
        XCTAssertFalse(
            recording.isActing,
            "nothing is being mutated during a convergence poll, and Stop must be reachable"
        )
        XCTAssertFalse(
            recording.reading.phase == .unsupported || recording.isActing,
            "the primary control is what a wearer presses to stop being recorded"
        )
        XCTAssertEqual(
            recording.primaryAction, .stop,
            "and the verb it carries while converging is Stop"
        )

        // Leave nothing running behind the test.
        client.sessionAfterRead = .known(try snapshot(state: "stopped"))
        recording.stop()
        await settle(recording)
    }

    /// A Stop that lands during convergence is accepted, cancels the poll, and
    /// ends the run.
    ///
    /// The cancelled Start does not simply stop where it was told to —
    /// cancellation in Swift is cooperative, so it unwinds through the main
    /// actor and reaches its completion block afterwards. If it were allowed to
    /// clear the flags there it would open the double-tap gate in the middle of
    /// the Stop that replaced it, which is what `run` prevents.
    func testStopDuringConvergenceIsAcceptedAndEndsTheRun() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()

        let recording = makeConvergingCoordinator(camera: camera, client: client)
        recording.start()
        await waitUntilConverging(recording)
        XCTAssertTrue(recording.reading.cameraStartedHere)

        client.sessionAfterRead = .known(try snapshot(state: "stopped"))
        recording.stop()
        await settle(recording)

        XCTAssertEqual(client.applied, [.start, .stop], "the Stop was sent, not dropped")
        XCTAssertEqual(
            camera.stops, 1,
            "the camera this screen started must end with the run that started it"
        )
        XCTAssertEqual(recording.phase, .stopped)
        XCTAssertFalse(recording.phase.isFollowingACapture)
        XCTAssertFalse(recording.isActing)
        XCTAssertFalse(
            recording.isSequenceRunning,
            "the superseded poll must not be left running behind the Stop"
        )
        XCTAssertEqual(recording.primaryAction, .start)
    }

    /// A reading pushed by the workspace's own watch loop is ignored while a
    /// sequence is running — convergence included.
    ///
    /// This is not a corner case. The watch loop polls the same session at
    /// `livenessRefreshInterval`, which is the same three seconds the
    /// convergence poll uses, so a reading lands on that publisher two or three
    /// times during every Start. `resting` turns an `active` session with
    /// nothing following into `notObserved` — the legal, documented shape of
    /// the first seconds after a Start — so an ungated push would print "asked
    /// for, and not observed" while the wait it describes was still running.
    func testAPushedReadingIsIgnoredWhileASequenceIsRunning() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()

        let recording = makeConvergingCoordinator(camera: camera, client: client)
        recording.start()
        await waitUntilConverging(recording)

        client.publish(ObjectMemorySessionState.known(try snapshot(state: "active", following: [])))
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(
            recording.phase, .waitingToBeFollowed,
            "the deadline owns this verdict, not a reading that arrived from elsewhere"
        )

        client.sessionAfterRead = .known(try snapshot(state: "stopped"))
        recording.stop()
        await settle(recording)
    }

    /// A reading that says a request is in flight leaves the last known phase
    /// alone, and a reading that says nothing has been read at all is `idle`
    /// rather than `stopped` — silence is not an answer.
    func testAnInFlightReadingHoldsAndAnUnreadOneIsIdle() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)
        XCTAssertEqual(recording.phase, .remembering)

        client.publish(ObjectMemorySessionState.working)
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(
            recording.phase, .remembering,
            "whatever was last true is still the best thing known; a flash of nothing is worse"
        )

        client.publish(ObjectMemorySessionState.unread)
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(
            recording.phase, .idle,
            "never read is not the same as stopped, and must not be drawn as it"
        )
    }

    // MARK: Pause is a Tower verb only

    /// Pause detaches a producer. It cannot pause a DAT stream, because no such
    /// call exists — so a running capture keeps running, and the reading has to
    /// say both things at once.
    func testPauseLeavesTheCameraRunningAndSaysSo() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)

        client.sessionAfterRead = .known(try snapshot(state: "paused"))
        recording.pause()
        await settle(recording)

        XCTAssertEqual(recording.phase, .paused)
        XCTAssertEqual(camera.stops, 0, "nothing in this app can pause a DAT stream")
        XCTAssertEqual(camera.claim, .running)
        XCTAssertNotEqual(
            ObjectMemoryCopy.recordingCameraLine(recording.reading),
            ObjectMemoryCopy.recordingCameraLine(
                ObjectMemoryRecordingReading(
                    phase: .paused,
                    camera: .unclaimed,
                    cameraStartedHere: false,
                    cameraIsReachable: true
                )
            ),
            "a paused session beside a live camera must not read like one beside no camera"
        )
    }

    /// The reproduced `SIGTERM` failure: the Tower reports the pause as
    /// honoured and still names a capture in `following`. Liveness is read from
    /// `following`, so the phase must keep claiming the writing has not ended.
    func testAPauseThatDidNotStopTheProducerKeepsClaimingLiveness() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(
            try snapshot(state: "paused", following: ["22e9d4289cb440fb"])
        )
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.pause()
        await settle(recording)

        XCTAssertEqual(recording.phase, .stillFollowing(after: .pause))
        XCTAssertTrue(recording.phase.isFollowingACapture)
    }

    // MARK: Finishing

    /// Stopping is usually the moment somebody wants to see what was written.
    func testStopRefreshesTheRecords() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "stopped"))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        let refreshes = RefreshCounter()
        recording.refreshRecords = { refreshes.bump() }
        recording.stop()
        await settle(recording)

        XCTAssertEqual(refreshes.count, 1)
    }

    /// **Only Stop refreshes.** Pause and Resume leave the session where a
    /// person can still add to it, so re-asking a question in the middle of a
    /// run would replace the reader's answer with a partial one and scroll
    /// their list out from under them. Stop is the moment the run is over and
    /// looking at what was written is the reason for pressing it.
    func testPauseAndResumeDoNotRefreshTheRecords() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "paused"))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        let refreshes = RefreshCounter()
        recording.refreshRecords = { refreshes.bump() }

        recording.pause()
        await settle(recording)
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        recording.resume()
        await settle(recording)

        XCTAssertEqual(refreshes.count, 0)
        XCTAssertEqual(client.applied, [.pause, .resume])
    }

    /// **A Stop the Tower did not confirm still ends the capture, and still
    /// refreshes.**
    ///
    /// Three things are pinned here, and the first is the one that matters. The
    /// `POST` has already gone; leaving the glasses streaming because the
    /// answer did not come back is the worst of the four available outcomes,
    /// and it is the one a "only stop the camera if the Tower said yes" guard
    /// would produce.
    ///
    /// The phase drops the liveness claim rather than reporting a stop, because
    /// this app genuinely cannot tell whether the producer detached. And the
    /// refresh is unconditional: it re-asks a question over HTTP, the records
    /// route is not the session route, and whatever was written before the
    /// Tower went quiet is exactly what somebody pressing Stop wants to see.
    func testAStopThatCouldNotBeConfirmedStillEndsTheCapture() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(following: ["22e9d4289cb440fb"]))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        recording.start()
        await settle(recording)
        XCTAssertTrue(recording.reading.cameraStartedHere)

        let failure = CartridgeFailure(
            kind: .transport, message: "The Tower did not answer: connection lost"
        )
        client.sessionAfterRead = .failed(failure)
        let refreshes = RefreshCounter()
        recording.refreshRecords = { refreshes.bump() }
        recording.stop()
        await settle(recording)

        XCTAssertEqual(
            camera.stops, 1,
            "a Stop that could not be confirmed must not leave the glasses streaming"
        )
        XCTAssertFalse(recording.reading.cameraStartedHere)
        XCTAssertEqual(recording.phase, .cannotTell(failure))
        XCTAssertFalse(
            recording.phase.isFollowingACapture,
            "a stop that could not be read is not a claim that anything is recording"
        )
        XCTAssertEqual(refreshes.count, 1)
    }

    // MARK: A build with no camera

    /// Release, and every preview. The Tower half works and the camera half
    /// reports that it cannot be reached rather than pretending it can.
    func testWithNoCameraTheTowerHalfStillWorksAndSaysSo() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))

        let recording = makeCoordinator(camera: nil, client: client)
        recording.start()
        await settle(recording)

        XCTAssertEqual(client.applied, [.start])
        XCTAssertFalse(recording.reading.cameraIsReachable)
        XCTAssertEqual(
            ObjectMemoryCopy.recordingCameraLine(recording.reading),
            ObjectMemoryCopy.recordingCameraNotInThisBuild
        )
    }

    // MARK: The two cadences

    /// The convergence poll and the workspace's liveness watch must stay the
    /// same interval. They are written out separately because one of them is
    /// isolated to the main actor and the other is a default argument; this is
    /// what keeps them from drifting into a screen that polls at one rate and
    /// converges at another.
    func testTheConvergenceCadenceMatchesTheWatchLoop() {
        XCTAssertEqual(
            ObjectMemoryRecordingCoordinator.convergenceInterval,
            ObjectMemoryViewModel.livenessRefreshInterval
        )
    }

    /// The primary control says Stop for every phase in which a session exists
    /// on the Tower — including `notObserved`, where offering Start would leave
    /// no way to clear a session that is open and doing nothing.
    ///
    /// **`cameraRefused` is that same shape and used to offer Start.** Every
    /// branch that produces it leaves the Tower session `active` on purpose,
    /// because the session is a gate rather than a recording and tearing it
    /// down for a refused camera would throw away correct work. So the wearer
    /// was offered the one verb that changes nothing, and the open session had
    /// no control that could close it.
    func testTheStopControlIsOfferedWheneverASessionExists() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
        let camera = FakeCaptureOwner()

        let recording = makeCoordinator(camera: camera, client: client)
        XCTAssertEqual(recording.primaryAction, .start)

        recording.start()
        await settle(recording)
        XCTAssertEqual(recording.phase, .notObserved)
        XCTAssertEqual(recording.primaryAction, .stop)

        client.sessionAfterRead = .known(try snapshot(state: "stopped"))
        recording.stop()
        await settle(recording)
        XCTAssertEqual(recording.primaryAction, .start)
    }

    /// Every one of the five camera refusals leaves a session open on the
    /// Tower, so every one of them offers Stop.
    ///
    /// Driven through the coordinator rather than asserted against a phase
    /// built by hand, because the claim being made is about a real Start whose
    /// Tower half stood while its camera half did not.
    func testACameraRefusalStillOffersTheControlThatClosesTheSession() async throws {
        // Named `refused` rather than `refusal`, which is the name of this
        // suite's 409 helper one scope up.
        for refused: CaptureStartRefusal in [
            .alreadyRunning, .deviceHasPausedCapture, .captureIsShuttingDown,
            .noActiveDevice, .cameraPermissionNotGranted,
            .datRefused("the device session could not be created"),
        ] {
            let client = StubObjectMemoryClient()
            client.sessionAfterRead = .known(try snapshot(state: "active", following: []))
            let camera = FakeCaptureOwner()
            camera.refusal = refused

            let recording = makeCoordinator(camera: camera, client: client)
            recording.start()
            await settle(recording)

            XCTAssertEqual(recording.phase, .cameraRefused(refused))
            XCTAssertEqual(client.applied, [.start], "the Tower's half stands")
            XCTAssertEqual(
                recording.primaryAction, .stop,
                """
                the session is open on the Tower and nothing is following it; \
                Start would change nothing and would leave no way to close it
                """
            )
        }
    }
}

// MARK: - Liveness belongs to the session claiming it

/// **A leftover producer must not light up a session that started nothing.**
///
/// The Tower's own contract carried this as a warning from 2026-08-27 until it
/// was fixed on 2026-08-29: `following` is supervisor-scoped, so a producer
/// that ignored `terminate()` on an earlier Stop stays registered and is
/// reported under the *next* session's id. Against this app's own rule — draw
/// liveness from `following`, never from `state` — that produced a false
/// positive: a brand-new session that attached nothing rendered as
/// "remembering".
///
/// The fix is additive on both sides. `following` keeps its full breadth,
/// because an un-killable producer is what a Stop failing open looks like and
/// it must stay visible; `following_this_session` is the scoped subset, and it
/// is what a "you are being recorded" claim is drawn from.
/// `@MainActor` for the same reason `ObjectMemoryRecordingTests` is: the last
/// case here builds a coordinator, and that type is main-actor isolated.
@MainActor
final class ObjectMemoryScopedLivenessTests: XCTestCase {

    private func snapshot(
        state: String,
        following: [String],
        followingThisSession: [String]? = nil
    ) throws -> CartridgeSessionSnapshot {
        try XCTUnwrap(
            CartridgeSessionDecoder.snapshot(
                from: ObjectMemoryFixtures.session(
                    state: state,
                    following: following,
                    followingThisSession: followingThisSession
                )
            )
        )
    }

    /// **The reproduced false positive.**
    func testASessionThatStartedNothingIsNotRenderedAsRemembering() throws {
        let reading = try snapshot(
            state: "active",
            following: ["a-leftover-producer"],
            followingThisSession: []
        )

        XCTAssertFalse(
            reading.isFollowingACapture,
            """
            this session attached nothing; the capture in `following` belongs \
            to an earlier session whose Stop did not reach its producer
            """
        )
        XCTAssertEqual(reading.recordingsThisControlDidNotStart, ["a-leftover-producer"])
    }

    /// And the leftover is still reported, because the alternative is silence
    /// about a camera that is recording.
    func testALeftoverProducerIsStillSaidOutLoud() throws {
        let reading = try snapshot(
            state: "active",
            following: ["a-leftover-producer"],
            followingThisSession: []
        )

        let line = try XCTUnwrap(ObjectMemoryCopy.leftoverProducerLine(reading))
        XCTAssertFalse(line.isEmpty)
        XCTAssertFalse(
            ObjectMemoryCopy.livenessLine(reading).contains("is writing into this"),
            "the liveness sentence must not claim a producer this session did not start"
        )
    }

    func testAProducerThisSessionStartedIsRememberingNormally() throws {
        let reading = try snapshot(
            state: "active",
            following: ["cap-1"],
            followingThisSession: ["cap-1"]
        )

        XCTAssertTrue(reading.isFollowingACapture)
        XCTAssertNil(
            ObjectMemoryCopy.leftoverProducerLine(reading),
            "there is nothing left over when the only producer is this session's"
        )
    }

    /// The count in the sentence must come from the same list the sentence
    /// branched on. Counting `following` while branching on the scoped list
    /// would let a leftover inflate a number describing this session.
    func testTheLivenessSentenceCountsOnlyThisSessionsCaptures() throws {
        let reading = try snapshot(
            state: "active",
            following: ["cap-1", "a-leftover-producer"],
            followingThisSession: ["cap-1"]
        )

        XCTAssertTrue(ObjectMemoryCopy.livenessLine(reading).contains("a recording"))
        XCTAssertFalse(ObjectMemoryCopy.livenessLine(reading).contains("2 recordings"))
    }

    /// **The alarm stays wide.** A Stop that did not stop is still a Stop that
    /// did not stop, whichever session started the producer.
    func testTheContradictionAlarmReadsTheWideField() throws {
        let stoppedWithALeftover = try snapshot(
            state: "stopped",
            following: ["a-leftover-producer"],
            followingThisSession: []
        )

        XCTAssertFalse(stoppedWithALeftover.isFollowingACapture)
        XCTAssertTrue(
            stoppedWithALeftover.intentContradictsLiveness,
            """
            scoping this alarm is the one narrowing that would make it useless: \
            something is recording, a person asked for it to stop, and this \
            screen is the only place that will say so
            """
        )
        XCTAssertNotNil(
            ObjectMemoryCopy.livenessContradictsIntentLine(stoppedWithALeftover)
        )
    }

    /// **`nil` is not `[]`.** A Tower that never sends the field has not said
    /// this session is following nothing; it has said nothing at all. Reporting
    /// a Tower with a live producer as one without would be the same class of
    /// error in the more dangerous direction.
    func testATowerThatDoesNotSendTheFieldFallsBackRatherThanGoingQuiet() throws {
        let older = try snapshot(state: "active", following: ["cap-1"])

        XCTAssertNil(older.followingThisSession)
        XCTAssertTrue(older.isFollowingACapture)
        XCTAssertEqual(
            older.recordingsThisControlDidNotStart,
            [],
            "nothing can be called a leftover when nothing is scoped"
        )
    }

    func testAnEmptyScopedListIsNotTheSameAsAnAbsentOne() throws {
        let absent = try snapshot(state: "active", following: ["cap-1"])
        let empty = try snapshot(
            state: "active", following: ["cap-1"], followingThisSession: []
        )

        XCTAssertTrue(absent.isFollowingACapture)
        XCTAssertFalse(empty.isFollowingACapture)
    }

    /// The composed lifecycle must inherit the distinction rather than
    /// re-deriving it: `ObjectMemoryRecordingCoordinator` reads
    /// `snapshot.isFollowingACapture`, so scoping it once scopes the phase too.
    func testTheComposedPhaseDoesNotRememberOnALeftover() async throws {
        let client = StubObjectMemoryClient()
        client.sessionAfterRead = .known(
            try snapshot(
                state: "active",
                following: ["a-leftover-producer"],
                followingThisSession: []
            )
        )
        let camera = FakeCaptureOwner()
        let coordinator = ObjectMemoryRecordingCoordinator(
            camera: camera, client: client,
            interval: .milliseconds(1), budget: .milliseconds(30)
        )

        coordinator.start()
        var turns = 0
        // `isSequenceRunning` and not `isActing`: the second now covers only
        // the mutating step and goes false when the convergence poll begins,
        // so waiting on it here would assert against a phase still in motion.
        while coordinator.isSequenceRunning && turns < 500 {
            try? await Task.sleep(nanoseconds: 2_000_000)
            turns += 1
        }
        try? await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertFalse(
            coordinator.phase.isFollowingACapture,
            "a leftover producer must not become this screen's remembering claim"
        )
    }
}

// MARK: - Two stores, two lifetimes

/// **A record's object picture and its context picture no longer live and die
/// together.**
///
/// `/crop` and `/frame` used to be two renders of one file in the capture
/// store. Object Memory now owns a small privacy-filtered crop per record,
/// under its own retention, so `/crop` keeps answering after the recording
/// behind it has been deleted while `/frame` honestly 410s. Two additive
/// fields carry that — `frame_available` and `imagery_source` — and the
/// contract identifier is deliberately unchanged, because no existing field
/// changed shape or meaning.
///
/// Three things have to hold at once, and each of them was a way to mislead a
/// wearer:
///
/// 1. An **older Tower**, which sends neither field, behaves exactly as it did.
/// 2. The automatic fallback to `/frame` must not walk away from a picture that
///    is being kept, towards one that is gone.
/// 3. The retention sentence must follow the store, because the two lifetimes
///    are opposites and they are said in the same slot on the same screen.
@MainActor
final class ObjectMemoryTwoStoreImageryTests: XCTestCase {

    private let onRead = "applied-on-read-the-stored-frame-is-unchanged"
    private let beforeWritten = "applied-before-this-file-was-written"

    private func description(
        subjectObscured: Any = 0.0,
        frameAvailable: Bool? = nil,
        imagerySource: String? = nil,
        filterMeans: Any = "applied-on-read-the-stored-frame-is-unchanged",
        imageryRetention: Any = "capture-side"
    ) throws -> ObjectMemoryImageryDescription {
        try XCTUnwrap(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(
                    subjectObscured: subjectObscured,
                    filterMeans: filterMeans,
                    frameAvailable: frameAvailable,
                    imagerySource: imagerySource,
                    imageryRetention: imageryRetention
                ),
                statusCode: 200
            )
        )
    }

    private func settle() async throws {
        try await Task.sleep(nanoseconds: 50_000_000)
    }

    // MARK: An older Tower is unchanged

    /// **The compatibility floor.** Neither field on the wire, and every
    /// derived answer is the one this build gave before the fields existed.
    func testAnOlderTowerThatSendsNeitherFieldBehavesAsItAlwaysDid() throws {
        let older = try description(subjectObscured: 0.42)

        XCTAssertNil(
            older.frameAvailable,
            "an absent field means the Tower did not say, which is not the same as no"
        )
        XCTAssertNil(older.imagerySource)
        XCTAssertTrue(
            older.frameCanBeAskedFor,
            "an unknown answer is a reason to ask the Tower, not to withhold a picture"
        )
        XCTAssertEqual(
            older.preferredKind, .frame,
            "the obscured-subject fallback is exactly the expression it always was"
        )
        XCTAssertEqual(
            ObjectMemoryCopy.pictureRetentionLine(older),
            ObjectMemoryCopy.unnamedSourceRetentionLine,
            "a Tower that names no store gets no claim about which retention holds this"
        )
    }

    /// **`nil` is not `false`.** Reading a missing field as "the frame is gone"
    /// would have made every older Tower report every context view as expired
    /// and would have taken a working control off the screen.
    func testAnAbsentFrameAvailabilityIsNotAnAbsentFrame() throws {
        let unknown = try description(subjectObscured: 0.42)
        let gone = try description(subjectObscured: 0.42, frameAvailable: false)
        let there = try description(subjectObscured: 0.42, frameAvailable: true)

        XCTAssertNil(unknown.frameAvailable)
        XCTAssertEqual(gone.frameAvailable, false)
        XCTAssertEqual(there.frameAvailable, true)
        XCTAssertTrue(unknown.frameCanBeAskedFor)
        XCTAssertFalse(gone.frameCanBeAskedFor)
        XCTAssertTrue(there.frameCanBeAskedFor)
    }

    // MARK: The fallback

    /// **The fallback must not walk away from the picture that is kept.**
    ///
    /// `preferredKind` returned `.frame` for any obscured subject. Once `/crop`
    /// started outliving the recording behind it, that sent a wearer with a
    /// partly-filled subject to a route that answers 410 — so they read "the
    /// memory is kept and the picture is gone" while the Tower was holding a
    /// crop for them.
    func testAFrameTheTowerSaysIsGoneIsNotPreferredOverAKeptCrop() throws {
        let obscuredAndGone = try description(
            subjectObscured: 0.42,
            frameAvailable: false,
            imagerySource: "object-memory-keyframe",
            filterMeans: beforeWritten
        )

        XCTAssertTrue(obscuredAndGone.subjectIsBehindAFill)
        XCTAssertEqual(
            obscuredAndGone.preferredKind, .crop,
            "falling back to a frame that is gone is worse than the crop it left"
        )
        XCTAssertNotNil(
            ObjectMemoryCopy.subjectObscuredLine(obscuredAndGone, kind: .crop),
            """
            the other half of the contract's instruction still applies: say the \
            subject is behind a fill
            """
        )
    }

    /// The fallback is intact wherever the frame can still be asked for —
    /// stated, and unknown.
    func testTheFallbackSurvivesWhereverTheFrameCanBeAskedFor() throws {
        XCTAssertEqual(
            try description(subjectObscured: 0.42, frameAvailable: true).preferredKind, .frame
        )
        XCTAssertEqual(try description(subjectObscured: 0.42).preferredKind, .frame)
        XCTAssertEqual(
            try description(subjectObscured: 0.0, frameAvailable: false).preferredKind, .crop,
            "nothing is obscured, so the crop was always the answer"
        )
    }

    /// End to end through the loader: no `/frame` request is ever made for a
    /// record whose frame the Tower has said is gone.
    func testTheLoaderNeverAsksForAFrameTheTowerSaysIsGone() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(
            try description(
                subjectObscured: 0.42,
                frameAvailable: false,
                imagerySource: "object-memory-keyframe",
                filterMeans: beforeWritten
            )
        )
        client.stubbedPicture = .picture(Data([0xFF, 0xD8, 0xFF]))

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()

        XCTAssertEqual(loader.kind, .crop)
        XCTAssertEqual(client.picturesAsked.count, 1)
        XCTAssertEqual(client.picturesAsked.first?.1, ObjectMemoryImageryKind.crop)
        XCTAssertFalse(
            client.picturesAsked.contains { $0.1 == .frame },
            "a request that can only 410 is not worth making on a wearer's behalf"
        )
    }

    /// The gate that made the Tower-side fix necessary: the loader asks for no
    /// bytes at all unless `available` is true, so `available` describing a
    /// `/frame` render meant a kept crop could never be fetched. Pinned here so
    /// the coupling is written down rather than rediscovered.
    func testNoBytesAreAskedForWhenTheDescriptionSaysThereAreNone() async throws {
        let client = StubObjectMemoryClient()
        client.stubbedImagery = .described(
            try XCTUnwrap(
                ObjectMemoryImageryDecoder.description(
                    from: ObjectMemoryFixtures.imagery(
                        available: false,
                        reason: "imagery-no-longer-available",
                        filter: NSNull(),
                        frameAvailable: false
                    ),
                    statusCode: 410
                )
            )
        )

        let loader = ObjectMemoryPictureLoader(client: client, observationID: "9f2c41b7ad0e6538")
        loader.load()
        try await settle()

        XCTAssertTrue(client.picturesAsked.isEmpty)
        if case .noPicture = loader.phase {} else {
            XCTFail("a described absence is a sentence, not a failure: \(loader.phase)")
        }
    }

    // MARK: The two filter meanings

    /// A keyframe payload decodes. It could not before: `filter_means` was
    /// checked against one constant, so the store whose filter runs *before*
    /// persistence would have failed the parse and rendered as an unreadable
    /// answer.
    func testAKeyframePayloadDecodesUnderTheSameContract() throws {
        let keyframe = try description(
            imagerySource: "object-memory-keyframe",
            filterMeans: beforeWritten,
            imageryRetention: "object-memory"
        )

        XCTAssertEqual(keyframe.contract, ObjectMemoryImageryContract.identifier)
        XCTAssertEqual(keyframe.imagerySource, ObjectMemoryImagerySource.objectMemoryKeyframe)
        XCTAssertEqual(keyframe.filterMeans, beforeWritten)
    }

    /// Widening a membership test is not dropping it. A third meaning is still
    /// refused, because the sentence beside the picture would be describing a
    /// transformation nobody here has read about.
    func testAFilterMeaningThisBuildHasNeverHeardOfIsStillRefused() {
        XCTAssertNil(
            ObjectMemoryImageryDecoder.description(
                from: ObjectMemoryFixtures.imagery(filterMeans: "applied-sometimes"),
                statusCode: 200
            )
        )
        XCTAssertNil(
            ObjectMemoryImageryDecoder.routes(
                from: ObjectMemoryFixtures.imageryRoutes(filterMeans: "applied-sometimes")
            )
        )
    }

    /// The envelope's route block accepts either meaning, so a Tower serving
    /// keyframes does not lose its imagery routes on decode.
    func testTheRouteBlockAcceptsEitherFilterMeaning() throws {
        for means in [onRead, beforeWritten] {
            XCTAssertNotNil(
                ObjectMemoryImageryDecoder.routes(
                    from: ObjectMemoryFixtures.imageryRoutes(filterMeans: means)
                ),
                "a Tower that filters before persistence still has routes"
            )
        }
    }

    /// The "when" clause follows `filter_means`, and the stronger claim is only
    /// made when the payload makes it.
    func testTheFilterSentenceFollowsWhatTheFilterMeaningSays() throws {
        let read = ObjectMemoryCopy.filterLine(try description(filterMeans: onRead))
        let written = ObjectMemoryCopy.filterLine(
            try description(imagerySource: "object-memory-keyframe", filterMeans: beforeWritten)
        )

        XCTAssertNotEqual(read, written)
        XCTAssertTrue(read.lowercased().contains("the stored frame is unchanged"))
        XCTAssertTrue(written.lowercased().contains("no unfiltered copy"))
        XCTAssertFalse(
            read.lowercased().contains("no unfiltered copy"),
            "a capture frame's unfiltered original is on disk; this app may not say otherwise"
        )
        for sentence in [read, written] {
            XCTAssertTrue(
                sentence.contains("display filter"),
                "the filter keeps its name under both meanings"
            )
            for word in [
                "redact", "anonymis", "anonymiz", "privacy-safe", "privacy safe",
                "de-identif", "deidentif", "scrubbed", "sanitis", "sanitiz",
            ] {
                XCTAssertFalse(sentence.lowercased().contains(word))
            }
        }
    }

    // MARK: Retention, per store

    /// **Three sentences, because there are three answers.** An owned keyframe
    /// goes when the record does and outlives the recording; a capture frame
    /// goes when capture-side retention says so and can vanish while the record
    /// stays; a Tower that names neither gets no claim about either.
    func testTheRetentionSentenceFollowsTheStoreThatServedTheBytes() throws {
        let keyframe = ObjectMemoryCopy.pictureRetentionLine(
            try description(
                imagerySource: "object-memory-keyframe",
                filterMeans: beforeWritten,
                imageryRetention: "object-memory"
            )
        )
        let captureFrame = ObjectMemoryCopy.pictureRetentionLine(
            try description(imagerySource: "capture-frame")
        )
        let unnamed = ObjectMemoryCopy.pictureRetentionLine(try description())

        XCTAssertEqual(Set([keyframe, captureFrame, unnamed]).count, 3)
        XCTAssertEqual(keyframe, ObjectMemoryCopy.keyframeRetentionLine)
        XCTAssertEqual(captureFrame, ObjectMemoryCopy.captureFrameRetentionLine)
        XCTAssertEqual(unnamed, ObjectMemoryCopy.unnamedSourceRetentionLine)

        XCTAssertTrue(keyframe.lowercased().contains("kept by this memory itself"))
        XCTAssertTrue(
            captureFrame.lowercased().contains("neither sets nor enforces"),
            "this cartridge does not own capture-side retention and must not imply it does"
        )
        XCTAssertTrue(unnamed.lowercased().contains("did not say"))

        // The one thing true of all three.
        for sentence in [keyframe, captureFrame, unnamed] {
            XCTAssertTrue(sentence.contains("not held on this phone"))
        }
    }

    /// The sentence is written from `imagery_source` rather than from
    /// `imagery_retention`, which is a label rather than a lifetime. A Tower
    /// that renames the label must not change what a wearer is told.
    func testTheRetentionSentenceIgnoresTheRetentionLabel() throws {
        XCTAssertEqual(
            ObjectMemoryCopy.pictureRetentionLine(
                try description(imagerySource: "capture-frame", imageryRetention: "something-else")
            ),
            ObjectMemoryCopy.captureFrameRetentionLine
        )
    }

    /// A store this build has never heard of gets the sentence that claims
    /// nothing, rather than the nearest one that does.
    func testAnUnrecognisedSourceClaimsNothingAboutRetention() throws {
        XCTAssertEqual(
            ObjectMemoryCopy.pictureRetentionLine(
                try description(imagerySource: "some-third-store")
            ),
            ObjectMemoryCopy.unnamedSourceRetentionLine
        )
        XCTAssertEqual(
            try description(imagerySource: "some-third-store").imagerySource,
            ObjectMemoryImagerySource(rawValue: "some-third-store"),
            "an unknown source survives the decode rather than failing the parse"
        )
    }

    // MARK: The sentences beside the picture

    /// Every sentence these two stores can produce, through the same
    /// forbidden-phrase discipline the rest of the picture copy gets. A
    /// sentence next to a first-person photograph is the most dangerous
    /// sentence this app writes, and that does not change because it is about
    /// retention.
    func testEveryTwoStoreSentenceMakesNoClaimItCannotSupport() throws {
        let classes = ["laptop", "cell phone"]
        var forbidden = [
            "still there", "is still", "right now", "at the moment", "on the map",
            "last known location", "location of", "we know where", "you left",
            "where you left", "go and get", "does not exist", "there is no ",
            "you have no ", "you do not have", "found your", "no results",
            "last seen in session", "seen in session",
        ]
        for objectClass in classes {
            forbidden.append(contentsOf: [
                "your \(objectClass)", "my \(objectClass)", "the \(objectClass) is",
                "\(objectClass) is in", "\(objectClass) is at", "\(objectClass) is on",
                "\(objectClass) is still", "\(objectClass) exists",
            ])
        }
        forbidden.append(contentsOf: [
            "redact", "anonymis", "anonymiz", "privacy-safe", "privacy safe",
            "de-identif", "deidentif", "scrubbed", "sanitis", "sanitiz",
        ])

        var sentences: [String] = []
        for source in [nil, "capture-frame", "object-memory-keyframe", "some-third-store"] {
            for frameAvailable in [nil, true, false] as [Bool?] {
                for obscured in [0.0, 0.42] {
                    let means = source == "object-memory-keyframe" ? beforeWritten : onRead
                    let described = try description(
                        subjectObscured: obscured,
                        frameAvailable: frameAvailable,
                        imagerySource: source,
                        filterMeans: means
                    )
                    for kind in ObjectMemoryImageryKind.allCases {
                        sentences.append(
                            contentsOf: ObjectMemoryCopy.everyString(for: described, kind: kind)
                        )
                    }
                }
            }
        }

        XCTAssertTrue(
            sentences.contains(ObjectMemoryCopy.wholeFrameIsGoneLine),
            "the gone-frame sentence must actually be generated, not merely written"
        )
        XCTAssertTrue(sentences.contains(ObjectMemoryCopy.keyframeRetentionLine))
        XCTAssertTrue(sentences.contains(ObjectMemoryCopy.captureFrameRetentionLine))
        XCTAssertTrue(sentences.contains(ObjectMemoryCopy.unnamedSourceRetentionLine))

        for sentence in sentences {
            let lowered = sentence.lowercased()
            XCTAssertFalse(lowered.isEmpty, "an empty string reached the screen")
            for phrase in forbidden {
                XCTAssertFalse(
                    lowered.contains(phrase),
                    "copy claims more than the sensor supports (\"\(phrase)\") in: \(sentence)"
                )
            }
        }
    }
}
