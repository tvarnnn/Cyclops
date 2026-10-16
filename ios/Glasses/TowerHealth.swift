//
//  TowerHealth.swift
//  Glasses
//

import Foundation

// MARK: - The Tower's own state, as reported by GET /health

/// Something the Tower's health report may or may not have mentioned.
///
/// ## Why three cases where a `nil` would fit
///
/// `GET /health` makes genuinely different statements about a subsystem, and
/// the difference is the whole reason this type exists:
///
/// - The key is **missing**. This Tower's health route says nothing about the
///   subsystem at all — an older build, or one whose shape has moved on. We
///   were not told.
/// - The key is present and **null**. The Tower is telling us something
///   specific: nothing is registered. `tower/tower/routes/health.py` is
///   explicit that `capture: null` means no recorder exists, "which is
///   different from a registered recorder that is idle", and that collapsing
///   the two would make "we are definitely not recording" indistinguishable
///   from "we are armed and one `stream_start` away".
/// - The key is present and carries an **object**, which may still leave any
///   individual field unsaid.
///
/// Folding the first two into one `nil` would answer a privacy question with a
/// shrug dressed up as a fact. `tower/docs/contracts/IOS-TO-TOWER-RECONCILIATION.md` §1.8 is a worked
/// example of the same failure in the other direction: reading the obvious
/// field there would have reported a confident, wrong **0**.
nonisolated enum TowerReported<Value: Equatable & Sendable>: Equatable, Sendable {
    /// The report did not mention this at all.
    case unreported
    /// The report mentioned it and said there is nothing there.
    case absent
    /// The report carried a value, whose own fields may still be unsaid.
    case present(Value)
    /// The report carried something this app could not read as this shape.
    /// Not the same as silence: the Tower spoke and we failed to understand.
    case unreadable

    var value: Value? {
        if case .present(let value) = self { return value }
        return nil
    }
}

/// The Tower's dataset recorder, as `/health` reports it.
///
/// ## Why every field is optional
///
/// Because the Tower's own error path sends `{"armed": true, "error":
/// "unavailable"}` — armed, and nothing else known — and because an omitted
/// `recording` must never be read as "not recording" on the one screen a
/// person would consult to find out. This is the project's `nil ≠ 0` rule at
/// the point where it costs the most to get wrong: while the recorder is
/// armed, the Tower fsyncs every frame it receives to disk **unredacted**
/// (`tower/tower/capture.py` declares `retains_raw_imagery: true`,
/// `redaction: "none"`, and tags the manifest `raw-imagery`,
/// `first-person`).
///
/// ## There is deliberately no way to arm it from here
///
/// `TOWER_CAPTURE_ROOT` arms the recorder at Tower start-up and
/// `stream_start`/`stream_stop` bound each recording; the reconciliation
/// document lists arming from iOS as **BLOCKED**, with no route to call. A
/// control on this side would be a fabricated capability, so this type is a
/// reading and nothing else.
nonisolated struct TowerCaptureState: Equatable, Sendable {
    /// A recorder is registered and will write whenever a stream is running.
    let armed: Bool?
    /// Frames are being written **right now**.
    let recording: Bool?
    /// The recording the counts below belong to.
    ///
    /// `TowerReported` rather than `String?`, because **three** different
    /// things arrive in this one field and only two of them are a `nil`:
    ///
    /// - `.present(id)` — a recording is open and this is its id.
    /// - `.absent` — the key arrived carrying `null`. That is a *positive*
    ///   statement: `tower/tower/routes/health.py` emits `capture_id` on every
    ///   success, and sends `null` in exactly one situation — a recorder is
    ///   registered and has not opened a recording yet.
    /// - `.unreported` — the key was not there at all, which happens only on
    ///   the Tower's `.error` branch (`{"armed": true, "error": "unavailable"}`)
    ///   and means the Tower could not read its own recorder.
    ///
    /// Flattening the middle two into one Optional is the same
    /// `unreported`-vs-`absent` conflation this enum exists to prevent, one
    /// level down at the field: it renders "The Tower did not say" over an
    /// answer the Tower gave clearly.
    ///
    /// `framesWritten` and `bytesWritten` below stay plain Optionals on
    /// purpose. They are `0` — a real, sent zero — in that same no-recording
    /// case, and absent only on the `.error` branch, so for them one `nil`
    /// carries exactly one meaning.
    let captureID: TowerReported<String>
    /// Frames written to disk in the latest recording. `nil` is "the Tower did
    /// not say"; `0` is the Tower saying zero, and the two must not be drawn
    /// the same way.
    let framesWritten: Int?
    let bytesWritten: Int?
    /// The Tower could not read its own recorder's state. It still knows the
    /// recorder is registered, which is why `armed` can be true beside this.
    let error: String?
}

/// Whether anything on the Tower is turning captures into anything.
///
/// `enabled: false` means nothing is configured to follow a capture;
/// `enabled: true` with no workers is correct between walks and wrong during
/// one. Carried here because it answers "why isn't World Builder changing?"
/// from the phone, which was previously only answerable by noticing that no
/// world directory had appeared on a machine nobody was looking at.
nonisolated struct TowerCaptureWorkers: Equatable, Sendable {
    let enabled: Bool?
    /// How many workers are running. `nil` when the Tower sent no `workers`
    /// list at all — an empty list it *did* send is a real `0`.
    let workerCount: Int?
    let error: String?
}

/// One `GET /health` answer.
///
/// Every field is optional and nothing is defaulted, because this type's only
/// job is to say what the Tower said. A build that omits a field has not
/// claimed anything about it.
nonisolated struct TowerHealth: Equatable, Sendable {
    let status: String?
    let service: String?
    let version: String?
    /// The module runtime's state, e.g. `active`.
    let moduleState: String?
    /// Which module is loaded, e.g. `experimental-cv`.
    let moduleID: String?
    let capture: TowerReported<TowerCaptureState>
    let captureWorkers: TowerReported<TowerCaptureWorkers>
}

/// Why a health read did not produce an answer.
///
/// The same split `ObjectMemoryFetchError` keeps, minus the cases that have no
/// meaning here: `/health` has no contract field to disagree about and no 404
/// that means anything but "this is not a Tower". Two cases, and they are not
/// interchangeable — an answer that arrived and could not be read is a
/// disagreement about the answer, not a failure to get one, and only the
/// second is evidence about the network.
nonisolated enum TowerHealthFetchError: Error, Equatable {
    /// The answer arrived and could not be read as a health report.
    case undecodable
    /// The request did not complete, or the Tower refused it.
    case transport(String)
}

/// Turns a `/health` body into what the Tower said, and nothing more.
///
/// Split out from the HTTP client so the shapes that actually matter — a
/// missing `capture`, an explicit null, an error object, a count the Tower
/// omitted — are testable against real payloads without standing up a server.
nonisolated enum TowerHealthDecoder {

    /// - Throws: `TowerHealthFetchError.undecodable` when the bytes are not a
    ///   JSON object. A *well-formed* object that mentions nothing is not an
    ///   error: it decodes to a health report in which everything is unsaid,
    ///   which is the truth about it.
    static func health(from data: Data) throws -> TowerHealth {
        // Parsed in its own `do` because `jsonObject(with:)` throws on a
        // malformed body rather than returning something the cast rejects,
        // and that throw must not be relabelled as a transport failure — the
        // Tower answered.
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw TowerHealthFetchError.undecodable
        }
        guard let json = parsed as? [String: Any] else {
            throw TowerHealthFetchError.undecodable
        }
        return health(from: json)
    }

    static func health(from json: [String: Any]) -> TowerHealth {
        TowerHealth(
            status: json["status"] as? String,
            service: json["service"] as? String,
            version: json["version"] as? String,
            moduleState: json["module_state"] as? String,
            moduleID: json["module_id"] as? String,
            capture: reported(json, key: "capture", as: captureState),
            captureWorkers: reported(json, key: "capture_workers", as: captureWorkers)
        )
    }

    /// The three-way read that `TowerReported` exists for, plus the fourth
    /// case for a value that is neither an object nor a null.
    private static func reported<Value>(
        _ json: [String: Any], key: String, as decode: ([String: Any]) -> Value
    ) -> TowerReported<Value> {
        guard let raw = json[key] else { return .unreported }
        if raw is NSNull { return .absent }
        guard let object = raw as? [String: Any] else { return .unreadable }
        return .present(decode(object))
    }

    private static func captureState(from json: [String: Any]) -> TowerCaptureState {
        TowerCaptureState(
            armed: json["armed"] as? Bool,
            recording: json["recording"] as? Bool,
            // Null on the wire until the recorder opens its first recording,
            // and that null is an answer rather than a silence — see
            // `TowerCaptureState.captureID`.
            captureID: reportedString(json, key: "capture_id"),
            framesWritten: json["frames_written"] as? Int,
            bytesWritten: json["bytes_written"] as? Int,
            error: json["error"] as? String
        )
    }

    /// The same three-way read for a field whose value is a bare string rather
    /// than an object.
    ///
    /// Separate from `reported(_:key:as:)` because that one's `decode` closure
    /// takes a dictionary — a scalar has no fields to project — and because the
    /// `.unreadable` case here means something narrower: the key arrived
    /// carrying neither a string nor a null, which is the Tower speaking in a
    /// shape this app cannot read.
    private static func reportedString(
        _ json: [String: Any], key: String
    ) -> TowerReported<String> {
        guard let raw = json[key] else { return .unreported }
        if raw is NSNull { return .absent }
        guard let string = raw as? String else { return .unreadable }
        return .present(string)
    }

    private static func captureWorkers(from json: [String: Any]) -> TowerCaptureWorkers {
        TowerCaptureWorkers(
            enabled: json["enabled"] as? Bool,
            // `nil` when no list was sent; `0` when an empty one was. The
            // Tower distinguishes them and so does this.
            workerCount: (json["workers"] as? [Any])?.count,
            error: json["error"] as? String
        )
    }
}

/// The one `GET`, and nothing else.
///
/// ## Read-only, and structurally so
///
/// There is no method here that changes anything on the Tower, and there is
/// nothing on the Tower's side to call: the dataset recorder is armed by
/// `TOWER_CAPTURE_ROOT` at start-up. This type can report the recorder's state
/// and can never alter it.
///
/// ## Why this mirrors `ObjectMemoryHTTPClient` rather than inventing anything
///
/// Same shape, same `JSONSerialization` decoding, same injectable
/// `URLSession`, same timeout policy. A second, differently-opinionated
/// networking layer in one app is two places for that policy to be wrong.
///
/// ## Bounded, and uncached
///
/// Rule 15. The request carries an explicit timeout and
/// `reloadIgnoringLocalCacheData` — a health answer served out of a URL cache
/// would report a recorder as idle after it had started writing, which is the
/// single worst staleness this particular reading can carry.
nonisolated struct TowerHealthHTTPClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = .shared
    /// Long enough for a Tailscale round trip, short enough that a dead Tower
    /// becomes a visible state rather than a spinner nobody can end. Matches
    /// `ObjectMemoryHTTPClient.timeout` deliberately.
    var timeout: TimeInterval = 10

    func health() async throws -> TowerHealth {
        let request = URLRequest(
            url: baseURL.appendingPathComponent("health"),
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )

        do {
            let (data, response) = try await session.data(for: request)
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                throw TowerHealthFetchError.transport("The Tower answered \(http.statusCode).")
            }
            return try TowerHealthDecoder.health(from: data)
        } catch let error as TowerHealthFetchError {
            throw error
        } catch {
            throw TowerHealthFetchError.transport(error.localizedDescription)
        }
    }
}

/// What a screen showing the Tower's state should be showing.
///
/// Four cases because there are four different things to say, and three of
/// them are routinely drawn as the fourth: "nobody has asked", "we are
/// asking", "here is what it said" and "we asked and could not find out" are
/// not the same, and a screen that renders the first and last identically has
/// turned a failure into silence.
nonisolated enum TowerHealthState: Equatable, Sendable {
    /// Nothing has been asked. **Not** "the Tower is fine", and not "the
    /// recorder is off".
    case notFetched
    case fetching
    /// An answer, stamped with when it arrived — because this reading goes
    /// stale the moment a capture starts, and an unstamped one invites being
    /// read as current.
    case fetched(TowerHealth, at: Date)
    case failed(TowerHealthFetchError, at: Date)
}
