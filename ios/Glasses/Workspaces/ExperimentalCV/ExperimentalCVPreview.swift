//
//  ExperimentalCVPreview.swift
//  Glasses
//

import Foundation
import SwiftUI
import UIKit

/// The CV Lab's live view: what the algorithm is looking at, drawn.
///
/// ## What this replaces
///
/// A screen of numbers. `edge_density: 0.071` is a true statement about a
/// frame and it tells a person wearing the glasses nothing about whether the
/// algorithm can see the doorway they are standing in. This is the panel that
/// answers that question, and everything else on the screen moved down to make
/// room for it.
///
/// ## Why this file holds no imagery it did not just draw
///
/// Every preview arrives with `treatment: "raw_ephemeral"`, which
/// `RedactionState` defines as *"Untreated imagery. Permitted only for the
/// live, in-memory view of what the wearer currently sees — never for anything
/// persisted, and never for anything a cartridge stored and re-served later."*
///
/// This app's half of that is structural rather than a promise:
///
/// - the session is `URLSessionConfiguration.ephemeral` with `urlCache = nil`,
///   so no byte reaches a disk cache or an in-memory one;
/// - every request asks for `.reloadIgnoringLocalAndRemoteCacheData`, because
///   the first point is a default somebody outside this file can change;
/// - `stop()` drops the image, and the view calls it `.onDisappear`, on pause,
///   on stop, when the run changes, and **when the app leaves the foreground**
///   — iOS captures an app-switcher snapshot on the way out, and a snapshot is
///   a copy on disk that nobody chose and no retention governs;
/// - exactly one `UIImage` is held at a time, replaced rather than appended to.
///
/// The Tower's half is `Cache-Control: no-store`, nothing written to disk, and
/// one image in memory per run. Neither side is trusting the other.
///
/// **A treatment this build does not recognise is not displayed.** There is no
/// lenient default here for the same reason `RedactionState` has none: an
/// unstated treatment is not a treatment, and a viewer that drew whatever
/// arrived would make the whole vocabulary decorative.

// MARK: - The descriptor

/// What the Tower says a preview would be, before any bytes are fetched.
///
/// Read off `run.annotation.artifact`, which was `null` on every previous
/// version of this contract. See `EXPERIMENTAL-CV-LAB.md` §5.
struct CVLivePreview: Equatable, Sendable {
    /// `experimental_cv.preview/2026-08-29`. Opaque, compared for equality.
    /// A mismatch means this build cannot read what the Tower is offering, and
    /// the honest response is to draw nothing rather than to guess.
    let contract: String
    /// How to READ the picture — `edge_map`, `relative_depth`, `keypoints`,
    /// `detections`, `flow_tracks`, `redaction_regions`, `frame_quality`.
    ///
    /// **Displayed, never switched on to decide what a picture means.** A
    /// `relative_depth` preview is not metres, and nothing about there being a
    /// picture changes what §4's provenance and unit rules already say.
    let visualKind: String
    /// The Tower's own sentence about what is drawn. Shown verbatim.
    let description: String
    /// How this imagery was treated. See the file comment.
    let redaction: RedactionState
    /// The Tower's process claim — `none` means no face detector ran on this
    /// path. Never an outcome claim, and never rendered as one.
    let faceFilter: String
    /// Where to fetch it. A path, not a URL: resolved against
    /// `TowerConfiguration.httpBaseURL`, the same base every other HTTP client
    /// in this app uses.
    let path: String
    /// The run these previews belong to. **The staleness guard**: sent back on
    /// every fetch, so a Tower that has moved on refuses rather than answering
    /// with a picture this app would draw under the wrong experiment's name.
    let runID: String?
    /// Past this age the Tower refuses rather than serving. Read so the panel
    /// can say "the picture stopped" with the Tower's own number in it.
    let maxAgeSeconds: Double
    /// What the Tower suggests polling at. Advisory, and taken: it is one
    /// number in one place, on the machine actually producing the frames,
    /// rather than a constant hardcoded on the phone that nobody can change
    /// from there.
    let pollIntervalSeconds: Double

    init(
        contract: String,
        visualKind: String,
        description: String,
        redaction: RedactionState,
        faceFilter: String,
        path: String,
        runID: String?,
        maxAgeSeconds: Double,
        pollIntervalSeconds: Double
    ) {
        self.contract = contract
        self.visualKind = visualKind
        self.description = description
        self.redaction = redaction
        self.faceFilter = faceFilter
        self.path = path
        self.runID = runID
        self.maxAgeSeconds = maxAgeSeconds
        self.pollIntervalSeconds = pollIntervalSeconds
    }

    /// One artifact block, or `nil` when the Tower sent something this build
    /// cannot use.
    ///
    /// Fails closed on every field that governs display. A missing `path`
    /// cannot be fetched; a missing or unrecognised `treatment` is `.unknown`,
    /// which this app handles exactly as strictly as raw and therefore does
    /// not draw. Neither is defaulted into something usable.
    init?(json: [String: Any]) {
        guard
            let contract = json["contract"] as? String,
            let visualKind = json["visual_kind"] as? String,
            let path = json["path"] as? String
        else { return nil }
        self.init(
            contract: contract,
            visualKind: visualKind,
            description: json["description"] as? String ?? "",
            redaction: CVWireRedaction.read(json["treatment"] as? String),
            faceFilter: json["face_filter"] as? String ?? "unstated",
            path: path,
            runID: json["run_id"] as? String,
            // Defaults chosen to be conservative rather than convenient: a
            // Tower that did not say how stale is too stale gets the shortest
            // window this app will draw, and one that did not suggest a rate
            // gets the slowest.
            maxAgeSeconds: json["max_age_s"] as? Double ?? 2.0,
            pollIntervalSeconds: json["poll_interval_s"] as? Double ?? 0.2
        )
    }

    /// Whether this build may draw it at all.
    ///
    /// `redacted` and `rawEphemeral` are both live-displayable — that is
    /// `rawEphemeral`'s entire documented purpose. `unknown` is not, and there
    /// is deliberately no third case.
    var isDrawable: Bool {
        contract == ExperimentalCVContract.preview && redaction.isDisplayableLive
    }

    /// Why nothing is drawn, when nothing is. `nil` when it may be.
    var withheldReason: String? {
        if contract != ExperimentalCVContract.preview {
            return """
                This Tower offers a live view in a format this app does not \
                read (\(contract)). Update the app.
                """
        }
        return redaction.isDisplayableLive ? nil : redaction.explanation
    }
}

/// Reads `treatment` off the wire into `RedactionState`.
///
/// One reader, in one place, for the same reason `CVWireProvenance` is one
/// reader: a second copy of the mapping is how two copies come to disagree,
/// and this one decides whether a picture of the wearer's room is drawn.
nonisolated enum CVWireRedaction {
    static func read(_ word: String?) -> RedactionState {
        switch word {
        case "redacted": return .redacted
        case "raw_ephemeral": return .rawEphemeral
        // Silence is not permission. The Tower states this on every preview
        // and never omits it, so reaching here means a document this build
        // could not read — which is exactly what `.unknown` is for, and
        // `.unknown` is not drawn.
        default: return .unknown
        }
    }
}

// MARK: - One fetched frame

/// A preview that arrived, and the identity that says which frame it is.
///
/// `@unchecked Sendable` because of `UIImage`, which UIKit declares the same
/// way. Safe here for a reason and not by assertion: this struct is `let`
/// throughout, the image is decoded once from bytes and never drawn into, and
/// it crosses exactly one boundary -- the `nonisolated` fetch handing a
/// finished frame to the `@MainActor` loader.
struct CVPreviewFrame: Equatable, @unchecked Sendable {
    let image: UIImage
    /// The run the Tower produced it under. Checked against the run this app
    /// is watching before anything is drawn — the second half of the guard
    /// whose first half is sending `run_id` on the request.
    let runID: String?
    /// The Tower's dense per-run ordinal. Not the phone's capture index, which
    /// skips by design and cannot order results.
    let resultSeq: Int?
    /// Seconds between the Tower producing this and answering for it.
    let ageSeconds: Double?
    /// When this device received it.
    ///
    /// Needed because `ageSeconds` is a fact about the moment the Tower
    /// answered and does not tick. A phone polling a Tower whose stream has
    /// stopped gets `304 Not Modified` for as long as the frame stays inside
    /// `max_age_s` — the picture is genuinely unchanged, and reporting its
    /// original age for those two seconds would say "Live" over a frozen
    /// image. `displayedAge` adds the time since.
    let arrivedAt: Date
    let treatment: RedactionState
    /// The `ETag` to send back, so the next poll costs a round trip instead of
    /// an encode on the Tower when nothing has changed.
    let etag: String?

    /// How old this picture is NOW, as far as this device can tell.
    ///
    /// Two clocks, and they are not being compared: `ageSeconds` is the
    /// Tower's own measurement of its own work, and the second term is this
    /// device measuring its own wait. Adding them is legitimate where
    /// subtracting two absolute timestamps would not be — which is the
    /// distinction `07-PLATFORM-CONSTRAINTS.md` Limitation 9 draws, and the
    /// reason there is still no end-to-end latency field anywhere here.
    var displayedAge: Double? {
        guard let ageSeconds else { return nil }
        return ageSeconds + max(0, Date().timeIntervalSince(arrivedAt))
    }

    static func == (lhs: CVPreviewFrame, rhs: CVPreviewFrame) -> Bool {
        lhs.runID == rhs.runID && lhs.resultSeq == rhs.resultSeq
            && lhs.etag == rhs.etag && lhs.image === rhs.image
    }
}

// MARK: - The HTTP client

/// Fetches one preview. Holds no state between calls except the session.
///
/// HTTP rather than the socket, because that is what the Tower serves and
/// because `ws.py` gives the frame path and the result sender one shared lock:
/// a 5–40 KB image several times a second on that socket would queue against
/// `frame_result`. Same reason `ObjectMemoryImageryHTTPClient` exists, and the
/// session configuration below is the same one for the same privacy reason.
nonisolated struct CVLivePreviewHTTPClient {
    enum Outcome: Sendable {
        /// New bytes.
        case frame(CVPreviewFrame)
        /// The Tower confirmed the frame this app already has is the newest
        /// one. Nothing to redraw and nothing was transferred.
        case unchanged
        /// The Tower is willing and has nothing right now — no frame yet, the
        /// newest aged out, previews switched off, or a render that failed.
        /// Carries the Tower's own sentence.
        case waiting(reason: String, message: String)
        /// The run moved on. Carries the run that is current, so the caller
        /// can stop asking about one that is gone.
        case runChanged(currentRunID: String?)
        /// This experiment has no picture, and asking again will not help.
        case noVisualOutput(message: String)
        case failed(CartridgeFailure)
    }

    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession
    /// Short, and deliberately shorter than the imagery client's twenty
    /// seconds. A preview is a *live* view: a fetch that has not landed in two
    /// seconds is describing a moment that has passed, and hanging on to it
    /// only delays the next one. Bounded either way (Rule 15).
    var timeout: TimeInterval = 2.0

    /// Ephemeral, with no URL cache at all. See the file comment.
    static func uncachedSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return URLSession(configuration: configuration)
    }

    init(
        baseURL: URL = TowerConfiguration.httpBaseURL,
        session: URLSession? = nil,
        timeout: TimeInterval = 2.0
    ) {
        self.baseURL = baseURL
        self.session = session ?? Self.uncachedSession()
        self.timeout = timeout
    }

    func fetch(
        path: String, runID: String?, ifNoneMatch etag: String?
    ) async -> Outcome {
        guard var components = URLComponents(
            url: URL(string: path, relativeTo: baseURL) ?? baseURL,
            resolvingAgainstBaseURL: true
        ) else {
            return .failed(
                CartridgeFailure(
                    kind: .notSupported,
                    message: "The Tower offered a live view at a path this app could not resolve."
                )
            )
        }
        if let runID {
            components.queryItems = [URLQueryItem(name: "run_id", value: runID)]
        }
        guard let url = components.url else {
            return .failed(
                CartridgeFailure(
                    kind: .notSupported,
                    message: "The Tower offered a live view at a path this app could not resolve."
                )
            )
        }

        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        if let etag {
            request.setValue(etag, forHTTPHeaderField: "If-None-Match")
        }

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return .failed(
                    CartridgeFailure(
                        kind: .undecodableResponse,
                        message: "The Tower's live view answered with something that was not an HTTP response."
                    )
                )
            }
            return decode(data: data, http: http)
        } catch is CancellationError {
            // The poll loop was torn down mid-flight. Not a failure and not
            // something to show anybody: the view is going away.
            return .unchanged
        } catch {
            return .failed(
                CartridgeFailure(
                    kind: .transport,
                    message: "The live view could not be fetched: \(error.localizedDescription)"
                )
            )
        }
    }

    private func decode(data: Data, http: HTTPURLResponse) -> Outcome {
        if http.statusCode == 304 { return .unchanged }

        if http.statusCode == 200 {
            // The treatment travels on the bytes as well as in the status
            // document, deliberately: an image whose treatment arrives
            // separately is an image whose treatment can be lost. Read from
            // here rather than assumed from the descriptor.
            let treatment = CVWireRedaction.read(
                http.value(forHTTPHeaderField: "X-CV-Preview-Treatment")
            )
            guard treatment.isDisplayableLive else {
                return .waiting(
                    reason: "treatment_not_displayable", message: treatment.explanation
                )
            }
            guard let image = UIImage(data: data) else {
                return .failed(
                    CartridgeFailure(
                        kind: .undecodableResponse,
                        message: "The Tower's live view sent bytes this device could not decode as an image."
                    )
                )
            }
            return .frame(
                CVPreviewFrame(
                    image: image,
                    runID: http.value(forHTTPHeaderField: "X-CV-Preview-Run"),
                    resultSeq: http.value(forHTTPHeaderField: "X-CV-Preview-Seq")
                        .flatMap(Int.init),
                    ageSeconds: http.value(forHTTPHeaderField: "X-CV-Preview-Age")
                        .flatMap(Double.init),
                    arrivedAt: Date(),
                    treatment: treatment,
                    etag: http.value(forHTTPHeaderField: "ETag")
                )
            )
        }

        // Every other status carries a JSON body naming the reason. The
        // Tower's rule, which this app follows: switch on the reason VALUE,
        // not on the status code.
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        let reason = body?["reason"] as? String ?? "unknown"
        let message = body?["message"] as? String
            ?? "The Tower did not say why it served no live view."

        switch reason {
        case "preview_run_changed":
            return .runChanged(currentRunID: body?["current_run_id"] as? String)
        case "experiment_has_no_visual_output":
            return .noVisualOutput(message: message)
        case "preview_disabled", "no_preview_yet", "preview_stale",
            "preview_render_failed", "lab_unavailable":
            return .waiting(reason: reason, message: message)
        default:
            // An unrecognised reason is a Tower this build does not fully
            // understand, not a broken one. Treated as "nothing right now"
            // and shown with the Tower's own sentence, which is strictly more
            // useful than this app's guess at what it meant.
            return .waiting(reason: reason, message: message)
        }
    }
}

// MARK: - The poll loop

/// Holds the newest preview, and nothing else.
///
/// One `Task`, one image, one ETag. `start` replaces whatever was running;
/// `stop` cancels it and drops the bytes. There is no queue, no history and no
/// buffer of frames — the phone does not need every processed frame and the
/// Tower does not keep them either.
@MainActor
final class CVLivePreviewLoader: ObservableObject {
    enum Phase: Equatable {
        /// Not started. Nothing is being fetched.
        case idle
        /// Started, and no picture has arrived yet.
        case connecting
        /// The newest picture the Tower had.
        case showing(CVPreviewFrame)
        /// The Tower is willing and has nothing right now — carries its own
        /// sentence, which is more specific than anything this app could say.
        case waiting(String)
        /// This experiment has no picture. Not an error.
        case noVisualOutput(String)
        /// The Tower has one and this app may not draw it. Carries the
        /// treatment's explanation.
        case withheld(String)
        case failed(CartridgeFailure)
    }

    @Published private(set) var phase: Phase = .idle

    /// Frames drawn this run, and fetches that came back `304`. Two counters,
    /// bounded by being counters, and they are how somebody reading the
    /// Tower's own `run.preview` block afterwards can tell what the PHONE saw
    /// from what the Tower produced.
    @Published private(set) var framesShown = 0
    @Published private(set) var unchangedResponses = 0

    private let client: CVLivePreviewHTTPClient
    /// The one poll loop. There is never a second: `start` cancels whatever
    /// was running before it begins, so a rapid sequence of starts leaves one
    /// loop rather than a pile of them all writing to `phase`.
    private var task: Task<Void, Never>?
    private var etag: String?
    /// The run this loader is watching. Every arriving frame is checked
    /// against it — the second half of the guard whose first half is sending
    /// `run_id` on the request. Belt and braces, because the two failures are
    /// different: the Tower refusing is a Tower that noticed, and this check
    /// is what happens if one ever does not.
    private var watchedRunID: String?

    init(client: CVLivePreviewHTTPClient = CVLivePreviewHTTPClient()) {
        self.client = client
    }

    /// Begin, or begin again on a different run.
    ///
    /// Idempotent for the same descriptor and run: calling it every time a
    /// status document lands — which is what the view does — must not restart
    /// the loop, or the picture would flicker at the status channel's rate
    /// instead of at the preview's.
    func start(_ preview: CVLivePreview, runID: String?) {
        guard preview.isDrawable else {
            stop()
            phase = .withheld(preview.withheldReason ?? preview.redaction.explanation)
            return
        }
        if task != nil, watchedRunID == runID { return }

        stop()
        watchedRunID = runID
        phase = .connecting
        let path = preview.path
        let interval = max(0.05, min(preview.pollIntervalSeconds, 1.0))
        // `[weak self]`, and re-acquired inside a scope that ENDS before the
        // sleep. That is what makes the loop self-healing: a view torn down
        // without `stop()` releases the loader, the next iteration finds
        // nothing, and the task ends on its own rather than polling the Tower
        // forever on behalf of a screen nobody is looking at. Holding `self`
        // strongly across the whole loop -- which is what the obvious
        // `await self?.poll(...)` does -- would make that leak permanent,
        // because the loop never returns.
        task = Task { [weak self] in
            var idleMultiplier = 1.0
            while !Task.isCancelled {
                do {
                    guard let loader = self else { return }
                    idleMultiplier = await loader.pollOnce(
                        path: path, runID: runID, idleMultiplier: idleMultiplier
                    )
                    if idleMultiplier == 0 { return }
                }
                try? await Task.sleep(
                    nanoseconds: UInt64(interval * max(idleMultiplier, 1.0) * 1e9)
                )
            }
        }
    }

    /// Cancel the loop and drop the bytes.
    ///
    /// Called on pause, on stop, when the run changes, and `.onDisappear`.
    /// Explicitly rather than left to `deinit`, because a `@StateObject`
    /// outlives a view scrolling off screen and the panel must stop holding a
    /// picture of the wearer's room the moment it stops being on screen.
    func stop() {
        task?.cancel()
        task = nil
        etag = nil
        watchedRunID = nil
        framesShown = 0
        unchangedResponses = 0
        phase = .idle
    }

    /// One fetch, applied. Returns the next idle multiplier, or `0` to stop.
    ///
    /// Split out from the loop so the loop can hold `self` for exactly the
    /// duration of a fetch and release it across the sleep -- see `start`.
    private func pollOnce(
        path: String, runID: String?, idleMultiplier: Double
    ) async -> Double {
        let outcome = await client.fetch(path: path, runID: runID, ifNoneMatch: etag)
        if Task.isCancelled { return 0 }

        switch outcome {
        case .frame(let frame):
            guard frame.runID == nil || frame.runID == watchedRunID else {
                // The Tower answered about a run this loader is not watching.
                // It should have refused; this is what happens if one ever
                // does not, and it is why the guard exists on both sides.
                return idleMultiplier
            }
            etag = frame.etag
            framesShown += 1
            phase = .showing(frame)
            return 1.0
        case .unchanged:
            unchangedResponses += 1
            return 1.0
        case .waiting(_, let message):
            // The picture goes with it. A frozen last frame under a "paused"
            // label reads as live to anybody not reading the label, and
            // `raw_ephemeral` means live-view-only in both directions.
            etag = nil
            phase = .waiting(message)
            // Backed off for the waiting case only. A run arming a model
            // takes up to two minutes, and asking ten times a second
            // throughout is two minutes of pointless round trips. A frame
            // resets it, so the live case is never slowed by the idle one.
            return min(idleMultiplier * 1.5, 8.0)
        case .runChanged:
            // Stop asking. The view calls `start` again with the new run when
            // the status document catches up; guessing the new id here would
            // be this app inventing identity.
            etag = nil
            phase = .waiting(
                "The run this view was watching has ended. Waiting for the next one."
            )
            return 0
        case .noVisualOutput(let message):
            phase = .noVisualOutput(message)
            return 0
        case .failed(let failure):
            phase = .failed(failure)
            return min(idleMultiplier * 1.5, 8.0)
        }
    }
}

// MARK: - The panel

/// The live view, drawn large and first.
///
/// ## Hierarchy, and why the picture is at the top
///
/// This workspace used to open with a list of numbers at equal weight —
/// `edge_density`, `spatial_coverage`, `entropy_bits`, `mean_relative_depth`,
/// backend, device, model, three timings and a capacity — every one of them
/// true and none of them answering the question a person wearing the glasses
/// is actually asking, which is *"can it see the doorway?"*
///
/// So: the picture, then a few figures that say whether it is healthy, then
/// everything else behind a disclosure. The diagnostics did not go away. They
/// stopped being the first thing.
///
/// ## What it will not do
///
/// - It does not switch on `visualKind` to decide what a picture MEANS. The
///   Tower's own `description` is drawn verbatim underneath, which is where
///   "these are NOT metres" comes from for a depth map — from the machine that
///   produced it, not from a string this app chose.
/// - It does not draw a picture whose treatment it could not read.
/// - It does not keep one. `.onDisappear` drops it.
struct CVLivePreviewPanel: View {
    let preview: CVLivePreview
    let runID: String?
    /// Whether the run is actually running. A paused or stopped run has no
    /// live view by design — the Tower empties its slot on the way out of
    /// `running` — and this stops the loop rather than letting it poll a Lab
    /// that will only refuse.
    let isRunning: Bool
    let experimentName: String

    @StateObject private var loader = CVLivePreviewLoader()
    /// Watched so the picture goes away before iOS takes its app-switcher
    /// snapshot. `.onDisappear` does not fire on backgrounding, so without
    /// this the one moment a derived view of the wearer's room reaches
    /// persistent storage is the moment they swipe up.
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            previewFrame
            caption
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 18))
        .onAppear { synchronise() }
        // Not `onChange(of: preview)`: the descriptor is rebuilt on every
        // status document, so comparing the whole struct would restart the
        // loop at the status channel's rate. The run id and the running flag
        // are the only two things that should ever restart it.
        .onChange(of: runID) { _, _ in synchronise() }
        .onChange(of: isRunning) { _, _ in synchronise() }
        // `.inactive` as well as `.background`: the snapshot is taken during
        // the inactive phase, on the way out, so waiting for `.background`
        // would be waiting until after the picture had already been copied.
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { synchronise() } else { loader.stop() }
        }
        .onDisappear { loader.stop() }
    }

    private func synchronise() {
        if isRunning, scenePhase == .active {
            loader.start(preview, runID: runID)
        } else {
            loader.stop()
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            SectionLabel("Live view")
            Spacer()
            // Gated on freshness, not merely on there being a picture. A frame
            // 1.9 s old is still `.showing` -- the Tower serves up to 2 s --
            // and a LIVE badge over a caption reading "1.9s behind" is the
            // screen contradicting itself.
            if case .showing(let shown) = loader.phase, isFresh(shown) {
                CVTag(text: "LIVE")
            }
        }
    }

    /// The one threshold on this screen, in one place. 0.35 s is about three
    /// preview intervals at the rate the Tower suggests: long enough not to
    /// flicker on a single missed poll, short enough that a stopped stream
    /// stops claiming to be live within half a second.
    private static let freshSeconds = 0.35

    private func isFresh(_ shown: CVPreviewFrame) -> Bool {
        guard let age = shown.displayedAge else { return true }
        return age < Self.freshSeconds
    }

    @ViewBuilder
    private var previewFrame: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.tertiarySystemFill))
            switch loader.phase {
            case .showing(let shown):
                Image(uiImage: shown.image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(.rect(cornerRadius: 12))
                    .accessibilityLabel(
                        "\(experimentName) live view. \(preview.description)"
                    )
            case .connecting:
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Waiting for the first frame.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            case .idle:
                placeholder("The live view runs while the experiment does.")
            case .waiting(let message):
                placeholder(message)
            case .noVisualOutput(let message):
                placeholder(message)
            case .withheld(let message):
                placeholder(message)
            case .failed(let failure):
                placeholder(failure.message)
            }
        }
        .frame(height: 240)
    }

    private func placeholder(_ message: String) -> some View {
        Text(message)
            .font(.caption)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .padding(20)
    }

    @ViewBuilder
    private var caption: some View {
        // The Tower's own sentence about what is drawn. Verbatim, and this is
        // where a depth map says it is not metres — from the machine that
        // produced it rather than from a string this app picked.
        if !preview.description.isEmpty {
            Text(preview.description)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        HStack(spacing: 10) {
            if case .showing(let shown) = loader.phase, let age = shown.displayedAge {
                // How old, rather than implying now. A picture is a statement
                // about a moment and the moment has a timestamp -- and this
                // one keeps ticking, because a `304` means the picture did not
                // change and not that it got younger. Redrawn on every poll:
                // `unchangedResponses` is `@Published` and moves whether or
                // not a new frame arrived.
                Text(
                    age < Self.freshSeconds
                        ? "Live" : String(format: "%.1fs behind", age)
                )
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
            }
            // What this app does with the pixels, in `RedactionState`'s own
            // words: "A live view. This app does not store it."
            Text(preview.redaction.explanation)
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Spacer(minLength: 0)
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}
