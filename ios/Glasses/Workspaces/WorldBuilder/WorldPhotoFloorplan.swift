//
//  WorldPhotoFloorplan.swift
//  Glasses
//
//  Fog of war v1.2 (FOW-V12-PHOTO-FLOORPLAN-DESIGN, "Wire, identity and
//  iOS"): the photo floor plan. Per landed component, one RGBA raster of
//  real camera pixels in that component's own basis, drawn over a neutral
//  gray plate: alpha 0 is "No usable photo here yet".
//
//  - The wire is read in ONE place (`WorldPhotoFloorplanWire`): the manifest
//    and the tile, fail-closed. A contract tweak is a change there only.
//  - The phone's switch, `IOS_FOW_PHOTO_FLOORPLAN`, is default OFF; OFF is
//    no request and no UI change.
//  - Fetched only on the existing pushes: a new landing in
//    `guidance.coverage`, retried a bounded number of times on the geometry
//    push while the Tower's side worker catches up, or on every push while
//    the Tower's `kept` notice holds the landing. No timer, no poll.
//  - What the phone holds is one walk's (`WalkScoped`). A validated photo
//    may briefly remain on its own receipt while the next landing loads,
//    or for as long as the Tower's `kept` notice keeps it for the current
//    receipt. A walk change empties it. Memory only, no disk cache.
//

import Combine
import CryptoKit
import Foundation
import UIKit

// MARK: - The switch

/// The phone's side of the photo floor plan. Read from the launch
/// environment or, for a home-screen launch, from the Info.plist key the
/// build fills from the same-named build setting: `on` exactly; anything
/// else, including an unset setting's empty string, is off.
nonisolated struct WorldPhotoFloorplanSwitch: Equatable, Sendable {
    static let variable = "IOS_FOW_PHOTO_FLOORPLAN"
    static let infoKey = "GlassesFOWPhotoFloorplan"
    static let off = WorldPhotoFloorplanSwitch(on: false)

    var on: Bool

    init(on: Bool) {
        self.on = on
    }

    init(environment: [String: String], info: [String: Any] = [:]) {
        self.init(on: environment[Self.variable] == "on" || info[Self.infoKey] as? String == "on")
    }

    static var current: WorldPhotoFloorplanSwitch {
        WorldPhotoFloorplanSwitch(environment: ProcessInfo.processInfo.environment,
                                  info: Bundle.main.infoDictionary ?? [:])
    }
}

// MARK: - The wire, decoded (the ONE place)

/// `raster_bounds_xy`: the 15%-padded footprint, `[min_x, min_y, max_x,
/// max_y]`, in the same coordinates as v8's `bounds_xy` -- x along the
/// component's `forward`, y along its `right` (`forward x up`), measured from
/// the solve frame's origin (not from `origin_xyz`).
nonisolated struct WorldPhotoBounds: Equatable, Sendable {
    let minX: Double
    let minY: Double
    let maxX: Double
    let maxY: Double

    var spanX: Double { maxX - minX }
    var spanY: Double { maxY - minY }

    func contains(x: Double, y: Double) -> Bool { (minX...maxX).contains(x) && (minY...maxY).contains(y) }
}

/// One component's raster, as the manifest names it.
nonisolated struct WorldPhotoFloorplanComponent: Equatable, Sendable {
    let referenceSegment: Int
    let basis: WorldCoverage.Basis
    let rasterBounds: WorldPhotoBounds
    /// The tile's pixel size, exactly.
    let width: Int
    let height: Int
    /// SHA-256 of the served bytes: 64 lowercase hex; the tile's only key.
    let digest: String
    /// The served byte count, exactly.
    let bytes: Int
    let projectedKeyframes: Int
    let imagerySource: String
}

/// `landed`: one landing's rasters, replaced as a whole.
nonisolated struct WorldPhotoFloorplanLanded: Equatable, Sendable {
    /// `solved_at`, `geometry_revision`, `frame_revision`: with the route's
    /// world and session, the landing's five-part identity.
    let landing: WorldImageryLanding
    let horizonKeyframes: Int
    /// Sorted by `reference_segment`, each once.
    let components: [WorldPhotoFloorplanComponent]
}

/// `kept`: the newest landing the Tower processed published no plan, and it
/// keeps serving `landed` on purpose (TOWER_WORLD_PHOTO_FLOORPLAN_KEPT).
nonisolated struct WorldPhotoFloorplanKept: Equatable, Sendable {
    enum Reason: Equatable, Sendable {
        case regressed, stalled, wallCap, superseded, failed, notNewer

        /// A reason the phone does not know is `failed`.
        init(wire: String) {
            switch wire {
            case "regressed": self = .regressed
            case "stalled": self = .stalled
            case "wall_cap": self = .wallCap
            case "superseded": self = .superseded
            case "not_newer": self = .notNewer
            default: self = .failed
            }
        }
    }

    /// The landing that published nothing: the phone's receipt, if it is current.
    let forLanding: WorldImageryLanding
    let reason: Reason
    /// The plan served instead: always `landed`'s identity.
    let planLanding: WorldImageryLanding
}

/// `GET .../photo-floorplan/manifest`: `landed` is `nil` for `null` (on, and
/// nothing yet). The optional `live` section is not read by the phone.
nonisolated struct WorldPhotoFloorplanManifest: Equatable, Sendable {
    let landed: WorldPhotoFloorplanLanded?
    /// `nil` when absent or malformed: a bad notice never refuses the plan.
    var kept: WorldPhotoFloorplanKept? = nil
}

nonisolated enum WorldPhotoFloorplanError: Error, Equatable {
    /// 404: the route is absent (the Tower's switch is off), or the digest is
    /// unknown or superseded.
    case notFound
    /// An answer the contract does not allow.
    case refused
    /// The tile's bytes are not the digest the manifest named.
    case digestMismatch
    case transport(String)
}

/// Every rule of the wire, and nothing else.
nonisolated enum WorldPhotoFloorplanWire {
    static let version = 1
    static let maxManifestBytes = 8 * 1024
    static let maxTileBytes = 180 * 1024
    static let maxRasterSide = 512
    static let maxComponents = WorldCoverage.maxComponents
    /// The only source the phone draws: the Exp1 cache's redacted frames
    /// (still real camera pixels). A raw source would need its own switch.
    /// "unredacted" is accepted while privacy is deferred for the walk-9 demo
    /// (owner, 2026-10-12): the Tower labels unfiltered cache frames honestly.
    static let imagerySources: Set<String> = ["redacted", "unredacted"]
    static let tileTypes = ["image/webp", "image/png"]

    static func manifestPath(worldID: String, sessionID: String) -> String {
        "worlds/\(worldID)/guidance/\(sessionID)/photo-floorplan/manifest"
    }

    static func tilePath(worldID: String, sessionID: String, digest: String) -> String {
        "worlds/\(worldID)/guidance/\(sessionID)/photo-floorplan/tile/\(digest)"
    }

    /// SHA-256's shape: 64 lowercase hex.
    static func isDigest(_ text: String) -> Bool {
        text.utf8.count == 64 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The manifest body, or `nil`: over 8 KiB, not JSON, a wrong version, a
    /// missing `landed` key, or any malformed known field of `landed`. Unknown
    /// fields are ignored; nothing is repaired. A malformed `kept` is read as
    /// absent, never as a refused manifest.
    static func manifest(_ data: Data) -> WorldPhotoFloorplanManifest? {
        guard data.count <= maxManifestBytes, let json = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }
        return manifest(json: json)
    }

    static func manifest(json: Any) -> WorldPhotoFloorplanManifest? {
        typealias R = WorldCoverageReader
        guard let object = json as? [String: Any], R.integer(object["version"], version...version) != nil,
              object.keys.contains("landed")
        else { return nil }
        if object["landed"] is NSNull { return WorldPhotoFloorplanManifest(landed: nil) }
        guard let section = object["landed"] as? [String: Any],
              let landing = landing(section),
              let horizon = R.integer(section["horizon_keyframes"], 0...65535),
              let rows = section["components"] as? [Any], rows.count <= maxComponents
        else { return nil }
        var components: [WorldPhotoFloorplanComponent] = []
        for row in rows {
            guard let component = component(row),
                  component.referenceSegment > (components.last?.referenceSegment ?? -1)
            else { return nil }
            components.append(component)
        }
        return WorldPhotoFloorplanManifest(
            landed: WorldPhotoFloorplanLanded(landing: landing, horizonKeyframes: horizon, components: components),
            kept: kept(object["kept"]))
    }

    /// `solved_at`, `geometry_revision`, `frame_revision`: a landing, as
    /// `landed` and both of `kept`'s landings carry it.
    private static func landing(_ section: [String: Any]) -> WorldImageryLanding? {
        typealias R = WorldCoverageReader
        guard let solvedAt = R.number(section["solved_at"]), solvedAt >= 0,
              let revision = section["geometry_revision"] as? String, (1...128).contains(revision.utf8.count),
              revision.unicodeScalars.allSatisfy(\.isASCII),
              let frame = R.integer(section["frame_revision"], 0...Int(Int32.max))
        else { return nil }
        return WorldImageryLanding(solvedAt: solvedAt, geometryRevision: revision, frameRevision: frame)
    }

    /// `kept`: both landings with their three keys and the reason a string;
    /// keys it does not know are ignored, like everywhere else. A missing or
    /// mistyped key is no notice at all.
    private static func kept(_ value: Any?) -> WorldPhotoFloorplanKept? {
        guard let object = value as? [String: Any],
              let forSection = object["for_landing"] as? [String: Any], let forLanding = landing(forSection),
              let planSection = object["plan_landing"] as? [String: Any], let planLanding = landing(planSection),
              let reason = object["reason"] as? String
        else { return nil }
        return WorldPhotoFloorplanKept(forLanding: forLanding, reason: .init(wire: reason), planLanding: planLanding)
    }

    private static func component(_ row: Any) -> WorldPhotoFloorplanComponent? {
        typealias R = WorldCoverageReader
        guard let object = row as? [String: Any],
              let segment = R.integer(object["reference_segment"], 0...Int(Int32.max)),
              let origin = R.vector(object["origin_xyz"], count: 3),
              let up = R.vector(object["up_xyz"], count: 3), isUnitish(up),
              let forward = R.vector(object["forward_xyz"], count: 3), isUnitish(forward),
              let b = R.vector(object["raster_bounds_xy"], count: 4), b[2] > b[0], b[3] > b[1],
              let width = R.integer(object["width"], 1...maxRasterSide),
              let height = R.integer(object["height"], 1...maxRasterSide),
              let digest = object["digest"] as? String, isDigest(digest),
              let bytes = R.integer(object["bytes"], 1...maxTileBytes),
              let projected = R.integer(object["projected_keyframes"], 0...65535),
              let source = object["imagery_source"] as? String, imagerySources.contains(source)
        else { return nil }
        return WorldPhotoFloorplanComponent(
            referenceSegment: segment, basis: WorldCoverage.Basis(origin: origin, up: up, forward: forward),
            rasterBounds: WorldPhotoBounds(minX: b[0], minY: b[1], maxX: b[2], maxY: b[3]),
            width: width, height: height, digest: digest, bytes: bytes, projectedKeyframes: projected,
            imagerySource: source)
    }

    private static func isUnitish(_ v: [Double]) -> Bool {
        let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).squareRoot()
        return abs(length - 1) < 1e-3
    }

    /// A tile's bytes, decoded, only as the contract allows: the declared
    /// byte count, at most 180 KiB, SHA-256 equal to the digest, PNG or WebP
    /// by its bytes, the declared pixel size, and an alpha channel.
    @concurrent static func rasterOffMain(_ data: Data, for component: WorldPhotoFloorplanComponent) async throws -> UIImage {
        #if DEBUG
        offMainProbe = !Thread.isMainThread
        #endif
        return try raster(data, for: component)
    }

    #if DEBUG
    /// Test hook: whether the last `rasterOffMain` ran off the main thread.
    nonisolated(unsafe) static var offMainProbe = false
    #endif

    static func raster(_ data: Data, for component: WorldPhotoFloorplanComponent) throws -> UIImage {
        guard data.count <= maxTileBytes, data.count == component.bytes else {
            throw WorldPhotoFloorplanError.refused
        }
        guard sha256(data) == component.digest else { throw WorldPhotoFloorplanError.digestMismatch }
        let png = data.starts(with: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        let webp = data.count >= 12 && data.prefix(4) == Data("RIFF".utf8)
            && data.subdata(in: 8..<12) == Data("WEBP".utf8)
        guard png || webp, let image = UIImage(data: data), let cgImage = image.cgImage,
              cgImage.width == component.width, cgImage.height == component.height,
              ![.none, .noneSkipFirst, .noneSkipLast].contains(cgImage.alphaInfo)
        else { throw WorldPhotoFloorplanError.refused }
        // Decoded once here, off the main actor, rather than on first draw.
        return image.preparingForDisplay() ?? image
    }
}

// MARK: - The routes

/// The manifest and tile routes. Memory only: an ephemeral session with no
/// URL cache, so no raster or manifest ever reaches the disk.
nonisolated struct WorldPhotoFloorplanClient {
    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = WorldImageryClient.memoryOnlySession
    var timeout: TimeInterval = 30

    func manifest(worldID: String, sessionID: String) async throws -> WorldPhotoFloorplanManifest {
        let data = try await get(WorldPhotoFloorplanWire.manifestPath(worldID: worldID, sessionID: sessionID),
                                 accept: ["application/json"])
        guard let manifest = WorldPhotoFloorplanWire.manifest(data) else { throw WorldPhotoFloorplanError.refused }
        return manifest
    }

    func raster(worldID: String, sessionID: String, component: WorldPhotoFloorplanComponent) async throws -> UIImage {
        guard WorldPhotoFloorplanWire.isDigest(component.digest) else { throw WorldPhotoFloorplanError.refused }
        let path = WorldPhotoFloorplanWire.tilePath(worldID: worldID, sessionID: sessionID, digest: component.digest)
        let data = try await get(path, accept: WorldPhotoFloorplanWire.tileTypes)
        return try await WorldPhotoFloorplanWire.rasterOffMain(data, for: component)
    }

    /// The body of a 200 of one of `accept`'s media types.
    private func get(_ path: String, accept: [String]) async throws -> Data {
        let request = URLRequest(url: baseURL.appendingPathComponent(path),
                                 cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: timeout)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw WorldPhotoFloorplanError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else { throw WorldPhotoFloorplanError.transport("no HTTP reply") }
        if http.statusCode == 404 { throw WorldPhotoFloorplanError.notFound }
        guard http.statusCode == 200 else { throw WorldPhotoFloorplanError.transport("HTTP \(http.statusCode)") }
        let type = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
        guard accept.contains(where: { type.hasPrefix($0) }) else { throw WorldPhotoFloorplanError.refused }
        return data
    }
}

// MARK: - What the phone holds

/// Decoded rasters by digest. Immutable per digest, so equal digests are
/// equal rasters.
nonisolated struct WorldPhotoRasters: Equatable {
    private var images: [String: UIImage] = [:]

    subscript(digest: String) -> UIImage? { images[digest] }

    var digests: Set<String> { Set(images.keys) }

    mutating func insert(_ image: UIImage, digest: String) { images[digest] = image }

    /// Only what `digests` names survives.
    mutating func keep(_ digests: Set<String>) { images = images.filter { digests.contains($0.key) } }

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.digests == rhs.digests }
}

/// What the Tower's manifest said for ONE walk, and the rasters in hand.
nonisolated struct WorldPhotoFloorplan: Equatable {
    var landed: WorldPhotoFloorplanLanded?
    var rasters = WorldPhotoRasters()
}

// MARK: - The fetching

/// Fetches the manifest on a new landing and the rasters it names for the
/// map's drawn pieces, for one walk at a time. Fetching has no poll timer;
/// a held photo only schedules its own expiry redraw.
@MainActor
final class WorldPhotoFloorplanModel: ObservableObject {
    /// A landing the manifest does not carry yet (the Tower's side worker
    /// runs after coverage publishes) is asked again on the next geometry
    /// push, at most this many times in all, first fetch included: a bounded retry, never a poll.
    static let landingRetries = 3
    /// A `kept` notice outlives at most this many failed asks in a row: an
    /// unreachable Tower is not a Tower still saying it.
    static let keptFailedAsks = 3

    let floorplanSwitch: WorldPhotoFloorplanSwitch
    private let client: WorldPhotoFloorplanClient
    private let holdSeconds: TimeInterval
    private let now: () -> TimeInterval

    /// Everything held, with the walk it is of. A new walk starts empty.
    @Published private(set) var plan = WalkScoped<WorldPhotoFloorplan>(walk: nil, value: WorldPhotoFloorplan())
    private var receipt = WalkScoped<WorldCoverageReceipt?>(walk: nil, value: nil)
    /// Only a plan proved against the receipt it carries can enter this slot.
    private var lastValid: WalkScoped<(plan: WorldPhotoPlan, supersededAt: TimeInterval?)>?
    private var holdExpiryTask: Task<Void, Never>?

    /// The landing asked for, the receipt's drawn pieces, whether it is
    /// answered, and how often asked.
    private var landing: WorldImageryLanding?
    private var pieces: Set<Int> = []
    private var landingAnswered = false
    private var landingAttempts = 0
    /// The Tower's `kept` notice for `landing`, as last asked: it keeps the
    /// remembered plan on purpose. Not an answer for good: while it holds,
    /// each push asks again, since only asking tells it still holds.
    @Published private var kept: WalkScoped<WorldPhotoFloorplanKept>?
    /// Asks of `landing` that failed in a row (no answer at all).
    private var failedAsks = 0
    private var manifestTask: Task<Void, Never>?
    private var rasterTask: Task<Void, Never>?
    /// Digests whose raster was refused or failed, until the next manifest.
    private var failed: Set<String> = []

    init(floorplanSwitch: WorldPhotoFloorplanSwitch = .current,
         client: WorldPhotoFloorplanClient = WorldPhotoFloorplanClient(),
         holdSeconds: TimeInterval? = nil,
         now: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }) {
        self.floorplanSwitch = floorplanSwitch
        self.client = client
        self.holdSeconds = holdSeconds ?? UITestHooks.photoHoldSeconds ?? 45
        self.now = now
    }

    /// The remembered plan is current on its own receipt. Once superseded,
    /// it stays on that receipt while the Tower's `kept` notice names the
    /// current receipt and that plan, else until the monotonic hold ends.
    func displayedPlan(for current: WalkScoped<WorldCoverageReceipt?>) -> WalkScoped<WorldPhotoDisplay>? {
        guard floorplanSwitch.on, let walk = current.walk, let currentReceipt = current.value,
              receipt.walk == walk, receipt.value?.coverage == currentReceipt.coverage
        else { return nil }
        guard let lastValid, lastValid.walk == walk else { return nil }
        let plan = lastValid.value.plan
        if plan.receipt.coverage == currentReceipt.coverage {
            return WalkScoped(walk: walk, value: WorldPhotoDisplay(plan: plan, held: nil))
        }
        if let notice = keptNotice(for: WorldImageryLanding(currentReceipt.coverage), walk: walk, plan: plan) {
            return WalkScoped(walk: walk, value: WorldPhotoDisplay(plan: plan, held: .init(notice.reason)))
        }
        guard let supersededAt = lastValid.value.supersededAt,
              now() - supersededAt < holdSeconds else { return nil }
        return WalkScoped(walk: walk, value: WorldPhotoDisplay(plan: plan, held: .updating))
    }

    /// The notice for `landing`, if it keeps exactly `plan` (its own landing).
    private func keptNotice(for landing: WorldImageryLanding, walk: WorldFinishWalk,
                            plan: WorldPhotoPlan) -> WorldPhotoFloorplanKept? {
        guard let kept, kept.walk == walk, kept.value.forLanding == landing,
              kept.value.planLanding == WorldImageryLanding(plan.receipt.coverage) else { return nil }
        return kept.value
    }

    /// The walk's `guidance.coverage`: the only trigger is a new landing in
    /// it (`solved_at`, `geometry_revision`, `frame_revision`).
    func coverageReported(_ coverage: WalkScoped<WorldCoverageReceipt?>, receiving: Bool) {
        guard floorplanSwitch.on, receiving, let walk = coverage.walk, let receipt = coverage.value else { return }
        adopt(walk)
        self.receipt = coverage
        let block = receipt.coverage
        rememberValidPlan()
        supersedeRememberedPlan(ifDifferentFrom: block, walk: walk)
        let next = WorldImageryLanding(block)
        guard next != landing else { return }
        manifestTask?.cancel()
        manifestTask = nil
        landing = next
        pieces = Set(block.components.map(\.referenceSegment))
        landingAnswered = false
        landingAttempts = 0
        failedAsks = 0
        if kept != nil { kept = nil }
        fetchManifest(walk: walk, landing: next)
    }

    /// The existing geometry push, heartbeat included: the bounded retry of
    /// a landing the manifest did not carry yet. A landing the Tower answered
    /// with a `kept` notice is asked on every push while the notice holds, so
    /// a plan published later for it, or the notice's end, is seen. Only for
    /// a walk received.
    func geometryPushed(_ coordinates: WorldGeometryCoordinates, receiving: Bool) {
        guard floorplanSwitch.on, receiving else { return }
        let walk = WorldFinishWalk(worldID: coordinates.worldID, sessionID: coordinates.sessionID)
        adopt(walk)
        guard let landing, !landingAnswered, manifestTask == nil else { return }
        if landingAttempts < Self.landingRetries || kept?.value.forLanding == landing {
            fetchManifest(walk: walk, landing: landing)
        }
    }

    /// Forget everything: the world was left, forgotten or pinned.
    func reset() {
        manifestTask?.cancel()
        rasterTask?.cancel()
        holdExpiryTask?.cancel()
        manifestTask = nil
        rasterTask = nil
        holdExpiryTask = nil
        lastValid = nil
        if kept != nil { kept = nil }
        failedAsks = 0
        receipt = WalkScoped(walk: nil, value: nil)
        failed = []
        landing = nil
        pieces = []
        landingAnswered = false
        landingAttempts = 0
        if plan.walk != nil || plan.value != WorldPhotoFloorplan() {
            plan = WalkScoped(walk: nil, value: WorldPhotoFloorplan())
        }
    }

    /// A walk other than the one held: nothing of the old one survives.
    private func adopt(_ walk: WorldFinishWalk) {
        guard plan.walk != walk else { return }
        reset()
        plan = WalkScoped(walk: walk, value: WorldPhotoFloorplan())
    }

    private func isCurrent(_ walk: WorldFinishWalk) -> Bool {
        !Task.isCancelled && plan.walk == walk
    }

    private func fetchManifest(walk: WorldFinishWalk, landing: WorldImageryLanding) {
        landingAttempts += 1
        manifestTask = Task { [weak self, client] in
            let result: Result<WorldPhotoFloorplanManifest, Error>
            do {
                result = .success(try await client.manifest(worldID: walk.worldID, sessionID: walk.sessionID))
            } catch {
                result = .failure(error)
            }
            guard let self, self.isCurrent(walk), self.landing == landing else { return }
            self.manifestTask = nil
            switch result {
            case .success(let manifest):
                self.failedAsks = 0
                // Only a notice for this landing that keeps the plan held is
                // one for the phone; any other is as if absent.
                let held = self.lastValid.flatMap {
                    $0.walk == walk ? WorldImageryLanding($0.value.plan.receipt.coverage) : nil
                }
                let notice = manifest.kept.flatMap {
                    $0.forLanding == landing && $0.planLanding == manifest.landed?.landing && $0.planLanding == held
                        ? $0 : nil
                }
                self.holdKept(notice.map { WalkScoped(walk: walk, value: $0) })
                if let landed = manifest.landed, landed.landing == landing {
                    self.landingAnswered = true
                    self.show(landed, walk: walk)
                }
                // `null`, or another landing's (the side worker is slow, or
                // it kept the prior image): never drawn for this landing;
                // asked again on the next push. A `kept` notice for this
                // landing holds the remembered plan (`displayedPlan`).
            case .failure(WorldPhotoFloorplanError.notFound), .failure(WorldPhotoFloorplanError.refused):
                // An answer: the route is absent, or said what the contract
                // does not allow. Nothing of it is drawn, and no notice holds.
                self.landingAnswered = true
                self.holdKept(nil)
            case .failure:
                self.failedAsks += 1
                if self.failedAsks >= Self.keptFailedAsks { self.holdKept(nil) }
            }
        }
    }

    /// A notice that ends gives its landing back the bounded retries, so a
    /// plan published for it later is still asked for.
    private func holdKept(_ notice: WalkScoped<WorldPhotoFloorplanKept>?) {
        guard kept != notice else { return }
        if notice == nil { landingAttempts = 0 }
        kept = notice
    }

    /// Replaced as a whole, never merged; rasters no longer named go.
    private func show(_ landed: WorldPhotoFloorplanLanded, walk: WorldFinishWalk) {
        failed = []
        var held = plan.value
        held.landed = landed
        held.rasters.keep(Set(landed.components.map(\.digest)))
        if held != plan.value { plan = WalkScoped(walk: walk, value: held) }
        rememberValidPlan()
        pumpRasters(walk: walk)
    }

    /// The first component of the map's drawn pieces with no raster in hand.
    private func nextMissing() -> WorldPhotoFloorplanComponent? {
        plan.value.landed?.components.first {
            pieces.contains($0.referenceSegment) && plan.value.rasters[$0.digest] == nil && !failed.contains($0.digest)
        }
    }

    /// One request at a time, re-reading the manifest in hand after each, so
    /// a newer landing is followed without a second loop. A raster lands only
    /// while the held manifest still names its digest.
    private func pumpRasters(walk: WorldFinishWalk) {
        guard rasterTask == nil else { return }
        rasterTask = Task { [weak self, client] in
            while true {
                guard let self, self.isCurrent(walk), let component = self.nextMissing() else { break }
                let image = try? await client.raster(worldID: walk.worldID, sessionID: walk.sessionID,
                                                     component: component)
                guard self.isCurrent(walk) else { break }
                guard let image else {
                    self.failed.insert(component.digest)
                    continue
                }
                guard self.plan.value.landed?.components.contains(where: { $0.digest == component.digest }) == true
                else { continue }
                var held = self.plan.value
                held.rasters.insert(image, digest: component.digest)
                self.plan = WalkScoped(walk: walk, value: held)
                self.rememberValidPlan()
            }
            guard let self, !Task.isCancelled else { return }
            self.rasterTask = nil
        }
    }

    private func rememberValidPlan() {
        guard floorplanSwitch.on, let walk = receipt.walk, plan.walk == walk,
              let current = receipt.value, let landed = plan.value.landed,
              let valid = WorldPhotoPlan(landed, rasters: plan.value.rasters, receipt: current),
              (lastValid?.value.plan.receipt.coverage != current.coverage ||
               lastValid?.value.plan.pieces != valid.pieces)
        else { return }
        lastValid = WalkScoped(walk: walk, value: (plan: valid, supersededAt: nil))
        holdExpiryTask?.cancel()
        holdExpiryTask = nil
    }

    /// The hold starts when the plan stops being the answer for the current
    /// receipt: when its own receipt is superseded, or a receipt the Tower
    /// kept it for (`kept`) is.
    private func supersedeRememberedPlan(ifDifferentFrom coverage: WorldCoverage, walk: WorldFinishWalk) {
        guard let lastValid, lastValid.walk == walk,
              lastValid.value.plan.receipt.coverage != coverage
        else { return }
        let leavingKept = landing.map {
            $0 != WorldImageryLanding(coverage) && keptNotice(for: $0, walk: walk, plan: lastValid.value.plan) != nil
        } ?? false
        guard lastValid.value.supersededAt == nil || leavingKept else { return }
        holdExpiryTask?.cancel()
        holdExpiryTask = nil
        self.lastValid = WalkScoped(walk: walk, value: (plan: lastValid.value.plan, supersededAt: now()))
        guard holdSeconds > 0 else { return }
        holdExpiryTask = Task { [weak self, holdSeconds] in
            try? await Task.sleep(for: .seconds(holdSeconds))
            guard !Task.isCancelled else { return }
            self?.objectWillChange.send()
        }
    }
}

// MARK: - What the map may draw

/// One piece's photo: its raster, where the raster lies, and the grid the
/// v8 receipt (and B2's thumbnails) use, in the same piece.
nonisolated struct WorldPhotoPiece: Equatable {
    let referenceSegment: Int
    let digest: String
    let image: UIImage
    let bounds: WorldPhotoBounds
    let projectedKeyframes: Int
    /// The receipt's grid: its centre `(dot(origin, forward), dot(origin,
    /// right))` and cell size, in `bounds`' coordinates.
    let gridCentre: (x: Double, y: Double)
    let gridCell: Double

    /// The raster's rect in a `side`-point square piece: the padded bounds
    /// filling the square along their longer span, centred on the other.
    func rasterRect(side: CGFloat) -> CGRect {
        let scale = Double(side) / max(bounds.spanX, bounds.spanY)
        let width = bounds.spanX * scale, height = bounds.spanY * scale
        return CGRect(x: (Double(side) - width) / 2, y: (Double(side) - height) / 2, width: width, height: height)
    }

    /// The receipt's 8-cell grid on this photo's fit: where B2's thumbnails
    /// (placed in grid cells) go over the raster. Not the fog field's fit.
    func gridFit(side: CGFloat) -> WorldCoverageField.Fit {
        let rect = rasterRect(side: side)
        let scale = Double(rect.width) / bounds.spanX
        let half = Double(WorldCoverage.gridSide) / 2
        return WorldCoverageField.Fit(
            cell: CGFloat(gridCell * scale),
            inset: rect.minX + CGFloat((gridCentre.x - half * gridCell - bounds.minX) * scale),
            insetY: rect.minY + CGFloat((gridCentre.y - half * gridCell - bounds.minY) * scale))
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.referenceSegment == rhs.referenceSegment && lhs.digest == rhs.digest && lhs.bounds == rhs.bounds
            && lhs.gridCentre == rhs.gridCentre && lhs.gridCell == rhs.gridCell
    }
}

/// The photo floor plan for ONE receipt: the pieces whose raster is in hand
/// and proved this receipt's.
nonisolated struct WorldPhotoPlan: Equatable {
    let receipt: WorldCoverageReceipt
    /// By `reference_segment`.
    let pieces: [Int: WorldPhotoPiece]

    /// `nil` unless the manifest's landing is the receipt's own (its walk is
    /// checked by the caller), and at least one component is proved: its
    /// reference a drawn piece of the receipt, its basis that piece's, its
    /// footprint around that piece's grid centre, its raster in hand.
    init?(_ landed: WorldPhotoFloorplanLanded, rasters: WorldPhotoRasters, receipt: WorldCoverageReceipt) {
        guard landed.landing == WorldImageryLanding(receipt.coverage) else { return nil }
        var pieces: [Int: WorldPhotoPiece] = [:]
        for component in landed.components {
            guard let image = rasters[component.digest],
                  let drawn = receipt.coverage.components.first(where: { $0.referenceSegment == component.referenceSegment }),
                  let basis = drawn.basis, Self.same(basis, component.basis),
                  let forward = Self.unit(basis.forward), let right = Self.unit(basis.right)
            else { continue }
            let centre = (x: Self.dot(basis.origin, forward), y: Self.dot(basis.origin, right))
            guard component.rasterBounds.contains(x: centre.x, y: centre.y) else { continue }
            pieces[component.referenceSegment] = WorldPhotoPiece(
                referenceSegment: component.referenceSegment, digest: component.digest, image: image,
                bounds: component.rasterBounds, projectedKeyframes: component.projectedKeyframes,
                gridCentre: centre, gridCell: drawn.cellSize)
        }
        guard !pieces.isEmpty else { return nil }
        self.receipt = receipt
        self.pieces = pieces
    }

    /// The manifest's basis is the receipt's for that component.
    static func same(_ a: WorldCoverage.Basis, _ b: WorldCoverage.Basis) -> Bool {
        func close(_ x: [Double], _ y: [Double]) -> Bool {
            x.count == y.count && zip(x, y).allSatisfy { abs($0 - $1) <= 1e-6 * max(1, abs($0), abs($1)) }
        }
        return close(a.origin, b.origin) && close(a.up, b.up) && close(a.forward, b.forward)
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }

    private static func unit(_ v: [Double]) -> [Double]? {
        guard v.count == 3, v.allSatisfy(\.isFinite) else { return nil }
        let length = dot(v, v).squareRoot()
        return length > 1e-9 ? v.map { $0 / length } : nil
    }
}

nonisolated struct WorldPhotoDisplay: Equatable {
    /// What the caption says under a held plan.
    enum Held: Equatable {
        /// The timed hold, or the Tower's transient `superseded` / `wall_cap`.
        case updating
        /// The Tower kept the plan on purpose: `regressed`, `stalled`,
        /// `failed`, `not_newer`, or a reason the phone does not know.
        case notImproved

        init(_ reason: WorldPhotoFloorplanKept.Reason) {
            switch reason {
            case .superseded, .wallCap: self = .updating
            case .regressed, .stalled, .failed, .notNewer: self = .notImproved
            }
        }
    }

    let plan: WorldPhotoPlan
    /// `nil`: the plan is the current receipt's own. Otherwise it is held,
    /// drawn on its own receipt, never as the current receipt's plan.
    let held: Held?
}

extension WorldBuilderViewModel {
    /// The display remains walk-scoped, including while its own receipt is held.
    var photoFloorplan: WalkScoped<WorldPhotoDisplay>? {
        photoFloorplanModel.displayedPlan(for: walkReport.map { $0.coverage })
    }
}
