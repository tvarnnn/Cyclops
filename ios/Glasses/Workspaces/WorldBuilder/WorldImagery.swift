//
//  WorldImagery.swift
//  Glasses
//
//  Fog of war v1.1, option B (FOW-V11B-WIRE-AND-IOS-SPEC-20261007, frozen):
//  redacted keyframe thumbnails, in two stages, each behind its own
//  default-OFF switch.
//
//  - B1, the live strip: the current segment's thumbnails, "Current piece ·
//    unplaced", newest first. Never placed on the room map or its grid. It
//    starts empty at every tracking break, and it is shown only while this
//    phone captures, exactly like the current piece.
//  - B2, solve-placed thumbnails: each landed thumbnail drawn on the v1
//    station grid at its keyframe's solve-placed centre, joined by
//    `keyframe_id` to the poses the geometry fetch already holds -- layered
//    over the grid, never replacing it, and only for the landing the grid on
//    screen comes from.
//
//  What the phone holds is the walk's own (`WalkScoped`): a fetch for one
//  walk never lands in another's, and a new walk starts with nothing. Every
//  tile is the Tower's redacted JPEG, kept in memory only (no disk cache), by
//  digest, at most 16 per stage.
//
//  OFF -- the phone's switch for a stage -- is no UI and no request. The
//  manifest is fetched only on the existing pushes: B1 on a new
//  `geometry.revision` (the geometry push), B2 on a new landing in
//  `guidance.coverage`. No timer, no poll.
//

import Combine
import Foundation
import UIKit

// MARK: - The switches

/// The phone's side of each stage's switch. The Tower's routes exist only
/// when its own switches are on (spec §0); these say whether the phone asks.
/// Read from the launch environment, `on` exactly; anything else is off.
nonisolated struct WorldImagerySwitches: Equatable, Sendable {
    static let liveVariable = "IOS_FOW_ROOM_IMAGERY_LIVE"
    static let landedVariable = "IOS_FOW_ROOM_IMAGERY_LANDED"
    static let off = WorldImagerySwitches(live: false, landed: false)

    var live: Bool
    var landed: Bool

    var anyOn: Bool { live || landed }

    init(live: Bool, landed: Bool) {
        self.live = live
        self.landed = landed
    }

    init(environment: [String: String]) {
        self.init(live: environment[Self.liveVariable] == "on", landed: environment[Self.landedVariable] == "on")
    }

    static var current: WorldImagerySwitches { WorldImagerySwitches(environment: ProcessInfo.processInfo.environment) }
}

// MARK: - The manifest, decoded (§1)

/// One stage's section of the manifest.
nonisolated enum WorldImagerySection<Value: Equatable & Sendable>: Equatable, Sendable {
    /// The key is absent: the Tower's switch for this stage is off.
    case off
    /// `null`: on, and nothing yet.
    case notYet
    case value(Value)

    var value: Value? {
        if case .value(let value) = self { return value }
        return nil
    }
}

/// One accepted thumbnail: complete, matching provenance only.
nonisolated struct WorldImageryEntry: Equatable, Sendable {
    let keyframeID: String
    /// `content_digest()`: 32 lowercase hex; the tile's only cache key.
    let digest: String
    /// Landed only: the component the solve placed it in.
    let componentReferenceSegment: Int?
    /// Live only: the segment it was taken in.
    let segmentIndex: Int?
}

/// `live`: the current segment's thumbnails, newest first.
nonisolated struct WorldImageryLive: Equatable, Sendable {
    let segmentIndex: Int
    let entries: [WorldImageryEntry]
}

/// A landing's identity: the values its `guidance.coverage` carries.
nonisolated struct WorldImageryLanding: Equatable, Sendable {
    let solvedAt: Double
    let geometryRevision: String
    let frameRevision: Int

    init(solvedAt: Double, geometryRevision: String, frameRevision: Int) {
        self.solvedAt = solvedAt
        self.geometryRevision = geometryRevision
        self.frameRevision = frameRevision
    }

    init(_ coverage: WorldCoverage) {
        self.init(solvedAt: coverage.solvedAt, geometryRevision: coverage.geometryRevision,
                  frameRevision: coverage.frameRevision)
    }
}

/// `landed`: one landing's selected thumbnails, replaced as a whole.
nonisolated struct WorldImageryLanded: Equatable, Sendable {
    let landing: WorldImageryLanding
    let entries: [WorldImageryEntry]
}

/// `GET /worlds/{world}/guidance/{session}/imagery/manifest`, read fail-closed:
/// `nil` for a wrong version or any malformed known field of a section;
/// unknown fields ignored; an entry without complete matching provenance is
/// dropped, never repaired.
nonisolated struct WorldImageryManifest: Equatable, Sendable {
    static let maxEntries = 16

    let live: WorldImagerySection<WorldImageryLive>
    let landed: WorldImagerySection<WorldImageryLanded>

    init(live: WorldImagerySection<WorldImageryLive>, landed: WorldImagerySection<WorldImageryLanded>) {
        self.live = live
        self.landed = landed
    }

    init?(json: Any?) {
        typealias R = WorldCoverageReader
        guard let object = json as? [String: Any], R.integer(object["version"], 1...1) != nil else { return nil }
        let index = 0...Int(Int32.max)

        if !object.keys.contains("live") {
            live = .off
        } else if object["live"] is NSNull {
            live = .notYet
        } else {
            guard let section = object["live"] as? [String: Any],
                  let segment = R.integer(section["segment_index"], index),
                  let rows = section["entries"] as? [Any], rows.count <= Self.maxEntries
            else { return nil }
            live = .value(WorldImageryLive(segmentIndex: segment, entries: Self.entries(rows, .live(segment))))
        }

        if !object.keys.contains("landed") {
            landed = .off
        } else if object["landed"] is NSNull {
            landed = .notYet
        } else {
            guard let section = object["landed"] as? [String: Any],
                  let solvedAt = R.number(section["solved_at"]), solvedAt >= 0,
                  let revision = section["geometry_revision"] as? String, (1...128).contains(revision.utf8.count),
                  revision.unicodeScalars.allSatisfy(\.isASCII),
                  let frame = R.integer(section["frame_revision"], index),
                  let rows = section["entries"] as? [Any], (1...Self.maxEntries).contains(rows.count)
            else { return nil }
            landed = .value(WorldImageryLanded(
                landing: WorldImageryLanding(solvedAt: solvedAt, geometryRevision: revision, frameRevision: frame),
                entries: Self.entries(rows, .landed)))
        }
    }

    private enum Provenance: Equatable {
        case live(Int)
        case landed
    }

    /// The accepted rows, in the Tower's order, each digest once.
    private static func entries(_ rows: [Any], _ provenance: Provenance) -> [WorldImageryEntry] {
        var seen = Set<String>()
        return rows.compactMap { row in
            guard let entry = entry(row, provenance), seen.insert(entry.digest).inserted else { return nil }
            return entry
        }
    }

    /// Complete matching provenance or nothing: `keyframe_id`, a digest,
    /// `redaction == "redacted"` and the section's `pose_source`, with the
    /// section's own index set and the other one absent or null.
    private static func entry(_ row: Any, _ provenance: Provenance) -> WorldImageryEntry? {
        typealias R = WorldCoverageReader
        let index = 0...Int(Int32.max)
        guard let object = row as? [String: Any],
              let keyframeID = object["keyframe_id"] as? String, !keyframeID.isEmpty,
              let digest = object["digest"] as? String, isDigest(digest),
              object["redaction"] as? String == "redacted"
        else { return nil }
        func isNull(_ key: String) -> Bool { object[key] == nil || object[key] is NSNull }
        switch provenance {
        case .live(let segment):
            guard object["pose_source"] as? String == "segment_local", isNull("component_reference_segment"),
                  R.integer(object["segment_index"], index) == segment
            else { return nil }
            return WorldImageryEntry(keyframeID: keyframeID, digest: digest, componentReferenceSegment: nil,
                                     segmentIndex: segment)
        case .landed:
            guard object["pose_source"] as? String == "landed_global_solve", isNull("segment_index"),
                  let component = R.integer(object["component_reference_segment"], index)
            else { return nil }
            return WorldImageryEntry(keyframeID: keyframeID, digest: digest, componentReferenceSegment: component,
                                     segmentIndex: nil)
        }
    }

    /// `content_digest()`'s shape: 32 lowercase hex.
    static func isDigest(_ text: String) -> Bool {
        text.utf8.count == 32 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}

// MARK: - The routes

nonisolated enum WorldImageryFetchError: Error, Equatable {
    /// 404: the route is not registered (the Tower's switch is off), or the
    /// tile is unknown or superseded.
    case notFound
    /// An answer the spec does not allow: a malformed manifest, or a tile
    /// that is not a JPEG of at most 16 KiB and 160 x 90.
    case refused
    case transport(String)
}

/// The manifest and tile routes. Memory only: an ephemeral session with no
/// URL cache, so no tile or manifest ever reaches the disk.
nonisolated struct WorldImageryClient {
    static let maxTileBytes = 16 * 1024
    static let maxTileSize = (width: 160, height: 90)

    static let memoryOnlySession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return URLSession(configuration: configuration)
    }()

    var baseURL: URL = TowerConfiguration.httpBaseURL
    var session: URLSession = Self.memoryOnlySession
    var timeout: TimeInterval = 30

    func manifest(worldID: String, sessionID: String) async throws -> WorldImageryManifest {
        let data = try await get("worlds/\(worldID)/guidance/\(sessionID)/imagery/manifest", accept: nil)
        guard let json = try? JSONSerialization.jsonObject(with: data), let manifest = WorldImageryManifest(json: json)
        else { throw WorldImageryFetchError.refused }
        return manifest
    }

    /// The tile, decoded, only as the spec allows it.
    func tile(worldID: String, sessionID: String, digest: String) async throws -> UIImage {
        guard WorldImageryManifest.isDigest(digest) else { throw WorldImageryFetchError.refused }
        let data = try await get("worlds/\(worldID)/guidance/\(sessionID)/imagery/tile/\(digest)", accept: "image/jpeg")
        guard data.count <= Self.maxTileBytes, let image = UIImage(data: data), let cgImage = image.cgImage,
              cgImage.width <= Self.maxTileSize.width, cgImage.height <= Self.maxTileSize.height
        else { throw WorldImageryFetchError.refused }
        // Decoded once here, off the main actor, rather than on first draw.
        return image.preparingForDisplay() ?? image
    }

    /// The body of a 200, of `accept`'s media type when one is named.
    private func get(_ path: String, accept: String?) async throws -> Data {
        let request = URLRequest(url: baseURL.appendingPathComponent(path),
                                 cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: timeout)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw WorldImageryFetchError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else { throw WorldImageryFetchError.transport("no HTTP reply") }
        if http.statusCode == 404 { throw WorldImageryFetchError.notFound }
        guard http.statusCode == 200 else { throw WorldImageryFetchError.transport("HTTP \(http.statusCode)") }
        if let accept {
            let type = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
            guard type.hasPrefix(accept) else { throw WorldImageryFetchError.refused }
        }
        return data
    }
}

// MARK: - The tiles in hand

/// Decoded tiles by digest, least recently used evicted beyond 16: one
/// stage's whole visible set, and no more.
nonisolated struct WorldImageryTiles: Equatable {
    static let capacity = 16

    /// Least recently used first.
    private(set) var digests: [String] = []
    private var images: [String: UIImage] = [:]

    subscript(digest: String) -> UIImage? { images[digest] }

    var count: Int { digests.count }

    mutating func insert(_ image: UIImage, digest: String) {
        digests.removeAll { $0 == digest }
        digests.append(digest)
        images[digest] = image
        while digests.count > Self.capacity {
            images[digests.removeFirst()] = nil
        }
    }

    /// The tiles a manifest shows are the most recently used.
    mutating func touch(_ wanted: [String]) {
        let present = wanted.filter { images[$0] != nil }
        digests.removeAll { present.contains($0) }
        digests.append(contentsOf: present)
    }

    /// Tiles are immutable per digest: the same digests are the same tiles.
    static func == (lhs: Self, rhs: Self) -> Bool { lhs.digests == rhs.digests }
}

/// What the Tower's manifest said for ONE walk, and the tiles in hand.
nonisolated struct WorldImagery: Equatable {
    var live: WorldImageryLive?
    var landed: WorldImageryLanded?
    var liveTiles = WorldImageryTiles()
    var landedTiles = WorldImageryTiles()
}

// MARK: - The fetching

/// Fetches the manifest on the existing pushes and the tiles it names, for
/// one walk at a time. The view model feeds it; nothing here has a timer.
@MainActor
final class WorldImageryModel: ObservableObject {
    /// A landing whose manifest did not yet carry it (the Tower's job is slow
    /// or failed, or a request failed) is asked again on the next geometry
    /// push, at most this many times: a bounded retry, never a poll.
    static let landingRetries = 3

    nonisolated enum Stage: Hashable, Sendable { case live, landed }

    let switches: WorldImagerySwitches
    private let client: WorldImageryClient

    /// Everything held, with the walk it is of. A new walk starts empty.
    @Published private(set) var imagery = WalkScoped<WorldImagery>(walk: nil, value: WorldImagery())

    /// B1: the `geometry.revision` whose manifest is in hand or out; `nil`
    /// after a failed request, so the next push asks again (the geometry
    /// fetch's own retry rule).
    private var liveRevision: String?
    private var liveTask: Task<Void, Never>?
    /// B2: the landing asked for, whether it is answered, and how often asked.
    private var landing: WorldImageryLanding?
    private var landingAnswered = false
    private var landingAttempts = 0
    private var landedTask: Task<Void, Never>?
    private var tileTasks: [Stage: Task<Void, Never>] = [:]
    /// Digests whose tile was refused or failed, per stage, until the next
    /// manifest names them again.
    private var failedTiles: [Stage: Set<String>] = [:]

    init(switches: WorldImagerySwitches = .current, client: WorldImageryClient = WorldImageryClient()) {
        self.switches = switches
        self.client = client
    }

    /// The existing geometry push, heartbeat included: B1's only trigger
    /// (a new revision), and B2's retry. Only for a walk being received.
    func geometryPushed(_ coordinates: WorldGeometryCoordinates, receiving: Bool) {
        guard switches.anyOn, receiving else { return }
        let walk = WorldFinishWalk(worldID: coordinates.worldID, sessionID: coordinates.sessionID)
        adopt(walk)
        if switches.live, coordinates.revision != liveRevision {
            fetchLive(walk: walk, revision: coordinates.revision)
        }
        if switches.landed, let landing, !landingAnswered, landedTask == nil,
           landingAttempts <= Self.landingRetries {
            fetchLanded(walk: walk, landing: landing)
        }
    }

    /// The walk's `guidance.coverage`: B2's only trigger is a new landing in
    /// it (`solved_at`, `geometry_revision`, `frame_revision`).
    func coverageReported(_ coverage: WalkScoped<WorldCoverage?>, receiving: Bool) {
        guard switches.landed, receiving, let walk = coverage.walk, let block = coverage.value else { return }
        adopt(walk)
        let next = WorldImageryLanding(block)
        guard next != landing else { return }
        landedTask?.cancel()
        landedTask = nil
        landing = next
        landingAnswered = false
        landingAttempts = 0
        fetchLanded(walk: walk, landing: next)
    }

    /// Forget everything: the world was left, forgotten or pinned.
    func reset() {
        liveTask?.cancel()
        landedTask?.cancel()
        tileTasks.values.forEach { $0.cancel() }
        liveTask = nil
        landedTask = nil
        tileTasks = [:]
        failedTiles = [:]
        liveRevision = nil
        landing = nil
        landingAnswered = false
        landingAttempts = 0
        if imagery.walk != nil || imagery.value != WorldImagery() {
            imagery = WalkScoped(walk: nil, value: WorldImagery())
        }
    }

    /// A walk other than the one held: nothing of the old one survives.
    private func adopt(_ walk: WorldFinishWalk) {
        guard imagery.walk != walk else { return }
        reset()
        imagery = WalkScoped(walk: walk, value: WorldImagery())
    }

    /// One assignment of the walk's imagery, its walk unchanged.
    private func update(_ change: (inout WorldImagery) -> Void) {
        var held = imagery.value
        change(&held)
        imagery = WalkScoped(walk: imagery.walk, value: held)
    }

    private func isCurrent(_ walk: WorldFinishWalk) -> Bool {
        !Task.isCancelled && imagery.walk == walk
    }

    // MARK: B1

    private func fetchLive(walk: WorldFinishWalk, revision: String) {
        liveTask?.cancel()
        liveRevision = revision
        liveTask = Task { [weak self, client] in
            let result: Result<WorldImageryManifest, Error>
            do {
                result = .success(try await client.manifest(worldID: walk.worldID, sessionID: walk.sessionID))
            } catch {
                result = .failure(error)
            }
            guard let self, self.isCurrent(walk), self.liveRevision == revision else { return }
            self.liveTask = nil
            switch result {
            case .success(let manifest):
                self.show(live: manifest.live.value, walk: walk)
            case .failure(WorldImageryFetchError.notFound), .failure(WorldImageryFetchError.refused):
                // An answer: the route is absent, or it said something the
                // spec does not allow. Nothing of it is drawn.
                self.show(live: nil, walk: walk)
            case .failure:
                // The previous strip stays; the next push asks again.
                self.liveRevision = nil
            }
        }
    }

    private func show(live: WorldImageryLive?, walk: WorldFinishWalk) {
        guard imagery.value.live != live else { return }
        failedTiles[.live] = nil
        update { held in
            held.live = live
            if let live { held.liveTiles.touch(live.entries.map(\.digest)) }
        }
        if live != nil { pumpTiles(.live, walk: walk) }
    }

    // MARK: B2

    private func fetchLanded(walk: WorldFinishWalk, landing: WorldImageryLanding) {
        landingAttempts += 1
        landedTask = Task { [weak self, client] in
            let result: Result<WorldImageryManifest, Error>
            do {
                result = .success(try await client.manifest(worldID: walk.worldID, sessionID: walk.sessionID))
            } catch {
                result = .failure(error)
            }
            guard let self, self.isCurrent(walk), self.landing == landing else { return }
            self.landedTask = nil
            switch result {
            case .success(let manifest):
                switch manifest.landed {
                case .value(let landed) where landed.landing == landing:
                    self.landingAnswered = true
                    self.show(landed: landed, walk: walk)
                case .value, .notYet:
                    // Not this landing's yet (its job is slow) or another
                    // landing's: never drawn; asked again on the next push.
                    break
                case .off:
                    self.landingAnswered = true
                }
            case .failure(WorldImageryFetchError.notFound), .failure(WorldImageryFetchError.refused):
                self.landingAnswered = true
            case .failure:
                break
            }
        }
    }

    /// Replaced as a whole, never merged.
    private func show(landed: WorldImageryLanded, walk: WorldFinishWalk) {
        guard imagery.value.landed != landed else { return }
        failedTiles[.landed] = nil
        update { held in
            held.landed = landed
            held.landedTiles.touch(landed.entries.map(\.digest))
        }
        pumpTiles(.landed, walk: walk)
    }

    // MARK: Tiles

    /// The first digest the stage's manifest names with no tile in hand.
    private func nextMissing(_ stage: Stage) -> String? {
        let entries: [WorldImageryEntry]
        let tiles: WorldImageryTiles
        switch stage {
        case .live: (entries, tiles) = (imagery.value.live?.entries ?? [], imagery.value.liveTiles)
        case .landed: (entries, tiles) = (imagery.value.landed?.entries ?? [], imagery.value.landedTiles)
        }
        let failed = failedTiles[stage] ?? []
        return entries.first { tiles[$0.digest] == nil && !failed.contains($0.digest) }?.digest
    }

    /// One request at a time per stage, re-reading the manifest in hand
    /// after each, so a newer manifest is followed without a second loop.
    private func pumpTiles(_ stage: Stage, walk: WorldFinishWalk) {
        guard tileTasks[stage] == nil else { return }
        tileTasks[stage] = Task { [weak self, client] in
            while true {
                guard let self, self.isCurrent(walk), let digest = self.nextMissing(stage) else { break }
                let image = try? await client.tile(worldID: walk.worldID, sessionID: walk.sessionID, digest: digest)
                guard self.isCurrent(walk) else { break }
                guard let image else {
                    self.failedTiles[stage, default: []].insert(digest)
                    continue
                }
                self.update { held in
                    switch stage {
                    case .live: held.liveTiles.insert(image, digest: digest)
                    case .landed: held.landedTiles.insert(image, digest: digest)
                    }
                }
            }
            guard let self, !Task.isCancelled else { return }
            self.tileTasks[stage] = nil
        }
    }
}

// MARK: - What the panel may draw

/// One thumbnail, its tile in hand.
nonisolated struct WorldImageryThumbnail: Equatable, Identifiable {
    let digest: String
    let image: UIImage

    var id: String { digest }

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.digest == rhs.digest }
}

/// B1: the current segment's strip, newest leading.
nonisolated struct WorldImageryStrip: Equatable {
    let segmentIndex: Int
    let thumbnails: [WorldImageryThumbnail]
}

/// B2: one thumbnail at its solve-placed centre on a piece's grid.
nonisolated struct WorldLandedThumbnail: Equatable, Identifiable {
    let digest: String
    let image: UIImage
    let referenceSegment: Int
    /// The true projected centre in grid cells, 0...8: `column` along the
    /// piece's sector-zero axis (screen right), `row` along its right axis
    /// (screen down) -- the v1 fans' basis. Never snapped to a cell.
    let column: Double
    let row: Double
    /// Look direction, degrees clockwise from sector zero, or `nil` when the
    /// camera looked (nearly) straight up or down.
    let heading: Double?

    var id: String { digest }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.digest == rhs.digest && lhs.referenceSegment == rhs.referenceSegment && lhs.column == rhs.column
            && lhs.row == rhs.row && lhs.heading == rhs.heading
    }
}

/// B2's overlay: the receipt whose grid the thumbnails are placed on.
nonisolated struct WorldLandedOverlay: Equatable {
    let receipt: WorldCoverageReceipt
    let thumbnails: [WorldLandedThumbnail]

    func thumbnails(on referenceSegment: Int) -> [WorldLandedThumbnail] {
        thumbnails.filter { $0.referenceSegment == referenceSegment }
    }
}

/// Where a landed thumbnail goes on its piece, from poses already fetched.
nonisolated enum WorldLandedPlacement {
    /// Each entry with its tile in hand, placed by joining `keyframe_id` to a
    /// posed keyframe of a chunk placed in the entry's component AND the
    /// landing's frame, projected into that component's grid. Dropped -- never
    /// guessed -- when the join fails, it falls outside the piece, or its cell
    /// is not a station with weak or supported evidence: a thumbnail never
    /// lands on grey.
    static func place(_ landed: WorldImageryLanded, on coverage: WorldCoverage,
                      chunks: some Sequence<WorldSegmentChunk>, tiles: WorldImageryTiles) -> [WorldLandedThumbnail] {
        var poses: [String: (pose: WorldPose, transform: WorldTransform)] = [:]
        for chunk in chunks {
            guard let transform = chunk.transformToWorld, transform.frameRevision == landed.landing.frameRevision
            else { continue }
            for pose in chunk.poses where poses[pose.keyframeID] == nil { poses[pose.keyframeID] = (pose, transform) }
        }
        return landed.entries.compactMap { entry in
            guard let image = tiles[entry.digest], let reference = entry.componentReferenceSegment,
                  let component = coverage.components.first(where: { $0.referenceSegment == reference }),
                  let joined = poses[entry.keyframeID], joined.transform.referenceSegment == reference,
                  let local = joined.pose.translation, local.count == 3, local.allSatisfy(\.isFinite),
                  let centre = joined.transform.apply(to: local),
                  let spot = project(centre, onto: component), isEvidence(spot, on: component)
            else { return nil }
            let forward = joined.pose.rotation.flatMap(cameraForward)
                .flatMap { rotate($0, by: joined.transform.rotationWXYZ) }
            return WorldLandedThumbnail(digest: entry.digest, image: image, referenceSegment: reference,
                                        column: spot.column, row: spot.row,
                                        heading: forward.flatMap { heading($0, on: component) })
        }
    }

    /// A solve-frame point in grid cells (0...8 each way), or `nil` outside
    /// the piece's square or without a usable basis.
    static func project(_ point: [Double], onto component: WorldCoverage.Component) -> (column: Double, row: Double)? {
        guard let basis = component.basis, let forward = unit(basis.forward), let right = unit(basis.right),
              point.count == 3, basis.origin.count == 3
        else { return nil }
        let offset = zip(point, basis.origin).map { $0 - $1 }
        let half = Double(WorldCoverage.gridSide) / 2
        let column = dot(offset, forward) / component.cellSize + half
        let row = dot(offset, right) / component.cellSize + half
        let side = Double(WorldCoverage.gridSide)
        guard column.isFinite, row.isFinite, (0...side).contains(column), (0...side).contains(row) else { return nil }
        return (column, row)
    }

    /// The spot's cell (the maximum edge belongs to cell 7) is a station with
    /// at least one weak or supported sector.
    static func isEvidence(_ spot: (column: Double, row: Double), on component: WorldCoverage.Component) -> Bool {
        let last = WorldCoverage.gridSide - 1
        let x = min(last, Int(spot.column.rounded(.down))), y = min(last, Int(spot.row.rounded(.down)))
        return component.stations.contains { $0.x == x && $0.y == y && ($0.weakMask | $0.supportedMask) != 0 }
    }

    /// Degrees clockwise from the piece's sector zero, as seen from above.
    static func heading(_ direction: [Double], on component: WorldCoverage.Component) -> Double? {
        guard let basis = component.basis, let forward = unit(basis.forward), let right = unit(basis.right),
              let direction = unit(direction)
        else { return nil }
        let along = dot(direction, forward), across = dot(direction, right)
        guard along * along + across * across > 0.04 else { return nil }
        let degrees = atan2(across, along) * 180 / .pi
        return degrees < 0 ? degrees + 360 : degrees
    }

    /// The camera's +z (OpenCV forward) under `T_world_camera`'s rotation.
    static func cameraForward(_ q: [Double]) -> [Double]? {
        rotate([0, 0, 1], by: q)
    }

    static func rotate(_ v: [Double], by q: [Double]) -> [Double]? {
        guard q.count == 4, v.count == 3, q.allSatisfy(\.isFinite) else { return nil }
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).squareRoot()
        guard norm > 1e-9 else { return nil }
        let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)
        let tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0])
        return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }

    private static func unit(_ v: [Double]) -> [Double]? {
        guard v.count == 3, v.allSatisfy(\.isFinite) else { return nil }
        let length = dot(v, v).squareRoot()
        return length > 1e-9 ? v.map { $0 / length } : nil
    }
}

extension WorldBuilderViewModel {
    /// B1: the current segment's strip, WITH the walk whose it is. Only
    /// while the published geometry is the revision the Tower last named and
    /// the manifest's segment is that geometry's last one: a tracking break
    /// retires the strip the moment either side sees it. The panel shows it
    /// only through the current piece's gate (this phone capturing).
    var liveStrip: WalkScoped<WorldImageryStrip>? {
        let held = imageryModel.imagery
        guard imageryModel.switches.live, let walk = held.walk, let live = held.value.live,
              let owner = geometryOwner, WorldFinishWalk(picture: owner) == walk,
              let published = publishedGeometryRevision, published == namedGeometryRevision,
              fragmentsModel.segments.last?.segmentIndex == live.segmentIndex
        else { return nil }
        let thumbnails = live.entries.compactMap { entry in
            held.value.liveTiles[entry.digest].map { WorldImageryThumbnail(digest: entry.digest, image: $0) }
        }
        return WalkScoped(walk: walk, value: WorldImageryStrip(segmentIndex: live.segmentIndex, thumbnails: thumbnails))
    }

    /// B2: the landed thumbnails on the walk's own map, WITH that walk. Only
    /// for the landing the map's receipt comes from, placed by this walk's
    /// own geometry.
    var landedOverlay: WalkScoped<WorldLandedOverlay>? {
        let held = imageryModel.imagery
        guard imageryModel.switches.landed, let walk = held.walk, let landed = held.value.landed,
              let receipt = walkReport.value(for: WorldRenderTarget(worldID: walk.worldID, sessionID: walk.sessionID))?
                .coverage,
              landed.landing == WorldImageryLanding(receipt.coverage),
              let owner = geometryOwner, WorldFinishWalk(picture: owner) == walk
        else { return nil }
        let thumbnails = WorldLandedPlacement.place(landed, on: receipt.coverage, chunks: geometryChunks.values,
                                                    tiles: held.value.landedTiles)
        return WalkScoped(walk: walk, value: WorldLandedOverlay(receipt: receipt, thumbnails: thumbnails))
    }
}
