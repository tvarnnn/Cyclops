//
//  WorldPhotoFloorplanTests.swift
//  GlassesTests
//
//  The photo floor plan: the manifest and tile read fail-closed, the raster
//  verified against its SHA-256, drawn as a gray plate plus the raster
//  (alpha 0 reads gray), and no walk's or landing's raster reaches
//  another.
//

import CryptoKit
import SwiftUI
import UIKit
import XCTest
@testable import Glasses

/// A synthetic 512 x 512 RGBA raster -- orange, its bottom-right quadrant
/// alpha 0 -- and a manifest naming it for a stop landing.
enum FOWPhotoFixtures {
    static let orange = (r: 224, g: 112, b: 32)
    static let gray = (r: 0x70, g: 0x75, b: 0x7B)

    static func png(side: Int = 512) -> Data {
        let size = CGSize(width: side, height: side)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        format.preferredRange = .standard
        return UIGraphicsImageRenderer(size: size, format: format).pngData { context in
            let cg = context.cgContext
            cg.setFillColor(CGColor(srgbRed: CGFloat(orange.r) / 255, green: CGFloat(orange.g) / 255,
                                    blue: CGFloat(orange.b) / 255, alpha: 1))
            cg.fill(CGRect(origin: .zero, size: size))
            cg.clear(CGRect(x: side / 2, y: side / 2, width: side / 2, height: side / 2))
        }
    }

    static let raster = png()
    static var digest: String { sha256(raster) }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The stop landing and its basis: a 16-unit square padded
    /// 15% to 18.4.
    static func manifest(solvedAt: String = "1791240313.0", revision: String = "g-stop-984",
                         digest: String? = nil, bytes: Int? = nil, component: String? = nil,
                         kept: String? = nil) -> String {
        let row = component ?? #"{"reference_segment":0,"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"raster_bounds_xy":[-9.2,-9.2,9.2,9.2],"width":512,"height":512,"digest":"\#(digest ?? self.digest)","bytes":\#(bytes ?? raster.count),"projected_keyframes":2,"imagery_source":"redacted"}"#
        let notice = kept.map { #","kept":\#($0)"# } ?? ""
        return #"{"version":1,"landed":{"solved_at":\#(solvedAt),"geometry_revision":"\#(revision)","frame_revision":1,"horizon_keyframes":984,"components":[\#(row)]}\#(notice)}"#
    }

    /// A `kept` notice: by default for the stop block re-solved (g-stop-990),
    /// keeping the stop landing's plan.
    static func kept(reason: String = "regressed", forSolvedAt: String = "1791240320.0",
                     forRevision: String = "g-stop-990", planSolvedAt: String = "1791240313.0",
                     planRevision: String = "g-stop-984") -> String {
        #"{"for_landing":{"solved_at":\#(forSolvedAt),"geometry_revision":"\#(forRevision)","frame_revision":1},"reason":"\#(reason)","plan_landing":{"solved_at":\#(planSolvedAt),"geometry_revision":"\#(planRevision)","frame_revision":1}}"#
    }

    static func decode(_ text: String) -> WorldPhotoFloorplanManifest? {
        WorldPhotoFloorplanWire.manifest(Data(text.utf8))
    }
}

@MainActor
final class WorldPhotoFloorplanTests: XCTestCase {
    private typealias F = FOWPhotoFixtures

    private var stop: WorldCoverage { FOWFixtures.coverage(FOWFixtures.stop)! }
    private var mid: WorldCoverage { FOWFixtures.coverage(FOWFixtures.mid)! }

    // MARK: The decoder

    func testTheManifestDecodes() throws {
        let landed = try XCTUnwrap(F.decode(F.manifest())?.landed)
        XCTAssertEqual(landed.landing, WorldImageryLanding(stop), "the five-part identity's landing")
        XCTAssertEqual(landed.horizonKeyframes, 984)
        let component = try XCTUnwrap(landed.components.first)
        XCTAssertEqual(component.referenceSegment, 0)
        XCTAssertEqual(component.rasterBounds, WorldPhotoBounds(minX: -9.2, minY: -9.2, maxX: 9.2, maxY: 9.2))
        XCTAssertEqual(component.basis, WorldCoverage.Basis(origin: [0, 0, 0], up: [0, 0, 1], forward: [0, 1, 0]))
        XCTAssertEqual(component.width, 512)
        XCTAssertEqual(component.height, 512)
        XCTAssertEqual(component.digest, F.digest)
        XCTAssertEqual(component.bytes, F.raster.count)
        XCTAssertEqual(component.projectedKeyframes, 2)
        XCTAssertEqual(component.imagerySource, "redacted")
        XCTAssertEqual(F.decode(#"{"version":1,"landed":null}"#), WorldPhotoFloorplanManifest(landed: nil))
        // Unknown fields, and the optional `live` section, are not read.
        XCTAssertNotNil(F.decode(F.manifest().replacingOccurrences(
            of: #"{"version":1,"#, with: #"{"version":1,"live":{"segment_index":4,"revision":"x"},"extra":[1],"#)))
    }

    func testMalformedManifestsAreRefused() {
        let good = F.manifest()
        func refused(_ text: String, _ why: String, file: StaticString = #filePath, line: UInt = #line) {
            XCTAssertNil(F.decode(text), why, file: file, line: line)
        }
        refused(good.replacingOccurrences(of: #""version":1"#, with: #""version":2"#), "version 2")
        refused(good.replacingOccurrences(of: #""version":1,"#, with: ""), "no version")
        refused(#"{"version":1}"#, "no landed key")
        refused(#"{"version":true,"landed":null}"#, "a boolean version")
        refused(good.replacingOccurrences(of: #""solved_at":1791240313.0"#, with: #""solved_at":"x""#), "solved_at")
        refused(good.replacingOccurrences(of: #""geometry_revision":"g-stop-984""#, with: #""geometry_revision":"""#),
                "an empty revision")
        refused(good.replacingOccurrences(of: #""frame_revision":1"#, with: #""frame_revision":-1"#), "frame")
        refused(good.replacingOccurrences(of: #""horizon_keyframes":984,"#, with: ""), "no horizon")
        refused(good.replacingOccurrences(of: #""redacted""#, with: #""raw""#), "a raw source")
        XCTAssertNotNil(WorldPhotoFloorplanWire.manifest(Data(good.replacingOccurrences(of: #""redacted""#, with: #""unredacted""#).utf8)),
                        "privacy deferred: an honestly labelled unredacted source is drawn")
        refused(good.replacingOccurrences(of: F.digest, with: String(F.digest.prefix(32))), "a 32-hex digest")
        refused(good.replacingOccurrences(of: F.digest, with: F.digest.uppercased()), "an uppercase digest")
        refused(good.replacingOccurrences(of: #""width":512"#, with: #""width":513"#), "wider than 512")
        refused(good.replacingOccurrences(of: #""height":512"#, with: #""height":0"#), "no height")
        refused(F.manifest(bytes: 180 * 1024 + 1), "over the 180 KiB hard cap")
        refused(good.replacingOccurrences(of: "[-9.2,-9.2,9.2,9.2]", with: "[9.2,-9.2,-9.2,9.2]"), "inverted bounds")
        refused(good.replacingOccurrences(of: "[-9.2,-9.2,9.2,9.2]", with: "[-9.2,-9.2,9.2]"), "three bounds")
        refused(good.replacingOccurrences(of: #""up_xyz":[0.0,0.0,1.0]"#, with: #""up_xyz":[0.0,0.0,2.0]"#),
                "a non-unit up")
        let twice = good.replacingOccurrences(of: #""imagery_source":"redacted"}]"#,
                                  with: #""imagery_source":"redacted"},"# + Self.component(reference: 0) + "]")
        refused(twice, "a reference twice")
        let unsorted = good.replacingOccurrences(of: #""components":[{"reference_segment":0"#,
                                                 with: #""components":["# + Self.component(reference: 3)
                                                     + #",{"reference_segment":0"#)
        refused(unsorted, "unsorted references")
        refused(good.replacingOccurrences(of: #"{"version":1,"#,
                                  with: #"{"version":1,"pad":""# + String(repeating: "x", count: 8 * 1024) + #"","#),
                "over 8 KiB")
    }

    /// `kept`: optional, decoded beside the plan; a malformed notice is
    /// absent and never refuses the manifest; keys it does not know are
    /// ignored; an unknown reason is `failed`.
    func testTheKeptNoticeDecodesAndAMalformedOneIsAbsent() throws {
        XCTAssertNil(F.decode(F.manifest())?.kept, "no notice")
        let manifest = try XCTUnwrap(F.decode(F.manifest(kept: F.kept())))
        XCTAssertEqual(manifest.landed, F.decode(F.manifest())?.landed, "the plan reads as without a notice")
        let kept = try XCTUnwrap(manifest.kept)
        XCTAssertEqual(kept.forLanding, WorldImageryLanding(solvedAt: 1791240320.0, geometryRevision: "g-stop-990",
                                                            frameRevision: 1))
        XCTAssertEqual(kept.planLanding, WorldImageryLanding(stop))
        XCTAssertEqual(kept.reason, .regressed)
        let reasons: [(String, WorldPhotoFloorplanKept.Reason, WorldPhotoDisplay.Held)] = [
            ("regressed", .regressed, .notImproved), ("stalled", .stalled, .notImproved),
            ("wall_cap", .wallCap, .updating), ("superseded", .superseded, .updating),
            ("failed", .failed, .notImproved), ("not_newer", .notNewer, .notImproved),
            ("frozen", .failed, .notImproved), ("", .failed, .notImproved), ("Regressed", .failed, .notImproved)]
        for (wire, reason, held) in reasons {
            let decoded = try XCTUnwrap(F.decode(F.manifest(kept: F.kept(reason: wire)))?.kept?.reason, wire)
            XCTAssertEqual(decoded, reason, wire)
            XCTAssertEqual(WorldPhotoDisplay.Held(decoded), held, wire)
        }
        XCTAssertEqual(WorldPhotoFloorplanCopy.held(.updating).words, "Updating…")
        XCTAssertEqual(WorldPhotoFloorplanCopy.held(.notImproved).words, "Couldn't improve this view yet")
        XCTAssertEqual(F.decode(#"{"version":1,"landed":null,"kept":"# + F.kept() + "}"),
                       WorldPhotoFloorplanManifest(landed: nil), "no plan, nothing kept")

        let good = F.kept()
        let additive: [(String, String)] = [
            (good.replacingOccurrences(of: #""reason":"regressed""#, with: #""reason":"regressed","why":"x""#),
             "an extra key"),
            (good.replacingOccurrences(of: #""frame_revision":1},"reason""#, with: #""frame_revision":1,"x":0},"reason""#),
             "an extra landing key")]
        for (extra, why) in additive {
            XCTAssertEqual(F.decode(F.manifest(kept: extra))?.kept, kept, why)
        }
        let malformed: [(String, String)] = [
            (good.replacingOccurrences(of: #","reason":"regressed""#, with: ""), "no reason"),
            (good.replacingOccurrences(of: #""reason":"regressed""#, with: #""reason":3"#), "a numeric reason"),
            (good.replacingOccurrences(of: #""reason":"regressed""#, with: #""reason":null"#), "a null reason"),
            (good.replacingOccurrences(of: #""solved_at":1791240320.0,"#, with: ""), "a landing without solved_at"),
            (good.replacingOccurrences(of: #""for_landing":"#, with: #""for":"#), "no for_landing"),
            (good.replacingOccurrences(of: #""solved_at":1791240320.0"#, with: #""solved_at":"1791240320.0""#),
             "a string solved_at"),
            (good.replacingOccurrences(of: #""geometry_revision":"g-stop-984""#, with: #""geometry_revision":7"#),
             "a numeric revision"),
            (good.replacingOccurrences(of: #""geometry_revision":"g-stop-984""#, with: #""geometry_revision":"""#),
             "an empty revision"),
            (good.replacingOccurrences(of: #""frame_revision":1}}"#, with: #""frame_revision":-1}}"#),
             "a negative frame"),
            (good.replacingOccurrences(of: #""plan_landing":{"#, with: #""plan_landing":[{"#)
                .replacingOccurrences(of: "}}", with: "}]}"), "an array landing"),
            ("[]", "an array"), ("null", "null"), (#""regressed""#, "a string"), ("{}", "an empty object")]
        for (bad, why) in malformed {
            let decoded = F.decode(F.manifest(kept: bad))
            XCTAssertNotNil(decoded?.landed, "\(why): the manifest was refused")
            XCTAssertNil(decoded?.kept, why)
        }
    }

    private static func component(reference: Int) -> String {
        #"{"reference_segment":\#(reference),"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"raster_bounds_xy":[-9.2,-9.2,9.2,9.2],"width":512,"height":512,"digest":"\#(FOWPhotoFixtures.digest)","bytes":\#(FOWPhotoFixtures.raster.count),"projected_keyframes":2,"imagery_source":"redacted"}"#
    }

    /// The tile: exactly the declared bytes, SHA-256 equal to the digest,
    /// PNG or WebP, the declared size, with alpha.
    func testTheTileDecodeRunsOffTheMainThread() async throws {
        let component = try XCTUnwrap(F.decode(F.manifest())?.landed?.components.first)
        _ = try await WorldPhotoFloorplanWire.rasterOffMain(F.raster, for: component)
        XCTAssertTrue(WorldPhotoFloorplanWire.offMainProbe)
    }

    func testATileIsVerifiedAgainstItsDigest() throws {
        let component = try XCTUnwrap(F.decode(F.manifest())?.landed?.components.first)
        let image = try WorldPhotoFloorplanWire.raster(F.raster, for: component)
        XCTAssertEqual(image.cgImage?.width, 512)

        var tampered = F.raster
        tampered[tampered.count - 20] ^= 0xFF
        XCTAssertThrowsError(try WorldPhotoFloorplanWire.raster(tampered, for: component)) {
            XCTAssertEqual($0 as? WorldPhotoFloorplanError, .digestMismatch)
        }
        XCTAssertThrowsError(try WorldPhotoFloorplanWire.raster(F.raster + Data([0]), for: component)) {
            XCTAssertEqual($0 as? WorldPhotoFloorplanError, .refused, "not the declared byte count")
        }
        let small = F.png(side: 384)
        let wrongSize = try XCTUnwrap(F.decode(F.manifest(digest: F.sha256(small), bytes: small.count))?
            .landed?.components.first)
        XCTAssertThrowsError(try WorldPhotoFloorplanWire.raster(small, for: wrongSize)) {
            XCTAssertEqual($0 as? WorldPhotoFloorplanError, .refused, "384 px declared as 512")
        }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let jpeg = UIGraphicsImageRenderer(size: CGSize(width: 512, height: 512), format: format)
            .jpegData(withCompressionQuality: 0.5) { _ in }
        let opaque = try XCTUnwrap(F.decode(F.manifest(digest: F.sha256(jpeg), bytes: jpeg.count))?
            .landed?.components.first)
        XCTAssertThrowsError(try WorldPhotoFloorplanWire.raster(jpeg, for: opaque)) {
            XCTAssertEqual($0 as? WorldPhotoFloorplanError, .refused, "a JPEG: no alpha")
        }
    }

    // MARK: Identity: only the receipt's own landing

    private func plan(_ manifest: String, receipt coverage: WorldCoverage, rasters: Bool = true) -> WorldPhotoPlan? {
        guard let landed = F.decode(manifest)?.landed else { return nil }
        var held = WorldPhotoRasters()
        if rasters, let component = landed.components.first,
           let image = try? WorldPhotoFloorplanWire.raster(F.raster, for: component) {
            held.insert(image, digest: component.digest)
        }
        return WorldPhotoPlan(landed, rasters: held,
                              receipt: WorldCoverageReceipt(coverage: coverage, keyframesAccepted: nil, clock: nil))
    }

    func testThePhotoIsDrawnOnlyForTheReceiptsOwnLandingAndPiece() {
        XCTAssertNotNil(plan(F.manifest(), receipt: stop)?.pieces[0])
        XCTAssertNil(plan(F.manifest(), receipt: mid), "another landing's receipt")
        XCTAssertNil(plan(F.manifest(solvedAt: "1791240320.0"), receipt: stop), "a re-dated landing")
        XCTAssertNil(plan(F.manifest(revision: "g-stop-990"), receipt: stop), "another geometry revision")
        XCTAssertNil(plan(F.manifest(), receipt: stop, rasters: false), "no raster in hand")
        XCTAssertNil(plan(F.manifest(component: Self.component(reference: 3)), receipt: stop), "no such piece")
        let turned = Self.component(reference: 0).replacingOccurrences(of: #""forward_xyz":[0.0,1.0,0.0]"#,
                                                                        with: #""forward_xyz":[1.0,0.0,0.0]"#)
        XCTAssertNil(plan(F.manifest(component: turned), receipt: stop), "not the piece's basis")
        let away = Self.component(reference: 0).replacingOccurrences(of: "[-9.2,-9.2,9.2,9.2]",
                                                                     with: "[20.0,20.0,38.4,38.4]")
        XCTAssertNil(plan(F.manifest(component: away), receipt: stop), "a footprint off the piece's grid")
    }

    /// The panel puts the photo only over the live receipt it was proved
    /// for: not over the DEBUG fixture, not over another landing's map.
    func testThePanelDrawsThePhotoOnlyOverItsOwnReceipt() throws {
        typealias Panel = WorldInlinePanel<EmptyView>
        let photo = try XCTUnwrap(plan(F.manifest(), receipt: stop))
        XCTAssertEqual(Panel.photo(photo, over: .coverage(photo.receipt)), photo)
        XCTAssertNil(Panel.photo(photo, over: .fixture(photo.receipt)), "over the DEBUG fixture")
        XCTAssertNil(Panel.photo(photo, over: .coverage(WorldCoverageReceipt(coverage: mid, keyframesAccepted: nil,
                                                                             clock: nil))), "over another landing")
        XCTAssertNil(Panel.photo(nil, over: .coverage(photo.receipt)))
    }

    // MARK: Rendering

    /// The raster fills the piece (its padded bounds are the piece's square)
    /// and its alpha-0 quadrant reads the plate's neutral gray.
    func testTheRasterFillsThePieceAndAlphaZeroReadsGray() throws {
        let piece = try XCTUnwrap(plan(F.manifest(), receipt: stop)?.pieces[0])
        let side = 128
        let renderer = ImageRenderer(content: WorldPhotoPieceView(piece: piece, index: 0, showsName: false)
            .frame(width: CGFloat(side), height: CGFloat(side)))
        renderer.scale = 1
        let image = try XCTUnwrap(renderer.cgImage)
        let pixels = try XCTUnwrap(Self.rgba(image, side: side))
        func at(_ x: Int, _ y: Int) -> (r: Int, g: Int, b: Int) {
            let i = (y * side + x) * 4
            return (Int(pixels[i]), Int(pixels[i + 1]), Int(pixels[i + 2]))
        }
        func close(_ a: (r: Int, g: Int, b: Int), _ b: (r: Int, g: Int, b: Int)) -> Bool {
            abs(a.r - b.r) <= 8 && abs(a.g - b.g) <= 8 && abs(a.b - b.b) <= 8
        }
        for (x, y) in [(1, 1), (side - 2, 1), (1, side - 2), (32, 32), (side / 2 - 2, side / 2 - 2)] {
            XCTAssertTrue(close(at(x, y), F.orange), "(\(x), \(y)) is \(at(x, y)), not the raster's photo")
        }
        for (x, y) in [(side - 2, side - 2), (96, 96), (side / 2 + 2, side / 2 + 2)] {
            XCTAssertTrue(close(at(x, y), F.gray), "(\(x), \(y)) is \(at(x, y)), not gray where alpha is 0")
        }
        XCTAssertEqual(piece.rasterRect(side: CGFloat(side)), CGRect(x: 0, y: 0, width: side, height: side))
    }

    /// B2's thumbnails, placed in the receipt's grid cells, land on the
    /// photo's fit of the same grid: the grid's centre is the footprint's
    /// centre, and its edge sits 1.2 units (the padding) in from the raster's.
    func testTheGridFitPutsLandedThumbnailsOnThePhoto() throws {
        let piece = try XCTUnwrap(plan(F.manifest(), receipt: stop)?.pieces[0])
        let fit = piece.gridFit(side: 184)
        XCTAssertEqual(fit.point(CGPoint(x: 4, y: 4)).x, 92, accuracy: 1e-6)
        XCTAssertEqual(fit.point(CGPoint(x: 4, y: 4)).y, 92, accuracy: 1e-6)
        XCTAssertEqual(fit.point(CGPoint(x: 0, y: 0)).x, 12, accuracy: 1e-6)
        XCTAssertEqual(fit.point(CGPoint(x: 8, y: 0)).x, 172, accuracy: 1e-6)
        XCTAssertEqual(fit.cell, 20, accuracy: 1e-6)
        // The fog field's own fit is unchanged.
        let fog = WorldCoverageField.fit(side: 130)
        XCTAssertEqual(fog.point(CGPoint(x: 0, y: 0)), CGPoint(x: 25, y: 25))
    }

    private static func rgba(_ image: CGImage, side: Int) -> [UInt8]? {
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: side, height: side, bitsPerComponent: 8,
                                          bytesPerRow: side * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
            return true
        }
        // An image drawn upright: its top row is the buffer's first.
        return drawn ? pixels : nil
    }
}

// MARK: - Fetching: no walk or landing reaches another

@MainActor
final class WorldPhotoFloorplanModelTests: XCTestCase {
    private typealias F = FOWPhotoFixtures
    private typealias Stub = FOWImageryStubProtocol

    private static let walkA = WorldFinishWalk(worldID: "w1", sessionID: "s9")
    private static let walkB = WorldFinishWalk(worldID: "w2", sessionID: "s2")
    private static let manifestA = "/worlds/w1/guidance/s9/photo-floorplan/manifest"
    private static let manifestB = "/worlds/w2/guidance/s2/photo-floorplan/manifest"
    private static var tileA: String { "/worlds/w1/guidance/s9/photo-floorplan/tile/\(F.digest)" }

    private var stop: WorldCoverage { FOWFixtures.coverage(FOWFixtures.stop)! }
    private var mid: WorldCoverage { FOWFixtures.coverage(FOWFixtures.mid)! }

    override func setUp() {
        super.setUp()
        Stub.reset()
        Stub.set(Self.manifestA, json: F.manifest())
        Stub.set(Self.tileA, status: 200, data: F.raster, type: "image/png")
    }

    private func model(holdSeconds: TimeInterval = 45,
                       now: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }) -> WorldPhotoFloorplanModel {
        WorldPhotoFloorplanModel(client: WorldPhotoFloorplanClient(baseURL: URL(string: "http://tower.test")!,
                                                                   session: Stub.makeSession(), timeout: 5),
                                 holdSeconds: holdSeconds, now: now)
    }

    private func scopedReceipt(_ coverage: WorldCoverage, walk: WorldFinishWalk? = nil)
        -> WalkScoped<WorldCoverageReceipt?> {
        WalkScoped(walk: walk ?? Self.walkA,
                   value: WorldCoverageReceipt(coverage: coverage, keyframesAccepted: nil, clock: nil))
    }

    private func report(_ model: WorldPhotoFloorplanModel, _ coverage: WorldCoverage, walk: WorldFinishWalk? = nil,
                        receiving: Bool = true) {
        model.coverageReported(scopedReceipt(coverage, walk: walk ?? Self.walkA), receiving: receiving)
    }

    private func push(_ model: WorldPhotoFloorplanModel, _ walk: WorldFinishWalk? = nil, revision: String,
                      receiving: Bool = true) {
        let walk = walk ?? Self.walkA
        model.geometryPushed(WorldGeometryCoordinates(worldID: walk.worldID, sessionID: walk.sessionID,
                                                      revision: revision), receiving: receiving)
    }

    private func expect(_ what: String = "", timeout: TimeInterval = 5, _ condition: () -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline { try? await Task.sleep(for: .milliseconds(20)) }
        XCTAssertTrue(condition(), what)
    }

    private func settle() async { try? await Task.sleep(for: .milliseconds(300)) }

    /// The stop block re-solved: the landing a `kept` notice names by default.
    private var newer: WorldCoverage {
        FOWFixtures.coverage(FOWFixtures.stop.replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320."))!
    }

    private var newest: WorldCoverage {
        FOWFixtures.coverage(FOWFixtures.stop.replacingOccurrences(of: "g-stop-984", with: "g-stop-991")
            .replacingOccurrences(of: "1791240313.", with: "1791240321."))!
    }

    private func held(_ model: WorldPhotoFloorplanModel, _ coverage: WorldCoverage) -> WorldPhotoDisplay.Held? {
        model.displayedPlan(for: scopedReceipt(coverage))?.value.held
    }

    /// The stop plan proved, then the newer receipt, the Tower answering with
    /// `manifest`.
    private func keptModel(_ manifest: String, time: @escaping () -> TimeInterval,
                           at supersession: () -> Void) async -> WorldPhotoFloorplanModel {
        let model = model(holdSeconds: 45, now: time)
        report(model, stop)
        await expect("first valid plan") { model.displayedPlan(for: self.scopedReceipt(self.stop)) != nil }
        Stub.set(Self.manifestA, json: manifest)
        supersession()
        report(model, newer)
        return model
    }

    // MARK: The Tower's `kept` notice

    /// A notice for the current receipt holds the plan, on its own receipt,
    /// for as long as the Tower sends it: the 45 s hold does not apply. Each
    /// push asks again while it holds.
    func testAKeptNoticeForTheCurrentReceiptHoldsPastTheHoldWithItsLabel() async throws {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept(reason: "regressed")), time: { time }) { time = 60 }
        await expect("the notice") { self.held(model, self.newer) == .notImproved }
        time = 60 + 45 * 10
        let shown = try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newer)), "the 45 s hold ended a kept plan")
        XCTAssertEqual(shown.value.held, .notImproved)
        XCTAssertEqual(shown.value.plan.receipt.coverage, stop, "the held plan keeps its own receipt")
        XCTAssertEqual(shown.walk, Self.walkA)

        Stub.set(Self.manifestA, json: F.manifest(kept: F.kept(reason: "wall_cap")))
        push(model, revision: "g1")
        await expect("a transient reason") { self.held(model, self.newer) == .updating }
        let asked = Stub.count(Self.manifestA)
        for revision in ["g2", "g3", "g4", "g5"] {
            await settle()
            push(model, revision: revision)
        }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), asked + 4, "a kept landing is asked on every push")
        XCTAssertEqual(held(model, newer), .updating)
    }

    /// The notice for another landing: the Tower says nothing about this
    /// receipt, so the 45 s hold from supersession, then the fog field.
    func testAKeptNoticeForAnotherLandingKeepsTheTimedHold() async {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept(forSolvedAt: "1791240321.0", forRevision: "g-stop-991")),
                                    time: { time }) { time = 60 }
        await expect("the manifest answered") { Stub.count(Self.manifestA) == 2 }
        await settle()
        time = 100
        XCTAssertEqual(held(model, newer), .updating)
        time = 106
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "a notice for another landing held the plan")
    }

    /// `plan_landing` that is not `landed`, or not the plan the phone holds,
    /// is no notice for it.
    func testAKeptNoticeNotNamingTheHeldPlanIsIgnored() async {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept(planSolvedAt: "1791240000.0", planRevision: "g-mid-113")),
                                    time: { time }) { time = 60 }
        await expect("the manifest answered") { Stub.count(Self.manifestA) == 2 }
        await settle()
        time = 106
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "plan_landing is not landed")

        // `landed` and `plan_landing` agree, on a plan the phone does not hold.
        Stub.set(Self.manifestA, json: F.manifest(solvedAt: "1791240000.0", revision: "g-mid-113",
                                                  kept: F.kept(planSolvedAt: "1791240000.0", planRevision: "g-mid-113")))
        push(model, revision: "g1")
        await expect("asked again") { Stub.count(Self.manifestA) == 3 }
        await settle()
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "a notice keeping another plan held this one")
        // Not for the phone, so not asked on every push: the bounded retries.
        for revision in ["g2", "g3", "g4", "g5"] {
            await settle()
            push(model, revision: revision)
        }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), 1 + WorldPhotoFloorplanModel.landingRetries,
                       "a notice for another plan was asked on every push")
    }

    /// The notice ends without a plan: the timed hold from supersession,
    /// long over, so the fog field.
    func testTheNoticeEndingFallsBackToTheTimedHold() async {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept()), time: { time }) { time = 60 }
        await expect("the notice") { self.held(model, self.newer) == .notImproved }
        time = 200
        Stub.set(Self.manifestA, json: F.manifest())
        push(model, revision: "g1")
        await expect("the notice ended") { model.displayedPlan(for: self.scopedReceipt(self.newer)) == nil }
    }

    /// Not stuck on a stale notice: a plan published later for the same
    /// receipt is seen on the next push and shown as that receipt's own.
    func testANewPlanAfterTheNoticeIsShownForTheSameReceipt() async throws {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept(reason: "stalled")), time: { time }) { time = 60 }
        await expect("the notice") { self.held(model, self.newer) == .notImproved }
        for revision in ["g1", "g2", "g3"] { // the bounded retries are spent
            await settle()
            push(model, revision: revision)
        }
        await settle()
        time = 500
        Stub.set(Self.manifestA, json: F.manifest(solvedAt: "1791240320.0", revision: "g-stop-990"))
        push(model, revision: "g4")
        await expect("the new plan") {
            model.displayedPlan(for: self.scopedReceipt(self.newer)).map { $0.value.held == nil } == true
        }
        XCTAssertEqual(try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newer))).value.plan.receipt.coverage, newer)
        let asked = Stub.count(Self.manifestA)
        push(model, revision: "g5")
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), asked, "an answered landing asked again")
    }

    /// The notice held past the bounded retries, then ended with the plan
    /// still the old one (the 8 KiB cap, the Tower's switch): the landing
    /// has its retries back, so a plan published for it later is seen.
    func testANoticeThatEndsGivesTheLandingBackItsRetries() async throws {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept()), time: { time }) { time = 60 }
        await expect("the notice") { self.held(model, self.newer) == .notImproved }
        for revision in ["g1", "g2", "g3", "g4"] {
            await settle()
            push(model, revision: revision)
        }
        await settle()
        XCTAssertEqual(held(model, newer), .notImproved, "held past the bounded retries")
        time = 500
        Stub.set(Self.manifestA, json: F.manifest())
        push(model, revision: "g5")
        await expect("the notice ended") { model.displayedPlan(for: self.scopedReceipt(self.newer)) == nil }
        await settle()
        Stub.set(Self.manifestA, json: F.manifest(solvedAt: "1791240320.0", revision: "g-stop-990"))
        push(model, revision: "g6")
        await expect("the plan published later") {
            model.displayedPlan(for: self.scopedReceipt(self.newer)).map { $0.value.held == nil } == true
        }
        XCTAssertEqual(try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newer))).value.plan.receipt.coverage, newer)
    }

    /// An unreachable Tower is not a Tower still saying it: the notice
    /// outlives at most three failed asks in a row, then the timed hold
    /// (long over) and the fog field; the landing keeps its retries.
    func testANoticeOutlivesAtMostThreeFailedAsks() async throws {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept(reason: "superseded")), time: { time }) { time = 60 }
        time = 500 // past the timed hold: only the notice says Updating…
        await expect("the notice") { self.held(model, self.newer) == .updating }
        Stub.set(Self.manifestA, status: 503, data: Data(), type: "application/json")
        for (index, revision) in ["g1", "g2", "g3"].enumerated() {
            let asked = Stub.count(Self.manifestA)
            push(model, revision: revision)
            await expect("ask \(index + 1)") { Stub.count(Self.manifestA) == asked + 1 }
            await settle()
            if index + 1 < WorldPhotoFloorplanModel.keptFailedAsks {
                XCTAssertEqual(held(model, newer), .updating, "the notice after \(index + 1) failed asks")
            }
        }
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "a notice outlived three failed asks")

        Stub.set(Self.manifestA, json: F.manifest(solvedAt: "1791240320.0", revision: "g-stop-990"))
        push(model, revision: "g4")
        await expect("the Tower back, its plan") {
            model.displayedPlan(for: self.scopedReceipt(self.newer)).map { $0.value.held == nil } == true
        }
    }

    /// Leaving a receipt the Tower kept the plan for starts the hold again:
    /// the next landing's answer is awaited, not met with the fog field.
    func testLeavingAKeptReceiptStartsTheHoldAgain() async {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept()), time: { time }) { time = 60 }
        await expect("the notice") { self.held(model, self.newer) == .notImproved }
        Stub.set(Self.manifestA, json: F.manifest())
        time = 300
        report(model, newest)
        XCTAssertEqual(held(model, newest), .updating, "the newest receipt's answer is awaited")
        await settle()
        time = 344
        XCTAssertEqual(held(model, newest), .updating)
        time = 345
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newest)), "45 s since the kept receipt was left")
    }

    /// A walk change or a reset: nothing kept survives.
    func testAKeptPlanClearsOnWalkChangeAndReset() async {
        var time: TimeInterval = 0
        let model = await keptModel(F.manifest(kept: F.kept()), time: { time }) { time = 60 }
        await expect("the notice") { self.held(model, self.newer) == .notImproved }
        time = 600
        report(model, newer, walk: Self.walkB)
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer, walk: Self.walkB)), "A's kept plan in B")
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "the old walk is no longer current")

        report(model, stop)
        await expect("A again") { model.displayedPlan(for: self.scopedReceipt(self.stop)) != nil }
        report(model, newer)
        await expect("the notice again") { self.held(model, self.newer) == .notImproved }
        model.reset()
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "reset forgot the kept plan")
    }

    func testTheHoldStartsWhenThePlanIsSupersededAfterALongLandingGap() async throws {
        var time: TimeInterval = 0
        let model = model(holdSeconds: 45, now: { time })
        report(model, stop)
        await expect("first valid plan") { model.displayedPlan(for: self.scopedReceipt(self.stop)) != nil }
        let first = try XCTUnwrap(model.displayedPlan(for: scopedReceipt(stop)))
        XCTAssertNil(first.value.held)

        let newer = FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320."))!
        time = 60
        report(model, newer)
        let held = try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newer)))
        XCTAssertEqual(held.value.held, .updating)
        XCTAssertEqual(held.value.plan.receipt, first.value.plan.receipt, "old raster keeps its own basis and receipt")
        XCTAssertEqual(WorldPhotoFloorplanCopy.updating, "Updating…")
        time = 100
        XCTAssertTrue(model.displayedPlan(for: scopedReceipt(newer))?.value.held == .updating)
        time = 106
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "45 seconds since supersession")

        Stub.set(Self.manifestA, json: F.manifest(solvedAt: "1791240320.0", revision: "g-stop-990"))
        await settle() // the unanswered landing's in-flight manifest has finished
        push(model, revision: "g-new")
        await expect("new valid plan") {
            model.displayedPlan(for: self.scopedReceipt(newer)).map { $0.value.held == nil } == true
        }
        let replacement = try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newer)))
        XCTAssertEqual(replacement.value.plan.receipt.coverage, newer)
        time = 200
        XCTAssertNil(try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newer))).value.held,
                       "a current plan has no hold deadline")
    }

    func testAnotherUnansweredLandingDoesNotRestartTheHold() async throws {
        var time: TimeInterval = 0
        let model = model(holdSeconds: 45, now: { time })
        report(model, stop)
        await expect("first valid plan") { model.displayedPlan(for: self.scopedReceipt(self.stop)) != nil }
        let newer = FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320."))!
        time = 60
        report(model, newer)
        XCTAssertTrue(model.displayedPlan(for: scopedReceipt(newer))?.value.held == .updating)

        let newest = FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-991")
            .replacingOccurrences(of: "1791240313.", with: "1791240321."))!
        time = 80
        report(model, newest)
        XCTAssertTrue(model.displayedPlan(for: scopedReceipt(newest))?.value.held == .updating)
        time = 100
        XCTAssertTrue(model.displayedPlan(for: scopedReceipt(newest))?.value.held == .updating)
        time = 106
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newest)), "the second landing did not restart the hold")
    }

    func testANewValidPlanDuringTheHoldClearsTheOldDeadline() async throws {
        var time: TimeInterval = 0
        let model = model(holdSeconds: 45, now: { time })
        report(model, stop)
        await expect("first valid plan") { model.displayedPlan(for: self.scopedReceipt(self.stop)) != nil }
        let newer = FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320."))!
        time = 60
        report(model, newer)
        XCTAssertTrue(model.displayedPlan(for: scopedReceipt(newer))?.value.held == .updating)

        time = 80
        Stub.set(Self.manifestA, json: F.manifest(solvedAt: "1791240320.0", revision: "g-stop-990"))
        await settle()
        push(model, revision: "g-new")
        await expect("replacement during hold") {
            model.displayedPlan(for: self.scopedReceipt(newer)).map { $0.value.held == nil } == true
        }
        time = 106
        XCTAssertNil(try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newer))).value.held)

        let newest = FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-991")
            .replacingOccurrences(of: "1791240313.", with: "1791240321."))!
        report(model, newest)
        let heldReplacement = try XCTUnwrap(model.displayedPlan(for: scopedReceipt(newest)))
        XCTAssertEqual(heldReplacement.value.held, .updating)
        XCTAssertEqual(heldReplacement.value.plan.receipt.coverage, newer, "the replacement became the remembered plan")
        time = 150
        XCTAssertTrue(model.displayedPlan(for: scopedReceipt(newest))?.value.held == .updating)
        time = 151
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newest)))
    }

    func testTheHoldClearsOnWalkChangeAndReset() async {
        let time: TimeInterval = 10
        let model = model(holdSeconds: 4, now: { time })
        report(model, stop)
        await expect { model.displayedPlan(for: self.scopedReceipt(self.stop)) != nil }
        let newer = FOWFixtures.coverage(FOWFixtures.stop
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320."))!
        report(model, newer)
        XCTAssertTrue(model.displayedPlan(for: scopedReceipt(newer))?.value.held == .updating)
        report(model, newer, walk: Self.walkB)
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer, walk: Self.walkB)), "old walk's plan")
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "old walk no longer current")
        report(model, stop, walk: Self.walkA)
        await expect { model.displayedPlan(for: self.scopedReceipt(self.stop)) != nil }
        model.reset()
        XCTAssertNil(model.displayedPlan(for: scopedReceipt(newer)), "reset forgot the plan")
    }

    /// One manifest per landing, the verified raster once; heartbeats and a
    /// repeated landing ask nothing more.
    func testANewLandingFetchesTheManifestAndItsVerifiedRaster() async {
        let model = model()
        report(model, stop)
        await expect("the raster") { model.plan.value.rasters[F.digest] != nil }
        XCTAssertEqual(model.plan.walk, Self.walkA)
        XCTAssertEqual(model.plan.value.landed?.landing, WorldImageryLanding(stop))
        report(model, stop)
        for revision in ["g1", "g2"] { push(model, revision: revision) }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), 1)
        XCTAssertEqual(Stub.count(Self.tileA), 1)
    }

    /// Bytes that are not the digest are never held, and never drawn.
    func testARasterWhoseBytesAreNotItsDigestIsNeverHeld() async {
        var tampered = F.raster
        tampered[tampered.count - 20] ^= 0xFF
        Stub.set(Self.tileA, status: 200, data: tampered, type: "image/png")
        let model = model()
        report(model, stop)
        await expect { Stub.count(Self.tileA) == 1 }
        await settle()
        XCTAssertNotNil(model.plan.value.landed)
        XCTAssertNil(model.plan.value.rasters[F.digest], "a raster that is not its digest")
    }

    // MARK: Walk mixing

    /// A stale walk: A's manifest, out when B is adopted, never lands in B,
    /// and A's raster is never asked for.
    func testAStaleWalksManifestNeverLandsInTheNextWalk() async {
        Stub.set(delay: 0.6, for: Self.manifestA)
        Stub.set(Self.manifestB, json: #"{"version":1,"landed":null}"#)
        let model = model()
        report(model, stop, walk: Self.walkA)
        await expect { Stub.count(Self.manifestA) == 1 }
        report(model, stop, walk: Self.walkB)
        await expect("B's own manifest") { Stub.count(Self.manifestB) == 1 }
        try? await Task.sleep(for: .milliseconds(900))
        XCTAssertEqual(model.plan.walk, Self.walkB)
        XCTAssertNil(model.plan.value.landed, "A's manifest landed in B")
        XCTAssertEqual(model.plan.value.rasters.digests, [])
        XCTAssertEqual(Stub.count(Self.tileA), 0, "A's raster was asked for")
    }

    /// A walk change empties what is held.
    func testAWalkChangeEmptiesTheRasters() async {
        let model = model()
        report(model, stop)
        await expect { model.plan.value.rasters[F.digest] != nil }
        push(model, Self.walkB, revision: "g1")
        XCTAssertEqual(model.plan.walk, Self.walkB)
        XCTAssertEqual(model.plan.value, WorldPhotoFloorplan(), "A's raster survived into B")
    }

    /// A superseded landing: the answer for a landing coverage has moved past
    /// is dropped, and a manifest still carrying the old landing is never
    /// shown for the new one; shown once the Tower carries the new one.
    func testASupersededLandingIsNeverShown() async {
        Stub.set(Self.manifestA, json: F.manifest(solvedAt: "1791240000.0", revision: "g-mid-113"))
        Stub.set(delay: 0.6, for: Self.manifestA)
        let model = model()
        report(model, mid)
        await expect { Stub.count(Self.manifestA) == 1 }
        // Coverage moves to the stop landing while mid's request is out, and
        // the Tower still serves mid's raster (its side worker is slow).
        report(model, stop)
        try? await Task.sleep(for: .milliseconds(1500))
        XCTAssertNil(model.plan.value.landed, "a landing other than the one asked for")
        XCTAssertEqual(Stub.count(Self.tileA), 0)

        Stub.set(delay: 0, for: Self.manifestA)
        Stub.set(Self.manifestA, json: F.manifest())
        push(model, revision: "g9")
        await expect("the stop landing") { model.plan.value.rasters[F.digest] != nil }
        XCTAssertEqual(model.plan.value.landed?.landing, WorldImageryLanding(stop))
    }

    /// A raster still out when its landing is superseded by one naming
    /// another raster is not held.
    func testALateRasterOfASupersededLandingIsNotHeld() async {
        Stub.set(delay: 0.8, for: Self.tileA)
        let model = model()
        report(model, stop)
        await expect { Stub.count(Self.tileA) == 1 }
        let other = F.png(side: 256)
        let newer = F.manifest(solvedAt: "1791240320.0", revision: "g-stop-990",
                               component: #"{"reference_segment":0,"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"raster_bounds_xy":[-9.2,-9.2,9.2,9.2],"width":256,"height":256,"digest":"\#(F.sha256(other))","bytes":\#(other.count),"projected_keyframes":3,"imagery_source":"redacted"}"#)
        Stub.set(Self.manifestA, json: newer)
        let newerTile = "/worlds/w1/guidance/s9/photo-floorplan/tile/\(F.sha256(other))"
        Stub.set(newerTile, status: 200, data: other, type: "image/png")
        let newerCoverage = FOWFixtures.coverage(FOWFixtures.stop.replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320."))!
        report(model, newerCoverage)
        await expect("the newer raster") { model.plan.value.rasters[F.sha256(other)] != nil }
        try? await Task.sleep(for: .milliseconds(900))
        XCTAssertNil(model.plan.value.rasters[F.digest], "the superseded landing's raster was held")
        XCTAssertEqual(model.plan.value.landed?.landing, WorldImageryLanding(newerCoverage))
    }

    func testASlowLandingIsAskedAgainOnlyABoundedNumberOfTimes() async {
        Stub.set(Self.manifestA, json: #"{"version":1,"landed":null}"#)
        let model = model()
        report(model, stop)
        for index in 0..<10 {
            await settle()
            push(model, revision: "g\(index)")
        }
        await settle()
        XCTAssertEqual(Stub.count(Self.manifestA), WorldPhotoFloorplanModel.landingRetries)
    }

    /// A late push after Stop: the Tower still reports the walk, and the
    /// phone asks nothing -- nothing is received any more.
    func testALatePushAfterStopAsksNothing() async {
        let model = model()
        push(model, revision: "g1", receiving: false)
        report(model, stop, receiving: false)
        await settle()
        XCTAssertEqual(Stub.requests, [])
        XCTAssertNil(model.plan.walk)
    }

    /// A cartridge switch builds a fresh view model and with it a fresh
    /// model: the old one's answer, still out, reaches nothing, and the new
    /// one holds nothing until its own landing.
    func testACartridgeSwitchStartsEmpty() async {
        Stub.set(delay: 0.6, for: Self.manifestA)
        var old: WorldPhotoFloorplanModel? = model()
        weak let gone = old
        report(old!, stop)
        await expect { Stub.count(Self.manifestA) == 1 }
        old = nil
        XCTAssertNil(gone, "the old model outlived its view model")
        let fresh = model()
        try? await Task.sleep(for: .milliseconds(900))
        XCTAssertNil(fresh.plan.walk)
        XCTAssertEqual(fresh.plan.value, WorldPhotoFloorplan())
        XCTAssertEqual(Stub.count(Self.tileA), 0, "the old model's answer asked for a raster")
    }

    /// The world left (inspection, forget): nothing held, nothing out.
    func testResetForgetsTheWalk() async {
        let model = model()
        report(model, stop)
        await expect { model.plan.value.rasters[F.digest] != nil }
        model.reset()
        XCTAssertNil(model.plan.walk)
        XCTAssertEqual(model.plan.value, WorldPhotoFloorplan())
    }
}
