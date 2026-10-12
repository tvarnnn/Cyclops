//
//  WorldPhotoFloorplanTests.swift
//  GlassesTests
//
//  FOW v1.2 photo floor plan (FOW-V12-PHOTO-FLOORPLAN-DESIGN, "Wire,
//  identity and iOS"): the manifest and tile read fail-closed, the raster
//  verified against its SHA-256, drawn as a gray plate plus the raster
//  (alpha 0 reads gray), OFF asks nothing, and no walk's or landing's raster
//  reaches another.
//

import CryptoKit
import SwiftUI
import UIKit
import XCTest
@testable import Glasses

/// A synthetic 512 x 512 RGBA raster -- orange, its bottom-right quadrant
/// alpha 0 -- and a manifest naming it for §9's stop landing.
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

    /// §9's stop landing; the stop block's basis; its 16-unit square padded
    /// 15% to 18.4.
    static func manifest(solvedAt: String = "1791240313.0", revision: String = "g-stop-984",
                         digest: String? = nil, bytes: Int? = nil, component: String? = nil) -> String {
        let row = component ?? #"{"reference_segment":0,"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"raster_bounds_xy":[-9.2,-9.2,9.2,9.2],"width":512,"height":512,"digest":"\#(digest ?? self.digest)","bytes":\#(bytes ?? raster.count),"projected_keyframes":2,"imagery_source":"redacted"}"#
        return #"{"version":1,"landed":{"solved_at":\#(solvedAt),"geometry_revision":"\#(revision)","frame_revision":1,"horizon_keyframes":984,"components":[\#(row)]}}"#
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

    // MARK: The switch

    func testTheSwitchIsOnOnlyWhenSaidOnExactly() {
        XCTAssertEqual(WorldPhotoFloorplanSwitch(environment: [:]), .off)
        XCTAssertEqual(WorldPhotoFloorplanSwitch(environment: ["IOS_FOW_PHOTO_FLOORPLAN": "ON"]), .off)
        XCTAssertEqual(WorldPhotoFloorplanSwitch(environment: ["IOS_FOW_PHOTO_FLOORPLAN": "1"]), .off)
        XCTAssertEqual(WorldPhotoFloorplanSwitch(environment: ["IOS_FOW_PHOTO_FLOORPLAN": "on"]),
                       WorldPhotoFloorplanSwitch(on: true))
        XCTAssertEqual(WorldPhotoFloorplanSwitch(environment: [:], info: ["GlassesFOWPhotoFloorplan": ""]), .off)
        XCTAssertEqual(WorldPhotoFloorplanSwitch(environment: [:], info: ["GlassesFOWPhotoFloorplan": "on"]),
                       WorldPhotoFloorplanSwitch(on: true))
        // The build carries the key for a home-screen launch; unset, it is off.
        XCTAssertNotNil(Bundle.main.infoDictionary?["GlassesFOWPhotoFloorplan"])
        XCTAssertEqual(WorldPhotoFloorplanSwitch.current, .off, "the test build turned it on")
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

// MARK: - Fetching: OFF asks nothing; no walk or landing reaches another

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

    private func model(on: Bool = true) -> WorldPhotoFloorplanModel {
        WorldPhotoFloorplanModel(floorplanSwitch: WorldPhotoFloorplanSwitch(on: on),
                                 client: WorldPhotoFloorplanClient(baseURL: URL(string: "http://tower.test")!,
                                                                   session: Stub.makeSession(), timeout: 5))
    }

    private func report(_ model: WorldPhotoFloorplanModel, _ coverage: WorldCoverage, walk: WorldFinishWalk? = nil,
                        receiving: Bool = true) {
        model.coverageReported(WalkScoped(walk: walk ?? Self.walkA, value: coverage), receiving: receiving)
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

    /// OFF: no request, whatever is pushed, and nothing held.
    func testOffAsksNothing() async {
        let off = model(on: false)
        report(off, stop)
        report(off, mid)
        for revision in ["g1", "g2", "g3"] { push(off, revision: revision) }
        await settle()
        XCTAssertEqual(Stub.requests, [], "OFF made a request")
        XCTAssertNil(off.plan.walk)
        XCTAssertEqual(off.plan.value, WorldPhotoFloorplan())
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
        weak var gone = old
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
