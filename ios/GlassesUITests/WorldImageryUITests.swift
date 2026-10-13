//
//  WorldImageryUITests.swift
//  GlassesUITests
//
//  FOW v1.1 B (FOW-V11B-WIRE-AND-IOS-SPEC-20261007 §5), against the mock
//  Tower serving the spec's §4 fixtures on the manifest and tile routes:
//  both switches off draw nothing and ask nothing (acceptance 2); B1's strip
//  shows one thumbnail and clears at a segment bump, and goes at Stop for
//  good (acceptance 3); B2's thumbnail sits inside its piece and a tap says
//  where it looked and how old it is (acceptance 4).
//

import CryptoKit
import UIKit
import XCTest

final class WorldImageryUITests: XCTestCase {

    private var app: XCUIApplication!
    private var mock: MockTowerHTTPServer!
    private var socket: PanelSocket!
    private var mockAuthority = ""
    private var seq = 10

    private static let liveDigest = "a3f1" + String(repeating: "0", count: 56) + "e02c"
    private static let liveDigest2 = "a3f2" + String(repeating: "0", count: 56) + "e02c"
    private static let landedDigest = "7cde" + String(repeating: "0", count: 56) + "1190"
    private static let liveFixture = #"{"version":1,"live":{"segment_index":812,"entries":[{"keyframe_id":"s9:4417","digest":"\#(liveDigest)","redaction":"redacted","pose_source":"segment_local","component_reference_segment":null,"segment_index":812}]}}"#
    private static let landedFixture = #"{"version":1,"landed":{"solved_at":1791240313.0,"geometry_revision":"g-stop-984","frame_revision":1,"entries":[{"keyframe_id":"s9:120","digest":"\#(landedDigest)","redaction":"redacted","pose_source":"landed_global_solve","component_reference_segment":0,"segment_index":null}]}}"#
    /// §9's stop block, solved 12 s before this report was sent.
    private static let sentAfterStopSolve = 1791240325.0

    override func setUpWithError() throws {
        continueAfterFailure = false
        mock = try MockTowerHTTPServer(.tower)
        mockAuthority = "127.0.0.1:\(try mock.start())"
        socket = PanelSocket(mock: mock)
        socket.pinnedReply = { subscription, world, session in
            PanelReport.savedWorld(subscription: subscription, world: world, session: session)
        }
        socket.install()
        mock.setRoute(CaptureHealthUITests.sessionStart, status: 200, body: DeadEndsUITests.session(state: "active"))
        mock.setRoute(CaptureHealthUITests.sessionStop, status: 200, body: DeadEndsUITests.session(state: "stopped"))
        // Every imagery route answers, so an OFF phone that asked would be seen asking.
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: Self.liveFixture)
        for (digest, color) in [(Self.liveDigest, UIColor.systemTeal), (Self.liveDigest2, .systemPink),
                                (Self.landedDigest, .systemOrange)] {
            mock.setDataRoute("GET \(Self.imagery)/tile/\(digest)", data: Self.jpeg(color), contentType: "image/jpeg")
        }
        // FOW v1.2: the photo floor plan's routes answer too, for §9's stop landing.
        mock.setRoute("GET \(Self.photo)/manifest", status: 200, body: Self.photoManifest)
        mock.setDataRoute("GET \(Self.photo)/tile/\(Self.photoDigest)", data: Self.photoRaster, contentType: "image/png")
    }

    override func tearDownWithError() throws {
        app?.terminate()
        mock?.stop()
    }

    private static let imagery = "/worlds/w1/guidance/s9/imagery"
    private static let photo = "/worlds/w1/guidance/s9/photo-floorplan"

    /// A 512 x 512 RGBA raster, orange with its bottom-right quadrant alpha 0.
    private static let photoRaster: Data = {
        let size = CGSize(width: 512, height: 512)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        format.preferredRange = .standard
        return UIGraphicsImageRenderer(size: size, format: format).pngData { context in
            context.cgContext.setFillColor(CGColor(srgbRed: 224 / 255, green: 112 / 255, blue: 32 / 255, alpha: 1))
            context.cgContext.fill(CGRect(origin: .zero, size: size))
            context.cgContext.clear(CGRect(x: 256, y: 256, width: 256, height: 256))
        }
    }()
    private static let photoDigest = SHA256.hash(data: photoRaster).map { String(format: "%02x", $0) }.joined()
    /// §9's stop landing and basis; the 16-unit square padded 15%.
    private static let photoManifest = #"{"version":1,"landed":{"solved_at":1791240313.0,"geometry_revision":"g-stop-984","frame_revision":1,"horizon_keyframes":984,"components":[{"reference_segment":0,"origin_xyz":[0.0,0.0,0.0],"up_xyz":[0.0,0.0,1.0],"forward_xyz":[0.0,1.0,0.0],"raster_bounds_xy":[-9.2,-9.2,9.2,9.2],"width":512,"height":512,"digest":"\#(photoDigest)","bytes":\#(photoRaster.count),"projected_keyframes":2,"imagery_source":"redacted"}]}}"#

    // MARK: Acceptance 2: OFF

    /// Both switches off: the map and the current piece as today, and no
    /// strip, no thumbnail, no request to an imagery route.
    func testBothSwitchesOffDrawNoImageryAndAskNothing() throws {
        launch(environment: [:])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        XCTAssertTrue(element("wb-current-piece").waitForExistence(timeout: 15), "today's current piece")
        XCTAssertTrue(element("wb-panel-map").waitForExistence(timeout: 10), "today's map")
        push(geometry: "gA", guidance: FOWGuidance.stop)
        Thread.sleep(forTimeInterval: 2)
        XCTAssertFalse(element("wb-imagery-live").exists, "a strip with B1 off")
        XCTAssertFalse(element("wb-imagery-live-thumb").exists)
        XCTAssertFalse(element("wb-imagery-landed-thumb").exists, "a thumbnail with B2 off")
        let asked = mock.requestLines.filter { $0.contains("/imagery/") }
        XCTAssertEqual(asked, [], "OFF asked the imagery routes")
        XCTAssertFalse(element("wb-panel-map-photo").exists, "a photo floor plan with its switch off")
        XCTAssertFalse(element("wb-panel-dial").exists, "a capture dial with its switch off")
        XCTAssertFalse(element("wb-panel-tab").exists, "a Dial|Map control with the dial's switch off")
        XCTAssertTrue(element("wb-panel-map-piece").exists, "today's fog piece")
        XCTAssertEqual(mock.requestLines.filter { $0.contains("/photo-floorplan/") }, [],
                       "OFF asked the photo floor plan routes")
        XCTAssertTrue(mock.requestLines.contains { $0.contains("/geometry/manifest") }, "the geometry was fetched")
        shoot("imagery-off")
    }

    // MARK: Acceptance 3: B1

    /// B1 on with §4's live fixture: one thumbnail, never on the map; a
    /// segment bump clears it; and after Stop it is gone, a late report of
    /// the walk never bringing it back, the loading view drawing none.
    func testTheLiveStripShowsOneThumbnailClearsAtABumpAndGoesAtStop() throws {
        launch(environment: ["IOS_FOW_ROOM_IMAGERY_LIVE": "on"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let strip = element("wb-imagery-live")
        let thumbs = app.descendants(matching: .any).matching(identifier: "wb-imagery-live-thumb")
        XCTAssertTrue(waitFor(timeout: 15) { thumbs.count == 1 }, "one thumbnail: \(thumbs.count)")
        XCTAssertEqual(text(element("wb-imagery-live-label")), "Current piece · unplaced")
        XCTAssertEqual(thumbs.firstMatch.label, "current piece photo, unplaced")
        let map = element("wb-panel-map")
        XCTAssertTrue(map.waitForExistence(timeout: 10), "the map")
        XCTAssertFalse(map.frame.intersects(strip.frame), "the strip is drawn against the map")
        XCTAssertFalse(element("wb-imagery-landed-thumb").exists, "a landed thumbnail with B2 off")
        reveal(strip)
        shoot("imagery-live-812")

        // A tracking break: segment 813 is current and its strip starts empty.
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200,
                      body: #"{"version":1,"live":{"segment_index":813,"entries":[]}}"#)
        serve(revision: "gB", segments: [Self.placedZero, Self.live812, Self.live813])
        push(geometry: "gB", guidance: FOWGuidance.stop)
        XCTAssertTrue(waitFor(timeout: 15) { thumbs.count == 0 }, "cleared at the bump: \(thumbs.count)")
        XCTAssertTrue(element("wb-imagery-live-empty").waitForExistence(timeout: 5), "the new strip, empty")
        shoot("imagery-live-813-empty")

        // 813's first photo, then Stop.
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: #"{"version":1,"live":{"segment_index":813,"entries":[{"keyframe_id":"s9:4500","digest":"\#(Self.liveDigest2)","redaction":"redacted","pose_source":"segment_local","component_reference_segment":null,"segment_index":813}]}}"#)
        serve(revision: "gC", segments: [Self.placedZero, Self.live812, Self.live813])
        push(geometry: "gC", guidance: FOWGuidance.stop)
        XCTAssertTrue(waitFor(timeout: 15) { thumbs.count == 1 }, "813's photo")
        let stop = app.buttons["Stop capture"]
        reveal(stop)
        XCTAssertTrue(tap(stop, until: !self.app.buttons["Stop capture"].exists), "Stop")
        XCTAssertTrue(waitFor(timeout: 5) { !strip.exists }, "the strip after the local Stop")
        // The Tower's late reports of the stopped walk, a new revision among them.
        serve(revision: "gD", segments: [Self.placedZero, Self.live812, Self.live813])
        for round in 0..<3 {
            push(geometry: round == 0 ? "gD" : "gC", guidance: FOWGuidance.stop)
            XCTAssertTrue(neverExists(strip, for: 1), "the strip came back after Stop (\(round))")
        }
        push(modelState: "finalizing", geometry: "gD", guidance: FOWGuidance.stop, buildInProgress: true)
        XCTAssertTrue(neverExists(strip, for: 1.5), "the strip in the loading view")
        XCTAssertFalse(element("wb-imagery-live-thumb").exists)
        XCTAssertFalse(element("wb-imagery-landed-thumb").exists)
        shoot("imagery-live-after-stop")
    }

    // MARK: Acceptance 4: B2

    /// B2 on with §4's landed fixture beside §9's stop block: the thumbnail
    /// inside its piece, a tap saying where it looked and how old it is; a
    /// newer landing's grid drops it.
    func testTheLandedThumbnailSitsInItsPieceAndATapSaysDirectionAndAge() throws {
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: Self.landedFixture)
        launch(environment: ["IOS_FOW_ROOM_IMAGERY_LANDED": "on"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let thumb = element("wb-imagery-landed-thumb")
        XCTAssertTrue(thumb.waitForExistence(timeout: 15), "the landed thumbnail")
        let piece = element("wb-panel-map-piece")
        XCTAssertTrue(piece.waitForExistence(timeout: 5), "its piece")
        print("IMAGERY-LANDED|thumb=\(thumb.frame)|piece=\(piece.frame)")
        XCTAssertTrue(piece.frame.insetBy(dx: -0.5, dy: -0.5).contains(thumb.frame),
                      "the thumbnail \(thumb.frame) outside its piece \(piece.frame)")
        // The §9 fixture is at (3.5, 5.5) on an eight-cell piece. The fog
        // field fits thirteen cells (2.5 padding on each side), but that fit
        // must move only the centre: it must not shrink the photo or its hit box.
        let side = piece.frame.width
        let expectedWidth = min(max(side / 8 * 1.6, 28), 64)
        XCTAssertEqual(thumb.frame.width, expectedWidth, accuracy: 1,
                       "the padded fog fit shrank the B2 thumbnail")
        XCTAssertEqual(thumb.frame.height, expectedWidth * 9 / 16, accuracy: 1)
        XCTAssertEqual(thumb.frame.midX, piece.frame.minX + side * 6 / 13, accuracy: 1,
                       "the thumbnail moved away from its mapped column")
        XCTAssertEqual(thumb.frame.midY, piece.frame.minY + side * 8 / 13, accuracy: 1,
                       "the thumbnail moved away from its mapped row")
        XCTAssertTrue(thumb.label.hasPrefix("redacted photo, look direction 90°, "), thumb.label)
        XCTAssertTrue(thumb.label.contains("seconds old"), thumb.label)
        XCTAssertEqual(text(piece), "Piece 1", "the v1 piece's label changed")
        XCTAssertFalse(element("wb-imagery-live").exists, "a strip with B1 off")
        reveal(thumb)
        shoot("imagery-landed")

        XCTAssertTrue(tap(thumb, until: self.element("wb-imagery-landed-direction").exists), "the sheet")
        XCTAssertEqual(text(element("wb-imagery-landed-direction")), "Look direction 90°")
        let age = text(element("wb-imagery-landed-age"))
        XCTAssertTrue(age.hasSuffix("seconds old") || age.hasSuffix("seconds old, stale"), age)
        shoot("imagery-landed-sheet")

        // A newer landing on the map -- the same grid, re-dated -- while the
        // Tower's manifest still carries the stop landing's thumbnails (its
        // job is slow): none is drawn on the newer landing's map, and the
        // open sheet of the old one closes.
        let newer = FOWGuidance.stop.replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320.")
        push(geometry: "gA", guidance: newer)
        XCTAssertTrue(waitFor(timeout: 10) { !self.element("wb-imagery-landed-direction").exists },
                      "a superseded landing's sheet stayed open")
        XCTAssertTrue(waitFor(timeout: 10) { !thumb.exists }, "a superseded landing's thumbnail")
        XCTAssertTrue(element("wb-panel-map-piece").exists, "the map stays")
        XCTAssertTrue(neverExists(thumb, for: 2), "the superseded thumbnail came back")
    }

    // MARK: FOW-RING: the capture dial's photos

    /// IOS_FOW_DIAL=on alone -- it asks for B2's tiles itself -- and §4's
    /// landed fixture: today's join places `s9:120` looking 90°, sector 3, a
    /// captured direction of §9's stop piece, so it is the dial's photo at
    /// 3 o'clock (start on top). Round 3: the same photo leads the "Recently
    /// captured" strip; a tap on either opens it large, with Close; no
    /// sentence of guidance; IOS_FOW_MAP_TAB unset, so no Dial|Map control.
    func testTheDialShowsTodaysLandedPhotoInItsSegment() throws {
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: Self.landedFixture)
        launch(environment: ["IOS_FOW_DIAL": "on"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let dial = element("wb-panel-dial")
        let photo = element("wb-panel-dial-photo")
        XCTAssertTrue(photo.waitForExistence(timeout: 20), "the dial's photo")
        XCTAssertEqual(app.descendants(matching: .any).matching(identifier: "wb-panel-dial-photo").count, 1)
        XCTAssertEqual(photo.label, "Photo at 3 o'clock")
        XCTAssertTrue(dial.exists)
        print("DIAL-PHOTO|dial=\(dial.frame)|photo=\(photo.frame)")
        XCTAssertGreaterThan(photo.frame.midX, dial.frame.midX + dial.frame.width * 0.2, "at 3 o'clock")
        XCTAssertEqual(photo.frame.midY, dial.frame.midY, accuracy: 4, "at 3 o'clock")
        XCTAssertEqual(dial.value as? String,
                       "4 of 12 directions captured; missing around 5 to 12 o'clock from where you started")
        XCTAssertFalse(element("wb-panel-dial-photos-unavailable").exists)
        XCTAssertFalse(element("wb-panel-dial-gap").exists, "no sentence of guidance")
        XCTAssertTrue(text(element("wb-panel-dial-stamp")).hasPrefix("4 of 12 directions · "),
                      text(element("wb-panel-dial-stamp")))
        reveal(dial)
        shoot("dial-photo-s2")
        let sheet = element("wb-panel-dial-photo-sheet")
        let close = element("wb-panel-dial-photo-close")
        centre(photo)
        // A finger on the wedge, wherever accessibility's hit test lands.
        XCTAssertTrue(tapCentre(photo, until: sheet.exists), "a wedge's photo opens large")
        shoot("dial-photo-sheet")
        XCTAssertTrue(tap(close, until: !sheet.exists), "the sheet closed")
        let recent = element("wb-panel-dial-recent")
        XCTAssertTrue(recent.waitForExistence(timeout: 5), "the recent strip")
        XCTAssertEqual(app.descendants(matching: .any).matching(identifier: "wb-panel-dial-recent").count, 1)
        XCTAssertEqual(recent.label, "Photo 1 of 1")
        reveal(recent)
        shoot("dial-recent-strip")
        XCTAssertTrue(tap(recent, until: sheet.exists), "a strip photo opens large")
        XCTAssertTrue(tap(close, until: !sheet.exists), "the sheet closed")
        let control = app.segmentedControls.matching(identifier: "wb-panel-tab").firstMatch
        XCTAssertFalse(control.exists, "IOS_FOW_MAP_TAB unset: no Dial|Map control")
        XCTAssertFalse(element("wb-panel-map").exists, "the dial is the only face")

        // After Stop: the same card -- its fixed title, no control -- the
        // walk's dial dimmed with the photo it showed, the strip kept, the
        // same value, and the progress in the footer, not over the ring.
        let stop = app.buttons["Stop capture"]
        centre(stop)
        XCTAssertTrue(tap(stop, until: !self.app.buttons["Stop capture"].exists), "Stop")
        push(modelState: "finalizing", geometry: "gA", guidance: FOWGuidance.stop, buildInProgress: true)
        XCTAssertTrue(element("wb-finish-line").waitForExistence(timeout: 15), "the progress, in the footer")
        XCTAssertEqual(text(element("wb-panel-dial-title")), "Room coverage", "no switch to the headline")
        XCTAssertFalse(element("wb-panel-headline").exists)
        XCTAssertFalse(control.exists, "no Dial|Map control after Stop either")
        XCTAssertFalse(element("wb-panel-dial-mini").exists, "no mini dial after Stop")
        XCTAssertTrue(element("wb-panel-dial-photo").exists, "the photo shown live stays on its segment")
        XCTAssertTrue(element("wb-panel-dial-recent").exists, "the strip stays after Stop")
        XCTAssertEqual(dial.value as? String,
                       "4 of 12 directions captured; missing around 5 to 12 o'clock from where you started")
        reveal(element("wb-panel-stage"))
        shoot("dial-photo-after-stop")
    }

    /// S3's additive fields: the Tower's own bin, `latest` and the walk cue.
    /// Round 3: start stays on top whatever `latest` says, so the photo the
    /// Tower binned at sector 3 sits at 3 o'clock; no sentence of guidance.
    func testS3LatestNeverTurnsTheDial() throws {
        let s3 = Self.landedFixture.replacingOccurrences(
            of: #""segment_index":null}]"#,
            with: #""segment_index":null,"sector":3}],"latest":{"component_reference_segment":0,"sector":3,"keyframe_id":"s9:120"},"motion_sector":4"#)
        mock.setRoute("GET \(Self.imagery)/manifest", status: 200, body: s3)
        launch(environment: ["IOS_FOW_DIAL": "on"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let dial = element("wb-panel-dial")
        let photo = element("wb-panel-dial-photo")
        XCTAssertTrue(photo.waitForExistence(timeout: 20), "the dial's photo")
        XCTAssertEqual(photo.label, "Photo at 3 o'clock", "start on top, never the newest view")
        XCTAssertGreaterThan(photo.frame.midX, dial.frame.midX + dial.frame.width * 0.2, "at 3 o'clock")
        XCTAssertEqual(photo.frame.midY, dial.frame.midY, accuracy: 4, "at 3 o'clock")
        XCTAssertEqual(dial.label, "Capture dial")
        XCTAssertEqual(dial.value as? String,
                       "4 of 12 directions captured; missing around 5 to 12 o'clock from where you started")
        XCTAssertFalse(element("wb-panel-dial-gap").exists, "no sentence of guidance")
        reveal(dial)
        shoot("dial-photo-s3")
    }

    // MARK: FOW v1.2: the photo floor plan

    /// ON, with a raster for the stop landing: the piece is the photo -- a
    /// gray plate, the raster's pixels over it, alpha 0 reading gray -- and
    /// the legend names the gray. A newer landing holds the proved photo on
    /// its own receipt briefly, then returns to the fog field at expiry.
    func testThePhotoFloorplanHoldsThenExpiresForANewerLanding() throws {
        launch(environment: ["IOS_FOW_PHOTO_FLOORPLAN": "on", "GLASSES_UITEST_PHOTO_HOLD_SECONDS": "10"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let photo = element("wb-panel-map-photo")
        XCTAssertTrue(photo.waitForExistence(timeout: 15), "the photo floor plan")
        XCTAssertFalse(element("wb-panel-map-piece").exists, "the fog field beside its own photo")
        XCTAssertTrue(element("wb-current-piece").exists, "the current piece, its own card")
        XCTAssertFalse(photo.frame.intersects(element("wb-current-piece").frame), "the piece drawn on the photo")
        XCTAssertTrue(element("wb-panel-map-legend").label.contains("No usable photo here yet"),
                      element("wb-panel-map-legend").label)
        XCTAssertTrue(mock.requestLines.contains { $0.contains("/photo-floorplan/tile/\(Self.photoDigest)") })
        // A newer landing, the manifest still at the stop landing.
        let newer = FOWGuidance.stop.replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320.")
        push(geometry: "gA", guidance: newer)
        let updating = element("wb-panel-map-updating")
        XCTAssertTrue(updating.waitForExistence(timeout: 2), "the Updating caption during the hold")
        XCTAssertEqual(updating.label, "Updating…")
        XCTAssertTrue(photo.exists, "the valid photo remains visible")
        XCTAssertFalse(element("wb-panel-map-piece").exists, "fog appeared during the hold")
        XCTAssertTrue(neverExists(element("wb-panel-map-piece"), for: 1), "fog flashed during the hold")
        XCTAssertTrue(element("wb-panel-map-piece").waitForExistence(timeout: 15), "fog after hold expiry")
        XCTAssertFalse(photo.exists, "the held photo expired")
        XCTAssertFalse(updating.exists, "the Updating caption expired")

        // When the Tower catches up, the newer plan takes over. Reveal it
        // before sampling pixels: an offscreen XCUIElement screenshot is black.
        let newerManifest = Self.photoManifest
            .replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320.")
        mock.setRoute("GET \(Self.photo)/manifest", status: 200, body: newerManifest)
        push(geometry: "gA", guidance: newer)
        XCTAssertTrue(photo.waitForExistence(timeout: 15), "the newer photo replaced the fog")
        XCTAssertFalse(element("wb-panel-map-piece").exists)
        reveal(photo)
        shoot("photo-floorplan")
        let shot = photo.screenshot().image
        let orange = try XCTUnwrap(Self.pixel(shot, at: CGPoint(x: 0.25, y: 0.25)))
        let gray = try XCTUnwrap(Self.pixel(shot, at: CGPoint(x: 0.75, y: 0.75)))
        print("PHOTO-FLOORPLAN|frame=\(photo.frame)|orange=\(orange)|gray=\(gray)")
        // The screenshot is display-converted: orange, not its exact sRGB.
        XCTAssertTrue(orange.r > 180 && orange.r - orange.b > 100 && orange.g < 170, "the raster: \(orange)")
        XCTAssertTrue(abs(gray.r - 0x70) < 16 && abs(gray.g - 0x75) < 16 && abs(gray.b - 0x7B) < 16,
                      "alpha 0 reads gray: \(gray)")
    }

    /// The Tower's `kept` notice for the phone's newer receipt, keeping the
    /// stop landing's plan ("regressed"): the photo stays on its own receipt,
    /// captioned, well past the hold, and no fog field appears. The notice
    /// gone, the hold is long over: the fog field.
    func testAKeptNoticeHoldsThePhotoPastTheHoldUntilItEnds() throws {
        launch(environment: ["IOS_FOW_PHOTO_FLOORPLAN": "on", "GLASSES_UITEST_PHOTO_HOLD_SECONDS": "4"])
        open(cartridge: "World Builder")
        XCTAssertTrue(waitFor(timeout: 15) { self.socket.liveSubscription != nil }, "the live subscription")
        try startCapture()
        serve(revision: "gA", segments: [Self.placedZero, Self.live812])
        push(geometry: "gA", guidance: FOWGuidance.stop)
        let photo = element("wb-panel-map-photo")
        let fog = element("wb-panel-map-piece")
        XCTAssertTrue(photo.waitForExistence(timeout: 15), "the photo floor plan")

        // The newer landing published nothing: `landed` stays the stop landing's.
        let kept = #","kept":{"for_landing":{"solved_at":1791240320.0,"geometry_revision":"g-stop-990","frame_revision":1},"reason":"regressed","plan_landing":{"solved_at":1791240313.0,"geometry_revision":"g-stop-984","frame_revision":1}}}"#
        mock.setRoute("GET \(Self.photo)/manifest", status: 200, body: String(Self.photoManifest.dropLast()) + kept)
        let newer = FOWGuidance.stop.replacingOccurrences(of: "g-stop-984", with: "g-stop-990")
            .replacingOccurrences(of: "1791240313.", with: "1791240320.")
        push(geometry: "gA", guidance: newer)
        let notImproved = element("wb-panel-map-not-improved")
        XCTAssertTrue(notImproved.waitForExistence(timeout: 10), "the kept caption")
        XCTAssertEqual(notImproved.label, "Couldn't improve this view yet")
        XCTAssertTrue(neverExists(fog, for: 12), "fog while the Tower keeps the plan (hold is 4 s)")
        XCTAssertTrue(photo.exists, "the kept photo")
        XCTAssertTrue(notImproved.exists, "the kept caption")
        XCTAssertFalse(element("wb-panel-map-updating").exists, "Updating… for a regressed landing")
        // A heartbeat asks again; the Tower still keeps it.
        let asked = mock.requestLines.filter { $0.contains("/photo-floorplan/manifest") }.count
        push(geometry: "gA", guidance: newer)
        XCTAssertTrue(neverExists(fog, for: 2), "fog after asking again while kept")
        XCTAssertGreaterThan(mock.requestLines.filter { $0.contains("/photo-floorplan/manifest") }.count, asked,
                             "a kept landing was not asked again")
        XCTAssertTrue(notImproved.exists)
        reveal(photo)
        shoot("photo-floorplan-kept")

        // The notice ends with no plan for the newer landing: today's expiry.
        mock.setRoute("GET \(Self.photo)/manifest", status: 200, body: Self.photoManifest)
        push(geometry: "gA", guidance: newer)
        XCTAssertTrue(fog.waitForExistence(timeout: 15), "fog once the notice ended")
        XCTAssertFalse(photo.exists, "the held photo after the notice ended")
        XCTAssertFalse(notImproved.exists, "the kept caption after the notice ended")
        XCTAssertFalse(element("wb-panel-map-updating").exists, "the hold had long ended")
    }

    private static func pixel(_ image: UIImage, at unit: CGPoint) -> (r: Int, g: Int, b: Int)? {
        guard let cg = image.cgImage else { return nil }
        var rgba = [UInt8](repeating: 0, count: 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: 1, height: 1, bitsPerComponent: 8,
                                          bytesPerRow: 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            let x = CGFloat(cg.width) * unit.x, y = CGFloat(cg.height) * (1 - unit.y)
            context.draw(cg, in: CGRect(x: -x, y: -y, width: CGFloat(cg.width), height: CGFloat(cg.height)))
            return true
        }
        return drawn ? (Int(rgba[0]), Int(rgba[1]), Int(rgba[2])) : nil
    }

    // MARK: The mock Tower's geometry

    private typealias Segment = (index: Int, hash: String, placed: Bool, poses: [String])

    private static func pose(_ id: String, _ t: [Double], yaw: Double = 0) -> String {
        let q = [cos(yaw / 2), 0, sin(yaw / 2), 0]
        return #"{"keyframe_id":"\#(id)","status":"solved","degeneracy":"","rotation":\#(q),"translation":\#(t)}"#
    }

    /// Placed in reference 0, frame 1, identity: `s9:120` at (3, -1, 0)
    /// looking along +x -- in §9's stop grid, station (3, 5), 90 degrees.
    private static let placedZero: Segment = (0, "h0", true, [pose("s9:120", [3, -1, 0], yaw: .pi / 2),
                                                              pose("s9:121", [3.2, -1, 0], yaw: .pi / 2)])
    private static let live812: Segment = (812, "h812", false, [pose("s9:4416", [0, 0, 0]),
                                                                pose("s9:4417", [0.4, 0, 0.8], yaw: 0.4)])
    private static let live813: Segment = (813, "h813", false, [pose("s9:4499", [0, 0, 0]),
                                                                pose("s9:4500", [-0.5, 0, 0.6], yaw: -0.6)])

    private static let contract = "world_builder.geometry/2026-08-25"

    private static func placement(_ placed: Bool) -> String {
        placed
            ? #""registered":true,"transform_to_world":{"rotation_wxyz":[1.0,0.0,0.0,0.0],"translation":[0.0,0.0,0.0],"scale":1.0,"reference_segment":0,"frame_revision":1}"#
            : #""registered":false,"transform_to_world":null"#
    }

    private func serve(revision: String, segments: [Segment]) {
        let rows = segments.map { segment in
            #"{"segment_index":\#(segment.index),"content_hash":"\#(segment.hash)","#
                + #""frame_id":"segment:\#(segment.index)",\#(Self.placement(segment.placed)),"#
                + #""resolution_state":"resolved","dominant_degeneracy":null,"#
                + #""keyframe_count":\#(segment.poses.count),"solved_count":\#(segment.poses.count),"#
                + #""point_count":1,"bounds":{"min":[0.0,0.0,0.0],"max":[1.0,1.0,1.0]}}"#
        }
        mock.setRoute("GET /worlds/w1/geometry/manifest", status: 200, body:
            #"{"contract":"\#(Self.contract)","world_id":"w1","session_id":"s9","#
            + #""geometry_revision":"\#(revision)","current":true,"pose_convention":{"pose_type":"T_world_camera","#
            + #""quaternion_order":"wxyz","handedness":"right","camera_axes":"opencv_x_right_y_down_z_forward","#
            + #""translation_units":"world","world_axes_origin":"first_keyframe_camera","up_axis":"unknown","#
            + #""pose_dtype":"float64","point_dtype":"float32"},"segment_count":\#(segments.count),"#
            + #""segments":[\#(rows.joined(separator: ","))]}"#)
        for segment in segments {
            mock.setRoute("GET /worlds/w1/geometry/segment/\(segment.index)", status: 200, body:
                #"{"contract":"\#(Self.contract)","segment_index":\#(segment.index),"#
                + #""content_hash":"\#(segment.hash)","frame_id":"segment:\#(segment.index)","#
                + #"\#(Self.placement(segment.placed)),"#
                + #""poses":[\#(segment.poses.joined(separator: ","))],"points":[[0.5,0.5,0.5]],"#
                + #""points_sent":1,"points_total":1,"point_sampling":"none"}"#)
        }
    }

    private func push(modelState: String = "receiving", geometry: String, guidance: String? = nil,
                      buildInProgress: Bool? = nil) {
        seq += 1
        let seq = self.seq
        socket.sendLive { subscription in
            PanelReport.text(subscription: subscription, seq: seq, modelState: modelState, world: "w1",
                             session: "s9", buildInProgress: buildInProgress, liveCapture: true,
                             guidance: guidance, accepted: 984, towerSentAt: Self.sentAfterStopSolve)
                .replacingOccurrences(of: #""revision":"g1"}"#, with: #""revision":"\#(geometry)"}"#)
        }
        Thread.sleep(forTimeInterval: 0.7)
    }

    private static func jpeg(_ color: UIColor) -> Data {
        let size = CGSize(width: 160, height: 90)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).jpegData(withCompressionQuality: 0.6) { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            UIColor.white.setFill()
            context.fill(CGRect(x: 60, y: 30, width: 40, height: 30))
        }
    }

    // MARK: Driving the app

    private func launch(environment: [String: String]) {
        let app = XCUIApplication()
        app.launchArguments = ["-UITestSkipOnboarding", "-UITestResetTowerAddress", "-UITestMockGlasses"]
        app.launchEnvironment["GLASSES_UITEST_AWAITING_BOUND_SECONDS"] = "600"
        switch DeadEndsUITests.cameraFeed {
        case .success(let feed): app.launchEnvironment["GLASSES_UITEST_MOCK_CAMERA_FEED"] = feed.path
        case .failure(let problem): XCTFail("the mock glasses' camera feed: \(problem.text)")
        }
        app.launchEnvironment["GLASSES_TOWER_AUTHORITY"] = mockAuthority
        for (key, value) in environment { app.launchEnvironment[key] = value }
        addUIInterruptionMonitor(withDescription: "system alert") { alert in
            for title in ["Allow", "Don't Allow", "OK", "Not Now"] where alert.buttons[title].exists {
                alert.buttons[title].tap()
                return true
            }
            return false
        }
        app.launch()
        self.app = app
    }

    private func startCapture() throws {
        let start = app.buttons["Start capture"]
        guard waitFor(timeout: 15, { start.exists && start.isEnabled }) else {
            throw XCTSkip("Mock Device Kit gave no active device here")
        }
        start.tap()
        XCTAssertTrue(app.buttons["Stop capture"].waitForExistence(timeout: 15), "the capture started")
    }

    private func open(cartridge name: String) {
        let cartridges = app.buttons["Cartridges"]
        XCTAssertTrue(cartridges.waitForExistence(timeout: 15), "the shell's Cartridges button")
        let drawerDone = app.buttons["Done"]
        XCTAssertTrue(tap(cartridges, until: drawerDone.exists), "the cartridge drawer opened")
        app.raiseTheCartridgeDrawerOnAnIPad()
        let row = app.buttons.containing(NSPredicate(format: "label BEGINSWITH %@", name)).firstMatch
        var found = waitFor(timeout: 3) { row.exists && row.isHittable }
        for _ in 0..<6 where !found {
            app.swipeUp(velocity: .slow)
            found = waitFor(timeout: 1) { row.exists && row.isHittable }
        }
        XCTAssertTrue(found, "a drawer row for \(name)")
        XCTAssertTrue(tap(row, until: !drawerDone.exists && app.navigationBars[name].exists),
                      "the \(name) workspace opened")
    }

    private func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func text(_ element: XCUIElement) -> String {
        element.exists ? element.label : "(gone)"
    }

    private func reveal(_ element: XCUIElement) {
        for _ in 0..<12 {
            let window = app.windows.firstMatch.frame
            let status = self.element("shell-status-bar")
            let top = max(window.minY + 100, status.exists ? status.frame.maxY + 16 : 0)
            let bottom = window.maxY - 40
            if element.exists {
                let frame = element.frame
                if element.isHittable && frame.minY >= top && frame.maxY <= bottom { return }
                if frame.minY < top {
                    app.swipeDown(velocity: .slow)
                    continue
                }
            }
            app.swipeUp(velocity: .slow)
        }
    }

    /// Small held drags until `element` sits mid-window: a slow swipe can
    /// step right over a small target under the status bar.
    private func centre(_ element: XCUIElement) {
        for _ in 0..<6 {
            guard element.exists else { return }
            let window = app.windows.firstMatch
            let delta = window.frame.midY - element.frame.midY
            if abs(delta) < 60 { return }
            let start = window.coordinate(withNormalizedOffset: CGVector(dx: 0.1, dy: 0.6))
            start.press(forDuration: 0.1, thenDragTo: start.withOffset(CGVector(dx: 0, dy: max(-250, min(250, delta)))),
                        withVelocity: .slow, thenHoldForDuration: 0.3)
        }
    }

    /// A touch at the centre of `element`'s frame, as a finger would.
    private func tapCentre(_ element: XCUIElement, until effect: @autoclosure () -> Bool) -> Bool {
        for _ in 0..<4 {
            if element.waitForExistence(timeout: 5) {
                element.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
            }
            if waitFor(timeout: 4, effect) { return true }
        }
        return effect()
    }

    @discardableResult
    private func tap(_ element: XCUIElement, until effect: @autoclosure () -> Bool, attempts: Int = 4) -> Bool {
        for _ in 0..<attempts {
            if element.waitForExistence(timeout: 5), element.isHittable { element.tap() }
            if waitFor(timeout: 4, effect) { return true }
        }
        return effect()
    }

    private func neverExists(_ element: XCUIElement, for seconds: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if element.exists { return false }
            Thread.sleep(forTimeInterval: 0.1)
        }
        return !element.exists
    }

    private func waitFor(timeout: TimeInterval = 10, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return condition()
    }

    private func shoot(_ name: String) {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        // As WorldInlinePanelUITests: to UINLINE_SHOTS_DIR when it is set.
        guard let dir = ProcessInfo.processInfo.environment["UINLINE_SHOTS_DIR"], !dir.isEmpty else { return }
        try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
    }
}
