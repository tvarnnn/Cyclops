//
//  WorldCaptureDialSnapshotTests.swift
//  GlassesTests
//
//  FOW-RING-SPEC §7: the capture dial's snapshots, for the owner's review.
//  Each renders the REAL walking panel (`WorldInlinePanel`, stage and
//  footer) in a window at an iPhone's width, from a real coverage block,
//  with 160 x 90 tiles drawn here as stand-ins for the Tower's redacted
//  keyframes (no real photo is committed). PNGs go to FOW_DIAL_SHOTS_DIR
//  (pass TEST_RUNNER_FOW_DIAL_SHOTS_DIR to xcodebuild) and are attached.
//

import SwiftUI
import UIKit
import XCTest
@testable import Glasses

@MainActor
final class WorldCaptureDialSnapshotTests: XCTestCase {
    private static let width: CGFloat = 402
    private static let walk = WorldFinishWalk(worldID: "w1", sessionID: "s9")

    private struct Scene {
        var name: String
        var block: String?
        var age: Double? = 12
        var latest: Int? = nil
        var latestPiece = 0
        var motion: Int? = nil
        /// Sectors with a tile; `nil`: every sector the hero has evidence in.
        var photos: [Int]? = nil
        var style: UIUserInterfaceStyle = .light
        var size: UIContentSizeCategory? = nil
        var reduceMotion = false
        var increaseContrast = false
        var phase: WorldPanelPhase = .walking(hasMap: true)
        var report = WorldPresentation()
    }

    // MARK: The shots

    func testTheShotsTheOwnerReviews() throws {
        let mid = Scene(name: "dial-light", block: "walk6-h362", latest: 0, motion: 2)
        var scenes = [mid]
        scenes.append(with(mid) { $0.name = "dial-dark"; $0.style = .dark })
        scenes.append(with(mid) { $0.name = "dial-ax5"; $0.size = .accessibilityExtraExtraExtraLarge })
        scenes.append(with(mid) { $0.name = "dial-reduce-motion"; $0.reduceMotion = true; $0.style = .dark })
        scenes.append(with(mid) { $0.name = "dial-increase-contrast"; $0.increaseContrast = true })
        scenes.append(Scene(name: "dial-two-pieces", block: "walk71-h563", latest: 10, motion: 9))
        scenes.append(with(mid) { $0.name = "dial-stale"; $0.age = 120; $0.style = .dark })
        scenes.append(Scene(name: "dial-one-view", block: "walk71-h104", latest: 1))
        scenes.append(Scene(name: "dial-s2-sparse", block: "walk7-h236", photos: [0, 9]))
        scenes.append(Scene(name: "dial-colour-only", block: "walk7-h236", age: nil, photos: []))
        scenes.append(Scene(name: "dial-all-12", block: "walk7-h991", latest: 4, style: .dark))
        scenes.append(Scene(name: "dial-before-first-landing", block: nil))
        // A live dial with a secondary piece: its mini dial in a row under
        // the ring (a "0/12" piece is never drawn).
        scenes.append(Scene(name: "dial-live-second-piece", block: "walk71-h563", latest: 10, motion: 9,
                            style: .dark))
        // After Stop: the photos shown live, dimmed, the words in the footer.
        let stopped = with(mid) {
            $0.name = "dial-post-stop"
            $0.style = .dark
            $0.phase = .finishing(target: WorldRenderTarget(worldID: "w1", sessionID: "s9"))
            $0.report = WorldPresentation(stage: .improving, finishLine: .finalPlacement)
        }
        scenes.append(stopped)
        scenes.append(with(stopped) { $0.name = "dial-post-stop-light"; $0.style = .light })
        for scene in scenes {
            try shoot(scene)
        }
        try shootMapTab()
    }

    private func with(_ scene: Scene, _ change: (inout Scene) -> Void) -> Scene {
        var copy = scene
        change(&copy)
        return copy
    }

    // MARK: Building a scene

    private func shoot(_ scene: Scene) throws {
        var source: WorldPanelMapSource?
        var overlay: WorldLandedOverlay?
        if let block = scene.block {
            let coverage = try WorldCaptureDialTests.coverage(block)
            let clock = scene.age.map { WorldTowerClock(towerSentAt: coverage.solvedAt + $0, receivedAt: .now) }
            let receipt = WorldCoverageReceipt(coverage: coverage, keyframesAccepted: coverage.horizonKeyframes + 9,
                                               clock: clock)
            source = .coverage(receipt)
            let hero = coverage.components[scene.latest == nil
                ? WorldCaptureDial(coverage: coverage).heroIndex ?? 0 : scene.latestPiece]
            let lit = (0..<12).filter { sector in
                hero.stations.contains { ($0.supportedMask | $0.weakMask) & (1 << sector) != 0 }
            }
            let entries = (scene.photos ?? lit).map { sector in
                WorldDialThumbnail(digest: String(format: "%032x", sector + 1), image: Self.tile(sector),
                                   referenceSegment: hero.referenceSegment, heading: Double(sector) * 30 + 14,
                                   sector: sector)
            }
            overlay = WorldLandedOverlay(
                receipt: receipt, thumbnails: [], dialEntries: entries,
                latest: scene.latest.map { .init(componentReferenceSegment: hero.referenceSegment, sector: $0,
                                                 keyframeID: "s9:\(coverage.horizonKeyframes)") },
                motionSector: scene.motion)
        }
        let panel = WorldInlinePanel(
            phase: scene.phase, report: WalkScoped(walk: Self.walk, value: scene.report),
            buildInProgress: nil, isTowerReachable: true, screenSaysOffline: false, showsHealth: false,
            map: source, landed: overlay,
            dial: WorldPanelDial(awaiting: source == nil),
            host: WorldInlineHost(), dismissBanner: {}, bannerAnnounced: { _ in }, expand: { _ in }
        ) { EmptyView() }
        // After Stop the stage is not a fixed height: at its ideal height, as
        // in the screen's scroll view (a 4000 pt proposal would fill it).
        try render(scene, scene.phase.isFinishing ? AnyView(panel.fixedSize(horizontal: false, vertical: true))
                                                  : AnyView(panel))
    }

    /// The Map tab: today's floor plan, unchanged, under the header's control.
    private func shootMapTab() throws {
        let coverage = try WorldCaptureDialTests.coverage("walk71-h563")
        let receipt = WorldCoverageReceipt(coverage: coverage, keyframesAccepted: 572,
                                           clock: WorldTowerClock(towerSentAt: coverage.solvedAt + 12, receivedAt: .now))
        let card = VStack(spacing: 0) {
            WorldCaptureDialHeader(tab: .constant(.map), showsToggle: true)
            WorldCaptureDialStage(dial: WorldCaptureDial(coverage: coverage), receipt: receipt, photos: .empty,
                                  tab: .map) {
                WorldCoverageMapView(coverage: coverage, isFrozen: false)
            }
            .frame(height: WorldPanelLayout.stageHeight(width: Self.width - 32, visibleHeight: 0))
            .background(Color(WorldRenderLoadingPanel.pageBackground))
            .clipShape(UnevenRoundedRectangle(topLeadingRadius: 18, topTrailingRadius: 18))
            .environment(\.colorScheme, .dark)
        }
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 18))
        try render(Scene(name: "dial-map-tab", block: nil, style: .dark), AnyView(card))
    }

    // MARK: Rendering

    private func render(_ scene: Scene, _ content: AnyView) throws {
        var root = AnyView(content
            .padding(16)
            .frame(width: Self.width)
            .background(Color(.systemGroupedBackground))
            .environment(\._accessibilityReduceMotion, scene.reduceMotion)
            .environment(\._colorSchemeContrast, scene.increaseContrast ? .increased : .standard))
        if let size = scene.size, let dynamic = DynamicTypeSize(size) {
            root = AnyView(root.environment(\.dynamicTypeSize, dynamic))
        }
        let host = UIHostingController(rootView: root)
        host.safeAreaRegions = []
        host.overrideUserInterfaceStyle = scene.style
        if let size = scene.size { host.traitOverrides.preferredContentSizeCategory = size }
        let windowScene = try XCTUnwrap(UIApplication.shared.connectedScenes.first as? UIWindowScene)
        let window = UIWindow(windowScene: windowScene)
        window.overrideUserInterfaceStyle = scene.style
        window.frame = CGRect(x: 0, y: 0, width: Self.width, height: 1400)
        window.rootViewController = host
        window.isHidden = false
        defer {
            window.isHidden = true
            window.rootViewController = nil
        }
        // Two passes: the card measures its own width, then its stage.
        for _ in 0..<2 {
            spin(0.5)
            let fitting = host.sizeThatFits(in: CGSize(width: Self.width, height: 4000))
            window.frame = CGRect(x: 0, y: 0, width: Self.width, height: ceil(fitting.height))
            host.view.frame = window.bounds
            host.view.layoutIfNeeded()
        }
        // The look-here pulses 0.35 <-> 1.0 every 1.2 s: keep the frame of one
        // period where it is strongest, so the shot shows it at full strength.
        var image = UIImage()
        var best = -1
        for _ in 0..<12 {
            spin(0.1)
            let frame = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
                window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
            }
            let amber = Self.amberPixels(frame)
            if amber > best {
                best = amber
                image = frame
            }
        }
        XCTAssertGreaterThan(image.size.height, 200, scene.name)
        let attachment = XCTAttachment(image: image)
        attachment.name = scene.name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let dir = ProcessInfo.processInfo.environment["FOW_DIAL_SHOTS_DIR"], !dir.isEmpty,
           let png = image.pngData() {
            try png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(scene.name).png"))
        }
    }

    /// Pixels near the look-here amber, `#FFB347`, on a 1/4-size copy.
    private static func amberPixels(_ image: UIImage) -> Int {
        guard let cg = image.cgImage else { return 0 }
        let width = cg.width / 4, height = cg.height / 4
        var data = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = data.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return 0 }
        var count = 0
        for i in stride(from: 0, to: data.count, by: 4) {
            let r = Int(data[i]), g = Int(data[i + 1]), b = Int(data[i + 2])
            if r > 200, g > 130, g < 210, b < 120, r - b > 110 { count += 1 }
        }
        return count
    }

    private func spin(_ seconds: TimeInterval) {
        RunLoop.main.run(until: Date().addingTimeInterval(seconds))
    }

    // MARK: Stand-in keyframes

    /// A 160 x 90 stand-in for a redacted keyframe: a wall, a floor and one
    /// thing in view, different per direction, so the ring reads like a
    /// walk around one space. Drawn here; no real photo is committed.
    static func tile(_ sector: Int) -> UIImage {
        let size = CGSize(width: 160, height: 90)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            let cg = context.cgContext
            let walls: [(CGFloat, CGFloat, CGFloat)] = [(0.90, 0.86, 0.79), (0.83, 0.85, 0.87), (0.80, 0.85, 0.78),
                                                        (0.93, 0.91, 0.87), (0.76, 0.82, 0.88), (0.88, 0.82, 0.76)]
            let wall = walls[sector % walls.count]
            func color(_ c: (CGFloat, CGFloat, CGFloat), _ k: CGFloat = 1, _ a: CGFloat = 1) -> CGColor {
                CGColor(srgbRed: c.0 * k, green: c.1 * k, blue: c.2 * k, alpha: a)
            }
            func fill(_ c: CGColor, _ rect: CGRect, radius: CGFloat = 0) {
                cg.setFillColor(c)
                cg.addPath(CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil))
                cg.fillPath()
            }
            func gradient(_ top: CGColor, _ bottom: CGColor, _ rect: CGRect) {
                cg.saveGState()
                cg.clip(to: rect)
                let g = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [top, bottom] as CFArray,
                                   locations: [0, 1])!
                cg.drawLinearGradient(g, start: CGPoint(x: rect.midX, y: rect.minY),
                                      end: CGPoint(x: rect.midX, y: rect.maxY), options: [])
                cg.restoreGState()
            }
            let horizon: CGFloat = 64 + CGFloat(sector % 3) * 3
            gradient(color(wall, 1.04), color(wall, 0.86), CGRect(x: 0, y: 0, width: 160, height: horizon))
            gradient(color((0.56, 0.40, 0.26)), color((0.38, 0.26, 0.16)),
                     CGRect(x: 0, y: horizon, width: 160, height: 90 - horizon))
            for plank in stride(from: CGFloat(-40), through: 200, by: 26) {
                cg.setStrokeColor(color((0.30, 0.20, 0.12), 1, 0.35))
                cg.setLineWidth(1)
                cg.move(to: CGPoint(x: 80 + (plank - 80) * 0.4, y: horizon))
                cg.addLine(to: CGPoint(x: plank, y: 90))
                cg.strokePath()
            }
            fill(color(wall, 0.7), CGRect(x: 0, y: horizon - 3, width: 160, height: 3))
            switch sector % 8 {
            case 0: // a window with daylight
                gradient(color((0.62, 0.78, 0.95)), color((0.90, 0.94, 0.98)), CGRect(x: 52, y: 10, width: 56, height: 42))
                cg.setStrokeColor(color((0.98, 0.98, 0.97)))
                cg.setLineWidth(3)
                cg.stroke(CGRect(x: 52, y: 10, width: 56, height: 42))
                cg.move(to: CGPoint(x: 80, y: 10)); cg.addLine(to: CGPoint(x: 80, y: 52)); cg.strokePath()
                fill(color((0.66, 0.30, 0.28), 0.9), CGRect(x: 40, y: 6, width: 11, height: 54))
                fill(color((0.66, 0.30, 0.28), 0.9), CGRect(x: 109, y: 6, width: 11, height: 54))
            case 1: // a sofa under a picture
                fill(color((0.20, 0.24, 0.30)), CGRect(x: 60, y: 14, width: 40, height: 26), radius: 1)
                gradient(color((0.85, 0.62, 0.40)), color((0.45, 0.55, 0.70)), CGRect(x: 63, y: 17, width: 34, height: 20))
                fill(color((0.30, 0.36, 0.48)), CGRect(x: 26, y: 46, width: 108, height: 30), radius: 6)
                fill(color((0.36, 0.43, 0.56)), CGRect(x: 32, y: 50, width: 30, height: 14), radius: 4)
                fill(color((0.80, 0.68, 0.40)), CGRect(x: 98, y: 50, width: 28, height: 14), radius: 4)
            case 2: // a bookshelf
                fill(color((0.36, 0.25, 0.17)), CGRect(x: 34, y: 4, width: 74, height: 72))
                let books: [(CGFloat, CGFloat, CGFloat)] = [(0.75, 0.25, 0.22), (0.20, 0.40, 0.60), (0.85, 0.70, 0.30),
                                                           (0.25, 0.55, 0.40), (0.90, 0.88, 0.82), (0.45, 0.30, 0.55)]
                for shelf in 0..<3 {
                    var x: CGFloat = 38
                    var n = shelf * 2 + sector
                    while x < 102 {
                        let w = CGFloat(4 + (n * 7) % 5)
                        fill(color(books[n % books.count]), CGRect(x: x, y: 8 + CGFloat(shelf) * 23 + CGFloat(n % 3),
                                                                   width: w, height: 19 - CGFloat(n % 3)))
                        x += w + 1
                        n += 1
                    }
                }
            case 3: // a doorway to a brighter room
                gradient(color((0.98, 0.95, 0.88)), color((0.80, 0.74, 0.64)), CGRect(x: 64, y: 8, width: 36, height: horizon - 8))
                cg.setStrokeColor(color((0.97, 0.96, 0.94)))
                cg.setLineWidth(4)
                cg.stroke(CGRect(x: 62, y: 6, width: 40, height: horizon - 4))
                fill(color((0.25, 0.22, 0.20), 1, 0.35), CGRect(x: 20, y: 30, width: 24, height: 18), radius: 2)
            case 4: // kitchen counter and cabinets
                fill(color((0.55, 0.62, 0.55)), CGRect(x: 0, y: 6, width: 160, height: 22))
                fill(color((0.92, 0.92, 0.90)), CGRect(x: 0, y: 46, width: 160, height: 6))
                fill(color((0.55, 0.62, 0.55)), CGRect(x: 0, y: 52, width: 160, height: horizon - 52))
                for x in stride(from: CGFloat(4), to: 160, by: 32) {
                    cg.setStrokeColor(color((0.35, 0.42, 0.35)))
                    cg.setLineWidth(1)
                    cg.stroke(CGRect(x: x, y: 8, width: 28, height: 18))
                }
                fill(color((0.70, 0.72, 0.74)), CGRect(x: 92, y: 34, width: 3, height: 12))
                fill(color((0.70, 0.72, 0.74)), CGRect(x: 86, y: 33, width: 12, height: 3), radius: 1)
            case 5: // a plant and a lamp
                fill(color((0.70, 0.38, 0.24)), CGRect(x: 48, y: 52, width: 22, height: 20), radius: 3)
                for (dx, dy, r) in [(0, 0, 14), (-10, 8, 10), (11, 6, 11), (-4, -10, 10), (8, -8, 9)] as [(CGFloat, CGFloat, CGFloat)] {
                    fill(color((0.22, 0.48, 0.28), 1 + dx / 80), CGRect(x: 59 + dx - r, y: 40 + dy - r,
                                                                        width: 2 * r, height: 2 * r), radius: r)
                }
                fill(color((0.25, 0.22, 0.20)), CGRect(x: 118, y: 26, width: 2, height: 44))
                fill(color((0.98, 0.90, 0.70)), CGRect(x: 106, y: 14, width: 26, height: 16), radius: 3)
            case 6: // a screen on a low cabinet
                fill(color((0.07, 0.08, 0.10)), CGRect(x: 40, y: 12, width: 80, height: 40), radius: 2)
                gradient(color((0.16, 0.22, 0.32)), color((0.08, 0.10, 0.14)), CGRect(x: 43, y: 15, width: 74, height: 34))
                fill(color((0.40, 0.30, 0.22)), CGRect(x: 30, y: 58, width: 100, height: 12), radius: 2)
            default: // a desk with a monitor
                fill(color((0.62, 0.48, 0.34)), CGRect(x: 24, y: 48, width: 112, height: 5))
                fill(color((0.50, 0.38, 0.27)), CGRect(x: 28, y: 53, width: 4, height: horizon - 53 + 8))
                fill(color((0.50, 0.38, 0.27)), CGRect(x: 128, y: 53, width: 4, height: horizon - 53 + 8))
                fill(color((0.10, 0.10, 0.12)), CGRect(x: 58, y: 18, width: 44, height: 26), radius: 2)
                gradient(color((0.55, 0.75, 0.90)), color((0.30, 0.45, 0.70)), CGRect(x: 60, y: 20, width: 40, height: 22))
                fill(color((0.15, 0.15, 0.17)), CGRect(x: 76, y: 44, width: 8, height: 4))
            }
            // Window light from one side, and the lens's own falloff.
            let side: CGFloat = sector % 2 == 0 ? 0 : 160
            let light = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                                   colors: [CGColor(srgbRed: 1, green: 0.97, blue: 0.9, alpha: 0.28),
                                            CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0)] as CFArray,
                                   locations: [0, 1])!
            cg.drawRadialGradient(light, startCenter: CGPoint(x: side, y: 10), startRadius: 0,
                                  endCenter: CGPoint(x: side, y: 10), endRadius: 150, options: [])
            let vignette = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                                      colors: [CGColor(gray: 0, alpha: 0), CGColor(gray: 0, alpha: 0.32)] as CFArray,
                                      locations: [0.55, 1])!
            cg.drawRadialGradient(vignette, startCenter: CGPoint(x: 80, y: 45), startRadius: 0,
                                  endCenter: CGPoint(x: 80, y: 45), endRadius: 95, options: [.drawsAfterEndLocation])
        }
    }
}
