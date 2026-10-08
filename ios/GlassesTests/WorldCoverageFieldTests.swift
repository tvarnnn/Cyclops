//
//  WorldCoverageFieldTests.swift
//  GlassesTests
//

import SwiftUI
import XCTest
@testable import Glasses

final class WorldCoverageFieldTests: XCTestCase {
    private typealias Station = WorldCoverage.Station

    private func component(_ stations: [Station]) -> WorldCoverage.Component {
        WorldCoverage.Component(referenceSegment: 0, posedKeyframes: 2, cellSize: 1, stations: stations)
    }

    func testMasksKeepClockwiseSectorAnglesAndStationCentres() {
        let field = WorldCoverageField(component: component([
            Station(x: 3, y: 5, keyframes: 2, weakMask: 1 << 11, supportedMask: (1 << 0) | (1 << 3)),
        ]))
        XCTAssertEqual(field.stations, [CGPoint(x: 3.5, y: 5.5)])
        XCTAssertEqual(field.wedges.map(\.sector), [0, 3, 11])
        XCTAssertEqual(field.wedges.map(\.startAngle), [0, 90, 330])
        XCTAssertEqual(field.wedges.map(\.endAngle), [30, 120, 360])
        XCTAssertEqual(field.wedges.map(\.center), Array(repeating: CGPoint(x: 3.5, y: 5.5), count: 3))
    }

    func testSupportedBeatsWeakAtAnOverlappingBit() {
        let field = WorldCoverageField(component: component([
            Station(x: 0, y: 0, keyframes: 2, weakMask: 1, supportedMask: 1),
        ]))
        XCTAssertEqual(field.wedges.count, 1)
        XCTAssertEqual(field.wedges[0].evidence, .supported)
        XCTAssertEqual(field.wedges[0].radiusInCells, 3)
        XCTAssertEqual(field.wedges[0].intensity, WorldCoverageField.supportedIntensity)
        XCTAssertEqual(field.supportedDirections, 1)
        XCTAssertEqual(field.weakDirections, 0)
    }

    func testUnconfirmedDirectionsLeaveFogUntouched() {
        let field = WorldCoverageField(component: component([
            Station(x: 7, y: 7, keyframes: 1, weakMask: 1 << 2, supportedMask: 0),
        ]))
        XCTAssertEqual(field.wedges.count, 1)
        XCTAssertEqual(field.wedges[0].evidence, .weak)
        XCTAssertEqual(field.wedges[0].radiusInCells, 2)
        XCTAssertEqual(field.wedges[0].intensity, WorldCoverageField.weakIntensity)
        XCTAssertFalse(field.wedges.contains { $0.sector == 0 })
    }

    func testSummaryCountsSeenDirectionsAndPlaces() {
        let first = component([
            Station(x: 0, y: 0, keyframes: 2, weakMask: 0b1, supportedMask: 0b110),
            Station(x: 1, y: 0, keyframes: 1, weakMask: 0b1000, supportedMask: 0),
        ])
        let field = WorldCoverageField(component: first)
        XCTAssertEqual(field.stationCount, 2)
        XCTAssertEqual(field.supportedDirections, 2)
        XCTAssertEqual(field.weakDirections, 2)
        XCTAssertEqual(field.seenDirections, 4)
        XCTAssertEqual(field.totalDirections, 24)
        XCTAssertEqual(field.accessibilitySummary, "Seen in 4 of 24 directions at 2 places")
        let second = component([
            Station(x: 2, y: 2, keyframes: 2, weakMask: 0, supportedMask: 1),
        ])
        XCTAssertEqual(WorldCoverageField.summary(for: [first, second]),
                       "Seen in 5 of 36 directions at 3 places")
    }

    func testEmptyComponentHasNoFieldAndAZeroSummary() {
        let field = WorldCoverageField(component: component([]))
        XCTAssertTrue(field.stations.isEmpty)
        XCTAssertTrue(field.wedges.isEmpty)
        XCTAssertEqual(field.stationCount, 0)
        XCTAssertEqual(field.seenDirections, 0)
        XCTAssertEqual(field.totalDirections, 0)
        XCTAssertEqual(field.accessibilitySummary, "Seen in 0 of 0 directions at 0 places")
    }

    func testRealCoverageBlocksParseAndMatchMaskCounts() throws {
        let files = try realBlockFiles()
        let sources: [(String, [String: Any])]
        if files.isEmpty {
            sources = [("fixture-mid", FOWFixtures.payload(FOWFixtures.mid)),
                       ("fixture-stop", FOWFixtures.payload(FOWFixtures.stop))]
        } else {
            sources = try files.map { ($0.lastPathComponent, try payload(at: $0)) }
        }
        for (name, payload) in sources {
            let coverage = try XCTUnwrap(WorldCoverageStatus(payload: payload).coverage, name)
            for component in coverage.components {
                let field = WorldCoverageField(component: component)
                let supported = component.stations.reduce(0) { $0 + $1.supportedMask.nonzeroBitCount }
                let weak = component.stations.reduce(0) { $0 + $1.weakMask.nonzeroBitCount }
                XCTAssertEqual(field.supportedDirections, supported, name)
                XCTAssertEqual(field.weakDirections, weak, name)
                XCTAssertEqual(field.wedges.count, supported + weak, name)
                XCTAssertEqual(field.totalDirections, component.stations.count * 12, name)
            }
        }
    }

    @MainActor
    func testRenderEveryCoverageBlockToPNG() throws {
#if targetEnvironment(simulator)
        let files = try realBlockFiles()
        let sources: [(String, [String: Any])]
        if files.isEmpty {
            sources = [("fixture-mid", FOWFixtures.payload(FOWFixtures.mid)),
                       ("fixture-stop", FOWFixtures.payload(FOWFixtures.stop))]
        } else {
            sources = try files.map { ($0.deletingPathExtension().lastPathComponent, try payload(at: $0)) }
        }
        let shots = URL(fileURLWithPath: "/Users/tristan/Projects/Glasses-scratch/fow-field/shots", isDirectory: true)
        try FileManager.default.createDirectory(at: shots, withIntermediateDirectories: true)
        for (name, payload) in sources {
            let coverage = try XCTUnwrap(WorldCoverageStatus(payload: payload).coverage, name)
            let component = try XCTUnwrap(coverage.components.first, name)
            let view = WorldCoveragePieceView(component: component, index: 0, showsName: false)
                .frame(width: 512, height: 512)
            let renderer = ImageRenderer(content: view)
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.uiImage, name)
            let png = try XCTUnwrap(image.pngData(), name)
            let destination = shots.appendingPathComponent("\(name).png")
            try png.write(to: destination, options: .atomic)
            XCTAssertGreaterThan(png.count, 1_000, name)
            let colors = try sampledColors(in: image)
            XCTAssertGreaterThan(colors.count, 20, "\(name): radial light must be visible in the fog")
        }
#else
        throw XCTSkip("Host-path PNG rendering runs on the simulator")
#endif
    }

    private func realBlockFiles() throws -> [URL] {
        let folder = URL(fileURLWithPath: "/Users/tristan/Projects/Glasses-scratch/fow-ios/real-blocks", isDirectory: true)
        guard FileManager.default.fileExists(atPath: folder.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func payload(at file: URL) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: file))
        return try XCTUnwrap(object as? [String: Any], file.lastPathComponent)
    }

    private func sampledColors(in image: UIImage) throws -> Set<UInt32> {
        let source = try XCTUnwrap(image.cgImage)
        let width = source.width, height = source.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let bitmap = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let context = try XCTUnwrap(CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                             bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: bitmap))
        context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        var colors: Set<UInt32> = []
        for y in stride(from: 0, to: height, by: 16) {
            for x in stride(from: 0, to: width, by: 16) {
                let offset = (y * width + x) * 4
                guard bytes[offset + 3] == 255 else {
                    XCTFail("fog fills the piece, including unseen space")
                    return colors
                }
                colors.insert(UInt32(bytes[offset]) << 16 | UInt32(bytes[offset + 1]) << 8
                              | UInt32(bytes[offset + 2]))
            }
        }
        return colors
    }
}
