//
//  IMURecorder.swift
//  Glasses
//

// Entire file is DEBUG-only, like the Motion probe beside it and the Developer
// Tools switch that arms it. It is walk-5 evidence tooling: the glasses IMU and
// the timing of every camera frame, written to a file on this phone and joined
// offline with the Tower's own capture. Nothing here is sent to the Tower (no
// contract change), shown to a wearer, or played as a sound or a haptic.
#if DEBUG

import Combine
import CoreMedia
import Foundation
import MWDATCore
import MWDATMotion
import os

// MARK: - Values

/// One Motion reading, reduced to the fields the log keeps.
///
/// The app's own type rather than `MotionSample`, so the log format can be
/// tested from fixed numbers without the SDK, and so Mock Device Kit's
/// same-named `MotionSample` never has to be disambiguated in a test.
nonisolated struct IMUMotionReading: Equatable, Sendable {
    /// `MotionSource`, spelled the way the log spells it.
    enum Source: String, Sendable {
        case glasses
        case neuralBand = "neural_band"
        case unknown
    }

    /// `MotionSample.timestampNs`: the glasses' own monotonic clock.
    let timestampNs: Int64
    let accelerometer: SIMD3<Float>?
    let gyroscope: SIMD3<Float>?
    /// The fused orientation as (w, x, y, z), the order the log writes it in.
    let orientation: SIMD4<Float>?
    let source: Source

    init(
        timestampNs: Int64,
        accelerometer: SIMD3<Float>?,
        gyroscope: SIMD3<Float>?,
        orientation: SIMD4<Float>?,
        source: Source
    ) {
        self.timestampNs = timestampNs
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.orientation = orientation
        self.source = source
    }

    init(_ sample: MotionSample) {
        let source: Source
        switch sample.source {
        case .glasses: source = .glasses
        case .neuralBand: source = .neuralBand
        case .unknown: source = .unknown
        }
        self.init(
            timestampNs: sample.timestampNs,
            accelerometer: sample.accelerometer.map { SIMD3($0.x, $0.y, $0.z) },
            gyroscope: sample.gyroscope.map { SIMD3($0.x, $0.y, $0.z) },
            orientation: sample.orientation.map { SIMD4($0.w, $0.x, $0.y, $0.z) },
            source: source
        )
    }
}

/// The two clocks read off one camera frame **on DAT's callback thread**,
/// before the main-actor hop, for the same reason `FramePTSProbe` reads there:
/// a receipt time taken after the hop carries main-actor queueing as well as
/// transport. The camera epoch is taken there too (`IMURecorder.stampFrame`).
nonisolated struct IMUFrameStamp: Equatable, Sendable {
    /// `CMSampleBufferGetPresentationTimeStamp` in microseconds, or `nil` when
    /// DAT's buffer carried an invalid or indefinite time.
    let ptsUs: Int64?
    /// This phone's `DispatchTime` (`mach_absolute_time`) in nanoseconds: the
    /// same base as `MonotonicClock` and as the Motion lines' `rx_mono_ns`.
    let rxMonoNs: UInt64
    /// The camera epoch the frame arrived in, fixed when it was stamped; see
    /// `IMUCameraEpochs`. `nil` before any camera start.
    let epoch: UUID?

    init(ptsUs: Int64?, rxMonoNs: UInt64, epoch: UUID? = nil) {
        self.ptsUs = ptsUs
        self.rxMonoNs = rxMonoNs
        self.epoch = epoch
    }

    init(sampleBuffer: CMSampleBuffer, rxMonoNs: UInt64, epoch: UUID? = nil) {
        self.init(
            ptsUs: Self.microseconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)),
            rxMonoNs: rxMonoNs,
            epoch: epoch
        )
    }

    static func microseconds(_ time: CMTime) -> Int64? {
        guard time.isValid, !time.isIndefinite, time.isNumeric, time.timescale > 0 else { return nil }
        return CMTimeConvertScale(time, timescale: 1_000_000, method: .roundHalfAwayFromZero).value
    }
}

/// One `addMotion` attempt, for the header: which rate was asked for, and why
/// it was refused if it was.
nonisolated struct IMUMotionAttempt: Equatable, Sendable {
    let hz: Int
    let error: String?
}

/// The glasses' `DeviceState`, as strings, so a change can be detected by
/// comparison and written without the SDK's types.
nonisolated struct IMUDeviceStateSnapshot: Equatable, Sendable {
    let thermal: String
    let batteryPercent: Int?
    let charging: String
    let don: String
    let hinge: String
    let link: String
    let compatibility: String

    init(thermal: String, batteryPercent: Int?, charging: String, don: String, hinge: String, link: String, compatibility: String) {
        self.thermal = thermal
        self.batteryPercent = batteryPercent
        self.charging = charging
        self.don = don
        self.hinge = hinge
        self.link = link
        self.compatibility = compatibility
    }

    init(_ state: DeviceState) {
        self.init(
            thermal: "\(state.thermalLevel)",
            batteryPercent: state.batteryLevel,
            charging: "\(state.chargingState)",
            don: "\(state.donState)",
            hinge: "\(state.hingeState)",
            link: "\(state.linkState)",
            compatibility: "\(state.compatibility)"
        )
    }
}

extension MotionSamplingRate {
    /// The rate in hertz, for the log and for gap detection.
    nonisolated var hertz: Int {
        switch self {
        case .hz5: return 5
        case .hz10: return 10
        case .hz15: return 15
        case .hz24: return 24
        case .hz30: return 30
        case .hz60: return 60
        }
    }
}

// MARK: - JSON lines

/// A JSON value as the log writes it. Hand-rolled rather than
/// `JSONSerialization` so the keys come out in the documented order, the
/// floats in their shortest exact form, and every line costs one string build.
nonisolated indirect enum IMUJSON: Sendable {
    case string(String)
    case int(Int)
    case uint(UInt64)
    case double(Double)
    case float(Float)
    case bool(Bool)
    case null
    case array([IMUJSON])
    case object([(String, IMUJSON)])

    static func optionalString(_ value: String?) -> IMUJSON { value.map(IMUJSON.string) ?? .null }
    static func optionalInt(_ value: Int?) -> IMUJSON { value.map { .int($0) } ?? .null }

    func write(into out: inout String) {
        switch self {
        case .string(let value): Self.writeString(value, into: &out)
        case .int(let value): out += String(value)
        case .uint(let value): out += String(value)
        case .double(let value): out += value.isFinite ? "\(value)" : "null"
        case .float(let value): out += value.isFinite ? "\(value)" : "null"
        case .bool(let value): out += value ? "true" : "false"
        case .null: out += "null"
        case .array(let values):
            out += "["
            for (index, value) in values.enumerated() {
                if index > 0 { out += "," }
                value.write(into: &out)
            }
            out += "]"
        case .object(let fields):
            Self.writeObject(fields, into: &out)
        }
    }

    static func writeObject(_ fields: [(String, IMUJSON)], into out: inout String) {
        out += "{"
        for (index, field) in fields.enumerated() {
            if index > 0 { out += "," }
            writeString(field.0, into: &out)
            out += ":"
            field.1.write(into: &out)
        }
        out += "}"
    }

    static func writeString(_ value: String, into out: inout String) {
        out += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
    }

    /// One complete log line, newline-terminated.
    static func line(_ fields: [(String, IMUJSON)]) -> String {
        var out = ""
        out.reserveCapacity(160)
        writeObject(fields, into: &out)
        out += "\n"
        return out
    }
}

/// The four line types. Pure functions of their inputs, so the format is
/// tested without a file, a queue or the SDK.
nonisolated enum IMULogFormat {
    /// Bumped on any change a reader must know about. 2: a frame's epoch is
    /// taken on DAT's thread when it arrives, a resume through `starting` opens
    /// an epoch, the file stays open (and written) while the phone is locked,
    /// and `counts` says what is on disk.
    static let schema = 2

    /// `{"t":"m","ts_ns":…,"rx_mono_ns":…,"a":[x,y,z]|null,"g":[x,y,z]|null,"q":[w,x,y,z]|null,"src":"glasses"}`
    static func motionLine(_ reading: IMUMotionReading, rxMonoNs: UInt64) -> String {
        IMUJSON.line([
            ("t", .string("m")),
            ("ts_ns", .int(Int(reading.timestampNs))),
            ("rx_mono_ns", .uint(rxMonoNs)),
            ("a", vector(reading.accelerometer.map { [$0.x, $0.y, $0.z] })),
            ("g", vector(reading.gyroscope.map { [$0.x, $0.y, $0.z] })),
            // Stored and written as (w, x, y, z).
            ("q", vector(reading.orientation.map { [$0[0], $0[1], $0[2], $0[3]] })),
            ("src", .string(reading.source.rawValue)),
        ])
    }

    /// `{"t":"f","pts_us":…,"epoch":"<uuid>","rx_mono_ns":…,"sent":true|false,"seq":N}`
    static func frameLine(_ stamp: IMUFrameStamp, sent: Bool, seq: Int) -> String {
        IMUJSON.line([
            ("t", .string("f")),
            ("pts_us", stamp.ptsUs.map { .int(Int($0)) } ?? .null),
            ("epoch", .optionalString(stamp.epoch?.uuidString)),
            ("rx_mono_ns", .uint(stamp.rxMonoNs)),
            ("sent", .bool(sent)),
            ("seq", .int(seq)),
        ])
    }

    /// `{"t":"ev","ev":"<name>","mono_ns":…, …fields}`
    static func eventLine(_ name: String, monoNs: UInt64, fields: [(String, IMUJSON)]) -> String {
        IMUJSON.line([("t", .string("ev")), ("ev", .string(name)), ("mono_ns", .uint(monoNs))] + fields)
    }

    private static func vector(_ values: [Float]?) -> IMUJSON {
        guard let values else { return .null }
        return .array(values.map(IMUJSON.float))
    }
}

/// Everything the header line records. Built by `GlassesConnection` once the
/// Motion attempt has an answer, so the header can say which rate was granted.
nonisolated struct IMULogHeader: Sendable {
    var sessionID: UUID
    var fileName: String
    var startMonoNs: UInt64
    var startWall: Date
    var motionRequestedHz: Int?
    var motionActualHz: Int?
    var motionAttempts: [IMUMotionAttempt]
    var glassesModel: String?
    var captureResolution: String
    var cameraFPSRequested: Int
    var towerTargetFPS: Double
    /// `nil` reads `.current` when the line is built, which is on the
    /// recorder's queue: it stats the app binary, and that is file I/O.
    var build: BuildInfo? = nil

    /// What this binary and this phone are. Read once per header.
    struct BuildInfo: Sendable {
        var appBuildSHA: String?
        var appVersion: String?
        var appBinaryModified: Date?
        var datSDKVersion: String?
        var datMotionVersion: String?
        var phoneModel: String?
        var phoneOS: String

        static var current: BuildInfo {
            let main = Bundle.main
            // `GlassesBuildSHA` is `$(GLASSES_BUILD_SHA)` in Info.plist: empty
            // unless the build passed `GLASSES_BUILD_SHA=<sha>` to xcodebuild.
            let sha = (main.object(forInfoDictionaryKey: "GlassesBuildSHA") as? String)
                .flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 }
            let short = main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
            let build = main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
            let modified = main.executableURL.flatMap {
                (try? FileManager.default.attributesOfItem(atPath: $0.path))?[.modificationDate] as? Date
            }
            return BuildInfo(
                appBuildSHA: sha,
                appVersion: [short, build.map { "(\($0))" }].compactMap { $0 }.joined(separator: " "),
                appBinaryModified: modified,
                datSDKVersion: frameworkVersion("com.facebook.MWDATCore"),
                datMotionVersion: frameworkVersion("com.facebook.MWDATMotion"),
                phoneModel: hardwareModel(),
                phoneOS: ProcessInfo.processInfo.operatingSystemVersionString
            )
        }

        /// DAT ships as dynamic frameworks, so each one's own Info.plist says
        /// which release is linked, rather than this file restating the pin.
        static func frameworkVersion(_ identifier: String) -> String? {
            Bundle(identifier: identifier)?.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        }

        /// `iPhone17,1` and the like; on the Simulator, the simulated model.
        static func hardwareModel() -> String? {
            if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
                return simulated
            }
            var info = utsname()
            uname(&info)
            let machine = withUnsafeBytes(of: &info.machine) { raw in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            return machine.isEmpty ? nil : machine
        }
    }

    func line() -> String {
        let build = self.build ?? .current
        return IMUJSON.line([
            ("t", .string("header")),
            ("schema", .int(IMULogFormat.schema)),
            ("file", .string(fileName)),
            ("session_id", .string(sessionID.uuidString)),
            ("app_build_sha", .optionalString(build.appBuildSHA)),
            ("app_version", .optionalString(build.appVersion)),
            ("app_binary_mtime", .optionalString(build.appBinaryModified.map(IMURecorder.isoExtended))),
            ("dat_sdk_version", .optionalString(build.datSDKVersion)),
            ("dat_motion_version", .optionalString(build.datMotionVersion)),
            ("motion_rate_requested_hz", .optionalInt(motionRequestedHz)),
            ("motion_rate_actual_hz", .optionalInt(motionActualHz)),
            ("motion_rate_attempts", .array(motionAttempts.map {
                .object([("hz", .int($0.hz)), ("error", .optionalString($0.error))])
            })),
            ("phone_model", .optionalString(build.phoneModel)),
            ("phone_os", .string(build.phoneOS)),
            ("glasses_model", .optionalString(glassesModel)),
            ("start_mono_ns", .uint(startMonoNs)),
            ("start_wall_unix_ms", .int(Int((startWall.timeIntervalSince1970 * 1000).rounded()))),
            ("start_wall_iso", .string(IMURecorder.isoExtended(startWall))),
            ("capture_resolution", .string(captureResolution)),
            ("camera_fps_requested", .int(cameraFPSRequested)),
            ("tower_target_fps", .double(towerTargetFPS)),
            ("clocks", .object([
                ("ts_ns", .string("MotionSample.timestampNs: the glasses' monotonic clock")),
                ("rx_mono_ns", .string("phone DispatchTime.uptimeNanoseconds (mach_absolute_time; stops while the phone sleeps), read on DAT's callback thread before any main-actor hop")),
                ("pts_us", .string("CMSampleBufferGetPresentationTimeStamp of VideoFrame.sampleBuffer, in microseconds")),
                ("mono_ns", .string("phone DispatchTime.uptimeNanoseconds when the event was noted")),
            ])),
            ("seq_semantics", .string("the app's 1-based DAT frame ordinal for this capture session: GlassesConnection.frameCount, sent to the Tower as frame.seq, which the Tower stores as source_seq in frames.jsonl (the app sends no source_seq)")),
            ("sent_semantics", .string("true when the 12 fps gate selected the frame and it was decoded and handed to TowerClient.sendFrame; the Tower may still not have received it (offline, send window, paused) -- frames.jsonl is the truth for receipt")),
            ("epoch_semantics", .string("a new UUID at each camera start: stream.start(), and every later return to streaming from any other state (a pause, a stop, starting, waitingForDevice) once the epoch has frames; the first streaming after stream.start() is that start. A frame's epoch is taken on DAT's callback thread when it arrives, in DAT's order against the stream states. The PTS freezes across a camera stop rather than resetting, so an epoch boundary is not visible in pts_us")),
        ])
    }
}

// MARK: - Recorder

/// Counters the recorder keeps. Also the body of the periodic `counts` event
/// and of the Developer Tools readout.
nonisolated struct IMURecorderCounts: Equatable, Sendable {
    /// Motion lines written: glasses-source samples.
    var motionGlasses = 0
    /// Samples not written because their source was not the glasses, by source.
    var motionOtherBySource: [String: Int] = [:]
    /// Consecutive glasses samples more than 1.5 intervals apart, on the
    /// glasses' clock, and how many samples those gaps are estimated to hold.
    /// DAT's sample stream keeps the newest 256 and drops silently, so a gap
    /// is the only trace an overflow leaves.
    var motionGapEvents = 0
    var motionEstimatedMissing = 0
    var motionMaxGapNs: Int64 = 0
    /// A timestamp that did not advance.
    var motionNonMonotonic = 0
    /// Frame lines written, and how many of them were forwarded.
    var frames = 0
    var framesSent = 0
    var framesWithoutPTS = 0
    var ptsDiscontinuities = 0
    /// Consecutive frames 0.5 s or more apart by receipt whose PTS advanced
    /// by less than half that: a stop or pause the PTS froze across.
    var ptsFrozen = 0
    /// Lines refused after a cap, after close, or after the file failed (a
    /// failed recorder holds nothing: there is no file to write it to).
    var droppedAfterCap = 0
    var droppedAfterClose = 0
    var droppedAfterFailure = 0
    /// The frames among `droppedAfterClose`: received by the phone after the
    /// session's log closed (still queued for the main actor at the stop).
    var framesAfterClose = 0

    var motionOther: Int { motionOtherBySource.values.reduce(0, +) }

    var jsonFields: [(String, IMUJSON)] {
        [
            ("m_glasses", .int(motionGlasses)),
            ("m_other", .int(motionOther)),
            ("m_other_by_src", .object(motionOtherBySource.sorted { $0.key < $1.key }.map { ($0.key, .int($0.value)) })),
            ("m_gap_events", .int(motionGapEvents)),
            ("m_est_missing", .int(motionEstimatedMissing)),
            ("m_max_gap_ms", .double(Double(motionMaxGapNs) / 1_000_000)),
            ("m_nonmonotonic", .int(motionNonMonotonic)),
            ("f_total", .int(frames)),
            ("f_sent", .int(framesSent)),
            ("f_no_pts", .int(framesWithoutPTS)),
            ("f_pts_discontinuities", .int(ptsDiscontinuities)),
            ("f_pts_frozen", .int(ptsFrozen)),
            ("dropped_after_cap", .int(droppedAfterCap)),
            ("dropped_after_close", .int(droppedAfterClose)),
            ("f_after_close", .int(framesAfterClose)),
            ("dropped_after_failure", .int(droppedAfterFailure)),
        ]
    }
}

/// What Developer Tools and the recording badge show. Republished at most
/// once a second, and on every change of phase or cap.
nonisolated struct IMURecorderStatus: Equatable, Sendable {
    enum Phase: Equatable, Sendable {
        case idle
        case recording
        case closed
        case failed(String)
        /// The recorder would not record: the file's Data Protection class
        /// could not be verified as `completeUnlessOpen` on a phone, so it
        /// might stop being writable at the first lock. Nothing is recorded.
        case refused(String)
    }

    var phase: Phase = .idle
    var fileName = ""
    var bytes = 0
    var counts = IMURecorderCounts()
    var epochs = 0
    var motionConfiguredHz: Int?
    /// Glasses samples per second over the last tick, by the glasses' clock.
    var motionRateHz: Double?
    /// Which cap stopped the data lines, if one has: `size`, `duration` or
    /// `folder`.
    var capReason: String?
    /// The phone is locked. The open file is still being written.
    var phoneLocked = false
}

/// The recording badge's three states.
nonisolated enum IMURecordingIndicator: Equatable, Sendable {
    /// No recorder: no badge.
    case off
    /// A recorder is writing this capture's IMU log.
    case writing
    /// A recorder exists but is not writing data lines: a cap was reached, or
    /// the file could not be written. The reason is short enough for a badge.
    case stopped(String)

    init(_ status: IMURecorderStatus) {
        switch status.phase {
        case .idle, .closed: self = .off
        case .failed: self = .stopped("error")
        case .refused: self = .stopped("protection")
        case .recording: self = status.capReason.map { .stopped("\($0) cap") } ?? .writing
        }
    }
}

/// Which file operations ran where. The recorder's promise is that every one of
/// them runs on its own serial queue and none on the main thread, and a test
/// reads this to hold it to that.
nonisolated struct IMUIOAudit: Equatable, Sendable {
    var operations = 0
    var onMainThread = 0
    var offQueue = 0
    /// Times the log file was opened. One per recorder: the handle is never
    /// closed and reopened, not even across a lock.
    var fileOpens = 0
}

/// The camera epoch, decided where DAT delivers: on its callback threads, in
/// the order they run, behind a lock.
///
/// ## Why not on the recorder's queue
///
/// The stream-state and frame listeners each hop to the main actor in their
/// own `Task`, and nothing orders two such `Task`s. When the epoch was decided
/// after those hops, the first frame of a resume could reach the log ahead of
/// the `.streaming` that opened its epoch and be written under the old one --
/// silently, because the PTS freezes across a camera stop instead of
/// resetting, so no discontinuity flags it. Here a frame's epoch is fixed when
/// it is stamped on DAT's thread, before any hop.
///
/// ## Two publishers, either order
///
/// DAT's stream states and frames come from two publishers, and nothing says
/// their callbacks run in the order DAT produced them: the first resumed frame
/// may run before the `.streaming` that announces the resume. So a pause or a
/// stop is a marker, and **whichever comes first after it opens the next
/// epoch** -- the return to `.streaming`, or a frame -- and the other joins
/// it. A frame within `stragglerWindowNs` of the marker is not a resume (no
/// pause is that short) but a frame from before it whose callback ran late,
/// and stays in the epoch it belongs to. The cost on DAT's thread is one
/// uncontended lock per frame.
nonisolated struct IMUCameraEpochs: Sendable {
    struct Rotation: Equatable, Sendable {
        let previous: UUID?
        let next: UUID
    }

    /// What a stream state or a frame did to the epoch.
    enum Effect: Equatable, Sendable {
        case none
        /// A new epoch, and why.
        case opened(Rotation, reason: String)
        /// The `.streaming` of a resume whose first frame already opened the
        /// epoch.
        case joined
    }

    /// A frame this close after a pause or stop marker is a late callback
    /// from before it, not a resumed frame.
    static let stragglerWindowNs: UInt64 = 250_000_000

    /// The current epoch; `nil` before the first camera start.
    private(set) var current: UUID?
    /// Frames stamped in `current`.
    private(set) var frames = 0
    /// The states since the stream last left `streaming`, oldest first (a
    /// few at most), for the new epoch's `reason`.
    private var sinceStreaming: [String] = []
    /// Set at a camera start and cleared by the first `streaming` after it,
    /// which is that start arriving, not a resume. Frames can precede it.
    private var awaitingFirstStreaming = false
    /// When the stream left `streaming` with frames in the epoch: the marker
    /// after which the next `streaming` or frame opens a new epoch.
    private var interruptedAtNs: UInt64?
    /// A frame opened the current epoch after a marker; the `streaming` still
    /// to come joins it rather than opening another.
    private var awaitingJoiningStreaming = false

    /// `stream.start()`: always a new epoch.
    mutating func cameraStart() -> Rotation {
        awaitingFirstStreaming = true
        interruptedAtNs = nil
        awaitingJoiningStreaming = false
        sinceStreaming = []
        return rotate()
    }

    /// Every `StreamState`, with the time read on DAT's thread.
    ///
    /// Any state but `streaming`, once the epoch has frames, is a marker
    /// (`starting` included: a resume may pass through it, and it used to be
    /// excluded, which left every such resume in the epoch before it) --
    /// except `starting` while a frame's epoch waits for its `streaming`,
    /// which is that same resume arriving. A `streaming` after a marker opens
    /// the next epoch, unless a frame already has, in which case it joins.
    mutating func streamState(_ state: String, monoNs: UInt64) -> Effect {
        guard state == "streaming" else {
            if sinceStreaming.count < 8 { sinceStreaming.append(state) }
            if awaitingJoiningStreaming, state == "starting" { return .none }
            awaitingJoiningStreaming = false
            if !awaitingFirstStreaming, frames > 0, interruptedAtNs == nil {
                interruptedAtNs = monoNs
            }
            return .none
        }
        let path = sinceStreaming
        sinceStreaming = []
        if awaitingFirstStreaming {
            awaitingFirstStreaming = false
            return .none
        }
        if awaitingJoiningStreaming {
            awaitingJoiningStreaming = false
            return .joined
        }
        guard interruptedAtNs != nil else { return .none }
        interruptedAtNs = nil
        return .opened(rotate(), reason: "resume from \(path.joined(separator: " > "))")
    }

    /// One frame, received at `rxMonoNs`: the epoch it belongs to, and
    /// whether it opened that epoch.
    mutating func frame(rxMonoNs: UInt64) -> (epoch: UUID?, effect: Effect) {
        var effect = Effect.none
        if let marker = interruptedAtNs, rxMonoNs >= marker &+ Self.stragglerWindowNs {
            interruptedAtNs = nil
            awaitingJoiningStreaming = true
            let path = sinceStreaming.joined(separator: " > ")
            effect = .opened(rotate(), reason: "frame after \(path), ahead of its streaming")
        }
        frames += 1
        return (current, effect)
    }

    private mutating func rotate() -> Rotation {
        let rotation = Rotation(previous: current, next: UUID())
        current = rotation.next
        frames = 0
        return rotation
    }
}

/// Whether the log's file may be written, from the Data Protection class it
/// reports once created (and, if that was wrong, once set on the file).
///
/// Walk 5 depends on the file staying writable through a lock, which only
/// `completeUnlessOpen` (held open) guarantees. A file under `.complete`
/// would record normally until the pocketed phone locked and then fail --
/// the walk would look recorded and not be. So on a phone anything but
/// `completeUnlessOpen`, including no class at all, is refused, with any
/// failure to set it in the reason. The Simulator has no Data Protection and
/// may report no class: that is recorded as unverified, not refused.
nonisolated enum IMUProtectionVerdict: Equatable, Sendable {
    case verified
    case unverified(String)
    case refused(String)

    init(file: FileProtectionType?, setErrors: [String], onSimulator: Bool) {
        let failures = setErrors.isEmpty ? "" : "; setting it failed (\(setErrors.joined(separator: "; ")))"
        switch file {
        case .some(.completeUnlessOpen):
            self = .verified
        case nil where onSimulator:
            self = .unverified("the Simulator reports no protection class (it has no Data Protection)" + failures)
        case nil:
            self = .refused("the log file reports no protection class, so completeUnlessOpen cannot be verified" + failures)
        case .some(let other):
            self = .refused("the log file is \(other.rawValue), not NSFileProtectionCompleteUnlessOpen" + failures)
        }
    }
}

/// Writes one capture session's IMU log: `Documents/imu-logs/<start>-<id>.jsonl`.
///
/// ## The one rule: nothing here runs on the caller's thread but an enqueue
///
/// The main actor already carries the frame path (gate, JPEG encode, base64,
/// JSON, viewfinder), so every public method does exactly one thing on the
/// calling thread -- `queue.async` with a small value -- and everything else,
/// formatting included, happens on this recorder's own serial utility queue.
/// Lines are buffered and written when 64 KB accumulate or once a second,
/// whichever is first, so a crash loses at most about a second -- locked or
/// not (see "Private").
///
/// ## Bounded, three ways
///
/// A file stops taking data lines (`m`, `f`) at 200 MB or 30 minutes, whichever
/// comes first, and the whole `imu-logs` folder is held to 1 GB: a file that
/// opens with less than 200 MB of folder budget left is capped at what is left,
/// and one that opens with almost none is not created at all. Events may use a
/// small reserve beyond the data cap, so the `cap_reached` line and the closing
/// `counts` still land. The first refusal is printed and logged; every later
/// one is counted.
///
/// ## Private
///
/// Motion is raw sensor data (docs/06-PRIVACY-DATA.md), so the file is
/// encrypted at rest with `FileProtectionType.completeUnlessOpen` (Data
/// Protection's "Protected Unless Open" class) and excluded from backup.
///
/// Not `.complete`. Walk 5 is walked with the phone in a pocket, so the phone
/// locks within minutes, and a `.complete` file cannot be written once it has:
/// the recorder used to close on lock and hold every later line in memory
/// until the unlock, so a pocketed walk lived in RAM -- lost to a crash or a
/// jetsam, and truncated at a 32 MB hold near 30 minutes. A
/// `completeUnlessOpen` file stays writable through a lock **for as long as it
/// stays open**, and once closed it cannot be opened again until the phone is
/// unlocked. So the file is created by the one `open` that writes it, that
/// handle is never closed until the log ends, and a lock is only a durable
/// point (flush and fsync) and an event in the log. A close while locked
/// closes at once. The class read back from the file is logged (`file_open`)
/// so the log itself says what protected it on the phone; the Simulator has no
/// Data Protection.
///
/// State below `queue` is confined to it; that is what the `@unchecked` asserts.
/// The camera epoch is the exception: it lives behind its own lock, because it
/// is decided on DAT's threads (`IMUCameraEpochs`).
nonisolated final class IMURecorder: @unchecked Sendable {

    struct Limits: Equatable, Sendable {
        /// Per file.
        var maxBytes = 200_000_000
        /// Per file, on this phone's monotonic clock from the recorder's start.
        var maxDuration: TimeInterval = 30 * 60
        /// The whole `imu-logs` folder, this file included.
        var folderMaxBytes = 1_000_000_000
        var eventReserveBytes = 64 * 1024
        var flushThresholdBytes = 64 * 1024
        var flushInterval: TimeInterval = 1.0
        /// A `counts` event every this many flush ticks (10 s by default).
        var countsEveryTicks = 10

        static let standard = Limits()
    }

    /// Directory under Documents that holds every log. `pull_imu_logs.sh`
    /// copies exactly this.
    static let directoryName = "imu-logs"

    /// `UIApplication.protectedDataWillBecomeUnavailableNotification` and its
    /// pair, by value, so this nonisolated file need not reach into UIKit's
    /// main-actor class for two constants. A test holds them equal to UIKit's.
    static let protectedDataWillBecomeUnavailable = Notification.Name("UIApplicationProtectedDataWillBecomeUnavailable")
    static let protectedDataDidBecomeAvailable = Notification.Name("UIApplicationProtectedDataDidBecomeAvailable")

    let sessionID: UUID
    let startWall: Date
    let startMonoNs: UInt64
    let limits: Limits
    /// `imu-logs/<ISO-start>-<first 8 of the session id>.jsonl`, relative to
    /// the base directory.
    let relativePath: String

    private let baseDirectory: URL?
    private let onStatus: (@Sendable (IMURecorderStatus) -> Void)?
    private let queue: DispatchQueue
    private let queueKey = DispatchSpecificKey<UInt8>()

    // MARK: Queue-confined state

    private var header: String?
    private var fileURL: URL?
    private var handle: FileHandle?
    private var buffer = Data()
    private var bytesAccepted = 0
    private var bytesWritten = 0
    /// `min(limits.maxBytes, folder budget left)`, fixed when the file opens.
    private var effectiveMaxBytes: Int
    private var capIsFolder = false
    private var capReason: String?
    private var closed = false
    private var closeCompletions: [@Sendable (IMURecorderSummary) -> Void] = []
    private var failure: String?
    private var timer: DispatchSourceTimer?
    private var ticks = 0
    /// Whether the phone is unlocked, as the observers and the sample have
    /// said; `true` until one of them says otherwise.
    private var protectedDataAvailable = true
    /// The file protection check refused this log; see `IMUProtectionVerdict`.
    private var refusal: String?
    private var counts = IMURecorderCounts()
    private var io = IMUIOAudit()

    /// Not queue-confined: written on DAT's threads and the main actor. See
    /// `IMUCameraEpochs`.
    private let epochs = OSAllocatedUnfairLock(initialState: IMUCameraEpochs())

    /// Not queue-confined either: installed on the caller's thread, before the
    /// open is queued, and removed on the queue. Behind `observerLock`.
    private let observerLock = NSLock()
    private var observerTokens: [NSObjectProtocol] = []
    private var observersInstalled = false

    /// Reads a file's Data Protection class. `IMURecorder.protection(of:)`
    /// in the app; a test seam, so both verdicts can be reached off a phone.
    private let readProtection: @Sendable (URL) -> FileProtectionType?
    /// Whether this is the Simulator, which has no Data Protection and may
    /// report no class. A test seam.
    private let onSimulator: Bool

    /// Camera epochs started, as written to the log.
    private var epochCount = 0
    /// The last PTS seen, and its epoch: a discontinuity is looked for only
    /// inside one epoch.
    private var lastPTSEpoch: UUID?
    private var lastPTSUs: Int64?
    /// The last frame with a PTS, in any epoch, for `pts_frozen`.
    private var lastFrameWithPTS: (ptsUs: Int64, rxMonoNs: UInt64, epoch: UUID?)?
    private var lastDeviceState: IMUDeviceStateSnapshot?
    private var motionIntervalNs: Int64?
    private var motionConfiguredHz: Int?
    private var lastMotionTsNs: Int64?
    private var tickFirstTsNs: Int64?
    private var tickLastTsNs: Int64?
    private var tickSamples = 0
    private var lastRateHz: Double?
    private var reportedFirstOtherSource = false

    /// - Parameters:
    ///   - baseDirectory: where `imu-logs/` is created; `nil` is this app's
    ///     Documents directory, resolved on the recorder's queue.
    ///   - readProtection, onSimulator: test seams for the protection check;
    ///     see `IMUProtectionVerdict`.
    ///   - onStatus: called on the recorder's queue, at most once a second and
    ///     on every change of phase or cap.
    init(
        sessionID: UUID = UUID(),
        startWall: Date = Date(),
        startMonoNs: UInt64 = DispatchTime.now().uptimeNanoseconds,
        baseDirectory: URL? = nil,
        limits: Limits = .standard,
        readProtection: @escaping @Sendable (URL) -> FileProtectionType? = { IMURecorder.protection(of: $0) },
        onSimulator: Bool = IMURecorder.runningOnSimulator,
        onStatus: (@Sendable (IMURecorderStatus) -> Void)? = nil
    ) {
        self.sessionID = sessionID
        self.startWall = startWall
        self.startMonoNs = startMonoNs
        self.baseDirectory = baseDirectory
        self.limits = limits
        self.readProtection = readProtection
        self.onSimulator = onSimulator
        self.onStatus = onStatus
        self.effectiveMaxBytes = limits.maxBytes
        let shortID = String(sessionID.uuidString.prefix(8))
        self.relativePath = "\(Self.directoryName)/\(Self.isoBasic(startWall))-\(shortID).jsonl"
        self.queue = DispatchQueue(label: "Glasses.IMURecorder.\(shortID)", qos: .utility)
        queue.setSpecific(key: queueKey, value: 1)
    }

    // MARK: Called from any thread; each is one enqueue

    #if targetEnvironment(simulator)
    static let runningOnSimulator = true
    #else
    static let runningOnSimulator = false
    #endif

    /// Writes the header, creates the file and starts the flush timer. Lines
    /// recorded before this are held in memory and written after the header,
    /// so the header is always the first line on disk. The lock observers are
    /// installed first, here, on the caller's thread, if they are not already.
    func open(header: IMULogHeader) {
        installObservers()
        queue.async { self.openOnQueue(header.line()) }
    }

    /// Installs the lock, unlock and thermal observers **now**, on the
    /// caller's thread. Idempotent. `open` calls it; `GlassesConnection` calls
    /// it earlier, at creation, and then samples the lock state
    /// (`noteProtectedDataSampled`) -- in that order, so a lock between the
    /// two is caught by one or the other and never missed. Installing them in
    /// the queued open, as before, left a window in which a lock reported
    /// nothing and the log said "unlocked".
    func installObservers() {
        let center = NotificationCenter.default
        observerLock.lock()
        defer { observerLock.unlock() }
        guard !observersInstalled else { return }
        observersInstalled = true
        // Weak: nothing waits on an unlock, so nothing here has to outlive the
        // recorder's owner. The close removes them.
        observerTokens = [
            center.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil) { [weak self] _ in
                self?.noteEvent("phone_thermal", fields: [("state", .string(Self.name(of: ProcessInfo.processInfo.thermalState)))])
            },
            center.addObserver(forName: Self.protectedDataWillBecomeUnavailable, object: nil, queue: nil) { [weak self] _ in
                self?.protectedDataWillBecomeUnavailable()
            },
            center.addObserver(forName: Self.protectedDataDidBecomeAvailable, object: nil, queue: nil) { [weak self] _ in
                self?.protectedDataDidBecomeAvailable()
            },
        ]
        let monoNs = DispatchTime.now().uptimeNanoseconds
        queue.async { self.eventOnQueue("lock_observers_installed", monoNs: monoNs, fields: []) }
    }

    /// The lock state, read (`UIApplication.isProtectedDataAvailable`, on the
    /// main actor) **after** `installObservers`. Logged either way.
    func noteProtectedDataSampled(_ available: Bool, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async { self.protectedDataSampledOnQueue(available, monoNs: monoNs) }
    }

    /// A frame's stamp: its two clocks and its camera epoch. Call it **on DAT's
    /// frame callback thread**, before any hop, so the epoch is the one current
    /// when DAT delivered the frame (see `IMUCameraEpochs`). One lock, no
    /// enqueue; the line itself is written later by `recordFrame`.
    func stampFrame(ptsUs: Int64?, rxMonoNs: UInt64) -> IMUFrameStamp {
        let (epoch, effect) = epochs.withLock { $0.frame(rxMonoNs: rxMonoNs) }
        if case .opened(let rotation, let reason) = effect {
            // Enqueued now, from DAT's thread, so it lands ahead of this
            // frame's line, which comes after the main-actor hop.
            queue.async { self.epochStartedOnQueue(rotation, reason: reason, monoNs: rxMonoNs) }
        }
        return IMUFrameStamp(ptsUs: ptsUs, rxMonoNs: rxMonoNs, epoch: epoch)
    }

    func stampFrame(sampleBuffer: CMSampleBuffer, rxMonoNs: UInt64) -> IMUFrameStamp {
        stampFrame(
            ptsUs: IMUFrameStamp.microseconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)),
            rxMonoNs: rxMonoNs
        )
    }

    /// Writes the frame's line under the epoch in its stamp, whenever it
    /// arrives here.
    func recordFrame(_ stamp: IMUFrameStamp, seq: Int, sent: Bool) {
        queue.async { self.frameOnQueue(stamp, seq: seq, sent: sent) }
    }

    func recordMotion(_ reading: IMUMotionReading, rxMonoNs: UInt64) {
        queue.async { self.motionOnQueue(reading, rxMonoNs: rxMonoNs) }
    }

    /// A camera start: a fresh epoch, which every frame stamped after this
    /// carries. Call it before `stream.start()`.
    func noteCameraStart(reason: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        let rotation = epochs.withLock { $0.cameraStart() }
        queue.async { self.epochStartedOnQueue(rotation, reason: reason, monoNs: monoNs) }
    }

    func noteCameraStop(reason: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        let (epoch, frames) = epochs.withLock { ($0.current, $0.frames) }
        queue.async {
            self.eventOnQueue("camera_stop", monoNs: monoNs, fields: [
                ("epoch", .optionalString(epoch?.uuidString)),
                ("reason", .string(reason)),
                ("epoch_frames", .int(frames)),
            ])
        }
    }

    /// Every camera `StreamState`. Call it **on DAT's stream-state callback
    /// thread**, before any hop, with the time read there: whether it opens
    /// (or joins) an epoch is decided now, against the frames being stamped
    /// on DAT's other thread; see `IMUCameraEpochs`.
    func noteStreamState(_ state: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        let effect = epochs.withLock { $0.streamState(state, monoNs: monoNs) }
        queue.async {
            var fields: [(String, IMUJSON)] = [("state", .string(state))]
            if effect == .joined { fields.append(("epoch", .string("joined the epoch its first frame opened"))) }
            self.eventOnQueue("stream_state", monoNs: monoNs, fields: fields)
            if case .opened(let rotation, let reason) = effect {
                self.epochStartedOnQueue(rotation, reason: reason, monoNs: monoNs)
            }
        }
    }

    /// Every `DeviceSessionState`: pause and resume arrive here.
    func noteSessionState(_ state: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async { self.eventOnQueue("session_state", monoNs: monoNs, fields: [("state", .string(state))]) }
    }

    /// Motion attached at `hz`. Also arms gap detection at that rate.
    func noteMotionStart(hz: Int, requestedHz: Int, attempts: [IMUMotionAttempt], monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async {
            self.motionConfiguredHz = hz
            self.motionIntervalNs = hz > 0 ? 1_000_000_000 / Int64(hz) : nil
            self.lastMotionTsNs = nil
            self.eventOnQueue("motion_start", monoNs: monoNs, fields: [
                ("rate_hz", .int(hz)),
                ("requested_hz", .int(requestedHz)),
                ("attempts", .array(attempts.map { .object([("hz", .int($0.hz)), ("error", .optionalString($0.error))]) })),
            ])
        }
    }

    func noteEvent(_ name: String, fields: [(String, IMUJSON)] = [], monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async { self.eventOnQueue(name, monoNs: monoNs, fields: fields) }
    }

    /// Written only when it differs from the last one written.
    func noteDeviceState(_ state: IMUDeviceStateSnapshot, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        queue.async {
            guard state != self.lastDeviceState else { return }
            self.lastDeviceState = state
            self.eventOnQueue("device_state", monoNs: monoNs, fields: [
                ("thermal", .string(state.thermal)),
                ("battery_pct", .optionalInt(state.batteryPercent)),
                ("charging", .string(state.charging)),
                ("don", .string(state.don)),
                ("hinge", .string(state.hinge)),
                ("link", .string(state.link)),
                ("compatibility", .string(state.compatibility)),
            ])
        }
    }

    /// The phone is locking: logged, and a durable point (flush and fsync).
    /// The handle stays open and writing goes on. Called by the notification
    /// this recorder observes, and by tests.
    func protectedDataWillBecomeUnavailable() {
        queue.async { self.lockOnQueue() }
    }

    /// The phone unlocked: logged. Nothing was held, so nothing is owed.
    func protectedDataDidBecomeAvailable() {
        queue.async { self.unlockOnQueue() }
    }

    /// Final counts, flush, fsync, close. Idempotent. `completion` runs on the
    /// recorder's queue once the file is closed, locked phone or not.
    func close(reason: String, monoNs: UInt64 = DispatchTime.now().uptimeNanoseconds, completion: (@Sendable (IMURecorderSummary) -> Void)? = nil) {
        queue.async {
            if let completion { self.closeCompletions.append(completion) }
            self.requestCloseOnQueue(reason: reason, monoNs: monoNs)
        }
    }

    // MARK: Test and diagnostics reads (block the caller until the queue drains)

    func waitUntilIdle() { queue.sync {} }

    func summary() -> IMURecorderSummary { queue.sync { summaryOnQueue() } }

    // MARK: Queue side

    private func openOnQueue(_ headerLine: String) {
        guard header == nil, !closed, failure == nil else { return }
        header = headerLine
        let headerData = Data(headerLine.utf8)
        buffer = headerData + buffer
        bytesAccepted += headerData.count

        var existingFolderBytes = 0
        var folderProtection: FileProtectionType?
        var fileProtection: FileProtectionType?
        var setErrors: [String] = []
        performIO {
            let base = try baseDirectory ?? FileManager.default.url(
                for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true
            )
            let url = base.appendingPathComponent(relativePath)
            var folder = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var excluded = URLResourceValues()
            excluded.isExcludedFromBackup = true
            try folder.setResourceValues(excluded)
            // The folder's class is the one a file created in it inherits, so
            // the `open` below creates the log in the right class even while
            // the phone is locked. Metadata, so it can be set then too. A
            // failure is recorded; the file's own class decides (below).
            do {
                try FileManager.default.setAttributes(
                    [.protectionKey: FileProtectionType.completeUnlessOpen], ofItemAtPath: folder.path
                )
            } catch {
                setErrors.append("folder: \(error.localizedDescription)")
            }
            folderProtection = readProtection(folder)
            existingFolderBytes = IMULogStore.totalBytes(in: folder)

            // The folder's budget, fixed now. A file that would open with less
            // room than its own event reserve is not created at all.
            let budget = limits.folderMaxBytes - existingFolderBytes
            guard budget >= limits.eventReserveBytes * 2 else {
                throw IMURecorderError.folderFull(existingBytes: existingFolderBytes, limit: limits.folderMaxBytes)
            }
            if budget < limits.maxBytes {
                effectiveMaxBytes = budget
                capIsFolder = true
            }

            // Created by the open that writes it, and that descriptor is the
            // only one this log ever has. `createFile` then `FileHandle(forWritingTo:)`
            // would close the new file and reopen it, and a
            // `completeUnlessOpen` file cannot be reopened while the phone is
            // locked.
            // `Darwin.` because `open(header:)` is this class's own. 0o644 is
            // what `createFile` gives a file, so `pull_imu_logs.sh` reads it as before.
            let descriptor = Darwin.open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0o644)
            guard descriptor >= 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO, userInfo: [NSFilePathErrorKey: url.path])
            }
            io.fileOpens += 1
            handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
            var fileURL = url
            self.fileURL = fileURL
            fileProtection = readProtection(url)
            if fileProtection != .completeUnlessOpen {
                // The folder's class did not reach the file (or it reports
                // none): set it on the file itself, which re-wraps its key and
                // leaves this handle open, then read it back.
                do {
                    try FileManager.default.setAttributes(
                        [.protectionKey: FileProtectionType.completeUnlessOpen], ofItemAtPath: url.path
                    )
                } catch {
                    setErrors.append("file: \(error.localizedDescription)")
                }
                fileProtection = readProtection(url)
            }
            try fileURL.setResourceValues(excluded)
        }
        guard failure == nil else { return }
        // What protected the file on this phone, whether it was locked when
        // the file was made, and what could not be set: the walk-5 pre-flight
        // reads this line.
        let fileOpen: [(String, IMUJSON)] = [
            ("protection", .optionalString(fileProtection?.rawValue)),
            ("folder_protection", .optionalString(folderProtection?.rawValue)),
            ("set_errors", .array(setErrors.map(IMUJSON.string))),
            ("protected_data_available", .bool(protectedDataAvailable)),
            ("on_simulator", .bool(onSimulator)),
        ]
        let verdict = IMUProtectionVerdict(file: fileProtection, setErrors: setErrors, onSimulator: onSimulator)
        if case .refused(let reason) = verdict {
            refuseOnQueue(reason, headerLine: headerLine, fileOpen: fileOpen)
            return
        }
        eventOnQueue("file_open", monoNs: DispatchTime.now().uptimeNanoseconds, fields: fileOpen)
        switch verdict {
        case .verified, .refused:
            break
        case .unverified(let reason):
            print("[Glasses][IMURec] file protection unverified: \(reason)")
            eventOnQueue("protection_unverified", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
                ("reason", .string(reason)),
            ])
        }
        print("[Glasses][IMURec] recording to \(fileURL?.path ?? relativePath) (protection \(fileProtection?.rawValue ?? "unreported"); folder already holds \(existingFolderBytes) bytes; this file may take \(effectiveMaxBytes))")
        flushOnQueue()

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(
            deadline: .now() + limits.flushInterval,
            repeating: limits.flushInterval,
            leeway: .milliseconds(100)
        )
        timer.setEventHandler { [weak self] in self?.tick() }
        timer.resume()
        self.timer = timer

        eventOnQueue("phone_thermal", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
            ("state", .string(Self.name(of: ProcessInfo.processInfo.thermalState))),
        ])
        publishStatus()
    }

    /// The file's class is not `completeUnlessOpen` on a phone, so it could
    /// stop being writable at the first lock -- the walk would look recorded
    /// and not be. Refuse instead: nothing held before the open is written
    /// (it is sensor data), the file keeps only its header and one
    /// `file_open` and one `protection_refused` line, the handle is closed,
    /// and every later line is refused. Developer Tools and the badge say so
    /// (`Phase.refused`).
    private func refuseOnQueue(_ reason: String, headerLine: String, fileOpen: [(String, IMUJSON)]) {
        let discarded = max(0, buffer.count - Data(headerLine.utf8).count)
        buffer = Data(headerLine.utf8)
        bytesAccepted = buffer.count
        // What was counted before the open was never written.
        counts = IMURecorderCounts()
        let monoNs = DispatchTime.now().uptimeNanoseconds
        eventOnQueue("file_open", monoNs: monoNs, fields: fileOpen)
        eventOnQueue("protection_refused", monoNs: monoNs, fields: [
            ("reason", .string(reason)),
            ("discarded_bytes", .int(discarded)),
        ])
        flushOnQueue()
        if let handle {
            performIO {
                try handle.synchronize()
                try handle.close()
            }
        }
        handle = nil
        refusal = reason
        if failure == nil { failure = "refused: \(reason)" }
        removeObservers()
        print("[Glasses][IMURec] NOT RECORDING: \(reason). \(relativePath) holds its header and the refusal only.")
        publishStatus()
    }

    /// Removes the observers `installObservers` added. On the queue.
    private func removeObservers() {
        observerLock.lock()
        let tokens = observerTokens
        observerTokens = []
        observerLock.unlock()
        tokens.forEach { NotificationCenter.default.removeObserver($0) }
    }

    /// The lock state as sampled after the observers went in. A locked phone
    /// is treated as a lock (flush and sync, if the file is open); either way
    /// it is logged.
    private func protectedDataSampledOnQueue(_ available: Bool, monoNs: UInt64) {
        eventOnQueue("protected_data_sampled", monoNs: monoNs, fields: [
            ("available", .bool(available)),
            ("was", .bool(protectedDataAvailable)),
        ])
        if available {
            protectedDataAvailable = true
        } else if protectedDataAvailable {
            lockOnQueue()
        }
        publishStatus()
    }

    private func frameOnQueue(_ stamp: IMUFrameStamp, seq: Int, sent: Bool) {
        // A stop or pause the PTS froze across: half a second or more between
        // receipts, and the PTS advanced by less than half of it. Flagged
        // whether or not the epoch changed -- a resume the epochs missed
        // shows here and nowhere else.
        if let pts = stamp.ptsUs {
            if let last = lastFrameWithPTS {
                let rxGapNs = Int64(bitPattern: stamp.rxMonoNs &- last.rxMonoNs)
                let ptsAdvanceNs = (pts - last.ptsUs) * 1_000
                if rxGapNs >= 500_000_000, ptsAdvanceNs < rxGapNs / 2 {
                    counts.ptsFrozen += 1
                    eventOnQueue("pts_frozen", monoNs: stamp.rxMonoNs, fields: [
                        ("seq", .int(seq)),
                        ("epoch", .optionalString(stamp.epoch?.uuidString)),
                        ("prev_epoch", .optionalString(last.epoch?.uuidString)),
                        ("same_epoch", .bool(stamp.epoch == last.epoch)),
                        ("prev_pts_us", .int(Int(last.ptsUs))),
                        ("pts_us", .int(Int(pts))),
                        ("pts_advance_ms", .double(Double(ptsAdvanceNs) / 1_000_000)),
                        ("rx_gap_ms", .double(Double(rxGapNs) / 1_000_000)),
                    ])
                }
            }
            lastFrameWithPTS = (pts, stamp.rxMonoNs, stamp.epoch)
        }
        if stamp.epoch != lastPTSEpoch {
            lastPTSEpoch = stamp.epoch
            lastPTSUs = nil
        }
        if let pts = stamp.ptsUs {
            // A capture clock that runs backwards, or jumps by more than any
            // frame interval, inside one epoch. Recorded, not repaired.
            if let last = lastPTSUs, pts <= last || pts - last > 1_000_000 {
                counts.ptsDiscontinuities += 1
                eventOnQueue("pts_discontinuity", monoNs: stamp.rxMonoNs, fields: [
                    ("epoch", .optionalString(stamp.epoch?.uuidString)),
                    ("seq", .int(seq)),
                    ("prev_pts_us", .int(Int(last))),
                    ("pts_us", .int(Int(pts))),
                ])
            }
            lastPTSUs = pts
        }
        guard append(IMULogFormat.frameLine(stamp, sent: sent, seq: seq), isEvent: false) else {
            if closed { counts.framesAfterClose += 1 }
            return
        }
        counts.frames += 1
        if sent { counts.framesSent += 1 }
        if stamp.ptsUs == nil { counts.framesWithoutPTS += 1 }
    }

    private func motionOnQueue(_ reading: IMUMotionReading, rxMonoNs: UInt64) {
        guard reading.source == .glasses else {
            counts.motionOtherBySource[reading.source.rawValue, default: 0] += 1
            if !reportedFirstOtherSource {
                reportedFirstOtherSource = true
                eventOnQueue("first_non_glasses_sample", monoNs: rxMonoNs, fields: [
                    ("src", .string(reading.source.rawValue)),
                    ("ts_ns", .int(Int(reading.timestampNs))),
                ])
            }
            return
        }
        guard append(IMULogFormat.motionLine(reading, rxMonoNs: rxMonoNs), isEvent: false) else { return }
        counts.motionGlasses += 1

        if let last = lastMotionTsNs {
            let delta = reading.timestampNs - last
            if delta <= 0 {
                counts.motionNonMonotonic += 1
            } else {
                counts.motionMaxGapNs = max(counts.motionMaxGapNs, delta)
                if let interval = motionIntervalNs, delta * 2 > interval * 3 {
                    counts.motionGapEvents += 1
                    let missing = Int((Double(delta) / Double(interval)).rounded()) - 1
                    counts.motionEstimatedMissing += max(0, missing)
                }
            }
        }
        lastMotionTsNs = reading.timestampNs
        if tickFirstTsNs == nil { tickFirstTsNs = reading.timestampNs }
        tickLastTsNs = reading.timestampNs
        tickSamples += 1
    }

    /// Writes an epoch that `IMUCameraEpochs` has already opened.
    private func epochStartedOnQueue(_ rotation: IMUCameraEpochs.Rotation, reason: String, monoNs: UInt64) {
        epochCount += 1
        eventOnQueue("camera_start", monoNs: monoNs, fields: [
            ("epoch", .string(rotation.next.uuidString)),
            ("reason", .string(reason)),
            ("prev_epoch", .optionalString(rotation.previous?.uuidString)),
        ])
    }

    private func eventOnQueue(_ name: String, monoNs: UInt64, fields: [(String, IMUJSON)]) {
        append(IMULogFormat.eventLine(name, monoNs: monoNs, fields: fields), isEvent: true)
    }

    /// Accepts one line into the buffer, or refuses it: for a cap, because the
    /// log is closed, or because the file failed.
    @discardableResult
    private func append(_ line: String, isEvent: Bool) -> Bool {
        guard !closed else {
            counts.droppedAfterClose += 1
            return false
        }
        // A failed file is never written again, so a line accepted now would
        // only sit in memory for the rest of the session (Motion keeps coming
        // at 60 Hz). Counted, not held.
        guard failure == nil else {
            counts.droppedAfterFailure += 1
            return false
        }
        if !isEvent, capReason == nil,
           Double(DispatchTime.now().uptimeNanoseconds &- startMonoNs) / 1_000_000_000 >= limits.maxDuration {
            reachCap("duration")
        }
        if !isEvent, capReason != nil {
            counts.droppedAfterCap += 1
            return false
        }
        let data = Data(line.utf8)
        let cap = isEvent ? effectiveMaxBytes : effectiveMaxBytes - limits.eventReserveBytes
        guard bytesAccepted + data.count <= cap else {
            counts.droppedAfterCap += 1
            if !isEvent { reachCap(capIsFolder ? "folder" : "size") }
            return false
        }
        buffer.append(data)
        bytesAccepted += data.count
        if header != nil, buffer.count >= limits.flushThresholdBytes {
            flushOnQueue()
        }
        return true
    }

    /// Stops the data lines for good, says so in the console and the log, and
    /// tells the badge.
    private func reachCap(_ reason: String) {
        guard capReason == nil else { return }
        capReason = reason
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds &- startMonoNs) / 1_000_000_000
        print("[Glasses][IMURec] \(reason) cap reached after \(String(format: "%.0f", elapsed)) s and \(bytesAccepted) bytes; no more m/f lines are written to \(relativePath)")
        eventOnQueue("cap_reached", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
            ("cap", .string(reason)),
            ("bytes", .int(bytesAccepted)),
            ("elapsed_s", .double(elapsed)),
            ("max_bytes", .int(effectiveMaxBytes)),
            ("max_duration_s", .double(limits.maxDuration)),
            ("folder_max_bytes", .int(limits.folderMaxBytes)),
        ])
        publishStatus()
    }

    private func flushOnQueue() {
        guard let handle, !buffer.isEmpty, failure == nil else { return }
        let pending = buffer
        performIO { try handle.write(contentsOf: pending) }
        guard failure == nil else { return }
        bytesWritten += pending.count
        buffer.removeAll(keepingCapacity: true)
    }

    /// The phone is locking. The handle stays open -- a `completeUnlessOpen`
    /// file stays writable through the lock only while it is open -- so this is
    /// an event, and a durable point before the pocket: flush, then fsync.
    private func lockOnQueue() {
        guard protectedDataAvailable, !closed else { return }
        protectedDataAvailable = false
        eventOnQueue("protected_data_unavailable", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
            ("bytes_written", .int(bytesWritten)),
        ])
        flushOnQueue()
        if let handle, failure == nil {
            performIO { try handle.synchronize() }
        }
        print("[Glasses][IMURec] phone locking: flushed and synced; the file stays open and is still written")
        publishStatus()
    }

    private func unlockOnQueue() {
        guard !protectedDataAvailable else { return }
        protectedDataAvailable = true
        // `bytes_written` against the lock's: what reached the disk while the
        // phone was locked. The walk-5 pre-flight reads this pair.
        eventOnQueue("protected_data_available", monoNs: DispatchTime.now().uptimeNanoseconds, fields: [
            ("bytes_written", .int(bytesWritten)),
        ])
        print("[Glasses][IMURec] phone unlocked: \(bytesWritten) bytes on disk")
        publishStatus()
    }

    /// The periodic and closing `counts` body: the counters, plus what is on
    /// disk and whether the phone is locked, so a pulled log shows the file
    /// growing through a lock.
    private func countsFieldsOnQueue() -> [(String, IMUJSON)] {
        counts.jsonFields + [
            ("bytes_written", .int(bytesWritten)),
            ("protected_data_available", .bool(protectedDataAvailable)),
        ]
    }

    private func tick() {
        flushOnQueue()
        ticks += 1
        if let first = tickFirstTsNs, let last = tickLastTsNs, tickSamples >= 2, last > first {
            lastRateHz = Double(tickSamples - 1) / (Double(last - first) / 1_000_000_000)
        } else {
            lastRateHz = tickSamples == 0 ? 0 : nil
        }
        tickFirstTsNs = nil
        tickLastTsNs = nil
        tickSamples = 0
        if limits.countsEveryTicks > 0, ticks % limits.countsEveryTicks == 0 {
            eventOnQueue("counts", monoNs: DispatchTime.now().uptimeNanoseconds, fields: countsFieldsOnQueue())
        }
        publishStatus()
    }

    /// Locked or not: the handle is open, so the last lines can be written now.
    private func requestCloseOnQueue(reason: String, monoNs: UInt64) {
        guard !closed else { return }
        finishCloseOnQueue(reason: reason, monoNs: monoNs)
    }

    private func finishCloseOnQueue(reason: String, monoNs: UInt64) {
        // Last, so they are the file's last lines.
        eventOnQueue("counts", monoNs: monoNs, fields: countsFieldsOnQueue())
        eventOnQueue("close", monoNs: monoNs, fields: [
            ("reason", .string(reason)),
            ("bytes", .int(bytesAccepted)),
            ("epochs", .int(epochCount)),
            ("cap", .optionalString(capReason)),
        ])
        timer?.cancel()
        timer = nil
        removeObservers()
        flushOnQueue()
        if let handle {
            performIO {
                try handle.synchronize()
                try handle.close()
            }
        }
        handle = nil
        closed = true
        if header != nil {
            print("[Glasses][IMURec] closed \(relativePath) (\(reason)): \(bytesWritten) bytes, m=\(counts.motionGlasses) f=\(counts.frames) sent=\(counts.framesSent) epochs=\(epochCount) cap=\(capReason ?? "none")")
        }
        publishStatus()
        let summary = summaryOnQueue()
        let completions = closeCompletions
        closeCompletions = []
        completions.forEach { $0(summary) }
    }

    /// Every file operation goes through here, so the audit is complete.
    ///
    /// The first failure stops the file for good: the handle is closed, the
    /// timer stopped, and whatever was buffered released, since nothing will
    /// ever write it. `append` then refuses every later line.
    private func performIO(_ body: () throws -> Void) {
        io.operations += 1
        if Thread.isMainThread { io.onMainThread += 1 }
        if DispatchQueue.getSpecific(key: queueKey) == nil { io.offQueue += 1 }
        do {
            try body()
        } catch {
            guard failure == nil else { return }
            failure = error.localizedDescription
            print("[Glasses][IMURec] recording stopped: \(error.localizedDescription)")
            try? handle?.close()
            handle = nil
            timer?.cancel()
            timer = nil
            buffer = Data()
            publishStatus()
        }
    }

    private func publishStatus() {
        guard let onStatus else { return }
        onStatus(statusOnQueue())
    }

    private func statusOnQueue() -> IMURecorderStatus {
        let phase: IMURecorderStatus.Phase
        if let refusal {
            phase = .refused(refusal)
        } else if let failure {
            phase = .failed(failure)
        } else if closed {
            phase = .closed
        } else if header != nil {
            phase = .recording
        } else {
            phase = .idle
        }
        return IMURecorderStatus(
            phase: phase,
            fileName: relativePath,
            bytes: bytesAccepted,
            counts: counts,
            epochs: epochCount,
            motionConfiguredHz: motionConfiguredHz,
            motionRateHz: lastRateHz,
            capReason: capReason,
            phoneLocked: !protectedDataAvailable
        )
    }

    private func summaryOnQueue() -> IMURecorderSummary {
        IMURecorderSummary(
            fileURL: fileURL,
            bytesWritten: bytesWritten,
            bytesHeld: buffer.count,
            counts: counts,
            epochs: epochCount,
            capReason: capReason,
            effectiveMaxBytes: effectiveMaxBytes,
            closed: closed,
            failure: failure,
            io: io
        )
    }

    // MARK: Formatting helpers

    /// `20260925T143012Z`: ISO 8601 basic format, safe in a file name.
    static func isoBasic(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        return formatter.string(from: date)
    }

    static func isoExtended(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    /// The Data Protection class a file or folder reports, or `nil` when it
    /// reports none (the Simulator may not). File I/O: call it on the queue.
    static func protection(of url: URL) -> FileProtectionType? {
        (try? FileManager.default.attributesOfItem(atPath: url.path))?[.protectionKey] as? FileProtectionType
    }

    static func name(of state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

nonisolated enum IMURecorderError: LocalizedError {
    case folderFull(existingBytes: Int, limit: Int)

    var errorDescription: String? {
        switch self {
        case .folderFull(let existing, let limit):
            return "imu-logs already holds \(existing) of its \(limit) bytes; delete the logs in Developer Tools to record again"
        }
    }
}

/// What a closed (or draining) recorder reports, for tests and the console.
nonisolated struct IMURecorderSummary: Equatable, Sendable {
    let fileURL: URL?
    let bytesWritten: Int
    /// Accepted but not yet on disk: the buffer, or what is held while locked.
    let bytesHeld: Int
    let counts: IMURecorderCounts
    let epochs: Int
    let capReason: String?
    let effectiveMaxBytes: Int
    let closed: Bool
    let failure: String?
    let io: IMUIOAudit
}

// MARK: - The folder

/// How many logs there are and how much they hold.
nonisolated struct IMULogInventory: Equatable, Sendable {
    var files = 0
    var bytes = 0
}

/// Reads and purges `imu-logs/`. Every function does file I/O, so callers run
/// them off the main thread.
nonisolated enum IMULogStore {
    static func folder(base: URL?) throws -> URL {
        let base = try base ?? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        return base.appendingPathComponent(IMURecorder.directoryName, isDirectory: true)
    }

    static func logFiles(in folder: URL) -> [URL] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]
        )) ?? []
        return contents.filter { $0.pathExtension == "jsonl" }
    }

    static func totalBytes(in folder: URL) -> Int {
        logFiles(in: folder).reduce(0) { total, url in
            total + ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
    }

    static func inventory(base: URL?) -> IMULogInventory {
        guard let folder = try? folder(base: base) else { return IMULogInventory() }
        let files = logFiles(in: folder)
        return IMULogInventory(files: files.count, bytes: totalBytes(in: folder))
    }

    /// Deletes every log in the folder -- on this phone only -- and returns what
    /// was deleted.
    static func purge(base: URL?) -> IMULogInventory {
        guard let folder = try? folder(base: base) else { return IMULogInventory() }
        var deleted = IMULogInventory()
        for url in logFiles(in: folder) {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            if (try? FileManager.default.removeItem(at: url)) != nil {
                deleted.files += 1
                deleted.bytes += size
            }
        }
        return deleted
    }
}

/// The recorder's readout for Developer Tools and the recording badge, on its
/// own object for the reason `MotionProbe` gives: nearly every screen observes
/// `GlassesConnection`, and only these two should redraw when the recorder
/// reports.
@MainActor
final class IMURecorderReadout: ObservableObject {
    @Published private(set) var status = IMURecorderStatus()
    @Published private(set) var indicator: IMURecordingIndicator = .off

    func update(_ status: IMURecorderStatus) {
        if status != self.status { self.status = status }
        setIndicator(IMURecordingIndicator(status))
    }

    func setIndicator(_ indicator: IMURecordingIndicator) {
        if indicator != self.indicator { self.indicator = indicator }
    }
}

#endif
