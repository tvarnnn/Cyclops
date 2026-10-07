//
//  MotionProbe.swift
//  Glasses
//

// Entire file is DEBUG-only, like the Developer Tools that read it. It is a
// spike: proof that DAT 1.0.0's experimental Motion capability reaches this
// app, and nothing else. No sample leaves the phone, nothing here is shown to
// a wearer, and nothing here plays a sound or a haptic.
#if DEBUG

import Combine
import Foundation
import MWDATMotion

/// One IMU reading, reduced to what the probe keeps.
///
/// Two clocks are carried on purpose and are never mixed. `deviceTimestampNs`
/// is `MotionSample.timestampNs`, which DAT documents as the glasses' own
/// monotonic clock; `receivedAt` is this phone's `MonotonicClock`, read the
/// moment the sample came off DAT's stream. The first says when the glasses
/// measured; the second says when the phone could act, and the difference is
/// the transport. A rate computed from one is not a rate computed from the
/// other.
nonisolated struct MotionProbeSample: Equatable, Sendable {
    let deviceTimestampNs: Int64
    let receivedAt: TimeInterval
    /// |gyroscope| in rad/s, or `nil` when the sample carried no gyroscope
    /// reading. DAT makes every sensor field independently optional, so a
    /// missing gyroscope is recorded as missing, never as zero.
    let gyroMagnitude: Float?

    init(deviceTimestampNs: Int64, receivedAt: TimeInterval, gyroMagnitude: Float?) {
        self.deviceTimestampNs = deviceTimestampNs
        self.receivedAt = receivedAt
        self.gyroMagnitude = gyroMagnitude
    }

    init(_ sample: MotionSample, receivedAt: TimeInterval) {
        self.init(
            deviceTimestampNs: sample.timestampNs,
            receivedAt: receivedAt,
            gyroMagnitude: sample.gyroscope.map { ($0.x * $0.x + $0.y * $0.y + $0.z * $0.z).squareRoot() }
        )
    }
}

/// The last few seconds of samples, bounded by receipt time.
///
/// A value type with no clock of its own, so every figure it reports can be
/// tested from fixed numbers.
nonisolated struct MotionSampleRing: Sendable {
    /// How much history is kept. Three seconds is enough for a stable rate at
    /// 30 Hz (~90 samples) and short enough that the readout follows the head.
    static let defaultWindow: TimeInterval = 3

    /// A hard ceiling independent of the window, so a burst that arrives
    /// faster than any configured rate cannot grow the buffer without bound.
    /// DAT's own sample stream holds 256 before it starts dropping.
    static let capacity = 256

    let window: TimeInterval
    private(set) var samples: [MotionProbeSample] = []
    /// Every sample ever appended, including those since evicted.
    private(set) var totalReceived = 0

    init(window: TimeInterval = MotionSampleRing.defaultWindow) {
        self.window = window
    }

    mutating func append(_ sample: MotionProbeSample) {
        samples.append(sample)
        totalReceived += 1
        let horizon = sample.receivedAt - window
        let stale = samples.prefix { $0.receivedAt < horizon }.count
        let overflow = max(0, samples.count - stale - Self.capacity)
        samples.removeFirst(stale + overflow)
    }

    mutating func removeAll() {
        samples.removeAll()
        totalReceived = 0
    }

    /// Samples per second as the **phone received them**, over the window.
    /// `nil` until two samples span a measurable interval.
    var arrivalRateHz: Double? {
        guard let first = samples.first, let last = samples.last, samples.count >= 2 else { return nil }
        let span = last.receivedAt - first.receivedAt
        guard span > 0 else { return nil }
        return Double(samples.count - 1) / span
    }

    /// Samples per second by the **glasses' own clock**, over the window.
    /// DAT documents the requested rate as a target, not a guarantee, and says
    /// to compute intervals from `timestampNs` rather than assume a cadence.
    var deviceRateHz: Double? {
        guard let first = samples.first, let last = samples.last, samples.count >= 2 else { return nil }
        let span = Double(last.deviceTimestampNs - first.deviceTimestampNs) / 1_000_000_000
        guard span > 0 else { return nil }
        return Double(samples.count - 1) / span
    }

    /// The most recent gyroscope magnitude, skipping samples that had none.
    var latestGyroMagnitude: Float? {
        samples.last { $0.gyroMagnitude != nil }?.gyroMagnitude
    }

    /// The largest gyroscope magnitude in the window.
    var peakGyroMagnitude: Float? {
        samples.compactMap(\.gyroMagnitude).max()
    }
}

/// What Developer Tools shows. Rebuilt at most twice a second.
nonisolated struct MotionProbeSnapshot: Equatable, Sendable {
    var state: String = "off"
    var lastError: String?
    var samplesInWindow = 0
    var totalReceived = 0
    var arrivalRateHz: Double?
    var deviceRateHz: Double?
    var latestGyroMagnitude: Float?
    var peakGyroMagnitude: Float?
}

/// Receives the Motion capability's samples for the Developer Tools readout.
///
/// A separate `ObservableObject` rather than `@Published` state on
/// `GlassesConnection`, because nearly every screen observes that object: a
/// property changing thirty times a second there would re-render the whole app
/// at the IMU rate. Only the Developer Tools section observes this, and it
/// republishes at 2 Hz, the same budget `SenderMetrics` uses.
@MainActor
final class MotionProbe: ObservableObject {
    static let publishInterval: TimeInterval = 0.5

    @Published private(set) var snapshot = MotionProbeSnapshot()

    private var ring = MotionSampleRing()
    private var state = "off"
    private var lastError: String?
    private var lastPublishedAt: TimeInterval = -.infinity

    /// The buffered window, for tests and for anything that needs the raw
    /// samples rather than the rounded readout.
    var samples: [MotionProbeSample] { ring.samples }
    var totalReceived: Int { ring.totalReceived }

    /// Clears everything, at the start of each attempt, so a previous
    /// session's samples can never be read as this one's.
    func reset() {
        ring.removeAll()
        state = "off"
        lastError = nil
        publish(at: MonotonicClock.now)
    }

    func record(_ sample: MotionProbeSample) {
        ring.append(sample)
        guard sample.receivedAt - lastPublishedAt >= Self.publishInterval else { return }
        publish(at: sample.receivedAt)
    }

    func noteState(_ description: String) {
        state = description
        publish(at: MonotonicClock.now)
    }

    func noteFailure(_ description: String) {
        lastError = description
        publish(at: MonotonicClock.now)
    }

    private func publish(at now: TimeInterval) {
        lastPublishedAt = now
        snapshot = MotionProbeSnapshot(
            state: state,
            lastError: lastError,
            samplesInWindow: ring.samples.count,
            totalReceived: ring.totalReceived,
            arrivalRateHz: ring.arrivalRateHz,
            deviceRateHz: ring.deviceRateHz,
            latestGyroMagnitude: ring.latestGyroMagnitude,
            peakGyroMagnitude: ring.peakGyroMagnitude
        )
    }
}

// MARK: - Live 30 s probe

import MWDATCore
import os

/// What the live probe shows. Pure value, so the arithmetic is testable.
nonisolated struct MotionLiveReadout: Equatable, Sendable {
    var state = "idle"
    var running = false
    var elapsed: TimeInterval = 0
    var samples = 0
    var accel = 0, gyro = 0, mag = 0, orientation = 0
    var samplesPerSecond: Double?
    var latestGyro: String = "-"
    var latestOrientation: String = "-"
    var error: String?
    var startAttempts = 0

    var summary: String {
        "state=\(state) samples=\(samples) rate=\(samplesPerSecond.map { String(format: "%.1f", $0) } ?? "-")/s "
            + "nonNull(accel=\(accel) gyro=\(gyro) mag=\(mag) orient=\(orientation)) "
            + "starts=\(startAttempts) error=\(error ?? "none")"
    }
}

/// Tap-driven, 30 s Motion probe. Adds Motion to an already STARTED session,
/// starts it, retries `start()` once after `sensorUnavailable`, then removes it.
/// Touches nothing of capture, streaming or Tower: the session is only handed in.
@MainActor
final class MotionLiveProbe: ObservableObject {
    static let duration: TimeInterval = 30
    private static let log = Logger(subsystem: "Glasses", category: "MotionProbe")

    @Published private(set) var readout = MotionLiveReadout()
    private var task: Task<Void, Never>?
    private let tokens = ListenerTokenBag()
    private var motion: Motion?
    private var retried = false

    func start(on session: DeviceSession) {
        guard task == nil else { return }
        readout = MotionLiveReadout()
        retried = false
        let motion: Motion
        do {
            guard let m = try session.addMotion(configuration: MotionConfiguration(samplingRate: .hz30)) else {
                readout.error = "addMotion returned nil (session not started)"
                finish(session)
                return
            }
            motion = m
        } catch {
            readout.error = "addMotion threw: \(error.description)"
            finish(session)
            return
        }
        self.motion = motion
        readout.running = true
        motion.statePublisher.listen { [weak self] state in
            Task { @MainActor [weak self] in self?.readout.state = state.description }
        }.store(in: tokens)
        motion.errorPublisher.listen { [weak self] error in
            Task { @MainActor [weak self] in self?.handle(error, motion: motion) }
        }.store(in: tokens)
        let began = Date()
        let samples = motion.samples
        readout.startAttempts = 1
        motion.start()
        task = Task { [weak self] in
            let reader = Task { [weak self] in
                for await s in samples { self?.ingest(s, since: began) }
            }
            while !Task.isCancelled, Date().timeIntervalSince(began) < Self.duration {
                try? await Task.sleep(nanoseconds: 250_000_000)
                self?.readout.elapsed = Date().timeIntervalSince(began)
            }
            reader.cancel()
            guard let self else { return }
            self.finish(session)
        }
    }

    func cancel() { task?.cancel() }

    private func ingest(_ s: MotionSample, since began: Date) {
        readout.samples += 1
        if s.accelerometer != nil { readout.accel += 1 }
        if s.gyroscope != nil { readout.gyro += 1 }
        if s.magnetometer != nil { readout.mag += 1 }
        if s.orientation != nil { readout.orientation += 1 }
        if let g = s.gyroscope { readout.latestGyro = String(format: "%.3f %.3f %.3f rad/s", g.x, g.y, g.z) }
        if let q = s.orientation { readout.latestOrientation = String(format: "x%.3f y%.3f z%.3f w%.3f", q.x, q.y, q.z, q.w) }
        let t = Date().timeIntervalSince(began)
        if t > 0.5 { readout.samplesPerSecond = Double(readout.samples) / t }
    }

    private func handle(_ error: MotionError, motion: Motion) {
        readout.error = "\(error) — \(error.description)"
        guard error == .sensorUnavailable, !retried, self.motion === motion else { return }
        retried = true
        Task { [weak self] in
            // start() is legal again only once the state is back to stopped.
            for _ in 0..<20 where motion.state != .stopped {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            guard let self, self.task != nil, self.motion === motion else { return }
            self.readout.startAttempts += 1
            motion.start()
        }
    }

    private func finish(_ session: DeviceSession) {
        tokens.clear()
        motion?.stop()
        motion = nil
        do { try session.removeMotion() } catch {
            Self.log.error("removeMotion failed: \(error.description, privacy: .public)")
        }
        readout.running = false
        task = nil
        Self.log.notice("Motion probe summary: \(self.readout.summary, privacy: .public)")
        print("[Glasses][MotionProbe] \(readout.summary)")
    }
}

/// The probe's entry exists only in DEBUG. Pinned by a test that is itself DEBUG.
enum MotionProbeEntry {
    static let isDebugOnly = true
    static let title = "Motion probe"
}

#endif
