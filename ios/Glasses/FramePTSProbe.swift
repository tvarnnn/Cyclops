//
//  FramePTSProbe.swift
//  Glasses
//

#if DEBUG
import CoreMedia
import Foundation

/// Measures what clock DAT's `VideoFrame.sampleBuffer` presentation timestamps
/// are on.
///
/// **Pure observation.** It reads a timestamp and prints. It does not touch the
/// frame rate gate, the send window, the Tower, or World Builder, and removing
/// this file removes the whole experiment.
///
/// ## Why the reading happens where it does
///
/// `MonotonicClock.now` in the frame listener is sampled *inside* a
/// `Task { @MainActor … }`, so it carries main-actor queueing latency as well
/// as transport latency. Comparing a PTS against that number would measure the
/// two together and attribute both to the wire. This probe is therefore fed
/// from the listener closure itself, on DAT's own callback thread, before any
/// hop — so `hostNow` is as close to "the frame arrived" as this process can
/// observe.
///
/// ## What the numbers can and cannot settle
///
/// `CMTime` carries no clock identity. Two readings are therefore compared:
///
/// - `hostNow` — `mach_absolute_time` via `DispatchTime`, the same base as
///   `CMClockGetHostTimeClock()`.
/// - `pts` — whatever DAT put in the buffer.
///
/// If `offset = hostNow − pts` is small and **stable**, the PTS is on the host
/// clock and was stamped near arrival. If `offset` is large but stable, the PTS
/// is on some other epoch with a fixed relationship to boot time. If `offset`
/// **drifts**, the two clocks run at different rates, which is what an
/// independent oscillator on the glasses would produce. Drift is the single
/// most diagnostic quantity here, so it is reported as a slope over the run.
///
/// A stable offset alone does **not** prove capture time: a phone-side stamp
/// applied on arrival also produces a stable offset. What separates them is
/// jitter. If the PTS were stamped on arrival it would inherit every Bluetooth
/// delay, so `pts` deltas and `hostNow` deltas would move together and their
/// difference would be near zero. A capture-side stamp is regular regardless of
/// how irregularly the frame arrives, so its deltas stay clean while arrival
/// deltas scatter. **The summary's `residual(d_pts-d_host)` is that
/// difference, and it is the measurement this probe exists for.**
///
/// The `[PTSProbe]` console lines are the probe's only output. Keep them
/// `print`s (stdout), word for word: 07-PLATFORM-CONSTRAINTS.md §9 was
/// measured by reading them.
nonisolated final class FramePTSProbe {

    static let shared = FramePTSProbe()

    private let lock = NSLock()

    private struct Sample {
        let seq: Int
        let pts: Double
        let host: Double
    }

    private var samples: [Sample] = []
    private var seq = 0
    private var firstReported = false
    private var lastSummaryAt: Double = 0
    private var lastPTS: Double?
    private var lastHost: Double?
    private var discontinuities = 0

    /// The last frame of the previous stream session, kept across the gap.
    ///
    /// This is the whole reconnect experiment. World Builder chains captures
    /// across a reconnect, so whether the capture clock's epoch survives one
    /// decides whether a PTS from before the drop is comparable with one after
    /// it. The session boundary line names one of three outcomes:
    ///
    /// - **free-running** — the new session's first PTS is roughly the old
    ///   session's last PTS plus the wall-clock gap, so `d_pts ≈ d_host`.
    /// - **per-session** — the count restarts, so `d_pts` is negative.
    /// - **suspended but monotonic** — `d_pts` is positive and far smaller
    ///   than `d_host`: the clock neither resets nor measures the gap. This is
    ///   what a stop measured (07-PLATFORM-CONSTRAINTS.md §9.1).
    private var previousSessionEnd: (pts: Double, host: Double, seq: Int)?
    private var sessionIndex = 1

    /// How often the rolling summary prints. Long enough that the console is
    /// readable during a walk, short enough to see a drift develop.
    private static let summaryInterval: Double = 5.0

    /// A PTS gap this large counts as a break in the capture clock.
    ///
    /// Judged on the capture clock alone: time running backwards, or a gap far
    /// longer than any frame interval this stream uses (24 fps nominal, worst
    /// observed 0.05 s). Never against `d_host`: a break called when `d_pts`
    /// exceeded `d_host` five-fold fired 24 times over a 45 s run, every one a
    /// false positive — `d_pts` was the nominal 0.04166 s while `d_host` was
    /// 2-8 ms, a *burst arrival* of evenly-captured frames, not a clock event.
    /// The capture and arrival clocks are the two things this probe exists to
    /// show are independent.
    private static let discontinuityGap: Double = 1.0

    func record(sampleBuffer: CMSampleBuffer, hostNow: Double) {
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        lock.lock()
        defer { lock.unlock() }

        seq += 1

        guard time.isValid, !time.isIndefinite else {
            if seq % 100 == 1 {
                print("[PTSProbe] frame #\(seq) PTS is INVALID (flags=\(time.flags.rawValue))")
            }
            return
        }

        let pts = CMTimeGetSeconds(time)

        // The first frame is the only place the raw, unreduced values are
        // visible, and they answer questions the deltas cannot: a timescale of
        // 1_000_000_000 says nanoseconds, 600 or 90_000 say a media pipeline,
        // and a PTS near zero says the stream is its own epoch while a PTS near
        // system uptime says it is not.
        if !firstReported {
            firstReported = true
            lastSummaryAt = hostNow
            let uptime = ProcessInfo.processInfo.systemUptime
            let hostClock = CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
            print("""
                [PTSProbe] FIRST FRAME session=\(sessionIndex) \
                pts_seconds=\(pts) \
                pts_value=\(time.value) pts_timescale=\(time.timescale) \
                pts_epoch=\(time.epoch) pts_flags=\(time.flags.rawValue) \
                host_now=\(hostNow) host_clock=\(hostClock) \
                process_uptime=\(uptime) \
                offset_host_minus_pts=\(hostNow - pts)
                """)

            // The answer, printed as the comparison rather than as two numbers
            // a reader has to subtract.
            if let previous = previousSessionEnd {
                let dPTS = pts - previous.pts
                let dHost = hostNow - previous.host
                // Three outcomes; see `previousSessionEnd`. The third is the
                // one that matters: the clock neither resets nor tracks wall
                // time, so it stays monotonic across a stop while silently
                // understating how long the stop was.
                let ratio = dHost > 0 ? dPTS / dHost * 100 : 0
                let verdict: String
                if dPTS < 0 {
                    verdict = "PER-SESSION (clock went backwards across the gap)"
                } else if dHost > 0, abs(dPTS - dHost) < 0.5 {
                    verdict = "FREE-RUNNING (capture clock ran through the gap)"
                } else if dHost > 0, dPTS < dHost {
                    verdict = "SUSPENDED-BUT-MONOTONIC (advanced \(String(format: "%.1f", ratio))% of the gap: did not reset, did not measure it either)"
                } else {
                    verdict = "UNEXPLAINED (clock advanced further than wall time)"
                }
                print("""
                    [PTSProbe] SESSION BOUNDARY \(previous.seq) -> \(seq) \
                    prev_last_pts=\(previous.pts) new_first_pts=\(pts) \
                    d_pts=\(dPTS) d_host=\(dHost) \
                    difference=\(dPTS - dHost) \
                    VERDICT=\(verdict)
                    """)
            }
        }

        if let lastPTS, let lastHost {
            let dPTS = pts - lastPTS
            let dHost = hostNow - lastHost
            // Backwards or wildly large steps are reported the moment they
            // happen: an average would hide exactly the event that matters.
            if dPTS <= 0 || dPTS > Self.discontinuityGap {
                discontinuities += 1
                print("""
                    [PTSProbe] DISCONTINUITY #\(discontinuities) at frame \(seq) \
                    d_pts=\(dPTS) d_host=\(dHost) pts=\(pts) host=\(hostNow)
                    """)
            }
        }
        lastPTS = pts
        lastHost = hostNow

        samples.append(Sample(seq: seq, pts: pts, host: hostNow))

        if hostNow - lastSummaryAt >= Self.summaryInterval {
            lastSummaryAt = hostNow
            summarise()
        }
    }

    /// Prints the whole run so far. Called on the interval and safe to call
    /// again at any time.
    private func summarise() {
        guard samples.count >= 3 else { return }

        let offsets = samples.map { $0.host - $0.pts }
        let meanOffset = offsets.reduce(0, +) / Double(offsets.count)
        let minOffset = offsets.min() ?? 0
        let maxOffset = offsets.max() ?? 0

        // Least-squares slope of offset against host time: seconds of drift per
        // second elapsed. A shared clock gives ~0; independent oscillators give
        // a consistent non-zero slope, and the sign says which runs faster.
        let t0 = samples[0].host
        let xs = samples.map { $0.host - t0 }
        let meanX = xs.reduce(0, +) / Double(xs.count)
        let meanY = meanOffset
        var num = 0.0
        var den = 0.0
        for (x, y) in zip(xs, offsets) {
            num += (x - meanX) * (y - meanY)
            den += (x - meanX) * (x - meanX)
        }
        let slope = den > 0 ? num / den : 0
        let elapsed = xs.last ?? 0

        // Per-frame deltas, and the residual that separates the two hypotheses.
        var dPTS: [Double] = []
        var dHost: [Double] = []
        var residual: [Double] = []
        for i in 1..<samples.count {
            let a = samples[i - 1]
            let b = samples[i]
            let p = b.pts - a.pts
            let h = b.host - a.host
            dPTS.append(p)
            dHost.append(h)
            residual.append(p - h)
        }

        func stats(_ v: [Double]) -> (mean: Double, sd: Double, min: Double, max: Double) {
            guard !v.isEmpty else { return (0, 0, 0, 0) }
            let m = v.reduce(0, +) / Double(v.count)
            let variance = v.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(v.count)
            return (m, variance.squareRoot(), v.min() ?? 0, v.max() ?? 0)
        }

        let p = stats(dPTS)
        let h = stats(dHost)
        let r = stats(residual)

        print("""
            [PTSProbe] SUMMARY n=\(samples.count) elapsed=\(String(format: "%.2f", elapsed))s \
            discontinuities=\(discontinuities)
            [PTSProbe]   offset host-pts: mean=\(String(format: "%.6f", meanOffset)) \
            min=\(String(format: "%.6f", minOffset)) max=\(String(format: "%.6f", maxOffset)) \
            spread=\(String(format: "%.6f", maxOffset - minOffset))
            [PTSProbe]   drift slope=\(String(format: "%.3e", slope)) s/s \
            (\(String(format: "%.2f", slope * 1_000_000)) ppm over \(String(format: "%.1f", elapsed))s)
            [PTSProbe]   d_pts  mean=\(String(format: "%.6f", p.mean)) sd=\(String(format: "%.6f", p.sd)) \
            min=\(String(format: "%.6f", p.min)) max=\(String(format: "%.6f", p.max))
            [PTSProbe]   d_host mean=\(String(format: "%.6f", h.mean)) sd=\(String(format: "%.6f", h.sd)) \
            min=\(String(format: "%.6f", h.min)) max=\(String(format: "%.6f", h.max))
            [PTSProbe]   residual(d_pts-d_host) mean=\(String(format: "%.6f", r.mean)) \
            sd=\(String(format: "%.6f", r.sd)) min=\(String(format: "%.6f", r.min)) max=\(String(format: "%.6f", r.max))
            """)
    }

    /// Emits the full run. Called when the stream stops, so a walk that ends
    /// still reports rather than leaving the last window unsummarised.
    func reportFinal(reason: String) {
        lock.lock()
        defer { lock.unlock() }
        guard !samples.isEmpty else { return }
        print("[PTSProbe] FINAL session=\(sessionIndex) (\(reason))")
        summarise()

        // Kept: the anchor the next session is measured against.
        if let lastPTS, let lastHost {
            previousSessionEnd = (pts: lastPTS, host: lastHost, seq: seq)
        }
        // Cleared: everything whose meaning does not survive the gap. The
        // inter-session interval is not a frame interval, and leaving it in
        // `samples` would put a multi-second outlier into the next session's
        // jitter statistics and swamp the millisecond effects being measured.
        samples.removeAll()
        lastPTS = nil
        lastHost = nil
        firstReported = false
        sessionIndex += 1
    }
}
#endif
