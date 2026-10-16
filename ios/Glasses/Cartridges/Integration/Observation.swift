//
//  Observation.swift
//  Glasses
//

import Foundation

/// Provenance types shared by every cartridge that displays something the Tower
/// inferred rather than measured.
///
/// These are not a convenience layer. `docs/02-DEVELOPMENT-RULES.md` Rule 16
/// and `docs/07-PLATFORM-CONSTRAINTS.md` make them obligations:
///
/// > **2. Inference ≠ Measurement.** ML-derived depth, object identity,
/// > semantic interpretation, and similar outputs must not be represented as
/// > equivalent to directly measured sensor values.
///
/// > **4. Confidence Must Survive the Pipeline.** Where an observation
/// > originates from probabilistic inference, uncertainty/confidence must not
/// > disappear simply because the observation is persisted, transmitted, fused,
/// > or consumed by another module.
///
/// This layer is the last link in that pipeline, and the easiest place for
/// confidence to disappear: a view can always choose not to draw a number it
/// was given. Encoding provenance *in the type* is what makes that omission a
/// deliberate act rather than an oversight — there is no way to hold a value
/// here without also holding where it came from.

// MARK: - Scale

/// How a spatial figure was arrived at.
///
/// Shared rather than World Builder's own because Scene Understanding needs the
/// identical distinction the moment it reports a distance to a tracked entity,
/// and a second copy of a rule this load-bearing is how the two copies come to
/// disagree. Spelled `World…` because that is the name every call site and
/// test uses.
///
/// This is not presentation garnish. `docs/modules/WORLD-BUILD.md` makes it a
/// hard requirement: the target glasses are **monocular RGB only**, with no
/// LiDAR and no stereo, so any distance the system produces is inferred rather
/// than measured. That document's rule is absolute —
///
/// > World Build must never represent monocularly inferred depth as ground-truth
/// > physical distance. Any distance figure derived from monocular inference
/// > must be identifiable as an estimate wherever it is stored, *displayed*, or
/// > consumed by another module.
///
/// "Displayed" is this layer. Encoding provenance in the type is what stops a
/// future view from rendering an inferred number as though it were measured:
/// there is no way to show a figure without also having said where it came
/// from.
nonisolated enum WorldScaleSemantics: Equatable, Sendable, CaseIterable {
    /// Structure and layout without any claimed absolute scale. The honest
    /// default for multi-view geometry on monocular input.
    case relative
    /// A distance estimate from ML monocular depth or geometric inference.
    /// Carries model uncertainty and must always be labelled as an estimate.
    case inferredMetric
    /// Depth from dedicated depth-sensing hardware. **Unreachable on the
    /// current target glasses** — kept because WORLD-BUILD.md requires the
    /// model to accommodate a future measured-depth source without rewriting
    /// the representation, and because its absence is what makes the other
    /// cases meaningful.
    case measuredMetric
    /// The Tower has not said, or has not established scale yet.
    case unknown

    /// Short label for a metric row.
    var displayName: String {
        switch self {
        case .relative: return "Relative"
        case .inferredMetric: return "Estimated"
        case .measuredMetric: return "Measured"
        case .unknown: return "Unknown"
        }
    }

    /// Whether a figure carrying this provenance must be presented as an
    /// estimate rather than as a fact. Views must consult this before rendering
    /// any distance.
    var isEstimate: Bool {
        switch self {
        case .inferredMetric: return true
        case .relative, .measuredMetric, .unknown: return false
        }
    }

    /// One sentence a person can act on, for a detail row or accessibility
    /// value.
    var explanation: String {
        switch self {
        case .relative:
            return "Shape and layout only. No real-world distances are claimed."
        case .inferredMetric:
            return "Distances are estimated from a single camera. Treat them as approximate, not measured."
        case .measuredMetric:
            return "Distances come from depth-sensing hardware."
        case .unknown:
            return "Scale has not been established."
        }
    }
}

// MARK: - Figures

/// Renders a number the Tower reported, with the unit the Tower named — and
/// bare when it named none.
///
/// ## Why this exists rather than a format string at each call site
///
/// Because `String(format: "%.1f m", value)` is how "metric" silently becomes
/// "metres". `WorldScaleSemantics.inferredMetric` says a figure is metric *in
/// kind*; it says nothing about what unit it counts in, and the Tower has named
/// none. `CVMetric` already took this position — *"the Tower's unit string, if
/// any. Never assumed"* — and two spatial screens were quietly doing the
/// opposite until this was extracted.
///
/// A bare number is not a worse answer than an invented unit. It is the honest
/// rendering of an unlabelled quantity.
enum ReportedFigure {
    static func format(_ value: Double, unit: String?) -> String {
        let number: String
        if value == value.rounded() && abs(value) < 1e9 {
            number = String(Int(value))
        } else {
            number = String(format: "%.1f", value)
        }
        guard let unit, !unit.isEmpty else { return number }
        return number + " " + unit
    }
}

// MARK: - Inference provenance

/// Whether a value the Tower reported was measured or inferred, and with what
/// confidence.
///
/// Core Principle 2 and 4 in one type. Used by Experimental CV Lab (every
/// experiment output is model inference unless validated against ground truth —
/// `docs/modules/EXPERIMENTAL-CV.md`, "Model Output vs. Measured Fact"),
/// Scene Understanding (detection and orientation), and Document Memory
/// (OCR/summarisation confidence). Three cartridges, one rule.
enum ObservationProvenance: Equatable, Sendable {
    /// A direct reading — a counter the Tower keeps, a timing it measured, a
    /// byte count. Not a model output.
    case measured
    /// A model output. `confidence` is `nil` when the Tower reported an
    /// inference without one, which is a *worse* state than a low confidence
    /// and must not be rendered as certainty.
    case inferred(confidence: Double?)
    /// The Tower did not say. Neither measured nor inferred may be assumed —
    /// Core Principle 1, an observation is evidence, not automatically fact.
    case unknown

    var isInference: Bool {
        if case .inferred = self { return true }
        return false
    }

    /// The confidence to show, if there is one. `nil` for `measured` because a
    /// measurement has no model confidence, and for `unknown` because inventing
    /// one is the failure this type exists to prevent.
    var confidence: Double? {
        if case .inferred(let confidence) = self { return confidence }
        return nil
    }

    /// A caveat that must accompany the value, or `nil` if none is owed.
    ///
    /// Deliberately not phrased as a percentage when the Tower gave no number:
    /// "estimated" with no figure is honest, "0% confident" is not.
    var caveat: String? {
        switch self {
        case .measured:
            return nil
        case .inferred(let confidence):
            guard let confidence else {
                return "Estimated by a model. The Tower did not report a confidence."
            }
            return "Estimated by a model, confidence \(Self.percent(confidence))."
        case .unknown:
            return "The Tower did not say whether this was measured or estimated."
        }
    }

    /// Confidence formatted for display, clamped so a malformed value from the
    /// Tower cannot render as "170%".
    ///
    /// Clamping is not the same as hiding: an out-of-range confidence is a Tower
    /// defect worth seeing in a log, and the decode site — not this formatter —
    /// is where it should be noticed.
    ///
    /// ## Why `isFinite` comes before the clamp
    ///
    /// The clamp alone does not clamp everything. **Swift's `min`/`max`
    /// propagate NaN** — every comparison with NaN is false, so
    /// `min(max(.nan, 0), 1) == .nan` — and `Int(Double.nan.rounded())` is a
    /// **trap**, not a garbage number, as is `Int` of `+∞` or of any value past
    /// `Int.max` (`1e300` is legal JSON that `JSONSerialization` hands over as
    /// a `Double`).
    ///
    /// This formatter is reached from `confidence`, `best_score`,
    /// `detector_score`, verification scores, `bounding_box_normalized`, and
    /// Object Memory's **required** `subject_obscured`, so without the guard a
    /// single malformed number anywhere on those paths would crash the app.
    /// Wire data is not trusted input.
    ///
    /// `isFinite` first, then the clamp. A non-finite value is not rendered as
    /// a percentage at all, because there is no percentage it could honestly
    /// stand for — and saying so is better than picking 0% or 100%, either of
    /// which would be this app inventing a figure the Tower did not send.
    static func percent(_ value: Double) -> String {
        guard value.isFinite else { return "not reported" }
        let clamped = min(max(value, 0), 1)
        return "\(Int((clamped * 100).rounded()))%"
    }
}

// MARK: - Time

/// When something happened, keeping the three clocks the platform requires be
/// kept apart.
///
/// `docs/07-PLATFORM-CONSTRAINTS.md` Core Principle 5 and Limitation 9:
///
/// > **5. Timestamps Represent Observation Time.** Network arrival time,
/// > processing time, and observation/capture time must remain conceptually
/// > distinct.
///
/// The temptation this type removes is a single `Date` that a view labels
/// "seen at" while it actually holds the moment the phone decoded a message.
/// Every cartridge that shows a time needs this — Document Memory shows when a
/// document was observed, Scene Understanding when a track was last seen — so
/// it is shared.
///
/// **`observedAt` is not derived on the phone.** An iOS `Date()` at decode time
/// is arrival time wearing observation time's label, which is precisely the
/// conflation forbidden above. If the Tower does not report an observation
/// time, this carries `nil` and the UI says the time is unknown.
nonisolated struct ObservationTime: Equatable, Sendable {
    /// When the glasses observed it, as reported by the Tower. `nil` when the
    /// Tower did not say.
    var observedAt: Date?
    /// When this app received the report. Always knowable locally, and never a
    /// substitute for `observedAt`.
    var receivedAt: Date?

    init(observedAt: Date? = nil, receivedAt: Date? = nil) {
        self.observedAt = observedAt
        self.receivedAt = receivedAt
    }

    /// The only time that may be presented as when something *happened*.
    ///
    /// A computed accessor rather than a stored field so there is no path by
    /// which `receivedAt` can be silently promoted: a caller that wants to show
    /// arrival time has to name it.
    var displayableObservationTime: Date? { observedAt }

    /// True when the report carries no observation time at all — the case a
    /// view must render as "time unknown" rather than falling back to arrival.
    var isObservationTimeUnknown: Bool { observedAt == nil }
}

// MARK: - Duration in view

/// How long something was within the camera's field of view.
///
/// ## Why this is not called "viewing duration"
///
/// `docs/07-PLATFORM-CONSTRAINTS.md` Limitation 8 is unambiguous:
///
/// > Something appearing in the glasses camera does not prove the user looked
/// > directly at it, noticed it, read it, understood it, or interacted with it.
/// > […] Describe camera-derived events as "observed by the system," never
/// > "seen by the user".
///
/// The mitigation for that limitation is classified **REQUIRES FUTURE
/// HARDWARE/API**, and the document says the current mitigation "is purely
/// linguistic/labeling discipline, not a technical fix". Linguistic discipline
/// that lives only in a comment does not survive a refactor; a type named for
/// the claim it is allowed to make does.
///
/// So this measures the camera, and its `label` says so. There is no gaze
/// signal on the current hardware and nothing here may imply one.
nonisolated struct ObservedDuration: Equatable, Sendable {
    /// Seconds the subject was in frame, as reported by the Tower.
    let seconds: TimeInterval

    /// `isFinite` first: `max(0, …)` alone neutralises NaN by accident (the
    /// comparison is false, so the `0` wins) but passes `+∞` and `1e300`
    /// straight through — and `label` below does `Int(seconds.rounded())`,
    /// which **traps** on both. This runs on every document record, so one
    /// malformed `observed_seconds` from the Tower must not crash the app.
    ///
    /// A non-finite duration becomes 0 rather than being rejected: this type is
    /// a rendering detail on a row that has other true things to say, and
    /// dropping the whole record over an unusable number would hide more than
    /// it protects. `label` states the floor case honestly.
    init(seconds: TimeInterval) {
        self.seconds = seconds.isFinite ? max(0, seconds) : 0
    }

    /// Rendered so the claim is about the camera, not the wearer.
    var label: String {
        let whole = Int(seconds.rounded())
        if whole < 60 { return "In view \(whole)s" }
        let minutes = whole / 60
        let remainder = whole % 60
        return remainder == 0 ? "In view \(minutes)m" : "In view \(minutes)m \(remainder)s"
    }

    /// The footnote any surface showing this owes the reader.
    static let attentionCaveat =
        "Time in the camera's view. The glasses cannot tell whether you looked at it."
}
