//
//  VisualArtifact.swift
//  Glasses
//

import Foundation

/// How a piece of imagery has been treated before this app was allowed to show
/// it.
///
/// ## The pipeline this sits at the end of
///
/// ```text
/// raw sensor data → ephemeral perception → derived structured state
///                 → redaction → persistence / display
/// ```
///
/// `docs/06-PRIVACY-DATA.md` is explicit that the reduction step is not itself a
/// safety guarantee:
///
/// > Selected crops or "reduced" imagery are not inherently safe: a cropped
/// > image can still contain a bystander's face, a private room, or a document.
/// > Treat any stored image/crop as potentially sensitive, regardless of size or
/// > selection method.
///
/// So this type does **not** describe how small an image is or where it came
/// from. It describes one thing: whether a redaction step was actually applied
/// by whoever produced it. That is a fact only the producer knows, which is why
/// there is no `.probablySafe` and no default — an artifact whose treatment was
/// not stated is `.unknown`, and `.unknown` is displayed with the same reserve
/// as raw.
///
/// ## What this cannot do
///
/// **It does not redact anything.** iOS applies no redaction and must not
/// pretend to: doing so here would mean the raw pixels had already reached the
/// phone, at which point the control is theatre. `docs/06-PRIVACY-DATA.md`
/// requires redaction to happen where the data is derived, and this app's job
/// is to refuse to display what has not been treated and to say which is which.
///
/// It is therefore **not a user-facing privacy control**. There is no toggle
/// backed by this. A switch the Tower cannot honour is worse than no switch,
/// because it converts a limitation into a false assurance.
enum RedactionState: String, Equatable, Sendable, CaseIterable {
    /// A redaction step was applied by the producer before this artifact was
    /// persisted or offered for display.
    case redacted
    /// Untreated imagery. Permitted only for the live, in-memory view of what
    /// the wearer currently sees — never for anything persisted, and never for
    /// anything a cartridge stored and re-served later.
    case rawEphemeral
    /// The producer did not state how the artifact was treated. Handled exactly
    /// as strictly as `rawEphemeral`: an unstated treatment is not a treatment.
    case unknown

    /// Whether an artifact may be shown on a surface that persists or
    /// re-displays it — a document thumbnail, a saved annotation, a memory row.
    ///
    /// The single decision this type exists to make, so no view re-derives it.
    var isDisplayableWhenPersisted: Bool { self == .redacted }

    /// Whether an artifact may be shown on a surface that draws it **live and
    /// keeps nothing** — a viewfinder, a running experiment's preview panel.
    ///
    /// Two of the three cases qualify, and that is not a relaxation: it is
    /// what `rawEphemeral` was defined for. Its own documentation says
    /// *"Permitted only for the live, in-memory view of what the wearer
    /// currently sees"*, and until there was such a view nothing had cause to
    /// ask. `unknown` does not qualify, here as everywhere, because an
    /// unstated treatment is not a treatment.
    ///
    /// The obligation this carries is on the CALLER and cannot be checked
    /// here: a surface that asks this question must not persist, cache or
    /// re-serve what it draws. `ExperimentalCVPreview.swift` is the one that
    /// does, and its file comment lists the four things it does to keep the
    /// promise structural rather than intentional.
    var isDisplayableLive: Bool { self != .unknown }

    /// What a person needs to know about how this image was treated.
    ///
    /// ## Why `redacted` describes the claim rather than the result
    ///
    /// It would read better as "people in this image were obscured". It would
    /// also be an invention: no Tower contract defines what redaction *does*.
    /// It could be face masking, text masking, a box drawn over nothing, or the
    /// producer's own definition of the word. Telling a person specifically
    /// what was removed turns an opaque flag into a checkable privacy
    /// guarantee — which is the "switch the Tower cannot honour" this type's
    /// own doc comment says is worse than no switch.
    ///
    /// So the sentence says who claimed what, and stops there.
    var explanation: String {
        switch self {
        case .redacted:
            return "The producer states this image was redacted before it was stored. What that removed is the producer's definition."
        case .rawEphemeral:
            // "It is not stored" would be a guarantee this enum cannot enforce
            // about the Tower. What this app knows is what *this app* does.
            return "A live view. This app does not store it."
        case .unknown:
            return "The Tower did not say whether this image was redacted, so it is not shown."
        }
    }
}

/// A piece of imagery a cartridge would like to show, and whether it may.
///
/// ## Why there is no image, URL, or identifier here
///
/// Because the Tower has not defined how artifacts are fetched, and inventing a
/// URL scheme or an artifact id format would be exactly the fabricated contract
/// this work is forbidden to produce. What *is* knowable without that contract
/// is the state machine around a fetch — is there one, is it in flight, did it
/// arrive, may it be shown — and that is what this models.
///
/// When a real fetch contract lands, `available` gains whatever payload the
/// Tower actually serves and every call site's handling of the other cases is
/// already written.
enum VisualArtifactState: Equatable, Sendable {
    /// The Tower reported no artifact for this item. Not an error: many
    /// observations legitimately have no image.
    case absent
    /// An artifact exists and has not been fetched. Distinct from `.fetching`
    /// so a list can show a placeholder without a spinner on every row.
    case notFetched(RedactionState)
    /// A fetch is genuinely in flight — the one state in which a progress
    /// indicator over a thumbnail is truthful.
    case fetching(RedactionState)
    /// The artifact is here. Carries its redaction state, so the display
    /// decision travels with it and cannot be lost between fetch and render.
    case available(RedactionState)
    /// The fetch failed.
    case failed(CartridgeFailure)

    /// The treatment the artifact carries, in every state where one is known.
    var redaction: RedactionState? {
        switch self {
        case .notFetched(let redaction), .fetching(let redaction), .available(let redaction):
            return redaction
        case .absent, .failed:
            return nil
        }
    }

    /// Whether a persisted surface may draw this artifact's pixels.
    ///
    /// Two conditions, both required: it has to be here, and it has to have
    /// been redacted. An artifact that arrived untreated is held and not shown —
    /// the strictness is the point, and it is what makes "redacted by default
    /// where practical" a property rather than an aspiration.
    var isDisplayable: Bool {
        guard case .available(let redaction) = self else { return false }
        return redaction.isDisplayableWhenPersisted
    }

    /// Why nothing is drawn, when nothing is drawn. `nil` when the artifact is
    /// displayable and there is nothing to explain.
    var withheldReason: String? {
        switch self {
        case .absent:
            return nil
        case .notFetched(let redaction), .fetching(let redaction):
            return redaction.isDisplayableWhenPersisted ? nil : redaction.explanation
        case .available(let redaction):
            return redaction.isDisplayableWhenPersisted ? nil : redaction.explanation
        case .failed(let failure):
            return failure.message
        }
    }
}
