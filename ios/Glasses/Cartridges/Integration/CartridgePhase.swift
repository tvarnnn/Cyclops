//
//  CartridgePhase.swift
//  Glasses
//

import Foundation

/// Where a cartridge's Tower-backed data currently stands, stated in the
/// smallest vocabulary that is true for every cartridge.
///
/// ## Why this exists at all
///
/// Each cartridge keeps its own state type, with its own domain-shaped cases —
/// `WorldModelState` says `receiving`/`finalized`, an experiment says
/// `running`/`completed`, a document query says `results`/`noMatch`. Those are
/// not the same thing and collapsing them into one generic enum would force
/// three different vocabularies through one shape and lose meaning at every
/// step.
///
/// What *is* genuinely shared is the coarse question a shell asks about any
/// cartridge: is there nothing here, is something in flight, is there data, did
/// it fail? That question has the same answers for every cartridge, and the app
/// needs it in exactly two places:
///
/// 1. shared presentation — one panel that renders "nothing yet" identically
///    everywhere, so a new cartridge cannot invent its own slightly different
///    way of saying the Tower has not answered;
/// 2. shared invariants — one test can assert, over every cartridge at once,
///    that a phase without data never carries data (see
///    `CartridgeIntegrationTests`). Without a common vocabulary that test would
///    have to be rewritten per cartridge and would rot.
///
/// So this carries **no payload**. It is a projection of each cartridge's own
/// state, never a replacement for it: a domain type exposes `var phase:
/// CartridgePhase` and keeps every one of its own cases. Nothing reads a phase
/// in order to decide what to *do* — only what to *say* when there is nothing
/// specific to show.
///
/// The `Value`-carrying generic that would let this replace the domain types
/// was deliberately not built. It would have to be `CartridgeDataState<Value>`
/// with cases covering every cartridge's lifecycle, which is how a small shared
/// vocabulary becomes a plugin framework.
enum CartridgePhase: String, Equatable, Sendable, CaseIterable {
    /// This Tower offers this app nothing it can use for the cartridge. Two
    /// causes reach it:
    ///
    /// - **no contract**: the Tower declared none for this cartridge. The
    ///   honest answer for the three cartridges with no Tower code anywhere
    ///   (Visual Q&A, Accessibility, Environmental Memory), and for any Tower
    ///   that genuinely has not heard of a cartridge;
    /// - **declared but not served**: the Tower knows the cartridge and says it
    ///   cannot serve it — an offer marked `available: false`, a status
    ///   document whose lifecycle is `unavailable`, a refusal or error saying
    ///   the cartridge is unavailable, or a route that answers that nothing is
    ///   configured. The Tower's own reason, where it gave one, is what the
    ///   screen shows.
    ///
    /// **Object Memory never reaches it through a declaration**: it is
    /// undeclared on the socket by the Tower's own design, and lands here only
    /// when its HTTP routes answer that no object memory is configured.
    case unsupported
    /// The capability exists but the Tower cannot be reached right now.
    ///
    /// Separate from `unsupported` because the two call for opposite responses
    /// — one waits for a Tower that may never do this, the other waits for a
    /// connection — and a shell that draws them identically tells a user to
    /// reconnect when reconnecting cannot help, or the reverse. The headline
    /// and the glyph are most of what a person actually reads, so the
    /// explanation string cannot carry the distinction alone.
    case disconnected
    /// The Tower does this, under an agreement this build cannot read.
    ///
    /// Separate from `unsupported` for exactly the reason `disconnected` is
    /// separate from it, one level up: the two call for **opposite** responses.
    /// `unsupported` is "this Tower may never do this", and there is nothing a
    /// person can do about it. This one is "this Tower already does this and
    /// the app is behind", which a person can act on immediately — and
    /// rendering it as "Nothing yet" tells someone a feature does not exist
    /// when in fact they are one update away from it.
    case needsUpdate
    /// The capability exists but nothing has been asked of it yet.
    case idle
    /// Something is genuinely in flight and the Tower has not answered.
    /// Distinct from `idle` because it is the only phase in which a progress
    /// indicator is honest.
    case waiting
    /// The Tower is actively producing or refining data, and it may still
    /// change. Covers "receiving world updates", "experiment running", and
    /// "finalising a world" alike: work is underway.
    case live
    /// There is data and the Tower is not changing it any more. A finished
    /// world, a completed experiment, a returned query result.
    case settled
    /// The attempt failed and the reason is known.
    case failed

    /// Whether a phase is permitted to carry data at all.
    ///
    /// The load-bearing half of this type. Every cartridge's state must satisfy
    /// `!phase.mayCarryData` ⇒ no payload, which is what makes "no fabricated
    /// placeholder data" a checked property rather than a convention.
    var mayCarryData: Bool {
        switch self {
        case .live, .settled: return true
        case .unsupported, .disconnected, .needsUpdate, .idle, .waiting, .failed: return false
        }
    }

    /// Whether a progress indicator is truthful. A spinner in any other phase
    /// claims work is underway when none is.
    var showsProgress: Bool { self == .waiting }
}
