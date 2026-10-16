//
//  CartridgeClient.swift
//  Glasses
//

import Foundation

/// What the Tower has told this app it can do, per cartridge.
///
/// The Tower declares its capabilities over the socket that is already open:
/// `{"type":"cartridges"}` after the pong, answered with the offers in
/// `TowerCartridgeDeclaration`. That is a *reply to a question about
/// capability*, not the dynamic module list `docs/04-MODULE-SYSTEM.md` forbids
/// before V1.0 and not the module-selection message
/// `docs/08-IOS-CARTRIDGE-SHELL.md` forbids — nothing here selects, starts, or
/// configures anything on the Tower, and the drawer is still the static
/// `Cartridge.catalog`.
///
/// Two sources answer:
///
/// - `declared`, a local table. Empty, and kept empty by rule (see there).
/// - `availability(for:declaredBy:isTowerReachable:)`, resolved against the
///   live declaration. The declaration is cached on `TowerClient`, which is the
///   object that owns the socket it arrived on; caching it in a mutable static
///   here would put connection state in a namespace nothing can scope to a
///   connection.
enum TowerCapabilities {

    /// Contracts declared by a local table rather than by the Tower.
    ///
    /// **Empty by rule:** a contract this app can reach is declared by the
    /// Tower, never hardcoded here. Duplicating one as a compile-time constant
    /// creates a second answer that can disagree with the Tower's, and the
    /// disagreement would be silent.
    ///
    /// **Object Memory is the deliberate exception, and it is not here either.**
    /// It is undeclared on purpose (the Tower's own §9), reached entirely over
    /// HTTP, and a `result_subscribe` for it is refused `unknown_cartridge`.
    /// Adding it to this table would be inventing a declaration the Tower
    /// pointedly declined to make.
    static let declared: [String: CartridgeContract] = [:]

    /// Contract identifiers this build implements.
    ///
    /// Kept separate from `declared` because they answer different questions
    /// about different machines, and because their disagreement is exactly what
    /// `CartridgeAvailability.unsupportedContract` exists to represent. A Tower
    /// offering `world_builder.status/2026-10-…` would land there rather than
    /// being decoded on a guess.
    ///
    /// A test pins this set, so widening it is a reviewed change, never a
    /// silent one.
    ///
    /// `document_memory` contributes **two** identifiers because the Tower
    /// declares two, deliberately: `status` is small and pushed on the socket,
    /// `library` is bulk text pulled over HTTP. They govern different
    /// transports with different failure modes, and a change to one is not a
    /// change to the other — collapsing them here would lose exactly that.
    ///
    /// **`world_builder.geometry/2026-08-25` is deliberately absent.** It is
    /// HTTP-only and compared for equality inside `WorldGeometryDecoder`, and
    /// this set is consulted against *subscription* offers. Adding it would
    /// make a fetch-only contract look subscribable.
    static let supported: Set<String> = [
        WorldBuilderResultContract.identifier,
        ExperimentalCVContract.status,
        SceneUnderstandingContract.identifier,
        DocumentMemoryContract.statusIdentifier,
        DocumentMemoryContract.libraryIdentifier
    ]

    /// This app's catalog id → the Tower's name for the same cartridge.
    ///
    /// The two vocabularies are genuinely different (`"world-build"` against
    /// `"world_builder"`, `"experimental-cv"` against `"experimental_cv"`), and
    /// the mapping is here rather than in the client so that a second cartridge
    /// cannot invent a different convention for it.
    ///
    /// **This dictionary is load-bearing in a way that is easy to miss.**
    /// `declaredContract(for:in:)` returns `nil` for any cartridge absent from
    /// it, so availability resolves `.noContract` and the phase is forced to
    /// `.unsupported` — *no matter what the Tower declared*. A cartridge with a
    /// complete client, a correct decoder and a live offer still shows a person
    /// "nothing here" until its name appears on this list.
    ///
    /// **Object Memory is absent deliberately**, and that is not an oversight
    /// to be tidied up. The Tower does not declare it, instructs clients to
    /// "learn nothing about it from the declaration", and refuses a
    /// subscription for it. It is reached over HTTP by its own client.
    static let towerCartridgeNames: [String: String] = [
        "world-build": WorldBuilderResultContract.towerCartridge,
        ExperimentalCVContract.catalogID: ExperimentalCVContract.towerCartridge,
        "scene-understanding": SceneUnderstandingContract.towerCartridge,
        "document-memory": DocumentMemoryContract.towerCartridge
    ]

    static func declaredContract(for cartridgeID: String) -> CartridgeContract? {
        declared[cartridgeID]
    }

    /// The contract the Tower declared over the socket for this cartridge, or
    /// `nil` if it declared none.
    ///
    /// Returned **whether or not the offer is currently `available`**. The two
    /// facts are separate: "the Tower speaks this contract" is what availability
    /// resolves, and "the Tower cannot serve it right now" is a reason the
    /// cartridge's own state carries, in the Tower's own words. Collapsing an
    /// unavailable offer to `nil` would render "no world root is configured" as
    /// "this Tower will never do this", which is a different and wrong claim.
    static func declaredContract(
        for cartridgeID: String,
        in declaration: TowerCartridgeDeclaration?
    ) -> CartridgeContract? {
        guard
            let declaration,
            let towerName = towerCartridgeNames[cartridgeID]
        else { return nil }
        if let offer = declaration.offer(forTowerCartridge: towerName) {
            return CartridgeContract(cartridgeID: cartridgeID, identifier: offer.contract)
        }
        // A capability the Tower serves over HTTP rather than by subscription.
        //
        // Checked **second, never first**: a cartridge declaring both (Document
        // Memory declares `status` on the socket and `library` over HTTP) must
        // resolve to its subscription, because that is the surface this app's
        // result channel binds to. Preferring the HTTP entry would hand the
        // subscribing client a library identifier it cannot subscribe with.
        //
        // Reached at all because a fetch-only cartridge is genuinely available
        // — a Tower serving `/documents` can answer questions about documents.
        // Resolving that to `.noContract` would tell a person the cartridge
        // does not exist while its route is answering.
        if let http = declaration.httpContract(forTowerCartridge: towerName) {
            return CartridgeContract(cartridgeID: cartridgeID, identifier: http.contract)
        }
        return nil
    }

    /// Availability for one cartridge from the local table alone, with no live
    /// declaration.
    ///
    /// What the `CartridgeClient` default answers with, so the precedence rules
    /// in `CartridgeAvailability.resolve` apply uniformly and no cartridge can
    /// quietly decide it is available on different grounds.
    static func availability(for cartridgeID: String, isTowerReachable: Bool) -> CartridgeAvailability {
        availability(for: cartridgeID, declaredBy: nil, isTowerReachable: isTowerReachable)
    }

    /// The same decision, given whatever the Tower has actually declared.
    ///
    /// A live declaration wins over the local table when it names this
    /// cartridge; otherwise the table answers, so a Tower that has declared
    /// nothing yet is indistinguishable from one that never will — which is
    /// correct, because from here it is.
    static func availability(
        for cartridgeID: String,
        declaredBy declaration: TowerCartridgeDeclaration?,
        isTowerReachable: Bool
    ) -> CartridgeAvailability {
        CartridgeAvailability.resolve(
            declared: declaredContract(for: cartridgeID, in: declaration)
                ?? declaredContract(for: cartridgeID),
            supported: supported,
            isTowerReachable: isTowerReachable,
            // A cartridge is "known to this build" exactly when it has a Tower
            // name here — that map is what makes a declaration for it legible
            // at all. Object Memory is deliberately absent and reaches its own
            // availability over HTTP, which is why it is not special-cased.
            knownToThisBuild: towerCartridgeNames[cartridgeID] != nil
        )
    }
}

/// The seam every cartridge's Tower client sits behind.
///
/// ## What it is for, and what it deliberately is not
///
/// This is the *shared* half of a cartridge client: which cartridge it serves,
/// and whether it can serve it. Everything else — what it fetches, what it
/// returns, what its lifecycle looks like — belongs to the cartridge's own
/// protocol, because those four things have nothing in common. World Builder
/// pushes a continuous stream; Document Memory answers point queries;
/// Experimental CV Lab runs a bounded job; Scene Understanding publishes a
/// changing set of tracks. Forcing one `fetch<T>()` over all four would be a
/// plugin framework wearing a protocol's clothes, and
/// docs/02-DEVELOPMENT-RULES.md Rule 10 exists to stop exactly that.
///
/// So: no `associatedtype`, no generic request/response, no transport. A
/// cartridge client conforms to this *and* to its own protocol, and the second
/// one is where the real work is described.
///
/// `@MainActor` and `AnyObject` to match `TowerClient` and `GlassesConnection`:
/// a Tower-backed implementation holds connection state and publishes into
/// SwiftUI on the main actor, as those do.
@MainActor
protocol CartridgeClient: AnyObject {
    /// The catalog id from `Cartridge.catalog`. Used to look the cartridge's
    /// contract up, and asserted in tests so a client cannot answer for a
    /// cartridge that is not its own.
    var cartridgeID: String { get }

    /// Whether this client can currently reach a Tower capability, and if not,
    /// why. Recomputed rather than stored, so it cannot go stale against a
    /// connection that changed underneath it.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability
}

extension CartridgeClient {
    /// The default, used by the unavailable stubs, and the reason
    /// `TowerCapabilities` is a lookup: a client does not get to decide it is
    /// available. The Tower-backed clients override it with what the Tower
    /// told them: the live declaration, or, for Object Memory, what its HTTP
    /// routes answered.
    func availability(isTowerReachable: Bool) -> CartridgeAvailability {
        TowerCapabilities.availability(for: cartridgeID, isTowerReachable: isTowerReachable)
    }
}
