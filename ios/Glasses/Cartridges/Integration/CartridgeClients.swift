//
//  CartridgeClients.swift
//  Glasses
//

import Foundation

/// The cartridge clients, owned above the workspace that uses them.
///
/// ## Why these do not live in the workspace views
///
/// A workspace's `@StateObject` view model is destroyed when the cartridge
/// changes. That is fine for a view model — it holds nothing real. It is
/// **not** fine for a client: a Tower-backed one holds a subscription to the
/// connection and whatever it has accumulated — a partly-built world, a live
/// set of scene tracks, a query result. Destroying that on a cartridge switch
/// loses work and, worse, tears down a subscription the user did not ask to
/// end. So anything that must *outlive* the workspace belongs on
/// `ProjectManager`, not in a view's `@StateObject`.
///
/// Injected rather than defaulted at the point of use: a default argument on
/// each view model making `UnavailableFooClient()` constructible there would
/// make the wrong wiring the path of least resistance.
///
/// ## What this is not
///
/// It is not a registry, not a plugin host, and not a lookup by cartridge id.
/// One `let` per cartridge, resolved at compile time. Adding a
/// `client(for: CartridgeWorkspace)` accessor would be the first step toward
/// the dynamic module discovery `docs/04-MODULE-SYSTEM.md` forbids before V1.0.
///
/// ## Runtime ownership
///
/// A container, not a manager. It starts nothing, connects to nothing, and owns
/// no transport. `ProjectManager`, which owns the connection, constructs the
/// socket-backed clients and passes them in.
@MainActor
final class CartridgeClients {
    let worldBuilder: any WorldBuilderClient
    let experimentalCV: any ExperimentalCVClient
    let documentMemory: any DocumentMemoryClient
    let sceneUnderstanding: any SceneUnderstandingClient
    /// The one client whose default is **not** an unavailable stub.
    ///
    /// Object Memory's Tower half exists and answers over HTTP, and its client
    /// discovers that by asking rather than by being told over the socket — so
    /// there is no declaration to wait for and nothing for a stub to stand in
    /// for. Constructed here rather than in the workspace because an answer
    /// must survive a cartridge switch, which destroys the view's
    /// `@StateObject`.
    ///
    /// It opens nothing on construction. No request is made until a person
    /// asks a question.
    let objectMemory: any ObjectMemoryClient

    /// Defaults are the unavailable stubs, Object Memory's excepted. They are a
    /// test convenience, not a description of the product: `ProjectManager`
    /// passes a Tower-backed client for every socket cartridge.
    ///
    /// The stubs stay for two reasons: a test needs to assert what a workspace
    /// does when a Tower declares nothing, and a stub is the only honest answer
    /// before a connection exists. They are kept deliberately dull: "this Tower
    /// has declared no contract" cannot go stale, where a claim about what the
    /// Tower does would become false the day the Tower changed.
    ///
    /// Injection points exist so a test can substitute one without reaching
    /// through `ProjectManager`.
    init(
        worldBuilder: (any WorldBuilderClient)? = nil,
        experimentalCV: (any ExperimentalCVClient)? = nil,
        documentMemory: (any DocumentMemoryClient)? = nil,
        sceneUnderstanding: (any SceneUnderstandingClient)? = nil,
        objectMemory: (any ObjectMemoryClient)? = nil
    ) {
        self.worldBuilder = worldBuilder ?? UnavailableWorldBuilderClient()
        self.experimentalCV = experimentalCV ?? UnavailableExperimentalCVClient()
        self.documentMemory = documentMemory ?? UnavailableDocumentMemoryClient()
        self.sceneUnderstanding = sceneUnderstanding ?? UnavailableSceneUnderstandingClient()
        self.objectMemory = objectMemory ?? TowerObjectMemoryClient()
    }
}
