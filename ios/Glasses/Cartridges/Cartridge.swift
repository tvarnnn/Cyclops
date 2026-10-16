//
//  Cartridge.swift
//  Glasses
//

import Foundation

/// What a person can do with a cartridge in this build.
///
/// A badge in the drawer answers *"is there anything here for me right now?"*,
/// not *"where does this module sit on the Tower roadmap?"*. The two questions
/// have different answers: a roadmap word can sound readier than a cartridge
/// with no Tower code at all. Roadmap position lives in `docs/03-ROADMAP.md`
/// and each cartridge's spec, and is never rendered to a wearer as though it
/// were a capability.
///
/// ## What these values are NOT
///
/// **This is static catalog data about *this build*, not a live claim about
/// the Tower you happen to be connected to.** Whether a given Tower can serve
/// a cartridge right now is resolved per-connection by
/// `CartridgeAvailability.resolve` and rendered inside each workspace. A badge
/// reading "Ready to test" beside a workspace saying "Nothing yet" is not a
/// contradiction: the app is ready and that Tower is not. Keeping those two
/// separate is the reason this type consults nothing at runtime.
nonisolated enum CartridgeStatus: Equatable, CaseIterable {
    /// This build implements the cartridge and can show real Tower data for it.
    ///
    /// Not a promise that any particular Tower is serving it this second —
    /// see the type doc. It is a promise that there is something here to try.
    case readyToTest
    /// The Tower implements it; no contract is offered on the wire yet.
    ///
    /// The workspace opens and refuses truthfully. This is deliberately
    /// distinct from `notBuilt`: the two call for opposite responses — one is
    /// waiting on a Tower decision that is already costed, the other is waiting
    /// on a backend nobody has written.
    case awaitingTower
    /// No backend exists anywhere. A specification only.
    case notBuilt

    var badge: String {
        switch self {
        case .readyToTest: return "Ready to test"
        case .awaitingTower: return "Awaiting Tower"
        case .notBuilt: return "Not built"
        }
    }

    /// Whether the badge should draw the eye.
    ///
    /// Exactly one status earns it. The drawer's job in one glance is to answer
    /// "which of these can I try?", and eight identically-styled capsules make
    /// a reader parse all eight to find out. Tinting more than one would put
    /// the question back.
    var isProminent: Bool { self == .readyToTest }
}

/// A module the platform intends to host.
///
/// Static catalog data. `status` describes what a person can do with the
/// cartridge in this build and carries no runtime behaviour; `workspace`
/// describes whether *this app* has a screen for it. The two are not
/// independent: a cartridge with nothing to open cannot be ready to test, and
/// `CartridgeCatalogTests` pins that correspondence in both directions.
struct Cartridge: Identifiable, Equatable {
    let id: String
    let name: String
    let summary: String
    let status: CartridgeStatus
    /// The module's spec. Not shown in the drawer; the catalog tests pin every
    /// path to a known spec file.
    let specPath: String
    /// The workspace this app can open for the cartridge, or `nil` if the app
    /// has no screen for it, in which case its drawer row is informational.
    var workspace: CartridgeWorkspace? = nil
}

extension Cartridge {
    /// Names and ordering follow the module specs in docs/modules/.
    ///
    /// **`status` comes from evidence, not from the specs:** what the Tower
    /// serves and what this build implements. A spec's roadmap position never
    /// sets a badge.
    static let catalog: [Cartridge] = [
        // `.readyToTest`: the Tower runs this module, and its experiment's
        // answer arrives in `frame_result` on every frame and is rendered in
        // this cartridge's workspace. What a person can and cannot do with it
        // is said on that screen, not in a capsule.
        Cartridge(
            id: "experimental-cv",
            name: "Experimental CV Lab",
            summary: "A sandbox for running and comparing vision models on the glasses feed.",
            status: .readyToTest,
            specPath: "docs/modules/EXPERIMENTAL-CV.md",
            workspace: .experimentalCV
        ),
        // The summary claims no location: `spatial_ref` is null in every
        // payload the contract can produce and is nulled on read, so nothing
        // here knows where anything is. The drawer row is the first sentence a
        // person reads about this module.
        //
        // `.readyToTest`: two live HTTP routes, a decoder pinned against a real
        // Tower's bytes, and a workspace that queries them. A Tower with no
        // object memory root answers 404 (`no object memory root is
        // configured`) and the workspace renders that; it is a fact about one
        // Tower's configuration, not this build, so the badge does not track
        // it (see `CartridgeStatus`).
        Cartridge(
            id: "object-memory",
            name: "Object Memory",
            summary: "Records that a category of thing was visible to the camera, and when.",
            status: .readyToTest,
            specPath: "docs/modules/OBJECT-MEMORY.md",
            workspace: .objectMemory
        ),
        Cartridge(
            id: "visual-qa",
            name: "Visual Q&A",
            summary: "Answers questions about what the camera can see, including reading text.",
            status: .notBuilt,
            specPath: "docs/modules/VISUAL-QA.md"
        ),
        // `.readyToTest`: the Tower offers its contract and this build has
        // walked it on a device.
        //
        // Named "World Builder" here, while the spec file it cites is titled
        // "World Build". Same module; the product surface and the Tower work
        // both use the longer name, and two names for one thing on screen would
        // be worse than the small mismatch with the document title.
        Cartridge(
            id: "world-build",
            name: "World Builder",
            summary: "Reconstructs a persistent spatial model of a space over time.",
            status: .readyToTest,
            specPath: "docs/modules/WORLD-BUILD.md",
            workspace: .worldBuilder
        ),
        Cartridge(
            id: "accessibility",
            name: "Accessibility",
            summary: "Experimental assistive descriptions. Not a safety or navigation device.",
            status: .notBuilt,
            specPath: "docs/modules/ACCESSIBILITY.md"
        ),
        Cartridge(
            id: "environmental-memory",
            name: "Environmental Memory",
            summary: "Search the physical world you have already walked through.",
            status: .notBuilt,
            specPath: "docs/modules/ENVIRONMENTAL-MEMORY.md"
        ),
        // The two entries below are narrower first versions of modules already
        // in this catalog, not new ambitions. Environmental Memory's own spec
        // asks a first version to "choose one constrained memory type, such as
        // […] searchable OCR history", and Object Memory's pipeline produces
        // the live tracks Scene Understanding reads. Each cites its own concept
        // seed, which says what it takes from its parent and what it leaves
        // behind.
        //
        // Both are `.readyToTest` on the same terms as the others: the Tower
        // declares their contracts and this build decodes them. Document
        // Memory declares `document_memory.status` on the socket and
        // `document_memory.library` under `http_contracts`. Scene
        // Understanding declares a `live` result rather than a `status` one:
        // the payload is the answer itself, so a cartridge that persists
        // nothing can still say what it can see right now. Neither badge
        // promises that a given Tower is configured for it; that is resolved
        // per connection by `CartridgeAvailability.resolve`.
        //
        // `.notBuilt` stays distinct for the three entries above with no Tower
        // code anywhere: those are a backend away, not a decision away.
        Cartridge(
            id: "document-memory",
            name: "Document Memory",
            summary: "Documents you passed, kept as text and summaries rather than as pictures.",
            status: .readyToTest,
            specPath: "docs/modules/DOCUMENT-MEMORY.md",
            workspace: .documentMemory
        ),
        Cartridge(
            id: "scene-understanding",
            name: "Scene Understanding",
            summary: "An anonymous read of how many people and objects the camera can see, and roughly where.",
            status: .readyToTest,
            specPath: "docs/modules/SCENE-UNDERSTANDING.md",
            workspace: .sceneUnderstanding
        ),
    ]
}
