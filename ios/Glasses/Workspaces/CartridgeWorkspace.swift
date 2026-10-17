//
//  CartridgeWorkspace.swift
//  Glasses
//

import Foundation

/// A primary workspace this app can present for a cartridge.
///
/// ## Why this is separate from `CartridgeStatus`
///
/// The two answer different questions, about different machines, and conflating
/// them would produce a lie in one direction or the other:
///
/// - `CartridgeStatus` describes a **module's position on the Tower roadmap**
///   — `next`, `planned`, `future`. It is deliberately missing an `available`
///   case, because no module runtime exists on the Tower at all: the module
///   container is V0.8 and the first module V0.9 (ios/Glasses/Project_Overview_Steps/docs/03-ROADMAP.md), and
///   `ProductShellTests` fails loudly if that ever silently changes.
///
/// - `CartridgeWorkspace` describes whether **this app has a screen** for the
///   cartridge. That is a fact about the iPhone, and it is true today for World
///   Builder whatever the Tower can or cannot do.
///
/// Keeping them apart is what lets the drawer open a World Builder workspace
/// without any of it implying the Tower is running World Builder. Opening a
/// workspace is local navigation. It sends nothing, selects nothing on the
/// Tower, and changes no Tower state — ios/Glasses/Project_Overview_Steps/docs/08-IOS-CARTRIDGE-SHELL.md forbids
/// inventing a module-selection message, and none is invented.
///
/// A cartridge with no workspace stays exactly as it was: an informational row.
///
/// ## Five cases, and still an enum
///
/// Product Shell V2 had one case and argued that a `switch` beat a workspace
/// registry. Somewhere around four or five is the number at which that argument
/// is usually abandoned in favour of a protocol and a lookup table, so it is
/// worth restating why it holds better now than it did then, not worse:
///
/// - the set is still **closed and compiled in**. A workspace is a SwiftUI view
///   in this binary; there is no dynamic module list to render
///   (`ios/Glasses/Project_Overview_Steps/docs/04-MODULE-SYSTEM.md` forbids building one before V1.0), so nothing
///   can appear here that was not compiled;
/// - exhaustiveness is the feature. Adding a case makes the compiler demand the
///   arm in `ContentView`, where a registry would silently fall back to Home;
/// - the anti-pattern the roadmap warns about is per-cartridge *conditionals
///   scattered through one enormous view*. There is exactly one `switch`, in
///   one place, whose arms are one-line constructor calls into separate files.
///
/// What did change is that the workspaces now share a *client* layer —
/// `CartridgeClient`, `CartridgeAvailability`, `CartridgePhase` — because that
/// is where the genuine commonality turned out to be. The screens have almost
/// nothing in common; the question "may this be used, and why not" is identical
/// for all four.
enum CartridgeWorkspace: String, Equatable, Sendable, CaseIterable {
    case worldBuilder
    case experimentalCV
    case documentMemory
    case sceneUnderstanding
    /// The fifth, and the first whose Tower half actually answers.
    ///
    /// Its cartridge `status` used to stay `.planned` on the argument that two
    /// read-only HTTP routes do not move a module's roadmap position. That was
    /// right about the roadmap and wrong about what the badge is for: it now
    /// answers what a person can do in this build, and two live routes with a
    /// decoder pinned against a real Tower's bytes is something they can try.
    /// `CartridgeCatalogTests` pins every entry's status against the evidence
    /// recorded beside it.
    case objectMemory
}

/// One row of the cartridge drawer, and the **only** place the question "may
/// this cartridge be opened?" is answered.
///
/// ## Why this type exists
///
/// It used to be answered in four places. `Cartridge.selectable` filtered the
/// catalog on `workspace != nil`; `workspaceCartridge(forID:)` re-checked the
/// same thing to reject a stale stored id; and the drawer — which did not
/// consult either — re-derived it twice more, once to decide whether to wrap a
/// row in a `Button` and once inside the row to choose an accessibility hint.
///
/// Four answers to one question, agreeing only because four people happened to
/// write the same expression. Nothing made them agree, so nothing would have
/// caught them diverging: a cartridge could have been tappable in the drawer
/// and absent from `selectable`, and the tests — which only ever asked
/// `selectable` — would have kept passing while describing a drawer that no
/// longer existed. `selectable`'s doc comment already claimed to be "every
/// cartridge the drawer may present as openable"; this type is what makes that
/// sentence true by construction rather than by coincidence.
///
/// ## Why an enum, and not a `Bool`
///
/// The same argument `CartridgeWorkspace` makes above, one level down. A row
/// carrying `isOpenable: Bool` alongside an optional workspace can still hold
/// the contradiction — `true` with nothing to open — and the drawer would have
/// to force-unwrap or silently do nothing. Here the openable case *carries* the
/// non-optional `CartridgeWorkspace`, so the impossible state cannot be
/// written down, and the `switch` in the drawer gets the workspace as a fact
/// rather than as a second lookup.
///
/// This is not a registry. Nothing is keyed by id and nothing is looked up
/// dynamically; the rows are a `map` over the compiled-in catalog in catalog
/// order, which is exactly what `ios/Glasses/Project_Overview_Steps/docs/04-MODULE-SYSTEM.md` permits before V1.0.
enum CartridgeDrawerRow: Identifiable, Equatable {
    /// The app ships a screen for this cartridge, and this is the screen.
    case openable(Cartridge, CartridgeWorkspace)
    /// The app has no screen for this cartridge, so its row is informational:
    /// no button, no tap target, exactly as every row was before workspaces
    /// existed. This is not a lesser state — three modules are genuinely in it.
    case informational(Cartridge)

    var cartridge: Cartridge {
        switch self {
        case .openable(let cartridge, _): return cartridge
        case .informational(let cartridge): return cartridge
        }
    }

    var id: String { cartridge.id }

    /// The single openability decision, for callers that need the answer
    /// without needing the workspace.
    var isOpenable: Bool {
        if case .openable = self { return true }
        return false
    }

    /// The hint VoiceOver reads, derived from the same decision that decides
    /// whether there is a `Button` to hear it on.
    ///
    /// It lives here rather than in the view because these two sentences are
    /// the only statement of openability a person who cannot see the screen
    /// receives. Derived separately, they would eventually promise a workspace
    /// that does not open — a lie told exclusively to the users least able to
    /// tell that nothing happened (Rule 3, Truthful State Only).
    var accessibilityHint: String {
        isOpenable ? "Opens this workspace" : "No workspace in this app yet"
    }
}

extension Cartridge {
    /// Every catalog entry, in catalog order, each paired with the one decision
    /// about whether it can be opened.
    ///
    /// All eight rows, deliberately: the drawer shows the modules the platform
    /// intends to host, and hiding the three with no screen would turn the
    /// roadmap into a feature list. What differs between them is whether the
    /// row does anything, which is the case of the row and nothing else.
    static var drawerRows: [CartridgeDrawerRow] {
        catalog.map { cartridge in
            guard let workspace = cartridge.workspace else {
                return .informational(cartridge)
            }
            return .openable(cartridge, workspace)
        }
    }

    /// Every cartridge the drawer may present as openable.
    ///
    /// Defined *as* the openable rows of the drawer, not as a parallel filter
    /// over the catalog that happens to select the same ones. A future
    /// cartridge cannot be openable here and informational there, because
    /// there is only one `here`.
    static var selectable: [Cartridge] {
        drawerRows.compactMap { row in
            switch row {
            case .openable(let cartridge, _): return cartridge
            case .informational: return nil
            }
        }
    }

    /// The cartridge whose workspace is currently open, resolved from a stored
    /// identifier.
    ///
    /// Returns `nil` for an unknown id **and** for a cartridge that has no
    /// workspace, so a persisted selection can never resurrect a screen that no
    /// longer exists or was never navigable. That matters across app updates:
    /// a stored id outlives the build that wrote it.
    ///
    /// Searches `selectable` rather than re-testing `workspace` on a catalog
    /// hit: "was never navigable" is the drawer's openability question asked
    /// about a remembered row, and asking it a second way is how a stored id
    /// starts reopening a screen the drawer would refuse to open.
    static func workspaceCartridge(forID id: String?) -> Cartridge? {
        guard let id, !id.isEmpty else { return nil }
        return selectable.first { $0.id == id }
    }
}
