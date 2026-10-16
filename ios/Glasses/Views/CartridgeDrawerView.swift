//
//  CartridgeDrawerView.swift
//  Glasses
//

import SwiftUI

/// The cartridge tray: the shell's workspace picker.
///
/// ## What selecting a cartridge does, and does not do
///
/// Choosing a cartridge here changes **which workspace this app shows**. The
/// choice itself sends nothing to the Tower and selects no module there —
/// there is no module-selection message in the protocol, and inventing one is
/// forbidden (docs/08-IOS-CARTRIDGE-SHELL.md). What a workspace does once it
/// opens is its own; the footer below says it to the person choosing.
///
/// The badge and being tappable are tied: a cartridge with nothing to open
/// cannot be `.readyToTest`, and one that ships a workspace is never
/// `.notBuilt`. `CartridgeCatalogTests` pins that in both directions.
/// The badge describes the *module's* position on the Tower roadmap; being
/// tappable describes whether *this app* ships a screen for it. Only cartridges
/// with a workspace are tappable — the rest stay exactly as they were, with no
/// `Button`, no `NavigationLink` and no tap target, because for them there
/// genuinely is nothing to open.
///
/// When the Tower can run modules, the active cartridge must be driven by what
/// the Tower reports is running, not by what was tapped here.
struct CartridgeDrawerView: View {
    @Binding var selectedCartridgeID: String

    /// Whether leaving the workspace on screen must ask first: a World
    /// Builder capture is running. Read at the tap.
    var leavingNeedsConfirmation: () -> Bool = { false }

    /// Stops the glasses camera, for "Stop capture". Leaving World Builder
    /// then asks the Tower to stop building, so both end together: the camera
    /// never keeps streaming with no build.
    var stopCapture: () -> Void = {}

    /// The row tapped while a capture runs, waiting on the alert.
    @State private var pendingSelection: String?

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        select("")
                    } label: {
                        HomeRow(isSelected: selectedCartridgeID.isEmpty)
                    }
                    .buttonStyle(.plain)
                }

                Section {
                    // Every catalog entry, in catalog order, and — critically —
                    // the openability decision is read off the row rather than
                    // re-derived here. `Cartridge.selectable` is defined as the
                    // openable rows of this same list, so the drawer and the
                    // rest of the app cannot come to different conclusions
                    // about what may be opened.
                    ForEach(Cartridge.drawerRows) { row in
                        switch row {
                        case .openable(let cartridge, _):
                            Button {
                                select(cartridge.id)
                            } label: {
                                CartridgeRow(
                                    row: row,
                                    isSelected: selectedCartridgeID == cartridge.id
                                )
                            }
                            .buttonStyle(.plain)
                        case .informational:
                            CartridgeRow(row: row, isSelected: false)
                        }
                    }
                } header: {
                    Text("Modules")
                        .foregroundStyle(.readableSecondary)
                } footer: {
                    // A person reads this *before* deciding whether opening a
                    // screen can make the Tower begin recording, so it must
                    // never understate what opening does. Opening a cartridge
                    // is not itself a recording, and the recording verbs
                    // (`cv_lab_start`, `POST /documents-session/start`,
                    // `POST /cartridges/object_memory/session/start`) live
                    // inside the workspaces. But two workspaces set the Tower
                    // working from `.onAppear`, with no control touched:
                    //
                    //   * World Builder POSTs its cartridge session `start`
                    //     (`WorldBuilderSessionController.workspaceDidAppear`).
                    //     An active session is what lets a builder attach to a
                    //     capture, so opening the screen while a camera is
                    //     streaming starts a world-build worker. That worker
                    //     writes: launched with `--root <world_root>`
                    //     (`tower/tower/main.py`), it opens a `WorldStore` on it
                    //     and writes sources, placements and a solve log — the
                    //     worlds "Saved worlds" lists. So opening this screen
                    //     over a live capture can put a reconstruction on the
                    //     Tower's disk.
                    //   * Scene Understanding subscribes to the live scene, and
                    //     a subscription *is* the watcher half of "somebody
                    //     streams and somebody watches", so opening the screen
                    //     over an open stream loads and runs a people detector.
                    //     It keeps nothing, and its own screen says so.
                    //
                    // That a `CartridgeSession` is not itself persisted does
                    // not mean nothing is kept: saying "keeps nothing" of both
                    // would be a false reassurance, in the one direction this
                    // sentence must never be wrong in.
                    Text("Opening a cartridge changes this app's workspace. Two of them also set the Tower working as they open: World Builder activates its session, and Scene Understanding starts watching the scene — so if a camera is already streaming, opening those is enough. They differ in what is left behind: Scene Understanding keeps nothing, while World Builder's session is what lets the Tower build and save a world from a capture that is already running. Recording is otherwise a deliberate act, and those controls live inside the workspace. Badges describe this app: \u{201C}Ready to test\u{201D} means there is something here to try, not that the Tower you are connected to is serving it right now. Each workspace says what that Tower can actually do.")
                        .foregroundStyle(.readableSecondary)
                        .padding(.top, 4)
                }
            }
            .navigationTitle("Cartridges")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            // Leaving World Builder mid-capture asks first, and "Stop capture"
            // stops the camera and the build together.
            //
            // An alert, not a `confirmationDialog`: on iOS 26 a confirmation
            // dialog can present as a popover, and a popover drops its
            // `.cancel` button (tapping outside stands in for it). Both
            // choices must be on screen, and an alert always draws both.
            .alert(
                CartridgeLeaveText.title,
                isPresented: Binding(
                    get: { pendingSelection != nil },
                    set: { if !$0 { pendingSelection = nil } }
                ),
                presenting: pendingSelection
            ) { id in
                Button(CartridgeLeaveText.stopCapture, role: .destructive) {
                    stopCapture()
                    selectedCartridgeID = id
                    pendingSelection = nil
                    dismiss()
                }
                Button(CartridgeLeaveText.keepCapturing, role: .cancel) {
                    pendingSelection = nil
                    dismiss()
                }
            } message: { _ in
                Text(CartridgeLeaveText.message)
            }
        }
    }

    /// A row's tap: switch at once, or ask first while a capture runs.
    private func select(_ id: String) {
        if id != selectedCartridgeID && leavingNeedsConfirmation() {
            pendingSelection = id
        } else {
            selectedCartridgeID = id
            dismiss()
        }
    }
}

/// The words of the leave-mid-capture alert.
enum CartridgeLeaveText {
    static let title = "Leave and stop the capture?"
    static let message = "Leaving World Builder stops the glasses camera and asks the Tower to stop building a world from this capture."
    static let stopCapture = "Stop capture"
    static let keepCapturing = "Keep capturing"
}

/// When leaving the workspace on screen must ask first: only World
/// Builder, and only while its capture runs or the glasses hold it paused.
/// Leaving there would ask the Tower to stop building while the camera kept
/// streaming.
enum CartridgeLeaveRule {
    static func needsConfirmation(workspace: CartridgeWorkspace?, claim: CaptureClaim) -> Bool {
        workspace == .worldBuilder && (claim == .running || claim == .devicePaused)
    }
}

private struct HomeRow: View {
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Home")
                    .font(.body.weight(.medium))
                Text("Infrastructure status and a plain capture session.")
                    .font(.caption)
                    .foregroundStyle(.readableSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // The words take the row's width and grow down, rather than
            // being offered only their one-line width beside a spacer.
            .frame(maxWidth: .infinity, alignment: .leading)
            if isSelected {
                Image(systemName: "checkmark")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.tint)
                    .accessibilityHidden(true)
            }
        }
        .padding(.vertical, 4)
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
        // Only `.isSelected`. The row is already wrapped in a `Button`, which
        // supplies the button trait; adding it here makes VoiceOver say it twice.
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

private struct CartridgeRow: View {
    /// The row, not the cartridge: openability arrives already decided rather
    /// than being worked out again here. The hint below is the second half of
    /// that decision and comes from the same place.
    let row: CartridgeDrawerRow
    let isSelected: Bool

    private var cartridge: Cartridge { row.cartridge }
    private var isOpenable: Bool { row.isOpenable }

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        // The badge under the words at the accessibility sizes, where a
        // trailing column squeezed both.
        let layout = dynamicTypeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
            : AnyLayout(HStackLayout(alignment: .top, spacing: 12))
        layout {
            VStack(alignment: .leading, spacing: 3) {
                Text(cartridge.name)
                    .font(.body.weight(.medium))
                Text(cartridge.summary)
                    .font(.caption)
                    .foregroundStyle(.readableSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: dynamicTypeSize.isAccessibilitySize ? .leading : .trailing, spacing: 6) {
                // Tinted for exactly one status, so the drawer answers "which
                // of these can I try?" in a glance rather than after reading
                // eight identical capsules. Colour is not the only carrier —
                // the badge still spells the status out — because a tint alone
                // would be invisible to a reader who cannot distinguish it.
                Text(cartridge.status.badge)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        cartridge.status.isProminent
                            ? AnyShapeStyle(Color.accentColor.opacity(0.18))
                            : AnyShapeStyle(Color(.tertiarySystemFill)),
                        in: .capsule
                    )
                    // The label colour on the tinted capsule: tint on a
                    // 0.18 tint was about 3:1, under the 4.5:1 text needs.
                    // The capsule still carries the colour.
                    .foregroundStyle(cartridge.status.isProminent ? AnyShapeStyle(.primary) : AnyShapeStyle(.readableSecondary))

                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.tint)
                        .accessibilityHidden(true)
                }
            }
        }
        .padding(.vertical, 4)
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
        // Two different truths, so two different hints. An unopenable row is
        // informational exactly as it always was. The strings live on the row
        // beside the decision that picks between them, so a row cannot be
        // untappable while announcing that it opens something.
        .accessibilityHint(row.accessibilityHint)
        // An openable row is already inside a `Button`, so only selection needs
        // stating. A row without a workspace gets no traits at all, which is
        // what makes it read as informational rather than actionable.
        .accessibilityAddTraits(isOpenable && isSelected ? .isSelected : [])
    }
}

#Preview {
    CartridgeDrawerView(selectedCartridgeID: .constant(""))
}
