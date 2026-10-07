//
//  WorldBuilderWorkspaceView.swift
//  Glasses
//

import MWDATCore
import SwiftUI

// `MWDATCamera` is imported for `StreamState`'s cases specifically. The target
// builds with `SWIFT_UPCOMING_FEATURE_MEMBER_IMPORT_VISIBILITY`, under which a
// type's members are only visible where the defining module is imported — so
// `== .streaming` does not resolve through `GlassesConnection` alone. Gated to
// match the property it reads.
#if DEBUG
import MWDATCamera
#endif

/// The World Builder workspace: what the glasses see, and what the Tower builds
/// from it.
///
/// ## What this screen may and may not claim
///
/// Those two halves are still in different states, and the whole design of this
/// view follows from saying so plainly.
///
/// **The capture half is real.** The iPhone can start the glasses camera, and
/// frames genuinely reach the Tower — that is the V0.7 pipeline, measured and
/// working.
///
/// **The world half is now reported rather than absent.** The Tower declares a
/// World Builder contract over the socket and reports what it has built, and
/// `TowerWorldBuilderClient` decodes it. What this screen shows is therefore
/// whatever the Tower says, and nothing else.
///
/// **Starting capture is still not the same as starting a build.** The Tower's
/// web process writes frames to a capture and answers `frame_result`; the
/// reconstruction runs in a *separate* process reading that capture from disk.
/// Whether one is running is not visible from the phone, and this app must not
/// imply it started one.
///
/// Three consequences, each of which is a deliberate refusal:
///
/// - **No "Start Mapping" button.** A verb-labelled primary button is the
///   strongest readiness claim a UI can make, and tapping it does not start a
///   build. The control says what it does — it starts a capture session — and
///   the world panel reports what the Tower says came of it.
/// - **No placeholder metrics.** A field the Tower did not report is not drawn
///   at all, rather than drawn as "—": six redacted values read as *broken*
///   rather than as *absent*.
/// - **No fabricated geometry.** No mesh, no spinner outside the two states
///   where work genuinely is underway, and no single world map. The Tower now
///   does send points and poses — over HTTP, per segment, never down the
///   socket that carries the frames — and this screen draws exactly those,
///   each segment in its own frame because the Tower has not registered them
///   into a shared one. What it will not do is composite them into a room
///   nobody measured.
///
/// The live preview presents `GlassesConnection.latestCapturedFrame` through
/// the same `ViewfinderCard` the Home workspace uses. It does not open a second
/// camera session; there is exactly one stream and one owner.
struct WorldBuilderWorkspaceView: View {
    @ObservedObject var glasses: GlassesConnection
    @ObservedObject var tower: TowerClient

    /// The world-model boundary. Two implementations exist:
    /// `TowerWorldBuilderClient`, which is what the app graph builds, and
    /// `UnavailableWorldBuilderClient`, which reports exactly one thing — that
    /// this screen is not connected to a world builder at all.
    ///
    /// A `@StateObject` so it is constructed once per workspace installation
    /// rather than on every render. It holds no runtime resources — no camera,
    /// no socket, no DAT reference — so losing it when the cartridge is
    /// deselected loses nothing real. Anything that must outlive the workspace
    /// belongs on `ProjectManager`, not here.
    @StateObject private var world: WorldBuilderViewModel

    /// Whether the saved-worlds sheet is up. View state, so it lives here.
    @State private var isShowingWorlds = false

    /// Set by the saved-worlds sheet's Connections button: the sheet goes
    /// first, and Connections opens once it has (U0.8 F09).
    @State private var opensConnectionsAfterWorlds = false

    /// The panel's one viewer model and web view, and the full-screen cover
    /// that same web view moves into (U-INLINE §3). Every room viewer this
    /// screen opens is the host's.
    @StateObject private var host = WorldInlineHost()

    /// A Saved worlds room row pinned a world: once the sheet has gone, the
    /// panel is scrolled into view, once (U-INLINE §2.4).
    @State private var scrollsToPanelAfterWorlds = false

    /// Whether this workspace is on screen (a cartridge switch takes it away).
    @State private var isOnScreen = false

    /// The panel's own full-screen cover took this screen away: not a
    /// cartridge switch, so the World Builder session is neither stopped
    /// when it goes up nor started again when it comes down.
    @State private var coveredByOwnCover = false

    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.worldPanelViewport) private var viewport
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Tells the Tower this workspace is on screen, so a builder may attach
    /// to a capture. View-owned on purpose — see the type's own doc comment
    /// for why it is not on the view model — and a `@StateObject` so the
    /// `stop` on disappear comes from the same object that sent `start`.
    @StateObject private var session: WorldBuilderSessionController

    #if DEBUG
    /// The operator's capture-health panel (U2-D0), fed by the same client.
    @StateObject private var health: CaptureHealthModel
    #endif

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    /// Connect and Settings, from the root (U0.8 F03, D1). `nil` in previews.
    @Environment(\.towerRecovery) private var recovery

    /// The client is injected rather than constructed here, and owned by
    /// `ProjectManager`. See `CartridgeClients` for why: this `@StateObject` is
    /// destroyed on every cartridge switch, and a Tower-backed client holding a
    /// subscription and a partly-built world must not be.
    init(
        glasses: GlassesConnection,
        tower: TowerClient,
        client: any WorldBuilderClient,
        session: WorldBuilderSessionController? = nil
    ) {
        self.glasses = glasses
        self.tower = tower
        _world = StateObject(wrappedValue: WorldBuilderViewModel(client: client))
        _session = StateObject(wrappedValue: session ?? WorldBuilderSessionController())
        #if DEBUG
        _health = StateObject(wrappedValue: CaptureHealthModel(client: client))
        #endif
    }

    /// Connectivity reaches the view model as a value, never as an object.
    ///
    /// This view genuinely needs `tower`: Start capture is off while the
    /// Tower is not connected (manager 137 D1), and the capture control says
    /// what happens to frames when it drops mid-capture. Frames are sent only
    /// while it is connected; frames taken while it is not are dropped
    /// (`TowerClient.sendFrame`) and never stored, and a reconnect re-opens
    /// the stream bracket (`ProjectManager`), so sending resumes. So the
    /// observation is not a dead dependency here, and reading the status
    /// costs nothing extra — passing the *fact* rather than the client is
    /// what keeps the view model free of a reference it could act on.
    ///
    /// The three cartridge workspaces that have no capture control receive this
    /// `Bool` from `TowerReachabilityReader` instead, and do not observe the
    /// connection at all.
    private var isTowerReachable: Bool { tower.status == .online }

    var body: some View {
        VStack(spacing: 16) {
            header

            // The look-back prompt (§6.5 as amended after walk 1): shown, with
            // a haptic, never spoken. `nil` unless following the live walk
            // this phone streams.
            if let banner = world.lookBackBanner {
                WorldLookBackBannerView(banner: banner)
            }

            #if DEBUG
            glassesPanel
            #endif

            WorldCanvasView(
                state: world.state,
                availability: world.availability(isTowerReachable: isTowerReachable),
                explanation: world.unavailableExplanation(isTowerReachable: isTowerReachable),
                inspection: world.inspection,
                sessionBinding: world.sessionBinding,
                fragments: world.fragmentsModel,
                geometryChunks: world.geometryChunks,
                report: world.walkPresentation,
                openReconstruction: { target in expand(target) },
                askAgain: { world.askTowerAgain() },
                goToCapture: goToCapture,
                awaitingIsOverdue: world.awaitingIsOverdue,
                panelNarratesFinishing: panelPhase.narratesFinishing,
                recentWorld: world.recentWorld,
                openRecent: { recent in
                    world.open(worldID: recent.worldID, sessionID: recent.sessionID)
                }
            )

            #if DEBUG
            captureControl
            #else
            HelperText("Capture is not available in this build.")
            #endif

            // The one World Builder panel (U-INLINE §1.1), in Capture health's
            // slot: below the capture control, so nothing it draws can move
            // Stop, and above the session line.
            panel

            // Under the capture control in both configurations: the Tower's
            // gate applies to a Release phone's neighbour as much as to a
            // DEBUG phone's own capture. One line, worded as what was asked
            // for — `active` is intent, and whether a builder attached is the
            // canvas's report, not this line's. A failed or refused request
            // can be asked again in place (U0.8 F02; manager 137 D2).
            VStack(spacing: 6) {
                HelperText(session.footnote)
                    .accessibilityIdentifier("wb-session-footnote")
                if session.canRetry {
                    Button(WorldBuilderSessionController.retryTitle) { session.retry() }
                        .font(.footnote)
                        .readableBorderedButton()
                        .accessibilityIdentifier("wb-session-retry")
                }
            }
        }
        .animation(.easeInOut(duration: 0.25), value: world.lookBackBanner)
        // Only one sheet can be up, so Connections waits for this one to go
        // (the root's `onDismiss` pattern).
        .sheet(isPresented: $isShowingWorlds, onDismiss: {
            if opensConnectionsAfterWorlds {
                opensConnectionsAfterWorlds = false
                recovery?.openConnections()
            }
            if scrollsToPanelAfterWorlds {
                scrollsToPanelAfterWorlds = false
                // Once, and animated only without Reduce Motion. Never
                // otherwise: at Stop the panel is already under the thumb.
                withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.3)) {
                    viewport.scrollTo?(Self.panelID)
                }
            }
        }) {
            WorldPickerView(
                world: world,
                onOpenConnections: {
                    opensConnectionsAfterWorlds = true
                    isShowingWorlds = false
                },
                // Back to the live screen, where Start capture is. It starts
                // nothing: the Start button is the only way in (U0.8 F10).
                onGoToCapture: {
                    world.returnToLive()
                    isShowingWorlds = false
                },
                // A room row pins its world and dismisses: the panel then
                // shows it, live inline (U-INLINE §2.4, manager 209).
                onOpenRoom: {
                    scrollsToPanelAfterWorlds = true
                    isShowingWorlds = false
                }
            )
        }
        // The panel's web view, moved here: no reload, the pose kept, the
        // bridge untouched (U-INLINE §3.2). The title, the note and the
        // notice come from the same `WorldPresentation` the canvas draws,
        // carried with the walk they describe (`world.coverWords`) -- and
        // shown only over that walk's picture (review 3, HIGH; review 4).
        // With the Tower gone the picture stays, its live stage goes, and
        // the notice says why (review MED 3).
        .fullScreenCover(isPresented: $host.isExpanded) {
            let words = host.coverWords.words(live: world.coverWords, towerReachable: isTowerReachable)
            WorldInlineCover(host: host, title: words.title, note: words.note, notice: words.notice,
                             progress: words.progress)
        }
        .onChange(of: world.coverWords) { _, words in
            host.coverReported(words)
        }
        .onChange(of: hostInputs, initial: true) { _, inputs in host.update(inputs) }
        // The World Builder cartridge session: `start` on appearance and
        // whenever the socket comes back while on screen, `stop` on
        // disappearance. Nothing else on the phone starts or stops a builder.
        .onAppear {
            isOnScreen = true
            if coveredByOwnCover {
                coveredByOwnCover = false
                return
            }
            session.workspaceDidAppear(isTowerReachable: isTowerReachable)
        }
        .onDisappear {
            // The world expanded over this screen is not a cartridge switch:
            // a walk in progress keeps its session (U-INLINE §3.2).
            if host.coverIsUp {
                coveredByOwnCover = true
                return
            }
            isOnScreen = false
            session.workspaceDidDisappear()
        }
        .onChange(of: isTowerReachable) { _, isReachable in
            session.towerReachabilityChanged(isReachable: isReachable)
        }
    }

    /// "Go to capture" for a saved world with nothing usable in it: back to
    /// the live screen, where Start capture is (U0.8 F12). Only while a saved
    /// world is pinned -- following live, Start capture is already here --
    /// and never in Release, which has no capture.
    private var goToCapture: (() -> Void)? {
        #if DEBUG
        world.inspection.isInspecting ? { world.returnToLive() } : nil
        #else
        nil
        #endif
    }

    // MARK: Header

    /// The two read-only controls, the sentence about the screen, and -- when
    /// a saved world is pinned -- which one, with the way back to live.
    ///
    /// No in-page "World Builder" title: the navigation bar already says it,
    /// directly above, and the two together read as the same title twice
    /// (UX audit, wb-header).
    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Side by side while both labels fit on one line each, and one
            // above the other when they do not. An HStack that never gave way
            // broke "Picture" and "Saved worlds" mid-word inside their
            // capsules at the accessibility sizes (UX audit, wb-header).
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    pictureButton
                    savedWorldsButton
                    Spacer(minLength: 0)
                }
                VStack(alignment: .leading, spacing: 8) {
                    pictureButton
                    savedWorldsButton
                }
            }

            Text("What the glasses see, and what the Tower reports it has built from that. Figures come from the Tower; absent ones are not drawn.")
                .font(.subheadline)
                .foregroundStyle(.readableSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if world.inspection.isInspecting {
                inspectingLine
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // Both read only, so neither is `#if DEBUG`: a Release build with no
    // camera can still look at what the Tower has stored.
    //
    // The picture button is disabled, not hidden, until the Tower has named a
    // world with geometry AND reported that world's walk: a control that
    // appears from nowhere is one nobody looks for, and a disabled one says
    // "not yet" truthfully. It offers the panel's picture (`panelTarget`),
    // never a target the report does not describe yet -- a Saved worlds pin
    // names its target before its report arrives (review 4, HIGH 1).
    private var pictureButton: some View {
        let target = Self.headerPicture(world)
        return Button {
            if let target { expand(target) }
        } label: {
            Label("Picture", systemImage: "cube.transparent")
                .font(.subheadline)
        }
        .readableBorderedButton()
        .disabled(target == nil)
        .accessibilityLabel("Interactive picture of the world")
        // Why it is off, for VoiceOver: the canvas says it on screen (U0.8
        // F14). Once here, so both of the header's layouts carry it.
        .accessibilityValue(target == nil ? WorldCanvasText.pictureOffValue : "")
        // Voice Control matches what is on screen: "Tap Picture" has to find
        // it, although VoiceOver reads the longer name.
        .accessibilityInputLabels(["Picture", "Interactive picture of the world"])
    }

    /// What the header's Picture button opens, or `nil` (off): the panel's
    /// picture, which the report describes (review 4, HIGH 1).
    static func headerPicture(_ world: WorldBuilderViewModel) -> WorldRenderTarget? {
        world.panelTarget
    }

    private var savedWorldsButton: some View {
        Button {
            isShowingWorlds = true
        } label: {
            Label("Saved worlds", systemImage: "archivebox")
                .font(.subheadline)
        }
        .readableBorderedButton()
    }

    /// Which saved world is pinned, and the way back. Wraps rather than
    /// truncating, and at the accessibility sizes puts the button under the
    /// sentence instead of squeezing both into one line.
    @ViewBuilder
    private var inspectingLine: some View {
        let layout = dynamicTypeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
            : AnyLayout(HStackLayout(alignment: .firstTextBaseline, spacing: 8))
        layout {
            // The world's NAME, or nothing. This line used to end in a raw
            // 32-character hex world id — `Looking at saved world
            // fcbca9e90b244785bdb671530b33c6a5.` — which is a database key on
            // the ordinary surface of the app. The id is still reachable: it
            // is in the canvas's Diagnostics disclosure and in the 3D viewer's
            // Details.
            Text(savedWorldLine)
                .font(.footnote)
                .foregroundStyle(.readableSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Back to live") {
                world.returnToLive()
            }
            .font(.footnote)
            .readableBorderedButton()
        }
    }

    /// What the inspecting line says. The Tower's display name when it gave
    /// one — 156 of the real root's 162 worlds have none — and the stage word
    /// otherwise, which is at least a fact about the world rather than a key
    /// into the Tower's filesystem.
    private var savedWorldLine: String {
        if let name = world.state.snapshot?.name, !name.isEmpty {
            return "Looking at saved world \(name)."
        }
        if let stage = world.presentation.stage {
            return "Looking at saved world · \(stage.label)."
        }
        return "Looking at saved world."
    }

    // MARK: The panel (U-INLINE)

    static let panelID = "wb-panel"

    /// Whether a capture runs on this phone. Never in Release.
    private var isCapturingNow: Bool {
        #if DEBUG
        isRunning
        #else
        false
        #endif
    }

    private var panelPhase: WorldPanelPhase {
        let presentation = world.presentation
        return WorldPanelPhase.phase(
            isCapturing: isCapturingNow,
            sessionActive: session.status == .active,
            towerReachable: isTowerReachable,
            pageLoaded: host.model?.state.html != nil,
            state: world.state,
            stage: presentation.stage,
            // Only a picture of the walk this report describes (review HIGH 1).
            target: world.panelTarget,
            needsRetrySentence: presentation.recoverability?.sentence,
            hasMap: panelMap != nil)
    }

    /// The panel's map: the walk's own coverage receipt, else the DEBUG fixture.
    private var panelMap: WorldPanelMapSource? {
        WorldPanelMap.source(live: world.coverage)
    }

    private var hostInputs: WorldInlineHost.Inputs {
        let phase = panelPhase
        return WorldInlineHost.Inputs(
            target: phase.target, isReady: phase.isReady, scenePhase: scenePhase,
            pickerShown: isShowingWorlds, isOnScreen: isOnScreen)
    }

    private var buildInProgress: Bool? {
        if case .finalizing(_, let buildInProgress) = world.state { return buildInProgress }
        return nil
    }

    @ViewBuilder
    private var panel: some View {
        WorldInlinePanel(
            phase: panelPhase,
            report: world.walkPresentation,
            buildInProgress: buildInProgress,
            isTowerReachable: isTowerReachable,
            screenSaysOffline: captureSaysTowerIsOff,
            showsHealth: showsHealthInPanel,
            map: panelMap,
            host: host,
            dismissBanner: { world.dismissFinishBanner() },
            bannerAnnounced: { walk in world.finishBannerAnnounced(for: walk) },
            expand: { target in expand(target) }
        ) {
            #if DEBUG
            CaptureHealthView(model: health, isLinked: isTowerReachable, isCapturing: isRunning)
            #else
            EmptyView()
            #endif
        }
        .id(Self.panelID)
    }

    /// The capture control already says the Tower is not connected (its
    /// `wb-capture-tower-line`): the panel then says it no third time.
    ///
    /// `false` in Release on purpose, not as a gap (review 3, LOW): Release
    /// compiles no capture control, so no `wb-capture-tower-line` exists to
    /// defer to. The rule -- never a THIRD "not connected" line -- then
    /// leaves the panel's line as the second, beside the canvas's own.
    private var captureSaysTowerIsOff: Bool {
        #if DEBUG
        guard !isTowerReachable else { return false }
        return controlMode == .start
            || WorldBuilderCaptureText.towerLine(status: tower.status, gaveUp: tower.reconnectGaveUp) != nil
        #else
        false
        #endif
    }

    private var showsHealthInPanel: Bool {
        #if DEBUG
        showsCaptureHealth
        #else
        false
        #endif
    }

    /// Full screen: the panel's web view moves into the cover (U-INLINE
    /// §3.2), with the report's words and the walk they are about; the
    /// cover shows them only over that walk's picture. Every route into the
    /// cover -- the panel, the header's Picture, the canvas -- comes here.
    /// Reduce Motion: no slide.
    private func expand(_ target: WorldRenderTarget) {
        var transaction = Transaction()
        transaction.disablesAnimations = reduceMotion
        withTransaction(transaction) {
            host.expand(target, words: world.coverWords)
        }
    }
}

// MARK: - DEBUG-only capture surface

#if DEBUG
/// The capture control's Tower sentences (U0.8 F04; manager 137 D1).
enum WorldBuilderCaptureText {
    static let framesRule = "Frames are sent only while it is connected; frames taken before then are not kept."

    /// While a capture runs and the Tower is not connected: what happens to
    /// the frames. `nil` while it is connected.
    static func towerLine(status: TowerStatus, gaveUp: Bool) -> String? {
        if status == .online { return nil }
        if gaveUp { return "The phone has stopped trying to reconnect to the Tower. " + framesRule }
        if status == .connecting { return "Connecting to the Tower. " + framesRule }
        return "The Tower is not connected. " + framesRule
    }

    /// Beside a Start capture that is off because the Tower is not
    /// connected: the reason and the next step.
    static func startOffReason(status: TowerStatus, gaveUp: Bool) -> String {
        if gaveUp {
            return "Start is off while the Tower is not connected, and the phone has stopped trying to reconnect. "
                + "Connect to it first."
        }
        if status == .connecting {
            return "Start is off until the Tower is connected. The phone is connecting to it now."
        }
        return "Start is off while the Tower is not connected. Connect to it first."
    }
}
#endif

// The camera path is DEBUG-only in the model, so the capture half of this
// workspace is gated to match. In Release the workspace still exists and still
// tells the truth about the world half — it simply has no capture controls,
// exactly as no other screen in the app does.
#if DEBUG
private extension WorldBuilderWorkspaceView {

    var isStreaming: Bool { glasses.cameraStreamState == .streaming }

    /// Includes the device-session states, not just the stream's. See
    /// `GlassesConnection.isCaptureEngaged` — deriving this from the stream
    /// alone left a window in which a session existed but the control still
    /// read "Start", and a tap in it did nothing observable.
    var isRunning: Bool { glasses.isCaptureEngaged }

    /// The capture-health panel is for a walk: a World Builder session the
    /// Tower honoured, or a capture running on this phone.
    var showsCaptureHealth: Bool {
        (session.status == .active || isRunning) && !UITestHooks.hidesCaptureHealth
    }

    /// What the wearer currently sees.
    @ViewBuilder
    var glassesPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel("What the glasses see")
            ViewfinderCard(
                frame: glasses.latestCapturedFrame,
                isStreaming: isStreaming,
                placeholderReason: placeholder,
                isPausedByGlasses: glasses.captureClaim == .devicePaused
            )
        }
    }

    /// The viewfinder's sentence, so the placeholder and Start's hint cannot
    /// drift apart.
    static var waitingForGlasses: String { ViewfinderText.waitingForGlasses }

    /// Start's VoiceOver hint: why it is off, or nothing.
    var startHint: String {
        if !glasses.hasActiveDevice { return Self.waitingForGlasses }
        if !isTowerReachable {
            return WorldBuilderCaptureText.startOffReason(status: tower.status, gaveUp: tower.reconnectGaveUp)
        }
        return ""
    }

    /// Never "Start…" while the camera is on or coming up (U0.8 F05).
    var placeholder: String {
        ViewfinderText.placeholder(
            isStreaming: isStreaming,
            isEngaged: glasses.isCaptureEngaged,
            isPausedByGlasses: glasses.captureClaim == .devicePaused,
            hasActiveDevice: glasses.hasActiveDevice,
            permission: glasses.cameraPermissionStatus,
            noun: "capture session"
        )
    }

    /// Stop, Stop while the glasses hold it paused, or Start (U0.8 F06).
    var controlMode: CaptureControlMode {
        CaptureControlMode.mode(isEngaged: glasses.isCaptureEngaged, claim: glasses.captureClaim)
    }

    /// Labelled for what it actually does. See the type's doc comment for why
    /// this is not "Start Mapping".
    @ViewBuilder
    var captureControl: some View {
        VStack(spacing: 8) {
            if controlMode != .start {
                // A capture the glasses paused is still a capture: a Start
                // then would be refused and do nothing observable (U0.8 F06).
                Button {
                    glasses.stopCameraSession()
                } label: {
                    Label("Stop capture", systemImage: "stop.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .readableBorderedButton()
                .disabled(glasses.cameraStreamState == .stopping)
                if controlMode == .stopWhilePaused {
                    HelperText(ViewfinderText.pausedControlLine(stopTitle: "Stop capture"))
                        .accessibilityIdentifier("capture-paused-line")
                }
            } else {
                Button {
                    glasses.startCameraSession()
                } label: {
                    Label("Start capture", systemImage: "play.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                // Off while the Tower is not connected, too (manager 137 D1):
                // a capture started then streams frames nobody keeps. Start
                // never does nothing; it is off, with the reason beside it.
                .disabled(!glasses.hasActiveDevice || !isTowerReachable)
                // Why it is disabled, at the control, for VoiceOver. On screen
                // the viewfinder above says the glasses' reason, and the line
                // under this control says the Tower's.
                .accessibilityHint(startHint)
            }

            // Neither string claims a build. The Tower reconstructs in a
            // separate process reading the capture from disk, and nothing on
            // the phone can see whether one is running — so the panel above,
            // which reports only what the Tower said, is where that question
            // is answered.
            HelperText(
                isRunning
                    ? "Frames are streaming to the Tower. What it builds from them is reported above."
                    : "Streams frames to the Tower. What it builds from them is reported above."
            )

            if controlMode == .start && !isTowerReachable {
                // Start is off because of the Tower (D1): the reason, beside
                // the control, and the way to end it. First, because it holds
                // whether or not the glasses are active -- the viewfinder
                // above already gives the glasses' reason.
                HelperText(WorldBuilderCaptureText.startOffReason(
                    status: tower.status, gaveUp: tower.reconnectGaveUp
                ))
                .accessibilityIdentifier("wb-capture-tower-line")
                if let recovery {
                    TowerRecoveryButtons(actions: recovery, identifierPrefix: "wb-capture")
                }
            } else if !glasses.hasActiveDevice && controlMode == .start {
                // Nothing: the viewfinder card at the top of this workspace
                // is showing this very sentence, and it read twice on one
                // screen (UX audit, wb-capture-idle). Kept as a branch so the
                // advice below still never shows while the glasses are away.
                EmptyView()
            } else if glasses.cameraPermissionStatus == .denied && controlMode == .start {
                // Advice, not a `.disabled` condition — see the equivalent
                // branch in `HomeWorkspaceView.sessionControl`.
                HelperText("Camera access is not granted. Allow it under Connections, then start capture.")
            } else if let line = WorldBuilderCaptureText.towerLine(
                status: tower.status, gaveUp: tower.reconnectGaveUp
            ) {
                // The Tower dropped mid-capture. What happens to the frames,
                // truthfully: they are sent only while it is connected, and
                // the ones taken before it reconnects are dropped and never
                // stored -- a reconnect re-opens the stream bracket, so
                // sending resumes (`ProjectManager`), but nothing refills the
                // gap. The words differ once the phone has stopped retrying,
                // because the remedy does: then only a tap on Connect helps.
                HelperText(line)
                    .accessibilityIdentifier("wb-capture-tower-line")
            }
        }
    }
}
#endif
