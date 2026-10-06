//
//  WorldInlinePanel.swift
//  Glasses
//
//  U-INLINE (manager rulings 209 and 210): one World Builder panel for the
//  whole walk, in Capture health's slot under the capture control. During
//  the walk it is Capture health (and, later, the fog-of-war map); after
//  Stop it shows the honest finishing progress (U0.6); when the world is
//  ready the same panel cross-fades into the live 3D world inline, and a tap
//  expands that same web view to full screen.
//
//  One web view at most (§4): `WorldInlineHost` creates it only while it is
//  wanted and tears it down within 5 s off-screen, at once in the
//  background, under Saved worlds, and when the workspace goes.
//

import Combine
import SwiftUI
import UIKit

// MARK: - The host: one model, one web view

/// Owns the panel's viewer model -- and through it the one `WKWebView` --
/// and the expanded cover's state (U-INLINE §3.1, §4).
@MainActor
final class WorldInlineHost: ObservableObject {
    enum Placement: Equatable { case inline, expanded }

    /// What the workspace knows that decides whether the web view is wanted.
    struct Inputs: Equatable {
        var target: WorldRenderTarget?
        var isReady = false
        var scenePhase: ScenePhase = .active
        var pickerShown = false
        var isOnScreen = true
    }

    @Published private(set) var model: WorldRenderViewerModel?
    /// Bound to the full-screen cover. Setting it `false` (Close) leaves the
    /// web view in the cover for the slide-down; `coverDisappeared()` brings
    /// it back.
    @Published var isExpanded = false
    /// Which container holds the web view: the panel's or the cover's.
    @Published private(set) var placement: Placement = .inline
    /// The cover shows one of the room's areas in the room's place (C1 E5,
    /// REPLACE): the room's web view is held off meanwhile.
    @Published private(set) var showsArea = false

    private(set) var inputs = Inputs()
    /// From `expand(_:)` until the cover has gone.
    private(set) var coverIsUp = false
    private(set) var expandedTarget: WorldRenderTarget?
    /// The panel has been off screen for `offscreenGrace`.
    private(set) var offscreenExpired = false

    private var lifecycle: Task<Void, Never>?
    private var offscreen: Task<Void, Never>?
    private var modelChanges: AnyCancellable?

    let offscreenGrace: Duration
    private let makeModel: @MainActor (WorldRenderTarget) -> WorldRenderViewerModel
    private let run: @MainActor (WorldRenderViewerModel) async -> Void

    init(
        offscreenGrace: Duration = .seconds(5),
        makeModel: @escaping @MainActor (WorldRenderTarget) -> WorldRenderViewerModel = {
            WorldRenderViewerModel(target: $0)
        },
        run: @escaping @MainActor (WorldRenderViewerModel) async -> Void = WorldInlineHost.lifecycle
    ) {
        self.offscreenGrace = offscreenGrace
        self.makeModel = makeModel
        self.run = run
    }

    /// The model's life on screen: fetch, the areas row, then follow.
    static func lifecycle(_ model: WorldRenderViewerModel) async {
        await model.load()
        guard !Task.isCancelled else { return }
        await model.refreshComponents()
        guard !Task.isCancelled else { return }
        await model.followRevisions()
    }

    /// Whether the web view may exist now (U-INLINE §4).
    var wantsWebView: Bool {
        guard !showsArea else { return false }
        guard coverIsUp || inputs.isOnScreen else { return false }
        return WorldInlineWebPolicy.wantsWebView(
            isReady: inputs.isReady, isExpanded: coverIsUp, panelVisible: !offscreenExpired,
            scenePhase: inputs.scenePhase, pickerShown: inputs.pickerShown)
    }

    func update(_ next: Inputs) {
        guard next != inputs else { return }
        inputs = next
        evaluate()
    }

    /// The panel entered or left the visible scroll area. Off screen it is
    /// kept for `offscreenGrace`, then suspended; back on screen it is
    /// wanted again (a new model and a fresh load: the imagery was dropped).
    func visibilityChanged(_ visible: Bool) {
        offscreen?.cancel()
        offscreen = nil
        if visible {
            guard offscreenExpired else { return }
            offscreenExpired = false
            evaluate()
            return
        }
        guard !offscreenExpired else { return }
        let grace = offscreenGrace
        offscreen = Task { [weak self] in
            try? await Task.sleep(for: grace)
            guard !Task.isCancelled, let self else { return }
            self.offscreen = nil
            self.offscreenExpired = true
            self.evaluate()
        }
    }

    /// Full screen: the same web view, moved into the cover (§3.2).
    func expand(_ target: WorldRenderTarget) {
        expandedTarget = target
        coverIsUp = true
        evaluate()
        isExpanded = true
    }

    /// The cover is on screen: its container takes the web view.
    func coverAppeared() {
        placement = .expanded
    }

    /// The cover has gone (its `onDisappear`): back to the panel's container,
    /// or torn down when the panel does not want it.
    func coverDisappeared() {
        isExpanded = false
        coverIsUp = false
        expandedTarget = nil
        showsArea = false
        placement = .inline
        evaluate()
    }

    func setShowsArea(_ shows: Bool) {
        guard shows != showsArea else { return }
        showsArea = shows
        evaluate()
    }

    func evaluate() {
        let target = coverIsUp ? (expandedTarget ?? inputs.target) : inputs.target
        if wantsWebView, let target {
            want(target)
        } else {
            suspend()
        }
    }

    /// A model for `target`: the one held when it is the same target;
    /// otherwise the old one is torn down BEFORE the new one exists.
    func want(_ target: WorldRenderTarget) {
        if let model, model.target == target { return }
        suspend()
        let model = makeModel(target)
        self.model = model
        // The panel draws from the model's state and chrome.
        modelChanges = model.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        let run = self.run
        lifecycle = Task { await run(model) }
    }

    /// The model, only when it is `target`'s (review HIGH 2): until the
    /// host catches up with a new phase it still holds the previous
    /// target's page, and the panel never draws that under this phase.
    func model(for target: WorldRenderTarget?) -> WorldRenderViewerModel? {
        guard let model, let target, model.target == target else { return nil }
        return model
    }

    /// Cancel the lifecycle, close the viewer (drops the imagery, PRIVACY
    /// §3.6, and resets the chrome), tear the web view down (`close` to the
    /// page's held `await`), forget the model.
    func suspend() {
        lifecycle?.cancel()
        lifecycle = nil
        modelChanges = nil
        guard let model else { return }
        self.model = nil
        model.viewerClosed()
    }
}

// MARK: - The scroll gate (U-INLINE §3.3)

/// Inline, the screen's `ScrollView` would steal every vertical drag from
/// the page. This pan decides first: horizontal drags and any two-finger
/// gesture go to the page (the scroll view's pan must wait for this one to
/// fail, and it does not); vertical drags fail it, and the screen scrolls --
/// the page then receives `pointercancel` and stops cleanly.
final class WorldInlineScrollGate: UIPanGestureRecognizer, UIGestureRecognizerDelegate {
    /// Found when the container moves to a window; weak, a dynamic
    /// dependency, so nothing outlives the gate.
    weak var enclosingScrollView: UIScrollView?

    /// Pure: whether a drag with this velocity and touch count belongs to
    /// the page. A 45° drag is vertical.
    nonisolated static func begins(velocity: CGPoint, touches: Int) -> Bool {
        touches >= 2 || abs(velocity.x) > abs(velocity.y)
    }

    init() {
        super.init(target: nil, action: nil)
        cancelsTouchesInView = false
        delaysTouchesBegan = false
        delaysTouchesEnded = false
        delegate = self
    }

    func gestureRecognizerShouldBegin(_ gestureRecognizer: UIGestureRecognizer) -> Bool {
        Self.begins(velocity: velocity(in: view), touches: numberOfTouches)
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        true
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldBeRequiredToFailBy other: UIGestureRecognizer) -> Bool {
        other === enclosingScrollView?.panGestureRecognizer
    }
}

// MARK: - The screen's scroll view, as the panel needs it

/// The visible height of the screen's scroll view (B, U-INLINE §1.1), and
/// how to bring the panel into view after a Saved worlds pin (§2.4).
struct WorldPanelViewport {
    var visibleHeight: CGFloat = 0
    var scrollTo: ((String) -> Void)?
}

extension EnvironmentValues {
    @Entry var worldPanelViewport = WorldPanelViewport()
}

// MARK: - The fog-of-war slot

/// S1's map slot. FOW v1 replaces the fixture; nothing else moves. The
/// fixture draws only behind `-WBPanelMapFixture` (a DEBUG launch argument).
enum WorldPanelMap {
    static let fixtureArgument = "-WBPanelMapFixture"

    static var fixtureEnabled: Bool {
        #if DEBUG
        ProcessInfo.processInfo.arguments.contains(fixtureArgument)
        #else
        false
        #endif
    }
}

/// A grey disc of twelve sectors, labelled *Fixture map*: the slot's
/// stand-in until the fog-of-war map lands.
struct WorldPanelMapFixtureView: View {
    let isFrozen: Bool

    var body: some View {
        Canvas { context, size in
            let radius = min(size.width, size.height) * 0.38
            let centre = CGPoint(x: size.width / 2, y: size.height / 2)
            for sector in 0..<12 {
                var path = Path()
                path.move(to: centre)
                path.addArc(center: centre, radius: radius,
                            startAngle: .degrees(Double(sector) * 30 + 1),
                            endAngle: .degrees(Double(sector + 1) * 30 - 1), clockwise: false)
                path.closeSubpath()
                let shade = 0.30 + 0.04 * Double(sector % 4)
                context.fill(path, with: .color(Color(white: shade)))
            }
        }
        .overlay(alignment: .bottomLeading) {
            // Not under the finish block: frozen, the map is only a backdrop.
            if !isFrozen {
                Text(WorldPanelCopy.mapFixture)
                    .font(.caption)
                    .foregroundStyle(WorldChromeStyle.secondary)
                    .padding(8)
            }
        }
        .opacity(isFrozen ? 0.35 : 1)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(WorldPanelCopy.mapFixture)
        .accessibilityIdentifier("wb-panel-map")
    }
}

// MARK: - The panel

/// U-INLINE §1.1: rows 0-5 under the capture control. Every row is present
/// only when the phase says so.
struct WorldInlinePanel<Health: View>: View {
    let phase: WorldPanelPhase
    let presentation: WorldPresentation
    /// `lifecycle.build_in_progress` of a finalizing state, for the spinner.
    let buildInProgress: Bool?
    let isTowerReachable: Bool
    /// The walk the report describes (world and session), or `nil`.
    let walk: WorldFinishWalk?
    /// The screen already says the Tower is not connected (the capture
    /// control's line): the offline stage then shows only its actions.
    let screenSaysOffline: Bool
    /// Capture health is drawn while walking (a session or a capture).
    let showsHealth: Bool
    /// The frozen map under the finishing wait, when there was one.
    let hasMap: Bool
    @ObservedObject var host: WorldInlineHost
    let dismissBanner: () -> Void
    /// The away banner was announced: never again for this walk.
    let bannerAnnounced: () -> Void
    let expand: (WorldRenderTarget) -> Void
    @ViewBuilder let health: () -> Health

    @Environment(\.towerRecovery) private var recovery
    @Environment(\.worldPanelViewport) private var viewport
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var width: CGFloat = 0
    /// The page has been ready for the model's `renderTimeout` without a
    /// drawn state: show it anyway, with its own message.
    @State private var readyTimedOut = false
    /// The finish block as it last stood while finishing, for its walk:
    /// what the wait overlay shows until the world is drawn, so the
    /// cross-fade starts from the picture the wearer was looking at, and
    /// never from words recomputed for the settled world (which no longer
    /// has a stage line) -- nor from another walk's wait.
    @State private var finish = WorldPanelFinishMemory<FinishingBlock>()

    /// What S2 draws: the spinner, the stage line, the elapsed line, or the
    /// canvas's own sentence when there is no stage line.
    struct FinishingBlock: Equatable {
        var showsSpinner: Bool
        var line: WorldFinishLine?
        var stoppedAt: ContinuousClock.Instant?
        var detail: String?
        /// The Tower dropped: the offline notice stands in for the claims.
        var offline = false

        /// With the Tower gone (an open preview keeps the phase finishing),
        /// the last stage, the spinner and the Tower's sentence are no
        /// longer current: none is claimed, the elapsed clock stays (it is
        /// the phone's), and the offline notice says why (review MED 3).
        func reachable(_ towerReachable: Bool) -> FinishingBlock {
            guard !towerReachable else { return self }
            return FinishingBlock(showsSpinner: false, line: nil, stoppedAt: stoppedAt, detail: nil, offline: true)
        }
    }

    private var currentFinishing: FinishingBlock? {
        guard phase.isFinishing else { return nil }
        return FinishingBlock(
            showsSpinner: presentation.showsLiveBuild(buildInProgress: buildInProgress),
            line: presentation.finishLine,
            stoppedAt: presentation.stoppedAt,
            detail: presentation.finishLine == nil
                ? WorldPresentation.finalizingDetail(buildInProgress: buildInProgress,
                                                     photographic: presentation.photographic)
                : nil)
        .reachable(isTowerReachable)
    }

    var body: some View {
        VStack(spacing: 12) {
            // Rows 0-4: one accessibility container, read in §5's order.
            // Capture health stays its own container beside it: nested as the
            // only child of this one, SwiftUI folded it in and its identity
            // ("capture-health") was lost.
            VStack(spacing: 12) {
                if showsBanner, let headline = presentation.headline {
                    WorldFinishBannerView(text: WorldFinishCopy.awayBanner(headline: headline),
                                          announces: presentation.finishClock.announcesAwayBanner,
                                          announced: bannerAnnounced, dismiss: dismissBanner)
                        .transition(reduceMotion ? .opacity : .move(edge: .top).combined(with: .opacity))
                        .accessibilitySortPriority(100)
                }
                if phase.drawsCard {
                    card
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("wb-panel")
            // Under the full-screen world the panel is not there to read:
            // its web view is in the cover, and its rows would be read twice.
            .accessibilityHidden(host.placement == .expanded)
            if case .walking = phase, showsHealth {
                health()
            }
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.25), value: showsBanner)
        .onScrollVisibilityChange(threshold: 0.2) { visible in host.visibilityChanged(visible) }
        .onChange(of: phase) { old, new in
            // One announcement, unless the away banner says it.
            if finish.phaseChanged(from: old, to: new), !showsBanner, let headline = presentation.headline {
                AccessibilityNotification.Announcement(WorldFinishCopy.finished(headline: headline)).post()
            }
        }
        .onChange(of: currentFinishing, initial: true) { _, block in
            finish.record(block, walk: walk)
        }
        .onChange(of: walk) { _, walk in
            finish.record(currentFinishing, walk: walk)
        }
        .onChange(of: webShown) { _, shown in
            guard shown else { return }
            // The wait overlay has gone with this commit: the chrome may now
            // be told the page finished, so `activate` reaches the page only
            // once the native chrome is visible (WORLDS §4c).
            let model = host.model(for: phase.target)
            Task { @MainActor in
                await Task.yield()
                model?.renderingPanelGone()
            }
        }
        .task(id: readyKey) {
            readyTimedOut = false
            guard readyKey != nil, let timeout = host.model(for: phase.target)?.renderTimeout else { return }
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled else { return }
            readyTimedOut = true
        }
    }

    private var showsBanner: Bool {
        guard presentation.finishClock.showsAwayBanner else { return false }
        switch phase {
        case .ready, .failed: return true
        case .hidden, .walking, .finishing, .offline: return false
        }
    }

    // MARK: The card

    private var card: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Row 1: the research marker, read directly -- in every chrome
            // mode, a fallback included (U-INLINE §2.2, the everNative MED).
            if let marker = host.model(for: phase.target)?.chrome.researchMarker {
                WorldChromeResearchBand(marker: marker)
                    .environment(\.colorScheme, .dark)
                    .accessibilitySortPriority(95)
            }
            if let heading {
                Text(heading)
                    .font(.headline)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .accessibilityAddTraits(.isHeader)
                    .accessibilitySortPriority(90)
                    .accessibilityIdentifier("wb-panel-headline")
            }
            stage
                .accessibilitySortPriority(80)
            footer
        }
        .frame(maxWidth: .infinity)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { new in
            if MeasuredHeight.moved(from: width, to: new) { width = new }
        }
    }

    /// The stage's own word, never one of the panel's (U-INLINE §2.2):
    /// *Interrupted* is never "Saved".
    private var heading: String? {
        switch phase {
        case .finishing, .ready, .failed, .offline: return presentation.headline
        case .hidden, .walking: return nil
        }
    }

    private var stageHeight: CGFloat {
        WorldPanelLayout.stageHeight(width: width, visibleHeight: viewport.visibleHeight)
    }

    // MARK: Row 3: the stage

    @ViewBuilder
    private var stage: some View {
        let height = stageHeight
        let fixed: Bool = {
            switch phase {
            case .ready, .walking: return true
            case .hidden, .finishing, .failed, .offline: return false
            }
        }()
        ZStack(alignment: .topLeading) {
            Color(WorldRenderLoadingPanel.pageBackground)
            stageContent
        }
        .frame(maxWidth: .infinity)
        .frame(height: fixed ? height : nil)
        .frame(minHeight: fixed ? nil : height)
        .clipShape(UnevenRoundedRectangle(topLeadingRadius: 18, topTrailingRadius: 18))
        .environment(\.colorScheme, .dark)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("wb-panel-stage")
    }

    @ViewBuilder
    private var stageContent: some View {
        switch phase {
        case .hidden:
            EmptyView()
        case .walking:
            WorldPanelMapFixtureView(isFrozen: false)
        case .finishing:
            if let block = currentFinishing { finishingContent(block) }
        case .ready(let target):
            readyContent(target)
        case .failed(let sentence):
            stageText(sentence ?? host.model?.state.failureMessage ?? presentation.recoverability?.sentence ?? "")
        case .offline:
            VStack(alignment: .leading, spacing: 12) {
                if let line = WorldPanelCopy.offlineLine(screenSaysOffline: screenSaysOffline) {
                    Text(line)
                        .font(.body)
                        .foregroundStyle(WorldChromeStyle.text)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("wb-panel-offline")
                }
                if let recovery {
                    TowerRecoveryButtons(actions: recovery, identifierPrefix: "wb-panel")
                        .environment(\.colorScheme, .light)
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// S2: the frozen map when there was one, and over it -- or alone on the
    /// plate -- the finish block. Never a duration or an estimate.
    private func finishingContent(_ block: FinishingBlock) -> some View {
        ZStack(alignment: .topLeading) {
            if hasMap {
                WorldPanelMapFixtureView(isFrozen: true)
            }
            VStack(alignment: .leading, spacing: 10) {
                WorldFinishLineView(showsSpinner: block.showsSpinner, line: block.line,
                                    stoppedAt: block.stoppedAt, style: .panel)
                if let detail = block.detail {
                    // The canvas's own true sentence: owed, unobservable, or
                    // not reported.
                    Text(detail)
                        .font(.body)
                        .foregroundStyle(WorldChromeStyle.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("wb-panel-finishing-detail")
                }
                if block.offline {
                    Text(WorldPanelCopy.finishingButOffline)
                        .font(.body)
                        .foregroundStyle(WorldChromeStyle.text)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("wb-panel-finishing-offline")
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func stageText(_ text: String) -> some View {
        Text(text)
            .font(.body)
            .foregroundStyle(WorldChromeStyle.text)
            .fixedSize(horizontal: false, vertical: true)
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityIdentifier("wb-panel-failed")
    }

    /// S3: the world, laid out at full size from the start and faded in at
    /// `webShown`. Never `.hidden()`: WebKit throttles a hidden view and the
    /// page would not draw.
    @ViewBuilder
    private func readyContent(_ target: WorldRenderTarget) -> some View {
        if let model = host.model(for: target), let html = model.state.html {
            let native = usesNativeChrome(model, html: html)
            let shown = webShown
            ZStack {
                WorldRenderWebView(
                    owner: model.web, html: html,
                    attempt: model.renderAttempt, budgetToken: model.renderBudgetToken,
                    isActive: host.placement == .inline, hosted: true, inline: true,
                    onTap: native ? { expand(target) } : nil,
                    onEvent: model.pageEvent)
                    .overlay {
                        if native {
                            WorldChromeCanvasLayer(
                                chrome: model.chrome, panel: .constant(nil),
                                onAction: { model.bridge.receive(.tapped($0)) },
                                onFirstStateDrawn: { model.bridge.receive(.firstStateDrawn) },
                                banner: { EmptyView() }, areas: { EmptyView() }, details: { EmptyView() },
                                presentation: .inline)
                            .tint(Color.readableTint)
                        }
                    }
                    .accessibilitySortPriority(2)
                    .opacity(shown ? 1 : 0)
                // Faded out and then gone, so nothing of the wait is left
                // for VoiceOver or a touch once the world is shown.
                if !shown {
                    waitOverlay(for: target)
                        .transition(.opacity)
                }
            }
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.35), value: shown)
        } else if let failure = host.model(for: target)?.state.failureMessage {
            stageText(failure)
        } else {
            waitOverlay(for: target)
        }
    }

    /// Until the web view is shown: the finishing content when the world
    /// came from ITS wait, otherwise a spinner and *Opening your world…* (a
    /// fetch is genuinely in flight).
    @ViewBuilder
    private func waitOverlay(for target: WorldRenderTarget) -> some View {
        if let block = finish.overlay(for: target) {
            finishingContent(block)
                .background(Color(WorldRenderLoadingPanel.pageBackground))
        } else {
            VStack(spacing: 10) {
                ProgressView().tint(WorldChromeStyle.text)
                Text(WorldPanelCopy.opening)
                    .font(.body)
                    .foregroundStyle(WorldChromeStyle.text)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(WorldRenderLoadingPanel.pageBackground))
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(WorldPanelCopy.openingLabel)
            .accessibilityIdentifier("wb-panel-opening")
        }
    }

    /// The page is drawn, or will never say: `.ready` and (today's chrome,
    /// the page's first drawn state, or the model's render timeout).
    private var webShown: Bool {
        guard phase.isReady, let model = host.model(for: phase.target), case .ready = model.state else { return false }
        return model.chrome.mode == .legacy || model.chrome.state?.drawn == true || readyTimedOut
    }

    /// Restarts the render-timeout wait for each model and each ready page.
    private var readyKey: String? {
        guard phase.isReady, let model = host.model(for: phase.target), case .ready = model.state else { return nil }
        return "\(ObjectIdentifier(model).hashValue)-\(model.renderAttempt)-\(model.pageFinishedToken)"
    }

    private func usesNativeChrome(_ model: WorldRenderViewerModel, html: String) -> Bool {
        !model.chrome.refusedForScreen && WorldChromeEcho.isOffered(in: html)
    }

    // MARK: Row 4: the footer

    @ViewBuilder
    private var footer: some View {
        switch phase {
        case .finishing(let target?):
            footerRow(line: nil) {
                actionButton(WorldPreviewCopy.openPreview, hint: WorldPanelCopy.fullScreenHint,
                             identifier: "wb-panel-expand") { expand(target) }
            }
        case .ready(let target):
            if let model = host.model(for: target), model.state.failureMessage != nil {
                footerRow(line: nil) {
                    actionButton(WorldPanelCopy.tryAgain, hint: nil, identifier: "wb-panel-retry") {
                        Task { await model.load() }
                    }
                }
            } else {
                footerRow(line: readyLine) {
                    actionButton(WorldPanelCopy.fullScreen, hint: WorldPanelCopy.fullScreenHint,
                                 identifier: "wb-panel-expand") { expand(target) }
                }
            }
        case .hidden, .walking, .finishing(nil), .failed, .offline:
            EmptyView()
        }
    }

    /// The ready footer's one line: the Tower dropped under a loaded page,
    /// or the page's status (native chrome), else nothing.
    private var readyLine: (text: String, identifier: String)? {
        if !isTowerReachable { return (WorldPanelCopy.readyButOffline, "wb-panel-footer-line") }
        // The inline subset's rule (U-INLINE §3.4): never under a message.
        guard let model = host.model(for: phase.target), model.chrome.isDrawingNative, let state = model.chrome.state,
              let status = state.status,
              WorldChromeCanvasLayer<EmptyView, EmptyView, EmptyView>.inlineElements(state).contains(.status)
        else { return nil }
        return (status, "world-chrome-status")
    }

    private func footerRow<Button: View>(line: (text: String, identifier: String)?,
                                         @ViewBuilder button: () -> Button) -> some View {
        let button = button()
        return ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 12) {
                footerLine(line)
                Spacer(minLength: 0)
                button
            }
            VStack(alignment: .leading, spacing: 8) {
                footerLine(line)
                button
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func footerLine(_ line: (text: String, identifier: String)?) -> some View {
        if let line {
            Text(verbatim: line.text)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.readableSecondary)
                .fixedSize(horizontal: false, vertical: true)
                // The page's status is the native chrome's pill, capped as
                // the pill is (U1.1 168 #3); the Tower sentence is not.
                .dynamicTypeSize(line.identifier == "world-chrome-status"
                                 ? DynamicTypeSize.xSmall...DynamicTypeSize.accessibility2
                                 : DynamicTypeSize.xSmall...DynamicTypeSize.accessibility5)
                .accessibilityShowsLargeContentViewer()
                .accessibilitySortPriority(70)
                .accessibilityIdentifier(line.identifier)
        }
    }

    private func actionButton(_ title: String, hint: String?, identifier: String,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.body.weight(.semibold))
                .frame(minWidth: 44, minHeight: 44)
        }
        .readableBorderedButton()
        .dynamicTypeSize(...DynamicTypeSize.accessibility1)
        .accessibilityShowsLargeContentViewer()
        .accessibilityInputLabels([title])
        .accessibilityHint(hint ?? "")
        .accessibilitySortPriority(60)
        .accessibilityIdentifier(identifier)
    }
}

// MARK: - The cover

/// The full-screen form of the panel's world: the SAME web view, moved here
/// (U-INLINE §3.2). An area opened from the room's areas row REPLACES the
/// room here (C1 E5): the room's web view is held off until *Back to the
/// room*, so one web view still holds.
struct WorldInlineCover: View {
    @ObservedObject var host: WorldInlineHost
    let title: String?
    let note: String?
    let notice: String?
    let progress: WorldViewerProgress?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private struct ShownArea: Equatable {
        let target: WorldRenderTarget
        let opening: WorldAreaOpening
    }

    @State private var area: ShownArea?

    var body: some View {
        NavigationStack {
            Group {
                if let area {
                    WorldRenderScene(
                        target: area.target, area: area.opening,
                        backToRoom: {
                            self.area = nil
                            host.setShowsArea(false)
                        })
                    .id(area.target.id)
                } else if let model = host.model {
                    WorldRenderScene(
                        model: model, hosted: true, webIsActive: host.placement == .expanded,
                        title: title, note: note, notice: notice, progress: progress,
                        openArea: { target, opening in
                            host.setShowsArea(true)
                            area = ShownArea(target: target, opening: opening)
                        })
                    .id(ObjectIdentifier(model))
                } else {
                    Color(WorldRenderLoadingPanel.pageBackground)
                        .ignoresSafeArea()
                }
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        var transaction = Transaction()
                        transaction.disablesAnimations = reduceMotion
                        withTransaction(transaction) { host.isExpanded = false }
                    }
                    .accessibilityIdentifier("wb-cover-close")
                }
            }
        }
        .onAppear { host.coverAppeared() }
        .onDisappear { host.coverDisappeared() }
    }
}
