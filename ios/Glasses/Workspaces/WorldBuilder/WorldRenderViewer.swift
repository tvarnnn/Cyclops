//
// WorldRenderViewer.swift
// Glasses
//

import Foundation
import SwiftUI
import UIKit

// The screen composes the state in `WorldRenderViewerState.swift`, the fetch and
// lifecycle in `WorldRenderViewerModel.swift`, and the owned WebKit view in
// `WorldRenderWebView.swift`. The same web view spans rendering and ready.

// MARK: - The 3D world on screen

/// The 3D world itself, without a stack around it. **The primary way a saved
/// world is seen.**
///
/// It fetches the Tower's composed page as it appears, shows the Tower's own
/// sentence when there is none, and hands the page to a web view that can go
/// nowhere else.
///
/// ## Why the ids left the caption
///
/// The caption used to show world and session ids in a monospaced font, over
/// the sentence "Sparse structure-from-motion
/// output: feature points and camera poses." Both are true and neither belongs
/// on the first screen a person sees after opening a saved world: one is a
/// database key and the other is the name of an algorithm. The ids moved into
/// Details, at the bottom, and nothing was deleted.
///
/// Details holds **only** the ids. A segmented control for the two renderings
/// briefly lived there too and was removed: it refetched a multi-megabyte page
/// to change one token, and the Tower's page already carries an instant toggle
/// of its own. See `details`.
///
/// The caption that replaced them still refuses the one claim this screen must
/// not make — that a point cloud is a scan of the room — and does it in words
/// that are true of both of the Tower's renderings.
struct WorldRenderScene: View {
    @StateObject private var model: WorldRenderViewerModel

    /// Hosted by `WorldInlineHost`: the host runs the model's
    /// lifecycle and decides when it closes, so this screen does neither.
    private let hosted: Bool
    /// Whether this screen's container holds the web view now. Always for a
    /// screen of its own; for the cover, once it has appeared.
    private let webIsActive: Bool
    /// What the workspace says about the wait, or `nil` (an area).
    private let progress: WorldViewerProgress?
    /// This screen showed a preview at some point, so a settled world is
    /// worth a word about whether the final picture is on screen yet.
    @State private var openedOnPreview = false

    /// What to call this world on screen: the Tower's display name, or a dated
    /// title from the picker. `nil` falls back to "Saved world" — never to the
    /// id, which is what the id being in Details is for.
    private let title: String?
    /// The ladder's note for a world that is not final, or `nil`. Composed by
    /// `WorldReconstruction`, not here.
    private let note: String?

    /// Whether the Details disclosure is open. View state, and deliberately
    /// not remembered across presentations: Details is where a person goes on
    /// purpose, and a sheet that opens with it expanded is back to being a
    /// debugger.
    @State private var isShowingDetails = false

    /// The screen's height below the navigation bar, for `CappedScroll`.
    @State private var screenHeight: CGFloat = 0

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    /// The most of the screen the words above the picture may take before
    /// they scroll -- at the accessibility sizes only -- and the most the
    /// areas and Details below it may take.
    ///
    /// At the largest text size the caption grew without limit: the 3D view
    /// measured 0 pt on an iPhone SE, 72 on the 17e and 87 on the 17 Pro,
    /// and with Details open the caption was
    /// pushed up under the navigation bar. Capped, at the accessibility sizes
    /// the words take at most 18 % of the screen, the controls between them
    /// and the picture at most 10 % (`controlsShare`), and the
    /// areas and Details at most 12 % -- so the picture keeps at least 60 %
    /// of the screen below the bar with a control shown, and 70 % without
    /// one. Each part scrolls in its own space, whole, to read and to
    /// VoiceOver.
    ///
    /// **No cap below the accessibility sizes**. A cap of
    /// 18 % there was about 106 pt on an SE, and a room's caption with the
    /// Tower's notice measures more than that at the default size: the
    /// notice was pushed into a scroller with no cue, where COMPONENTS §8
    /// says to show it. Below the accessibility sizes the words take their
    /// own height, as they did before the cap; the areas and Details keep the
    /// room they had.
    private var captionShare: CGFloat? { dynamicTypeSize.isAccessibilitySize ? 0.18 : nil }
    /// The controls' cap, at the accessibility sizes only, as for the words.
    /// On an SE, 10 % is about one AX5 caption line and its
    /// padding; below the accessibility sizes they take their own height.
    private var controlsShare: CGFloat? { dynamicTypeSize.isAccessibilitySize ? 0.10 : nil }
    private var belowShare: CGFloat { dynamicTypeSize.isAccessibilitySize ? 0.12 : 0.3 }

    /// When the current wait began: the fetch (the screen's opening, a
    /// Reload, a Try again), or the drawing that follows it or a reload of
    /// the page. The loading panel counts each step from here.
    @State private var waitingSince = Date.now

    /// `finalization.notice` for the walk this room belongs to (COMPONENTS §8): what
    /// the Tower could not do and who can fix it, shown verbatim below the
    /// room caption as a plain note. `nil` shows nothing -- every older world,
    /// and every walk with nothing owed -- and an area viewer never shows it.
    private let notice: String?
    /// For an area viewer: its number, the walk's area count and its capture
    /// spans, from the list it was opened from. `nil` for the room.
    private let area: WorldAreaOpening?
    /// Opens one of this room's areas in place of the room. `nil`
    /// where the host cannot, and then the areas row offers no control.
    private let openArea: ((WorldRenderTarget, WorldAreaOpening) -> Void)?
    /// *Back to the room*, for an area viewer.
    private let backToRoom: (() -> Void)?

    /// The one panel open over the canvas under native chrome.
    /// VoiceOver focus for it is kept by the panel and its toggle
    /// (`WorldChromeCaptionPanel`, `WorldChromeToggle`), each bound to its own
    /// element, so no focus state here outlives what it points at.
    @State private var chromePanel: WorldChromePanelKind?

    /// Whether this screen is laid out for native chrome (WORLDS §4c).
    /// From the moment a page that echoes `wb-chrome=native` arrives -- so the
    /// first drawn frame does not resize the canvas -- until the screen falls
    /// back for good. Covers both `.pending` and `.native`; what is drawn
    /// over the canvas is the chrome model's to say.
    private var usesNativeChrome: Bool {
        guard let html = model.state.html, !model.chrome.refusedForScreen else { return false }
        return WorldChromeEcho.isOffered(in: html)
    }

    /// Why an area's picture stopped following, when the Tower says so (§5.2).
    static let areaNoLongerServedSentence = "The Tower no longer serves this area -- the walk may have been "
        + "finished again. The room's list of areas is current."

    init(
        target: WorldRenderTarget,
        title: String? = nil,
        note: String? = nil,
        client: WorldRenderClient = WorldRenderClient(),
        components: WorldComponents? = nil,
        notice: String? = nil,
        area: WorldAreaOpening? = nil,
        openArea: ((WorldRenderTarget, WorldAreaOpening) -> Void)? = nil,
        backToRoom: (() -> Void)? = nil,
        progress: WorldViewerProgress? = nil
    ) {
        _model = StateObject(wrappedValue: WorldRenderViewerModel(
            target: target, client: client, components: components))
        self.hosted = false
        self.webIsActive = true
        self.progress = target.isArea ? nil : progress
        self.title = title
        self.note = note
        // Guarded: machine output never reaches the wearer verbatim.
        self.notice = target.isArea ? nil : WorldNoticeGuard.displayText(notice)
        self.area = area
        self.openArea = openArea
        self.backToRoom = backToRoom
    }

    /// The screen over a model the World Builder panel's host owns: the same
    /// web view, moved here from the panel.
    init(
        model: WorldRenderViewerModel,
        hosted: Bool,
        webIsActive: Bool,
        title: String? = nil,
        note: String? = nil,
        notice: String? = nil,
        progress: WorldViewerProgress? = nil,
        openArea: ((WorldRenderTarget, WorldAreaOpening) -> Void)? = nil
    ) {
        _model = StateObject(wrappedValue: model)
        self.hosted = hosted
        self.webIsActive = webIsActive
        self.progress = model.target.isArea ? nil : progress
        self.title = title
        self.note = note
        self.notice = model.target.isArea ? nil : WorldNoticeGuard.displayText(notice)
        self.area = nil
        self.openArea = openArea
        self.backToRoom = nil
    }

    /// The area's header, or the host's title, or "Saved world".
    private var screenTitle: String {
        if model.target.isArea { return area?.header ?? "Area — not placed in the room" }
        return title ?? "Saved world"
    }

    /// Compose the viewer around one stable web view. Height caps and
    /// `layoutPriority` keep the picture visible; one `.task` (skipped when
    /// hosted) runs load, component refresh, then the revision follower.
    var body: some View {
        let native = usesNativeChrome
        VStack(spacing: 0) {
            if native {
                // The research marker (when raised), then one row. The
                // web view below is exactly the strip between this and the bar.
                WorldChromeTopBand(
                    chrome: model.chrome, isArea: model.target.isArea, note: note,
                    progress: progressLines(style: .band), notice: notice,
                    showsAreas: !(model.components?.areas.isEmpty ?? true),
                    wordsCap: captionShare.map { screenHeight * $0 } ?? 0,
                    backToRoom: model.target.isArea ? backToRoom : nil,
                    panel: $chromePanel
                )
                .tint(Color.readableTint)
                .environment(\.colorScheme, .dark)
            } else {
                // The research marker: after a fallback from native chrome,
                // or raised by the served header (IOS §10); otherwise this is
                // today's screen.
                if let marker = model.chrome.legacyResearchMarker {
                    WorldChromeResearchBand(marker: marker)
                        .environment(\.colorScheme, .dark)
                }
                // The words and the controls take the readable tint: system blue is about
                // 4.0:1 on white, under the 4.5:1
                // text needs, and "Details" and the offers are text. Not the
                // picture: the loading panel's spinner keeps its own colour.
                CappedScroll(cap: captionShare.map { screenHeight * $0 } ?? 0) {
                    caption
                }
                .tint(Color.readableTint)
                controls
                    .tint(Color.readableTint)
            }
            // The SAME view in every mode (slot 2 of this stack), so SwiftUI
            // keeps one `WKWebView` when the chrome's mode changes.
            content
                .layoutPriority(1)
            if native {
                WorldChromeBar(chrome: model.chrome) { model.bridge.receive(.tapped($0)) }
                    .environment(\.colorScheme, .dark)
            } else {
                CappedScroll(cap: screenHeight * belowShare) {
                    VStack(spacing: 0) {
                        if !model.target.isArea { areasRow }
                        details
                    }
                }
                .tint(Color.readableTint)
            }
        }
        // Written only when it moved by half a point or more. The caps above
        // are fractions of this height, and at AX5 a sub-pixel difference fed
        // back through them: the push of this screen re-ran its layout
        // without end, `_screenHeight changed` 23,030 times in 30 s, every
        // value printing as 708 pt.
        .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height in
            if MeasuredHeight.moved(from: screenHeight, to: height) { screenHeight = height }
        }
        .navigationTitle(screenTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if model.target.isArea, let area {
                ToolbarItem(placement: .principal) {
                    VStack(spacing: 0) {
                        Text(area.numberLine).font(.headline)
                        Text(WorldAreaOpening.notPlacedLine).font(.caption).foregroundStyle(.readableSecondary)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel(area.header)
                }
            }
        }
        // One fetch, for the life of the screen. `model.target` is a `let`, so
        // there is nothing for a `.task(id:)` to key on; a second rendering of
        // the same world is a second screen, and the page's own button switches
        // in place without fetching at all.
        .task {
            // A hosted model's lifecycle is the host's.
            guard !hosted else { return }
            await model.load()
            await model.refreshComponents()
            await model.followRevisions()
        }
        .onAppear { if progress?.isPreview == true { openedOnPreview = true } }
        .onChange(of: progress?.isPreview) { _, isPreview in
            if isPreview == true { openedOnPreview = true }
        }
        .onChange(of: model.state.isFetching) { _, isFetching in
            if isFetching { waitingSince = .now }
        }
        .onChange(of: model.state.isRendering) { _, isRendering in
            if isRendering { waitingSince = .now }
        }
        // `WORLD-BUILDER-IOS.md` §10: in-memory reuse within one open viewer is fine;
        // drop it when the viewer closes. Nothing did.
        .onDisappear { if !hosted { model.viewerClosed() } }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                // ## Why there is always a way to ask again
                //
                // Every other control on this screen is conditional on the
                // model's state, and the state that needed one most had none:
                // the page reports `didFinish`, so the screen is `.ready`, the
                // failure view and its "Try again" are not shown -- and the
                // page underneath is showing its own sentence, because it
                // placed no imagery, or its fetches were all refused, or it
                // boots into a world whose appearance is withdrawn. The wearer
                // was left with a message and a Close.
                //
                // This is not a second "Try again": it is the same `load()`,
                // reachable whatever the screen is showing. It is deliberately
                // plain and in the toolbar rather than under the caption, so it
                // is never mistaken for a claim about the picture.
                Button("Reload") { Task { await model.load() } }
                    .disabled(model.state.isFetching)
                    .accessibilityIdentifier("world-render-reload")
            }
        }
    }

    /// Under the ladder's note (IOS §3c), the live stage line and the
    /// elapsed line while the world is a preview, then what the picture on
    /// screen is known to be. `nil` when there is nothing to say.
    private func progressLines(style: WorldFinishLineView.Style) -> AnyView? {
        let finish = progress.map { $0.isPreview } ?? false
        let settle = WorldPreviewCopy.settleLine(
            isPreview: progress?.isPreview ?? false, walkEnded: progress?.walkEnded ?? false,
            openedOnPreview: openedOnPreview, basisOnScreen: model.basisOnScreen,
            noteSaysWalkTime: progress?.noteSaysWalkTime ?? false)
        guard finish || settle != nil else { return nil }
        return AnyView(VStack(alignment: .leading, spacing: 2) {
            if let progress, progress.isPreview {
                WorldFinishLineView(showsSpinner: false, line: progress.line, stoppedAt: progress.stoppedAt,
                                    style: style)
            }
            if let settle {
                Text(settle)
                    .font(.caption)
                    .foregroundStyle(style == .band ? AnyShapeStyle(WorldChromeStyle.secondary)
                                                    : AnyShapeStyle(.readableSecondary))
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("world-render-settle-line")
            }
        })
    }

    private var caption: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let note {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.readableSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let lines = progressLines(style: .viewer) { lines }
            // Follows the page. "Not a surface" was load-bearing while every
            // page was points; the Tower can now serve a surface, and a caption
            // that denies what is on screen is as wrong as one that overclaims.
            // Before the page arrives it claims neither.
            Text(nativeCaption)
                .font(.caption2)
                .foregroundStyle(.readableSecondary)
                .fixedSize(horizontal: false, vertical: true)
            // The walk's notice (COMPONENTS §8): verbatim, below the room caption, a
            // plain note and not an error -- no icon, no tint, no control.
            if let notice {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(.readableSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("world-render-notice")
            }
            // When the Tower has said this area will not be served (§5.2): why
            // the picture stopped following. Words, so they scroll with the
            // words; *Back to the room* is pinned below them.
            if model.target.isArea, model.areaNoLongerServed {
                Text(Self.areaNoLongerServedSentence)
                    .font(.caption)
                    .foregroundStyle(.readableSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    /// The controls that act on the picture, between the words and the
    /// picture and never inside the words' scroller: at the
    /// accessibility sizes the words scroll in their capped space, and a
    /// control at the end of them -- the offer, which comes last and late --
    /// started below the fold with nothing to say it was there.
    @ViewBuilder
    private var controls: some View {
        let back = model.target.isArea ? self.backToRoom : nil
        // One offer at a time: with both up, the controls alone
        // could take the SE's whole screen at AX5.
        let offer = WorldRenderOffer.current(
            newerPictureAvailable: model.newerPictureAvailable,
            newerPictureRefused: model.newerPictureRefused
        )
        if back != nil || offer != WorldRenderOffer.none {
            // Capped at the accessibility sizes, and scrolling beyond it
            // as the words above are.
            CappedScroll(cap: controlsShare.map { screenHeight * $0 } ?? 0) {
                VStack(alignment: .leading, spacing: 6) {
                    if let back {
                        Button("Back to the room") { back() }
                            .font(.caption)
                            .accessibilityIdentifier("world-render-back-to-room")
                    }
                    // A rebuild of the rung on screen is offered, never forced: the
                    // swap reloads the page and resets the camera the reader is
                    // using. A better rung replaces the picture without asking.
                    if offer == .newerPicture {
                        Button("A newer reconstruction is ready. Show it") {
                            Task { await model.showNewerPicture() }
                        }
                        .font(.caption)
                        .accessibilityIdentifier("world-render-newer-picture")
                    }
                    // After a refresh could not be drawn the old picture is back
                    // and the screen is `.ready`, so the failure view's "Try again"
                    // is not there. This is that control: `load()` forgets every
                    // refusal.
                    if offer == .retryRefused {
                        Button("A newer reconstruction could not be drawn on this phone. Try again") {
                            Task { await model.load() }
                        }
                        .font(.caption)
                        .accessibilityIdentifier("world-render-retry-refused")
                    }
                }
                .multilineTextAlignment(.leading)
                // Their whole height, never squeezed to a truncated line by the
                // picture's layout priority.
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
            }
        }
    }

    /// The caption line under the title. For an area: that it is an area, when
    /// it was captured, that it is not placed, and not to scale
    /// (`WORLD-BUILDER-COMPONENTS.md` §5.4). For the room: the rung's caption,
    /// plus ` · N more areas shown separately` when the walk has areas (§4).
    private var nativeCaption: String {
        if model.target.isArea {
            return WorldComponentsPresentation.areaCaption(
                representation: model.state.representation, spans: area?.spans ?? [],
                rawImagery: model.chrome.imageryRaw)
        }
        // The same source as the research marker: no redaction claimed once
        // the viewer knows the imagery is not redacted.
        return WorldRenderRepresentation.caption(for: model.state.representation,
                                                 rawImagery: model.chrome.imageryRaw)
            + (WorldComponentsPresentation.roomCaptionSuffix(model.components) ?? "")
    }

    /// Below the room: the walk's areas, and the stretches it can only count
    /// (§8). Nothing at all for `components: null` -- every older world -- and
    /// nothing for a walk the gate kept whole, so both look exactly as today.
    @ViewBuilder
    private var areasRow: some View {
        if let components = model.components, !components.isWhole {
            VStack(alignment: .leading, spacing: 6) {
                if let heading = WorldComponentsPresentation.areasHeading(components) {
                    Text(heading)
                        .font(.caption)
                        .foregroundStyle(.readableSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                let areas = components.areas
                ForEach(Array(areas.enumerated()), id: \.element.id) { index, area in
                    areaEntry(number: index + 1, of: areas.count, area: area)
                }
                if let footer = WorldComponentsPresentation.footer(components) {
                    Text(footer)
                        .font(.caption2)
                        .foregroundStyle(.readableSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .accessibilityIdentifier("world-render-areas")
        }
    }

    @ViewBuilder
    private func areaEntry(number: Int, of total: Int, area: WorldComponent) -> some View {
        let line = WorldComponentsPresentation.areaLine(number: number, area: area)
        let availability = WorldComponentsPresentation.availability(of: area)
        if availability == .opens, let openArea {
            Button {
                openArea(model.target.area(area.id),
                         WorldAreaOpening(number: number, total: total, spans: area.captureSpans))
            } label: {
                HStack {
                    Text(line).font(.footnote)
                    Spacer()
                    Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
        } else {
            HStack {
                Text(line).font(.footnote).foregroundStyle(.readableSecondary)
                Spacer()
                if let word = WorldComponentsPresentation.availabilityWord(availability) {
                    Text(word).font(.caption2).foregroundStyle(.readableSecondary)
                }
            }
        }
    }

    /// The page, the two waits, and the failure.
    ///
    /// ## Why this is not a `switch`
    ///
    /// `.rendering` and `.ready` hold the same page, and the web view must be
    /// the **same view** across both: separate arms of a `switch` inside a
    /// `ViewBuilder` are separate `_ConditionalContent` branches, so SwiftUI
    /// would destroy the `WKWebView` and build a new one at the instant the
    /// first one finished loading — discarding the render, and starting a
    /// second one that would finish and be discarded in turn. One `if let` over
    /// `state.html` keeps one web view for the life of the page.
    @ViewBuilder
    private var content: some View {
        if let html = model.state.html {
            WorldRenderWebView(owner: model.web, html: html,
                               attempt: model.renderAttempt, budgetToken: model.renderBudgetToken,
                               isActive: webIsActive, hosted: hosted,
                               onEvent: model.pageEvent)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                // The native chrome over the canvas. It never resizes
                // the web view; the rendering overlay stays outermost.
                .overlay { if usesNativeChrome { chromeCanvasLayer } }
                .modifier(WorldRenderCover(isRendering: model.state.isRendering,
                                           finished: model.pageFinishedToken,
                                           onGone: { model.renderingPanelGone() }) {
                    WorldRenderLoadingPanel(sentence: "Drawing the world…", step: 2, since: waitingSince)
                })
        } else if model.state.failureMessage != nil {
            failureView
        } else {
            WorldRenderLoadingPanel(sentence: "Fetching this world from the Tower…",
                                    step: 1, since: waitingSince)
        }
    }

    /// Everything the phone draws over the canvas under native chrome.
    private var chromeCanvasLayer: some View {
        WorldChromeCanvasLayer(
            chrome: model.chrome, panel: $chromePanel,
            onAction: { model.bridge.receive(.tapped($0)) },
            onFirstStateDrawn: { model.bridge.receive(.firstStateDrawn) },
            banner: { chromeBanner },
            areas: { areasRow },
            details: { details }
        )
        .tint(Color.readableTint)
        .environment(\.colorScheme, .dark)
    }

    /// The phone's own offers over the canvas, one at a time, and why an
    /// area stopped following. Floating: they never resize the canvas.
    @ViewBuilder
    private var chromeBanner: some View {
        let offer = WorldRenderOffer.current(
            newerPictureAvailable: model.newerPictureAvailable,
            newerPictureRefused: model.newerPictureRefused
        )
        let areaGone = model.target.isArea && model.areaNoLongerServed
        if offer != WorldRenderOffer.none || areaGone {
            VStack(alignment: .leading, spacing: 6) {
                if areaGone {
                    Text(Self.areaNoLongerServedSentence)
                        .font(.caption)
                        .foregroundStyle(WorldChromeStyle.text)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if offer == .newerPicture {
                    WorldChromeButton(word: "A newer reconstruction is ready. Show it",
                                      identifier: "world-render-newer-picture") {
                        Task { await model.showNewerPicture() }
                    }
                    .dynamicTypeSize(...DynamicTypeSize.accessibility1)
                }
                if offer == .retryRefused {
                    WorldChromeButton(word: "A newer reconstruction could not be drawn on this phone. Try again",
                                      identifier: "world-render-retry-refused") {
                        Task { await model.load() }
                    }
                    .dynamicTypeSize(...DynamicTypeSize.accessibility1)
                }
            }
            .padding(8)
            .background(RoundedRectangle(cornerRadius: 8).fill(WorldChromeStyle.pill))
        }
    }

    private var failureView: some View {
        VStack(spacing: 12) {
            Image(systemName: "cube.transparent")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            Text(model.state.failureMessage ?? "")
                .font(.footnote)
                .foregroundStyle(.readableSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
            if model.state.failureIsRetryable {
                Button("Try again") {
                    Task { await model.load() }
                }
                .readableBorderedButton()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The identifiers and the other rendering, both one tap away and neither
    /// on the way to anything.
    ///
    /// `.ignoresSafeArea` came off the web view above when this arrived: a
    /// disclosure underneath a view that draws into the home-indicator area
    /// would be drawn over by it.
    private var details: some View {
        DisclosureGroup("Details", isExpanded: $isShowingDetails) {
            VStack(alignment: .leading, spacing: 8) {
                Text((model.target.sessionID.map { "World \(model.target.worldID)\nSession \($0)" }
                      ?? "World \(model.target.worldID)\nNewest session with geometry")
                     + (model.target.areaID.map { "\nArea \($0)" } ?? ""))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.readableSecondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)

                // ## There is no rendering switch here, deliberately
                //
                // A segmented control stood here, bound to `model.show(_:)`.
                // It worked and it was the wrong control, for two independent
                // reasons.
                //
                // **It refetched the world to change one token.** Switching the
                // view changes `WorldRenderTarget.id`, which restarts
                // `.task(id:)`, which pulls the whole page again — several
                // megabytes of point coordinates over Tailscale — to flip a
                // single `let mode` in a script that is already on screen.
                //
                // **The page already has the control, and it is better.** The
                // Tower's page carries `<button id="mode">` and a `d` key
                // binding (`toggleMode()` in
                // `tower/world_builder/render.py`), both of which switch in
                // place with no request at all, reset the camera, and relabel
                // themselves. It is on screen, above this disclosure, in front
                // of the reader. Duplicating it worse, one level down, in a
                // collapsed section, is not an affordance — it is a second
                // way to do the same thing that costs a download.
                //
                // `WorldRenderView` and the `?view=diagnostics` parameter stay
                // and are still used: they choose which rendering a page OPENS
                // in, which is a fetch that was going to happen anyway. The
                // world screen's Diagnostics section uses that, so a reader who
                // wants the solver's view gets it without first loading the
                // one they did not ask for.
                //
                // Only the SPARSE page has that button. A surface or dense
                // page has none, and a sentence pointing "above" at a button
                // that is not there would be false copy. A
                // page with no declared rung is from a Tower older than the
                // ladder, whose only page was the sparse one.
                if model.state.representation == .sparse || model.state.representation == nil {
                    Text("The page's own Diagnostics button, above, switches to the solver's view of this session without fetching anything. The solver's geometry is also drawn in full under Diagnostics on the world screen.")
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("The solver's own view of this session is under Diagnostics on the world screen.")
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
        }
        .font(.footnote)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}

/// The viewer's wait, drawn in the page's own colours: the dark surface the
/// page will paint, so opening a world never flashes a white panel and then
/// cuts to black. It uses the page's background rather than pure black.
///
/// The two waits are worded for what is actually happening: fetching the page
/// from the Tower, then drawing it. (The first once read "Building the 3D
/// world…", which described neither half: the Tower composed the page before
/// this screen existed.) The panel says which of the two it is, as a step of
/// two, and how long that step has taken once it is more than a moment:
/// 6–11 s on the real Tower, with nothing moving but a spinner, read as stuck.
/// The "Drawing the world…" panel over the web view while the page has not
/// reported finishing, and the hand-over that follows it.
///
/// Opaque on purpose: a half-transparent spinner over an unpainted page
/// reads as a stuck page rather than as a page arriving. It disappears on
/// `didFinish`, and if that never comes the watchdog replaces the whole
/// thing with a sentence.
///
/// It also covers the native chrome, which is drawn under it. So
/// `onGone` -- the chrome's `pageFinished`, which may send the page
/// `activate` -- runs only after the commit that removed the panel: from a
/// task SwiftUI starts in that update, one main-actor turn later, once per
/// `finished` (the same yield as the chrome's first drawn state). The
/// `didFinish` callback that asked for the removal is too early: the panel
/// is still on screen in its turn (WORLDS §4c).
struct WorldRenderCover<Panel: View>: ViewModifier {
    let isRendering: Bool
    /// Bumped by each `didFinish` that leaves the page ready.
    let finished: Int
    let onGone: @MainActor () -> Void
    @ViewBuilder let panel: () -> Panel

    func body(content: Content) -> some View {
        content
            .overlay { if isRendering { panel() } }
            .task(id: finished) {
                guard finished > 0 else { return }
                await Task.yield()
                guard !Task.isCancelled else { return }
                onGone()
            }
    }
}

struct WorldRenderLoadingPanel: View {
    let sentence: String
    let step: Int
    let since: Date

    /// `--bg` in `appearance_viewer.html` and `surface_viewer.html`.
    static let pageBackground = UIColor(red: 0x0B / 255, green: 0x0D / 255, blue: 0x10 / 255, alpha: 1)

    var body: some View {
        TimelineView(.periodic(from: since, by: 1)) { context in
            let seconds = Int(context.date.timeIntervalSince(since))
            VStack(spacing: 12) {
                ProgressView()
                    .accessibilityHidden(true)
                // In the dark scheme forced below, `.primary` is near-white
                // (about 18:1 on the page background) and the readable
                // secondary is #AAAAB0 (8:1).
                Text(sentence)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.primary)
                Text(Self.progress(step: step, seconds: seconds))
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.readableSecondary)
            }
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 24)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityElement(children: .combine)
        }
        .background(Color(uiColor: Self.pageBackground))
        .environment(\.colorScheme, .dark)
        .accessibilityIdentifier("world-render-loading")
    }

    /// "Step 1 of 2", then "Step 1 of 2 · 3 s" from the second second on.
    static func progress(step: Int, seconds: Int) -> String {
        let steps = "Step \(step) of 2"
        return seconds >= 2 ? "\(steps) · \(seconds) s" : steps
    }
}

/// Which offer the viewer's controls show. Only one at a time: both
/// together could leave the picture no room at the largest text size.
enum WorldRenderOffer: Equatable {
    case none, newerPicture, retryRefused

    /// Showing the newer picture is itself a fresh attempt, so it wins. If it
    /// draws, the refusal is cleared; if it fails, the refusal is set again
    /// and the retry comes back.
    static func current(newerPictureAvailable: Bool, newerPictureRefused: Bool) -> WorldRenderOffer {
        if newerPictureAvailable { return .newerPicture }
        if newerPictureRefused { return .retryRefused }
        return .none
    }
}

/// Whether a measured height is written back: only when it moved by half a
/// point or more. The caps on the viewer are fractions of measured heights,
/// and at AX5 a sub-pixel difference fed back through them re-ran the layout
/// without end. What it ignores is under half a point (1.5 px at
/// 3x): a stored height is never further than that from the real one.
nonisolated enum MeasuredHeight {
    static let threshold: CGFloat = 0.5

    static func moved(from old: CGFloat, to new: CGFloat) -> Bool {
        abs(new - old) >= threshold
    }
}

/// Its content at its own height, up to `cap`, and scrolling beyond it.
///
/// A plain `ScrollView` takes all the height it is offered, which would take
/// the 3D view's; this one is exactly as tall as what it holds until that is
/// taller than `cap`. A `cap` of zero (no cap, or before the screen has been
/// measured) means no cap.
struct CappedScroll<Content: View>: View {
    let cap: CGFloat
    @ViewBuilder let content: Content

    @State private var contentHeight: CGFloat = 0
    /// How many times the content has become taller than its cap.
    @State private var overflows = 0

    private var isOverflowing: Bool { cap > 0 && contentHeight > cap }

    var body: some View {
        ScrollView {
            content
                // Half a point or more, for the reason `WorldRenderScene`
                // gives: a sub-pixel change must not feed back into a cap.
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height in
                    if MeasuredHeight.moved(from: contentHeight, to: height) { contentHeight = height }
                }
        }
        .scrollBounceBehavior(.basedOnSize)
        // When the words outgrow the cap, say so: the indicator shows once
        // that there is more to read. On the overflow itself, not as the
        // screen opens: the caption's rung and the areas
        // arrive with the Tower's answers, after the screen has appeared, and
        // a flash at the opening came before there was anything to scroll to.
        .scrollIndicatorsFlash(trigger: overflows)
        .onChange(of: isOverflowing) { _, isOverflowing in
            if isOverflowing { overflows += 1 }
        }
        .frame(height: cap > 0 ? min(contentHeight, cap) : contentHeight)
    }
}
