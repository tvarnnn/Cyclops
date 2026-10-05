//
//  WorldChromeOverlay.swift
//  Glasses
//
//  U1.1 native chrome (WORLD-BUILDER-WORLDS.md §4c; IOS §10): what the phone
//  draws in place of the appearance page's own chrome. A top band, the web
//  view as the canvas, and a bar; over the canvas, floating chrome that never
//  resizes it.
//
//  Every word drawn here for the page is the page's, from `hello.labels` and
//  `state`, shown with `Text(verbatim:)`. The few iOS words ("Areas", "Back to
//  the room", the offers) are the phone's own and are listed in
//  `WorldChromeStyle.iOSWords`. This file holds none of the page's words, and
//  `WorldChromeTests` scans it to keep it that way.
//
//  Hidden means absent: every conditional element is included or omitted by
//  `if`, never by opacity, so a hidden control neither hit-tests nor reaches
//  assistive technology.
//

import SwiftUI

// MARK: - Style

/// The page's dark palette, used for the chrome in every system appearance.
enum WorldChromeStyle {
    static func rgb(_ hex: UInt32, _ alpha: Double = 1) -> Color {
        Color(.sRGB, red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
              blue: Double(hex & 0xFF) / 255, opacity: alpha)
    }

    /// `#0B0D10`, the page's background.
    static let background = rgb(0x0B0D10)
    /// The band and the bar: the background at 0.96, or opaque under
    /// Increase Contrast.
    static func plate(_ contrast: ColorSchemeContrast) -> Color {
        rgb(0x0B0D10, contrast == .increased ? 1 : 0.96)
    }
    /// Pills and the banner: at least 0.90, so their text keeps 4.5:1 over
    /// a white frame.
    static let pill = rgb(0x0B0D10, 0.9)
    static let text = rgb(0xE8E9EC)
    static let secondary = rgb(0xAAAAB0)
    static let rawHead = rgb(0xFFB4A2)
    static let buttonFill = rgb(0x14171C)
    static let buttonBorder = rgb(0x23262C)
    static let panel = Color(.sRGB, red: 8 / 255, green: 10 / 255, blue: 13 / 255, opacity: 0.92)
    static let panelBorder = Color(.sRGB, red: 1, green: 1, blue: 1, opacity: 0.08)
    static let panelLine = rgb(0xAAB1BA)
    static let sectionTitle = rgb(0x8B939D)
    static let sectionBody = rgb(0xD3D7DD)
    static let researchText = rgb(0x2A0B06)
    static let researchBack = rgb(0xFFB4A2)
    static let researchRule = rgb(0xC96A52)
    static let chevron = rgb(0xE8E9EC, 0.72)

    /// The iOS-owned words the native chrome adds, for `WorldChromeTests`.
    static let areasWord = "Areas"
    static let backToRoomWord = "Back to the room"
    static var iOSWords: [String] { [areasWord, backToRoomWord] }

    static func colour(_ c: WorldChromeRingGeometry.Colour) -> Color {
        Color(.sRGB, red: c.red / 255, green: c.green / 255, blue: c.blue / 255, opacity: c.alpha)
    }
}

/// Which panel is open over the canvas: the page's caption, or the room's
/// areas. One at a time.
enum WorldChromePanelKind: Equatable {
    case caption, areas
}

/// Where VoiceOver focus goes when the caption panel opens and closes.
enum WorldChromeFocus: Hashable {
    case panel, toggle
}

// MARK: - The research band

/// APPEARANCE §6.6's marker, drawn by the phone above everything. Never
/// dismissible; sticky for the life of the viewer.
struct WorldChromeResearchBand: View {
    let marker: String

    var body: some View {
        Text(verbatim: marker.uppercased())
            .font(.caption2.weight(.semibold))
            .foregroundStyle(WorldChromeStyle.researchText)
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 4)
            .background(WorldChromeStyle.researchBack)
            .overlay(alignment: .bottom) {
                WorldChromeStyle.researchRule.frame(height: 1)
            }
            .dynamicTypeSize(...DynamicTypeSize.accessibility1)
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            .accessibilitySortPriority(100)
            .accessibilityIdentifier("world-chrome-research")
    }
}

// MARK: - Controls

/// One chrome control: the page's visible words, its accessible name, a
/// 44 pt target, and the large content viewer.
struct WorldChromeButton: View {
    let text: String
    let name: String
    var enabled = true
    let identifier: String
    let action: () -> Void

    init(_ label: WorldChromeLabel, enabled: Bool, identifier: String, action: @escaping () -> Void) {
        self.text = label.text
        self.name = label.name
        self.enabled = enabled
        self.identifier = identifier
        self.action = action
    }

    /// A control whose visible words are its name.
    init(word: String, identifier: String, action: @escaping () -> Void) {
        self.text = word
        self.name = word
        self.identifier = identifier
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            WorldChromeButtonFace(text: text)
        }
        .buttonStyle(.plain)
        .opacity(enabled ? 1 : 0.4)
        .disabled(!enabled)
        .accessibilityLabel(Text(verbatim: name))
        .accessibilityInputLabels([Text(verbatim: text), Text(verbatim: name)])
        .accessibilityShowsLargeContentViewer { Text(verbatim: text) }
        .accessibilityIdentifier(identifier)
    }
}

/// The drawn part of a control, also used (hidden) to reserve the bar.
struct WorldChromeButtonFace: View {
    let text: String

    var body: some View {
        Text(verbatim: text)
            .font(.footnote.weight(.semibold))
            .foregroundStyle(WorldChromeStyle.text)
            .padding(.horizontal, 10)
            .frame(minWidth: 44, minHeight: 44)
            .background(RoundedRectangle(cornerRadius: 8).fill(WorldChromeStyle.buttonFill))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(WorldChromeStyle.buttonBorder, lineWidth: 1))
            .contentShape(Rectangle())
    }
}

// MARK: - The top band

/// Above the canvas: the research marker (when raised), then one row -- the
/// words (room) or *Back to the room* (area), then *Areas* and the caption
/// toggle.
struct WorldChromeTopBand: View {
    @ObservedObject var chrome: WorldChromeModel
    let isArea: Bool
    let note: String?
    let notice: String?
    let showsAreas: Bool
    /// The most the words may take before they scroll; 0 is no cap.
    let wordsCap: CGFloat
    let backToRoom: (() -> Void)?
    @Binding var panel: WorldChromePanelKind?
    var focus: AccessibilityFocusState<WorldChromeFocus?>.Binding

    @Environment(\.colorSchemeContrast) private var contrast

    /// The page's head and toggle, only once there is something to see.
    private var head: (text: String, raw: Bool)? {
        guard chrome.isDrawingNative, let state = chrome.state, state.drawn, state.message == nil,
              state.phase != .failed, let caption = state.caption
        else { return nil }
        return (caption.head, state.research.raw)
    }

    var body: some View {
        VStack(spacing: 0) {
            if let marker = chrome.researchMarker {
                WorldChromeResearchBand(marker: marker)
            }
            HStack(alignment: .top, spacing: 8) {
                if isArea {
                    if let backToRoom {
                        WorldChromeButton(word: WorldChromeStyle.backToRoomWord,
                                          identifier: "world-render-back-to-room", action: backToRoom)
                            .dynamicTypeSize(...DynamicTypeSize.accessibility1)
                            .accessibilitySortPriority(87)
                    }
                    Spacer(minLength: 0)
                } else {
                    // A scroll view is an accessibility container: the words
                    // are ordered inside it, and it is ordered here (90-88).
                    CappedScroll(cap: wordsCap) { words }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilitySortPriority(90)
                }
                trailing
            }
            .frame(minHeight: 44, alignment: .top)
            .padding(.horizontal, 16)
            .padding(.vertical, 4)
        }
        .background(WorldChromeStyle.plate(contrast))
    }

    private var words: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let note {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(WorldChromeStyle.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilitySortPriority(90)
            }
            if let head {
                Text(verbatim: head.text)
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(head.raw ? WorldChromeStyle.rawHead : WorldChromeStyle.text)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilitySortPriority(89)
                    .accessibilityIdentifier("world-chrome-head")
            }
            if let notice {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(WorldChromeStyle.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilitySortPriority(88)
                    .accessibilityIdentifier("world-render-notice")
            }
        }
        .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
    }

    @ViewBuilder
    private var trailing: some View {
        HStack(spacing: 6) {
            if showsAreas, !isArea {
                WorldChromeButton(word: WorldChromeStyle.areasWord, identifier: "world-chrome-areas") {
                    panel = panel == .areas ? nil : .areas
                }
                .accessibilitySortPriority(86)
            }
            if chrome.isDrawingNative, toggleShown, let labels = chrome.hello?.labels {
                let open = panel == .caption
                WorldChromeButton(word: open ? labels.aboutClose : labels.aboutOpen,
                                  identifier: "world-chrome-about") {
                    if panel == .caption {
                        panel = nil
                        focus.wrappedValue = .toggle
                    } else {
                        panel = .caption
                        Task { @MainActor in
                            try? await Task.sleep(for: .milliseconds(150))
                            focus.wrappedValue = .panel
                        }
                    }
                }
                .accessibilityFocused(focus, equals: .toggle)
                .accessibilitySortPriority(85)
            }
        }
        .dynamicTypeSize(...DynamicTypeSize.accessibility1)
    }

    /// The toggle follows the head's rule, on the area viewer too, whose
    /// band has no words.
    private var toggleShown: Bool {
        guard let state = chrome.state else { return false }
        return state.drawn && state.message == nil && state.phase != .failed && state.caption != nil
    }
}

// MARK: - The bar

/// The five controls, from `hello.labels`. One row, or two when one does not
/// fit; the region is reserved in native geometry whether or not the
/// controls are shown, so the canvas does not move when they arrive.
struct WorldChromeBar: View {
    @ObservedObject var chrome: WorldChromeModel
    let onAction: (WorldChromeAction) -> Void

    @Environment(\.colorSchemeContrast) private var contrast

    private var shown: (labels: WorldChromeLabels, state: WorldChromeState)? {
        guard chrome.isDrawingNative, let labels = chrome.hello?.labels, let state = chrome.state,
              state.drawn, state.message == nil, state.phase != .failed
        else { return nil }
        return (labels, state)
    }

    var body: some View {
        Group {
            if let shown {
                rows(shown.labels, shown.state)
            } else {
                reserved
            }
        }
        .dynamicTypeSize(...DynamicTypeSize.accessibility1)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(WorldChromeStyle.plate(contrast).ignoresSafeArea(edges: .bottom))
    }

    private func rows(_ labels: WorldChromeLabels, _ state: WorldChromeState) -> some View {
        let best = control(labels.best, .best, state, "world-chrome-best", 40)
        let face = control(labels.face, .face, state, "world-chrome-face", 39)
        let previous = control(labels.previous, .previous, state, "world-chrome-previous", 38)
            .accessibilityValue(Text(verbatim: state.walk ?? ""))
        let next = control(labels.next, .next, state, "world-chrome-next", 37)
            .accessibilityValue(Text(verbatim: state.walk ?? ""))
        let reset = control(labels.reset, .reset, state, "world-chrome-reset", 36)
        return ViewThatFits(in: .horizontal) {
            HStack(spacing: 6) { best; face; previous; next; Spacer(minLength: 6); reset }
            VStack(spacing: 6) {
                HStack(spacing: 6) { best; face; Spacer(minLength: 0) }
                HStack(spacing: 6) { previous; next; Spacer(minLength: 6); reset }
            }
        }
    }

    private func control(_ label: WorldChromeLabel, _ action: WorldChromeAction, _ state: WorldChromeState,
                         _ identifier: String, _ priority: Double) -> some View {
        WorldChromeButton(label, enabled: state.buttons.isEnabled(action), identifier: identifier) {
            onAction(action)
        }
        .accessibilitySortPriority(priority)
    }

    /// The bar's height without its controls: the same faces, hidden, or one
    /// 44 pt row before the page has said what they are.
    @ViewBuilder
    private var reserved: some View {
        if let labels = chrome.hello?.labels {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 6) {
                    WorldChromeButtonFace(text: labels.best.text); WorldChromeButtonFace(text: labels.face.text)
                    WorldChromeButtonFace(text: labels.previous.text); WorldChromeButtonFace(text: labels.next.text)
                    Spacer(minLength: 6); WorldChromeButtonFace(text: labels.reset.text)
                }
                VStack(spacing: 6) {
                    HStack(spacing: 6) {
                        WorldChromeButtonFace(text: labels.best.text); WorldChromeButtonFace(text: labels.face.text)
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: 6) {
                        WorldChromeButtonFace(text: labels.previous.text); WorldChromeButtonFace(text: labels.next.text)
                        Spacer(minLength: 6); WorldChromeButtonFace(text: labels.reset.text)
                    }
                }
            }
            .hidden()
            .accessibilityHidden(true)
            .allowsHitTesting(false)
        } else {
            Color.clear.frame(height: 44).accessibilityHidden(true)
        }
    }
}

// MARK: - Over the canvas

/// Everything the phone draws over the web view. Hit testing is only on
/// content: empty space passes touches to the page's canvas. None of it
/// resizes the web view.
struct WorldChromeCanvasLayer<Banner: View, AreasPanel: View, Details: View>: View {
    @ObservedObject var chrome: WorldChromeModel
    @Binding var panel: WorldChromePanelKind?
    var focus: AccessibilityFocusState<WorldChromeFocus?>.Binding
    let onAction: (WorldChromeAction) -> Void
    let onFirstStateDrawn: () -> Void
    @ViewBuilder let banner: () -> Banner
    @ViewBuilder let areas: () -> AreasPanel
    @ViewBuilder let details: () -> Details

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                if chrome.isDrawingNative, let state = chrome.state, let labels = chrome.hello?.labels {
                    drawn(state, labels, canvas: geometry.size)
                }
                banner()
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .accessibilityElement(children: .contain)
                    .accessibilitySortPriority(80)
                if chrome.isDrawingNative, let state = chrome.state, let labels = chrome.hello?.labels,
                   Self.isFull(state), let panel {
                    panelView(panel, state, labels, canvas: geometry.size)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .topLeading)
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.25), value: animationKey)
        .task(id: chrome.firstStateToken) {
            guard chrome.firstStateToken > 0 else { return }
            // After SwiftUI has committed a frame with this state on screen
            // (and the research band, when raised).
            await Task.yield()
            guard !Task.isCancelled, chrome.isDrawingNative else { return }
            onFirstStateDrawn()
        }
        .onChange(of: chrome.hello?.pageID) { _, _ in panel = nil }
    }

    /// Something to see, and nothing in its way.
    static func isFull(_ state: WorldChromeState) -> Bool {
        state.drawn && !isBlocked(state)
    }

    /// A message, or a failed page: the message alone, and the iOS rows.
    static func isBlocked(_ state: WorldChromeState) -> Bool {
        state.message != nil || state.phase == .failed
    }

    private var animationKey: [String] {
        guard let state = chrome.state else { return [] }
        return [state.drawn ? "d" : "", state.dark ? "k" : "", state.hint?.text ?? "", state.edge?.rawValue ?? "",
                state.status ?? "", state.message ?? ""]
    }

    @ViewBuilder
    private func drawn(_ state: WorldChromeState, _ labels: WorldChromeLabels, canvas: CGSize) -> some View {
        let full = Self.isFull(state)
        let face = { if state.buttons.face { onAction(.face) } }
        if let message = state.message {
            WorldChromeMessageView(message: message)
        }
        if full, state.ring.shown, panel == nil {
            WorldChromeRingView(chrome: chrome, ring: state.ring, labels: labels, canFace: state.buttons.face,
                                face: face)
                .padding(.top, 8)
                .padding(.trailing, 10)
                .frame(maxWidth: .infinity, alignment: .topTrailing)
        }
        if full, let edge = state.edge {
            WorldChromeEdgeChevron(edge: edge, face: face)
                .frame(maxWidth: .infinity, maxHeight: .infinity,
                       alignment: edge == .left ? .leading : .trailing)
        }
        if !Self.isBlocked(state) {
            VStack(spacing: 6) {
                Spacer(minLength: 0)
                WorldChromeNotices(hint: state.hint, dark: full && state.dark ? labels : nil,
                                   canFace: state.buttons.face, face: face, canvasWidth: canvas.width)
                HStack(alignment: .bottom, spacing: 8) {
                    if let status = state.status {
                        WorldChromePill(text: status, font: .caption.monospacedDigit(),
                                        colour: WorldChromeStyle.secondary)
                            .dynamicTypeSize(...DynamicTypeSize.accessibility2)
                            .accessibilitySortPriority(50)
                            .accessibilityIdentifier("world-chrome-status")
                    }
                    Spacer(minLength: 0)
                    if full, let walk = state.walk {
                        WorldChromePosition(walk: walk)
                    }
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    @ViewBuilder
    private func panelView(_ kind: WorldChromePanelKind, _ state: WorldChromeState, _ labels: WorldChromeLabels,
                           canvas: CGSize) -> some View {
        let cap = dynamicTypeSize.isAccessibilitySize ? max(0, canvas.height - 16) : canvas.height * 0.44
        WorldChromePanel(cap: cap) {
            switch kind {
            case .caption:
                if let caption = state.caption {
                    WorldChromeCaptionPanel(caption: caption, focus: focus, details: details)
                }
            case .areas:
                areas()
            }
        }
        .padding(8)
        .accessibilitySortPriority(84)
    }
}

/// A small text on a plate of at least 0.90.
struct WorldChromePill: View {
    let text: String
    let font: Font
    let colour: Color

    var body: some View {
        Text(verbatim: text)
            .font(font)
            .foregroundStyle(colour)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 6).fill(WorldChromeStyle.pill))
    }
}

/// The walk position, bottom-trailing: the arrows' accessibility value, so
/// hidden from VoiceOver.
struct WorldChromePosition: View {
    let walk: String
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOver

    var body: some View {
        WorldChromePill(text: walk, font: .caption2.monospacedDigit(), colour: WorldChromeStyle.secondary)
            .dynamicTypeSize(...DynamicTypeSize.accessibility1)
            .accessibilityHidden(voiceOver)
            .accessibilityIdentifier("world-chrome-position")
    }
}

// MARK: - Notices

/// The hint and the dark line, stacked: they never overlap each other or the
/// status line, at any text size (the page lets them share one slot).
struct WorldChromeNotices: View {
    let hint: WorldChromeHint?
    /// The labels when the dark line is shown, else `nil`.
    let dark: WorldChromeLabels?
    let canFace: Bool
    let face: () -> Void
    let canvasWidth: CGFloat

    var body: some View {
        VStack(spacing: 6) {
            if let hint {
                WorldChromePill(text: hint.text, font: .caption, colour: WorldChromeStyle.text)
                    .opacity(max(0.4, hint.opacity))
                    .multilineTextAlignment(.center)
                    .dynamicTypeSize(...DynamicTypeSize.accessibility2)
                    .accessibilitySortPriority(55)
                    .accessibilityIdentifier("world-chrome-hint")
            }
            if let dark {
                Button(action: face) {
                    VStack(spacing: 2) {
                        Text(verbatim: dark.darkTitle).font(.footnote.weight(.semibold))
                            .foregroundStyle(WorldChromeStyle.text)
                        Text(verbatim: dark.darkTap).font(.caption2)
                            .foregroundStyle(WorldChromeStyle.secondary)
                    }
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .frame(minHeight: 44)
                    .frame(maxWidth: min(canvasWidth * 0.84, 360))
                    .background(RoundedRectangle(cornerRadius: 8).fill(WorldChromeStyle.pill))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(!canFace)
                .dynamicTypeSize(...DynamicTypeSize.accessibility1)
                .accessibilityLabel(Text(verbatim: dark.darkName))
                .accessibilityShowsLargeContentViewer {
                    Text(verbatim: dark.darkTitle + "\n" + dark.darkTap)
                }
                .accessibilitySortPriority(60)
                .accessibilityIdentifier("world-chrome-dark")
            }
        }
    }
}

// MARK: - The ring

/// The page's ring, drawn by its own rule (`WorldChromeRingGeometry`), with
/// its label under it. A tap turns the view, as the page's own ring does; its
/// name is the page's. (`WorldChromeRing` is the protocol's data.)
struct WorldChromeRingView: View {
    @ObservedObject var chrome: WorldChromeModel
    let ring: WorldChromeRing
    let labels: WorldChromeLabels
    let canFace: Bool
    let face: () -> Void

    var body: some View {
        VStack(spacing: 2) {
            WorldChromeRingCanvas(heading: chrome.heading, ring: ring, centre: labels.ringCenter)
                .frame(width: WorldChromeRingGeometry.size, height: WorldChromeRingGeometry.size)
                .contentShape(Circle())
                .onTapGesture { if canFace { face() } }
            Text(verbatim: labels.ringLabel)
                .font(.caption2)
                .foregroundStyle(WorldChromeStyle.text)
                .opacity(0.8)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 62)
                .dynamicTypeSize(...DynamicTypeSize.accessibility1)
                .allowsHitTesting(false)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: labels.ringName))
        .accessibilityAddTraits(.isImage)
        .accessibilityAction(named: Text(verbatim: labels.face.text)) { if canFace { face() } }
        .accessibilitySortPriority(70)
        .accessibilityIdentifier("world-chrome-ring")
    }
}

/// The drawing, observing only the per-frame heading.
struct WorldChromeRingCanvas: View {
    @ObservedObject var heading: WorldChromeHeadingModel
    let ring: WorldChromeRing
    let centre: String

    var body: some View {
        let view = heading.view
        Canvas { context, size in
            let g = WorldChromeRingGeometry.self
            let c = CGPoint(x: size.width / 2, y: size.height / 2)
            let headingRad = view?.headingRad ?? 0
            context.fill(Path(ellipseIn: CGRect(x: c.x - g.discRadius, y: c.y - g.discRadius,
                                                width: g.discRadius * 2, height: g.discRadius * 2)),
                         with: .color(WorldChromeStyle.colour(g.disc)))
            for i in 0..<WorldChromeLimits.ringBins {
                let arc = g.binArc(i, sense: ring.sense, headingRad: headingRad)
                var path = Path()
                path.addArc(center: c, radius: g.arcRadius, startAngle: .radians(arc.start),
                            endAngle: .radians(arc.end), clockwise: false)
                context.stroke(path, with: .color(WorldChromeStyle.colour(g.colour(support: g.support(ring.lit, i)))),
                               style: StrokeStyle(lineWidth: g.lineWidth, lineCap: .butt))
            }
            if let view {
                let spread = g.sectorArc(halfFovRad: view.halfFovRad)
                var sector = Path()
                sector.move(to: c)
                sector.addArc(center: c, radius: g.sectorRadius, startAngle: .radians(spread.start),
                              endAngle: .radians(spread.end), clockwise: false)
                sector.closeSubpath()
                context.fill(sector, with: .color(WorldChromeStyle.colour(g.sector)))
            }
            let points = g.pointerPoints(centre: c.x)
            var pointer = Path()
            pointer.move(to: CGPoint(x: points[0].x, y: points[0].y))
            pointer.addLine(to: CGPoint(x: points[1].x, y: points[1].y))
            pointer.addLine(to: CGPoint(x: points[2].x, y: points[2].y))
            pointer.closeSubpath()
            context.fill(pointer, with: .color(WorldChromeStyle.colour(g.pointer)))
            context.draw(Text(verbatim: centre).font(.system(size: 7, weight: .semibold))
                            .foregroundColor(WorldChromeStyle.colour(g.centre)),
                         at: c)
        }
        .accessibilityHidden(true)
    }
}

// MARK: - The edge chevron

/// The way back from a push past the edge: a triangle at the side the page
/// names. A duplicate of the face control, so hidden from VoiceOver.
struct WorldChromeEdgeChevron: View {
    let edge: WorldChromeEdge
    let face: () -> Void
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOver

    var body: some View {
        Button(action: face) {
            Path { path in
                // 26 pt tall, 17 pt deep, pointing at its edge.
                let tip: CGFloat = edge == .left ? 13.5 : 30.5
                let base: CGFloat = edge == .left ? 30.5 : 13.5
                path.move(to: CGPoint(x: tip, y: 32))
                path.addLine(to: CGPoint(x: base, y: 19))
                path.addLine(to: CGPoint(x: base, y: 45))
                path.closeSubpath()
            }
            .fill(WorldChromeStyle.chevron)
            .shadow(color: .black.opacity(0.6), radius: 3)
            .frame(width: 44, height: 64)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(edge == .left ? .leading : .trailing, 2)
        .accessibilityHidden(voiceOver)
        .accessibilityIdentifier("world-chrome-edge")
        .accessibilityValue(Text(verbatim: edge.rawValue))
    }
}

// MARK: - The message

/// The page's full-screen message, in place of the picture. It blocks
/// touches, as the page's does.
struct WorldChromeMessageView: View {
    let message: String

    var body: some View {
        ScrollView {
            Text(verbatim: message)
                .font(.callout)
                .foregroundStyle(WorldChromeStyle.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(28)
                .frame(maxWidth: .infinity)
                .accessibilityIdentifier("world-chrome-message")
        }
        .accessibilitySortPriority(78)
        .scrollBounceBehavior(.basedOnSize)
        .defaultScrollAnchor(.center)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(WorldChromeStyle.background)
        .contentShape(Rectangle())
    }
}

// MARK: - Panels

/// A panel below the canvas's top edge: the page's plate, a hairline, and its
/// content scrolling within `cap`.
struct WorldChromePanel<Content: View>: View {
    let cap: CGFloat
    @ViewBuilder let content: () -> Content

    var body: some View {
        CappedScroll(cap: cap) {
            content()
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(RoundedRectangle(cornerRadius: 8).fill(WorldChromeStyle.panel))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(WorldChromeStyle.panelBorder, lineWidth: 1))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("world-chrome-panel")
    }
}

/// The page's caption, whole: the head, the count line, the titled sections
/// and the tail; then the phone's own Details.
struct WorldChromeCaptionPanel<Details: View>: View {
    let caption: WorldChromeCaption
    var focus: AccessibilityFocusState<WorldChromeFocus?>.Binding
    @ViewBuilder let details: () -> Details

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(verbatim: caption.head)
                .font(.footnote.bold())
                .foregroundStyle(WorldChromeStyle.text)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityFocused(focus, equals: .panel)
            if let line = caption.line {
                Text(verbatim: line)
                    .font(.caption)
                    .foregroundStyle(WorldChromeStyle.panelLine)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(Array(caption.sections.enumerated()), id: \.offset) { _, section in
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: section.title.uppercased())
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(WorldChromeStyle.sectionTitle)
                        .accessibilityAddTraits(.isHeader)
                        .accessibilityLabel(Text(verbatim: section.title))
                    Text(verbatim: section.body)
                        .font(.footnote)
                        .foregroundStyle(WorldChromeStyle.sectionBody)
                }
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("world-chrome-section")
            }
            if let tail = caption.tail {
                Text(verbatim: tail)
                    .font(.caption)
                    .foregroundStyle(WorldChromeStyle.panelLine)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Divider().overlay(WorldChromeStyle.panelBorder)
            details()
        }
    }
}
