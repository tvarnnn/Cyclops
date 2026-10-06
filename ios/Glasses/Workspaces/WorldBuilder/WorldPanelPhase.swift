//
//  WorldPanelPhase.swift
//  Glasses
//
//  U-INLINE §2.1 and §4: the World Builder panel's one phase, and whether
//  its one web view may exist. Pure functions of what the screen already
//  knows; table-tested (P1, P2, P6).
//

import CoreGraphics
import SwiftUI

/// What the panel under the capture control is showing: the walk, the
/// finishing wait, the finished world, or why there is none.
enum WorldPanelPhase: Equatable {
    case hidden
    /// Capture health, under the fog-of-war map when there is one.
    case walking(hasMap: Bool)
    /// After Stop: the honest finishing progress. A target means a preview
    /// exists (U0.6 `isPreview`).
    case finishing(target: WorldRenderTarget?)
    /// The world, live, inline.
    case ready(WorldRenderTarget)
    /// `nil`: say the model's `failureMessage`.
    case failed(sentence: String?)
    case offline

    /// The rules of U-INLINE §2.1; the first match wins.
    static func phase(isCapturing: Bool, sessionActive: Bool, towerReachable: Bool, pageLoaded: Bool,
                      state: WorldModelState, stage: WorldStage?, target: WorldRenderTarget?,
                      needsRetrySentence: String?, hasMap: Bool) -> WorldPanelPhase {
        if isCapturing { return .walking(hasMap: hasMap) }
        if !towerReachable, !pageLoaded { return .offline }
        if case .receiving = state { return .walking(hasMap: hasMap) }
        if stage?.isStillChanging == true { return .finishing(target: target) }
        if let stage, stage == .saved || stage == .partial || stage == .interrupted, let target {
            return .ready(target)
        }
        if stage == .needsRetry { return .failed(sentence: needsRetrySentence) }
        if case .failed(let failure) = state, failure.kind == .towerReportedFailure {
            return .failed(sentence: needsRetrySentence ?? failure.message)
        }
        // Capture health alone, as today before Start: no walk, no map.
        if sessionActive { return .walking(hasMap: false) }
        return .hidden
    }

    /// The panel is narrating the finishing wait: the canvas's own spinner,
    /// "The Tower is finishing this world." and its fallback sentence are
    /// then not drawn (review 2, MED 1) -- one voice for the wait, and the
    /// panel's is the more specific.
    var narratesFinishing: Bool { isFinishing }

    /// The world this phase can show or open, if any.
    var target: WorldRenderTarget? {
        switch self {
        case .finishing(let target): return target
        case .ready(let target): return target
        case .hidden, .walking, .failed, .offline: return nil
        }
    }

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var isFinishing: Bool {
        if case .finishing = self { return true }
        return false
    }

    /// Rows 2-4 (heading, stage, footer) are drawn: every phase but hidden,
    /// and walking without a map.
    var drawsCard: Bool {
        switch self {
        case .hidden: return false
        case .walking(let hasMap): return hasMap
        case .finishing, .ready, .failed, .offline: return true
        }
    }
}

/// What the ready panel's wait overlay may show of the finishing wait
/// before it (U-INLINE §2.2; review HIGH 2): the finish block as it last
/// stood, kept WITH the walk it was drawn for (`WalkScoped`). A ready world
/// shows it only when it came straight from THAT walk's wait; a ready world
/// replaced by another forgets it, so B's wait is never A's finish block.
struct WorldPanelFinishMemory<Block: Equatable>: Equatable {
    /// The last finish block drawn, and its walk.
    private(set) var block: WalkScoped<Block>?
    /// The ready world that arrived straight from the finishing wait.
    private(set) var readyFromFinishing: WorldRenderTarget?

    /// The finish block as it stands now, with the walk it describes. A
    /// `nil` block (not finishing) keeps what there is.
    mutating func record(_ current: WalkScoped<Block?>) {
        guard let value = current.value else { return }
        block = WalkScoped(walk: current.walk, value: value)
    }

    /// The phase moved. `true` when the world arrived from ITS OWN finishing
    /// wait: the one moment the panel announces it. A world that replaced
    /// the wait without being its walk -- a pin to another world while
    /// finishing -- is not "finished" (review 3).
    @discardableResult
    mutating func phaseChanged(from old: WorldPanelPhase, to new: WorldPanelPhase) -> Bool {
        guard case .ready(let target) = new else {
            readyFromFinishing = nil
            return false
        }
        if old.isFinishing {
            guard let block, block.describes(target) else {
                readyFromFinishing = nil
                return false
            }
            readyFromFinishing = target
            return true
        }
        if case .ready(let previous) = old, previous != target {
            // Replaced: nothing of the first world's wait is kept.
            readyFromFinishing = nil
            block = nil
        }
        return false
    }

    /// The finish block the wait overlay over `target` shows, or `nil` for
    /// "Opening your world…".
    func overlay(for target: WorldRenderTarget) -> Block? {
        guard readyFromFinishing == target else { return nil }
        return block?.value(for: target)
    }
}

/// What the full-screen cover says over its picture.
struct WorldCoverText: Equatable {
    var title: String?
    var note: String?
    var notice: String?
    var progress: WorldViewerProgress?
    /// The walk these words describe is still changing. Not drawn: it picks
    /// the cover's notice when the report has moved to another walk.
    var inProgress = false
}

/// The cover's words, bound to the walk of the picture it shows (review 3,
/// HIGH; review 4). The cover keeps its picture while it is open, whatever
/// the report moves on to, and its title, note, notice and progress are the
/// report's only while the report describes THAT picture's walk -- asked of
/// the scoped words themselves (`WalkScoped.value(for:)`), so words that
/// arrive with another walk can never be taken for the picture's. Otherwise
/// the cover keeps the picture's own last words -- static, with no live
/// stage -- and says so; or, when the picture's walk has not been reported
/// since the cover opened, says nothing of any walk. Closed, it says nothing.
struct WorldCoverBinding: Equatable {
    /// The picture the cover shows, from expand to close.
    private(set) var picture: WorldRenderTarget?
    /// The picture's own walk's words as they last stood, or `nil` when the
    /// report has not described that walk since the cover opened.
    private(set) var held: WorldCoverText?

    mutating func opened(picture: WorldRenderTarget, live: WalkScoped<WorldCoverText>) {
        self.picture = picture
        held = live.value(for: picture)
    }

    /// The report now says `live`: kept only when it is about the picture's
    /// walk.
    mutating func reported(_ live: WalkScoped<WorldCoverText>) {
        guard let picture, let words = live.value(for: picture) else { return }
        held = words
    }

    mutating func closed() {
        self = WorldCoverBinding()
    }

    /// What the cover shows over its picture, given what the report says
    /// now (`live`, with the walk it is about).
    func words(live: WalkScoped<WorldCoverText>, towerReachable: Bool) -> WorldCoverText {
        guard let picture else { return WorldCoverText() }
        var words: WorldCoverText
        var moved: String?
        if let own = live.value(for: picture) {
            words = own
        } else {
            words = held ?? WorldCoverText()
            // The picture's walk is not what is reported: its last stage is
            // not current.
            words.progress?.line = nil
            if held != nil, live.walk != nil {
                moved = live.value.inProgress ? WorldPanelCopy.coverNewWalk : WorldPanelCopy.coverOtherWalk
            }
        }
        let notice = [moved, words.notice].compactMap { $0 }.joined(separator: " ")
        words.notice = WorldPanelCopy.coverNotice(notice.isEmpty ? nil : notice, towerReachable: towerReachable)
        words.progress = words.progress?.reachable(towerReachable)
        return words
    }
}

/// U-INLINE §4: the one `WKWebView` exists only while it is wanted.
enum WorldInlineWebPolicy {
    /// Expanded, or ready and on screen with nothing over it. Never in the
    /// background, expanded or not; `.inactive` (a notification shade, the
    /// app switcher's first frame) keeps what there is.
    static func wantsWebView(isReady: Bool, isExpanded: Bool, panelVisible: Bool,
                             scenePhase: ScenePhase, pickerShown: Bool) -> Bool {
        guard scenePhase != .background else { return false }
        return isExpanded || (isReady && panelVisible && !pickerShown)
    }
}

/// VoiceOver's order inside the ready stage (U-INLINE §5; review 2, MED 2).
enum WorldPanelOrder {
    /// The 3D world: read before the inline chrome drawn over it
    /// (`WorldChromeInlineOrder`).
    static let world: Double = 70
}

/// The panel's words that are not the stage's (U-INLINE §2.2, §5).
enum WorldPanelCopy {
    static let fullScreen = "Full screen"
    static let fullScreenHint = "Opens the 3D world full screen."
    static let tryAgain = "Try again"
    static let opening = "Opening your world…"
    static let openingLabel = "Opening your world"
    static let notConnected = "The Tower is not connected."
    static let readyButOffline = "The Tower is not connected. This picture will not change until it is."
    /// The finishing wait when the Tower drops under an open preview: its
    /// last stage and spinner are no longer current (review MED 3).
    static let finishingButOffline = "The Tower is not connected. Its progress shows again once it is."
    static let mapFixture = "Fixture map"
    /// Over the full-screen picture once the report has moved to another
    /// walk (review 3, HIGH): the picture and its words stay as they were.
    static let coverNewWalk = "A new walk is in progress. This picture stays as it is until you close it."
    static let coverOtherWalk = "The Tower is now reporting another walk. This picture stays as it is until you close it."

    /// The offline stage's sentence, or `nil` when the screen already says
    /// it (the capture control's Tower line): then the panel shows only its
    /// own actions, never a third "not connected" line (the lead's ruling).
    static func offlineLine(screenSaysOffline: Bool) -> String? {
        screenSaysOffline ? nil : notConnected
    }

    /// The full-screen preview's notice: the walk's own, and -- with the
    /// Tower gone -- that the picture will not change until it is back.
    static func coverNotice(_ notice: String?, towerReachable: Bool) -> String? {
        guard !towerReachable else { return notice }
        return [readyButOffline, notice].compactMap { $0 }.joined(separator: " ")
    }

    /// Every sentence the panel itself writes, for the banned-phrase audit (P9).
    static var all: [String] {
        [fullScreen, fullScreenHint, tryAgain, opening, openingLabel, notConnected, readyButOffline,
         finishingButOffline, mapFixture, coverNewWalk, coverOtherWalk,
         WorldFinishCopy.finishing, WorldPreviewCopy.openPreview, WorldPreviewCopy.building,
         WorldPreviewCopy.finalComing, WorldPreviewCopy.finalShown, WorldPreviewCopy.walkTimePicture,
         WorldPreviewCopy.walkTimePreview, WorldFinishCopy.awayBanner(headline: "Saved"),
         WorldFinishCopy.finished(headline: "Saved")]
    }
}

/// The stage area's height: `min(round(W × 0.75), round(0.5 × B))`, where B
/// is the visible scroll height (U-INLINE §1.1). With no B yet, 3:4 of W.
enum WorldPanelLayout {
    static func stageHeight(width: CGFloat, visibleHeight: CGFloat) -> CGFloat {
        let byWidth = (width * 0.75).rounded()
        guard visibleHeight > 0 else { return byWidth }
        return min(byWidth, (visibleHeight * 0.5).rounded())
    }
}

extension WorldViewerProgress {
    /// With the Tower gone, the live stage line is no longer current: the
    /// preview stays a preview and the elapsed clock stays true, but no
    /// stage is claimed (review MED 3).
    func reachable(_ towerReachable: Bool) -> WorldViewerProgress {
        guard !towerReachable else { return self }
        var progress = self
        progress.line = nil
        return progress
    }
}
