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

    /// The picture the panel may pair with the report's stage: `target`
    /// only when it names the walk the report describes -- the same world
    /// AND the same session (review HIGH 1). A Saved worlds pin names its
    /// target before its own report arrives, and a new session of the same
    /// world reports before its coordinates do: either way the stage on
    /// screen is another walk's, and another walk's picture is never shown
    /// (or offered) under it.
    static func target(_ target: WorldRenderTarget?, matching walk: WorldFinishWalk?) -> WorldRenderTarget? {
        guard let target, let walk, target.worldID == walk.worldID, target.sessionID == walk.sessionID
        else { return nil }
        return target
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
/// stood, kept for the walk it was drawn for. A ready world shows it only
/// when it came straight from THAT walk's wait; a ready world replaced by
/// another forgets it, so B's wait is never A's finish block.
struct WorldPanelFinishMemory<Block: Equatable>: Equatable {
    private(set) var block: Block?
    /// The walk `block` was drawn for.
    private(set) var walk: WorldFinishWalk?
    /// The ready world that arrived straight from the finishing wait.
    private(set) var readyFromFinishing: WorldRenderTarget?

    /// The finish block as it stands now, for the walk the report describes.
    /// `nil` (not finishing) keeps what there is.
    mutating func record(_ block: Block?, walk: WorldFinishWalk?) {
        guard let block else { return }
        self.block = block
        self.walk = walk
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
            guard block != nil, let walk, walk.worldID == target.worldID, walk.sessionID == target.sessionID
            else {
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
            walk = nil
        }
        return false
    }

    /// The finish block the wait overlay over `target` shows, or `nil` for
    /// "Opening your world…".
    func overlay(for target: WorldRenderTarget) -> Block? {
        guard readyFromFinishing == target, let walk,
              walk.worldID == target.worldID, walk.sessionID == target.sessionID
        else { return nil }
        return block
    }
}

/// What the full-screen cover says over its picture.
struct WorldCoverText: Equatable {
    var title: String?
    var note: String?
    var notice: String?
    var progress: WorldViewerProgress?
}

/// The cover's words, bound to the walk of the picture it shows (review 3,
/// HIGH). The cover keeps its picture while it is open, whatever the report
/// moves on to, so its title, note, notice and progress are the report's
/// only while the report still describes that walk. Once it describes
/// another, the cover keeps the picture's own last words -- static, with no
/// live stage -- and says so; it never pairs B's words with A's picture.
/// Closed, the panel's own rules apply again.
struct WorldCoverBinding: Equatable {
    /// The walk the report described when the cover opened: the picture's
    /// (the panel's target always names it, `WorldPanelPhase.target`).
    private(set) var walk: WorldFinishWalk?
    /// That walk's own words as they last stood; `nil` while closed.
    private(set) var held: WorldCoverText?

    mutating func opened(walk: WorldFinishWalk?, words: WorldCoverText) {
        self.walk = walk
        held = words
    }

    /// The report now says `words` about `walk`: kept only while it is the
    /// picture's walk.
    mutating func reported(_ words: WorldCoverText, walk: WorldFinishWalk?) {
        guard held != nil, walk == self.walk else { return }
        held = words
    }

    mutating func closed() {
        self = WorldCoverBinding()
    }

    /// What the cover shows. `live`: the report's words now, about
    /// `presented`; `presentedInProgress`: that walk is still changing.
    func words(live: WorldCoverText, presented: WorldFinishWalk?, presentedInProgress: Bool,
               towerReachable: Bool) -> WorldCoverText {
        var words = live
        var moved: String?
        if let held, presented != walk {
            words = held
            // The picture's walk is no longer reported: its last stage is
            // not current.
            words.progress?.line = nil
            if presented != nil {
                moved = presentedInProgress ? WorldPanelCopy.coverNewWalk : WorldPanelCopy.coverOtherWalk
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
