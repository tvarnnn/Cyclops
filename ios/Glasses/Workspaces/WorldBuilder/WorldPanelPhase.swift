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

/// The panel's words that are not the stage's (U-INLINE §2.2, §5).
enum WorldPanelCopy {
    static let fullScreen = "Full screen"
    static let fullScreenHint = "Opens the 3D world full screen."
    static let tryAgain = "Try again"
    static let opening = "Opening your world…"
    static let openingLabel = "Opening your world"
    static let notConnected = "The Tower is not connected."
    static let readyButOffline = "The Tower is not connected. This picture will not change until it is."
    static let mapFixture = "Fixture map"

    /// Every sentence the panel itself writes, for the banned-phrase audit (P9).
    static var all: [String] {
        [fullScreen, fullScreenHint, tryAgain, opening, openingLabel, notConnected, readyButOffline, mapFixture,
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
