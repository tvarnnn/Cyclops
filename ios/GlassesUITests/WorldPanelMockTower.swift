//
//  WorldPanelMockTower.swift
//  GlassesUITests
//
//  U-INLINE / U0.6: the Tower's World Builder socket, scripted on
//  `MockTowerHTTPServer` as `DeadEndsUITests` does, plus the status reports
//  the panel follows -- live ones the test pushes, and the pinned world's
//  report sent straight after a Saved worlds pin is acknowledged, as the
//  Tower answers a pinned subscribe.
//

import Foundation

/// The socket half of a mock Tower: `ping` → `pong`; `cartridges` → World
/// Builder's declaration; `result_subscribe` → `result_subscribed` echoing
/// the pin, then (if one is set) a first report for it.
final class PanelSocket: @unchecked Sendable {
    private let lock = NSLock()
    private let mock: MockTowerHTTPServer
    private var count = 0
    private var live: String?
    private var pinnedReplyValue: ((_ subscription: String, _ world: String, _ session: String?) -> String?)?
    private var liveReplyValue: ((_ subscription: String) -> String?)?

    init(mock: MockTowerHTTPServer) {
        self.mock = mock
    }

    /// The report a pinned subscribe is answered with, after its ack.
    var pinnedReply: ((_ subscription: String, _ world: String, _ session: String?) -> String?)? {
        get { lock.withLock { pinnedReplyValue } }
        set { lock.withLock { pinnedReplyValue = newValue } }
    }

    /// The report an unpinned subscribe is answered with, after its ack.
    var liveReply: ((_ subscription: String) -> String?)? {
        get { lock.withLock { liveReplyValue } }
        set { lock.withLock { liveReplyValue = newValue } }
    }

    /// The id of the last unpinned subscription, or `nil`.
    var liveSubscription: String? { lock.withLock { live } }

    func install() {
        mock.acceptsWebSocket = true
        mock.onSocketText = { [weak self] text in self?.handle(text) }
    }

    /// One report to the live subscription, built for its id.
    func sendLive(_ make: (String) -> String) {
        guard let subscription = liveSubscription else { return }
        mock.sendSocket(text: make(subscription))
    }

    private func handle(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String
        else { return }
        switch type {
        case "ping":
            mock.sendSocket(text: #"{"type":"pong"}"#)
        case "cartridges":
            lock.withLock { count = 0; live = nil }
            mock.sendSocket(text: """
                {"type":"cartridges",
                 "envelope_contract":"cartridge_results.envelope/2026-08-23",
                 "cartridges":[{"cartridge":"world_builder","result_type":"status",
                    "contract":"\(DeadEndsUITests.worldBuilderContract)","available":true,
                    "unavailable_reason":null,"snapshot_only":true}],
                 "not_offered":[]}
                """)
        case "result_subscribe":
            let world = json["world_id"] as? String
            let session = json["session_id"] as? String
            let subscription: String = lock.withLock {
                count += 1
                let id = "sub-\(count)"
                if world == nil { live = id }
                return id
            }
            let worldJSON = world.map { "\"\($0)\"" } ?? "null"
            let sessionJSON = session.map { "\"\($0)\"" } ?? "null"
            mock.sendSocket(text: """
                {"type":"result_subscribed",
                 "envelope_contract":"cartridge_results.envelope/2026-08-23",
                 "subscription_id":"\(subscription)","cartridge":"world_builder",
                 "result_type":"status","contract":"\(DeadEndsUITests.worldBuilderContract)",
                 "snapshot_only":true,"world_id":\(worldJSON),"session_id":\(sessionJSON),
                 "cursor_status":"absent"}
                """)
            if let world {
                if let report = pinnedReply?(subscription, world, session) { mock.sendSocket(text: report) }
            } else if let report = liveReply?(subscription) {
                mock.sendSocket(text: report)
            }
        default:
            break
        }
    }
}

/// World Builder status reports as the Tower sends them, with `lifecycle`.
enum PanelReport {
    /// `elements` and `poses` of 0 is a world with nothing usable in it.
    static func text(subscription: String, seq: Int, modelState: String, world: String = "w1",
                     session: String = "s1", name: String = "Probe Room", elements: Int = 1360, poses: Int = 40,
                     reason: String? = nil, buildInProgress: Bool? = nil, finalizationState: String? = nil,
                     finalSolve: String? = nil, processing: String? = nil, photographic: String? = nil,
                     notice: String? = nil) -> String {
        var lifecycle: [String] = []
        if let buildInProgress { lifecycle.append(#""build_in_progress":\#(buildInProgress)"#) }
        if let finalizationState {
            let solve = finalSolve.map { "\"\($0)\"" } ?? "null"
            let noticeField = notice.map { #","notice":"\#($0)""# } ?? ""
            lifecycle.append(#""finalization":{"state":"\#(finalizationState)","final_solve":\#(solve)\#(noticeField)}"#)
        }
        if let photographic { lifecycle.append(#""photographic":\#(photographic)"#) }
        if let processing { lifecycle.append(#""processing":\#(processing)"#) }
        let lifecycleJSON = lifecycle.isEmpty ? "" : #","lifecycle":{\#(lifecycle.joined(separator: ","))}"#
        let reasonJSON = reason.map { "\"\($0)\"" } ?? "null"
        return """
            {"type":"cartridge_result",
             "envelope_contract":"cartridge_results.envelope/2026-08-23",
             "subscription_id":"\(subscription)","cartridge":"world_builder","result_type":"status",
             "contract":"\(DeadEndsUITests.worldBuilderContract)","seq":\(seq),"revision":"r\(seq)",
             "revision_changed":true,"coalesced":0,"cursor_status":null,
             "snapshot":true,"tower_sent_at":1787463092.9,"time_basis":"tower-receipt",
             "payload":{"model_state":"\(modelState)","model_state_reason":\(reasonJSON),
               "session":{"session_id":"\(session)","started_at":1788895000.0},
               "world_snapshot":{"name":"\(name)","world_id":"\(world)",
                 "keyframe_count":\(poses),"revision":"r\(seq)",
                 "tracking":"good","scale":"relative","mapping_seconds":12.5,"calibration":"calibrated",
                 "geometry":{"representation":"sparse point cloud","element_count":\(elements),
                             "is_incremental":false},
                 "trajectory":{"pose_count":\(poses),"path_length":2.85,
                               "path_length_unit":"world units","scale":"relative"},
                 "persistence":{"state":"saved","revision":"p1"}},
               "geometry":{"available":true,"current":true,"built_from_keyframes":\(poses),
                           "keyframes_now":\(poses),"revision":"g1"}\(lifecycleJSON)}}
            """
    }

    /// The saved fixture world `w1/s1` as a pinned subscribe is answered.
    static func savedWorld(subscription: String, world: String = "w1", session: String? = "s1",
                           notice: String? = nil) -> String {
        text(subscription: subscription, seq: 1, modelState: "finalized", world: world, session: session ?? "s1",
             name: "Appearance fixture (Mac)", finalizationState: "complete", finalSolve: "solved",
             photographic: #"{"state":"complete","stage":"appearance"}"#, notice: notice)
    }
}
