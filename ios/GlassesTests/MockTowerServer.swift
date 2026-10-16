//
//  MockTowerServer.swift
//  GlassesTests
//
//  Minimal local WebSocket server used to exercise TowerClient's receive
//  loop, delegate close handling, and send-failure paths against a real
//  socket, without depending on the actual Tower. Built directly on
//  Network.framework's NWProtocolWebSocket (Apple-native, matching the
//  client's use of URLSessionWebSocketTask — no third-party WebSocket
//  library on either side of these tests).
//

import Foundation
import Network

final class MockTowerServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "MockTowerServer")
    private var connection: NWConnection?

    /// Invoked (on an arbitrary queue) for every text message the connected
    /// client sends.
    var onText: (@Sendable (String) -> Void)?

    init() throws {
        let parameters = NWParameters(tls: nil)
        parameters.allowLocalEndpointReuse = true
        let wsOptions = NWProtocolWebSocket.Options()
        wsOptions.autoReplyPing = true
        parameters.defaultProtocolStack.applicationProtocols.insert(wsOptions, at: 0)
        listener = try NWListener(using: parameters)
    }

    /// Starts listening and resolves once the listener is ready, returning
    /// the system-assigned port.
    func start() async throws -> UInt16 {
        try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { [weak self] state in
                switch state {
                case .ready:
                    if let port = self?.listener.port {
                        continuation.resume(returning: port.rawValue)
                    }
                case .failed(let error):
                    continuation.resume(throwing: error)
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] newConnection in
                self?.accept(newConnection)
            }
            listener.start(queue: queue)
        }
    }

    private func accept(_ newConnection: NWConnection) {
        connection = newConnection
        newConnection.start(queue: queue)
        receiveNext(on: newConnection)
    }

    private func receiveNext(on connection: NWConnection) {
        connection.receiveMessage { [weak self] data, context, _, error in
            guard let self else { return }
            if let data, let context, !data.isEmpty, self.isText(context) {
                let text = String(data: data, encoding: .utf8) ?? ""
                self.onText?(text)
            }
            if error == nil {
                self.receiveNext(on: connection)
            }
        }
    }

    private func isText(_ context: NWConnection.ContentContext) -> Bool {
        guard
            let metadata = context.protocolMetadata(definition: NWProtocolWebSocket.definition)
                as? NWProtocolWebSocket.Metadata
        else {
            return false
        }
        return metadata.opcode == .text
    }

    /// Sends one WebSocket text frame to the connected client.
    func send(text: String) {
        guard let connection else { return }
        let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
        let context = NWConnection.ContentContext(identifier: "text", metadata: [metadata])
        connection.send(
            content: text.data(using: .utf8),
            contentContext: context,
            isComplete: true,
            completion: .contentProcessed { _ in }
        )
    }

    /// Sends one WebSocket binary frame to the connected client.
    func send(data: Data) {
        guard let connection else { return }
        let metadata = NWProtocolWebSocket.Metadata(opcode: .binary)
        let context = NWConnection.ContentContext(identifier: "binary", metadata: [metadata])
        connection.send(
            content: data,
            contentContext: context,
            isComplete: true,
            completion: .contentProcessed { _ in }
        )
    }

    /// Abruptly drops the connection with no close handshake, to exercise
    /// the client's receive-failure / send-failure paths.
    func dropConnection() {
        connection?.forceCancel()
        connection = nil
    }

    func stop() {
        connection?.cancel()
        connection = nil
        listener.cancel()
    }
}
