//
//  MockTowerHTTPServer.swift
//  GlassesUITests
//
//  The HTTP half of a Tower, as far as Settings' "Test connection" can see it,
//  served from the UI-test process on the Mac's loopback — which the app in
//  the Simulator shares. Built like `GlassesTests/MockTowerServer.swift` (the
//  WebSocket mock), on Network.framework's `NWListener`, and for the same
//  reason: no third-party server on either side of the test. That one speaks
//  only WebSocket frames, which a plain `GET` never reaches, so this is its
//  sibling rather than a reuse.
//
//  It records every request line it is sent, which is how a test proves the
//  app sent exactly one `GET /cartridges` and nothing that could change a
//  Tower.
//
//  U0.8 (H0, H1): a route table in front of the personality, and -- only when
//  `acceptsWebSocket` is set -- the Tower's socket on `/ws` of the same
//  authority, since the app dials `ws://<authority>/ws` beside its HTTP. One
//  listener serves both. The socket half is a hand-rolled RFC 6455 server
//  (handshake, masked client frames, unmasked server frames), so a test can
//  script the Tower's side of the conversation and drop it at will.
//

import CryptoKit
import Foundation
import Network

final class MockTowerHTTPServer: @unchecked Sendable {
    enum Personality {
        /// Answers `GET /cartridges` with a Tower's declaration; 404 otherwise.
        case tower
        /// Answers everything 404, like a web server that is not a Tower.
        case notATower
    }

    static let contract = "cartridge_results.envelope/2026-08-23"
    private static let declaration = """
        {"type":"cartridges","envelope_contract":"\(contract)","cartridges":[],"not_offered":[],"http_contracts":[]}
        """

    private let listener: NWListener
    private let personality: Personality
    private let queue = DispatchQueue(label: "MockTowerHTTPServer")
    private var lines: [String] = []
    private var startFailure: Error?

    // Everything below is read and written on `queue`.

    /// `"GET /worlds"` (the method, then the path without its query) →
    /// the answer. Looked up before the personality.
    private var routes: [String: (status: Int, body: String)] = [:]
    private var acceptsWebSocketValue = false
    private var refusesSocketsValue = false
    private var onSocketTextValue: ((String) -> Void)?
    private var texts: [String] = []
    /// The socket the app holds now; a reconnect replaces it.
    private var socket: NWConnection?

    init(_ personality: Personality) throws {
        self.personality = personality
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        // Loopback only: nothing off this Mac can reach the mock.
        parameters.requiredInterfaceType = .loopback
        listener = try NWListener(using: parameters)
    }

    /// Starts listening and returns the system-assigned port.
    func start(timeout: TimeInterval = 5) throws -> UInt16 {
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready: ready.signal()
            case .failed(let error): self?.startFailure = error; ready.signal()
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.start(queue: queue)
        guard ready.wait(timeout: .now() + timeout) == .success else {
            throw NSError(domain: "MockTowerHTTPServer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "the listener never became ready"])
        }
        if let failure = queue.sync(execute: { startFailure }) { throw failure }
        guard let port = listener.port?.rawValue else {
            throw NSError(domain: "MockTowerHTTPServer", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "the listener has no port"])
        }
        return port
    }

    /// Every request line received so far, like `GET /cartridges HTTP/1.1`.
    var requestLines: [String] { queue.sync { lines } }

    func stop() {
        queue.sync {
            socket?.forceCancel()
            socket = nil
        }
        listener.cancel()
    }

    // MARK: Routes (H0)

    /// Answers `methodAndPath` -- `"GET /worlds"`, the path without its
    /// query -- with `status` and `body`, ahead of the personality. Setting
    /// it again replaces the answer for the next request.
    func setRoute(_ methodAndPath: String, status: Int, body: String) {
        queue.sync { routes[methodAndPath] = (status, body) }
    }

    private static func statusText(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 404: return "Not Found"
        case 409: return "Conflict"
        case 500: return "Internal Server Error"
        case 503: return "Service Unavailable"
        default: return "Status"
        }
    }

    // MARK: The socket (H1)

    /// Off by default: `/ws` then answers 404, as it always did, so a test
    /// that never asks for the socket is untouched.
    var acceptsWebSocket: Bool {
        get { queue.sync { acceptsWebSocketValue } }
        set { queue.sync { acceptsWebSocketValue = newValue } }
    }

    /// While true, `/ws` answers 503, so a reconnect fails.
    var refusesSockets: Bool {
        get { queue.sync { refusesSocketsValue } }
        set { queue.sync { refusesSocketsValue = newValue } }
    }

    /// Every text frame the app sends, called on the mock's own queue. It may
    /// call `sendSocket(text:)`, which never waits on that queue, and must not
    /// read any other property of the mock.
    var onSocketText: ((String) -> Void)? {
        get { queue.sync { onSocketTextValue } }
        set { queue.sync { onSocketTextValue = newValue } }
    }

    /// Every text frame the app has sent, in order.
    var socketTexts: [String] { queue.sync { texts } }

    /// One text frame to the app's current socket, if it holds one.
    func sendSocket(text: String) {
        queue.async { [weak self] in
            guard let self, let socket = self.socket else { return }
            socket.send(content: Self.frame(opcode: 0x1, payload: Data(text.utf8)),
                        completion: .contentProcessed { _ in })
        }
    }

    /// Force-cancels the app's socket, as a Tower that went away would.
    func dropSocket() {
        queue.sync {
            socket?.forceCancel()
            socket = nil
        }
    }

    // MARK: Connections

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(on: connection, buffered: Data())
    }

    private func receive(on connection: NWConnection, buffered: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffered = buffered
            if let data { buffered.append(data) }
            if let end = buffered.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffered[..<end.lowerBound], as: UTF8.self)
                let rest = Data(buffered[end.upperBound...])
                let headLines = head.components(separatedBy: "\r\n")
                let headers = Self.headers(headLines.dropFirst())
                // A body is read to its end before the answer goes out, so
                // closing the connection never discards bytes the app is
                // still sending (a reset instead of an answer).
                let length = Int(headers["content-length"] ?? "") ?? 0
                if rest.count < length, error == nil, !isComplete {
                    self.receive(on: connection, buffered: buffered)
                    return
                }
                let line = headLines.first ?? ""
                self.lines.append(line)
                if self.upgrade(connection, line: line, headers: headers, rest: rest) { return }
                self.respond(on: connection, to: line)
            } else if error == nil, !isComplete {
                self.receive(on: connection, buffered: buffered)
            } else {
                connection.cancel()
            }
        }
    }

    private static func headers(_ lines: ArraySlice<String>) -> [String: String] {
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        return headers
    }

    /// `"GET /ws"` from a request line, the query dropped.
    private static func methodAndPath(_ line: String) -> String {
        let parts = line.split(separator: " ")
        guard parts.count >= 2 else { return line }
        let path = parts[1].split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false).first ?? ""
        return "\(parts[0]) \(path)"
    }

    private func respond(on connection: NWConnection, to line: String) {
        let status: Int
        let body: String
        if let route = routes[Self.methodAndPath(line)] {
            (status, body) = route
        } else if personality == .tower && line.hasPrefix("GET /cartridges ") {
            (status, body) = (200, Self.declaration)
        } else {
            (status, body) = (404, #"{"detail":"Not Found"}"#)
        }
        let response = "HTTP/1.1 \(status) \(Self.statusText(status))\r\nContent-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n\(body)"
        connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    /// Takes over a `GET /ws` upgrade when the socket is on: 503 while
    /// refusing, otherwise the 101 and frame mode. `false` leaves the request
    /// to the ordinary answer (404 unless a route says otherwise).
    private func upgrade(_ connection: NWConnection, line: String, headers: [String: String], rest: Data) -> Bool {
        guard Self.methodAndPath(line) == "GET /ws",
              headers["upgrade"]?.lowercased() == "websocket",
              let key = headers["sec-websocket-key"],
              acceptsWebSocketValue
        else { return false }
        if refusesSocketsValue {
            let body = #"{"detail":"refused by the test"}"#
            let response = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                + "Content-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n\(body)"
            connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in
                connection.cancel()
            })
            return true
        }
        let digest = Insecure.SHA1.hash(data: Data((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").utf8))
        let accept = Data(digest).base64EncodedString()
        let response = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            + "Sec-WebSocket-Accept: \(accept)\r\n\r\n"
        socket?.forceCancel()
        socket = connection
        connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in })
        readFrames(on: connection, buffered: rest, message: nil)
        return true
    }

    // MARK: Frames

    /// A message being reassembled from fragments: its opcode and payload.
    private typealias Partial = (opcode: UInt8, payload: Data)

    private func readFrames(on connection: NWConnection, buffered: Data, message: Partial?) {
        var buffered = buffered
        var message = message
        while case let (fin, opcode, payload, consumed)? = Self.parseFrame(buffered) {
            buffered.removeFirst(consumed)
            switch opcode {
            case 0x0:
                message?.payload.append(payload)
                if fin, let whole = message {
                    message = nil
                    deliver(opcode: whole.opcode, payload: whole.payload)
                }
            case 0x1, 0x2:
                if fin { deliver(opcode: opcode, payload: payload) } else { message = (opcode, payload) }
            case 0x8:
                connection.send(content: Data([0x88, 0x00]), completion: .contentProcessed { _ in
                    connection.cancel()
                })
                if socket === connection { socket = nil }
                return
            case 0x9:
                connection.send(content: Self.frame(opcode: 0xA, payload: payload),
                                completion: .contentProcessed { _ in })
            default:
                break
            }
        }
        let pending = buffered
        let partial = message
        connection.receive(minimumIncompleteLength: 1, maximumLength: 256 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            guard error == nil, !isComplete || data != nil else {
                if self.socket === connection { self.socket = nil }
                connection.cancel()
                return
            }
            var next = pending
            if let data { next.append(data) }
            self.readFrames(on: connection, buffered: next, message: partial)
        }
    }

    private func deliver(opcode: UInt8, payload: Data) {
        // Binary frames (a camera's) are not the conversation a test scripts.
        guard opcode == 0x1 else { return }
        let text = String(decoding: payload, as: UTF8.self)
        texts.append(text)
        onSocketTextValue?(text)
    }

    /// One whole client frame from the front of `data`, unmasked, or `nil`
    /// until all of it has arrived.
    private static func parseFrame(_ data: Data) -> (Bool, UInt8, Data, Int)? {
        let bytes = [UInt8](data.prefix(14))
        guard bytes.count >= 2 else { return nil }
        let fin = bytes[0] & 0x80 != 0
        let opcode = bytes[0] & 0x0F
        let masked = bytes[1] & 0x80 != 0
        var length = Int(bytes[1] & 0x7F)
        var offset = 2
        if length == 126 {
            guard bytes.count >= 4 else { return nil }
            length = Int(bytes[2]) << 8 | Int(bytes[3])
            offset = 4
        } else if length == 127 {
            guard bytes.count >= 10 else { return nil }
            length = (2..<10).reduce(0) { $0 << 8 | Int(bytes[$1]) }
            offset = 10
        }
        var mask: [UInt8] = []
        if masked {
            guard bytes.count >= offset + 4 else { return nil }
            mask = Array(bytes[offset..<offset + 4])
            offset += 4
        }
        guard data.count >= offset + length else { return nil }
        let start = data.startIndex + offset
        var payload = [UInt8](data[start..<start + length])
        if masked {
            for index in payload.indices { payload[index] ^= mask[index % 4] }
        }
        return (fin, opcode, Data(payload), offset + length)
    }

    /// A server frame: final, unmasked.
    private static func frame(opcode: UInt8, payload: Data) -> Data {
        var frame = Data([0x80 | opcode])
        if payload.count < 126 {
            frame.append(UInt8(payload.count))
        } else if payload.count <= 0xFFFF {
            frame.append(126)
            frame.append(contentsOf: [UInt8(payload.count >> 8 & 0xFF), UInt8(payload.count & 0xFF)])
        } else {
            frame.append(127)
            frame.append(contentsOf: (0..<8).reversed().map { UInt8(payload.count >> ($0 * 8) & 0xFF) })
        }
        frame.append(payload)
        return frame
    }
}
