import Foundation
import Network

final class VoiceEngineSocket {
    private let path: String
    private let queue = DispatchQueue(label: "voice.engine.socket")
    private var connection: NWConnection?
    private var reconnectTimer: DispatchSourceTimer?
    private var buffer = Data()

    var onEvent: (([String: Any]) -> Void)?

    init(path: String) {
        self.path = path
    }

    func start() {
        queue.async { [weak self] in
            self?.connect()
        }
    }

    func stop() {
        queue.async { [weak self] in
            self?.reconnectTimer?.cancel()
            self?.reconnectTimer = nil
            self?.connection?.stateUpdateHandler = nil
            self?.connection?.cancel()
            self?.connection = nil
            self?.buffer.removeAll(keepingCapacity: false)
        }
    }

    private func connect() {
        if connection != nil {
            return
        }

        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        let connection = NWConnection(to: .unix(path: path), using: params)
        self.connection = connection

        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.receive()
            case .failed, .cancelled:
                self.connection = nil
                self.scheduleReconnect()
            default:
                break
            }
        }

        connection.start(queue: queue)
    }

    private func scheduleReconnect() {
        if reconnectTimer != nil {
            return
        }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 1.0)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            self.reconnectTimer?.cancel()
            self.reconnectTimer = nil
            self.connect()
        }
        reconnectTimer = timer
        timer.resume()
    }

    private func receive() {
        connection?.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, isComplete, error in
            guard let self else { return }

            if let data, !data.isEmpty {
                self.buffer.append(data)
                self.processBuffer()
            }

            if error != nil || isComplete {
                self.connection?.cancel()
                self.connection = nil
                self.scheduleReconnect()
                return
            }

            self.receive()
        }
    }

    private func processBuffer() {
        while let range = buffer.firstRange(of: Data([0x0A])) {
            let line = buffer.subdata(in: 0..<range.lowerBound)
            buffer.removeSubrange(0...range.lowerBound)

            guard !line.isEmpty else { continue }
            guard let json = try? JSONSerialization.jsonObject(with: line),
                  let dict = json as? [String: Any] else {
                continue
            }
            DispatchQueue.main.async { [weak self] in
                self?.onEvent?(dict)
            }
        }
    }
}
