import Foundation

final class VoiceEngineSocket {
    private let path: String
    private var running = false
    private var thread: Thread?
    private var fd: Int32 = -1

    var onEvent: (([String: Any]) -> Void)?

    init(path: String) {
        self.path = path
    }

    func start() {
        guard !running else { return }
        running = true
        thread = Thread { [weak self] in
            self?.runLoop()
        }
        thread?.start()
    }

    func stop() {
        running = false
        closeSocket()
    }

    func sendCommand(_ command: String, payload: [String: Any] = [:]) {
        var message: [String: Any] = [
            "type": "command",
            "command": command
        ]
        if !payload.isEmpty {
            message["payload"] = payload
        }
        guard let data = try? JSONSerialization.data(withJSONObject: message) else {
            return
        }
        var packet = data
        packet.append(0x0A)

        if fd >= 0 {
            _ = packet.withUnsafeBytes { rawBuffer in
                guard let base = rawBuffer.baseAddress else { return 0 }
                return Darwin.write(fd, base, rawBuffer.count)
            }
            return
        }

        let tempFd = socket(AF_UNIX, SOCK_STREAM, 0)
        if tempFd < 0 {
            return
        }
        guard var addr = makeSocketAddress() else {
            _ = Darwin.close(tempFd)
            return
        }
        let addrLen = socklen_t(MemoryLayout.size(ofValue: addr))
        let result = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(tempFd, $0, addrLen)
            }
        }
        if result == 0 {
            _ = packet.withUnsafeBytes { rawBuffer in
                guard let base = rawBuffer.baseAddress else { return 0 }
                return Darwin.write(tempFd, base, rawBuffer.count)
            }
        }
        _ = Darwin.close(tempFd)
    }

    private func runLoop() {
        while running {
            if connectSocket() {
                readLoop()
            } else {
                Thread.sleep(forTimeInterval: 1.0)
            }
        }
    }

    private func connectSocket() -> Bool {
        closeSocket()
        fd = socket(AF_UNIX, SOCK_STREAM, 0)
        if fd < 0 {
            return false
        }
        guard var addr = makeSocketAddress() else {
            closeSocket()
            return false
        }
        let addrLen = socklen_t(MemoryLayout.size(ofValue: addr))
        let result = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, addrLen)
            }
        }
        if result != 0 {
            closeSocket()
            return false
        }
        return true
    }

    private func makeSocketAddress() -> sockaddr_un? {
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(path.utf8)
        if pathBytes.count >= MemoryLayout.size(ofValue: addr.sun_path) {
            return nil
        }
        withUnsafeMutableBytes(of: &addr.sun_path) { rawBuffer in
            rawBuffer.copyBytes(from: [UInt8](repeating: 0, count: rawBuffer.count))
            rawBuffer.copyBytes(from: pathBytes)
        }
        return addr
    }

    private func readLoop() {
        var buffer = Data()
        var readBuf = [UInt8](repeating: 0, count: 4096)
        while running {
            let count = read(fd, &readBuf, readBuf.count)
            if count <= 0 {
                break
            }
            buffer.append(readBuf, count: count)
            while let range = buffer.firstRange(of: Data([0x0A])) { // newline
                let line = buffer.subdata(in: 0..<range.lowerBound)
                buffer.removeSubrange(0...range.lowerBound)
                handleLine(line)
            }
        }
        closeSocket()
    }

    private func handleLine(_ line: Data) {
        guard !line.isEmpty else { return }
        guard let obj = try? JSONSerialization.jsonObject(with: line),
              let dict = obj as? [String: Any] else {
            return
        }
        DispatchQueue.main.async { [weak self] in
            self?.onEvent?(dict)
        }
    }

    private func closeSocket() {
        if fd >= 0 {
            _ = Darwin.close(fd)
            fd = -1
        }
    }
}
