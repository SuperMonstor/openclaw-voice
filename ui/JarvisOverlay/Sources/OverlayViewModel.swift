import Combine
import Foundation

enum OverlayState: String, CaseIterable {
    case idle
    case listening
    case transcribing
    case responding
    case speaking
}

struct CLIArgs {
    let state: OverlayState?
    let testCycle: Bool
    let socketPath: String

    static func parse() -> CLIArgs {
        return parse(from: ProcessInfo.processInfo.arguments)
    }

    static func parse(from args: [String]) -> CLIArgs {
        var state: OverlayState?
        var testCycle = false
        var socketPath = "/tmp/openclaw_voice.sock"

        if let idx = args.firstIndex(of: "--state"), idx + 1 < args.count {
            state = OverlayState(rawValue: args[idx + 1].lowercased())
        }
        if args.contains("--test-cycle") {
            testCycle = true
        }
        if let idx = args.firstIndex(of: "--socket"), idx + 1 < args.count {
            socketPath = args[idx + 1]
        }

        return CLIArgs(state: state, testCycle: testCycle, socketPath: socketPath)
    }
}

final class OverlayViewModel: ObservableObject {
    @Published var state: OverlayState = .idle
    @Published var liveTranscription: String = ""
    @Published var responseText: String = ""
    @Published var statusText: String = "Ready"

    private var timer: Timer?
    private var socket: VoiceEngineSocket?

    func apply(args: CLIArgs) {
        if let state = args.state {
            self.state = state
        }
        if args.testCycle {
            startTestCycle()
            return
        }
        connectSocket(path: args.socketPath)
    }

    private func startTestCycle() {
        let states = OverlayState.allCases
        var idx = 0
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.state = states[idx % states.count]
            self.statusText = "State: \(self.state.rawValue)"
            self.liveTranscription = self.state == .listening ? "Listening to you..." : ""
            self.responseText = self.state == .responding ? "Thinking in streaming chunks..." : ""
            idx += 1
        }
    }

    private func connectSocket(path: String) {
        socket?.stop()
        let socket = VoiceEngineSocket(path: path)
        socket.onEvent = { [weak self] message in
            self?.handleEvent(message)
        }
        self.socket = socket
        socket.start()
    }

    private func handleEvent(_ message: [String: Any]) {
        guard let type = message["type"] as? String, type == "event" else {
            return
        }
        guard let event = message["event"] as? String else {
            return
        }
        let payload = message["payload"] as? [String: Any] ?? [:]

        switch event {
        case "state_change":
            if let stateRaw = payload["state"] as? String,
               let state = OverlayState(rawValue: stateRaw.lowercased()) {
                self.state = state
                handleStateReset(for: state)
            }
        case "transcription":
            if let text = payload["text"] as? String {
                applyTextUpdate(text, payload: payload, target: &liveTranscription)
            }
        case "response_chunk":
            if let text = payload["text"] as? String {
                applyTextUpdate(text, payload: payload, target: &responseText)
            }
        default:
            break
        }
    }

    private func handleStateReset(for state: OverlayState) {
        switch state {
        case .idle:
            statusText = "Ready"
            liveTranscription = ""
            responseText = ""
        case .listening:
            statusText = "Listening"
            liveTranscription = ""
            responseText = ""
        case .transcribing:
            statusText = "Transcribing"
            responseText = ""
        case .responding:
            statusText = "Responding"
        case .speaking:
            statusText = "Speaking"
        }
    }

    private func applyTextUpdate(_ text: String, payload: [String: Any], target: inout String) {
        if let append = payload["append"] as? Bool, append {
            target += text
        } else {
            target = text
        }
    }
}
