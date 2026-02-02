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

    static func parse() -> CLIArgs {
        return parse(from: ProcessInfo.processInfo.arguments)
    }

    static func parse(from args: [String]) -> CLIArgs {
        var state: OverlayState?
        var testCycle = false

        if let idx = args.firstIndex(of: "--state"), idx + 1 < args.count {
            state = OverlayState(rawValue: args[idx + 1].lowercased())
        }
        if args.contains("--test-cycle") {
            testCycle = true
        }

        return CLIArgs(state: state, testCycle: testCycle)
    }
}

final class OverlayViewModel: ObservableObject {
    @Published var state: OverlayState = .idle
    @Published var transcription: String = "Ready"

    private var timer: Timer?

    func apply(args: CLIArgs) {
        if let state = args.state {
            self.state = state
        }
        if args.testCycle {
            startTestCycle()
        }
    }

    private func startTestCycle() {
        let states = OverlayState.allCases
        var idx = 0
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.state = states[idx % states.count]
            self.transcription = "State: \(self.state.rawValue)"
            idx += 1
        }
    }
}
