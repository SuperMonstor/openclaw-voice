import SwiftUI

struct RingView: View {
    let state: OverlayState

    private var color: Color {
        switch state {
        case .idle: return Color.white.opacity(0.2)
        case .listening: return Color.green
        case .transcribing: return Color.cyan
        case .responding: return Color.orange
        case .speaking: return Color.pink
        }
    }

    private var isActive: Bool {
        state != .idle
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.white.opacity(0.15), lineWidth: 4)

            Circle()
                .trim(from: 0.0, to: isActive ? 0.75 : 0.25)
                .stroke(color, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(isActive ? 360 : 0))
                .animation(
                    isActive ? .linear(duration: 1.2).repeatForever(autoreverses: false) : .default,
                    value: isActive
                )
        }
    }
}
