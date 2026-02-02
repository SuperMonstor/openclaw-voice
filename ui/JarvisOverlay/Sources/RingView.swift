import SwiftUI

struct RingView: View {
    let state: OverlayState

    private struct DotSpec: Identifiable {
        let id = UUID()
        let radius: CGFloat
        let size: CGFloat
        let speed: Double
        let phase: Double
        let opacity: Double
        let blur: CGFloat
    }

    private struct RGB {
        let r: Double
        let g: Double
        let b: Double
    }

    private let dots: [DotSpec] = [
        DotSpec(radius: 16, size: 3.5, speed: 0.9, phase: 0.0, opacity: 0.8, blur: 0.5),
        DotSpec(radius: 22, size: 2.5, speed: 1.3, phase: 1.2, opacity: 0.7, blur: 0.3),
        DotSpec(radius: 28, size: 4.0, speed: 0.7, phase: 2.4, opacity: 0.85, blur: 0.6),
        DotSpec(radius: 34, size: 2.2, speed: 1.1, phase: 3.1, opacity: 0.6, blur: 0.2),
        DotSpec(radius: 20, size: 3.0, speed: 1.6, phase: 4.4, opacity: 0.75, blur: 0.4),
        DotSpec(radius: 26, size: 2.8, speed: 0.8, phase: 5.3, opacity: 0.65, blur: 0.4)
    ]

    private var activityLevel: Double {
        switch state {
        case .idle: return 0.1
        case .listening: return 1.0
        case .transcribing: return 0.7
        case .responding: return 0.8
        case .speaking: return 0.6
        }
    }

    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let responsePulse = (sin(time * 1.2) + 1) / 2
            let accent = accentColor(progress: responsePulse)
            let ringMotion = CGFloat(state == .listening ? 6.0 : 2.0)

            ZStack {
                ForEach(0..<3, id: \.self) { index in
                    let base = CGFloat(18 + index * 8)
                    let wobble = CGFloat(sin(time * 1.4 + Double(index) * 1.3)) * ringMotion
                    let orbit = CGFloat(cos(time * 0.5 + Double(index))) * (state == .listening ? 4.0 : 1.5)
                    let orbitY = CGFloat(sin(time * 0.5 + Double(index))) * (state == .listening ? 4.0 : 1.5)
                    Circle()
                        .trim(from: 0.05, to: 0.82)
                        .stroke(
                            accent.opacity(0.55 - Double(index) * 0.15),
                            style: StrokeStyle(lineWidth: 2.0, lineCap: .round)
                        )
                        .frame(width: (base + wobble) * 2, height: (base + wobble) * 2)
                        .rotationEffect(.degrees(time * 50 + Double(index) * 90))
                        .offset(x: orbit, y: orbitY)
                }

                ForEach(dots) { dot in
                    let angle = time * dot.speed + dot.phase
                    let x = CGFloat(cos(angle)) * dot.radius
                    let y = CGFloat(sin(angle)) * dot.radius
                    Circle()
                        .fill(accent.opacity(dot.opacity))
                        .frame(width: dot.size, height: dot.size)
                        .offset(x: x, y: y)
                        .blur(radius: dot.blur)
                }

                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                accent.opacity(0.85),
                                accent.opacity(0.35),
                                Color.black.opacity(0.05)
                            ],
                            center: .center,
                            startRadius: 2,
                            endRadius: 24
                        )
                    )
                    .frame(width: 22 + CGFloat(activityLevel) * 4, height: 22 + CGFloat(activityLevel) * 4)
                    .overlay(
                        Circle()
                            .stroke(accent.opacity(0.35), lineWidth: 1)
                    )

                Circle()
                    .stroke(accent.opacity(0.25), lineWidth: 3)
                    .frame(width: 46, height: 46)
                    .blur(radius: 0.5)
            }
        }
    }

    private func accentColor(progress: Double) -> Color {
        let orange = RGB(r: 1.0, g: 0.45, b: 0.12)
        let blue = RGB(r: 0.2, g: 0.62, b: 1.0)
        let teal = RGB(r: 0.35, g: 0.95, b: 0.78)
        let cyan = RGB(r: 0.35, g: 0.82, b: 1.0)
        let pink = RGB(r: 0.92, g: 0.55, b: 0.8)
        let idle = RGB(r: 0.82, g: 0.84, b: 0.9)

        switch state {
        case .idle:
            return color(idle, opacity: 0.4)
        case .listening:
            return color(teal)
        case .transcribing:
            return color(cyan)
        case .responding:
            return color(mix(orange, blue, progress))
        case .speaking:
            return color(pink)
        }
    }

    private func mix(_ a: RGB, _ b: RGB, _ t: Double) -> RGB {
        let clamped = min(max(t, 0.0), 1.0)
        return RGB(
            r: a.r + (b.r - a.r) * clamped,
            g: a.g + (b.g - a.g) * clamped,
            b: a.b + (b.b - a.b) * clamped
        )
    }

    private func color(_ rgb: RGB, opacity: Double = 1.0) -> Color {
        Color(red: rgb.r, green: rgb.g, blue: rgb.b, opacity: opacity)
    }
}
