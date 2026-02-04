import SwiftUI

struct ConstellationView: View {
    let state: OverlayState
    let micLevel: Double

    private struct DotSeed: Identifiable {
        let id: Int
        let angle: Double
        let radiusFactor: Double
        let speed: Double
        let phase: Double
        let size: Double
        let drift: Double
    }

    private struct RGB {
        let r: Double
        let g: Double
        let b: Double
    }

    private static let dotSeeds: [DotSeed] = ConstellationView.makeSeeds(count: 70)

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                render(context: context, size: size, time: timeline.date.timeIntervalSinceReferenceDate)
            }
        }
    }

    private func render(context: GraphicsContext, size: CGSize, time: TimeInterval) {
        let responsePulse = (sin(time * 1.2) + 1) / 2
        let accent = accentColor(progress: responsePulse)
        let baseColor = accent.opacity(0.9)

        let centerX = Double(size.width / 2)
        let centerY = Double(size.height / 2)
        let minSide = Double(min(size.width, size.height))
        let baseRadius = minSide * 0.38
        let clampedMic = min(max(micLevel, 0), 1)
        let energy = stateEnergy()
        let breath = 0.03 * sin(time * 0.9)
        let expansion = 0.08 + clampedMic * 0.22 + energy * 0.08
        let clusterScale = 1 + breath + expansion
        let linkDistance = minSide * (0.28 + clampedMic * 0.14)
        let wiggleBase = minSide * (0.02 + energy * 0.03)
        let wiggleBoost = minSide * (0.08 * clampedMic)

        var points: [CGPoint] = []
        points.reserveCapacity(Self.dotSeeds.count)

        for seed in Self.dotSeeds {
            let baseX = cos(seed.angle) * baseRadius * seed.radiusFactor
            let baseY = sin(seed.angle) * baseRadius * seed.radiusFactor
            let rotation = time * (0.08 + seed.speed * 0.04) + seed.phase * 0.2
            let cosRot = cos(rotation)
            let sinRot = sin(rotation)
            let rotatedX = baseX * cosRot - baseY * sinRot
            let rotatedY = baseX * sinRot + baseY * cosRot
            let wiggle = wiggleBase + wiggleBoost
            let wiggleX = sin(time * (1.1 + seed.drift * 0.2) + seed.phase) * wiggle
            let wiggleY = cos(time * (1.3 + seed.drift * 0.15) + seed.phase * 1.1) * wiggle
            let point = CGPoint(
                x: CGFloat(centerX + rotatedX * clusterScale + wiggleX),
                y: CGFloat(centerY + rotatedY * clusterScale + wiggleY)
            )
            points.append(point)
        }

        for firstIndex in 0..<points.count {
            for secondIndex in (firstIndex + 1)..<points.count {
                let firstPoint = points[firstIndex]
                let secondPoint = points[secondIndex]
                let dx = Double(firstPoint.x - secondPoint.x)
                let dy = Double(firstPoint.y - secondPoint.y)
                let distance = sqrt(dx * dx + dy * dy)
                if distance < linkDistance {
                    let opacity = (1 - (distance / linkDistance)) * 0.5
                    var path = Path()
                    path.move(to: firstPoint)
                    path.addLine(to: secondPoint)
                    context.stroke(path, with: .color(baseColor.opacity(opacity)), lineWidth: 1)
                }
            }
        }

        for (index, point) in points.enumerated() {
            let seed = Self.dotSeeds[index]
            let radius = (seed.size + clampedMic * 1.6) * (minSide / 120)
            let dotRect = CGRect(
                x: point.x - CGFloat(radius),
                y: point.y - CGFloat(radius),
                width: CGFloat(radius * 2),
                height: CGFloat(radius * 2)
            )
            context.fill(Path(ellipseIn: dotRect), with: .color(baseColor))
        }
    }

    private func stateEnergy() -> Double {
        switch state {
        case .idle:
            return 0.12
        case .listening:
            return 0.9
        case .transcribing:
            return 0.7
        case .responding:
            return 0.8
        case .speaking:
            return 0.6
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
            return color(idle, opacity: 0.45)
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

    private func mix(_ first: RGB, _ second: RGB, _ progress: Double) -> RGB {
        let clamped = min(max(progress, 0.0), 1.0)
        return RGB(
            r: first.r + (second.r - first.r) * clamped,
            g: first.g + (second.g - first.g) * clamped,
            b: first.b + (second.b - first.b) * clamped
        )
    }

    private func color(_ rgb: RGB, opacity: Double = 1.0) -> Color {
        Color(red: rgb.r, green: rgb.g, blue: rgb.b, opacity: opacity)
    }

    private static func makeSeeds(count: Int) -> [DotSeed] {
        var seeds: [DotSeed] = []
        seeds.reserveCapacity(count)
        for index in 0..<count {
            let angle = seededValue(index, salt: 1.1) * Double.pi * 2
            let radiusFactor = sqrt(seededValue(index, salt: 2.3))
            let speed = 0.6 + seededValue(index, salt: 3.7) * 0.9
            let phase = seededValue(index, salt: 4.9) * Double.pi * 2
            let size = 1.6 + seededValue(index, salt: 6.2) * 2.4
            let drift = seededValue(index, salt: 7.4) * 2 - 1
            seeds.append(
                DotSeed(
                    id: index,
                    angle: angle,
                    radiusFactor: radiusFactor,
                    speed: speed,
                    phase: phase,
                    size: size,
                    drift: drift
                )
            )
        }
        return seeds
    }

    private static func seededValue(_ index: Int, salt: Double) -> Double {
        let value = sin(Double(index) * 12.9898 + salt) * 43758.5453
        return value - floor(value)
    }
}
