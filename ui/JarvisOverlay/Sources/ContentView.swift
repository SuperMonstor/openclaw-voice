import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: OverlayViewModel

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(LinearGradient(
                    colors: [
                        Color.black.opacity(0.9),
                        Color(red: 0.06, green: 0.07, blue: 0.1).opacity(0.95),
                        Color.black.opacity(0.75)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(Color.white.opacity(0.12), lineWidth: 1)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color.white.opacity(0.08),
                                    Color.clear
                                ],
                                center: .topLeading,
                                startRadius: 20,
                                endRadius: 180
                            )
                        )
                )

            HStack(spacing: 16) {
                RingView(state: viewModel.state)
                    .frame(width: 72, height: 72)

                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Text("OpenClaw")
                            .font(.custom("Avenir Next Demi Bold", size: 15))
                            .foregroundStyle(Color.white)
                        Text(viewModel.statusText.uppercased())
                            .font(.custom("Avenir Next", size: 10))
                            .foregroundStyle(Color.white.opacity(0.8))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(
                                Capsule()
                                    .fill(Color.white.opacity(0.12))
                            )
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .top, spacing: 6) {
                            Text("YOU")
                                .font(.custom("Avenir Next Demi Bold", size: 9))
                                .foregroundStyle(Color.white.opacity(0.6))
                                .padding(.top, 1)
                            Text(viewModel.liveTranscription.isEmpty ? "Say something..." : viewModel.liveTranscription)
                                .font(.custom("Avenir Next", size: 12))
                                .foregroundStyle(Color.white.opacity(viewModel.liveTranscription.isEmpty ? 0.5 : 0.9))
                                .lineLimit(2)
                        }

                        HStack(alignment: .top, spacing: 6) {
                            Text("CLAW")
                                .font(.custom("Avenir Next Demi Bold", size: 9))
                                .foregroundStyle(Color.white.opacity(0.5))
                                .padding(.top, 1)
                            Text(viewModel.responseText.isEmpty ? " " : viewModel.responseText)
                                .font(.custom("Avenir Next", size: 12))
                                .foregroundStyle(Color.white.opacity(0.7))
                                .lineLimit(2)
                        }
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(16)
        }
        .frame(width: 380, height: 160)
    }
}
