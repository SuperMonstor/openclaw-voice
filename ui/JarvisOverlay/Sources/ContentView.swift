import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: OverlayViewModel

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(LinearGradient(
                    colors: [Color.black.opacity(0.75), Color.black.opacity(0.45)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                )

            HStack(spacing: 12) {
                RingView(state: viewModel.state)
                    .frame(width: 44, height: 44)

                VStack(alignment: .leading, spacing: 4) {
                    Text("OpenClaw")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Color.white)
                    Text(viewModel.transcription)
                        .font(.system(size: 12))
                        .foregroundStyle(Color.white.opacity(0.75))
                        .lineLimit(2)
                }
                Spacer(minLength: 0)
            }
            .padding(12)
        }
        .frame(width: 320, height: 120)
    }
}
