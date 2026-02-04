import AVFoundation

final class MicrophoneLevelMonitor {
    private let audioEngine = AVAudioEngine()
    private var isRunning = false

    var onLevel: ((Double) -> Void)?

    func start() {
        guard !isRunning else { return }
        let isAppBundle = Bundle.main.bundlePath.hasSuffix(".app")
        if isAppBundle,
           Bundle.main.object(forInfoDictionaryKey: "NSMicrophoneUsageDescription") == nil {
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            startEngine()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.startEngine()
                    }
                }
            }
        default:
            break
        }
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
    }

    private func startEngine() {
        guard !isRunning else { return }
        isRunning = true
        let inputNode = audioEngine.inputNode
        let format = inputNode.inputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.process(buffer: buffer)
        }
        do {
            try audioEngine.start()
        } catch {
            stop()
        }
    }

    private func process(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return }
        let samples = channelData[0]
        var sumSquares: Float = 0
        for index in 0..<frameCount {
            let sample = samples[index]
            sumSquares += sample * sample
        }
        let rms = sqrt(sumSquares / Float(frameCount))
        let clampedRms = max(Double(rms), 0.000_000_1)
        let db = 20 * log10(clampedRms)
        let minDb = -55.0
        let maxDb = -10.0
        let normalized = (db - minDb) / (maxDb - minDb)
        let level = min(max(normalized, 0), 1)
        DispatchQueue.main.async { [weak self] in
            self?.onLevel?(level)
        }
    }
}
