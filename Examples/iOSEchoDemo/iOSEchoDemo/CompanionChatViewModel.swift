import AVFoundation
import Foundation
import os
import Observation
import Qwen3ASR
import SpeechVAD

enum MessageRole { case user, assistant, system }

/// Message displayed in chat UI.
struct ChatBubbleMessage: Identifiable {
    let id = UUID()
    let role: MessageRole
    var text: String
    let timestamp = Date()
}

private let pipelineLog = Logger(subsystem: "audio.soniqo.iOSEchoDemo", category: "Qwen3PseudoStreaming")

@Observable
@MainActor
final class CompanionChatViewModel {
    // MARK: - UI State

    var messages: [ChatBubbleMessage] = []
    var inputText = ""
    var currentPartial = ""
    var isLoading = false
    var isListening = false
    var isSpeechDetected = false
    var pipelineState = "idle"
    var audioLevel: Float = 0
    var loadProgress: Double = 0
    var loadingStatus = ""
    var errorMessage: String?
    /// Which compute backend the loaded ASR encoder is using. Qwen3 CoreML
    /// loads with `.all`, so on device this means ANE preferred.
    var asrBackend = "—"

    private var _modelsLoaded = false
    var modelsLoaded: Bool { _modelsLoaded }

    let diagnostics = DiagnosticsMonitor()

    // MARK: - Private State

    private var vadModel: SileroVADModel?
    private var asrModel: CoreMLASRModel?
    private var recognizer: Qwen3PseudoStreamingASR?
    private var audioEngine: AVAudioEngine?
    private var micRecordBuffer: [Float] = []
    private var debugLog: [String] = []
    private var agcRecentPeak: Float = 0

    private func dbg(_ msg: String) {
        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent().truncatingRemainder(dividingBy: 1000))
        let line = "[\(ts)] \(msg)"
        debugLog.append(line)
        pipelineLog.warning("\(line, privacy: .public)")
    }

    // MARK: - Load Models

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        loadProgress = 0

        do {
            let vadDir = BundledModelStore.preferredDirectory(for: .sileroVAD)
            loadingStatus = "Loading VAD (\(vadDir.loadingModeDescription))..."
            loadProgress = 0.05
            vadModel = try await Task.detached {
                try await SileroVADModel.fromPretrained(
                    engine: .coreml,
                    cacheDir: vadDir.url,
                    offlineMode: vadDir.isBundled
                ) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.05 + progress * 0.20
                        if !status.isEmpty { self?.loadingStatus = "VAD: \(status)" }
                    }
                }
            }.value

            let asrDir = BundledModelStore.preferredDirectory(for: .qwen3ASR)
            loadingStatus = "Loading Qwen3 ASR (\(asrDir.loadingModeDescription))..."
            loadProgress = 0.25
            asrModel = try await Task.detached {
                let model = try await CoreMLASRModel.fromPretrained(
                    cacheDir: asrDir.url,
                    offlineMode: asrDir.isBundled
                ) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.25 + progress * 0.70
                        if !status.isEmpty { self?.loadingStatus = "Qwen3 ASR: \(status)" }
                    }
                }
                try model.warmUp()
                return model
            }.value
            asrBackend = asrDir.isBundled ? "ANE (bundled)" : "ANE"

            loadProgress = 1.0
            loadingStatus = "Ready"
            _modelsLoaded = true
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
        }

        isLoading = false
    }

    // MARK: - Pipeline Start/Stop

    func startListening() {
        guard !isListening, let vad = vadModel, let asr = asrModel else { return }

        let worker = Qwen3CoreMLASRWorker(model: asr)
        recognizer = Qwen3PseudoStreamingASR(
            asrWorker: worker,
            vadModel: vad,
            config: .default,
            eventHandler: { [weak self] event in
                DispatchQueue.main.async { self?.handleASREvent(event) }
            }
        )

        dbg("[START] Qwen3 pseudo-streaming recognizer created")
        isListening = true
        pipelineState = "listening"
        diagnostics.start()
        startMicrophone()
        dbg("[START] mic started, recognizer running")
    }

    func stopListening() {
        diagnostics.stop()
        stopMicrophone()
        recognizer?.stop()
        recognizer = nil
        isListening = false
        isSpeechDetected = false
        currentPartial = ""
        audioLevel = 0
        pipelineState = "idle"
        saveDebugRecording()
    }

    // MARK: - ASR Events

    private func handleASREvent(_ event: Qwen3PseudoStreamingEvent) {
        switch event {
        case .speechStarted:
            dbg("speechStarted")
            currentPartial = ""
            isSpeechDetected = true
            pipelineState = "listening..."

        case .partial(let text, _, _, _):
            dbg("partial: '\(text)'")
            currentPartial = text
            pipelineState = "transcribing partial..."

        case .final(let text, _, _, _, let forced):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            dbg("final\(forced ? " forced" : ""): '\(trimmed)'")
            isSpeechDetected = false
            currentPartial = ""
            guard !trimmed.isEmpty else {
                pipelineState = "listening"
                return
            }

            messages.append(ChatBubbleMessage(role: .user, text: trimmed))
            pipelineState = isListening ? "listening" : "idle"

        case .forceSplit(let duration):
            dbg("forceSplit after \(String(format: "%.1f", duration))s")
            messages.append(ChatBubbleMessage(
                role: .system,
                text: "Segment limit reached (\(Int(duration))s). Transcribing what was captured."
            ))
            pipelineState = "segment limit, transcribing..."

        case .error(let message):
            dbg("ERROR: \(message)")
            errorMessage = message
            isSpeechDetected = false
            currentPartial = ""
            pipelineState = "error"
        }
    }

    // MARK: - Text fallback

    func send(_ text: String) {
        messages.append(ChatBubbleMessage(role: .user, text: text))
        inputText = ""
    }

    func clearChat() {
        messages.removeAll()
        currentPartial = ""
    }

    // MARK: - Microphone

    private func startMicrophone() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()

        switch AVAudioApplication.shared.recordPermission {
        case .undetermined:
            AVAudioApplication.requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.startMicrophone()
                    } else {
                        self?.errorMessage = "Microphone permission denied"
                    }
                }
            }
            return
        case .denied:
            errorMessage = "Microphone permission denied. Enable in Settings."
            return
        case .granted:
            break
        @unknown default:
            break
        }

        do {
            try session.setCategory(.playAndRecord, mode: .default,
                                    options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true)
            try session.overrideOutputAudioPort(.speaker)
        } catch {
            errorMessage = "Mic access failed: \(error.localizedDescription)"
            return
        }
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000,
            channels: 1, interleaved: false
        ) else { return }

        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: hwFormat.sampleRate,
            channels: 1, interleaved: false
        ) else { return }

        guard let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            guard let monoBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat,
                                                     frameCapacity: buffer.frameCapacity) else { return }
            monoBuffer.frameLength = buffer.frameLength
            memcpy(monoBuffer.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            let outFrameCount = AVAudioFrameCount(Double(frameLen) * 16000.0 / hwFormat.sampleRate)
            guard outFrameCount > 0,
                  let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat,
                                                    frameCapacity: outFrameCount) else { return }

            var error: NSError?
            resampler.convert(to: outBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return monoBuffer
            }
            if error != nil { return }

            guard let outData = outBuffer.floatChannelData else { return }
            let count = Int(outBuffer.frameLength)
            guard count > 0 else { return }
            let samples = Array(UnsafeBufferPointer(start: outData[0], count: count))

            var sum: Float = 0
            for s in samples { sum += s * s }
            let rms = sqrt(sum / max(Float(count), 1))
            DispatchQueue.main.async {
                self.audioLevel = rms
                self.diagnostics.updateVAD(rms)
            }

            self.micRecordBuffer.append(contentsOf: samples)
            let maxMicSamples = 16000 * 60
            if self.micRecordBuffer.count > maxMicSamples {
                self.micRecordBuffer.removeFirst(self.micRecordBuffer.count - maxMicSamples)
            }

            self.recognizer?.pushAudio(self.applyAGC(to: samples))
        }

        do {
            try engine.start()
            audioEngine = engine
        } catch {
            errorMessage = "Mic error: \(error.localizedDescription)"
        }
    }

    private func stopMicrophone() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
    }

    /// Per-buffer automatic gain control. Tracks a decaying peak of the
    /// loudest recent sample and scales the buffer so that peak hits a
    /// target level (~0.5). Caps at 10× to avoid amplifying silence /
    /// background noise to speech-trigger levels.
    private func applyAGC(to samples: [Float]) -> [Float] {
        let targetPeak: Float = 0.5
        let decay: Float = 0.95
        let maxGain: Float = 10
        let minPeakForGain: Float = 0.005

        var bufferPeak: Float = 0
        for s in samples {
            let abs = s < 0 ? -s : s
            if abs > bufferPeak { bufferPeak = abs }
        }
        agcRecentPeak = max(bufferPeak, agcRecentPeak * decay)

        guard agcRecentPeak >= minPeakForGain else { return samples }
        let gain = min(maxGain, targetPeak / agcRecentPeak)
        guard gain > 1.05 else { return samples }
        return samples.map { max(-1, min(1, $0 * gain)) }
    }

    // MARK: - Debug Recording

    private func saveDebugRecording() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("debug_audio")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        if !micRecordBuffer.isEmpty {
            let url = dir.appendingPathComponent("mic_debug.wav")
            writeWAV(samples: micRecordBuffer, sampleRate: 16000, to: url)
            pipelineLog.warning("DEBUG MIC: \(url.path) (\(self.micRecordBuffer.count / 16000)s)")
            micRecordBuffer.removeAll()
        }

        if !debugLog.isEmpty {
            let logUrl = dir.appendingPathComponent("pipeline_debug.log")
            try? debugLog.joined(separator: "\n").write(to: logUrl, atomically: true, encoding: .utf8)
            debugLog.removeAll()
        }
    }

    private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) {
        var data = Data()
        let dataSize = samples.count * 2
        data.append(contentsOf: "RIFF".utf8)
        var fileSize = UInt32(36 + dataSize); data.append(Data(bytes: &fileSize, count: 4))
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        var fmtSize: UInt32 = 16; data.append(Data(bytes: &fmtSize, count: 4))
        var fmt: UInt16 = 1; data.append(Data(bytes: &fmt, count: 2))
        var ch: UInt16 = 1; data.append(Data(bytes: &ch, count: 2))
        var sr = UInt32(sampleRate); data.append(Data(bytes: &sr, count: 4))
        var byteRate = UInt32(sampleRate * 2); data.append(Data(bytes: &byteRate, count: 4))
        var blockAlign: UInt16 = 2; data.append(Data(bytes: &blockAlign, count: 2))
        var bps: UInt16 = 16; data.append(Data(bytes: &bps, count: 2))
        data.append(contentsOf: "data".utf8)
        var dSize = UInt32(dataSize); data.append(Data(bytes: &dSize, count: 4))
        for s in samples {
            var pcm = Int16(max(-1, min(1, s)) * 32767)
            data.append(Data(bytes: &pcm, count: 2))
        }
        try? data.write(to: url)
    }
}
