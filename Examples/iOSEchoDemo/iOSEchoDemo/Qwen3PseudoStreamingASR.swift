import Foundation
import Qwen3ASR
import SpeechVAD

struct Qwen3PseudoStreamingConfig {
    var language: String?
    var vadConfig: VADConfig
    var preSpeechDuration: Double
    var partialInterval: Double
    var minimumPartialDuration: Double
    var maxSegmentDuration: Double
    var hardMaxSegmentDuration: Double
    var partialMaxTokens: Int
    var finalMaxTokens: Int

    static let `default` = Qwen3PseudoStreamingConfig(
        language: nil,
        vadConfig: VADConfig(
            onset: 0.2,
            offset: 0.15,
            minSpeechDuration: 0.20,
            minSilenceDuration: 0.80,
            windowDuration: 0.032,
            stepRatio: 1.0
        ),
        preSpeechDuration: 0.30,
        partialInterval: 1.80,
        minimumPartialDuration: 1.20,
        maxSegmentDuration: 10.0,
        hardMaxSegmentDuration: 26.0,
        partialMaxTokens: 160,
        finalMaxTokens: 448
    )
}

enum Qwen3PseudoStreamingEvent {
    case speechStarted(startTime: Double)
    case partial(text: String, startTime: Double, endTime: Double, segmentIndex: Int)
    case final(text: String, startTime: Double, endTime: Double, segmentIndex: Int, forced: Bool)
    case forceSplit(duration: Double)
    case error(String)
}

final class Qwen3CoreMLASRWorker {
    private let model: CoreMLASRModel
    private let queue = DispatchQueue(label: "audio.soniqo.iOSEchoDemo.qwen3-asr")

    init(model: CoreMLASRModel) {
        self.model = model
    }

    func transcribe(
        samples: [Float],
        language: String?,
        maxTokens: Int,
        completion: @escaping (Result<String, Error>) -> Void
    ) {
        queue.async { [model] in
            do {
                let text = try model.transcribe(
                    audio: samples,
                    sampleRate: 16000,
                    language: language,
                    maxTokens: maxTokens
                )
                completion(.success(text))
            } catch {
                completion(.failure(error))
            }
        }
    }
}

final class Qwen3PseudoStreamingASR {
    private let vadProcessor: StreamingVADProcessor
    private let asrWorker: Qwen3CoreMLASRWorker
    private let config: Qwen3PseudoStreamingConfig
    private let eventHandler: (Qwen3PseudoStreamingEvent) -> Void
    private let queue = DispatchQueue(label: "audio.soniqo.iOSEchoDemo.qwen3-pseudostream")

    private var preSpeechBuffer: [Float] = []
    private var utteranceBuffer: [Float] = []
    private var isSpeechActive = false
    private var utteranceStartTime: Double = 0
    private var lastPartialTime: Double = 0
    private var segmentIndex = 0
    private var partialRequestID = 0
    private var partialInFlight = false
    private var finalInFlight = false

    init(
        asrWorker: Qwen3CoreMLASRWorker,
        vadModel: SileroVADModel,
        config: Qwen3PseudoStreamingConfig,
        eventHandler: @escaping (Qwen3PseudoStreamingEvent) -> Void
    ) {
        self.asrWorker = asrWorker
        self.config = config
        self.eventHandler = eventHandler
        self.vadProcessor = StreamingVADProcessor(model: vadModel, config: config.vadConfig)
    }

    func pushAudio(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        queue.async { [weak self] in
            self?.process(samples)
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            let events = self.vadProcessor.flush()
            self.handle(events: events)
            self.resetState()
        }
    }

    private func process(_ samples: [Float]) {
        appendPreSpeech(samples)

        if isSpeechActive {
            utteranceBuffer.append(contentsOf: samples)
        }

        let events = vadProcessor.process(samples: samples)
        handle(events: events)

        guard isSpeechActive else { return }

        let now = Double(vadProcessor.currentTime)
        let duration = now - utteranceStartTime
        if duration >= config.hardMaxSegmentDuration || duration >= config.maxSegmentDuration {
            eventHandler(.forceSplit(duration: duration))
            submitFinal(endTime: now, forced: true)
            startNextForcedSegment(at: now)
            return
        }

        maybeSubmitPartial(now: now)
    }

    private func handle(events: [VADEvent]) {
        for event in events {
            switch event {
            case .speechStarted(let time):
                guard !isSpeechActive else { continue }
                isSpeechActive = true
                utteranceStartTime = max(0, Double(time) - config.preSpeechDuration)
                lastPartialTime = Double(time)
                utteranceBuffer = preSpeechBuffer
                eventHandler(.speechStarted(startTime: utteranceStartTime))

            case .speechEnded(let segment):
                guard isSpeechActive else { continue }
                submitFinal(endTime: Double(segment.endTime), forced: false)
                isSpeechActive = false
                utteranceBuffer.removeAll(keepingCapacity: true)
            }
        }
    }

    private func maybeSubmitPartial(now: Double) {
        guard !partialInFlight, !finalInFlight else { return }
        guard now - utteranceStartTime >= config.minimumPartialDuration else { return }
        guard now - lastPartialTime >= config.partialInterval else { return }

        let snapshot = utteranceBuffer
        guard snapshot.count >= Int(config.minimumPartialDuration * 16_000) else { return }

        partialInFlight = true
        partialRequestID += 1
        let requestID = partialRequestID
        let currentSegment = segmentIndex
        let start = utteranceStartTime
        let end = now
        lastPartialTime = now

        asrWorker.transcribe(
            samples: snapshot,
            language: config.language,
            maxTokens: config.partialMaxTokens
        ) { [weak self] result in
            self?.queue.async {
                guard let self else { return }
                self.partialInFlight = false
                guard requestID == self.partialRequestID,
                      currentSegment == self.segmentIndex,
                      self.isSpeechActive
                else {
                    return
                }

                switch result {
                case .success(let text):
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        self.eventHandler(.partial(
                            text: trimmed,
                            startTime: start,
                            endTime: end,
                            segmentIndex: currentSegment
                        ))
                    }
                case .failure(let error):
                    self.eventHandler(.error(error.localizedDescription))
                }
            }
        }
    }

    private func submitFinal(endTime: Double, forced: Bool) {
        let snapshot = utteranceBuffer
        guard !snapshot.isEmpty else { return }

        finalInFlight = true
        partialRequestID += 1
        let currentSegment = segmentIndex
        let start = utteranceStartTime

        asrWorker.transcribe(
            samples: snapshot,
            language: config.language,
            maxTokens: config.finalMaxTokens
        ) { [weak self] result in
            self?.queue.async {
                guard let self else { return }
                self.finalInFlight = false

                switch result {
                case .success(let text):
                    self.eventHandler(.final(
                        text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                        startTime: start,
                        endTime: endTime,
                        segmentIndex: currentSegment,
                        forced: forced
                    ))
                case .failure(let error):
                    self.eventHandler(.error(error.localizedDescription))
                }
            }
        }

        segmentIndex += 1
    }

    private func startNextForcedSegment(at time: Double) {
        isSpeechActive = true
        utteranceStartTime = time
        lastPartialTime = time
        utteranceBuffer.removeAll(keepingCapacity: true)
        partialInFlight = false
        finalInFlight = false
    }

    private func appendPreSpeech(_ samples: [Float]) {
        preSpeechBuffer.append(contentsOf: samples)
        let maxSamples = max(0, Int(config.preSpeechDuration * 16_000))
        if preSpeechBuffer.count > maxSamples {
            preSpeechBuffer.removeFirst(preSpeechBuffer.count - maxSamples)
        }
    }

    private func resetState() {
        preSpeechBuffer.removeAll(keepingCapacity: true)
        utteranceBuffer.removeAll(keepingCapacity: true)
        isSpeechActive = false
        lastPartialTime = 0
        partialRequestID += 1
        partialInFlight = false
        finalInFlight = false
    }
}
