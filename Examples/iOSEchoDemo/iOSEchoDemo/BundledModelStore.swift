import Foundation

struct ModelDirectory {
    let url: URL?
    let isBundled: Bool

    var loadingModeDescription: String {
        isBundled ? "bundled" : "download/cache"
    }
}

enum BundledModelStore {
    enum Model: String {
        case qwen3ASR = "qwen3-asr-coreml"
        case sileroVAD = "silero-vad-coreml"
        case kokoroTTS = "kokoro-tts-coreml"
    }

    static func preferredDirectory(for model: Model) -> ModelDirectory {
        if let bundled = bundledDirectory(for: model) {
            return ModelDirectory(url: bundled, isBundled: true)
        }
        return ModelDirectory(url: nil, isBundled: false)
    }

    private static func bundledDirectory(for model: Model) -> URL? {
        for bundle in resourceBundles {
            guard let root = bundle.url(forResource: "BundledModels", withExtension: nil) else {
                continue
            }
            let dir = root.appendingPathComponent(model.rawValue, isDirectory: true)
            if directoryHasModelContent(dir) {
                return dir
            }
        }
        return nil
    }

    private static var resourceBundles: [Bundle] {
        var bundles = [Bundle.main]
        #if SWIFT_PACKAGE
        bundles.append(Bundle.module)
        #endif
        return bundles
    }

    private static func directoryHasModelContent(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
              isDir.boolValue
        else {
            return false
        }

        let knownFiles = [
            "config.json",
            "encoder.mlmodelc",
            "silero_vad.mlmodelc",
            "kokoro_5s.mlmodelc",
        ]
        return knownFiles.contains { name in
            FileManager.default.fileExists(
                atPath: url.appendingPathComponent(name).path
            )
        }
    }
}
