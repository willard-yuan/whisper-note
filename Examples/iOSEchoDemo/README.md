# iOS Echo Demo

On-device Qwen3-ASR pseudo-streaming transcription.

- **Device**: Qwen3-ASR CoreML INT8 pseudo-streaming
- **Simulator**: Qwen3-ASR CoreML INT8 pseudo-streaming

## Setup

```bash
cd Examples/iOSEchoDemo
xcodegen generate
open iOSEchoDemo.xcodeproj
```

Set your signing team in Xcode, build and run.

Models download from HuggingFace on first launch unless bundled.

To create an offline build, prefill the app resource directory before
generating/building the Xcode project:

```bash
cd Examples/iOSEchoDemo
scripts/download_bundled_models.sh
xcodegen generate
```

The app looks for bundled models first:

```text
iOSEchoDemo/BundledModels/
  qwen3-asr-coreml/
  silero-vad-coreml/
```

If those directories are empty, it falls back to the library's default model
cache and downloads missing files.

## Features

- Voice activity detection (Silero VAD)
- Qwen3-ASR pseudo-streaming partials with final VAD commits
- Force-cut at 10s with system message
- Diagnostics view (CPU, memory, VAD level)
