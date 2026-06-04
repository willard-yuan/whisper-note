#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT_DIR/iOSEchoDemo/BundledModels"

if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "hf is required. Install with: python3 -m pip install huggingface_hub"
  exit 1
fi

mkdir -p "$DEST_DIR/qwen3-asr-coreml" "$DEST_DIR/silero-vad-coreml"

echo "Downloading Qwen3-ASR CoreML INT8..."
"$HF_CLI" download aufklarer/Qwen3-ASR-CoreML \
  --local-dir "$DEST_DIR/qwen3-asr-coreml" \
  --max-workers 1 \
  --include "encoder.mlmodelc/**" \
  --include "embedding.mlmodelc/**" \
  --include "decoder_part1.mlmodelc/**" \
  --include "decoder_part2.mlmodelc/**" \
  --include "config.json"

echo "Downloading Qwen3 tokenizer files..."
"$HF_CLI" download aufklarer/Qwen3-ASR-0.6B-MLX-4bit \
  --local-dir "$DEST_DIR/qwen3-asr-coreml" \
  --max-workers 1 \
  --include "vocab.json" \
  --include "merges.txt" \
  --include "tokenizer_config.json"

echo "Downloading Silero VAD CoreML..."
"$HF_CLI" download aufklarer/Silero-VAD-v5-CoreML \
  --local-dir "$DEST_DIR/silero-vad-coreml" \
  --max-workers 1

find "$DEST_DIR" -name ".cache" -type d -prune -exec rm -rf {} +
find "$DEST_DIR" -name ".DS_Store" -type f -delete

echo "Bundled models are ready in $DEST_DIR"
