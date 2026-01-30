# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenClaw Voice is a local voice interface for OpenClaw that provides wake word detection, speech-to-text, and text-to-speech with a JARVIS-style overlay UI.

## Architecture

```
macOS Overlay (SwiftUI) ←─ stdin/stdout JSON ─→ Voice Engine (Python) ←─ WebSocket ─→ OpenClaw Gateway
```

**Two main components:**
1. **Voice Engine** (`voice/`) - Python, handles audio pipeline: wake word → STT → Gateway → TTS
2. **Overlay UI** (`ui/JarvisOverlay/`) - SwiftUI macOS app, floating waveform visualization

**Voice flow:** OpenWakeWord (80ms chunks) → mlx-whisper transcription → OpenClaw Gateway (ws://127.0.0.1:18789) → Piper TTS

## Build Commands

```bash
# Voice Engine (Python)
cd voice
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run voice engine
python -m src.main

# Run tests
pytest

# Lint
ruff check src/
black --check src/
```

## Key Dependencies

- `openwakeword` - Wake word detection (processes 80ms audio frames, returns 0-1 confidence)
- `mlx-whisper` - Apple Silicon optimized speech-to-text
- `piper-tts` - Neural text-to-speech
- `sounddevice` - Audio capture/playback
- `websockets` - OpenClaw Gateway connection

## Configuration

Config loaded from `config.yaml` via `Config.from_yaml()`. Key settings:
- `audio.chunk_size`: 1280 samples = 80ms at 16kHz (required by OpenWakeWord)
- `wake_word.threshold`: Detection confidence threshold (0-1)
- `gateway.url`: OpenClaw WebSocket endpoint (default: ws://127.0.0.1:18789)

## OpenClaw Gateway Protocol

WebSocket JSON-RPC at `ws://127.0.0.1:18789`. Key methods:
- `sessions_send` - Send user message
- Streaming response events for assistant replies
