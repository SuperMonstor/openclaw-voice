# OpenClaw Voice Layer - Plan

## Goals
- Ship a reliable local voice+UI layer for OpenClaw on macOS.
- Always-on wake word by default with an optional push-to-talk mode.
- Low-latency streaming UI updates (transcription + response chunks).
- Clean separation between UI, voice engine, and gateway protocol.

## Current Status (Already Done)
- `voice/slice0.py` proves the end-to-end flow (wake word → VAD → STT → gateway → TTS).
- Core scaffolding exists in `voice/src/` (config, state machine, VAD, TTS, UI socket).
- SwiftUI overlay exists in `ui/JarvisOverlay` and listens to `/tmp/openclaw_voice.sock`.

## Decisions
- Engine runs as a **single Python process** with **asyncio tasks** for audio, wake word/VAD, gateway, and TTS.
- **Wake word always-on** by default; **push-to-talk** is user-selectable via config/CLI.
- Gateway **protocol v3** stays as in `slice0.py`, wrapped in a dedicated client module for future changes.

## Phase 1 - Real Engine (Core MVP)
1) Engine entrypoint
   - Add `voice/src/main.py` to orchestrate the pipeline and state transitions.
   - Run a single asyncio loop with tasks for wake word, capture, STT, gateway, TTS.

2) Modularize `slice0.py` into reusable components
   - `voice/src/audio.py` (streaming capture, pre-roll buffer)
   - `voice/src/wake_word.py` (openwakeword wrapper)
   - `voice/src/stt.py` (mlx-whisper wrapper + optional mock)
   - `voice/src/gateway.py` (v3 auth + streaming response)

3) UI event contract (engine → overlay)
   - Formalize JSON-lines events: `state_change`, `transcription`, `response_chunk`, `status`.
   - Support partial transcription updates (`append: true/false`).

4) Config + CLI
   - `config.yaml` support via `voice/src/config.py`.
   - CLI flags: `--config`, `--gateway`, `--socket`, `--mode` (always-on | push-to-talk), `--mic`.

5) Minimal tests
   - Unit tests for state transitions and gateway client parsing.
   - Smoke test for UI socket event emission.

**Exit criteria:** running `python -m src.main` drives the overlay and completes a full voice turn.

## Phase 2 - Robustness + UX
- Barge-in (wake word interrupts TTS).
- Gateway reconnect/backoff and error surfacing to UI.
- Device change handling + explicit mic selection.
- Whisper warm-loading to reduce cold start.
- Optional waveform visualization (RMS/FFT at ~20Hz).

## Phase 3 - Packaging + Ops
- One-shot launcher script to run engine + UI together.
- Model download/setup script.
- README runbook and troubleshooting.
