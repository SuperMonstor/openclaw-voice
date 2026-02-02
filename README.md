# OpenClaw Voice

OpenClaw Voice is a local, Siri/Alexa-style voice interface for OpenClaw on macOS.
It listens for a wake word, captures speech, transcribes it, sends it to the OpenClaw
Gateway, and speaks the response back. A floating SwiftUI overlay sits at the top-right
of the screen, showing live transcription and a status animation while the assistant
is thinking/responding.

## Target Experience

- Wake word detection (always-on, low latency)
- Live transcription text in a top-right overlay
- A loading indicator animation around the overlay border while the AI responds
- Spoken reply via TTS

## Architecture (Planned)

macOS Overlay (SwiftUI) <-> stdin/stdout JSON <-> Voice Engine (Python) <-> WebSocket <-> OpenClaw Gateway

## Status

Early scaffolding exists for the Python voice engine and a working end-to-end demo
script (`voice/slice0.py`). The SwiftUI overlay and the full engine pipeline are not
built out yet.

## Repository Layout

- `voice/`: Python voice engine (config + prototypes)
- `ui/`: SwiftUI overlay app (to be implemented)
- `models/`: Local model assets (wake word, whisper, piper)

## Notes

This repo is intended for local, on-device operation on macOS.
