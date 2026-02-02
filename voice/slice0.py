#!/usr/bin/env python3
"""
Slice 0: Ugly but working end-to-end voice loop.

This proves all integration points work:
- Audio capture (sounddevice)
- Speech-to-text (mlx-whisper)
- Gateway communication (websockets)
- Text-to-speech (macOS say command)

Usage:
    python slice0.py
    python slice0.py --gateway ws://127.0.0.1:18789

Press Enter to start recording, speak, wait for response.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import argparse
import uuid
from datetime import datetime
from pathlib import Path

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # Fixed recording duration for slice0


def log(state: str, msg: str):
    """Log with timestamp and state."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{state:^12}] {msg}")


def list_input_devices():
    """Return list of (device_index, name) for input-capable devices."""
    import sounddevice as sd

    devices = sd.query_devices()
    input_devices = []
    for idx, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            input_devices.append((idx, info.get("name", f"Device {idx}")))
    return input_devices


def select_microphone() -> int | None:
    """Prompt user to select an input device. Returns selected device index."""
    import sounddevice as sd

    input_devices = list_input_devices()
    if not input_devices:
        log("ERROR", "No input devices found")
        return None

    current_default = sd.default.device[0] if sd.default.device else None
    print("\nAvailable input devices:")
    for idx, name in input_devices:
        default_mark = " (default)" if idx == current_default else ""
        print(f"  [{idx}] {name}{default_mark}")

    choice = input("Select input device index (blank to cancel): ").strip()
    if not choice:
        log("IDLE", "Microphone selection canceled")
        return None

    try:
        selected = int(choice)
    except ValueError:
        log("ERROR", f"Invalid device index: {choice}")
        return None

    if selected not in {idx for idx, _ in input_devices}:
        log("ERROR", f"Device index {selected} is not an input device")
        return None

    sd.default.device = (selected, sd.default.device[1])
    log("IDLE", f"Selected input device {selected}")
    return selected


def record_audio(duration: float = RECORD_SECONDS, device: int | None = None) -> bytes:
    """Record audio from microphone."""
    import sounddevice as sd
    import numpy as np

    log("LISTENING", f"Recording for {duration}s... Speak now!")

    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        device=device,
    )
    sd.wait()

    log("LISTENING", "Recording complete")
    return recording


def save_wav(audio_data, path: Path):
    """Save audio data to WAV file."""
    import soundfile as sf

    sf.write(str(path), audio_data, SAMPLE_RATE)


def transcribe(audio_path: Path) -> str:
    """Transcribe audio using mlx-whisper."""
    log("TRANSCRIBING", "Loading whisper model...")

    try:
        import mlx_whisper

        whisper_repo = os.environ.get(
            "OPENCLAW_WHISPER_REPO", "mlx-community/whisper-base-mlx"
        )
        log("TRANSCRIBING", f"Transcribing {audio_path} with {whisper_repo}...")
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=whisper_repo,
        )
        text = result.get("text", "").strip()
        log("TRANSCRIBING", f"Result: {text}")
        return text
    except ImportError:
        log("TRANSCRIBING", "mlx-whisper not installed, using mock")
        return "Hello, this is a test message."


def speak(text: str):
    """Speak text using macOS say command."""
    log("SPEAKING", f"Speaking: {text[:50]}...")
    subprocess.run(["say", text], check=True)
    log("SPEAKING", "Done speaking")


async def send_to_gateway(url: str, message: str) -> str:
    """Send message to OpenClaw Gateway and collect response."""
    import websockets

    log("GATEWAY", f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        # Wait for challenge
        challenge = await asyncio.wait_for(ws.recv(), timeout=5.0)
        challenge_data = json.loads(challenge)

        if challenge_data.get("event") == "connect.challenge":
            # Authenticate
            connect_req = {
                "type": "req",
                "id": 1,
                "method": "connect",
                "params": {
                    "role": "operator",
                    "scopes": ["operator.read", "operator.write"],
                    "clientType": "voice",
                    "clientVersion": "0.1.0",
                },
            }
            await ws.send(json.dumps(connect_req))

            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            response_data = json.loads(response)

            if not response_data.get("ok"):
                raise Exception(f"Auth failed: {response_data.get('error')}")

            log("GATEWAY", "Authenticated")

        # Send chat message
        session_key = "voice"
        idempotency_key = str(uuid.uuid4())

        log("GATEWAY", f"Sending: {message[:50]}...")
        send_req = {
            "type": "req",
            "id": 2,
            "method": "chat.send",
            "params": {
                "sessionKey": session_key,
                "message": message,
                "idempotencyKey": idempotency_key,
                "timeoutMs": 60000,
            },
        }
        await ws.send(json.dumps(send_req))

        # Collect streaming response
        log("GATEWAY", "Waiting for response...")
        full_response = ""
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < 120:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(msg)

                msg_type = data.get("type")
                if msg_type == "event":
                    event_name = data.get("event", "")
                    payload = data.get("payload", {})

                    if event_name == "agent":
                        phase = payload.get("phase")
                        if phase == "end":
                            log("GATEWAY", "Response complete")
                            break
                    elif "delta" in event_name or "text" in event_name.lower():
                        text = payload.get("text", payload.get("content", ""))
                        if text:
                            full_response += text
                            print(text, end="", flush=True)

            except asyncio.TimeoutError:
                # Check if we have a response and it's been quiet
                if full_response and (asyncio.get_event_loop().time() - start > 10):
                    break
                continue

        if full_response:
            print()  # newline

        return full_response


async def voice_loop(gateway_url: str):
    """Main voice interaction loop."""
    print("\n" + "=" * 60)
    print("  OpenClaw Voice - Slice 0 (Ugly but Working)")
    print("=" * 60)
    print(f"  Gateway: {gateway_url}")
    print(f"  Recording: {RECORD_SECONDS}s fixed duration")
    print("  Press Enter to speak, 'm' to select microphone, Ctrl+C to quit")
    print("=" * 60 + "\n")

    selected_input_device = None

    while True:
        try:
            try:
                cmd = input("\n>>> Press Enter to start recording (m=mic, q=quit)...")
            except KeyboardInterrupt:
                print("\n")
                log("IDLE", "Goodbye!")
                break
            cmd = cmd.strip().lower()
            if cmd in {"q", "quit", "exit"}:
                log("IDLE", "Goodbye!")
                break
            if cmd in {"m", "mic", "microphone"}:
                selected_input_device = select_microphone()
                continue
            if cmd:
                log("IDLE", f"Unknown command '{cmd}', press Enter to record")
                continue

            log("IDLE", "Starting voice interaction")

            # Record audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = Path(f.name)

            audio_data = record_audio(device=selected_input_device)
            save_wav(audio_data, audio_path)

            # Transcribe
            transcription = transcribe(audio_path)

            if not transcription:
                log("IDLE", "No speech detected, try again")
                continue

            # Send to gateway
            log("WAITING", "Sending to OpenClaw...")
            try:
                response = await send_to_gateway(gateway_url, transcription)
            except ConnectionRefusedError:
                log("ERROR", "Gateway not running. Start with: openclaw gateway")
                continue
            except Exception as e:
                log("ERROR", f"Gateway error: {e}")
                continue

            if not response:
                log("IDLE", "No response received")
                continue

            # Speak response
            speak(response)

            log("IDLE", "Ready for next interaction")

        except KeyboardInterrupt:
            print("\n")
            log("IDLE", "Goodbye!")
            break
        except Exception as e:
            log("ERROR", f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Slice 0: Voice loop demo")
    parser.add_argument(
        "--gateway",
        default="ws://127.0.0.1:18789",
        help="Gateway WebSocket URL",
    )
    args = parser.parse_args()

    # Check dependencies
    missing = []
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")

    try:
        import websockets
    except ImportError:
        missing.append("websockets")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return

    # Check mlx-whisper (optional - will use mock if not available)
    try:
        import mlx_whisper
    except ImportError:
        print("Note: mlx-whisper not installed. Install with: pip install mlx-whisper")
        print("      Continuing with mock transcription...\n")

    asyncio.run(voice_loop(args.gateway))


if __name__ == "__main__":
    main()
