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
import queue
import subprocess
import tempfile
import argparse
import uuid
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # Fixed recording duration for slice0
PROTOCOL_VERSION = 3
WAKEWORD_MODEL = os.environ.get("OPENCLAW_WAKEWORD_MODEL", "hey_jarvis")
WAKEWORD_THRESHOLD = float(os.environ.get("OPENCLAW_WAKEWORD_THRESHOLD", "0.5"))
WAKEWORD_DEBOUNCE_SEC = 1.0
WAKEWORD_DEBUG = os.environ.get("OPENCLAW_WAKEWORD_DEBUG", "0") == "1"


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


def get_default_input_device_name() -> str | None:
    """Return the default input device name if available."""
    try:
        import sounddevice as sd
    except ImportError:
        return None
    try:
        default_idx = sd.default.device[0] if sd.default.device else None
        if default_idx is None:
            return None
        info = sd.query_devices(default_idx)
        return info.get("name")
    except Exception:
        return None


def load_gateway_token() -> str | None:
    """Load gateway auth token from env or OpenClaw config."""
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
    if token:
        return token

    config_path = os.environ.get(
        "OPENCLAW_CONFIG_PATH", str(Path.home() / ".openclaw" / "openclaw.json")
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("gateway", {}).get("auth", {}).get("token")
    except FileNotFoundError:
        return None
    except Exception as exc:
        log("ERROR", f"Failed to read OpenClaw config: {type(exc).__name__}")
        return None


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


def init_wake_word_model():
    """Initialize openWakeWord model."""
    try:
        import openwakeword
    except ImportError:
        log("ERROR", "openwakeword not installed. Install with: pip install openwakeword")
        return None, None

    def ensure_file(url: str, path: Path):
        if path.exists():
            return
        log("WAKEWORD", f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)
        log("WAKEWORD", f"Downloaded {path.name}")

    desired_inference = None
    if os.path.exists(WAKEWORD_MODEL):
        if WAKEWORD_MODEL.endswith(".onnx"):
            desired_inference = "onnx"
        elif WAKEWORD_MODEL.endswith(".tflite"):
            desired_inference = "tflite"

    if desired_inference == "tflite" or desired_inference is None:
        try:
            import tflite_runtime.interpreter  # type: ignore
            inference_framework = "tflite"
        except Exception:
            inference_framework = None
    else:
        inference_framework = None

    if desired_inference == "onnx" or inference_framework is None:
        try:
            import onnxruntime  # type: ignore
            inference_framework = "onnx"
        except Exception:
            inference_framework = None

    if inference_framework is None:
        log("ERROR", "Missing tflite-runtime and onnxruntime")
        log("ERROR", "Install one of: pip install tflite-runtime | pip install onnxruntime")
        return None, None

    target_dir = Path(__file__).resolve().parent.parent / "models" / "wakeword"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure feature models for the selected framework
    feature_suffix = ".onnx" if inference_framework == "onnx" else ".tflite"
    melspec_url = openwakeword.FEATURE_MODELS["melspectrogram"]["download_url"].replace(
        ".tflite", feature_suffix
    )
    embed_url = openwakeword.FEATURE_MODELS["embedding"]["download_url"].replace(
        ".tflite", feature_suffix
    )
    melspec_path = target_dir / f"melspectrogram{feature_suffix}"
    embed_path = target_dir / f"embedding_model{feature_suffix}"
    ensure_file(melspec_url, melspec_path)
    ensure_file(embed_url, embed_path)

    if WAKEWORD_MODEL in openwakeword.MODELS:
        model_name = WAKEWORD_MODEL
        model_info = openwakeword.MODELS[model_name]
        base_path = model_info["model_path"].replace(".tflite", feature_suffix)
        base_url = model_info.get("download_url")
        if base_url:
            base_url = base_url.replace(".tflite", feature_suffix)
        target_path = target_dir / os.path.basename(base_path)
        if base_url:
            ensure_file(base_url, target_path)
        elif os.path.exists(base_path):
            target_path = Path(base_path)
        else:
            log("ERROR", f"Wake word model not found: {base_path}")
            return None, None
        model_path = str(target_path)
    elif os.path.exists(WAKEWORD_MODEL):
        model_name = os.path.splitext(os.path.basename(WAKEWORD_MODEL))[0]
        model_path = WAKEWORD_MODEL
        if model_path.endswith(".onnx"):
            inference_framework = "onnx"
            feature_suffix = ".onnx"
        elif model_path.endswith(".tflite"):
            inference_framework = "tflite"
            feature_suffix = ".tflite"
    else:
        log("ERROR", f"Unknown wake word model: {WAKEWORD_MODEL}")
        return None, None

    try:
        model = openwakeword.Model(
            wakeword_models=[model_path],
            inference_framework=inference_framework,
            melspec_model_path=str(melspec_path),
            embedding_model_path=str(embed_path),
        )
    except Exception as exc:
        log("ERROR", f"Failed to load wake word model: {type(exc).__name__}")
        return None, None
    return model, model_name


def wait_for_wake_word(device: int | None = None):
    """Block until wake word is detected."""
    import sounddevice as sd
    import numpy as np

    model, model_name = init_wake_word_model()
    if not model:
        raise RuntimeError("openwakeword model failed to initialize")

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            log("WAKEWORD", f"Audio status: {status}")
        audio_q.put(indata.copy())

    log("WAKEWORD", f"Listening for '{model_name}' (threshold {WAKEWORD_THRESHOLD})")
    last_trigger = 0.0
    last_debug = 0.0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=1280,
        device=device,
        callback=callback,
    ):
        try:
            while True:
                audio = audio_q.get()
                audio = np.squeeze(audio)
                scores = model.predict(audio)
                score = scores.get(model_name, 0.0)
                now = time.time()
                if WAKEWORD_DEBUG and (now - last_debug) >= 1.0:
                    rms = float(np.sqrt(np.mean(audio**2)))
                    log("WAKEWORD", f"Score {score:.3f} RMS {rms:.4f}")
                    last_debug = now
                if score >= WAKEWORD_THRESHOLD and (now - last_trigger) >= WAKEWORD_DEBOUNCE_SEC:
                    last_trigger = now
                    log("WAKEWORD", f"Detected '{model_name}' (score {score:.2f})")
                    return True
        except KeyboardInterrupt:
            return False


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
        challenge_payload = challenge_data.get("payload", {})
        nonce = challenge_payload.get("nonce")
        ts = challenge_payload.get("ts")

        if challenge_data.get("event") == "connect.challenge":
            gateway_token = load_gateway_token()
            auth_token = gateway_token
            if not auth_token:
                raise Exception(
                    "Missing gateway token. Set OPENCLAW_GATEWAY_TOKEN or run OpenClaw onboarding."
                )

            # Authenticate (protocol v3)
            connect_req = {
                "type": "req",
                "id": str(uuid.uuid4()),
                "method": "connect",
                "params": {
                    "minProtocol": PROTOCOL_VERSION,
                    "maxProtocol": PROTOCOL_VERSION,
                    "client": {
                        "id": "gateway-client",
                        "version": "0.1.0",
                        "platform": "macos",
                        "mode": "backend",
                    },
                    "role": "operator",
                    "scopes": ["operator.admin", "operator.approvals", "operator.pairing"],
                    "caps": [],
                    "commands": [],
                    "permissions": {},
                    "auth": {"token": auth_token},
                    "locale": "en-US",
                    "userAgent": "openclaw-voice/0.1.0",
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
            "id": str(uuid.uuid4()),
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
                    elif event_name == "chat":
                        state = payload.get("state")
                        message = payload.get("message", {})
                        content = message.get("content", [])
                        if isinstance(content, dict):
                            content = [content]
                        if isinstance(content, list):
                            for item in content:
                                if not isinstance(item, dict):
                                    continue
                                if item.get("type") != "text":
                                    continue
                                text = item.get("text", "")
                                if text:
                                    full_response += text
                                    print(text, end="", flush=True)
                        if state in {"final", "error", "aborted"}:
                            log("GATEWAY", f"Chat state: {state}")
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
    print(f"  Wake word: {WAKEWORD_MODEL} (threshold {WAKEWORD_THRESHOLD})")
    default_mic = get_default_input_device_name()
    if default_mic:
        print(f"  Default mic: {default_mic}")
    print("  Press Enter to start wake-word listening, 'm' to select microphone, Ctrl+C to quit")
    print("=" * 60 + "\n")

    selected_input_device = None

    while True:
        try:
            try:
                cmd = input("\n>>> Press Enter to start wake-word listening (m=mic, q=quit)...")
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
                log("IDLE", f"Unknown command '{cmd}', press Enter to listen")
                continue

            log("WAKEWORD", "Waiting for wake word...")
            if wait_for_wake_word(device=selected_input_device) is False:
                log("IDLE", "Goodbye!")
                break
            log("IDLE", "Wake word detected, starting voice interaction")

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
