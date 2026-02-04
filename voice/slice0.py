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
import tempfile
import argparse
import uuid
import time
import urllib.request
import threading
from datetime import datetime
from pathlib import Path
import re

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # Fixed recording duration for slice0
VAD_THRESHOLD = float(os.environ.get("OPENCLAW_VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH = float(os.environ.get("OPENCLAW_VAD_MIN_SPEECH", "0.2"))
VAD_MIN_SILENCE = float(os.environ.get("OPENCLAW_VAD_MIN_SILENCE", "1.0"))
PROTOCOL_VERSION = 3
DEFAULT_WAKEWORD_MODEL = "hey_jarvis"
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


def load_openclaw_config(log_errors: bool = False) -> dict | None:
    """Load OpenClaw config from disk."""
    config_path = os.environ.get(
        "OPENCLAW_CONFIG_PATH", str(Path.home() / ".openclaw" / "openclaw.json")
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        if log_errors:
            log("ERROR", f"Failed to read OpenClaw config: {type(exc).__name__}")
        return None


def load_gateway_token() -> str | None:
    """Load gateway auth token from env or OpenClaw config."""
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
    if token:
        return token

    config = load_openclaw_config(log_errors=True)
    if not config:
        return None
    return config.get("gateway", {}).get("auth", {}).get("token")


def load_assistant_name() -> str | None:
    """Load assistant name from OpenClaw workspace identity."""
    env_name = os.environ.get("OPENCLAW_ASSISTANT_NAME")
    if env_name:
        return env_name.strip() or None

    config = load_openclaw_config()
    workspace = None
    if config:
        workspace = (
            config.get("agents", {})
            .get("defaults", {})
            .get("workspace")
        )
    if not workspace:
        workspace = str(Path.home() / ".openclaw" / "workspace")

    identity_path = Path(workspace) / "IDENTITY.md"
    if not identity_path.exists():
        return None

    try:
        with open(identity_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^\s*-\s*\*\*Name:\*\*\s*(.+?)\s*$", line)
                if match:
                    name = match.group(1).strip()
                    return name or None
    except Exception:
        return None
    return None


def resolve_wakeword_model() -> str:
    """Resolve wake word model name from env or assistant identity."""
    override = os.environ.get("OPENCLAW_WAKEWORD_MODEL")
    if override:
        return override
    assistant_name = load_assistant_name()
    if not assistant_name:
        return DEFAULT_WAKEWORD_MODEL
    slug = re.sub(r"[^a-z0-9]+", "_", assistant_name.lower()).strip("_")
    if not slug:
        return DEFAULT_WAKEWORD_MODEL
    return f"hey_{slug}"


WAKEWORD_MODEL = resolve_wakeword_model()

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

    model_candidate = WAKEWORD_MODEL
    if model_candidate not in openwakeword.MODELS and not os.path.exists(model_candidate):
        if model_candidate != DEFAULT_WAKEWORD_MODEL:
            log(
                "WAKEWORD",
                f"Unknown wake word model '{model_candidate}', falling back to '{DEFAULT_WAKEWORD_MODEL}'",
            )
        model_candidate = DEFAULT_WAKEWORD_MODEL

    if model_candidate in openwakeword.MODELS:
        model_name = model_candidate
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
    elif os.path.exists(model_candidate):
        model_name = os.path.splitext(os.path.basename(model_candidate))[0]
        model_path = model_candidate
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

    # Model keys are derived from filenames (e.g., hey_jarvis_v0.1)
    model_keys = list(model.models.keys())
    if model_keys:
        model_name = model_keys[0]
    return model, model_name


def wait_for_wake_word(
    device: int | None = None, stop_event: threading.Event | None = None
):
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
                if stop_event is not None and stop_event.is_set():
                    return False
                try:
                    audio = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                audio = np.squeeze(audio)
                audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
                scores = model.predict(audio_int16)
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


def record_audio_until_silence(
    device: int | None = None, stop_event: threading.Event | None = None
) -> bytes | None:
    """Record audio until VAD detects end-of-utterance."""
    import sounddevice as sd
    import numpy as np

    from src.vad import SileroVAD, VADConfig

    vad = SileroVAD(
        VADConfig(
            threshold=VAD_THRESHOLD,
            min_speech_duration=VAD_MIN_SPEECH,
            min_silence_duration=VAD_MIN_SILENCE,
        )
    )

    frames: list[np.ndarray] = []
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames_count, time_info, status):
        if status:
            log("LISTENING", f"Audio status: {status}")
        audio_q.put(indata.copy())

    log("LISTENING", "Recording... Speak now!")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=1280,
        device=device,
        callback=callback,
    ):
        while True:
            if stop_event is not None and stop_event.is_set():
                log("LISTENING", "Stop requested, ending recording")
                return None
            try:
                frame = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            frame = np.squeeze(frame)
            frames.append(frame)
            audio_int16 = np.clip(frame * 32768.0, -32768, 32767).astype(np.int16)
            if vad.stream_until_silence(audio_int16, SAMPLE_RATE):
                break

    recording = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
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


_TTS_ENGINE = None


def speak(text: str):
    """Speak text using AVSpeechSynthesizer (fallback to `say`)."""
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        from src.tts import TTS

        _TTS_ENGINE = TTS(logger=log)
    log("SPEAKING", f"Speaking: {text[:50]}...")
    _TTS_ENGINE.speak(text)
    log("SPEAKING", "Done speaking")


def _extract_content_from_message(message) -> str:
    """Match OpenClaw's message text extraction (content-only, no thinking)."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        stop_reason = message.get("stopReason", "")
        if stop_reason == "error":
            error_message = message.get("errorMessage", "")
            return str(error_message).strip()
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    if not parts:
        stop_reason = message.get("stopReason", "")
        if stop_reason == "error":
            error_message = message.get("errorMessage", "")
            return str(error_message).strip()
    return "\n".join(part.strip() for part in parts if part).strip()


def _append_text(full_response: str, text: str, on_chunk) -> str:
    if not text:
        return full_response
    full_response += text
    print(text, end="", flush=True)
    if on_chunk:
        on_chunk(text)
    return full_response


async def send_to_gateway(url: str, message: str, on_chunk=None) -> str:
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
                if msg_type == "res":
                    payload = data.get("payload", {})
                    if isinstance(payload, dict) and payload.get("message"):
                        text = _extract_content_from_message(payload.get("message"))
                        full_response = _append_text(full_response, text, on_chunk)
                elif msg_type == "event":
                    event_name = data.get("event", "")
                    payload = data.get("payload", {})

                    if event_name == "agent":
                        stream = payload.get("stream")
                        data_payload = payload.get("data", {})
                        if stream == "assistant" and isinstance(data_payload, dict):
                            text = data_payload.get("text")
                            if isinstance(text, str):
                                full_response = _append_text(full_response, text, on_chunk)
                        phase = data_payload.get("phase") if isinstance(data_payload, dict) else None
                        if phase == "end":
                            log("GATEWAY", "Response complete")
                            break
                    elif event_name.startswith("chat"):
                        state = payload.get("state")
                        message = payload.get("message")
                        if isinstance(message, str):
                            full_response = _append_text(full_response, message, on_chunk)
                        else:
                            text = _extract_content_from_message(message)
                            full_response = _append_text(full_response, text, on_chunk)
                        if state in {"final", "error", "aborted"}:
                            if not full_response:
                                log(
                                    "GATEWAY",
                                    "Chat final with no text; payload="
                                    f"{json.dumps(payload)[:200]}",
                                )
                            log("GATEWAY", f"Chat state: {state}")
                            break
                    elif "delta" in event_name or "text" in event_name.lower():
                        if isinstance(payload, dict) and payload.get("message"):
                            text = _extract_content_from_message(payload.get("message"))
                            full_response = _append_text(full_response, text, on_chunk)

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
    print(
        "  Recording: VAD-based (threshold "
        f"{VAD_THRESHOLD}, min_speech {VAD_MIN_SPEECH}s, min_silence {VAD_MIN_SILENCE}s)"
    )
    print(f"  Wake word: {WAKEWORD_MODEL} (threshold {WAKEWORD_THRESHOLD})")
    default_mic = get_default_input_device_name()
    if default_mic:
        print(f"  Default mic: {default_mic}")
    print("  Listening for wake word immediately. Ctrl+C to quit.")
    print("=" * 60 + "\n")

    selected_input_device = None
    env_device = os.environ.get("OPENCLAW_INPUT_DEVICE")
    if env_device is not None:
        try:
            selected_input_device = int(env_device)
            log("IDLE", f"Using input device {selected_input_device} from OPENCLAW_INPUT_DEVICE")
        except ValueError:
            log("ERROR", f"Invalid OPENCLAW_INPUT_DEVICE: {env_device}")

    from src.ui_socket import UISocketServer

    ui_socket = UISocketServer()
    stop_event = threading.Event()

    def handle_command(command: str, payload: dict):
        if command == "shutdown":
            log("IDLE", "Shutdown requested by UI")
            stop_event.set()

    ui_socket.on_command = handle_command
    ui_socket.start()
    ui_socket.send_event("state_change", {"state": "idle"})

    try:
        while True:
            if stop_event.is_set():
                break
            log("WAKEWORD", "Waiting for wake word...")
            ui_socket.send_event("state_change", {"state": "idle"})
            if wait_for_wake_word(
                device=selected_input_device, stop_event=stop_event
            ) is False:
                log("IDLE", "Goodbye!")
                break
            ui_socket.send_event("state_change", {"state": "listening"})
            log("IDLE", "Wake word detected, starting voice interaction")

            # Record audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = Path(f.name)

            audio_data = record_audio_until_silence(
                device=selected_input_device, stop_event=stop_event
            )
            if stop_event.is_set() or audio_data is None:
                log("IDLE", "Stopping voice loop")
                break
            save_wav(audio_data, audio_path)

            # Transcribe
            ui_socket.send_event("state_change", {"state": "transcribing"})
            transcription = transcribe(audio_path)
            if transcription:
                ui_socket.send_event("transcription", {"text": transcription})

            if not transcription:
                log("IDLE", "No speech detected, try again")
                continue

            # Send to gateway
            log("WAITING", "Sending to OpenClaw...")
            ui_socket.send_event("state_change", {"state": "responding"})
            try:
                response = await send_to_gateway(
                    gateway_url,
                    transcription,
                    on_chunk=lambda chunk: ui_socket.send_event(
                        "response_chunk", {"text": chunk}
                    ),
                )
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
            ui_socket.send_event("state_change", {"state": "speaking"})
            speak(response)

            ui_socket.send_event("state_change", {"state": "idle"})
            log("IDLE", "Ready for next interaction")
    finally:
        ui_socket.send_event("state_change", {"state": "idle"})
        ui_socket.stop()
        if _TTS_ENGINE is not None:
            _TTS_ENGINE.stop()


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

    try:
        asyncio.run(voice_loop(args.gateway))
    except KeyboardInterrupt:
        # Suppress asyncio.run() KeyboardInterrupt traceback on Ctrl+C.
        pass


if __name__ == "__main__":
    main()
