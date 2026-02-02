"""Configuration management for OpenClaw Voice."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class WakeWordConfig:
    model: str = "hey_jarvis"
    threshold: float = 0.5


@dataclass
class STTConfig:
    model: str = "mlx-community/whisper-small"
    language: str = "en"


@dataclass
class TTSConfig:
    model: str = "en_US-lessac-medium"
    rate: float = 1.0


@dataclass
class GatewayConfig:
    url: str = "ws://127.0.0.1:18789"
    session_id: Optional[str] = None


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_size: int = 1280  # 80ms at 16kHz
    silence_threshold: float = 0.01
    silence_duration: float = 1.5  # seconds of silence to end recording
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.2


@dataclass
class Config:
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            wake_word=WakeWordConfig(**data.get("wake_word", {})),
            stt=STTConfig(**data.get("stt", {})),
            tts=TTSConfig(**data.get("tts", {})),
            gateway=GatewayConfig(**data.get("gateway", {})),
            audio=AudioConfig(**data.get("audio", {})),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            "wake_word": {
                "model": self.wake_word.model,
                "threshold": self.wake_word.threshold,
            },
            "stt": {
                "model": self.stt.model,
                "language": self.stt.language,
            },
            "tts": {
                "model": self.tts.model,
                "rate": self.tts.rate,
            },
            "gateway": {
                "url": self.gateway.url,
                "session_id": self.gateway.session_id,
            },
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "chunk_size": self.audio.chunk_size,
                "silence_threshold": self.audio.silence_threshold,
                "silence_duration": self.audio.silence_duration,
                "vad_threshold": self.audio.vad_threshold,
                "min_speech_duration": self.audio.min_speech_duration,
            },
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
