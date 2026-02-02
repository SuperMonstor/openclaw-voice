"""Voice activity detection using Silero via openwakeword."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_speech_duration: float = 0.2
    min_silence_duration: float = 1.0


class SileroVAD:
    """Thin wrapper around openwakeword's Silero VAD."""

    def __init__(self, config: VADConfig):
        try:
            import openwakeword
        except ImportError as exc:
            raise RuntimeError("openwakeword is required for VAD") from exc

        self._ensure_vad_model(openwakeword)

        self._vad = openwakeword.VAD()
        self._threshold = float(config.threshold)
        self._min_speech = float(config.min_speech_duration)
        self._min_silence = float(config.min_silence_duration)

        self._speech_started = False
        self._speech_duration = 0.0
        self._silence_duration = 0.0

    def reset(self):
        self._speech_started = False
        self._speech_duration = 0.0
        self._silence_duration = 0.0

    @staticmethod
    def _ensure_vad_model(openwakeword_module):
        """Download Silero VAD model if missing."""
        import os
        import urllib.request
        from pathlib import Path

        vad_info = openwakeword_module.VAD_MODELS["silero_vad"]
        model_path = vad_info["model_path"]
        if os.path.exists(model_path):
            return

        target_dir = Path(model_path).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        url = vad_info["download_url"]
        urllib.request.urlretrieve(url, model_path)

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        """Return True if this frame is speech above threshold."""
        score = float(self._vad.predict(frame, sample_rate))
        return score >= self._threshold

    def stream_until_silence(self, frame: np.ndarray, sample_rate: int) -> bool:
        """Update internal counters. Return True if end-of-utterance is detected."""
        frame_duration = len(frame) / float(sample_rate)

        if self.is_speech(frame, sample_rate):
            self._speech_duration += frame_duration
            if self._speech_duration >= self._min_speech:
                self._speech_started = True
            self._silence_duration = 0.0
        else:
            if self._speech_started:
                self._silence_duration += frame_duration

        return self._speech_started and self._silence_duration >= self._min_silence
