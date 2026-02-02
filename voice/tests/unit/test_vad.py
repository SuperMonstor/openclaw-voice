import numpy as np

from src.vad import SileroVAD, VADConfig


def test_stream_until_silence_transitions():
    sample_rate = 16000
    frame_samples = 1280  # 80ms

    vad = SileroVAD(
        VADConfig(
            threshold=0.5,
            min_speech_duration=0.16,  # 2 frames
            min_silence_duration=0.24,  # 3 frames
        )
    )

    # Monkeypatch is_speech for deterministic behavior.
    speech_pattern = ([True] * 4) + ([False] * 3)
    calls = {"i": 0}

    def fake_is_speech(frame, sr):
        idx = min(calls["i"], len(speech_pattern) - 1)
        calls["i"] += 1
        return speech_pattern[idx]

    vad.is_speech = fake_is_speech  # type: ignore[assignment]

    frame = np.zeros(frame_samples, dtype=np.int16)

    triggered = False
    for _ in range(len(speech_pattern)):
        triggered = vad.stream_until_silence(frame, sample_rate)

    assert triggered is True
