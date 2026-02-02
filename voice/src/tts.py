"""Text-to-speech helpers with AVSpeechSynthesizer and fallback to `say`."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Callable


LogFn = Callable[[str, str], None]


@dataclass
class TTSConfig:
    voice: str | None = None
    rate: float = 1.0
    prefer_avspeech: bool = True


class TTS:
    """TTS wrapper that prefers AVSpeechSynthesizer with cancellation support."""

    def __init__(self, config: TTSConfig | None = None, logger: LogFn | None = None):
        self._config = config or TTSConfig()
        self._log = logger
        self._engine = self._init_engine()

    def _init_engine(self):
        if self._config.prefer_avspeech:
            try:
                return _AVSpeechSynthesizerEngine(self._config, self._log)
            except Exception:
                if self._log:
                    self._log("TTS", "AVSpeechSynthesizer unavailable, falling back to say")
        return _SayEngine(self._config, self._log)

    def speak(self, text: str) -> None:
        self._engine.speak(text)

    def stop(self) -> None:
        self._engine.stop()


class _SayEngine:
    def __init__(self, config: TTSConfig, logger: LogFn | None):
        self._config = config
        self._log = logger
        self._proc: subprocess.Popen[str] | None = None

    def speak(self, text: str) -> None:
        self.stop()
        cmd = ["say"]
        if self._config.voice:
            cmd.extend(["-v", self._config.voice])
        cmd.append(text)
        if self._log:
            self._log("TTS", "Using macOS say command")
        self._proc = subprocess.Popen(cmd)
        self._proc.wait()
        self._proc = None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait()
        self._proc = None


class _AVSpeechSynthesizerEngine:
    def __init__(self, config: TTSConfig, logger: LogFn | None):
        self._config = config
        self._log = logger
        self._synthesizer, self._delegate = self._init_synthesizer()

    def _init_synthesizer(self):
        try:
            import objc  # type: ignore
            from AVFoundation import (  # type: ignore
                AVSpeechBoundaryImmediate,
                AVSpeechSynthesisVoice,
                AVSpeechSynthesizer,
                AVSpeechUtterance,
                AVSpeechUtteranceDefaultSpeechRate,
            )
            from Foundation import NSObject  # type: ignore
        except Exception as exc:
            raise RuntimeError("AVFoundation not available") from exc

        class _Delegate(NSObject):
            def init(self):
                self = objc.super(_Delegate, self).init()
                self.done = False
                return self

            def speechSynthesizer_didFinishSpeechUtterance_(self, synthesizer, utterance):
                self.done = True

            def speechSynthesizer_didCancelSpeechUtterance_(self, synthesizer, utterance):
                self.done = True

        self._objc = objc
        self._AVSpeechBoundaryImmediate = AVSpeechBoundaryImmediate
        self._AVSpeechSynthesisVoice = AVSpeechSynthesisVoice
        self._AVSpeechSynthesizer = AVSpeechSynthesizer
        self._AVSpeechUtterance = AVSpeechUtterance
        self._AVSpeechUtteranceDefaultSpeechRate = AVSpeechUtteranceDefaultSpeechRate
        self._NSObject = NSObject

        synthesizer = AVSpeechSynthesizer.alloc().init()
        delegate = _Delegate.alloc().init()
        synthesizer.setDelegate_(delegate)
        return synthesizer, delegate

    def _select_voice(self, voice_name: str | None):
        if not voice_name:
            return None
        voice = self._AVSpeechSynthesisVoice.voiceWithIdentifier_(voice_name)
        if voice:
            return voice
        voices = self._AVSpeechSynthesisVoice.speechVoices()
        for candidate in voices:
            if candidate.name() == voice_name:
                return candidate
        return None

    def speak(self, text: str) -> None:
        self.stop()
        utterance = self._AVSpeechUtterance.speechUtteranceWithString_(text)

        voice = self._select_voice(self._config.voice)
        if voice is not None:
            utterance.setVoice_(voice)

        base_rate = float(self._AVSpeechUtteranceDefaultSpeechRate)
        rate = max(0.1, min(2.0, float(self._config.rate)))
        utterance.setRate_(base_rate * rate)

        if self._log:
            self._log("TTS", "Using AVSpeechSynthesizer")

        self._delegate.done = False
        self._synthesizer.speakUtterance_(utterance)

        from Foundation import NSDate, NSRunLoop  # type: ignore

        run_loop = NSRunLoop.currentRunLoop()
        while not self._delegate.done:
            run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))

    def stop(self) -> None:
        self._synthesizer.stopSpeakingAtBoundary_(self._AVSpeechBoundaryImmediate)
        self._delegate.done = True
