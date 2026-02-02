import subprocess

import pytest

import src.tts as tts


class DummyProcess:
    def __init__(self):
        self.terminated = False
        self.waited = False
        self._poll = None

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0

    def wait(self):
        self.waited = True


def test_tts_falls_back_to_say_when_avspeech_unavailable(monkeypatch):
    class FailingAV:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("no avspeech")

    monkeypatch.setattr(tts, "_AVSpeechSynthesizerEngine", FailingAV)
    engine = tts.TTS()
    assert isinstance(engine._engine, tts._SayEngine)


def test_say_engine_terminates_active_process(monkeypatch):
    dummy = DummyProcess()

    def fake_popen(_cmd):
        return dummy

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    engine = tts._SayEngine(tts.TTSConfig(), None)
    engine.speak("hello")
    assert dummy.waited
    engine._proc = dummy
    engine.stop()
    assert dummy.terminated
