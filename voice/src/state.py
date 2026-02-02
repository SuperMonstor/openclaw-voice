"""State machine for the voice engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VoiceState(Enum):
    IDLE = 0
    WAKE_DETECTED = 1
    LISTENING = 2
    TRANSCRIBING = 3
    RESPONDING = 4
    SPEAKING = 5


class VoiceEvent(Enum):
    WAKE_WORD_DETECTED = "wake_word_detected"
    LISTEN_START = "listen_start"
    UTTERANCE_END = "utterance_end"
    TRANSCRIPTION_DONE = "transcription_done"
    RESPONSE_START = "response_start"
    RESPONSE_END = "response_end"
    TTS_DONE = "tts_done"
    CANCEL = "cancel"


_TRANSITIONS: dict[tuple[VoiceState, VoiceEvent], VoiceState] = {
    (VoiceState.IDLE, VoiceEvent.WAKE_WORD_DETECTED): VoiceState.WAKE_DETECTED,
    (VoiceState.WAKE_DETECTED, VoiceEvent.LISTEN_START): VoiceState.LISTENING,
    (VoiceState.WAKE_DETECTED, VoiceEvent.CANCEL): VoiceState.IDLE,
    (VoiceState.LISTENING, VoiceEvent.UTTERANCE_END): VoiceState.TRANSCRIBING,
    (VoiceState.LISTENING, VoiceEvent.CANCEL): VoiceState.IDLE,
    (VoiceState.TRANSCRIBING, VoiceEvent.TRANSCRIPTION_DONE): VoiceState.RESPONDING,
    (VoiceState.TRANSCRIBING, VoiceEvent.CANCEL): VoiceState.IDLE,
    (VoiceState.RESPONDING, VoiceEvent.RESPONSE_START): VoiceState.SPEAKING,
    (VoiceState.RESPONDING, VoiceEvent.RESPONSE_END): VoiceState.IDLE,
    (VoiceState.RESPONDING, VoiceEvent.CANCEL): VoiceState.IDLE,
    (VoiceState.SPEAKING, VoiceEvent.TTS_DONE): VoiceState.IDLE,
    (VoiceState.SPEAKING, VoiceEvent.WAKE_WORD_DETECTED): VoiceState.LISTENING,
    (VoiceState.SPEAKING, VoiceEvent.CANCEL): VoiceState.IDLE,
}


def transition_for(state: VoiceState, event: VoiceEvent) -> VoiceState:
    try:
        return _TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise ValueError(f"Invalid transition from {state} on {event}") from exc


def valid_events(state: VoiceState) -> set[VoiceEvent]:
    return {event for (src, event), _ in _TRANSITIONS.items() if src == state}


@dataclass
class VoiceStateMachine:
    state: VoiceState = VoiceState.IDLE

    def transition(self, event: VoiceEvent) -> VoiceState:
        self.state = transition_for(self.state, event)
        return self.state
