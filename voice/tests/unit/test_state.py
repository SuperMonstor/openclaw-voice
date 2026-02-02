import pytest

from src.state import VoiceEvent, VoiceState, VoiceStateMachine, transition_for, valid_events


def test_transition_table_allows_happy_path():
    state = VoiceState.IDLE
    state = transition_for(state, VoiceEvent.WAKE_WORD_DETECTED)
    assert state is VoiceState.WAKE_DETECTED

    state = transition_for(state, VoiceEvent.LISTEN_START)
    assert state is VoiceState.LISTENING

    state = transition_for(state, VoiceEvent.UTTERANCE_END)
    assert state is VoiceState.TRANSCRIBING

    state = transition_for(state, VoiceEvent.TRANSCRIPTION_DONE)
    assert state is VoiceState.RESPONDING

    state = transition_for(state, VoiceEvent.RESPONSE_START)
    assert state is VoiceState.SPEAKING

    state = transition_for(state, VoiceEvent.TTS_DONE)
    assert state is VoiceState.IDLE


def test_barge_in_from_speaking_to_listening():
    state = transition_for(VoiceState.SPEAKING, VoiceEvent.WAKE_WORD_DETECTED)
    assert state is VoiceState.LISTENING


def test_invalid_transition_raises():
    with pytest.raises(ValueError):
        transition_for(VoiceState.IDLE, VoiceEvent.TTS_DONE)


def test_state_machine_updates_state():
    machine = VoiceStateMachine()
    assert machine.state is VoiceState.IDLE

    machine.transition(VoiceEvent.WAKE_WORD_DETECTED)
    assert machine.state is VoiceState.WAKE_DETECTED


def test_valid_events_exposes_available_inputs():
    events = valid_events(VoiceState.WAKE_DETECTED)
    assert VoiceEvent.LISTEN_START in events
    assert VoiceEvent.CANCEL in events
