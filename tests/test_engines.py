import pytest
from autosub_player import get_transcription_engine, WhisperEngine, CanaryQwenEngine, Qwen3Engine

def test_explicit_engine():
    assert isinstance(get_transcription_engine("whisper", None, False, "tiny"), WhisperEngine)
    assert isinstance(get_transcription_engine("canary-qwen", None, False, "tiny"), CanaryQwenEngine)
    assert isinstance(get_transcription_engine("qwen3-asr", None, False, "tiny"), Qwen3Engine)

def test_auto_selection_english():
    assert isinstance(get_transcription_engine(None, "en", False, "tiny"), CanaryQwenEngine)
    assert isinstance(get_transcription_engine(None, "english", False, "tiny"), CanaryQwenEngine)
    assert isinstance(get_transcription_engine(None, "EN", False, "tiny"), CanaryQwenEngine)

def test_auto_selection_other():
    assert isinstance(get_transcription_engine(None, "ja", False, "tiny"), Qwen3Engine)
    assert isinstance(get_transcription_engine(None, "de", False, "tiny"), Qwen3Engine)

def test_auto_selection_none():
    assert isinstance(get_transcription_engine(None, None, False, "tiny"), Qwen3Engine)

def test_small_model_flag():
    engine = get_transcription_engine("qwen3-asr", None, True, "tiny")
    assert isinstance(engine, Qwen3Engine)
    assert engine.use_small_model is True
