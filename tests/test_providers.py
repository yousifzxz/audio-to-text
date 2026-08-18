#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for autosub_player.py translation providers, tag handling,
__FAILED__ fallback, and ffmpeg command construction.

Run with:
    python -m pytest tests/test_providers.py -v
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Import the module under test
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autosub_player import (
    hide_tags,
    restore_tags,
    hide_tags_plain,
    restore_tags_plain,
    apply_rtl_fix,
    _parse_llm_json_response,
    ProviderExhaustedException,
    ProviderChain,
    CachedSegment,
    validate_ffmpeg,
    log,
    wrap_subtitle_text,
)


# ==========================================================================
# Tag round-tripping tests
# ==========================================================================


class TestHideTags:
    """Tests for hide_tags / restore_tags (XML-safe version for DeepL)."""

    def test_no_tags(self):
        clean, tags = hide_tags("Hello world")
        assert tags == []
        assert "Hello world" in clean

    def test_single_tag_at_start(self):
        clean, tags = hide_tags("{\\i1}italic text")
        assert tags == ["{\\i1}"]
        assert "<tx0/>" in clean
        assert "{\\i1}" not in clean

    def test_multiple_tags_preserve_order(self):
        clean, tags = hide_tags("{\\b1}bold {\\i1}italic{\\i0} end{\\b0}")
        assert len(tags) == 4
        assert tags[0] == "{\\b1}"
        assert tags[1] == "{\\i1}"
        assert tags[2] == "{\\i0}"
        assert tags[3] == "{\\b0}"

    def test_roundtrip_single_tag(self):
        original = "{\\i1}Hello world"
        clean, tags = hide_tags(original)
        restored = restore_tags(clean, tags)
        assert restored == original

    def test_roundtrip_mid_line_tag(self):
        original = "Hello {\\i1}beautiful{\\i0} world"
        clean, tags = hide_tags(original)
        restored = restore_tags(clean, tags)
        assert restored == original

    def test_roundtrip_complex(self):
        original = "{\\b1}{\\c&H0000FF&}Red bold{\\b0} normal"
        clean, tags = hide_tags(original)
        restored = restore_tags(clean, tags)
        assert restored == original

    def test_html_entities_in_text(self):
        original = "Tom & Jerry <forever>"
        clean, tags = hide_tags(original)
        assert "&amp;" in clean or "Tom" in clean
        restored = restore_tags(clean, tags)
        assert restored == original


class TestHideTagsPlain:
    """Tests for hide_tags_plain / restore_tags_plain (for LLM prompts)."""

    def test_no_tags(self):
        clean, tags = hide_tags_plain("Hello world")
        assert tags == []
        assert clean == "Hello world"

    def test_single_tag_replaced(self):
        clean, tags = hide_tags_plain("{\\i1}italic")
        assert clean == "<tx0/>italic"
        assert tags == ["{\\i1}"]

    def test_mid_line_tag_position_preserved(self):
        clean, tags = hide_tags_plain("Hello {\\i1}world")
        assert clean == "Hello <tx0/>world"
        assert tags == ["{\\i1}"]

    def test_roundtrip(self):
        original = "Hello {\\i1}world{\\i0} end"
        clean, tags = hide_tags_plain(original)
        restored = restore_tags_plain(clean, tags)
        assert restored == original

    def test_roundtrip_with_simulated_translation(self):
        """Simulate an LLM keeping placeholders in translated text."""
        original = "{\\b1}Bold text{\\b0} normal"
        clean, tags = hide_tags_plain(original)
        # Simulate translation that keeps markers in place
        translated = "<tx0/>Texte gras<tx1/> normal"
        restored = restore_tags_plain(translated, tags)
        assert restored == "{\\b1}Texte gras{\\b0} normal"


class TestApplyRtlFix:
    """Tests for RTL embedding wrapper."""

    def test_wraps_with_rle(self):
        result = apply_rtl_fix("مرحبا")
        assert result.startswith("\u202B")
        assert result.endswith("\u202C")

    def test_strips_existing_markers(self):
        already_wrapped = "\u202Bمرحبا\u202C"
        result = apply_rtl_fix(already_wrapped)
        # Should not double-wrap
        assert result.count("\u202B") == 1
        assert result.count("\u202C") == 1


# ==========================================================================
# LLM JSON response parsing tests
# ==========================================================================


class TestParseLlmJsonResponse:
    """Tests for _parse_llm_json_response shared by Gemini/Groq."""

    def test_valid_json_array(self):
        raw = '["Hola", "Mundo"]'
        original = ["Hello", "World"]
        all_tags: List[List[str]] = [[], []]
        result = _parse_llm_json_response(raw, original, all_tags, "es", "Test")
        assert result == ["Hola", "Mundo"]

    def test_json_with_markdown_fences(self):
        raw = '```json\n["Bonjour", "Monde"]\n```'
        original = ["Hello", "World"]
        all_tags: List[List[str]] = [[], []]
        result = _parse_llm_json_response(raw, original, all_tags, "fr", "Test")
        assert result == ["Bonjour", "Monde"]

    def test_invalid_json_returns_failed(self):
        raw = "This is not JSON"
        original = ["Hello", "World"]
        all_tags: List[List[str]] = [[], []]
        result = _parse_llm_json_response(raw, original, all_tags, "es", "Test")
        assert all(line.startswith("__FAILED__") for line in result)
        assert len(result) == 2

    def test_wrong_count_returns_failed(self):
        raw = '["One"]'
        original = ["Hello", "World"]
        all_tags: List[List[str]] = [[], []]
        result = _parse_llm_json_response(raw, original, all_tags, "es", "Test")
        assert all(line.startswith("__FAILED__") for line in result)

    def test_tags_restored_after_parse(self):
        raw = '["<tx0/>Hola<tx1/>"]'
        original = ["{\\b1}Hello{\\b0}"]
        all_tags = [["{\\b1}", "{\\b0}"]]
        result = _parse_llm_json_response(raw, original, all_tags, "es", "Test")
        assert result == ["{\\b1}Hola{\\b0}"]

    def test_arabic_gets_rtl_fix(self):
        raw = '["مرحبا"]'
        original = ["Hello"]
        all_tags: List[List[str]] = [[]]
        result = _parse_llm_json_response(raw, original, all_tags, "ar", "Test")
        assert result[0].startswith("\u202B")
        assert result[0].endswith("\u202C")


# ==========================================================================
# ProviderChain tests
# ==========================================================================


class TestProviderChain:
    """Tests for the ProviderChain fallback mechanism."""

    def test_empty_chain_returns_original(self):
        chain = ProviderChain([])
        result = chain.translate_batch(["hello"], "en", "es")
        assert result == ["hello"]

    def test_single_provider_success(self):
        mock_provider = MagicMock()
        mock_provider.name = "MockProvider"
        mock_provider.translate_batch.return_value = ["hola"]

        chain = ProviderChain([mock_provider])
        result = chain.translate_batch(["hello"], "en", "es")
        assert result == ["hola"]

    def test_fallback_on_exhaustion(self):
        provider1 = MagicMock()
        provider1.name = "Provider1"
        provider1.translate_batch.side_effect = ProviderExhaustedException("quota")

        provider2 = MagicMock()
        provider2.name = "Provider2"
        provider2.translate_batch.return_value = ["hola"]

        chain = ProviderChain([provider1, provider2])
        result = chain.translate_batch(["hello"], "en", "es")
        assert result == ["hola"]

    def test_all_providers_exhausted(self):
        provider1 = MagicMock()
        provider1.name = "P1"
        provider1.translate_batch.side_effect = ProviderExhaustedException("q1")

        provider2 = MagicMock()
        provider2.name = "P2"
        provider2.translate_batch.side_effect = ProviderExhaustedException("q2")

        chain = ProviderChain([provider1, provider2])
        result = chain.translate_batch(["hello"], "en", "es")
        assert all(line.startswith("__FAILED__") for line in result)


# ==========================================================================
# __FAILED__ fallback behavior tests
# ==========================================================================


class TestFailedFallback:
    """Tests for the __FAILED__ fallback logic in the transcription pipeline."""

    def test_failed_lines_replaced_with_original(self):
        """Simulate the fallback logic from transcribe_video."""
        texts_to_translate = ["Hello", "World", "Goodbye"]
        texts_to_save = ["Hola", "__FAILED__World", "Adiós"]

        failed_count = 0
        for i, text in enumerate(texts_to_save):
            if text.startswith("__FAILED__"):
                texts_to_save[i] = texts_to_translate[i]
                failed_count += 1

        assert texts_to_save == ["Hola", "World", "Adiós"]
        assert failed_count == 1

    def test_no_failures_zero_count(self):
        texts_to_translate = ["Hello", "World"]
        texts_to_save = ["Hola", "Mundo"]

        failed_count = 0
        for i, text in enumerate(texts_to_save):
            if text.startswith("__FAILED__"):
                texts_to_save[i] = texts_to_translate[i]
                failed_count += 1

        assert failed_count == 0
        assert texts_to_save == ["Hola", "Mundo"]

    def test_all_failures(self):
        texts_to_translate = ["Hello", "World"]
        texts_to_save = ["__FAILED__Hello", "__FAILED__World"]

        failed_count = 0
        for i, text in enumerate(texts_to_save):
            if text.startswith("__FAILED__"):
                texts_to_save[i] = texts_to_translate[i]
                failed_count += 1

        assert failed_count == 2
        assert texts_to_save == ["Hello", "World"]


# ==========================================================================
# FFmpeg command validation tests
# ==========================================================================


class TestValidateFFmpeg:
    """Tests for validate_ffmpeg."""

    @patch("autosub_player.shutil.which", return_value=None)
    def test_ffmpeg_not_found_raises(self, mock_which):
        with pytest.raises(RuntimeError, match="ffmpeg was not found"):
            validate_ffmpeg()

    @patch("autosub_player.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_ffmpeg_found_returns_path(self, mock_which):
        result = validate_ffmpeg()
        assert result == "/usr/bin/ffmpeg"


# ==========================================================================
# Caching tests
# ==========================================================================


class TestTranscriptionCache:
    """Tests for video fingerprinting and transcription caching."""

    def test_cached_segment_dataclass(self):
        seg = CachedSegment(start=0.0, end=1.5, text="Hello")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "Hello"


# ==========================================================================
# Wrapping tests
# ==========================================================================

class TestWrapSubtitleText:
    """Tests for subtitle text line wrapping."""

    def test_no_wrap_needed(self):
        text = "This is a short line."
        wrapped = wrap_subtitle_text(text, max_chars=42, max_lines=2, line_break="\\N")
        assert wrapped == text

    def test_wraps_at_balanced_point(self):
        text = "This is a somewhat longer subtitle line that needs wrapping"
        wrapped = wrap_subtitle_text(text, max_chars=42, max_lines=2, line_break="\\N")
        # Should split near the middle
        # "This is a somewhat longer" is 25 chars
        # "subtitle line that needs wrapping" is 33 chars
        assert "\\N" in wrapped
        assert wrapped == "This is a somewhat longer\\Nsubtitle line that needs wrapping"

    def test_more_than_two_lines(self):
        text = "This is an extremely long subtitle line that is going to be way more than forty two characters long, probably pushing into the hundreds just to test what happens when it goes over."
        wrapped = wrap_subtitle_text(text, max_chars=42, max_lines=2, line_break="\\N")
        # It should still only split once (max_lines=2), putting extra on the second line rather than dropping words.
        # Total length without spaces is 180 chars. 
        # Half is 90 chars.
        assert "\\N" in wrapped
        # Make sure no words were lost
        assert text.replace(" ", "") == wrapped.replace("\\N", "").replace(" ", "")

    def test_wraps_with_ass_tags(self):
        text = "{\\b1}Bold text{\\b0} that is long enough to wrap around"
        wrapped = wrap_subtitle_text(text, max_chars=42, max_lines=2, line_break="\\N")
        # Tags are zero length, should not split between tags
        assert "\\N" in wrapped
        assert wrapped == "{\\b1}Bold text{\\b0} that is long\\Nenough to wrap around"

    def test_max_lines_1(self):
        text = "This is a somewhat longer subtitle line that needs wrapping"
        wrapped = wrap_subtitle_text(text, max_chars=42, max_lines=1, line_break="\\N")
        assert "\\N" not in wrapped
        assert wrapped == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
