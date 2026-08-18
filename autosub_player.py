#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autosub_player.py
==================

Automatically transcribes a video's audio track using faster-whisper,
generates a subtitle file (.srt or .ass) with pysubs2, and launches
mpv with the subtitles pre-loaded.

--------------------------------------------------------------------
requirements.txt
--------------------------------------------------------------------
faster-whisper>=1.0.0
pysubs2>=1.6.0
ctranslate2>=4.0.0
deepl
google-genai
groq

# System dependencies (not installable via pip):
#   - mpv media player (https://mpv.io) must be on PATH for auto-playback.
#   - ffmpeg must be on PATH (faster-whisper/ctranslate2 relies on it for
#     audio decoding of arbitrary container formats such as .mkv/.webm).
#   - Tkinter is part of the Python standard library on most platforms,
#     but on some Linux distros it must be installed separately, e.g.:
#       sudo apt-get install python3-tk
#
# Optional (for GPU acceleration):
#   - NVIDIA CUDA toolkit + cuDNN for float16 inference on CUDA devices.
#   - ROCm-enabled ctranslate2 build for AMD GPUs (falls back to CPU
#     automatically if unavailable).
--------------------------------------------------------------------

Usage:
    python autosub_player.py --video movie.mkv --model small --lang en --format srt
    python autosub_player.py                     # opens a file picker dialog
    python autosub_player.py --video clip.mp4 --no-play

Author: Autosub Player
License: MIT
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import os
import re
import html
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mkv", ".mp4", ".avi", ".mov", ".webm",
)

SUPPORTED_SUBTITLE_FORMATS: tuple[str, ...] = ("srt", "ass")

DEFAULT_MODEL_SIZE = "small"
DEFAULT_LANGUAGE: Optional[str] = None  # None => auto-detect
DEFAULT_FORMAT = "srt"

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Container for the outcome of a transcription run."""

    subtitle_path: Path
    detected_language: str
    language_probability: float
    duration_seconds: float
    processing_seconds: float
    segment_count: int


# --------------------------------------------------------------------------
# Device / compute type selection
# --------------------------------------------------------------------------


def select_device_and_compute_type() -> tuple[str, str]:
    """
    Determine the best available device and matching compute type for
    faster-whisper / ctranslate2.

    Returns:
        A tuple of (device, compute_type). device is one of "cuda" or
        "cpu" (ctranslate2 does not expose a distinct "rocm" device
        string -- ROCm-enabled builds are still addressed as "cuda"
        internally by ctranslate2's HIP backend, so we probe with
        torch when available and otherwise fall back safely to CPU).
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            # Covers both genuine CUDA GPUs and ROCm/HIP builds of torch,
            # which report themselves through the same torch.cuda API.
            return "cuda", "float16"
    except ImportError:
        # torch is not a hard dependency of this script; if it is not
        # installed we simply cannot probe for a GPU and fall back to CPU.
        pass
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] GPU probing failed ({exc}); falling back to CPU.")

    return "cpu", "int8"


# --------------------------------------------------------------------------
# File picker dialog (Tkinter, dark theme)
# --------------------------------------------------------------------------


def prompt_for_settings() -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """
    Launch a minimal, dark-themed Tkinter dialog allowing the user to
    configure languages and browse for a video file.

    Returns:
        A tuple of (video_path, source_language, target_language).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError:
        print(
            "[error] Tkinter is not available in this Python installation. "
            "Please supply --video explicitly.",
            file=sys.stderr,
        )
        return None, None, None

    bg = "#1e1e1e"
    fg = "#e6e6e6"
    accent = "#3b82f6"

    root = tk.Tk()
    root.title("Autosub Player")
    root.configure(bg=bg)
    root.geometry("460x320")
    root.resizable(False, False)

    # Center the window on screen.
    root.update_idletasks()
    width, height = 460, 320
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    selected: dict[str, any] = {"path": None, "source": None, "target": None}

    title_label = tk.Label(
        root,
        text="Autosub Player",
        bg=bg,
        fg=fg,
        font=("Segoe UI", 16, "bold"),
    )
    title_label.pack(pady=(20, 4))

    subtitle_label = tk.Label(
        root,
        text="Select languages, then browse for a video.",
        bg=bg,
        fg="#a0a0a0",
        font=("Segoe UI", 10),
    )
    subtitle_label.pack(pady=(0, 16))

    LANGUAGES = {
        "Auto-detect": "auto",
        "English": "en",
        "Arabic": "ar",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese (Simplified)": "zh-CN",
    }
    TARGET_LANGUAGES = {"None (Original)": "none"}
    TARGET_LANGUAGES.update({k: v for k, v in LANGUAGES.items() if v != "auto"})

    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("TCombobox", fieldbackground="#2a2a2a", background="#3a3a3a", foreground="white", bordercolor="#2a2a2a")

    dropdown_frame = tk.Frame(root, bg=bg)
    dropdown_frame.pack(pady=(0, 15))

    tk.Label(dropdown_frame, text="Source Language:", bg=bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=10, pady=2)
    source_var = tk.StringVar(value="Auto-detect")
    source_cb = ttk.Combobox(dropdown_frame, textvariable=source_var, values=list(LANGUAGES.keys()), state="readonly", width=18)
    source_cb.grid(row=1, column=0, padx=10, pady=(0, 10))

    tk.Label(dropdown_frame, text="Translation (Optional):", bg=bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(row=0, column=1, sticky="w", padx=10, pady=2)
    target_var = tk.StringVar(value="None (Original)")
    target_cb = ttk.Combobox(dropdown_frame, textvariable=target_var, values=list(TARGET_LANGUAGES.keys()), state="readonly", width=18)
    target_cb.grid(row=1, column=1, padx=10, pady=(0, 10))

    def browse() -> None:
        filetypes = [
            ("Video files", " ".join(f"*{ext}" for ext in SUPPORTED_VIDEO_EXTENSIONS)),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=filetypes,
        )
        if path:
            selected["path"] = Path(path)
            selected["source"] = LANGUAGES[source_var.get()]
            selected["target"] = TARGET_LANGUAGES[target_var.get()]
            root.destroy()

    def cancel() -> None:
        selected["path"] = None
        root.destroy()

    button_frame = tk.Frame(root, bg=bg)
    button_frame.pack(pady=10)

    browse_button = tk.Button(
        button_frame,
        text="Browse for Video…",
        command=browse,
        bg=accent,
        fg="white",
        activebackground="#2563eb",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=16,
        pady=8,
        borderwidth=0,
        cursor="hand2",
    )
    browse_button.grid(row=0, column=0, padx=6)

    cancel_button = tk.Button(
        button_frame,
        text="Cancel",
        command=cancel,
        bg="#2a2a2a",
        fg=fg,
        activebackground="#3a3a3a",
        activeforeground=fg,
        relief="flat",
        font=("Segoe UI", 10),
        padx=16,
        pady=8,
        borderwidth=0,
        cursor="hand2",
    )
    cancel_button.grid(row=0, column=1, padx=6)

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    if selected["path"] is None:
        return None, None, None
    return selected["path"], selected["source"], selected["target"]


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def validate_video_path(path: Path) -> None:
    """
    Validate that a given path points to an existing, supported video file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not a supported video format.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
        raise ValueError(
            f"Unsupported video format '{path.suffix}'. "
            f"Supported formats: {supported}"
        )


def validate_subtitle_format(fmt: str) -> str:
    """
    Normalize and validate a requested subtitle output format.

    Raises:
        ValueError: If the format is not supported.
    """
    normalized = fmt.lower().lstrip(".")
    if normalized not in SUPPORTED_SUBTITLE_FORMATS:
        supported = ", ".join(SUPPORTED_SUBTITLE_FORMATS)
        raise ValueError(
            f"Unsupported subtitle format '{fmt}'. Supported formats: {supported}"
        )
    return normalized


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------
# Multi-Provider Translation Engine
# --------------------------------------------------------------------------

BATCH_SIZE = 40
MAX_RETRIES = 3
RETRY_DELAY = 2.0

LANGUAGE_CODES_TO_NAMES = {
    "auto": "Auto-detect",
    "en": "English",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Chinese (Simplified)"
}

def apply_rtl_fix(text: str) -> str:
    clean = text.replace("\u202B", "").replace("\u202C", "")
    return f"\u202B{clean}\u202C"

_TAG_RE = re.compile(r"\{[^}]*\}")

def hide_tags(line: str):
    tags = _TAG_RE.findall(line)
    clean = line
    placeholders = {}
    for idx, tag in enumerate(tags):
        ph = f"<tx{idx}/>"
        clean = clean.replace(tag, f"\x00PH{idx}\x00", 1)
        placeholders[idx] = tag

    parts = re.split(r"\x00PH(\d+)\x00", clean)
    escaped_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            escaped_parts.append(html.escape(part))
        else:
            escaped_parts.append(f"<tx{part}/>")
    clean_escaped = "".join(escaped_parts)
    return clean_escaped, tags

def restore_tags(line: str, tags: list) -> str:
    result = html.unescape(line)
    for idx, tag in enumerate(tags):
        result = result.replace(f"<tx{idx}/>", tag)
    return result

class ProviderExhaustedException(Exception): pass

class TranslationProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def translate_batch(self, lines: list[str], source_lang: str, target_lang: str) -> list[str]: ...

def _strip_ass_tags(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    clean_lines = []
    all_tags = []
    for line in lines:
        tags = _TAG_RE.findall(line)
        clean = line
        for tag in tags:
            clean = clean.replace(tag, "", 1)
        clean_lines.append(clean.strip())
        all_tags.append(tags)
    return clean_lines, all_tags

def _reattach_tags(translated: list[str], all_tags: list[list[str]]) -> list[str]:
    result = []
    for text, tags in zip(translated, all_tags):
        prefix = "".join(tags)
        result.append(prefix + text if prefix else text)
    return result

class DeepLProvider(TranslationProvider):
    name = "DeepL"
    def __init__(self, api_key: str):
        import deepl
        self._translator = deepl.Translator(api_key)

    def translate_batch(self, lines: list[str], source_lang: str, target_lang: str) -> list[str]:
        import deepl
        batch_tags = []
        xml_parts = ['<?xml version="1.0" encoding="utf-8"?><doc>']
        for i, line in enumerate(lines):
            clean_line, tags = hide_tags(line)
            batch_tags.append(tags)
            xml_parts.append(f'<p id="{i}">{clean_line}</p>')
        xml_parts.append("</doc>")
        xml_payload = "".join(xml_parts)

        max_tags = max((len(t) for t in batch_tags), default=0)
        ignore_tags = [f"tx{i}" for i in range(max_tags)]

        delay = RETRY_DELAY
        last_exc = None

        target_deepl = target_lang.upper()
        if target_deepl == "EN": target_deepl = "EN-US"
        if target_deepl == "PT": target_deepl = "PT-BR"

        src = source_lang.upper() if source_lang and source_lang != "auto" else None

        for attempt in range(MAX_RETRIES):
            try:
                result = self._translator.translate_text(
                    xml_payload,
                    source_lang=src,
                    target_lang=target_deepl,
                    tag_handling="xml",
                    ignore_tags=ignore_tags,
                )
                translated_xml = result.text
                break
            except deepl.QuotaExceededException:
                raise ProviderExhaustedException("DeepL quota exceeded")
            except deepl.TooManyRequestsException:
                last_exc = None
                time.sleep(delay)
                delay *= 2
            except deepl.DeepLException as exc:
                if "quota" in str(exc).lower():
                    raise ProviderExhaustedException(f"DeepL quota: {exc}")
                last_exc = exc
                time.sleep(delay)
                delay *= 2
        else:
            raise ProviderExhaustedException(f"DeepL exhausted after {MAX_RETRIES} retries: {last_exc}")

        final_lines = []
        for i in range(len(lines)):
            match = re.search(rf'<p[^>]*\bid="{i}"[^>]*>(.*?)</p>', translated_xml, re.DOTALL)
            if match:
                ara_text = match.group(1).strip()
                restored = restore_tags(ara_text, batch_tags[i])
                final_lines.append(apply_rtl_fix(restored) if target_lang == "ar" else restored)
            else:
                final_lines.append("__FAILED__" + lines[i])
        return final_lines

_LLM_TRANSLATE_PROMPT = """\
You are a professional subtitle translator. Translate the following subtitle lines from {source} to {target}.

RULES:
- Return ONLY a JSON array of strings, one translated string per input line.
- Keep the same number of elements as the input.
- Make it sound natural for dialogue.
- Keep character names untranslated (transliterate to target script if appropriate).
- Do NOT add any explanation, markdown, or wrapping — just the raw JSON array.

INPUT LINES:
{lines_json}
"""

class GeminiProvider(TranslationProvider):
    name = "Gemini"
    def __init__(self, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def translate_batch(self, lines: list[str], source_lang: str, target_lang: str) -> list[str]:
        clean_lines, all_tags = _strip_ass_tags(lines)
        source_name = LANGUAGE_CODES_TO_NAMES.get(source_lang, "Original") if source_lang else "Original"
        target_name = LANGUAGE_CODES_TO_NAMES.get(target_lang, target_lang)

        prompt = _LLM_TRANSLATE_PROMPT.format(source=source_name, target=target_name, lines_json=json.dumps(clean_lines, ensure_ascii=False))

        delay = RETRY_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "quota" in exc_str or "resource" in exc_str:
                    if attempt == MAX_RETRIES - 1:
                        raise ProviderExhaustedException(f"Gemini quota/rate limit: {exc}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ProviderExhaustedException(f"Gemini error: {exc}")

        return self._parse_llm_response(response.text, lines, clean_lines, all_tags, target_lang)

    def _parse_llm_response(self, raw: str, original: list[str], clean: list[str], all_tags: list[list[str]], target_lang: str) -> list[str]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()
        try:
            translated = json.loads(text)
        except json.JSONDecodeError:
            print(f"[!] Gemini returned unparseable JSON, marking batch as failed")
            return ["__FAILED__" + line for line in original]

        if not isinstance(translated, list) or len(translated) != len(original):
            print(f"[!] Gemini returned {len(translated) if isinstance(translated, list) else 'non-list'} items, expected {len(original)}")
            return ["__FAILED__" + line for line in original]

        tagged = _reattach_tags(translated, all_tags)
        if target_lang == "ar":
            return [apply_rtl_fix(t) for t in tagged]
        return tagged

class GroqProvider(TranslationProvider):
    name = "Groq"
    def __init__(self, api_key: str):
        import groq as groq_sdk
        self._client = groq_sdk.Groq(api_key=api_key)

    def translate_batch(self, lines: list[str], source_lang: str, target_lang: str) -> list[str]:
        clean_lines, all_tags = _strip_ass_tags(lines)
        source_name = LANGUAGE_CODES_TO_NAMES.get(source_lang, "Original") if source_lang else "Original"
        target_name = LANGUAGE_CODES_TO_NAMES.get(target_lang, target_lang)

        prompt = _LLM_TRANSLATE_PROMPT.format(source=source_name, target=target_name, lines_json=json.dumps(clean_lines, ensure_ascii=False))

        delay = RETRY_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "rate" in exc_str or "quota" in exc_str or "limit" in exc_str:
                    if attempt == MAX_RETRIES - 1:
                        raise ProviderExhaustedException(f"Groq quota/rate limit: {exc}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ProviderExhaustedException(f"Groq error: {exc}")

        raw = response.choices[0].message.content
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()
        try:
            translated = json.loads(text)
        except json.JSONDecodeError:
            print(f"[!] Groq returned unparseable JSON, marking batch as failed")
            return ["__FAILED__" + line for line in lines]
        if not isinstance(translated, list) or len(translated) != len(lines):
            print(f"[!] Groq returned bad format")
            return ["__FAILED__" + line for line in lines]
        
        tagged = _reattach_tags(translated, all_tags)
        if target_lang == "ar":
            return [apply_rtl_fix(t) for t in tagged]
        return tagged

class ProviderChain:
    def __init__(self, providers: list[TranslationProvider]):
        self._providers = providers
        self._current = 0
        if self._providers:
            print(f"[info] Using translation provider: {self._providers[self._current].name}")

    def translate_batch(self, lines: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not self._providers:
            return lines
        while self._current < len(self._providers):
            provider = self._providers[self._current]
            try:
                return provider.translate_batch(lines, source_lang, target_lang)
            except ProviderExhaustedException as exc:
                print(f"[warn] {provider.name} exhausted: {exc}")
                self._current += 1
                if self._current < len(self._providers):
                    next_name = self._providers[self._current].name
                    print(f"[info] Switched translation provider to {next_name}")
                else:
                    print("[warn] All translation providers exhausted")
                    return ["__FAILED__" + line for line in lines]
        return ["__FAILED__" + line for line in lines]

def build_provider_chain() -> ProviderChain:
    providers = []
    deepl_key = os.environ.get("DEEPL_API_KEY", "").strip()
    if deepl_key:
        try:
            providers.append(DeepLProvider(deepl_key))
            print("[init] DeepL provider ready")
        except Exception as exc:
            print(f"[init] DeepL init failed: {exc}")

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            providers.append(GeminiProvider(gemini_key))
            print("[init] Gemini provider ready")
        except Exception as exc:
            print(f"[init] Gemini init failed: {exc}")

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            providers.append(GroqProvider(groq_key))
            print("[init] Groq provider ready")
        except Exception as exc:
            print(f"[init] Groq init failed: {exc}")

    if not providers:
         print("[warn] No API keys found in environment variables (DEEPL_API_KEY, GEMINI_API_KEY, GROQ_API_KEY). Translation will be skipped.")
    
    return ProviderChain(providers)

# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def transcribe_video(
    video_path: Path,
    model_size: str,
    language: Optional[str],
    target_language: Optional[str],
    subtitle_format: str,
    output_path: Optional[Path],
) -> TranscriptionResult:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Install it with: pip install faster-whisper") from exc

    try:
        import pysubs2
    except ImportError as exc:
        raise RuntimeError("pysubs2 is not installed. Install it with: pip install pysubs2") from exc

    translator_chain = None
    if target_language and target_language != "none":
        translator_chain = build_provider_chain()

    device, compute_type = select_device_and_compute_type()
    print(f"[info] Using device='{device}' compute_type='{compute_type}'")
    print(f"[info] Loading faster-whisper model '{model_size}'...")

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        if device != "cpu":
            print(f"[warn] Failed to load model on '{device}' ({exc}). Retrying on CPU...")
            try:
                device, compute_type = "cpu", "int8"
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as cpu_exc:
                raise RuntimeError(f"Failed to load model on CPU: {cpu_exc}") from cpu_exc
        else:
            raise RuntimeError(f"Failed to load faster-whisper model: {exc}") from exc

    print(f"[info] Transcribing '{video_path.name}'...")
    start_time = time.monotonic()

    try:
        segments_iter, info = model.transcribe(
            str(video_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Transcription failed for '{video_path}': {exc}") from exc

    print(f"[info] Detected language: {info.language} (confidence: {info.language_probability:.2%}), duration: {info.duration:.1f}s")
    
    collected_segments = []
    try:
        for segment in segments_iter:
            collected_segments.append(segment)
            elapsed = time.monotonic() - start_time
            speed = segment.end / elapsed if elapsed > 0 else 0.0
            print(f"[progress] transcribed | t={segment.start:7.1f}s-{segment.end:7.1f}s | speed={speed:5.2f}x realtime")
    except Exception as exc:
        raise RuntimeError(f"Error while iterating transcription segments: {exc}") from exc

    segment_count = len(collected_segments)
    if segment_count == 0:
        raise RuntimeError("No speech segments were detected in the audio track.")

    processing_seconds = time.monotonic() - start_time
    print(f"[info] Transcription completed in {processing_seconds:.1f}s. Proceeding to translation if enabled.")

    texts_to_translate = [seg.text.strip() for seg in collected_segments]
    
    if translator_chain and translator_chain._providers:
        translated_texts = []
        for i in range(0, len(texts_to_translate), BATCH_SIZE):
            batch = texts_to_translate[i:i + BATCH_SIZE]
            print(f"[info] Translating batch {i//BATCH_SIZE + 1}...")
            src = language if language else info.language
            trans_batch = translator_chain.translate_batch(batch, src, target_language)
            translated_texts.extend(trans_batch)
        
        texts_to_save = translated_texts
    else:
        texts_to_save = texts_to_translate

    subs = pysubs2.SSAFile()
    for seg, text in zip(collected_segments, texts_to_save):
        event = pysubs2.SSAEvent(
            start=pysubs2.make_time(s=seg.start),
            end=pysubs2.make_time(s=seg.end),
            text=text,
        )
        subs.events.append(event)

    if output_path is None:
        output_path = video_path.with_suffix(f".{subtitle_format}")
    else:
        output_path = output_path.with_suffix(f".{subtitle_format}")

    try:
        subs.save(str(output_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to save subtitle file to '{output_path}': {exc}") from exc

    print(f"[info] Wrote {segment_count} subtitle events to '{output_path}'")

    return TranscriptionResult(
        subtitle_path=output_path,
        detected_language=info.language,
        language_probability=info.language_probability,
        duration_seconds=info.duration,
        processing_seconds=processing_seconds,
        segment_count=segment_count,
    )
# --------------------------------------------------------------------------


def launch_mpv(video_path: Path, subtitle_path: Path) -> None:
    """
    Launch mpv with the given video and subtitle file pre-loaded.

    If mpv is not found on PATH, prints a clear message with the location
    of the generated subtitle file instead of raising.
    """
    mpv_binary = shutil.which("mpv")

    if mpv_binary is None:
        print(
            "[warn] mpv was not found on PATH. Please install mpv "
            "(https://mpv.io) to enable auto-playback.\n"
            f"[info] Your subtitle file has been saved at: {subtitle_path}"
        )
        return

    command: List[str] = [
        mpv_binary,
        str(video_path),
        f"--sub-file={subtitle_path}",
    ]

    print(f"[info] Launching mpv: {' '.join(command)}")

    try:
        subprocess.Popen(command)
    except FileNotFoundError:
        print(
            "[warn] mpv could not be executed even though it was found on "
            f"PATH. Your subtitle file has been saved at: {subtitle_path}"
        )
    except Exception as exc:
        print(
            f"[warn] Failed to launch mpv ({exc}). "
            f"Your subtitle file has been saved at: {subtitle_path}"
        )


# --------------------------------------------------------------------------
# CLI argument parsing
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse.ArgumentParser for this application."""
    parser = argparse.ArgumentParser(
        prog="autosub_player.py",
        description=(
            "Transcribe a video's audio with faster-whisper, generate "
            "subtitles, and auto-play the result in mpv."
        ),
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to the input video file. If omitted, a file picker dialog is shown.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        help=(
            "faster-whisper model size/name (e.g. tiny, base, small, medium, "
            "large-v3). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Force a source language (ISO-639-1 code, e.g. 'en'). Default: auto-detect.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="none",
        help="Target language for translation (e.g., 'es'). Default: none (original text).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=DEFAULT_FORMAT,
        choices=SUPPORTED_SUBTITLE_FORMATS,
        help="Subtitle output format. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output path for the subtitle file (extension is normalized).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip auto-launching mpv after transcription.",
    )
    return parser


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Application entry point.

    Args:
        argv: Optional argument vector for testing; defaults to sys.argv.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Resolve video path -------------------------------------------------
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        source_lang = args.lang
        target_lang = args.target_lang
    else:
        print("[info] No --video supplied; opening file picker dialog...")
        video_path_opt, source_lang_opt, target_lang_opt = prompt_for_settings()
        if video_path_opt is None:
            print("[error] No video file was selected. Exiting.", file=sys.stderr)
            return 1
        video_path = video_path_opt.expanduser().resolve()
        source_lang = None if source_lang_opt == "auto" else source_lang_opt
        target_lang = target_lang_opt

    try:
        validate_video_path(video_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    # Resolve subtitle format ---------------------------------------------
    try:
        subtitle_format = validate_subtitle_format(args.format)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else None

    # Transcribe -----------------------------------------------------------
    try:
        result = transcribe_video(
            video_path=video_path,
            model_size=args.model,
            language=source_lang,
            target_language=target_lang,
            subtitle_format=subtitle_format,
            output_path=output_path,
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[warn] Transcription interrupted by user.", file=sys.stderr)
        return 130

    print(
        f"[summary] language={result.detected_language} "
        f"({result.language_probability:.2%}) | "
        f"duration={result.duration_seconds:.1f}s | "
        f"processing={result.processing_seconds:.1f}s | "
        f"segments={result.segment_count} | "
        f"subtitles='{result.subtitle_path}'"
    )

    # Playback ---------------------------------------------------------------
    if args.no_play:
        print(f"[info] --no-play set. Subtitle file saved at: {result.subtitle_path}")
        return 0

    try:
        launch_mpv(video_path, result.subtitle_path)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        print(
            f"[warn] Unexpected error while launching playback: {exc}. "
            f"Subtitle file saved at: {result.subtitle_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
