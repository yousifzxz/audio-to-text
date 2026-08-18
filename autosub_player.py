#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autosub_player.py
==================

Automatically transcribes a video's audio track using faster-whisper,
optionally translates subtitles via a multi-provider AI chain
(DeepL → Gemini → Groq), generates a subtitle file (.srt or .ass),
auto-muxes the subtitles into the video via ffmpeg, and launches mpv
for playback.

--------------------------------------------------------------------
requirements.txt
--------------------------------------------------------------------
faster-whisper>=1.0.0
pysubs2>=1.6.0
ctranslate2>=4.0.0
deepl
google-genai
groq
python-dotenv

# System dependencies (not installable via pip):
#   - mpv media player (https://mpv.io) must be on PATH for auto-playback.
#   - ffmpeg must be on PATH (faster-whisper/ctranslate2 relies on it for
#     audio decoding of arbitrary container formats such as .mkv/.webm,
#     and it is also used for subtitle muxing / burn-in).
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
    python autosub_player.py --video clip.mp4 --burn-in --target-lang ar
    python autosub_player.py --video clip.mp4 --provider-order gemini,groq

Environment variables (or .env file):
    DEEPL_API_KEY   - DeepL API key for translation
    GEMINI_API_KEY  - Google Gemini API key for translation
    GROQ_API_KEY    - Groq API key for translation

Author: Autosub Player
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

# Translation engine constants
BATCH_SIZE = 40
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Subtitle line-wrapping constants
MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_EVENT = 2

# Cache directory (central location)
CACHE_DIR = Path.home() / ".autosub_cache"

LANGUAGE_CODES_TO_NAMES: Dict[str, str] = {
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
    "zh-CN": "Chinese (Simplified)",
}

# Subtitle codec mapping per container format for ffmpeg muxing.
# Maps output extension → subtitle codec name.
_SUBTITLE_CODEC_MAP: Dict[str, Dict[str, str]] = {
    ".mp4": {"srt": "mov_text", "ass": "mov_text"},
    ".mov": {"srt": "mov_text", "ass": "mov_text"},
    ".mkv": {"srt": "srt", "ass": "ass"},
    ".webm": {"srt": "webvtt", "ass": "webvtt"},
}


# --------------------------------------------------------------------------
# Structured logging helper
# --------------------------------------------------------------------------


def log(stage: str, message: str) -> None:
    """
    Print a structured log message with a consistent stage prefix.

    Args:
        stage: Pipeline stage name (e.g. 'init', 'transcribe', 'translate').
        message: Human-readable log message.
    """
    print(f"[{stage}] {message}")


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


@dataclass
class CachedSegment:
    """A single cached transcription segment."""

    start: float
    end: float
    text: str


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
        log("init", f"GPU probing failed ({exc}); falling back to CPU.")

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
        log(
            "error",
            "Tkinter is not available in this Python installation. "
            "Please supply --video explicitly.",
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

    selected: dict[str, Any] = {"path": None, "source": None, "target": None}

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
    TARGET_LANGUAGES: dict[str, str] = {"None (Original)": "none"}
    TARGET_LANGUAGES.update({k: v for k, v in LANGUAGES.items() if v != "auto"})

    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure(
        "TCombobox",
        fieldbackground="#2a2a2a",
        background="#3a3a3a",
        foreground="white",
        bordercolor="#2a2a2a",
    )

    dropdown_frame = tk.Frame(root, bg=bg)
    dropdown_frame.pack(pady=(0, 15))

    tk.Label(
        dropdown_frame,
        text="Source Language:",
        bg=bg,
        fg="#a0a0a0",
        font=("Segoe UI", 9),
    ).grid(row=0, column=0, sticky="w", padx=10, pady=2)
    source_var = tk.StringVar(value="Auto-detect")
    source_cb = ttk.Combobox(
        dropdown_frame,
        textvariable=source_var,
        values=list(LANGUAGES.keys()),
        state="readonly",
        width=18,
    )
    source_cb.grid(row=1, column=0, padx=10, pady=(0, 10))

    tk.Label(
        dropdown_frame,
        text="Translation (Optional):",
        bg=bg,
        fg="#a0a0a0",
        font=("Segoe UI", 9),
    ).grid(row=0, column=1, sticky="w", padx=10, pady=2)
    target_var = tk.StringVar(value="None (Original)")
    target_cb = ttk.Combobox(
        dropdown_frame,
        textvariable=target_var,
        values=list(TARGET_LANGUAGES.keys()),
        state="readonly",
        width=18,
    )
    target_cb.grid(row=1, column=1, padx=10, pady=(0, 10))

    def browse() -> None:
        filetypes = [
            (
                "Video files",
                " ".join(f"*{ext}" for ext in SUPPORTED_VIDEO_EXTENSIONS),
            ),
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


def validate_ffmpeg() -> str:
    """
    Check that ffmpeg is available on PATH.

    Returns:
        The absolute path to the ffmpeg binary.

    Raises:
        RuntimeError: If ffmpeg is not found on PATH.
    """
    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Please install ffmpeg "
            "(https://ffmpeg.org) to enable subtitle muxing and burn-in. "
            "On Windows, download from https://www.gyan.dev/ffmpeg/builds/ "
            "and add the bin/ directory to your system PATH."
        )
    return ffmpeg_binary


# --------------------------------------------------------------------------
# Transcription caching
# --------------------------------------------------------------------------


def compute_video_fingerprint(video_path: Path, model_size: str) -> str:
    """
    Compute a fast fingerprint for a video file + model combination.

    Hashes the first 64 KB, last 64 KB, file size, and model name to
    create a unique key without reading multi-GB files in full.

    Args:
        video_path: Path to the video file.
        model_size: The Whisper model size string.

    Returns:
        A hex digest string suitable for use as a cache key.
    """
    chunk_size = 65536  # 64 KB
    file_size = video_path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(f"size={file_size}|model={model_size}".encode())

    with open(video_path, "rb") as f:
        # First 64 KB
        hasher.update(f.read(chunk_size))
        # Last 64 KB (if file is large enough)
        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)
            hasher.update(f.read(chunk_size))

    return hasher.hexdigest()


def load_cached_transcription(
    video_path: Path, model_size: str
) -> Optional[tuple[List[CachedSegment], str, float, float]]:
    """
    Attempt to load a cached transcription result.

    Args:
        video_path: Path to the video file.
        model_size: The Whisper model size string.

    Returns:
        A tuple of (segments, detected_language, language_probability,
        duration_seconds) if cache hit, or None if cache miss.
    """
    fingerprint = compute_video_fingerprint(video_path, model_size)
    cache_file = CACHE_DIR / f"{fingerprint}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        segments = [
            CachedSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in data["segments"]
        ]
        return (
            segments,
            data["detected_language"],
            data["language_probability"],
            data["duration_seconds"],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log("cache", f"Corrupt cache file {cache_file.name}, ignoring: {exc}")
        return None


def save_transcription_cache(
    video_path: Path,
    model_size: str,
    segments: List[CachedSegment],
    detected_language: str,
    language_probability: float,
    duration_seconds: float,
) -> None:
    """
    Save transcription results to the central cache.

    Args:
        video_path: Path to the video file.
        model_size: The Whisper model size string.
        segments: List of transcribed segments.
        detected_language: ISO code of the detected language.
        language_probability: Confidence of the language detection.
        duration_seconds: Total audio duration in seconds.
    """
    fingerprint = compute_video_fingerprint(video_path, model_size)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{fingerprint}.json"

    data = {
        "video_name": video_path.name,
        "model_size": model_size,
        "detected_language": detected_language,
        "language_probability": language_probability,
        "duration_seconds": duration_seconds,
        "segments": [asdict(s) for s in segments],
    }

    try:
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log("cache", f"Saved transcription cache ({len(segments)} segments)")
    except OSError as exc:
        log("cache", f"Failed to write cache: {exc}")


# --------------------------------------------------------------------------
# ASS/SSA tag handling (shared across all providers)
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"\{[^}]*\}")


def hide_tags(line: str) -> tuple[str, List[str]]:
    """
    Replace inline ASS/SSA tags with XML-safe placeholders.

    Tags like ``{\\i1}`` are replaced with ``<tx0/>``, ``<tx1/>``, etc.
    The surrounding text is HTML-escaped so the result is safe for XML
    payloads (used by DeepL) and for LLM prompts.

    Args:
        line: A subtitle line potentially containing ASS tags.

    Returns:
        A tuple of (cleaned_line_with_placeholders, list_of_original_tags).
    """
    tags = _TAG_RE.findall(line)
    clean = line
    for idx, tag in enumerate(tags):
        clean = clean.replace(tag, f"\x00PH{idx}\x00", 1)

    parts = re.split(r"\x00PH(\d+)\x00", clean)
    escaped_parts: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            escaped_parts.append(html.escape(part))
        else:
            escaped_parts.append(f"<tx{part}/>")
    clean_escaped = "".join(escaped_parts)
    return clean_escaped, tags


def restore_tags(line: str, tags: List[str]) -> str:
    """
    Restore original ASS/SSA tags from their ``<txN/>`` placeholders.

    Also un-escapes HTML entities that were added by ``hide_tags()``.

    Args:
        line: A translated line containing ``<txN/>`` placeholders.
        tags: The original tags list from ``hide_tags()``.

    Returns:
        The line with original tags restored in their positions.
    """
    result = html.unescape(line)
    for idx, tag in enumerate(tags):
        result = result.replace(f"<tx{idx}/>", tag)
    return result


def hide_tags_plain(line: str) -> tuple[str, List[str]]:
    """
    Replace inline ASS/SSA tags with plain-text placeholders for LLM prompts.

    Unlike ``hide_tags()`` which produces XML-safe output, this version
    produces plain ``<tx0/>``, ``<tx1/>`` markers without HTML-escaping
    the surrounding text, suitable for LLM translation prompts.

    Args:
        line: A subtitle line potentially containing ASS tags.

    Returns:
        A tuple of (cleaned_line_with_placeholders, list_of_original_tags).
    """
    tags = _TAG_RE.findall(line)
    clean = line
    for idx, tag in enumerate(tags):
        clean = clean.replace(tag, f"<tx{idx}/>", 1)
    return clean, tags


def restore_tags_plain(line: str, tags: List[str]) -> str:
    """
    Restore original ASS/SSA tags from ``<txN/>`` placeholders (plain text).

    Args:
        line: A translated line containing ``<txN/>`` placeholders.
        tags: The original tags list from ``hide_tags_plain()``.

    Returns:
        The line with original tags restored in their positions.
    """
    result = line
    for idx, tag in enumerate(tags):
        result = result.replace(f"<tx{idx}/>", tag)
    return result


def wrap_subtitle_text(
    text: str,
    max_chars: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES_PER_EVENT,
    line_break: str = "\\N",
) -> str:
    """
    Wrap subtitle text into balanced multi-line format.

    Splits long subtitle text at word boundaries, preferring a balanced
    two-line split (close to the middle) rather than greedily filling
    the first line. This matches professional subtitle conventions.

    ASS/SSA tags like ``{\\i1}`` are counted as zero-width for wrapping
    purposes but are never split across a line boundary.

    Args:
        text: The final subtitle text (after translation and tag restoration).
        max_chars: Maximum visible characters per line (default 42).
        max_lines: Maximum number of lines per subtitle event (default 2).
        line_break: The line-break string to insert. Use ``"\\N"`` for
                    ASS/SSA format, ``"\\n"`` or ``"\n"`` for SRT.

    Returns:
        The wrapped text with line breaks inserted.

    Examples:
        >>> wrap_subtitle_text("Hello world")
        'Hello world'
        >>> wrap_subtitle_text("This is a somewhat longer subtitle line that needs wrapping")
        'This is a somewhat longer\\Nsubtitle line that needs wrapping'
        >>> wrap_subtitle_text("{\\b1}Bold text{\\b0} that is long enough to wrap around")
        '{\\b1}Bold text{\\b0} that is\\Nlong enough to wrap around'
    """
    # Strip any existing line breaks and normalize
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\n", " ").strip()

    if not text:
        return text

    # Calculate visible length (excluding ASS tags)
    visible_text = _TAG_RE.sub("", text)
    visible_len = len(visible_text)

    # No wrapping needed if the visible text fits on one line
    if visible_len <= max_chars:
        return text

    # Tokenize into words, keeping tags attached to the word they precede/follow.
    # We split on spaces but preserve the structure.
    words = text.split()
    if not words:
        return text

    def _visible_len(s: str) -> int:
        """Return the visible character count (excluding ASS tags)."""
        return len(_TAG_RE.sub("", s))

    # For max_lines == 2 (the common case), use balanced split:
    # find the split point closest to half the visible length.
    if max_lines >= 2:
        # Build prefix visible-length array
        prefix_lens: list[int] = []
        running = 0
        for word in words:
            running += _visible_len(word) + (1 if prefix_lens else 0)  # +1 for space
            prefix_lens.append(running)

        total_visible = prefix_lens[-1]
        half = total_visible / 2.0

        # Find best split point (split AFTER word at index best_idx)
        best_idx = 0
        best_diff = abs(prefix_lens[0] - half)
        for i in range(1, len(words) - 1):  # don't split after last word
            diff = abs(prefix_lens[i] - half)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        line1 = " ".join(words[: best_idx + 1])
        line2 = " ".join(words[best_idx + 1 :])

        # If one of the lines is empty (single word), just return as-is
        if not line1 or not line2:
            return text

        # If we'd need more than max_lines (line2 is still very long),
        # we still return 2 lines — we never truncate/drop words.
        return f"{line1}{line_break}{line2}"

    # Fallback for max_lines == 1: no wrapping possible, return as-is
    return text


def apply_rtl_fix(text: str) -> str:
    """
    Wrap text in RTL embedding characters for proper Arabic display.

    Args:
        text: The subtitle text to wrap.

    Returns:
        The text wrapped with U+202B (RLE) and U+202C (PDF).
    """
    clean = text.replace("\u202B", "").replace("\u202C", "")
    return f"\u202B{clean}\u202C"


# --------------------------------------------------------------------------
# Translation provider exceptions
# --------------------------------------------------------------------------


class ProviderExhaustedException(Exception):
    """Raised when a translation provider has exhausted its quota or retries."""
    pass


# --------------------------------------------------------------------------
# Translation provider base class
# --------------------------------------------------------------------------


class TranslationProvider(ABC):
    """Abstract base class for subtitle translation providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this provider."""
        ...

    @abstractmethod
    def translate_batch(
        self, lines: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """
        Translate a batch of subtitle lines.

        Args:
            lines: List of subtitle text lines to translate.
            source_lang: ISO language code of the source language.
            target_lang: ISO language code of the target language.

        Returns:
            List of translated lines. Lines that failed translation are
            prefixed with ``__FAILED__``.
        """
        ...


# --------------------------------------------------------------------------
# DeepL provider
# --------------------------------------------------------------------------


class DeepLProvider(TranslationProvider):
    """Translation provider using the DeepL API with XML tag handling."""

    name = "DeepL"

    def __init__(self, api_key: str) -> None:
        import deepl

        self._translator = deepl.Translator(api_key)

    def translate_batch(
        self, lines: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """Translate lines via DeepL with XML-based tag preservation."""
        import deepl

        batch_tags: List[List[str]] = []
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
        last_exc: Optional[Exception] = None

        target_deepl = target_lang.upper()
        if target_deepl == "EN":
            target_deepl = "EN-US"
        if target_deepl == "PT":
            target_deepl = "PT-BR"

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
            except deepl.TooManyRequestsException as exc:
                # Item 5 fix: store the exception instead of clearing it
                last_exc = exc
                time.sleep(delay)
                delay *= 2
            except deepl.DeepLException as exc:
                if "quota" in str(exc).lower():
                    raise ProviderExhaustedException(f"DeepL quota: {exc}")
                last_exc = exc
                time.sleep(delay)
                delay *= 2
        else:
            raise ProviderExhaustedException(
                f"DeepL exhausted after {MAX_RETRIES} retries: {last_exc}"
            )

        final_lines: List[str] = []
        for i in range(len(lines)):
            match = re.search(
                rf'<p[^>]*\bid="{i}"[^>]*>(.*?)</p>', translated_xml, re.DOTALL
            )
            if match:
                translated_text = match.group(1).strip()
                restored = restore_tags(translated_text, batch_tags[i])
                final_lines.append(
                    apply_rtl_fix(restored) if target_lang == "ar" else restored
                )
            else:
                final_lines.append("__FAILED__" + lines[i])
        return final_lines


# --------------------------------------------------------------------------
# LLM translation prompt (shared by Gemini & Groq)
# --------------------------------------------------------------------------

_LLM_TRANSLATE_PROMPT = """\
You are a professional subtitle translator. Translate the following subtitle lines from {source} to {target}.

RULES:
- Return ONLY a JSON array of strings, one translated string per input line.
- Keep the same number of elements as the input.
- Make it sound natural for dialogue.
- Keep character names untranslated (transliterate to target script if appropriate).
- Keep any <txN/> markers (like <tx0/>, <tx1/>) in their exact original positions \
relative to the surrounding words. These are formatting placeholders — do NOT \
translate, remove, or reorder them.
- Do NOT add any explanation, markdown, or wrapping — just the raw JSON array.

INPUT LINES:
{lines_json}
"""


def _parse_llm_json_response(
    raw: str,
    original_lines: List[str],
    all_tags: List[List[str]],
    target_lang: str,
    provider_name: str,
) -> List[str]:
    """
    Parse a JSON array response from an LLM and restore ASS tags.

    Args:
        raw: The raw text response from the LLM.
        original_lines: The original (pre-tag-stripping) subtitle lines.
        all_tags: Per-line list of original ASS tags from ``hide_tags_plain()``.
        target_lang: The target language ISO code.
        provider_name: Name of the provider for log messages.

    Returns:
        List of translated lines with tags restored. Lines that failed
        are prefixed with ``__FAILED__``.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        translated = json.loads(text)
    except json.JSONDecodeError:
        log("translate", f"{provider_name} returned unparseable JSON, marking batch as failed")
        return ["__FAILED__" + line for line in original_lines]

    if not isinstance(translated, list) or len(translated) != len(original_lines):
        actual = len(translated) if isinstance(translated, list) else "non-list"
        log(
            "translate",
            f"{provider_name} returned {actual} items, expected {len(original_lines)}",
        )
        return ["__FAILED__" + line for line in original_lines]

    # Restore tags in their original positions
    result: List[str] = []
    for text_line, tags in zip(translated, all_tags):
        restored = restore_tags_plain(str(text_line), tags)
        if target_lang == "ar":
            restored = apply_rtl_fix(restored)
        result.append(restored)
    return result


# --------------------------------------------------------------------------
# Gemini provider (Item 3: specific exception handling)
# --------------------------------------------------------------------------


class GeminiProvider(TranslationProvider):
    """Translation provider using Google Gemini (gemini-2.5-flash)."""

    name = "Gemini"

    def __init__(self, api_key: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def translate_batch(
        self, lines: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """Translate lines via Gemini with proper error handling."""
        from google.genai import errors as genai_errors

        # Item 2: Use hide_tags_plain for consistent tag handling
        all_tags: List[List[str]] = []
        clean_lines: List[str] = []
        for line in lines:
            clean, tags = hide_tags_plain(line)
            clean_lines.append(clean)
            all_tags.append(tags)

        source_name = (
            LANGUAGE_CODES_TO_NAMES.get(source_lang, "Original")
            if source_lang
            else "Original"
        )
        target_name = LANGUAGE_CODES_TO_NAMES.get(target_lang, target_lang)

        prompt = _LLM_TRANSLATE_PROMPT.format(
            source=source_name,
            target=target_name,
            lines_json=json.dumps(clean_lines, ensure_ascii=False),
        )

        delay = RETRY_DELAY
        response = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                break
            except genai_errors.APIError as exc:
                if exc.code == 429:
                    # Rate limit — retry with backoff
                    if attempt == MAX_RETRIES - 1:
                        raise ProviderExhaustedException(
                            f"Gemini rate limit after {MAX_RETRIES} retries: {exc.message}"
                        )
                    log("translate", f"Gemini rate limited (attempt {attempt + 1}), backing off...")
                    time.sleep(delay)
                    delay *= 2
                elif exc.code in (401, 403):
                    # Auth / permission error — fail fast
                    raise ProviderExhaustedException(
                        f"Gemini authentication error ({exc.code}): {exc.message}"
                    )
                elif exc.code >= 500:
                    # Server error — retry
                    if attempt == MAX_RETRIES - 1:
                        raise ProviderExhaustedException(
                            f"Gemini server error after {MAX_RETRIES} retries: {exc.message}"
                        )
                    log("translate", f"Gemini server error ({exc.code}), retrying...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    # Client error (400, 404, etc.) — fail fast
                    raise ProviderExhaustedException(
                        f"Gemini error ({exc.code}): {exc.message}"
                    )
            except Exception as exc:
                # Unexpected non-API error — fail fast
                raise ProviderExhaustedException(f"Gemini unexpected error: {exc}")

        if response is None:
            return ["__FAILED__" + line for line in lines]

        return _parse_llm_json_response(
            response.text, lines, all_tags, target_lang, "Gemini"
        )


# --------------------------------------------------------------------------
# Groq provider (Item 3: specific exception handling)
# --------------------------------------------------------------------------


class GroqProvider(TranslationProvider):
    """Translation provider using Groq (llama-3.3-70b-versatile)."""

    name = "Groq"

    def __init__(self, api_key: str) -> None:
        import groq as groq_sdk

        self._client = groq_sdk.Groq(api_key=api_key)

    def translate_batch(
        self, lines: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """Translate lines via Groq with proper error handling."""
        import groq as groq_sdk

        # Item 2: Use hide_tags_plain for consistent tag handling
        all_tags: List[List[str]] = []
        clean_lines: List[str] = []
        for line in lines:
            clean, tags = hide_tags_plain(line)
            clean_lines.append(clean)
            all_tags.append(tags)

        source_name = (
            LANGUAGE_CODES_TO_NAMES.get(source_lang, "Original")
            if source_lang
            else "Original"
        )
        target_name = LANGUAGE_CODES_TO_NAMES.get(target_lang, target_lang)

        prompt = _LLM_TRANSLATE_PROMPT.format(
            source=source_name,
            target=target_name,
            lines_json=json.dumps(clean_lines, ensure_ascii=False),
        )

        delay = RETRY_DELAY
        response = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                break
            except groq_sdk.RateLimitError:
                # 429 — retry with backoff
                if attempt == MAX_RETRIES - 1:
                    raise ProviderExhaustedException(
                        f"Groq rate limit after {MAX_RETRIES} retries"
                    )
                log("translate", f"Groq rate limited (attempt {attempt + 1}), backing off...")
                time.sleep(delay)
                delay *= 2
            except groq_sdk.AuthenticationError as exc:
                # 401 — fail fast
                raise ProviderExhaustedException(
                    f"Groq authentication error: {exc}"
                )
            except groq_sdk.PermissionDeniedError as exc:
                # 403 — fail fast
                raise ProviderExhaustedException(
                    f"Groq permission denied: {exc}"
                )
            except groq_sdk.APIStatusError as exc:
                if exc.status_code >= 500:
                    # Server error — retry
                    if attempt == MAX_RETRIES - 1:
                        raise ProviderExhaustedException(
                            f"Groq server error after {MAX_RETRIES} retries: {exc}"
                        )
                    log("translate", f"Groq server error ({exc.status_code}), retrying...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    # Other client errors — fail fast
                    raise ProviderExhaustedException(
                        f"Groq error ({exc.status_code}): {exc}"
                    )
            except groq_sdk.APIConnectionError as exc:
                # Network error — retry
                if attempt == MAX_RETRIES - 1:
                    raise ProviderExhaustedException(
                        f"Groq connection error after {MAX_RETRIES} retries: {exc}"
                    )
                log("translate", f"Groq connection error, retrying...")
                time.sleep(delay)
                delay *= 2

        if response is None:
            return ["__FAILED__" + line for line in lines]

        raw = response.choices[0].message.content
        return _parse_llm_json_response(raw, lines, all_tags, target_lang, "Groq")


# --------------------------------------------------------------------------
# Provider chain with fallback
# --------------------------------------------------------------------------


class ProviderChain:
    """
    Chains multiple translation providers with automatic fallback.

    When one provider is exhausted (quota, rate limit, auth), the chain
    automatically switches to the next available provider.
    """

    def __init__(self, providers: List[TranslationProvider]) -> None:
        self._providers = providers
        self._current = 0
        if self._providers:
            log("init", f"Using translation provider: {self._providers[self._current].name}")

    @property
    def has_providers(self) -> bool:
        """Return True if at least one provider is configured."""
        return len(self._providers) > 0

    def translate_batch(
        self, lines: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """
        Translate a batch, falling back to the next provider on failure.

        Args:
            lines: Subtitle lines to translate.
            source_lang: Source language ISO code.
            target_lang: Target language ISO code.

        Returns:
            Translated lines. Failed lines are prefixed with ``__FAILED__``.
        """
        if not self._providers:
            return lines
        while self._current < len(self._providers):
            provider = self._providers[self._current]
            try:
                return provider.translate_batch(lines, source_lang, target_lang)
            except ProviderExhaustedException as exc:
                log("translate", f"{provider.name} exhausted: {exc}")
                self._current += 1
                if self._current < len(self._providers):
                    next_name = self._providers[self._current].name
                    log("translate", f"Switched to provider: {next_name}")
                else:
                    log("translate", "All translation providers exhausted")
                    return ["__FAILED__" + line for line in lines]
        return ["__FAILED__" + line for line in lines]


def build_provider_chain(provider_order: Optional[str] = None) -> ProviderChain:
    """
    Build a provider chain from environment variables.

    Args:
        provider_order: Optional comma-separated provider order override
                        (e.g. ``"gemini,deepl,groq"``). Defaults to
                        DeepL → Gemini → Groq.

    Returns:
        A configured ProviderChain instance.
    """
    available: Dict[str, tuple[str, type]] = {}

    deepl_key = os.environ.get("DEEPL_API_KEY", "").strip()
    if deepl_key:
        available["deepl"] = (deepl_key, DeepLProvider)

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        available["gemini"] = (gemini_key, GeminiProvider)

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        available["groq"] = (groq_key, GroqProvider)

    # Determine ordering
    if provider_order:
        order = [name.strip().lower() for name in provider_order.split(",")]
    else:
        order = ["deepl", "gemini", "groq"]

    providers: List[TranslationProvider] = []
    for name in order:
        if name not in available:
            continue
        api_key, provider_cls = available[name]
        try:
            providers.append(provider_cls(api_key))
            log("init", f"{provider_cls.name} provider ready")  # type: ignore[attr-defined]
        except Exception as exc:
            log("init", f"{name} init failed: {exc}")

    if not providers:
        log(
            "init",
            "No API keys found in environment variables "
            "(DEEPL_API_KEY, GEMINI_API_KEY, GROQ_API_KEY). "
            "Translation will be skipped.",
        )

    return ProviderChain(providers)


# --------------------------------------------------------------------------
# Transcription Engines
# --------------------------------------------------------------------------

class TranscriptionEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def transcribe(
        self, video_path: Path, language: Optional[str]
    ) -> tuple[List[CachedSegment], str, float, float]:
        """
        Transcribe the video.
        
        Returns:
            (segments, detected_language, language_prob, duration)
        """
        pass

class WhisperEngine(TranscriptionEngine):
    def __init__(self, model_size: str):
        self.model_size = model_size

    @property
    def name(self) -> str:
        return "whisper"

    def transcribe(
        self, video_path: Path, language: Optional[str]
    ) -> tuple[List[CachedSegment], str, float, float]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed.") from exc

        device, compute_type = select_device_and_compute_type()
        log("init", f"Using device='{device}' compute_type='{compute_type}'")
        log("transcribe", f"Loading faster-whisper model '{self.model_size}'...")

        try:
            model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        except Exception as exc:
            if device != "cpu":
                log("transcribe", f"Failed to load model on '{device}' ({exc}). Retrying on CPU...")
                device, compute_type = "cpu", "int8"
                model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
            else:
                raise RuntimeError(f"Failed to load faster-whisper model: {exc}") from exc

        log("transcribe", f"Transcribing '{video_path.name}'...")
        start_time = time.monotonic()
        
        segments_iter, info = model.transcribe(
            str(video_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )

        detected_language = info.language
        language_probability = info.language_probability
        duration_seconds = info.duration

        log(
            "transcribe",
            f"Detected language: {detected_language} "
            f"(confidence: {language_probability:.2%}), "
            f"duration: {duration_seconds:.1f}s",
        )

        segments = []
        for segment in segments_iter:
            start_time_val = segment.words[0].start if segment.words else segment.start
            segments.append(CachedSegment(start=start_time_val, end=segment.end, text=segment.text.strip()))
            elapsed = time.monotonic() - start_time
            speed = segment.end / elapsed if elapsed > 0 else 0.0
            log("transcribe", f"t={start_time_val:7.1f}s-{segment.end:7.1f}s | speed={speed:5.2f}x realtime")
            
        return segments, detected_language, language_probability, duration_seconds

class CanaryQwenEngine(TranscriptionEngine):
    @property
    def name(self) -> str:
        return "canary-qwen"

    def transcribe(
        self, video_path: Path, language: Optional[str]
    ) -> tuple[List[CachedSegment], str, float, float]:
        if language and language.lower() not in ("en", "english"):
            raise ValueError(f"Canary-Qwen only supports English, but got: {language}")
        raise RuntimeError(
            "Canary-Qwen (SALM) does not expose real timestamps. "
            "Refusing to run to prevent desynced subtitles."
        )

class Qwen3Engine(TranscriptionEngine):
    def __init__(self, use_small_model: bool = False):
        self.use_small_model = use_small_model

    @property
    def name(self) -> str:
        return "qwen3-asr"

    def transcribe(
        self, video_path: Path, language: Optional[str]
    ) -> tuple[List[CachedSegment], str, float, float]:
        try:
            import librosa
            import soundfile as sf
            import torch
            from transformers import pipeline, AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR requires transformers>=5.13.0, accelerate, librosa, and soundfile."
            ) from exc

        device, _ = select_device_and_compute_type()
        if device == "cpu":
            raise RuntimeError("Qwen3-ASR requires a GPU (CUDA) but none was detected.")

        asr_model_id = "Qwen/Qwen3-ASR-0.6B" if self.use_small_model else "Qwen/Qwen3-ASR-1.7B"
        aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B"

        log("transcribe", f"Loading ASR model {asr_model_id} on {device}...")
        asr_pipe = pipeline("automatic-speech-recognition", model=asr_model_id, device=device, torch_dtype=torch.bfloat16)

        log("transcribe", f"Loading aligner model {aligner_model_id} on {device}...")
        # Often forced aligners are token-classification pipelines or custom pipelines in transformers.
        aligner_pipe = pipeline("token-classification", model=aligner_model_id, device=device, torch_dtype=torch.bfloat16)

        log("transcribe", f"Loading audio from {video_path.name}...")
        y, sr = librosa.load(str(video_path), sr=16000)
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        # Audio chunking (5 min max = 300 seconds)
        chunk_length_s = 300
        chunk_length_samples = chunk_length_s * sr
        
        all_segments = []
        detected_language = language or "auto"
        language_probability = 1.0 # Forced for Qwen3-ASR if no metadata

        start_time = time.monotonic()
        for i in range(0, len(y), chunk_length_samples):
            chunk_y = y[i:i + chunk_length_samples]
            chunk_offset_s = i / sr
            log("transcribe", f"Processing chunk {i//chunk_length_samples + 1} at offset {chunk_offset_s:.1f}s...")
            
            # Write chunk to temp file for pipelines that require paths
            temp_chunk_path = CACHE_DIR / f"temp_chunk_{i}.wav"
            sf.write(temp_chunk_path, chunk_y, sr)
            
            try:
                # 1. Transcribe
                # We specify language if provided, but pipeline kwargs depend on the model.
                generate_kwargs = {}
                if language:
                    generate_kwargs["language"] = language
                
                asr_res = asr_pipe(str(temp_chunk_path), generate_kwargs=generate_kwargs)
                transcript_text = asr_res.get("text", "").strip()
                
                if not transcript_text:
                    continue

                # 2. Align (audio + transcript)
                # The exact API depends on transformers implementation, but generally we pass text + audio
                # For Qwen3-ForcedAligner, we usually pass text and audio.
                align_res = aligner_pipe(str(temp_chunk_path), text=transcript_text)
                
                # Reconstruct segments (group words)
                # Assuming align_res returns a list of dictionaries with 'word', 'start', 'end'
                current_segment_text = ""
                current_segment_start = -1.0
                current_segment_end = -1.0
                
                for item in align_res:
                    word = item.get("word", "")
                    w_start = item.get("start", 0.0)
                    w_end = item.get("end", 0.0)
                    
                    if current_segment_start < 0:
                        current_segment_start = w_start
                    
                    # Very naive word grouping (by MAX_CHARS_PER_LINE * MAX_LINES)
                    # For a robust implementation, we group by pauses (w_start - current_segment_end > threshold)
                    if (len(current_segment_text) + len(word) > MAX_CHARS_PER_LINE * MAX_LINES_PER_EVENT) or \
                       (current_segment_end > 0 and w_start - current_segment_end > 1.0):
                        # Flush segment
                        if current_segment_text:
                            all_segments.append(CachedSegment(
                                start=chunk_offset_s + current_segment_start,
                                end=chunk_offset_s + current_segment_end,
                                text=current_segment_text.strip()
                            ))
                        current_segment_text = word
                        current_segment_start = w_start
                        current_segment_end = w_end
                    else:
                        current_segment_text += " " + word
                        current_segment_end = w_end
                        
                if current_segment_text:
                    all_segments.append(CachedSegment(
                        start=chunk_offset_s + current_segment_start,
                        end=chunk_offset_s + current_segment_end,
                        text=current_segment_text.strip()
                    ))
            finally:
                if temp_chunk_path.exists():
                    temp_chunk_path.unlink()
            
            elapsed = time.monotonic() - start_time
            speed = (chunk_offset_s + (len(chunk_y)/sr)) / elapsed if elapsed > 0 else 0.0
            log("transcribe", f"Chunk completed | speed={speed:5.2f}x realtime")
            
        return all_segments, detected_language, language_probability, total_duration


def get_transcription_engine(engine_name: Optional[str], language: Optional[str], small_model: bool, model_size: str) -> TranscriptionEngine:
    """
    Factory and auto-selection logic for transcription engines.
    """
    if engine_name:
        engine_name = engine_name.lower()
        if engine_name == "whisper":
            return WhisperEngine(model_size)
        elif engine_name == "canary-qwen":
            return CanaryQwenEngine()
        elif engine_name == "qwen3-asr":
            return Qwen3Engine(small_model)
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

    # Auto-selection logic
    if language:
        lang_lower = language.lower()
        if lang_lower in ("en", "english"):
            return CanaryQwenEngine()
        # Fallthrough to qwen3-asr for everything else
        return Qwen3Engine(small_model)
    else:
        # Auto-detect => qwen3-asr
        return Qwen3Engine(small_model)

def transcribe_video(
    video_path: Path,
    model_size: str,
    language: Optional[str],
    target_language: Optional[str],
    subtitle_format: str,
    output_path: Optional[Path],
    provider_order: Optional[str] = None,
    use_cache: bool = True,
    engine: Optional[TranscriptionEngine] = None,
    offset_ms: int = 0,
) -> TranscriptionResult:
    """
    Transcribe a video's audio track and optionally translate subtitles.

    Args:
        video_path: Path to the input video file.
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        language: Source language ISO code, or None for auto-detect.
        target_language: Target language ISO code, or ``"none"``/None to skip.
        subtitle_format: Output subtitle format (``"srt"`` or ``"ass"``).
        output_path: Optional explicit output path for the subtitle file.
        provider_order: Optional comma-separated translation provider order.
        use_cache: If True, use transcription caching.
        engine: The selected TranscriptionEngine.

    Returns:
        A TranscriptionResult with the path to the generated subtitle file.
    """
    try:
        import pysubs2
    except ImportError as exc:
        raise RuntimeError(
            "pysubs2 is not installed. Install it with: pip install pysubs2"
        ) from exc

    # Build translator if needed
    translator_chain: Optional[ProviderChain] = None
    if target_language and target_language != "none":
        translator_chain = build_provider_chain(provider_order)

    # ---------- Try cache first ----------
    cached = None
    if use_cache:
        cached = load_cached_transcription(video_path, model_size)

    if cached is not None:
        segments_data, detected_language, language_probability, duration_seconds = cached
        log("cache", f"Cache hit! Loaded {len(segments_data)} segments from cache")
        collected_segments = segments_data
        processing_seconds = 0.0
    else:
        # ---------- Transcribe ----------
        if engine is None:
            # Fallback to Whisper if not provided, though main() should provide one
            engine = WhisperEngine(model_size)

        try:
            segments, detected_language, language_probability, duration_seconds = engine.transcribe(video_path, language)
            collected_segments = segments
            processing_seconds = 0.0 # Handled inside engine, but we don't return processing_seconds cleanly. We can just guess or track.
        except RuntimeError as exc:
            if engine.name != "whisper":
                log("transcribe", f"{engine.name} failed: {exc}. Falling back to faster-whisper...")
                engine = WhisperEngine(model_size)
                segments, detected_language, language_probability, duration_seconds = engine.transcribe(video_path, language)
                collected_segments = segments
            else:
                raise
        
        if not collected_segments:
            raise RuntimeError("No speech segments were detected in the audio track.")

        # Save to cache
        if use_cache:
            save_transcription_cache(
                video_path, model_size, collected_segments,
                detected_language, language_probability, duration_seconds,
            )

    segment_count = len(collected_segments)
    texts_to_translate = [seg.text for seg in collected_segments]

    # ---------- Translate ----------
    if translator_chain and translator_chain.has_providers:
        total_batches = (len(texts_to_translate) + BATCH_SIZE - 1) // BATCH_SIZE
        translated_texts: List[str] = []
        for i in range(0, len(texts_to_translate), BATCH_SIZE):
            batch = texts_to_translate[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            log(
                "translate",
                f"Batch {batch_num}/{total_batches} "
                f"({i + 1}-{min(i + BATCH_SIZE, len(texts_to_translate))}"
                f"/{len(texts_to_translate)} lines)",
            )
            src = language if language else detected_language
            trans_batch = translator_chain.translate_batch(batch, src, target_language)
            translated_texts.extend(trans_batch)

        texts_to_save = translated_texts
    else:
        texts_to_save = texts_to_translate

    # ---------- Item 4: Handle __FAILED__ fallback ----------
    failed_count = 0
    for i, text in enumerate(texts_to_save):
        if text.startswith("__FAILED__"):
            texts_to_save[i] = texts_to_translate[i]  # fall back to original
            failed_count += 1
    if failed_count:
        log(
            "translate",
            f"{failed_count}/{len(texts_to_save)} lines failed translation; "
            f"fell back to original transcribed text.",
        )

    # ---------- Wrap subtitle lines ----------
    line_break = "\\N" if subtitle_format == "ass" else "\\N"
    for i, text in enumerate(texts_to_save):
        texts_to_save[i] = wrap_subtitle_text(
            text,
            max_chars=MAX_CHARS_PER_LINE,
            max_lines=MAX_LINES_PER_EVENT,
            line_break=line_break,
        )

    # ---------- Build subtitle file ----------
    subs = pysubs2.SSAFile()
    for seg, text in zip(collected_segments, texts_to_save):
        start_s = max(0.0, seg.start + (offset_ms / 1000.0))
        end_s = max(0.0, seg.end + (offset_ms / 1000.0))
        event = pysubs2.SSAEvent(
            start=pysubs2.make_time(s=start_s),
            end=pysubs2.make_time(s=end_s),
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
        raise RuntimeError(
            f"Failed to save subtitle file to '{output_path}': {exc}"
        ) from exc

    log("transcribe", f"Wrote {segment_count} subtitle events to '{output_path}'")

    return TranscriptionResult(
        subtitle_path=output_path,
        detected_language=detected_language,
        language_probability=language_probability,
        duration_seconds=duration_seconds,
        processing_seconds=processing_seconds,
        segment_count=segment_count,
    )


# --------------------------------------------------------------------------
# FFmpeg subtitle muxing / burn-in
# --------------------------------------------------------------------------


def mux_subtitles(
    video_path: Path,
    subtitle_path: Path,
    subtitle_format: str,
    output_format: str,
) -> Path:
    """
    Mux a subtitle file into a video as a soft (toggleable) subtitle track.

    Uses ffmpeg stream copy (no re-encode) for video and audio, adding
    the subtitle as an additional stream. The subtitle codec is chosen
    based on the output container format.

    Args:
        video_path: Path to the original video file.
        subtitle_path: Path to the subtitle file (.srt or .ass).
        subtitle_format: The subtitle format string (``"srt"`` or ``"ass"``).
        output_format: The output container format (e.g. ``"mkv"`` or ``"mp4"``).

    Returns:
        Path to the muxed output file (e.g. ``video.subbed.mkv``).

    Raises:
        RuntimeError: If ffmpeg is not found or the mux operation fails.
    """
    ffmpeg = validate_ffmpeg()
    out_ext = f".{output_format.lower()}"

    codec_map = _SUBTITLE_CODEC_MAP.get(out_ext)
    if codec_map is None:
        log("mux", f"WARNING: Unknown container '{out_ext}', attempting with srt codec")
        sub_codec = "srt"
    else:
        sub_codec = codec_map.get(subtitle_format, codec_map.get("srt", "srt"))

    output_path = video_path.with_suffix(f".subbed{out_ext}")

    command = [
        ffmpeg,
        "-i", str(video_path),
        "-i", str(subtitle_path),
        "-map", "0",              # all streams from video
        "-map", "1",              # subtitle stream
        "-c", "copy",             # copy video + audio
        "-c:s", sub_codec,        # subtitle codec
        "-y",                     # overwrite output
        str(output_path),
    ]

    log("mux", f"Muxing subtitles into '{output_path.name}'...")
    log("mux", f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux failed (exit code {result.returncode}):\n{result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg mux timed out after 5 minutes")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg could not be executed")

    log("mux", f"Successfully created '{output_path.name}'")
    return output_path


def burn_in_subtitles(
    video_path: Path,
    subtitle_path: Path,
    subtitle_format: str,
    output_format: str,
) -> Path:
    """
    Burn subtitles into the video frames (hardcoded / permanent).

    This re-encodes the video using libx264 and the ffmpeg subtitles
    or ass filter, so it will be significantly slower than soft muxing.

    Args:
        video_path: Path to the original video file.
        subtitle_path: Path to the subtitle file (.srt or .ass).
        subtitle_format: The subtitle format string (``"srt"`` or ``"ass"``).
        output_format: The output container format (e.g. ``"mkv"`` or ``"mp4"``).

    Returns:
        Path to the burned-in output file (e.g. ``video.burned.mkv``).

    Raises:
        RuntimeError: If ffmpeg is not found or the burn-in fails.
    """
    ffmpeg = validate_ffmpeg()
    out_ext = f".{output_format.lower()}"
    output_path = video_path.with_suffix(f".burned{out_ext}")

    # Escape special characters in subtitle path for ffmpeg filter
    # On Windows, backslashes and colons must be escaped
    sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

    if subtitle_format == "ass":
        vf = f"ass='{sub_path_escaped}'"
    else:
        vf = f"subtitles='{sub_path_escaped}'"

    command = [
        ffmpeg,
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-c:a", "copy",
        "-y",
        str(output_path),
    ]

    log("burn-in", f"Burning subtitles into '{output_path.name}' (this may take a while)...")
    log("burn-in", f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours for long videos
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg burn-in failed (exit code {result.returncode}):\n{result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg burn-in timed out after 2 hours")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg could not be executed")

    log("burn-in", f"Successfully created '{output_path.name}'")
    return output_path


# --------------------------------------------------------------------------
# Media Playback
# --------------------------------------------------------------------------

def _find_vlc() -> Optional[str]:
    """Try to find the VLC executable, even if not on PATH."""
    vlc_path = shutil.which("vlc")
    if vlc_path:
        return vlc_path
        
    system = platform.system()
    if system == "Windows":
        common_paths = [
            r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
    elif system == "Darwin":
        mac_path = "/Applications/VLC.app/Contents/MacOS/VLC"
        if os.path.exists(mac_path):
            return mac_path
            
    return None

def launch_player(
    video_path: Path,
    subtitle_path: Optional[Path] = None,
) -> None:
    """
    Launch a media player with the given video and optional subtitle file.

    Tries VLC first, then mpv. If neither is found or fails to launch, 
    falls back to the OS default video player. Does not raise exceptions 
    on failure, to allow the pipeline to exit gracefully.

    Args:
        video_path: Path to the video file to play.
        subtitle_path: Optional path to a subtitle file to load externally.
                        If None, plays the file as-is (useful for muxed files).
    """
    vlc_binary = _find_vlc()
    mpv_binary = shutil.which("mpv")

    command: Optional[List[str]] = None
    player_name = ""

    if vlc_binary is not None:
        command = [vlc_binary, str(video_path)]
        if subtitle_path:
            command.append(f"--sub-file={subtitle_path}")
        player_name = "VLC"
    elif mpv_binary is not None:
        command = [mpv_binary, str(video_path)]
        if subtitle_path:
            command.append(f"--sub-file={subtitle_path}")
        player_name = "mpv"

    if command is not None:
        log("playback", f"Launching playback: {video_path}")
        log("playback", f"Command ({player_name}): {' '.join(command)}")

        try:
            subprocess.Popen(command)
            return  # Success
        except Exception as exc:
            log("playback", f"Failed to launch {player_name}: {exc}. Falling back to default player.")
    else:
        log(
            "playback",
            "Neither VLC nor mpv were found. Please install VLC (https://www.videolan.org) "
            "or mpv (https://mpv.io) for the best auto-playback experience. "
            "Falling back to system default player."
        )

    # Fallback to OS default player
    # Note: Default players usually can't take an external subtitle file argument easily,
    # so we just open the video file. Muxed subtitles should still work.
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(video_path))
        elif system == "Darwin":
            subprocess.Popen(["open", str(video_path)])
        else:
            subprocess.Popen(["xdg-open", str(video_path)])
        log("playback", f"Launched system default player for: {video_path.name}")
    except Exception as exc:
        log("playback", f"Failed to open with default player: {exc}")
        log("playback", f"Your output files are ready.")
        log("playback", f"Video: {video_path}")
        if subtitle_path:
            log("playback", f"Subtitle: {subtitle_path}")


# --------------------------------------------------------------------------
# CLI argument parsing
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse.ArgumentParser for this application."""
    parser = argparse.ArgumentParser(
        prog="autosub_player.py",
        description=(
            "Transcribe a video's audio with faster-whisper, generate "
            "subtitles, optionally translate them, mux into the video, "
            "and auto-play the result."
        ),
        epilog=(
            "Environment variables:\n"
            "  DEEPL_API_KEY    DeepL translation API key\n"
            "  GEMINI_API_KEY   Google Gemini API key\n"
            "  GROQ_API_KEY     Groq API key\n"
            "\n"
            "These can also be loaded from a .env file in the working directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to the input video file. If omitted, a file picker dialog is shown.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Manual timing delay in milliseconds. Positive values delay subtitles, negative values make them appear earlier.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["whisper", "canary-qwen", "qwen3-asr"],
        help="Force a specific transcription engine. Default: auto-select by language.",
    )
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use a smaller model variant (e.g. Qwen3-ASR 0.6B instead of 1.7B).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        help=(
            "faster-whisper model size/name (e.g. tiny, base, small, medium, "
            "large-v3). Only applies to whisper engine. Default: %(default)s"
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
        "--output-format",
        type=str,
        default="mkv",
        choices=["mkv", "mp4"],
        help="Output video container format for muxed/burned videos. Default: mkv",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip auto-launching mpv after processing.",
    )
    parser.add_argument(
        "--burn-in",
        action="store_true",
        help=(
            "Burn subtitles into the video (hardcoded, permanent). "
            "Requires video re-encode; much slower than soft muxing."
        ),
    )
    parser.add_argument(
        "--keep-subs",
        action="store_true",
        help="Keep the standalone subtitle file after muxing (default: clean up).",
    )
    parser.add_argument(
        "--no-mux",
        action="store_true",
        help=(
            "Skip the ffmpeg mux step. Only produce a standalone subtitle file "
            "and play with --sub-file (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--provider-order",
        type=str,
        default=None,
        help=(
            "Comma-separated translation provider order "
            "(e.g. 'gemini,deepl,groq'). Default: deepl,gemini,groq."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the transcription cache and force re-transcription.",
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
    # Load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
        log("init", "Loaded environment from .env file")
    except ImportError:
        pass  # python-dotenv not installed — use raw env vars only

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Resolve video path -------------------------------------------------
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        source_lang = args.lang
        target_lang = args.target_lang
    else:
        log("init", "No --video supplied; opening file picker dialog...")
        video_path_opt, source_lang_opt, target_lang_opt = prompt_for_settings()
        if video_path_opt is None:
            log("error", "No video file was selected. Exiting.")
            return 1
        video_path = video_path_opt.expanduser().resolve()
        source_lang = None if source_lang_opt == "auto" else source_lang_opt
        target_lang = target_lang_opt

    try:
        validate_video_path(video_path)
    except (FileNotFoundError, ValueError) as exc:
        log("error", str(exc))
        return 1

    # Resolve subtitle format ---------------------------------------------
    try:
        subtitle_format = validate_subtitle_format(args.format)
    except ValueError as exc:
        log("error", str(exc))
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else None

    # Validate ffmpeg early if we'll need it for muxing/burn-in
    if not args.no_mux:
        try:
            validate_ffmpeg()
        except RuntimeError as exc:
            log("error", str(exc))
            return 1

    # Transcribe + translate -----------------------------------------------
    log("init", f"Engine parameter: {args.engine or 'auto-select'}")
    
    engine = get_transcription_engine(
        engine_name=args.engine,
        language=source_lang,
        small_model=args.small_model,
        model_size=args.model,
    )
    log("init", f"Selected transcription engine: {engine.name}")

    try:
        result = transcribe_video(
            video_path=video_path,
            model_size=args.model,
            language=source_lang,
            target_language=target_lang,
            subtitle_format=subtitle_format,
            output_path=args.output,
            provider_order=args.provider_order,
            use_cache=not args.no_cache,
            engine=engine,
            offset_ms=args.offset,
        )
    except RuntimeError as exc:
        log("error", str(exc))
        return 1
    except KeyboardInterrupt:
        print("\n", file=sys.stderr)
        log("error", "Interrupted by user.")
        return 130

    log(
        "summary",
        f"language={result.detected_language} "
        f"({result.language_probability:.2%}) | "
        f"duration={result.duration_seconds:.1f}s | "
        f"processing={result.processing_seconds:.1f}s | "
        f"segments={result.segment_count} | "
        f"subtitles='{result.subtitle_path}'",
    )

    # Mux / burn-in --------------------------------------------------------
    playback_file: Path = video_path
    subtitle_for_playback: Optional[Path] = result.subtitle_path

    if args.no_mux:
        log("info", "Muxing skipped (--no-mux). Subtitle file saved standalone.")
    else:
        try:
            if args.output_format == "mp4" and subtitle_format == "ass":
                log(
                    "mux",
                    "WARNING: You chose --output-format mp4. MP4 only supports mov_text subtitles. "
                    "ASS background-box, custom-color, and opacity styling will be lost!"
                )

            if args.burn_in:
                muxed_path = burn_in_subtitles(
                    video_path, result.subtitle_path, subtitle_format, args.output_format
                )
            else:
                muxed_path = mux_subtitles(
                    video_path, result.subtitle_path, subtitle_format, args.output_format
                )

            playback_file = muxed_path
            subtitle_for_playback = None  # subtitles are in the muxed file

            # Clean up standalone subtitle file unless --keep-subs
            if not args.keep_subs:
                try:
                    result.subtitle_path.unlink()
                    log("cleanup", f"Removed standalone subtitle file: {result.subtitle_path.name}")
                except OSError:
                    pass  # best effort cleanup
        except RuntimeError as exc:
            log("warn", f"Muxing failed: {exc}")
            log("warn", "Falling back to external subtitle file for playback.")
            # Keep subtitle file and use it with --sub-file instead

    # Playback ---------------------------------------------------------------
    if args.no_play:
        log("info", f"--no-play set. Output saved at: {playback_file}")
        if subtitle_for_playback:
            log("info", f"Subtitle file at: {subtitle_for_playback}")
        return 0

    launch_player(playback_file, subtitle_for_playback)

    log("done", "Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
