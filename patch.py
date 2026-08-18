import re

with open('autosub_player.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace requirements.txt section
content = content.replace('faster-whisper>=1.0.0\npysubs2>=1.6.0\nctranslate2>=4.0.0', 
                          'faster-whisper>=1.0.0\npysubs2>=1.6.0\nctranslate2>=4.0.0\ndeepl\ngoogle-genai\ngroq')

# Add imports
imports_to_add = '''import os
import re
import html
import json
from abc import ABC, abstractmethod'''
content = content.replace('import sys\nimport time', 'import sys\nimport time\n' + imports_to_add)

transcribe_idx = content.find('def transcribe_video(')
transcribe_header_idx = content.rfind('# --------------------------------------------------------------------------', 0, transcribe_idx)

translation_logic = '''# --------------------------------------------------------------------------
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
    clean = text.replace("\\u202B", "").replace("\\u202C", "")
    return f"\\u202B{clean}\\u202C"

_TAG_RE = re.compile(r"\\{[^}]*\\}")

def hide_tags(line: str):
    tags = _TAG_RE.findall(line)
    clean = line
    placeholders = {}
    for idx, tag in enumerate(tags):
        ph = f"<tx{idx}/>"
        clean = clean.replace(tag, f"\\x00PH{idx}\\x00", 1)
        placeholders[idx] = tag

    parts = re.split(r"\\x00PH(\\d+)\\x00", clean)
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
            match = re.search(rf'<p[^>]*\\bid="{i}"[^>]*>(.*?)</p>', translated_xml, re.DOTALL)
            if match:
                ara_text = match.group(1).strip()
                restored = restore_tags(ara_text, batch_tags[i])
                final_lines.append(apply_rtl_fix(restored) if target_lang == "ar" else restored)
            else:
                final_lines.append("__FAILED__" + lines[i])
        return final_lines

_LLM_TRANSLATE_PROMPT = """\\
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
            text = re.sub(r"^```\\w*\\n?", "", text)
            text = re.sub(r"\\n?```$", "", text)
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
            text = re.sub(r"^```\\w*\\n?", "", text)
            text = re.sub(r"\\n?```$", "", text)
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
'''

launch_idx = content.find('def launch_mpv(')
playback_header_idx = content.rfind('# -----', 0, launch_idx)

content = content[:transcribe_header_idx] + translation_logic + content[playback_header_idx:]

with open('autosub_player.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("done")
