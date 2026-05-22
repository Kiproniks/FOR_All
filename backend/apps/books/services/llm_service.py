from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from typing import Any

import pymorphy3
import requests
from razdel import sentenize

logger = logging.getLogger(__name__)
morph = pymorphy3.MorphAnalyzer()

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
MAX_CHAPTER_MAP_CHARS = 6500
MAX_BLOCK_PROMPT_CHARS = 1200
DEFAULT_EXPLANATION = "Concept is grounded in this logical block and explained by the source fragment."

STOP_WORDS = {
    "книга",
    "автор",
    "текст",
    "пример",
    "задача",
    "глава",
    "раздел",
    "данный",
    "этот",
    "тема",
}

GENERIC_SUMMARY_MARKERS_RU = {
    "в этом блоке рассматривается",
    "в данной главе рассматривается",
    "в тексте рассматривается",
    "данный раздел посвящен",
    "основная идея заключается",
    "данный материал",
    "main meaning",
    "2-4 предложения",
    "конкретикой из блоков",
    "по сути, с конкретикой",
}
GENERIC_SUMMARY_MARKERS_EN = {
    "this section discusses",
    "this chapter discusses",
    "this block discusses",
    "the main idea is",
    "this material describes",
    "main meaning",
    "2-4 sentences",
}

_OLLAMA_DISABLED_UNTIL = 0.0


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    clean = (text or "").strip()
    if len(clean) <= max_chars:
        return clean
    head = clean[: max_chars - 220]
    tail = clean[-180:]
    return f"{head}\n...\n{tail}"


def _text_language_hint(text: str) -> str:
    cyr_count = len(CYRILLIC_RE.findall(text or ""))
    lat_count = len(LATIN_RE.findall(text or ""))
    if cyr_count > 0 and cyr_count >= lat_count:
        return "ru"
    if lat_count > 0:
        return "en"
    return "auto"


def _is_language_compatible(value: str, lang_hint: str) -> bool:
    if not value:
        return False
    if CJK_RE.search(value):
        return False
    if lang_hint == "ru":
        if CYRILLIC_RE.search(value):
            return True
        if LATIN_RE.search(value):
            return True
        upper_latin = re.fullmatch(r"[A-Z0-9\-]{2,8}", value.strip())
        return bool(upper_latin)
    return True


def _ollama_enabled() -> bool:
    return os.getenv("LLM_PROVIDER", "ollama").lower() == "ollama"


def _ollama_generate(prompt: str, *, expect_json: bool = False) -> str | None:
    global _OLLAMA_DISABLED_UNTIL

    if not _ollama_enabled():
        return None

    now = time.time()
    if now < _OLLAMA_DISABLED_UNTIL:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    endpoint = f"{base_url}/api/generate"
    timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
    max_tokens = int(
        os.getenv(
            "OLLAMA_MAX_TOKENS_JSON" if expect_json else "OLLAMA_MAX_TOKENS_TEXT",
            "220" if expect_json else "140",
        )
    )
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096,
            "num_predict": max_tokens,
        },
    }
    if expect_json:
        payload["format"] = "json"

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()
    except requests.exceptions.ReadTimeout:
        logger.warning("Ollama read timeout, fallback branch will be used")
        timeout_cooldown = int(os.getenv("OLLAMA_TIMEOUT_COOLDOWN_SECONDS", "180"))
        _OLLAMA_DISABLED_UNTIL = time.time() + max(15, timeout_cooldown)
        return None
    except Exception:
        logger.exception("Ollama call failed")
        cooldown = int(os.getenv("OLLAMA_RETRY_COOLDOWN_SECONDS", "120"))
        _OLLAMA_DISABLED_UNTIL = time.time() + max(15, cooldown)
        return None


def _clean_json_candidate(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    match = JSON_BLOCK_RE.search(value)
    if match:
        return match.group(1).strip()
    return value


def _safe_json_parse(raw: str) -> Any:
    value = _clean_json_candidate(raw)
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        left = value.find("[")
        right = value.rfind("]")
        if left != -1 and right != -1 and left < right:
            try:
                return json.loads(value[left : right + 1])
            except json.JSONDecodeError:
                pass

        left = value.find("{")
        right = value.rfind("}")
        if left != -1 and right != -1 and left < right:
            try:
                return json.loads(value[left : right + 1])
            except json.JSONDecodeError:
                return None
    return None


def _concept_items_from_parsed(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("concepts", "items", "data", "results"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _concept_items_from_partial_json(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    list_start = raw.find("[")
    if list_start == -1:
        return []

    items: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = list_start + 1
    length = len(raw)

    while idx < length:
        while idx < length and raw[idx] in " \r\n\t,":
            idx += 1
        if idx >= length or raw[idx] == "]":
            break
        try:
            obj, end_idx = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            next_obj = raw.find("{", idx + 1)
            if next_obj == -1:
                break
            idx = next_obj
            continue
        if isinstance(obj, dict):
            items.append(obj)
        idx = end_idx

    return items


def _pick_fallback_sentences(block_text: str, limit: int = 3) -> str:
    sentences = [item.text.strip() for item in sentenize(block_text) if len(item.text.strip()) >= 35]
    if not sentences:
        sentences = [segment.strip() for segment in re.split(r"[.!?]+", block_text) if len(segment.strip()) >= 35]
    return " ".join(sentences[:limit]).strip() or (block_text or "")[:500].strip()


def _lemma_content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in WORD_RE.findall(text or ""):
        token = raw.lower()
        if len(token) < 3 or token.isdigit():
            continue
        if token in STOP_WORDS:
            continue
        parsed = morph.parse(token)[0]
        lemma = parsed.normal_form
        if len(lemma) < 3 or lemma in STOP_WORDS:
            continue
        tokens.add(lemma)
    return tokens


def _is_generic_summary(summary: str) -> bool:
    low = (summary or "").lower()
    if "main meaning" in low:
        return True
    if "2-4 предложения" in low or "2-4 sentences" in low:
        return True
    if "по сути, с конкретикой из блоков" in low:
        return True
    ru_hits = sum(1 for marker in GENERIC_SUMMARY_MARKERS_RU if marker in low)
    en_hits = sum(1 for marker in GENERIC_SUMMARY_MARKERS_EN if marker in low)
    if ru_hits >= 2 or en_hits >= 2:
        return True
    if len(low) < 65 and (ru_hits >= 1 or en_hits >= 1):
        return True
    return False


def _grounding_overlap(summary: str, evidence_text: str) -> float:
    summary_tokens = _lemma_content_tokens(summary)
    evidence_tokens = _lemma_content_tokens(evidence_text)
    if not summary_tokens or not evidence_tokens:
        return 0.0
    overlap = len(summary_tokens & evidence_tokens)
    return overlap / max(1, len(summary_tokens))


def extractive_theme_summary_from_digests(block_digests: list[dict[str, Any]], limit_sentences: int = 3) -> str:
    parts: list[str] = []
    for item in block_digests:
        text = str(item.get("summary", "")).strip()
        if not text:
            continue
        parts.append(text)
        if len(parts) >= max(1, limit_sentences):
            break
    return " ".join(parts).strip()[:1900]


def ensure_grounded_summary(summary: str, evidence_text: str, fallback_summary: str = "") -> str:
    candidate = " ".join((summary or "").split()).strip()
    if not candidate:
        candidate = " ".join((fallback_summary or "").split()).strip()
    if not candidate:
        candidate = _pick_fallback_sentences(evidence_text, limit=3)

    overlap = _grounding_overlap(candidate, evidence_text)
    if overlap < 0.18 or _is_generic_summary(candidate):
        repaired = " ".join((fallback_summary or "").split()).strip()
        if not repaired:
            repaired = _pick_fallback_sentences(evidence_text, limit=3)
        candidate = repaired

    return candidate[:2000]


def summarize_logical_block(block_text: str) -> str:
    lang_hint = _text_language_hint(block_text)
    prompt_text = _truncate_for_prompt(block_text, MAX_BLOCK_PROMPT_CHARS)
    prompt = (
        "You analyze an educational or scientific book fragment.\n"
        "Return a concise semantic summary without adding facts outside the source.\n"
        "Length: 2-4 sentences.\n"
        "Language rule: if source contains Cyrillic, answer in Russian. Otherwise use source language.\n\n"
        f"Text:\n{prompt_text}\n\n"
        "Return plain text only."
    )
    llm_result = _ollama_generate(prompt, expect_json=False)
    if llm_result:
        if lang_hint != "ru" or CYRILLIC_RE.search(llm_result):
            return llm_result
    return _pick_fallback_sentences(block_text, limit=3)


def _fallback_extract_concepts(block_text: str, limit: int = 7) -> list[dict[str, Any]]:
    words = [item.lower() for item in WORD_RE.findall(block_text)]
    noun_counter: Counter[str] = Counter()
    for word in words:
        if len(word) <= 2 or word.isdigit():
            continue
        parsed = morph.parse(word)[0]
        lemma = parsed.normal_form
        if "NOUN" not in parsed.tag:
            continue
        if lemma in STOP_WORDS:
            continue
        noun_counter[lemma] += 1

    concepts: list[dict[str, Any]] = []
    quote = _pick_fallback_sentences(block_text, limit=1)[:300]
    top_items = noun_counter.most_common(limit)
    for rank, (name, freq) in enumerate(top_items, start=1):
        score = max(0.1, min(1.0, freq / (top_items[0][1] if top_items else 1)))
        concepts.append(
            {
                "name": name,
                "short_explanation": DEFAULT_EXPLANATION,
                "source_quote": quote,
                "importance_score": round(score * (1 - rank * 0.03), 3),
            }
        )
    if concepts:
        return concepts[: max(3, min(limit, len(concepts)))]
    return []


def _clean_concept_items(items: list[dict[str, Any]], *, lang_hint: str) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        name = str(item.get("name", "")).strip()
        explanation = str(item.get("short_explanation", "")).strip()
        quote = str(item.get("source_quote", "")).strip()
        score = item.get("importance_score", 0.5)

        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.5
        score_value = max(0.0, min(1.0, score_value))

        if not name or len(name) > 140:
            continue
        if lang_hint in {"ru", "en"} and not _is_language_compatible(name, lang_hint):
            continue

        dedupe_key = name.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if lang_hint == "ru" and explanation and not _is_language_compatible(explanation, "ru"):
            explanation = ""
        if lang_hint == "ru" and quote and not _is_language_compatible(quote, "ru"):
            quote = ""

        cleaned.append(
            {
                "name": name,
                "short_explanation": explanation or DEFAULT_EXPLANATION,
                "source_quote": quote[:800],
                "importance_score": score_value,
            }
        )

    cleaned.sort(key=lambda item: item["importance_score"], reverse=True)
    return cleaned[:10]


def _top_up_concepts_with_fallback(
    concepts: list[dict[str, Any]],
    block_text: str,
    *,
    minimum: int = 3,
) -> list[dict[str, Any]]:
    if len(concepts) >= minimum:
        return concepts
    fallback = _fallback_extract_concepts(block_text)
    seen = {item["name"].strip().lower() for item in concepts if item.get("name")}
    for item in fallback:
        name = item.get("name", "").strip().lower()
        if not name or name in seen:
            continue
        concepts.append(item)
        seen.add(name)
        if len(concepts) >= minimum:
            break
    return concepts[:10]


def extract_concepts_from_block(block_text: str, block_summary: str) -> list[dict[str, Any]]:
    lang_hint = _text_language_hint(block_text)
    def _build_prompt(max_chars: int) -> str:
        text_for_prompt = _truncate_for_prompt(block_text, max_chars)
        return (
            "You analyze one logical block of a book.\n"
            "Extract only meaningful concepts that are explicitly supported by the text.\n"
            "Do not output generic terms like book, text, author, chapter.\n"
            "Language rule: if source contains Cyrillic, all fields must be in Russian.\n"
            "Return strict JSON object with key `concepts`.\n"
            "Schema:\n"
            "{\n"
            '  "concepts": [\n'
            "    {\n"
            '      "name": "string",\n'
            '      "short_explanation": "string",\n'
            '      "source_quote": "string",\n'
            '      "importance_score": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Concept count: 3..6.\n\n"
            "Keep the answer compact:\n"
            "- short_explanation: up to 20 words\n"
            "- source_quote: up to 140 characters\n"
            "- total response size: up to 1200 characters\n\n"
            f"Block text:\n{text_for_prompt}\n\n"
            f"Block summary:\n{_truncate_for_prompt(block_summary, 700)}"
        )

    for max_chars in (MAX_BLOCK_PROMPT_CHARS, 800):
        llm_result = _ollama_generate(_build_prompt(max_chars), expect_json=True)
        if not llm_result:
            continue
        parsed = _safe_json_parse(llm_result)
        items = _concept_items_from_parsed(parsed)
        if not items:
            items = _concept_items_from_partial_json(llm_result)
        cleaned = _clean_concept_items(items, lang_hint=lang_hint)
        if cleaned:
            return _top_up_concepts_with_fallback(cleaned, block_text)

    fallback = _fallback_extract_concepts(block_text)
    if fallback:
        return fallback
    return []


def summarize_book(block_summaries: list[str]) -> str:
    joined = "\n".join(f"- {item}" for item in block_summaries if item.strip())
    lang_hint = _text_language_hint(joined)
    prompt = (
        "Build a concise structured book summary from block summaries only.\n"
        "Do not add facts outside provided blocks.\n"
        "Language rule: if input contains Cyrillic, answer in Russian.\n\n"
        f"Blocks:\n{_truncate_for_prompt(joined, 14000)}\n\n"
        "Return:\n1. General theme\n2. Key ideas\n3. Learning outcome"
    )
    llm_result = _ollama_generate(prompt, expect_json=False)
    if llm_result and (lang_hint != "ru" or CYRILLIC_RE.search(llm_result)):
        return llm_result

    if not block_summaries:
        return ""
    return (
        "1. General theme: source material explains key ideas from the original text.\n"
        f"2. Key ideas: {'; '.join(block_summaries[:3])}\n"
        "3. Learning outcome: reader understands the core concepts and links between them."
    )


def compare_concept_mentions(concept_name: str, mentions: list[dict[str, Any]]) -> str:
    mention_lines = []
    for item in mentions:
        mention_lines.append(
            f"- book={item.get('book_title')} | block={item.get('block_title')} | explanation={item.get('short_explanation')}"
        )

    lang_hint = _text_language_hint(concept_name + " " + " ".join(mention_lines))
    prompt = (
        "Compare how the same concept is explained across sources.\n"
        "Use only provided evidence.\n"
        "Language rule: if input contains Cyrillic, answer in Russian.\n\n"
        f"Concept:\n{concept_name}\n\n"
        f"Sources:\n{_truncate_for_prompt(chr(10).join(mention_lines), 12000)}\n\n"
        "Return:\n"
        "1. Common points\n2. Differences\n3. Where simpler\n4. Where deeper\n5. Final takeaway"
    )
    llm_result = _ollama_generate(prompt, expect_json=False)
    if llm_result and (lang_hint != "ru" or CYRILLIC_RE.search(llm_result)):
        return llm_result

    if not mentions:
        return "Not enough data to compare this concept."

    books = ", ".join(sorted({str(item.get("book_title", "")) for item in mentions if item.get("book_title")}))
    return (
        f"1. Common points: concept '{concept_name}' appears in multiple user sources.\n"
        "2. Differences: emphasis changes by context of each source.\n"
        "3. Where simpler: shorter and more applied explanations.\n"
        "4. Where deeper: longer blocks with expanded reasoning.\n"
        f"5. Final takeaway: comparison built from books {books or 'without source labels'}."
    )


def _fallback_chapter_boundaries(
    paragraphs: list[str],
    min_words: int,
    target_words: int,
    max_words: int,
) -> list[int]:
    boundaries: list[int] = []
    acc_words = 0

    for index, paragraph in enumerate(paragraphs, start=1):
        acc_words += _count_words(paragraph)
        if acc_words >= max_words:
            boundaries.append(index)
            acc_words = 0
        elif acc_words >= target_words:
            boundaries.append(index)
            acc_words = 0

    if not boundaries or boundaries[-1] != len(paragraphs):
        boundaries.append(len(paragraphs))

    cleaned: list[int] = []
    prev = 0
    for item in boundaries:
        if item <= prev:
            continue
        chunk_words = sum(_count_words(paragraphs[i]) for i in range(prev, item))
        if cleaned and chunk_words < max(40, min_words // 2):
            cleaned[-1] = item
        else:
            cleaned.append(item)
        prev = item

    return cleaned


def suggest_chapter_boundaries(
    chapter_title: str,
    paragraphs: list[str],
    *,
    min_words: int,
    target_words: int,
    max_words: int,
) -> list[int]:
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [1]

    total_words = sum(_count_words(item) for item in paragraphs)
    desired = max(1, round(total_words / max(1, target_words)))
    min_blocks = max(1, math.ceil(total_words / max(1, max_words)))
    max_blocks = max(1, math.ceil(total_words / max(1, min_words)))
    desired = max(min_blocks, min(max_blocks, desired))

    lines = []
    total_chars = 0
    for idx, paragraph in enumerate(paragraphs, start=1):
        digest = _pick_fallback_sentences(paragraph, limit=1)[:220]
        line = f"{idx}. {digest} [words={_count_words(paragraph)}]"
        if total_chars + len(line) > MAX_CHAPTER_MAP_CHARS:
            break
        lines.append(line)
        total_chars += len(line) + 1

    prompt = (
        "Split chapter paragraphs into coherent semantic thought units.\n"
        "Output strict JSON object:\n"
        '{"boundaries": [end_paragraph_index, ...]}\n\n'
        "Rules:\n"
        "- boundaries are 1-based paragraph end indexes\n"
        "- strictly increasing\n"
        f"- expected number of blocks around {desired}\n"
        f"- min words per block about {min_words}, max about {max_words}\n"
        "- include final paragraph index as last boundary\n\n"
        f"Chapter title: {chapter_title or 'Untitled chapter'}\n"
        f"Total paragraphs: {len(paragraphs)}\n"
        f"Total words: {total_words}\n\n"
        f"Paragraph digest:\n{chr(10).join(lines)}"
    )

    llm_result = _ollama_generate(prompt, expect_json=True)
    if llm_result:
        parsed = _safe_json_parse(llm_result)
        if isinstance(parsed, dict) and isinstance(parsed.get("boundaries"), list):
            cleaned: list[int] = []
            seen = set()
            for raw in parsed["boundaries"]:
                try:
                    item = int(raw)
                except (TypeError, ValueError):
                    continue
                if item < 1 or item > len(paragraphs):
                    continue
                if item in seen:
                    continue
                seen.add(item)
                cleaned.append(item)
            cleaned.sort()
            if not cleaned or cleaned[-1] != len(paragraphs):
                cleaned.append(len(paragraphs))
            if cleaned and cleaned[0] <= 0:
                cleaned = [item for item in cleaned if item > 0]
            if len(cleaned) > len(paragraphs):
                cleaned = cleaned[: len(paragraphs)]
            if cleaned:
                return cleaned

    return _fallback_chapter_boundaries(
        paragraphs,
        min_words=min_words,
        target_words=target_words,
        max_words=max_words,
    )


def extract_theme_hierarchy_for_chapter(
    chapter_title: str,
    block_digests: list[dict[str, Any]],
    *,
    desired_themes: int = 3,
) -> list[dict[str, Any]]:
    if not block_digests:
        return []

    safe_lines = []
    total_chars = 0
    for item in block_digests:
        index = int(item.get("index", 0))
        start_paragraph = int(item.get("start_paragraph", 0))
        end_paragraph = int(item.get("end_paragraph", 0))
        summary = _truncate_for_prompt(str(item.get("summary", "")).strip(), 260)
        line = f"[{index}] p{start_paragraph}-{end_paragraph}: {summary}"
        if total_chars + len(line) > 6500:
            break
        total_chars += len(line) + 1
        safe_lines.append(line)

    prompt = (
        "Ты анализируешь главу книги и выделяешь главные смысловые темы.\n"
        "Требование качества: summary каждой темы должен точно описывать, о чем именно этот фрагмент главы.\n"
        "Запрещены общие формулировки вроде 'в тексте рассматривается' без конкретики.\n"
        "Используй только данные из блоков, ничего не выдумывай.\n"
        "Нужен ответ строго в JSON без markdown.\n\n"
        "Верни формат:\n"
        "{\n"
        '  "themes": [\n'
        "    {\n"
        '      "title": "крупная тема",\n'
        '      "summary": "main meaning темы: 2-4 предложения по сути, с конкретикой из блоков",\n'
        '      "start_block": 1,\n'
        '      "end_block": 2,\n'
        '      "subthemes": [\n'
        "        {\n"
        '          "name": "подтема или закон",\n'
        '          "summary": "краткое раскрытие подтемы",\n'
        '          "source_quote": "короткая цитата",\n'
        '          "importance_score": 0.0,\n'
        '          "start_block": 1,\n'
        '          "end_block": 1\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Ожидаемое число тем: {max(1, desired_themes)}.\n"
        "Для каждой темы выделяй 2-4 подтемы.\n"
        "Для каждой темы соблюдай диапазон блоков start_block..end_block.\n"
        "summary темы должен соответствовать только своему диапазону блоков.\n\n"
        f"Глава: {chapter_title or 'Без названия'}\n"
        f"Блоки:\n{chr(10).join(safe_lines)}"
    )
    raw = _ollama_generate(prompt, expect_json=True)
    if not raw:
        return []

    parsed = _safe_json_parse(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("themes"), list):
        return [item for item in parsed["themes"] if isinstance(item, dict)]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []
