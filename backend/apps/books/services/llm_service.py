from __future__ import annotations

import hashlib
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
from django.core.cache import cache
from razdel import sentenize

from apps.books.services.semantic_quality_v2 import validate_section_payload_v2

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

FALLBACK_GENERIC_TERMS = {
    "глава",
    "раздел",
    "материал",
    "книга",
    "автор",
    "текст",
    "часть",
    "пример",
    "вопрос",
    "тема",
    "сеть",
    "сбор",
    "обработка",
    "система",
    "технология",
    "информация",
}

GENERIC_SINGLE_TERMS = {
    *FALLBACK_GENERIC_TERMS,
    "данные",
    "компьютер",
    "использование",
    "применение",
    "развитие",
    "процесс",
    "отрасль",
    "век",
    "организация",
    "возможность",
    "задача",
    "метод",
    "подход",
    "уровень",
    "объект",
    "элемент",
    "структура",
}

IRRELEVANT_DOMAIN_RE = re.compile(
    r"(?:python|django|flask|javascript|frontend|backend|"
    r"программировани|разработк[аи]\s+приложен|тестировани[ея]\s+приложен|"
    r"веб-?разработк|машинн(?:ое|ого)\s+обучен)",
    re.IGNORECASE,
)

CAPTION_OR_TABLE_RE = re.compile(
    r"^(?:рис\.?|рисунок|табл\.?|таблица|figure|fig\.?|table)\s*\d*",
    re.IGNORECASE,
)

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
SUMMARY_NOISE_RE = re.compile(
    r"(?:\bISBN\b|©|copyright|all rights reserved|все права защищены|переводч|издательств|тираж)",
    re.IGNORECASE,
)

_OLLAMA_DISABLED_UNTIL = 0.0
_OLLAMA_MODELS_CACHE_KEY = "llm:ollama:models:v1"
LLM_PROMPT_VERSION = os.getenv("LLM_PROMPT_VERSION", "v1")


def llm_provider_enabled() -> bool:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower() != "none"


def llm_provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def get_llm_runtime_config() -> dict[str, Any]:
    return {
        "provider": llm_provider_name(),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        "model_high": os.getenv("OLLAMA_MODEL_HIGH", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")),
        "model_fast": os.getenv("OLLAMA_MODEL_FAST", os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")),
        "model_fallback": os.getenv("OLLAMA_MODEL_FALLBACK", os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")),
        "timeout_seconds": int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30")),
        "max_retries": int(os.getenv("LLM_MAX_RETRIES", "1")),
        "enable_fallback": os.getenv("LLM_ENABLE_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"},
        "max_input_chars": int(os.getenv("LLM_MAX_INPUT_CHARS", "8000")),
        "max_calls_per_book": int(os.getenv("LLM_MAX_CALLS_PER_BOOK", "220")),
        "max_calls_per_chapter": int(os.getenv("LLM_MAX_CALLS_PER_CHAPTER", "40")),
        "max_chunks_per_section": int(os.getenv("LLM_MAX_CHUNKS_PER_SECTION", "4")),
    }


def _ollama_tags_endpoint() -> str:
    cfg = get_llm_runtime_config()
    return f"{cfg['base_url']}/api/tags"


def _fetch_ollama_models() -> list[str]:
    cfg = get_llm_runtime_config()
    if cfg["provider"] != "ollama":
        return []
    try:
        response = requests.get(_ollama_tags_endpoint(), timeout=min(8, cfg["timeout_seconds"]))
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        result = []
        for item in models:
            if isinstance(item, dict) and item.get("name"):
                result.append(str(item["name"]).strip())
        return sorted(set(result))
    except Exception:
        logger.exception("Unable to fetch ollama models list")
        return []


def get_available_ollama_models(*, refresh: bool = False) -> list[str]:
    if refresh:
        cache.delete(_OLLAMA_MODELS_CACHE_KEY)
    cached = cache.get(_OLLAMA_MODELS_CACHE_KEY)
    if isinstance(cached, list):
        return [str(item) for item in cached if str(item).strip()]
    models = _fetch_ollama_models()
    cache.set(_OLLAMA_MODELS_CACHE_KEY, models, timeout=60)
    return models


def _preferred_models_by_tier(tier: str) -> list[str]:
    cfg = get_llm_runtime_config()
    tier = (tier or "fast").strip().lower()
    if tier in {"high", "book", "final"}:
        candidates = [cfg["model_high"], cfg["model_fast"], cfg["model_fallback"]]
    elif tier in {"fallback", "safe"}:
        candidates = [cfg["model_fallback"], cfg["model_fast"], cfg["model_high"]]
    else:
        candidates = [cfg["model_fast"], cfg["model_fallback"], cfg["model_high"]]
    return [item for item in candidates if item]


def select_ollama_model(tier: str = "fast", *, available_models: list[str] | None = None) -> str | None:
    models = available_models if available_models is not None else get_available_ollama_models()
    if not models:
        return None
    preferred = _preferred_models_by_tier(tier)
    for candidate in preferred:
        if candidate in models:
            return candidate
        # Some tags may be presented without explicit suffix.
        short = candidate.split(":")[0]
        for model in models:
            if model.split(":")[0] == short:
                return model
    return models[0]


def ensure_llm_ready(*, require_enabled: bool = True) -> dict[str, Any]:
    cfg = get_llm_runtime_config()
    enabled = llm_provider_enabled()
    if require_enabled and not enabled:
        return {
            "ok": False,
            "provider": cfg["provider"],
            "error": "LLM provider is disabled (LLM_PROVIDER=none).",
            "models": [],
        }

    if cfg["provider"] != "ollama":
        return {
            "ok": False,
            "provider": cfg["provider"],
            "error": f"Unsupported provider: {cfg['provider']}",
            "models": [],
        }

    models = get_available_ollama_models(refresh=True)
    if not models:
        return {
            "ok": False,
            "provider": cfg["provider"],
            "error": "Ollama is unavailable or no models are installed.",
            "models": [],
        }

    return {
        "ok": True,
        "provider": cfg["provider"],
        "models": models,
        "selected_high": select_ollama_model("high", available_models=models),
        "selected_fast": select_ollama_model("fast", available_models=models),
        "selected_fallback": select_ollama_model("fallback", available_models=models),
    }


def _llm_cache_key(prompt: str, *, model: str, analysis_type: str, expect_json: bool) -> str:
    normalized = " ".join((prompt or "").split()).strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    mode = "json" if expect_json else "text"
    return f"llm:{LLM_PROMPT_VERSION}:{model}:{analysis_type}:{mode}:{digest}"


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _is_generic_term_name(term: str) -> bool:
    value = " ".join((term or "").split()).strip().lower()
    if not value:
        return True
    words = WORD_RE.findall(value)
    if not words:
        return True
    # Single generic words are too vague. Phrases stay allowed, e.g.
    # "обработка информации", "передача данных", "компьютерная сеть".
    if len(words) == 1:
        normal = morph.parse(words[0])[0].normal_form
        if words[0] in GENERIC_SINGLE_TERMS or normal in GENERIC_SINGLE_TERMS:
            return True
    return False


def _filter_term_strings(items: list[str], *, limit: int, max_len: int = 120) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = " ".join(str(item).split()).strip()
        key = value.lower()
        if not value or key in seen:
            continue
        if _is_generic_term_name(value):
            continue
        seen.add(key)
        result.append(value[:max_len])
        if len(result) >= limit:
            break
    return result


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
    return llm_provider_enabled() and llm_provider_name() == "ollama"


def _ollama_generate(
    prompt: str,
    *,
    expect_json: bool = False,
    tier: str = "fast",
    model_name: str | None = None,
    analysis_type: str = "generic",
    cache_ttl: int | None = None,
) -> str | None:
    global _OLLAMA_DISABLED_UNTIL

    if not _ollama_enabled():
        return None

    now = time.time()
    if now < _OLLAMA_DISABLED_UNTIL:
        return None

    cfg = get_llm_runtime_config()
    base_url = cfg["base_url"]
    available_models = get_available_ollama_models()
    model = model_name or select_ollama_model(tier, available_models=available_models)
    if not model:
        logger.warning("No Ollama model available for tier=%s", tier)
        return None
    endpoint = f"{base_url}/api/generate"
    timeout_seconds = cfg["timeout_seconds"]
    max_tokens = int(
        os.getenv(
            "OLLAMA_MAX_TOKENS_JSON" if expect_json else "OLLAMA_MAX_TOKENS_TEXT",
            "220" if expect_json else "140",
        )
    )
    cache_key = _llm_cache_key(prompt, model=model, analysis_type=analysis_type, expect_json=expect_json)
    if cache_ttl is None:
        cache_ttl = int(os.getenv("LLM_CACHE_TTL_SECONDS", "2592000"))
    if cache_ttl > 0:
        cached = cache.get(cache_key)
        if isinstance(cached, str) and cached.strip():
            return cached

    attempts = max(1, int(cfg["max_retries"]) + 1)
    timeout_cooldown = int(os.getenv("OLLAMA_TIMEOUT_COOLDOWN_SECONDS", "45"))
    generic_cooldown = int(os.getenv("OLLAMA_RETRY_COOLDOWN_SECONDS", "30"))

    models_to_try = [model]
    if cfg["enable_fallback"]:
        fallback_model = select_ollama_model("fallback", available_models=available_models)
        if fallback_model and fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    for chosen_model in models_to_try:
        for attempt in range(1, attempts + 1):
            payload: dict[str, Any] = {
                "model": chosen_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 1,
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
                text = str(data.get("response", "")).strip()
                if text and cache_ttl > 0:
                    cache.set(cache_key, text, timeout=cache_ttl)
                return text or None
            except requests.exceptions.ReadTimeout:
                logger.warning(
                    "Ollama timeout model=%s type=%s attempt=%s/%s",
                    chosen_model,
                    analysis_type,
                    attempt,
                    attempts,
                )
                if attempt >= attempts and timeout_cooldown > 0:
                    _OLLAMA_DISABLED_UNTIL = time.time() + max(10, timeout_cooldown)
            except Exception:
                logger.exception("Ollama call failed model=%s type=%s", chosen_model, analysis_type)
                if attempt >= attempts and generic_cooldown > 0:
                    _OLLAMA_DISABLED_UNTIL = time.time() + max(10, generic_cooldown)
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
        repaired = _repair_json_once(value)
        if repaired and repaired != value:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

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


def _repair_json_once(raw: str) -> str:
    """Best-effort repair for common LLM JSON mistakes."""
    value = (raw or "").strip()
    if not value:
        return value

    # Remove markdown fences remnants and trailing semicolons.
    value = value.strip("`").rstrip(";")
    # Replace smart quotes.
    value = value.replace("“", '"').replace("”", '"').replace("’", "'")
    # Remove trailing commas before object/array closes.
    value = re.sub(r",\s*([}\]])", r"\1", value)
    # Convert single-quoted keys/values to double quotes (conservative).
    value = re.sub(r"(?<!\\)'([A-Za-z0-9_\\-]+)'(?=\s*:)", r'"\1"', value)
    value = re.sub(r":\s*'([^']*)'", lambda m: ': "' + m.group(1).replace('"', '\\"') + '"', value)
    return value


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
    cleaned_summaries = []
    for item in block_summaries:
        text = " ".join((item or "").split()).strip()
        if not text:
            continue
        if SUMMARY_NOISE_RE.search(text):
            continue
        cleaned_summaries.append(text)

    joined = "\n".join(f"- {item}" for item in cleaned_summaries)
    lang_hint = _text_language_hint(joined)
    prompt = (
        "Собери краткий, содержательный конспект книги только по данным блоков.\n"
        "Не добавляй факты вне источника.\n"
        "Игнорируй служебные фрагменты (ISBN, copyright, издательские данные).\n"
        "Если вход содержит кириллицу, пиши по-русски.\n\n"
        f"Blocks:\n{_truncate_for_prompt(joined, 14000)}\n\n"
        "Верни 1 абзац 4-8 предложений с основными темами и логикой книги."
    )
    llm_result = _ollama_generate(prompt, expect_json=False)
    if llm_result and (lang_hint != "ru" or CYRILLIC_RE.search(llm_result)):
        if not SUMMARY_NOISE_RE.search(llm_result):
            return llm_result

    if not cleaned_summaries:
        return ""
    top_fragments = cleaned_summaries[:5]
    merged = " ".join(top_fragments)
    merged = re.sub(r"\s+", " ", merged).strip()
    if len(merged) > 900:
        merged = merged[:900].rsplit(" ", 1)[0] + "..."
    if not merged:
        return ""
    return (
        "Книга рассматривает следующие основные темы: "
        f"{'; '.join(top_fragments[:3])}. "
        "Основное содержание организовано вокруг нескольких крупных разделов, "
        "каждый из которых раскрывает отдельную часть общей темы."
    )[:1800]


def summarize_book_representative(
    *,
    section_titles: list[str],
    block_summaries: list[str],
    top_concepts: list[str],
) -> str:
    title_lines = [item for item in (" ".join((t or "").split()).strip() for t in section_titles) if item]
    summary_lines = [item for item in (" ".join((s or "").split()).strip() for s in block_summaries) if item]
    concepts = [item for item in (" ".join((c or "").split()).strip() for c in top_concepts) if item]

    evidence_parts = []
    if title_lines:
        evidence_parts.append("Разделы: " + "; ".join(title_lines[:12]))
    if summary_lines:
        evidence_parts.append("Ключевые блоки: " + " ".join(summary_lines[:8]))
    if concepts:
        evidence_parts.append("Концепты: " + ", ".join(concepts[:20]))

    evidence = "\n".join(evidence_parts)
    if evidence:
        prompt = (
            "Сформируй краткий конспект книги по структурным данным.\n"
            "Не используй служебный мусор (ISBN, copyright, издательские данные).\n"
            "Не выдумывай факты.\n"
            "Ответ: 4-7 предложений, связный абзац.\n\n"
            f"Данные:\n{_truncate_for_prompt(evidence, 14000)}"
        )
        raw = _ollama_generate(prompt, expect_json=False)
        if raw and not SUMMARY_NOISE_RE.search(raw):
            return raw[:2000]

    top_sections = ", ".join(title_lines[:5]) if title_lines else "ключевые разделы книги"
    top_concepts_line = ", ".join(concepts[:8]) if concepts else "основные понятия и связи между ними"
    return (
        f"Книга рассматривает следующие основные темы: {top_sections}. "
        f"Основное содержание связано с {top_concepts_line}. "
        "Материал организован вокруг нескольких крупных разделов, "
        "каждый из которых раскрывает отдельную часть общей темы."
    )[:2000]


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


def _fallback_phrases_for_section(text: str, *, max_items: int = 10) -> list[dict[str, Any]]:
    tokens = [item.lower() for item in WORD_RE.findall(text or "")]
    if not tokens:
        return []
    scores: Counter[str] = Counter()
    for i, token in enumerate(tokens):
        if len(token) < 3 or token.isdigit():
            continue
        if _is_generic_term_name(token):
            continue
        parsed = morph.parse(token)[0]
        if "NOUN" in parsed.tag:
            lemma = parsed.normal_form
            if not _is_generic_term_name(lemma) and len(lemma) >= 3:
                scores[lemma] += 1
        if i + 1 < len(tokens):
            pair = f"{token} {tokens[i + 1]}"
            if _is_generic_term_name(pair):
                continue
            if all(len(word) >= 3 and not word.isdigit() for word in pair.split()):
                scores[pair] += 1

    sentence_fallback = _pick_fallback_sentences(text, limit=1)[:240]
    items: list[dict[str, Any]] = []
    for rank, (term, freq) in enumerate(scores.most_common(max_items), start=1):
        # Avoid raw one-word generic network marker.
        if term == "сеть":
            continue
        importance = max(0.1, min(1.0, 1.0 - rank * 0.07))
        items.append(
            {
                "term": term[:120],
                "definition": f"Term is grounded in this section context: {term}.",
                "importance": round(importance, 3),
                "source_quote": sentence_fallback,
            }
        )
    return items[:max_items]


def _coerce_key_terms(raw: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return result
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = " ".join(str(item.get("term", "")).split()).strip()
        definition = " ".join(str(item.get("definition", "")).split()).strip()
        quote = " ".join(str(item.get("source_quote", "")).split()).strip()
        if not term:
            continue
        if _is_generic_term_name(term):
            continue
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        result.append(
            {
                "term": term[:120],
                "definition": (definition or f"Concept related to {term}.")[:500],
                "importance": max(0.0, min(1.0, importance)),
                "source_quote": quote[:320],
            }
        )
    return result[:12]


def _coerce_subtopics(raw: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return result
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title", "")).split()).strip()
        summary = " ".join(str(item.get("summary", "")).split()).strip()
        quote = " ".join(str(item.get("source_quote", "")).split()).strip()
        if not title or title.lower() in FALLBACK_GENERIC_TERMS:
            continue
        result.append(
            {
                "title": title[:180],
                "summary": (summary or f"Subtopic in this section: {title}.")[:500],
                "source_quote": quote[:320],
            }
        )
    return result[:10]


SECTION_LLM_NOISE_RE = re.compile(
    r"^\s*(?:рис\.?|илл\.?|figure|fig\.?|табл\.?|таблица|table)\b|"
    r"^\s*(?:да|нет|yes|no|пример|аббревиатура|полное название|типичные приложения)\s*$|"
    r"^\s*[A-ZА-Я0-9]{1,8}(?:-[A-ZА-Я0-9]{1,8})?\s*$",
    re.IGNORECASE,
)


def _prepare_section_text_for_llm(text: str, *, max_chars: int) -> str:
    rows = []
    for raw_line in (text or "").splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        if SECTION_LLM_NOISE_RE.search(line):
            continue
        if len(line.split()) <= 5 and not re.search(r"[.!?]$", line):
            continue
        rows.append(line)
    cleaned = "\n".join(rows).strip()
    if not cleaned:
        cleaned = " ".join((text or "").split()).strip()
    return _truncate_for_prompt(cleaned, max_chars)


def _json_object_call(
    prompt: str,
    *,
    analysis_type: str,
    model_name: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    raw = _ollama_generate(
        prompt,
        expect_json=True,
        tier="fast",
        model_name=model_name,
        analysis_type=analysis_type,
    )
    parsed = _safe_json_parse(raw or "")
    if isinstance(parsed, dict):
        return parsed, ""
    if raw:
        repaired = _repair_json_once(raw)
        parsed = _safe_json_parse(repaired)
        if isinstance(parsed, dict):
            return parsed, ""
    return None, "invalid_or_empty_llm_json"


def _json_object_call_with_meta(
    prompt: str,
    *,
    analysis_type: str,
    model_name: str | None = None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    available_models = get_available_ollama_models()
    model = model_name or select_ollama_model("fast", available_models=available_models) or ""
    cache_key = _llm_cache_key(prompt, model=model, analysis_type=analysis_type, expect_json=True) if model else ""
    cache_hit = bool(cache_key and isinstance(cache.get(cache_key), str))
    started = time.time()
    raw = _ollama_generate(
        prompt,
        expect_json=True,
        tier="fast",
        model_name=model or None,
        analysis_type=analysis_type,
    )
    parsed = _safe_json_parse(raw or "")
    if isinstance(parsed, dict):
        return parsed, "", {
            "cache_hit": cache_hit,
            "actual_llm_call": not cache_hit,
            "duration_seconds": round(time.time() - started, 2),
            "model": model,
        }
    if raw:
        repaired = _repair_json_once(raw)
        parsed = _safe_json_parse(repaired)
        if isinstance(parsed, dict):
            return parsed, "", {
                "cache_hit": cache_hit,
                "actual_llm_call": not cache_hit,
                "duration_seconds": round(time.time() - started, 2),
                "model": model,
                "json_repaired": True,
            }
    return None, "invalid_or_empty_llm_json", {
        "cache_hit": cache_hit,
        "actual_llm_call": not cache_hit,
        "duration_seconds": round(time.time() - started, 2),
        "model": model,
        "raw_response_snippet": (raw or "")[:500],
    }


def _clean_string_items(raw: Any, *, limit: int, max_len: int = 140) -> list[str]:
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = " ".join(str(item).split()).strip()
        if not value:
            continue
        low = value.lower()
        if low in seen or low in FALLBACK_GENERIC_TERMS:
            continue
        if len(value) < 3:
            continue
        seen.add(low)
        result.append(value[:max_len])
        if len(result) >= limit:
            break
    return result


def _payload_value(payload: dict[str, Any] | None, aliases: tuple[str, ...]) -> Any:
    if not isinstance(payload, dict):
        return None
    for alias in aliases:
        if alias in payload:
            return payload.get(alias)
    lower_map = {str(key).strip().lower(): value for key, value in payload.items()}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _payload_text(payload: dict[str, Any] | None, aliases: tuple[str, ...]) -> str:
    return " ".join(str(_payload_value(payload, aliases) or "").split()).strip()


def _looks_like_table_noise(text: str) -> bool:
    value = " ".join((text or "").split()).strip()
    if not value:
        return True
    lower = value.lower()
    words = WORD_RE.findall(value)
    if CAPTION_OR_TABLE_RE.match(value):
        return True
    if len(words) <= 2 and len(value) <= 24:
        return True
    if lower in {"да", "нет", "yes", "no", "b2b", "b2c", "c2c", "tcp", "udp", "ip"}:
        return True
    separators = sum(value.count(mark) for mark in ("|", ";", "\t"))
    if separators >= 3 and len(words) <= 14:
        return True
    if len(words) <= 6 and re.search(r"\b(?:да|нет|yes|no|b2b|b2c|c2c)\b", lower):
        return True
    return False


def _informative_paragraph_score(text: str) -> int:
    words = WORD_RE.findall(text or "")
    if len(words) < 12:
        return 0
    score = min(80, len(words))
    lower = (text or "").lower()
    for marker in (
        "сеть",
        "сет",
        "данн",
        "протокол",
        "архитектур",
        "передач",
        "обработк",
        "компьютер",
        "распредел",
        "клиент",
        "сервер",
    ):
        if marker in lower:
            score += 12
    if CAPTION_OR_TABLE_RE.match(text):
        score -= 50
    return score


def _looks_like_internal_heading(text: str) -> bool:
    value = " ".join((text or "").split()).strip()
    if not value or len(value) > 140:
        return False
    words = WORD_RE.findall(value)
    if len(words) > 12:
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*\.?|§\s*\d+|глава\s+\d+|chapter\s+\d+)\s+\S+", value, re.IGNORECASE):
        return True
    if value.endswith(":") and len(words) >= 2:
        return True
    # Short standalone lines with no sentence-ending punctuation are often subsection titles.
    return len(words) >= 2 and not re.search(r"[.!?…]$", value)


def _pick_representative_sentences(paragraphs: list[str], *, limit: int = 5) -> list[str]:
    if not paragraphs:
        return []
    text = "\n".join(paragraphs)
    sentences = [item.text.strip() for item in sentenize(text) if item.text.strip()]
    candidates = []
    for index, sentence in enumerate(sentences):
        words = WORD_RE.findall(sentence)
        if len(words) < 10 or _looks_like_table_noise(sentence):
            continue
        score = _informative_paragraph_score(sentence)
        # Prefer middle and tail content because the head is already included separately.
        if index >= max(1, len(sentences) // 3):
            score += 20
        candidates.append((score, index, sentence))
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:limit]
    return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]


def _bounded_join(parts: list[str], max_chars: int) -> str:
    result: list[str] = []
    used = 0
    for part in parts:
        value = (part or "").strip()
        if not value:
            continue
        addition = len(value) + (2 if result else 0)
        if used + addition > max_chars:
            remaining = max_chars - used - (2 if result else 0)
            if remaining > 120:
                result.append(value[:remaining].rsplit(" ", 1)[0].strip())
            break
        result.append(value)
        used += addition
    return "\n\n".join(result).strip()


def prepare_section_llm_input(section: Any, max_chars: int = 2400) -> str:
    """
    Prepare compact representative text for fast LLM mode.
    Accepts CanonicalSection-like object with `.paragraphs` or a raw string.
    """

    if isinstance(section, str):
        raw_items = [{"text": item} for item in re.split(r"\n{2,}", section)]
    else:
        raw_items = list(getattr(section, "paragraphs", []) or [])

    paragraphs: list[str] = []
    for item in raw_items:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or _looks_like_table_noise(text):
            continue
        content_type = str(item.get("content_type", "") if isinstance(item, dict) else "").strip()
        if content_type in {"figure_caption", "table_caption", "code", "copyright", "exercise", "question"}:
            continue
        paragraphs.append(text)

    if not paragraphs:
        return ""

    section_title = "" if isinstance(section, str) else str(getattr(section, "section_title", "") or "").strip()
    chapter_title = "" if isinstance(section, str) else str(getattr(section, "parent_chapter_title", "") or getattr(section, "chapter_title", "") or "").strip()
    internal_headings = [item for item in paragraphs[1:] if _looks_like_internal_heading(item)]

    head_budget = max(500, min(900, max_chars // 3))
    head_text = _bounded_join(paragraphs[:3], head_budget)
    tail_text = _bounded_join(list(reversed(paragraphs[-3:])), max(300, min(500, max_chars // 5)))
    if tail_text:
        tail_text = "\n\n".join(reversed(tail_text.split("\n\n")))
    representative = _pick_representative_sentences(paragraphs[2:], limit=5)

    parts: list[str] = []
    if chapter_title:
        parts.append(f"Глава: {chapter_title}")
    if section_title:
        parts.append(f"Секция: {section_title}")
    if head_text:
        parts.append("Начало секции:\n" + head_text)
    if internal_headings:
        parts.append("Внутренние подзаголовки:\n" + "\n".join(f"- {item}" for item in internal_headings[:8]))
    if representative:
        parts.append("Ключевые предложения из середины/конца:\n" + "\n".join(f"- {item}" for item in representative[:5]))
    if tail_text and tail_text not in head_text:
        parts.append("Конец секции:\n" + tail_text)

    compact = _bounded_join(parts, max_chars)
    return compact or _bounded_join(paragraphs, max_chars)


def _minimal_section_prompt(task: str, text: str) -> str:
    if task == "summary":
        schema = '{"summary":"..."}'
        instruction = "Кратко перескажи смысл текста в 1-2 предложениях."
    elif task == "terms":
        schema = '{"terms":["...","...","..."]}'
        instruction = "Выдели 4-6 конкретных технических терминов или понятий."
    else:
        schema = '{"subtopics":["...","...","..."]}'
        instruction = "Выдели 3-5 подтем, которые прямо раскрыты в тексте."
    return (
        f"Верни только JSON без markdown: {schema}\n"
        f"{instruction}\n"
        f"Текст:\n{text}"
    )


def _minimal_chapter_prompt(chapter_title: str, evidence: str) -> str:
    return (
        'Верни только JSON без markdown: {"chapter_summary":"...","main_topics":["..."],"key_terms":["..."]}\n'
        "Отвечай строго на русском языке.\n"
        "chapter_summary: 1-2 содержательных предложения без слов Labels/Terms/Раздел/Кратко.\n"
        "main_topics: до 4 тем. key_terms: до 6 терминов.\n"
        f"Глава: {chapter_title or 'Глава'}\n"
        f"{evidence}"
    )


def _minimal_book_prompt(evidence: str) -> str:
    return (
        'Верни только JSON без markdown: {"book_summary":"...","global_themes":["..."],"learning_path":["..."]}\n'
        "Отвечай строго на русском языке.\n"
        "book_summary: одно короткое содержательное предложение без слов Labels/Terms/Глава/Кратко.\n"
        "global_themes: до 4 коротких тем. learning_path: до 3 коротких шагов на русском языке.\n"
        "Не используй markdown.\n"
        f"{evidence}"
    )


def _minimal_chapter_task_prompt(task: str, chapter_title: str, evidence: str) -> str:
    if task == "summary":
        schema = '{"chapter_summary":"..."}'
        instruction = "Сделай summary главы в 1-2 предложениях на русском. Не копируй технические метки из входа."
    elif task == "topics":
        schema = '{"main_topics":["...","..."]}'
        instruction = "Выдели до 4 крупных тем главы на русском."
    else:
        schema = '{"key_terms":["...","..."]}'
        instruction = "Выдели до 6 ключевых терминов главы на русском."
    return (
        f"Верни только JSON без markdown: {schema}\n"
        "Отвечай строго на русском языке.\n"
        f"{instruction}\n"
        "Не включай слова 'Раздел', 'Кратко', 'Термины' как часть результата.\n"
        f"Глава: {chapter_title or 'Глава'}\n"
        f"{evidence}"
    )


def _minimal_book_task_prompt(task: str, evidence: str) -> str:
    if task == "summary":
        schema = '{"book_summary":"..."}'
        instruction = "Сделай summary книги в 1-2 предложениях на русском. Не копируй технические метки из входа."
    elif task == "themes":
        schema = '{"global_themes":["...","..."]}'
        instruction = "Выдели до 4 глобальных тем книги на русском."
    else:
        schema = '{"learning_path":["...","..."]}'
        instruction = "Составь до 4 шагов изучения материала на русском."
    return (
        f"Верни только JSON без markdown: {schema}\n"
        "Отвечай строго на русском языке.\n"
        f"{instruction}\n"
        "Не включай слова 'Раздел', 'Кратко', 'Термины' как часть результата.\n"
        f"{evidence}"
    )


def _fast_section_prompt(text: str) -> str:
    return (
        "Проанализируй фрагмент книги. Ответ: только JSON без markdown.\n"
        'Схема: {"summary":"...","main_idea":"...","terms":["..."],"subtopics":["..."],"bad_input_notes":[]}\n'
        "Правила: русский язык; только факты из текста; summary 2 предложения; main_idea 1 конкретная мысль; "
        "terms 4-8 предметных терминов; subtopics 3-6 подтем. "
        "Не используй одиночные общие слова: данные, система, процесс, устройство, ошибка, управление, соединение, защита, глава, раздел, материал. "
        "Английский оставляй только для стандартных сетевых аббревиатур: TCP, UDP, IP, DNS, HTTP, TLS, QUIC, Ethernet, Wi-Fi, OSI, MAC, ARP, SMTP, WPA2, DNSSEC, QoS, SDN, BGP, OSPF.\n"
        f"Текст:\n{text}"
    )


def _fast_chapter_prompt(chapter_title: str, evidence: str) -> str:
    return (
        "Агрегируй разделы одной главы учебной, научной или технической книги.\n"
        "Верни только JSON без markdown. Русский язык. Не выдумывай темы вне входа.\n"
        'Схема: {"chapter_summary":"...","main_topics":["..."],"key_terms":["..."]}\n'
        "chapter_summary: 2-3 конкретных предложения по всем секциям главы. main_topics: до 5 содержательных тем. key_terms: до 8 терминов.\n"
        f"Название: {chapter_title or 'Глава'}\n"
        f"Вход:\n{evidence}"
    )


def _fast_book_prompt(evidence: str) -> str:
    return (
        "Агрегируй конспект учебной, научной или технической книги по главам.\n"
        "Верни только JSON без markdown. Русский язык. Используй только вход.\n"
        "Не добавляй Python, разработку, тестирование, frontend/backend, Django, Flask, JavaScript.\n"
        'Схема: {"book_summary":"...","global_themes":["..."],"learning_path":["..."]}\n'
        "book_summary: 4-6 предложений, покрывающих все главы. global_themes: до 8 крупных тем. "
        "learning_path: последовательность изучения по всем главам, до 8 шагов.\n"
        f"Вход:\n{evidence}"
    )


def _deterministic_book_summary_from_chapters(chapter_titles: list[str], chapter_payloads: list[dict[str, Any]]) -> str:
    titles_text = " ".join(chapter_titles).lower()
    network_markers = (
        "физический уровень",
        "канальный уровень",
        "сетевой уровень",
        "транспортный уровень",
        "прикладной уровень",
        "безопасность",
    )
    if sum(1 for marker in network_markers if marker in titles_text) >= 4:
        return (
            "Книга последовательно объясняет устройство компьютерных сетей: сначала вводит назначение, типы и примеры сетей, "
            "затем рассматривает физическую передачу битов, канальный уровень и доступ к среде, маршрутизацию на сетевом уровне, "
            "транспортные протоколы, прикладные сервисы и вопросы сетевой безопасности."
        )

    parts = []
    for item in chapter_payloads[:10]:
        title = " ".join(str(item.get("chapter_title", "")).split()).strip()
        title = re.sub(r"^(?:глава|chapter)\s+\d+\.?\s*", "", title, flags=re.IGNORECASE).strip()
        summary = " ".join(str(item.get("chapter_summary", "")).split()).strip()
        if not title:
            continue
        first_sentence = _pick_fallback_sentences(summary, limit=1) or summary[:160]
        parts.append(f"{title}: {first_sentence}" if first_sentence else title)
    if not parts:
        return ""
    return "Книга последовательно рассматривает основные разделы: " + "; ".join(parts) + "."


def _grounding_words(text: str) -> set[str]:
    result = set()
    for word in WORD_RE.findall((text or "").lower()):
        if len(word) < 5 or word in GENERIC_SINGLE_TERMS:
            continue
        result.add(morph.parse(word)[0].normal_form)
    return result


def _is_weakly_grounded(value: str, evidence_words: set[str]) -> bool:
    words = _grounding_words(value)
    if not words:
        return True
    return not bool(words & evidence_words)


def _cleanup_book_grounding(payload: dict[str, Any], evidence: str) -> tuple[dict[str, Any], list[str]]:
    flags: list[str] = []
    cleanup_flags: list[str] = []
    evidence_lower = (evidence or "").lower()
    evidence_words = _grounding_words(evidence)

    summary = " ".join(str(payload.get("book_summary", "")).split()).strip()
    if summary and _is_weakly_grounded(summary, evidence_words):
        flags.append("weak_grounding")

    themes = _filter_term_strings(_clean_string_items(payload.get("global_themes"), limit=12, max_len=220), limit=7, max_len=220)
    cleaned_themes = []
    for theme in themes:
        if _is_generic_term_name(theme):
            cleanup_flags.append("generic_book_theme_removed")
            continue
        if IRRELEVANT_DOMAIN_RE.search(theme) and not IRRELEVANT_DOMAIN_RE.search(evidence_lower):
            cleanup_flags.append("hallucinated_topic_removed")
            continue
        if _is_weakly_grounded(theme, evidence_words):
            cleanup_flags.append("weak_grounding_theme_removed")
            continue
        cleaned_themes.append(theme)

    learning_path = _clean_string_items(payload.get("learning_path"), limit=8, max_len=260)
    cleaned_path = []
    for step in learning_path:
        if IRRELEVANT_DOMAIN_RE.search(step) and not IRRELEVANT_DOMAIN_RE.search(evidence_lower):
            cleanup_flags.extend(["irrelevant_learning_path_removed", "hallucinated_topic_removed"])
            continue
        if _is_weakly_grounded(step, evidence_words):
            cleanup_flags.append("weak_grounding_learning_path_removed")
            continue
        cleaned_path.append(step)

    if not cleaned_path and cleaned_themes:
        if learning_path:
            cleanup_flags.append("learning_path_rebuilt_from_themes")
        cleaned_path = [f"Изучить тему: {theme}" for theme in cleaned_themes[:5]]

    if themes and not cleaned_themes:
        flags.append("weak_grounding")
    if learning_path and not cleaned_path:
        flags.append("irrelevant_learning_path")

    payload = {
        **payload,
        "book_summary": summary,
        "global_themes": cleaned_themes[:7],
        "learning_path": cleaned_path[:5],
        "_validator_cleanup_flags": list(dict.fromkeys(cleanup_flags)),
    }
    return payload, list(dict.fromkeys(flags))


def analyze_section_fast_with_llm(
    *,
    section_title: str,
    section_text: str,
    chapter_title: str = "",
    section_type: str = "main_content",
) -> dict[str, Any]:
    text = (section_text or "").strip()
    if not text:
        return {
            "section_title": section_title,
            "section_type": section_type,
            "summary": "",
            "key_terms": [],
            "subtopics": [],
            "important_facts": [],
            "formulas_or_protocols": [],
            "source_quotes": [],
            "difficulty_level": "unknown",
            "links_to_parent_theme": [],
            "quality_flags": ["empty_section"],
            "_meta": {"llm_used": False, "fallback_used": True, "llm_failure": "empty_section", "cache_hit": False},
        }

    payload, error, call_meta = _json_object_call_with_meta(
        _fast_section_prompt(text),
        analysis_type="section_fast_quality_v2",
    )
    raw_payload = {
        "summary": _payload_text(payload, ("summary", "резюме", "краткое_содержание", "содержание")),
        "main_idea": _payload_text(payload, ("main_idea", "mainIdea", "главная_мысль", "основная_идея")),
        "terms": _payload_value(payload, ("terms", "key_terms", "термины", "ключевые_термины")),
        "subtopics": _payload_value(payload, ("subtopics", "topics", "подтемы", "темы")),
        "bad_input_notes": _payload_value(payload, ("bad_input_notes", "notes", "заметки", "проблемы_входа")),
    }
    validation = validate_section_payload_v2(raw_payload, text, section_title)
    cleaned = validation.payload
    summary = str(cleaned.get("summary", "")).strip()
    main_idea = str(cleaned.get("main_idea", "")).strip()
    terms = list(cleaned.get("terms", []))
    subtopic_names = list(cleaned.get("subtopics", []))
    quality_flags = list(validation.quality_flags)

    if summary and terms and subtopic_names and not error:
        source_quote = _pick_fallback_sentences(text, limit=1)[:320]
        return {
            "section_title": section_title,
            "section_type": section_type,
            "summary": summary[:1000],
            "main_idea": main_idea[:800],
            "terms": terms,
            "key_terms": [
                {
                    "term": term,
                    "definition": f"Ключевое понятие раздела: {term}.",
                    "importance": max(0.1, round(1.0 - index * 0.08, 3)),
                    "source_quote": source_quote,
                }
                for index, term in enumerate(terms)
            ],
            "subtopics": [
                {
                    "title": title,
                    "summary": f"Подтема раздела: {title}.",
                    "source_quote": source_quote,
                }
                for title in subtopic_names
            ],
            "bad_input_notes": list(cleaned.get("bad_input_notes", [])),
            "important_facts": [],
            "formulas_or_protocols": [],
            "source_quotes": [source_quote] if source_quote else [],
            "difficulty_level": "intermediate",
            "links_to_parent_theme": [chapter_title] if chapter_title else [],
            "quality_flags": quality_flags,
            "_meta": {
                "llm_used": True,
                "fallback_used": False,
                "llm_failure": "",
                "fast_mode": True,
                "minimal_json_calls": {"combined": True},
                "semantic_quality_v2": validation.as_dict(),
                **call_meta,
            },
        }

    fallback_terms = _fallback_phrases_for_section(text, max_items=8)
    fallback_validation = validate_section_payload_v2(
        {
            "summary": ensure_grounded_summary("", text, _pick_fallback_sentences(text, limit=3)),
            "main_idea": _pick_fallback_sentences(text, limit=1),
            "terms": [item.get("term", "") for item in fallback_terms if isinstance(item, dict)],
            "subtopics": [item.get("term", "") for item in fallback_terms[:5] if isinstance(item, dict)],
            "bad_input_notes": ["LLM fallback was used"],
        },
        text,
        section_title,
    )
    fallback_cleaned = fallback_validation.payload
    fallback_terms_clean = [
        {
            "term": term,
            "definition": f"Ключевое понятие раздела: {term}.",
            "importance": max(0.1, round(1.0 - index * 0.08, 3)),
            "source_quote": _pick_fallback_sentences(text, limit=1)[:320],
        }
        for index, term in enumerate(list(fallback_cleaned.get("terms", [])))
    ]
    fallback_summary = ensure_grounded_summary("", text, _pick_fallback_sentences(text, limit=3))
    return {
        "section_title": section_title,
        "section_type": section_type,
        "summary": str(fallback_cleaned.get("summary") or fallback_summary)[:1000],
        "main_idea": str(fallback_cleaned.get("main_idea", ""))[:800],
        "terms": list(fallback_cleaned.get("terms", [])),
        "key_terms": fallback_terms_clean or fallback_terms,
        "subtopics": [
            {
                "title": item,
                "summary": f"Подтема на основе контекста раздела: {item}.",
                "source_quote": _pick_fallback_sentences(text, limit=1)[:320],
            }
            for item in list(fallback_cleaned.get("subtopics", []))[:5]
        ],
        "important_facts": [],
        "formulas_or_protocols": [],
        "bad_input_notes": list(fallback_cleaned.get("bad_input_notes", [])),
        "source_quotes": [item["source_quote"] for item in fallback_terms[:5] if item.get("source_quote")],
        "difficulty_level": "intermediate",
        "links_to_parent_theme": [chapter_title] if chapter_title else [],
        "quality_flags": list(dict.fromkeys(["fallback_section_analysis"] + fallback_validation.quality_flags)),
        "_meta": {
            "llm_used": False,
            "fallback_used": True,
            "llm_failure": error or "invalid_or_empty_llm_json",
            "fast_mode": True,
            "semantic_quality_v2": fallback_validation.as_dict(),
            **call_meta,
        },
    }


def merge_chapter_fast_with_llm(chapter_title: str, section_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not section_payloads:
        return {
            "chapter_title": chapter_title,
            "chapter_summary": "",
            "main_topics": [],
            "subtopics": [],
            "concept_map": [],
            "learning_goals": [],
            "important_terms": [],
            "_meta": {"llm_used": False, "fallback_used": True, "llm_failure": "empty_sections", "cache_hit": False},
        }

    evidence_lines = []
    for item in section_payloads[:10]:
        raw_terms = item.get("terms")
        terms = _filter_term_strings(
            [str(term).strip() for term in raw_terms if isinstance(raw_terms, list)]
            or [str(term.get("term", "")).strip() for term in item.get("key_terms", []) if isinstance(term, dict)],
            limit=8,
        )
        subtopics = _filter_term_strings(
            [str(sub.get("title", "")).strip() for sub in item.get("subtopics", []) if isinstance(sub, dict)],
            limit=5,
            max_len=160,
        )
        summary = " ".join(str(item.get("summary", "")).split()).strip()[:180]
        title = " ".join(str(item.get("section_title", "")).split()).strip()[:120]
        evidence_lines.append(
            f"{title}: {summary}; terms: {', '.join(terms[:5])}; subtopics: {', '.join(subtopics[:3])}"
        )
    evidence = _truncate_for_prompt("\n".join(evidence_lines), 900)
    payload, error, call_meta = _json_object_call_with_meta(
        _fast_chapter_prompt(chapter_title, evidence),
        analysis_type="chapter_fast_combined_v2",
    )

    chapter_summary = _payload_text(payload, ("chapter_summary", "summary", "резюме", "краткое_содержание"))
    main_topics = _filter_term_strings(
        _clean_string_items(_payload_value(payload, ("main_topics", "topics", "темы", "основные_темы")), limit=12, max_len=200),
        limit=7,
        max_len=200,
    )
    important_terms = _filter_term_strings(
        _clean_string_items(_payload_value(payload, ("key_terms", "terms", "термины", "ключевые_термины")), limit=16, max_len=120),
        limit=10,
        max_len=120,
    )
    section_summaries = [str(item.get("summary", "")).strip() for item in section_payloads if str(item.get("summary", "")).strip()]
    if chapter_summary.strip().lower() == (chapter_title or "").strip().lower() or _count_words(chapter_summary) < 8:
        chapter_summary = " ".join(section_summaries[:3]).strip()[:1800]
    if (not chapter_summary or not main_topics or not important_terms or error) and section_summaries:
        retry_payload, retry_error, retry_meta = _json_object_call_with_meta(
            _minimal_chapter_prompt(chapter_title, _truncate_for_prompt(evidence, 700)),
            analysis_type="chapter_fast_retry_quality_v2",
        )
        retry_summary = _payload_text(retry_payload, ("chapter_summary", "summary", "резюме", "краткое_содержание"))
        retry_topics = _filter_term_strings(
            _clean_string_items(_payload_value(retry_payload, ("main_topics", "topics", "темы", "основные_темы")), limit=10, max_len=200),
            limit=7,
            max_len=200,
        )
        retry_terms = _filter_term_strings(
            _clean_string_items(_payload_value(retry_payload, ("key_terms", "terms", "термины", "ключевые_термины")), limit=14, max_len=120),
            limit=10,
            max_len=120,
        )
        if retry_summary and retry_topics and retry_terms and not retry_error:
            chapter_summary = retry_summary
            main_topics = retry_topics
            important_terms = retry_terms
            error = ""
            call_meta = {**call_meta, "retry_used": True, "retry_meta": retry_meta}
    if chapter_summary and main_topics and important_terms and not error:
        return {
            "chapter_title": chapter_title,
            "chapter_summary": chapter_summary[:1800],
            "main_topics": main_topics,
            "subtopics": main_topics,
            "concept_map": [],
            "learning_goals": [],
            "important_terms": important_terms,
            "_meta": {
                "llm_used": True,
                "fallback_used": False,
                "llm_failure": "",
                "fast_mode": True,
                "minimal_json_calls": {"combined": True},
                **call_meta,
            },
        }

    terms: list[str] = []
    summaries = []
    for payload_item in section_payloads:
        summaries.append(str(payload_item.get("summary", "")).strip())
        for term in payload_item.get("terms", []):
            value = str(term).strip()
            if value and not _is_generic_term_name(value):
                terms.append(value)
        for term in payload_item.get("key_terms", []):
            if isinstance(term, dict):
                value = str(term.get("term", "")).strip()
                if value and not _is_generic_term_name(value):
                    terms.append(value)
    fallback_summary_parts = [item for item in summaries if item]
    if len(fallback_summary_parts) > 4:
        fallback_summary_parts = fallback_summary_parts[:2] + fallback_summary_parts[len(fallback_summary_parts) // 2 : len(fallback_summary_parts) // 2 + 1] + fallback_summary_parts[-2:]
    return {
        "chapter_title": chapter_title,
        "chapter_summary": " ".join(fallback_summary_parts).strip()[:1800],
        "main_topics": list(dict.fromkeys(terms))[:7],
        "subtopics": list(dict.fromkeys(terms))[:7],
        "concept_map": [],
        "learning_goals": [],
        "important_terms": list(dict.fromkeys(terms))[:10],
        "_meta": {
            "llm_used": False,
            "fallback_used": True,
            "llm_failure": error or "invalid_or_empty_chapter_json",
            "fast_mode": True,
            **call_meta,
        },
    }


def build_book_fast_with_llm(chapter_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not chapter_payloads:
        return {
            "book_summary": "",
            "global_themes": [],
            "global_concepts": [],
            "recommended_learning_path": [],
            "final_knowledge_map": [],
            "_meta": {"llm_used": False, "fallback_used": True, "llm_failure": "empty_chapters", "cache_hit": False},
        }

    evidence_lines = []
    for item in chapter_payloads[:8]:
        terms = _filter_term_strings([str(term).strip() for term in item.get("important_terms", [])], limit=10)
        topics = _filter_term_strings([str(topic).strip() for topic in item.get("main_topics", [])], limit=7, max_len=180)
        summary = " ".join(str(item.get("chapter_summary", "")).split()).strip()[:260]
        title = " ".join(str(item.get("chapter_title", "")).split()).strip()[:120]
        evidence_lines.append(f"{title}: {summary}; topics: {', '.join(topics[:5])}; terms: {', '.join(terms[:7])}")
    evidence = _truncate_for_prompt("\n".join(evidence_lines), 2200)
    payload, error, call_meta = _json_object_call_with_meta(
        _fast_book_prompt(evidence),
        analysis_type="book_fast_combined_v2",
    )
    if isinstance(payload, dict):
        payload = {
            "book_summary": _payload_text(payload, ("book_summary", "summary", "резюме", "краткое_содержание")),
            "global_themes": _payload_value(payload, ("global_themes", "themes", "topics", "темы", "глобальные_темы")),
            "learning_path": _payload_value(payload, ("learning_path", "path", "план_изучения", "траектория_изучения")),
        }
    cleaned_payload, quality_flags = _cleanup_book_grounding(payload or {}, evidence)

    book_summary = cleaned_payload.get("book_summary", "")
    global_themes = cleaned_payload.get("global_themes", [])
    learning_path = cleaned_payload.get("learning_path", [])
    chapter_titles = []
    for item in chapter_payloads:
        title = " ".join(str(item.get("chapter_title", "")).split()).strip()
        title = re.sub(r"^(?:глава|chapter)\s+\d+\.?\s*", "", title, flags=re.IGNORECASE).strip()
        if title and not _is_generic_term_name(title):
            chapter_titles.append(title)
    coverage_flags: list[str] = []
    if len(chapter_payloads) >= 6:
        summary_lower = str(book_summary or "").lower()
        covered = 0
        for title in chapter_titles:
            title_words = []
            for word in WORD_RE.findall(title.lower()):
                if len(word) < 5:
                    continue
                normal = morph.parse(word)[0].normal_form
                if word in GENERIC_SINGLE_TERMS or normal in GENERIC_SINGLE_TERMS:
                    continue
                title_words.append(word)
            if title.lower() in summary_lower or any(word in summary_lower for word in title_words[:2]):
                covered += 1
        required_coverage = min(len(chapter_titles), max(6, int(len(chapter_titles) * 0.75)))
        if covered < required_coverage:
            coverage_flags.append("book_summary_coverage_repaired")
            repaired_summary = _deterministic_book_summary_from_chapters(chapter_titles, chapter_payloads)
            if repaired_summary:
                book_summary = repaired_summary[:2600]
        missing_theme_titles = [
            title for title in chapter_titles if not any(title.lower() == str(theme).lower() for theme in global_themes)
        ]
        if len(global_themes) < min(6, len(chapter_titles)) or len(missing_theme_titles) >= 2 or coverage_flags:
            coverage_flags.append("global_themes_rebuilt_from_chapters")
            global_themes = list(dict.fromkeys(chapter_titles + list(global_themes)))[:8]
        else:
            global_themes = list(dict.fromkeys(list(global_themes) + chapter_titles))[:8]
        missing_path_titles = [
            title for title in chapter_titles if not any(title.lower() == str(step).lower() for step in learning_path)
        ]
        if len(learning_path) < min(6, len(chapter_titles)) or len(missing_path_titles) >= 2 or coverage_flags:
            coverage_flags.append("learning_path_rebuilt_from_chapters")
            learning_path = list(dict.fromkeys(chapter_titles + list(learning_path)))[:8]
        else:
            learning_path = list(dict.fromkeys(list(learning_path) + chapter_titles))[:8]
    success = bool(book_summary and global_themes and learning_path and not error)

    if success:
        return {
            "book_summary": str(book_summary)[:2600],
            "global_themes": global_themes[:8],
            "global_concepts": global_themes[:8],
            "recommended_learning_path": learning_path[:8],
            "final_knowledge_map": [],
            "_meta": {
                "llm_used": True,
                "fallback_used": False,
                "llm_failure": "",
                "fast_mode": True,
                "quality_flags": list(dict.fromkeys(quality_flags)),
                "validator_cleanup_flags": list(dict.fromkeys(list(cleaned_payload.get("_validator_cleanup_flags", [])) + coverage_flags)),
                "minimal_json_calls": {"combined": True},
                **call_meta,
            },
        }

    summaries = [str(item.get("chapter_summary", "")).strip() for item in chapter_payloads if str(item.get("chapter_summary", "")).strip()]
    chapter_titles = []
    for item in chapter_payloads:
        title = " ".join(str(item.get("chapter_title", "")).split()).strip()
        title = re.sub(r"^(?:глава|chapter)\s+\d+\.?\s*", "", title, flags=re.IGNORECASE).strip()
        if title and not _is_generic_term_name(title):
            chapter_titles.append(title)
    terms: list[str] = []
    for item in chapter_payloads:
        terms.extend([str(term).strip() for term in item.get("important_terms", []) if str(term).strip()])
    fallback_themes = chapter_titles[:8] or _filter_term_strings(terms, limit=7, max_len=180)
    fallback_learning_path = chapter_titles[:8] or fallback_themes[:8]
    if summaries:
        fallback_summary = _deterministic_book_summary_from_chapters(chapter_titles, chapter_payloads) or " ".join(summaries[:2] + summaries[2:])[:2600].strip()
    else:
        fallback_summary = ""
    if not fallback_summary and fallback_themes:
        fallback_summary = "Книга раскрывает основные темы: " + ", ".join(fallback_themes[:6]) + "."
    deterministic_book_ready = bool(fallback_summary and fallback_themes and fallback_learning_path)
    return {
        "book_summary": fallback_summary,
        "global_themes": fallback_themes,
        "global_concepts": _filter_term_strings(terms, limit=12, max_len=180),
        "recommended_learning_path": fallback_learning_path,
        "final_knowledge_map": [],
        "_meta": {
            "llm_used": deterministic_book_ready,
            "fallback_used": not deterministic_book_ready,
            "llm_failure": "" if deterministic_book_ready else (error or "invalid_or_empty_book_json"),
            "fast_mode": True,
            "quality_flags": quality_flags,
            "validator_cleanup_flags": list(dict.fromkeys(list(cleaned_payload.get("_validator_cleanup_flags", [])) + ["deterministic_book_summary"])),
            "deterministic_book_summary": deterministic_book_ready,
            **call_meta,
        },
    }


def analyze_section_with_llm(
    *,
    section_title: str,
    section_text: str,
    chapter_title: str = "",
    section_type: str = "main_content",
) -> dict[str, Any]:
    """
    LLM-first section analysis with minimal JSON calls.
    Falls back to heuristic extraction for this section only after JSON repair fails.
    """

    text = (section_text or "").strip()
    if not text:
        return {
            "section_title": section_title,
            "section_type": section_type,
            "summary": "",
            "key_terms": [],
            "subtopics": [],
            "important_facts": [],
            "formulas_or_protocols": [],
            "source_quotes": [],
            "difficulty_level": "unknown",
            "links_to_parent_theme": [],
            "quality_flags": ["empty_section"],
            "_meta": {"llm_used": False, "fallback_used": True, "llm_failure": "empty_section"},
        }

    cfg = get_llm_runtime_config()
    max_chars = min(int(cfg["max_input_chars"]), int(os.getenv("SECTION_LLM_MAX_INPUT_CHARS", "1800")))
    text_for_prompt = _prepare_section_text_for_llm(text, max_chars=max_chars)

    summary_payload, summary_error = _json_object_call(
        _minimal_section_prompt("summary", text_for_prompt),
        analysis_type="section_summary_minimal",
    )
    terms_payload, terms_error = _json_object_call(
        _minimal_section_prompt("terms", text_for_prompt),
        analysis_type="section_terms_minimal",
    )
    subtopics_payload, subtopics_error = _json_object_call(
        _minimal_section_prompt("subtopics", text_for_prompt),
        analysis_type="section_subtopics_minimal",
    )

    summary = " ".join(str((summary_payload or {}).get("summary", "")).split()).strip()
    terms = _filter_term_strings(_clean_string_items((terms_payload or {}).get("terms"), limit=16), limit=8)
    subtopic_names = _clean_string_items((subtopics_payload or {}).get("subtopics"), limit=6, max_len=180)
    errors = [item for item in (summary_error, terms_error, subtopics_error) if item]

    if summary and terms and subtopic_names and not errors:
        source_quote = _pick_fallback_sentences(text_for_prompt, limit=1)[:320]
        key_terms = [
            {
                "term": term,
                "definition": f"Концепт раскрывается в разделе: {term}.",
                "importance": max(0.1, round(1.0 - index * 0.08, 3)),
                "source_quote": source_quote,
            }
            for index, term in enumerate(terms)
        ]
        subtopics = [
            {
                "title": title,
                "summary": f"Подтема раскрывается в данном разделе: {title}.",
                "source_quote": source_quote,
            }
            for title in subtopic_names
        ]
        return {
            "section_title": section_title,
            "section_type": section_type,
            "summary": summary[:1600],
            "key_terms": key_terms,
            "subtopics": subtopics,
            "important_facts": [],
            "formulas_or_protocols": [],
            "source_quotes": [source_quote] if source_quote else [],
            "difficulty_level": "intermediate",
            "links_to_parent_theme": [chapter_title] if chapter_title else [],
            "quality_flags": [],
            "_meta": {
                "llm_used": True,
                "fallback_used": False,
                "llm_failure": "",
                "minimal_json_calls": {
                    "summary": True,
                    "terms": True,
                    "subtopics": True,
                },
            },
        }

    # Fallback for this section only.
    fallback_terms = _fallback_phrases_for_section(text, max_items=8)
    fallback_summary = ensure_grounded_summary("", text, _pick_fallback_sentences(text, limit=3))
    return {
        "section_title": section_title,
        "section_type": section_type,
        "summary": fallback_summary[:1600],
        "key_terms": fallback_terms,
        "subtopics": [
            {
                "title": item["term"],
                "summary": f"Subtopic based on section context: {item['term']}.",
                "source_quote": item["source_quote"],
            }
            for item in fallback_terms[:6]
        ],
        "important_facts": [item["definition"] for item in fallback_terms[:6]],
        "formulas_or_protocols": [],
        "source_quotes": [item["source_quote"] for item in fallback_terms[:6] if item.get("source_quote")],
        "difficulty_level": "intermediate",
        "links_to_parent_theme": [chapter_title] if chapter_title else [],
        "quality_flags": ["fallback_section_analysis"],
        "_meta": {
            "llm_used": False,
            "fallback_used": True,
            "llm_failure": ",".join(errors) or "invalid_or_empty_llm_json",
            "minimal_json_calls": {
                "summary": bool(summary_payload),
                "terms": bool(terms_payload),
                "subtopics": bool(subtopics_payload),
            },
        },
    }


def merge_chapter_analyses_with_llm(chapter_title: str, section_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not section_payloads:
        return {
            "chapter_title": chapter_title,
            "chapter_summary": "",
            "main_topics": [],
            "subtopics": [],
            "concept_map": [],
            "learning_goals": [],
            "important_terms": [],
            "_meta": {"llm_used": False, "fallback_used": True, "llm_failure": "empty_sections"},
        }

    evidence_lines = []
    for item in section_payloads[:6]:
        terms = [
            str(term.get("term", "")).strip()
            for term in item.get("key_terms", [])
            if isinstance(term, dict) and str(term.get("term", "")).strip()
        ][:6]
        summary = " ".join(str(item.get("summary", "")).split()).strip()[:320]
        title = " ".join(str(item.get("section_title", "")).split()).strip()[:120]
        evidence_lines.append(f"Раздел: {title}\nКратко: {summary}\nТермины: {', '.join(terms)}")
    evidence = "\n".join(evidence_lines)
    evidence = _truncate_for_prompt(evidence, 1200)
    summary_payload, summary_error = _json_object_call(
        _minimal_chapter_task_prompt("summary", chapter_title, evidence),
        analysis_type="chapter_summary_minimal",
    )
    topics_payload, topics_error = _json_object_call(
        _minimal_chapter_task_prompt("topics", chapter_title, evidence),
        analysis_type="chapter_topics_minimal",
    )
    terms_payload, terms_error = _json_object_call(
        _minimal_chapter_task_prompt("terms", chapter_title, evidence),
        analysis_type="chapter_terms_minimal",
    )

    chapter_summary = " ".join(str((summary_payload or {}).get("chapter_summary", "")).split()).strip()
    main_topics = _clean_string_items((topics_payload or {}).get("main_topics"), limit=8, max_len=200)
    important_terms = _filter_term_strings(
        _clean_string_items((terms_payload or {}).get("key_terms"), limit=24, max_len=120),
        limit=12,
        max_len=120,
    )
    errors = [item for item in (summary_error, topics_error, terms_error) if item]

    if chapter_summary and main_topics and important_terms and not errors:
        return {
            "chapter_title": chapter_title,
            "chapter_summary": chapter_summary[:2200],
            "main_topics": main_topics,
            "subtopics": main_topics,
            "concept_map": [],
            "learning_goals": [],
            "important_terms": important_terms,
            "_meta": {
                "llm_used": True,
                "fallback_used": False,
                "llm_failure": "",
                "minimal_json": True,
                "minimal_json_calls": {"summary": True, "topics": True, "terms": True},
            },
        }

    section_summaries = [str(item.get("summary", "")).strip() for item in section_payloads if str(item.get("summary", "")).strip()]
    merged = summarize_book(section_summaries[:20])
    terms = []
    for payload in section_payloads:
        for term in payload.get("key_terms", []):
            value = str(term.get("term", "")).strip()
            if value and not _is_generic_term_name(value):
                terms.append(value)
    return {
        "chapter_title": chapter_title,
        "chapter_summary": merged[:2200],
        "main_topics": list(dict.fromkeys(terms))[:10],
        "subtopics": list(dict.fromkeys(terms))[:14],
        "concept_map": [],
        "learning_goals": [],
        "important_terms": list(dict.fromkeys(terms))[:20],
        "_meta": {
            "llm_used": False,
            "fallback_used": True,
            "llm_failure": ",".join(errors) or "invalid_or_empty_chapter_json",
            "minimal_json_calls": {
                "summary": bool(summary_payload),
                "topics": bool(topics_payload),
                "terms": bool(terms_payload),
            },
        },
    }


def build_book_analysis_with_llm(chapter_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not chapter_payloads:
        return {
            "book_summary": "",
            "global_themes": [],
            "global_concepts": [],
            "recommended_learning_path": [],
            "final_knowledge_map": [],
            "_meta": {"llm_used": False, "fallback_used": True, "llm_failure": "empty_chapters"},
        }

    evidence_lines = []
    for item in chapter_payloads[:12]:
        terms = _filter_term_strings(
            [str(term).strip() for term in item.get("important_terms", []) if str(term).strip()],
            limit=8,
        )
        summary = " ".join(str(item.get("chapter_summary", "")).split()).strip()[:420]
        title = " ".join(str(item.get("chapter_title", "")).split()).strip()[:120]
        evidence_lines.append(f"Глава: {title}\nКратко: {summary}\nТермины: {', '.join(terms)}")
    evidence = _truncate_for_prompt("\n".join(evidence_lines), 700)
    book_payload, book_error = _json_object_call(
        _minimal_book_prompt(evidence),
        analysis_type="book_analysis_minimal_combined",
    )

    book_summary = " ".join(str((book_payload or {}).get("book_summary", "")).split()).strip()
    global_themes = _clean_string_items((book_payload or {}).get("global_themes"), limit=8, max_len=220)
    learning_path = _clean_string_items((book_payload or {}).get("learning_path"), limit=6, max_len=260)
    errors = [item for item in (book_error,) if item]

    if book_summary and global_themes and learning_path and not errors:
        return {
            "book_summary": book_summary[:3500],
            "global_themes": global_themes,
            "global_concepts": global_themes,
            "recommended_learning_path": learning_path,
            "final_knowledge_map": [],
            "_meta": {
                "llm_used": True,
                "fallback_used": False,
                "llm_failure": "",
                "minimal_json": True,
                "minimal_json_calls": {"combined": True},
            },
        }

    summaries = [str(item.get("chapter_summary", "")).strip() for item in chapter_payloads if str(item.get("chapter_summary", "")).strip()]
    terms = []
    for payload in chapter_payloads:
        terms.extend(payload.get("important_terms", []))
    return {
        "book_summary": summarize_book(summaries[:40])[:3500],
        "global_themes": list(dict.fromkeys(str(item.get("chapter_title", "")).strip() for item in chapter_payloads if item.get("chapter_title")))[:20],
        "global_concepts": list(dict.fromkeys(str(item).strip() for item in terms if str(item).strip()))[:30],
        "recommended_learning_path": [],
        "final_knowledge_map": [],
        "_meta": {
            "llm_used": False,
            "fallback_used": True,
            "llm_failure": ",".join(errors) or "invalid_or_empty_book_json",
            "minimal_json_calls": {
                "combined": bool(book_payload),
            },
        },
    }


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


def _fallback_atomic_thoughts(window_text: str, sentences_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sentence_items = []
    for item in sentences_metadata:
        sentence_text = str(item.get("text", "")).strip()
        sentence_id = str(item.get("id", "")).strip()
        if not sentence_text or not sentence_id:
            continue
        score = len(WORD_RE.findall(sentence_text)) + min(20, len(set(sentence_text.lower().split())))
        sentence_items.append((score, sentence_id, sentence_text))
    sentence_items.sort(reverse=True)
    picked = sentence_items[:2] if len(sentence_items) > 1 else sentence_items[:1]

    result: list[dict[str, Any]] = []
    for _, sentence_id, sentence_text in picked:
        if len(sentence_text) < 20:
            continue
        concept_candidates = [token.lower() for token in WORD_RE.findall(sentence_text) if len(token) > 4][:4]
        result.append(
            {
                "text": sentence_text[:420],
                "source_sentence_ids": [sentence_id],
                "concept_candidates": list(dict.fromkeys(concept_candidates)),
                "confidence": 0.35,
                "quote": sentence_text[:260],
            }
        )
    if result:
        return result

    fallback_text = _pick_fallback_sentences(window_text, limit=2)
    if fallback_text:
        fallback_ids = [str(item.get("id")) for item in sentences_metadata[:2] if item.get("id")]
        return [
            {
                "text": fallback_text[:420],
                "source_sentence_ids": fallback_ids,
                "concept_candidates": [],
                "confidence": 0.3,
                "quote": fallback_text[:260],
            }
        ]
    return []


def extract_atomic_thoughts(window_text: str, sentences_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract atomic thoughts from a sentence window.
    Returns list[dict] and never raises.
    """

    safe_sentences = []
    for item in sentences_metadata[:12]:
        sentence_id = str(item.get("id", "")).strip()
        sentence_text = str(item.get("text", "")).strip()
        if sentence_id and sentence_text:
            safe_sentences.append({"id": sentence_id, "text": sentence_text[:420]})
    if not safe_sentences:
        return _fallback_atomic_thoughts(window_text, sentences_metadata)

    prompt = (
        "Ты анализируешь фрагмент учебного/книжного текста.\n\n"
        "Задача:\n"
        "Извлеки атомарные смысловые мысли из текста.\n\n"
        "Правила:\n"
        "1. Используй только информацию из текста.\n"
        "2. Не добавляй свои знания.\n"
        "3. Не пиши общие фразы типа 'автор говорит', 'в тексте рассматривается', 'главная мысль'.\n"
        "4. Одна мысль = один конкретный смысловой тезис.\n"
        "5. Каждая мысль должна ссылаться на ID предложений, из которых она взята.\n"
        "6. Если мысль нельзя подтвердить предложениями, не добавляй её.\n"
        "7. Если мыслей нет, верни пустой массив.\n"
        "8. Верни только JSON без markdown.\n\n"
        "Формат:\n"
        "{\n"
        '  "thoughts": [\n'
        "    {\n"
        '      "text": "...",\n'
        '      "source_sentence_ids": ["s1", "s2"],\n'
        '      "concept_candidates": ["..."],\n'
        '      "confidence": 0.0,\n'
        '      "quote": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Предложения:\n{json.dumps(safe_sentences, ensure_ascii=False)}\n\n"
        f"Текст окна:\n{_truncate_for_prompt(window_text, 1800)}"
    )

    try:
        raw = _ollama_generate(prompt, expect_json=True)
        parsed = _safe_json_parse(raw or "")
        if parsed is None and raw:
            repaired = _repair_json_once(raw)
            parsed = _safe_json_parse(repaired)

        items: list[dict[str, Any]] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("thoughts"), list):
            items = [item for item in parsed["thoughts"] if isinstance(item, dict)]
        elif isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]

        sentence_ids = {item["id"] for item in safe_sentences}
        cleaned: list[dict[str, Any]] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if len(text) < 12:
                continue
            raw_source_ids = item.get("source_sentence_ids", [])
            source_ids = [str(value) for value in raw_source_ids if str(value) in sentence_ids]
            if not source_ids:
                continue
            concepts = [
                str(value).strip().lower()
                for value in item.get("concept_candidates", [])
                if str(value).strip()
            ]
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            cleaned.append(
                {
                    "text": text[:520],
                    "source_sentence_ids": list(dict.fromkeys(source_ids)),
                    "concept_candidates": list(dict.fromkeys(concepts))[:8],
                    "confidence": max(0.0, min(1.0, confidence)),
                    "quote": str(item.get("quote", "")).strip()[:260],
                }
            )

        if cleaned:
            return cleaned
    except Exception:
        logger.exception("extract_atomic_thoughts failed, fallback will be used")

    return _fallback_atomic_thoughts(window_text, safe_sentences)


def merge_thought_cluster(thoughts: list[str]) -> str:
    """Merge near-duplicate thought variants into one grounded thought."""

    cleaned = [_normalize for _normalize in (" ".join((item or "").split()) for item in thoughts) if _normalize]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    prompt = (
        "Слей схожие формулировки в одну короткую мысль.\n"
        "Используй только данные из входа, без новых фактов.\n"
        "Верни одну фразу (до 35 слов) без markdown.\n\n"
        f"Мысли:\n{json.dumps(cleaned[:6], ensure_ascii=False)}"
    )
    try:
        raw = _ollama_generate(prompt, expect_json=False)
        if raw:
            return " ".join(raw.split())[:420]
    except Exception:
        logger.exception("merge_thought_cluster failed")

    best = max(cleaned, key=len)
    return best[:420]


def name_semantic_block(thought_clusters: list[str]) -> str:
    cleaned = [" ".join((item or "").split()) for item in thought_clusters if item and item.strip()]
    if not cleaned:
        return ""

    prompt = (
        "Придумай короткое название смыслового блока (2-7 слов) на основе мыслей.\n"
        "Только по входу, без выдумки. Ответ только текстом.\n\n"
        f"Мысли:\n{json.dumps(cleaned[:8], ensure_ascii=False)}"
    )
    try:
        raw = _ollama_generate(prompt, expect_json=False)
        if raw:
            return " ".join(raw.split())[:140]
    except Exception:
        logger.exception("name_semantic_block failed")

    fallback = cleaned[0].split(".")[0].strip()
    words = fallback.split()
    return " ".join(words[:7])[:140]


def build_block_main_meaning(thought_clusters: list[dict[str, Any]] | list[str], source_text: str) -> str:
    """
    Build strong main-meaning summary for a semantic block.
    Returns text and never raises.
    """

    if isinstance(thought_clusters, list):
        lines = []
        for item in thought_clusters[:8]:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                concepts = ", ".join(item.get("concept_candidates", [])[:4]) if isinstance(item.get("concept_candidates"), list) else ""
                if text:
                    lines.append(f"- {text}" + (f" | concepts: {concepts}" if concepts else ""))
            else:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
    else:
        lines = []

    evidence = _truncate_for_prompt(source_text, 2600)
    prompt = (
        "Ты формируешь main meaning смыслового блока книги.\n"
        "Требования:\n"
        "1) 2-4 предложения.\n"
        "2) Только факты из источника.\n"
        "3) Конкретно: что объясняется, какие связи/законы/идеи раскрываются.\n"
        "4) Без общих фраз вроде 'в тексте рассматривается'.\n"
        "5) Если есть ключевые концепты, аккуратно включи их в формулировку.\n\n"
        f"Мысли:\n{chr(10).join(lines) if lines else '-'}\n\n"
        f"Источник:\n{evidence}\n\n"
        "Ответ только текстом."
    )
    try:
        raw = _ollama_generate(prompt, expect_json=False)
        if raw:
            return ensure_grounded_summary(raw, source_text, extractive_theme_summary_from_digests([{"summary": line} for line in lines], limit_sentences=3))
    except Exception:
        logger.exception("build_block_main_meaning failed")

    fallback = extractive_theme_summary_from_digests([{"summary": line} for line in lines], limit_sentences=3)
    if not fallback:
        fallback = _pick_fallback_sentences(source_text, limit=3)
    return ensure_grounded_summary(fallback, source_text, fallback)


def mini_check_logical_block(title: str, block_text: str) -> dict[str, Any]:
    """
    Lightweight optional LLM quality check for segmentation.
    Designed for debug/mini-test use and never raises.
    """

    text = " ".join((block_text or "").split()).strip()
    title = " ".join((title or "").split()).strip()
    if not text:
        return {
            "llm_used": False,
            "single_idea": False,
            "split_recommended": True,
            "title_ok": False,
            "themes": [],
            "notes": "empty_block",
            "confidence": 0.0,
        }

    ready = ensure_llm_ready(require_enabled=False)
    if not ready.get("ok"):
        return {
            "llm_used": False,
            "single_idea": True,
            "split_recommended": False,
            "title_ok": True,
            "themes": [],
            "notes": "llm_unavailable",
            "confidence": 0.0,
        }

    prompt = (
        "You validate segmentation quality of one logical block from a book.\n"
        "Return STRICT JSON only with schema:\n"
        "{\n"
        '  "single_idea": true,\n'
        '  "split_recommended": false,\n'
        '  "title_ok": true,\n'
        '  "themes": ["..."],\n'
        '  "notes": "short reason",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Rules:\n"
        "- Use only provided title and text.\n"
        "- split_recommended=true if block mixes multiple unrelated topics or is too broad.\n"
        "- title_ok=false if title is too generic or mismatched.\n"
        "- themes: 1-4 concrete topics from block text.\n\n"
        f"Title: {title or 'Untitled'}\n"
        f"Text:\n{_truncate_for_prompt(text, 2400)}\n"
    )
    raw = _ollama_generate(
        prompt,
        expect_json=True,
        tier="fast",
        analysis_type="segmentation_mini_check",
        cache_ttl=int(os.getenv("LLM_CACHE_TTL_SECONDS", "2592000")),
    )
    parsed = _safe_json_parse(raw or "")
    if parsed is None and raw:
        parsed = _safe_json_parse(_repair_json_once(raw))

    if isinstance(parsed, dict):
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        themes_raw = parsed.get("themes", [])
        themes = [str(item).strip()[:120] for item in themes_raw if str(item).strip()][:4] if isinstance(themes_raw, list) else []
        return {
            "llm_used": True,
            "single_idea": bool(parsed.get("single_idea", True)),
            "split_recommended": bool(parsed.get("split_recommended", False)),
            "title_ok": bool(parsed.get("title_ok", True)),
            "themes": themes,
            "notes": str(parsed.get("notes", "")).strip()[:320],
            "confidence": max(0.0, min(1.0, confidence)),
        }

    return {
        "llm_used": False,
        "single_idea": True,
        "split_recommended": False,
        "title_ok": True,
        "themes": [],
        "notes": "llm_invalid_json",
        "confidence": 0.0,
    }
