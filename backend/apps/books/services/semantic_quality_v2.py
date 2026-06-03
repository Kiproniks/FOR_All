from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pymorphy3


morph = pymorphy3.MorphAnalyzer()

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-]*")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")

STANDARD_ABBREVIATIONS = {
    "TCP",
    "UDP",
    "IP",
    "DNS",
    "HTTP",
    "HTTPS",
    "TLS",
    "QUIC",
    "ETHERNET",
    "WI-FI",
    "WIFI",
    "OSI",
    "MAC",
    "ARP",
    "SMTP",
    "WPA2",
    "DNSSEC",
    "QOS",
    "QOE",
    "SDN",
    "BGP",
    "OSPF",
    "ADSL",
    "ASCII",
    "BLUETOOTH",
    "CMTS",
    "CSMA",
    "CSMA/CD",
    "DOCSIS",
    "E-MAIL",
    "IEEE",
    "IEEE 802.3",
    "IPV4",
    "IPV6",
    "LAN",
    "MAN",
    "PAN",
    "PGP",
    "PPP",
    "RSA",
    "SONET",
    "TELNET",
    "WAN",
    "WWW",
}

TECH_TERM_PATTERNS = (
    re.compile(r"^IEEE\s+802(?:\.\d+)+$", re.IGNORECASE),
    re.compile(r"^CSMA/CD$", re.IGNORECASE),
    re.compile(r"^TCP[-\s]?соединение$", re.IGNORECASE),
    re.compile(r"^(?:протокол|алгоритм|стандарт|служба|уровень|мосты?)\s+[A-Za-z0-9][A-Za-z0-9./-]*$", re.IGNORECASE),
    re.compile(r"^[A-Za-z0-9][A-Za-z0-9./-]*\s+(?:протокол|стандарт|алгоритм|уровень|сеть|сети|соединение)$", re.IGNORECASE),
)

GENERIC_SINGLE_TERMS = {
    "автор",
    "блок",
    "вопрос",
    "глава",
    "данные",
    "документ",
    "задача",
    "защита",
    "информация",
    "книга",
    "компьютер",
    "контакты",
    "материал",
    "метод",
    "обработка",
    "объект",
    "ошибка",
    "передача",
    "подход",
    "пример",
    "процесс",
    "развитие",
    "раздел",
    "сбор",
    "связь",
    "сеть",
    "система",
    "соединение",
    "структура",
    "текст",
    "тема",
    "технология",
    "устройство",
    "уровень",
    "управление",
    "часть",
    "элемент",
    "архитектура",
    "год",
    "канал",
    "компания",
    "контроль",
    "предложение",
    "протокол",
    "секция",
    "служба",
    "способность",
    "стандарт",
}

GENERIC_PHRASES = {
    "большое количество",
    "важный вопрос",
    "данный материал",
    "данный раздел",
    "другой способ",
    "обработка данных",
    "обработка информации",
    "основная проблема",
    "передача данных",
    "передача информации",
    "разработка системы",
    "система управления",
    "с одной стороны",
    "с другой стороны",
    "следующий раздел",
    "такой образ",
    "таким образом",
    "текущий раздел",
    "этот пример",
    "эта глава",
}

SUSPICIOUS_TERMS = {
    "30 tb data capacity",
    "centrally controlled",
    "commutation of packets",
    "mаgnetic tape cartridge",
    "packet switched communication",
    "splitted node",
    "внутриполисные коммутаторы",
    "квадратичные",
    "лайт-фрейм",
    "робот-телевизор",
    "секурити-задача",
    "сопротивление",
    "сплitted node",
    "тензоры",
}

TRANSLATABLE_ENGLISH_JUNK = {
    "bandwidth",
    "bridges",
    "minislots",
}


@dataclass
class SectionValidation:
    payload: dict[str, Any]
    quality_flags: list[str]
    removed_generic_terms: list[str]
    removed_suspicious_terms: list[str]
    mixed_language_artifacts: list[str]
    weak_grounding_terms: list[str]
    generic_terms_ratio: float
    should_retry: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "quality_flags": self.quality_flags,
            "removed_generic_terms": self.removed_generic_terms,
            "removed_suspicious_terms": self.removed_suspicious_terms,
            "mixed_language_artifacts": self.mixed_language_artifacts,
            "weak_grounding_terms": self.weak_grounding_terms,
            "generic_terms_ratio": self.generic_terms_ratio,
            "should_retry": self.should_retry,
        }


def clean_text(value: Any, *, max_len: int = 2000) -> str:
    return " ".join(str(value or "").split()).strip()[:max_len]


def _words(value: str) -> list[str]:
    return WORD_RE.findall((value or "").lower())


def _lemma(word: str) -> str:
    if LATIN_RE.fullmatch(word):
        return word.lower()
    try:
        return morph.parse(word)[0].normal_form
    except Exception:
        return word.lower()


def _lemma_set(value: str) -> set[str]:
    result = set()
    for word in _words(value):
        if len(word) < 3:
            continue
        result.add(_lemma(word))
    return result


def _dedupe(items: list[str], *, limit: int, max_len: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = clean_text(item, max_len=max_len)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def extract_strings(raw: Any, *, value_key: str | None = None, limit: int = 12, max_len: int = 180) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return _dedupe([raw], limit=limit, max_len=max_len)
    if not isinstance(raw, list):
        return []

    values: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            if value_key and item.get(value_key):
                values.append(str(item.get(value_key)))
            elif item.get("term"):
                values.append(str(item.get("term")))
            elif item.get("title"):
                values.append(str(item.get("title")))
            elif item.get("name"):
                values.append(str(item.get("name")))
        else:
            values.append(str(item))
    return _dedupe(values, limit=limit, max_len=max_len)


def is_standard_abbreviation(value: str) -> bool:
    token = clean_text(value).upper()
    return token in STANDARD_ABBREVIATIONS


def is_allowed_technical_term(value: str) -> bool:
    clean = clean_text(value, max_len=220)
    if not clean:
        return False
    if is_standard_abbreviation(clean):
        return True
    return any(pattern.search(clean) for pattern in TECH_TERM_PATTERNS)


def is_generic_term(value: str) -> bool:
    clean = clean_text(value, max_len=220).lower()
    if not clean:
        return True
    if clean in GENERIC_PHRASES:
        return True
    if is_allowed_technical_term(clean):
        return False
    words = _words(clean)
    if len(words) == 1:
        return words[0] in GENERIC_SINGLE_TERMS or _lemma(words[0]) in GENERIC_SINGLE_TERMS
    if len(words) > 8:
        return True
    meaningful = [word for word in words if word not in GENERIC_SINGLE_TERMS and _lemma(word) not in GENERIC_SINGLE_TERMS]
    return not meaningful


def is_mixed_language_artifact(value: str) -> bool:
    clean = clean_text(value, max_len=220)
    if not clean:
        return False
    if is_allowed_technical_term(clean):
        return False
    lower = clean.lower()
    if lower in TRANSLATABLE_ENGLISH_JUNK:
        return True
    if lower in SUSPICIOUS_TERMS:
        return True
    latin_words = re.findall(r"[A-Za-z]{3,}", clean)
    cyrillic = bool(CYRILLIC_RE.search(clean))
    if len(latin_words) >= 2:
        upper_tokens = {token.upper() for token in latin_words}
        if not upper_tokens <= STANDARD_ABBREVIATIONS:
            return True
    # Mixed-script words are usually OCR/LLM artifacts, not real terms.
    for word in re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", clean):
        if LATIN_RE.search(word) and CYRILLIC_RE.search(word):
            return True
    return cyrillic and len(latin_words) >= 3


def is_suspicious_term(value: str) -> bool:
    clean = clean_text(value, max_len=220).lower()
    if clean in SUSPICIOUS_TERMS:
        return True
    return any(term in clean for term in SUSPICIOUS_TERMS if len(term) > 8)


def is_weakly_grounded(value: str, source_text: str) -> bool:
    clean = clean_text(value, max_len=220)
    if not clean:
        return True
    lower_source = (source_text or "").lower()
    lower_value = clean.lower()
    if lower_value in lower_source:
        return False
    if is_allowed_technical_term(clean) and clean.upper() in lower_source.upper():
        return False

    value_lemmas = {
        item
        for item in _lemma_set(clean)
        if item not in GENERIC_SINGLE_TERMS and len(item) >= 4
    }
    if not value_lemmas:
        return True
    source_lemmas = _lemma_set(source_text)
    overlap = len(value_lemmas & source_lemmas)
    required = 1 if len(value_lemmas) <= 2 else max(2, len(value_lemmas) // 2)
    return overlap < required


def clean_term_list_v2(items: list[str], source_text: str, *, limit: int = 8) -> tuple[list[str], dict[str, list[str] | float]]:
    removed_generic: list[str] = []
    removed_suspicious: list[str] = []
    mixed_artifacts: list[str] = []
    weak_grounding: list[str] = []
    accepted: list[str] = []
    seen: set[str] = set()

    for item in items:
        value = clean_text(item, max_len=180)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        if is_generic_term(value):
            removed_generic.append(value)
            continue
        if is_suspicious_term(value):
            removed_suspicious.append(value)
            continue
        if is_mixed_language_artifact(value):
            mixed_artifacts.append(value)
            continue
        if is_weakly_grounded(value, source_text):
            weak_grounding.append(value)
            continue
        accepted.append(value)
        if len(accepted) >= limit:
            break

    total = max(1, len([item for item in items if clean_text(item)]))
    return accepted, {
        "removed_generic_terms": removed_generic,
        "removed_suspicious_terms": removed_suspicious,
        "mixed_language_artifacts": mixed_artifacts,
        "weak_grounding_terms": weak_grounding,
        "generic_terms_ratio": len(removed_generic) / total,
    }


def _is_title_echo(text: str, section_title: str) -> bool:
    summary = clean_text(text, max_len=500).lower()
    title = clean_text(section_title, max_len=180).lower()
    if not summary or not title:
        return False
    title_words = {word for word in _words(title) if len(word) >= 4}
    summary_words = {word for word in _words(summary) if len(word) >= 4}
    if not title_words:
        return False
    overlap = len(title_words & summary_words) / max(1, len(title_words))
    return overlap >= 0.8 and len(summary_words) <= len(title_words) + 5


def validate_section_payload_v2(payload: dict[str, Any], source_text: str, section_title: str) -> SectionValidation:
    summary = clean_text(payload.get("summary"), max_len=1500)
    main_idea = clean_text(payload.get("main_idea"), max_len=800)
    terms_raw = extract_strings(payload.get("terms") if payload.get("terms") is not None else payload.get("key_terms"), limit=16)
    subtopics_raw = extract_strings(payload.get("subtopics"), limit=14, max_len=220)
    bad_input_notes = extract_strings(payload.get("bad_input_notes"), limit=8, max_len=180)

    terms, term_stats = clean_term_list_v2(terms_raw, source_text, limit=8)
    subtopics, subtopic_stats = clean_term_list_v2(subtopics_raw, source_text, limit=6)

    removed_generic = list(term_stats["removed_generic_terms"]) + list(subtopic_stats["removed_generic_terms"])
    removed_suspicious = list(term_stats["removed_suspicious_terms"]) + list(subtopic_stats["removed_suspicious_terms"])
    mixed = list(term_stats["mixed_language_artifacts"]) + list(subtopic_stats["mixed_language_artifacts"])
    weak = list(term_stats["weak_grounding_terms"]) + list(subtopic_stats["weak_grounding_terms"])
    original_count = len(terms_raw) + len(subtopics_raw)
    generic_ratio = len(removed_generic) / max(1, original_count)

    flags: list[str] = []
    if len(summary) < 80:
        flags.append("too_short_summary")
    if _is_title_echo(summary, section_title) or _is_title_echo(main_idea, section_title):
        flags.append("title_echo")
    if generic_ratio > 0.35:
        flags.append("too_many_generic_terms")
    if mixed:
        flags.append("mixed_language_artifact")
    if removed_suspicious:
        flags.append("suspicious_terms")
    if weak and len(weak) >= max(2, int(original_count * 0.35)):
        flags.append("weak_grounding")
    if len(terms) < 3:
        flags.append("too_few_terms")
    if len(subtopics) < 2:
        flags.append("too_few_subtopics")

    retry_flags = {
        "too_short_summary",
        "title_echo",
        "too_many_generic_terms",
        "mixed_language_artifact",
        "suspicious_terms",
        "weak_grounding",
        "too_few_terms",
        "too_few_subtopics",
    }
    cleaned_payload = {
        "summary": summary,
        "main_idea": main_idea,
        "terms": terms,
        "subtopics": subtopics,
        "bad_input_notes": bad_input_notes,
    }
    return SectionValidation(
        payload=cleaned_payload,
        quality_flags=list(dict.fromkeys(flags)),
        removed_generic_terms=list(dict.fromkeys(removed_generic)),
        removed_suspicious_terms=list(dict.fromkeys(removed_suspicious)),
        mixed_language_artifacts=list(dict.fromkeys(mixed)),
        weak_grounding_terms=list(dict.fromkeys(weak)),
        generic_terms_ratio=round(generic_ratio, 4),
        should_retry=bool(set(flags) & retry_flags),
    )


def semantic_problem_flags(flags: list[str] | None) -> bool:
    if not flags:
        return False
    blocking = {
        "fallback_section_analysis",
        "mixed_language_artifact",
        "suspicious_terms",
        "title_echo",
        "too_few_subtopics",
        "too_few_terms",
        "too_many_generic_terms",
        "too_short_summary",
        "weak_grounding",
    }
    return bool(set(flags) & blocking)


def fatal_flags_for_section(entry: dict[str, Any]) -> list[str]:
    flags = list(entry.get("flags") or [])
    fatal: list[str] = []
    deterministic_cleanup = bool(entry.get("deterministic_cleanup"))
    summary = clean_text(entry.get("summary", ""), max_len=800)
    section_title = clean_text(entry.get("section_title", ""), max_len=220)
    title_echo = "title_echo" in flags or _is_title_echo(summary, section_title)

    if "missing_section_analysis" in flags:
        fatal.append("missing_section_analysis")
    if "timeout" in flags:
        fatal.append("timeout")
    if "invalid_json" in flags and not deterministic_cleanup:
        fatal.append("invalid_json")
    if "fallback_section_analysis" in flags and not deterministic_cleanup:
        fatal.append("fallback_section_analysis")
    if title_echo and not deterministic_cleanup:
        fatal.append("title_echo")
    if "too_short_summary" in flags and title_echo and not deterministic_cleanup:
        fatal.append("too_short_summary")
    if "suspicious_terms" in flags:
        fatal.append("suspicious_terms")
    if "mixed_language_artifact" in flags and entry.get("mixed_language_artifacts"):
        fatal.append("mixed_language_artifact")
    if "weak_grounding" in flags and entry.get("weak_grounding_terms"):
        fatal.append("weak_grounding")
    return list(dict.fromkeys(fatal))


def warning_flags_for_section(entry: dict[str, Any]) -> list[str]:
    fatal = set(fatal_flags_for_section(entry))
    return [flag for flag in list(entry.get("flags") or []) if flag not in fatal]


def clean_chapter_terms(items: list[str], evidence: str, *, limit: int = 10) -> list[str]:
    cleaned, _ = clean_term_list_v2(items, evidence, limit=limit)
    return cleaned
