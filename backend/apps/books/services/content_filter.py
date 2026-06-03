from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


ISBN_RE = re.compile(r"\bISBN\b", re.IGNORECASE)
COPYRIGHT_RE = re.compile(
    r"(?:©|copyright|all rights reserved|все права защищены|издательств|изд-во|переводч|тираж|подписано в печать)",
    re.IGNORECASE,
)
DEDICATION_RE = re.compile(r"^(?:посвящ[ае]тся|dedicated to)\b", re.IGNORECASE)
ACK_RE = re.compile(r"(?:благодарност|acknowledg|thanks to)", re.IGNORECASE)
BIB_RE = re.compile(r"(?:литератур[аы]|references|bibliography|список литературы)", re.IGNORECASE)
INDEX_RE = re.compile(r"(?:предметный указатель|index)\b", re.IGNORECASE)
TOC_RE = re.compile(r"(?:содержание|оглавление|table of contents)\b", re.IGNORECASE)
ACRONYM_HEADER_RE = re.compile(r"(?:список сокращений|abbreviations|acronyms)\b", re.IGNORECASE)
FRONT_MATTER_TITLE_RE = re.compile(
    r"(?:переводчик|об авторах|от издательства|благодарност|acknowledg|about the author|foreword|предисловие|посвящени|copyright|all rights reserved|список аббревиатур|материалы для студентов)",
    re.IGNORECASE,
)
EDITORIAL_ROLE_RE = re.compile(
    r"(?:переводчик|редактор|научн[а-я]* редакц|корректор|верстк|издательств|publisher|editor|translated by|translation)",
    re.IGNORECASE,
)
FIGURE_RE = re.compile(r"^(?:рис\.?|илл\.?|figure\s+\d+|fig\.?\s*\d+)", re.IGNORECASE)
TABLE_RE = re.compile(r"^(?:табл\.?|таблица\s+\d+|table\s+\d+)", re.IGNORECASE)
QUESTION_RE = re.compile(
    r"^(?:\d+[.)]\s*)?(?:докажите|объясните|рассмотрите|чему равна|почему|какие|укажите|найдите|определите|каков|какая|какой)",
    re.IGNORECASE,
)
CODE_RE = re.compile(
    r"(?:#include\b|#define\b|typedef\b|struct\b|int\s+main\b|void\b|class\s+\w+|def\s+\w+|import\s+\w+|if\s*\(|while\s*\(|for\s*\(|case\s+\w+\s*:)",
    re.IGNORECASE,
)
FORMULA_RE = re.compile(r"(?:[=<>±∑∫√]|\b\w+\s*=\s*\w+|\d+\s*[+\-*/]\s*\d+)")
HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,4}[.)]?|глава\s+\d+|часть\s+[ivxlcdm\d]+|раздел\s+\d+|chapter\s+\d+|part\s+[ivxlcdm\d]+|section\s+\d+)\b",
    re.IGNORECASE,
)
GENERIC_QUESTION_MARK_RE = re.compile(r"\?\s*$")
DIALOGUE_RE = re.compile(r"^[—\-]\s+|^«.+»$")
ABBR_TOKEN_RE = re.compile(r"\b[A-ZА-ЯЁ]{2,8}\b")
NAME_LINE_RE = re.compile(r"^(?:[A-ZА-ЯЁ][a-zа-яё'-]+(?:\s+[A-ZА-ЯЁ]\.){0,2}\s*){2,}$")
NAME_TOKEN_RE = re.compile(r"^(?:[A-ZА-ЯЁ][a-zа-яё'-]+|[A-ZА-ЯЁ]\.)$")


CONTENT_TYPES = {
    "main_text",
    "title",
    "subtitle",
    "dialogue",
    "definition",
    "example",
    "formula",
    "code",
    "figure_caption",
    "table_caption",
    "exercise",
    "question",
    "copyright",
    "dedication",
    "acknowledgements",
    "bibliography",
    "acronym_list",
    "index",
    "toc",
    "empty_or_noise",
}


@dataclass(slots=True)
class ClassifiedParagraph:
    text: str
    paragraph_index: int
    content_type: str
    chapter_title: str
    section_title: str
    metadata: dict[str, Any]


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _is_noise(text: str) -> bool:
    if not text:
        return True
    if len(text) < 2:
        return True
    if re.fullmatch(r"[\W_]+", text):
        return True
    return False


def _looks_like_subtitle(text: str) -> bool:
    if not text or len(text) > 180:
        return False
    if text.endswith(":") and len(text.split()) <= 12:
        return True
    if HEADING_RE.match(text):
        return True
    if text.istitle() and len(text.split()) <= 10 and not re.search(r"[.!?]", text):
        return True
    return False


def _is_acronym_list(text: str) -> bool:
    parts = [chunk.strip() for chunk in re.split(r"[,;]", text) if chunk.strip()]
    if len(parts) < 4:
        return False
    abbr_parts = 0
    for part in parts[:15]:
        token = part.split()[0]
        if ABBR_TOKEN_RE.fullmatch(token):
            abbr_parts += 1
    return abbr_parts >= max(3, int(len(parts) * 0.5))


def _looks_like_person_name_cluster(text: str) -> bool:
    clean = _normalize_text(text)
    if not clean:
        return False
    if len(clean) > 180:
        return False
    if re.search(r"\d", clean):
        return False

    tokens = re.findall(r"[A-Za-zА-Яа-яЁё]+\.?", clean)
    if len(tokens) < 2:
        return False

    proper_like = sum(1 for token in tokens if NAME_TOKEN_RE.fullmatch(token))
    ratio = proper_like / max(1, len(tokens))
    if ratio < 0.72:
        return False

    # Typical title-page author lists are comma-separated or short compact lines.
    return clean.count(",") >= 1 or len(tokens) <= 8


def is_front_matter_title(title: str) -> bool:
    clean = _normalize_text(title)
    if not clean:
        return False
    if FRONT_MATTER_TITLE_RE.search(clean):
        return True
    return _looks_like_person_name_cluster(clean)


def classify_paragraph(
    text: str,
    *,
    chapter_title: str = "",
    section_title: str = "",
) -> str:
    clean = _normalize_text(text)
    low = clean.lower()

    if _is_noise(clean):
        return "empty_or_noise"
    if ISBN_RE.search(clean) or COPYRIGHT_RE.search(clean):
        return "copyright"
    if EDITORIAL_ROLE_RE.search(clean):
        return "copyright"
    if DEDICATION_RE.search(low) or (len(clean.split()) <= 8 and low.startswith("посв")):
        return "dedication"
    if NAME_LINE_RE.match(clean) and len(clean.split()) <= 8:
        return "dedication"
    if _looks_like_person_name_cluster(clean):
        return "dedication"
    if ACK_RE.search(low):
        return "acknowledgements"
    if TOC_RE.search(low):
        return "toc"
    if BIB_RE.search(low):
        return "bibliography"
    if INDEX_RE.search(low):
        return "index"
    if ACRONYM_HEADER_RE.search(low) or _is_acronym_list(clean):
        return "acronym_list"
    if FIGURE_RE.match(clean):
        return "figure_caption"
    if TABLE_RE.match(clean):
        return "table_caption"
    if CODE_RE.search(clean) or clean.count("{") >= 1 or clean.count(";") >= 2:
        return "code"
    if FORMULA_RE.search(clean) and len(clean.split()) <= 30:
        return "formula"
    if QUESTION_RE.match(clean):
        return "exercise"
    if GENERIC_QUESTION_MARK_RE.search(clean) and len(clean.split()) <= 24:
        return "question"
    if DIALOGUE_RE.match(clean):
        return "dialogue"
    if _looks_like_subtitle(clean):
        return "subtitle"

    low_start = low[:120]
    if " — это " in low or " называется " in low_start or low_start.startswith("под ") and "понимается" in low:
        return "definition"
    if low.startswith("пример") or low.startswith("example"):
        return "example"

    return "main_text"


def _keep_types_for_mode(mode: str, *, allow_dialogue: bool) -> set[str]:
    base = {
        "main_text",
        "title",
        "subtitle",
        "definition",
    }
    if mode == "summary":
        keep = set(base) | {"example"}
    elif mode == "themes":
        keep = set(base)
    elif mode == "concepts":
        keep = set(base) | {"example"}
    elif mode == "blocks":
        keep = set(base) | {"example", "formula", "dialogue", "code"}
    else:
        keep = set(base)

    if allow_dialogue:
        keep.add("dialogue")
    return keep


def filter_content_for_analysis(
    paragraphs: list[dict[str, Any]] | list[str],
    *,
    mode: str = "summary",
    chapter_title: str = "",
    section_title: str = "",
    allow_dialogue: bool = True,
) -> dict[str, Any]:
    """
    Classify and filter paragraphs for a target analysis mode.

    Returns:
    {
      "classified": list[ClassifiedParagraph as dict],
      "kept": list[ClassifiedParagraph as dict],
      "stats": {...}
    }
    """

    keep_types = _keep_types_for_mode(mode, allow_dialogue=allow_dialogue)
    front_matter = is_front_matter_title(chapter_title) or is_front_matter_title(section_title)
    classified_rows: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []

    counter: dict[str, int] = {key: 0 for key in CONTENT_TYPES}

    for idx, raw in enumerate(paragraphs, start=1):
        if isinstance(raw, dict):
            text = _normalize_text(str(raw.get("text", "")))
            paragraph_index = int(raw.get("paragraph_index") or idx)
            row_meta = {k: v for k, v in raw.items() if k != "text"}
        else:
            text = _normalize_text(str(raw))
            paragraph_index = idx
            row_meta = {}

        content_type = classify_paragraph(text, chapter_title=chapter_title, section_title=section_title)
        if front_matter and mode in {"summary", "themes", "concepts"} and content_type in {"main_text", "subtitle", "example"}:
            content_type = "acknowledgements"
        if content_type not in CONTENT_TYPES:
            content_type = "empty_or_noise"

        counter[content_type] += 1

        row = {
            "text": text,
            "paragraph_index": paragraph_index,
            "content_type": content_type,
            "chapter_title": chapter_title,
            "section_title": section_title,
            "metadata": row_meta,
            "keep": content_type in keep_types,
        }
        classified_rows.append(row)

        if row["keep"] and text:
            kept_rows.append(row)

    total = len(classified_rows)
    kept = len(kept_rows)
    removed = total - kept

    stats = {
        "total_paragraphs": total,
        "kept_count": kept,
        "removed_count": removed,
        "main_text_count": counter["main_text"],
        "titles_count": counter["title"] + counter["subtitle"],
        "code_count": counter["code"],
        "exercises_count": counter["exercise"] + counter["question"],
        "captions_count": counter["figure_caption"] + counter["table_caption"],
        "copyright_count": counter["copyright"],
        "acronym_list_count": counter["acronym_list"],
        "noise_count": counter["empty_or_noise"],
        "by_type": counter,
    }

    return {
        "classified": classified_rows,
        "kept": kept_rows,
        "stats": stats,
    }
