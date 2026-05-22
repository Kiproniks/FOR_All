# Intelligent Library (Python-only Web UI)

This project is now focused on Python-only runtime components:
- Django backend (API + server-rendered web UI)
- Celery background processing
- PostgreSQL
- Redis
- aiogram Telegram bot
- Optional Ollama for LLM mode

The React/Vite frontend is no longer required to run the project.

## What the system does

1. Uploads FB2/PDF books.
2. Validates XML and file size (max 50 MB).
3. Computes SHA-256 hash and reuses global cache for duplicate books.
4. Splits text into logical blocks.
5. Groups blocks into major themes.
6. Extracts 2-4 subtopics for each theme.
7. Stores concept mentions linked to source blocks.
8. Builds full book summary.
9. Supports concept search and concept comparison across user books.
10. Exports analysis to PDF/TXT/CSV/JSON.
11. Builds an interactive concept/subtheme map with filters and drill-down.

## Python-only web interface

Main pages are implemented as Django templates:
- `/login/`
- `/register/`
- `/library/`
- `/library/books/<book_id>/`
- `/library/books/<book_id>/blocks/<block_id>/`
- `/library/concepts/`
- `/library/concepts/<concept_id>/`
- `/library/concepts/<concept_id>/compare/`
- `/library/map/`

## Smart semantic split (chapter -> logical thoughts)

The pipeline now splits text in two levels:
1. Structural chapters from source (`FB2 section/title` or `PDF page` fallback).
2. Semantic logical blocks inside each chapter using:
   - paragraph embeddings,
   - cosine drop points between neighboring paragraphs,
   - transition cue words,
   - minimum/target/maximum block size constraints.

Tuning variables in `.env`:
- `BLOCK_MIN_WORDS`
- `BLOCK_TARGET_WORDS`
- `BLOCK_MAX_WORDS`

## Core models

- `GlobalBookCache`
- `UserBook`
- `LogicalBlock`
- `BookTheme`
- `ThemeSubtopic`
- `Concept`
- `ConceptMention`
- `UserConceptEdit`
- `BookSummary`

Legacy glossary models are kept only for migration compatibility.

## API (kept and working)

Auth:
- `POST /api/auth/register/`
- `POST /api/auth/login/`
- `POST /api/auth/logout/`
- `GET /api/auth/me/`

Books:
- `GET /api/books/`
- `POST /api/books/upload/`
- `GET /api/books/{id}/`
- `DELETE /api/books/{id}/`
- `POST /api/books/{id}/protect/`
- `POST /api/books/{id}/reanalyze/`
- `GET /api/books/{id}/summary/`
- `GET /api/books/{id}/blocks/`
- `GET /api/books/{id}/blocks/{block_id}/`
- `GET /api/books/{id}/concepts/`
- `GET /api/books/{id}/export/?format=pdf|txt|csv|json`

Concepts:
- `GET /api/concepts/`
- `GET /api/concepts/search/?q=...`
- `GET /api/concepts/{concept_id}/`
- `GET /api/concepts/{concept_id}/compare/`
- `PATCH /api/concepts/mentions/{mention_id}/edit/`
- `POST /api/concepts/mentions/{mention_id}/reset/`

Stats:
- `GET /api/stats/`

## Environment

Copy `.env.example` to `.env` and set values.

Important vars:
- `DATABASE_URL` or `POSTGRES_*`
- `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
- `LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `VECTOR_STORE`, `CHROMA_PATH`, `EMBEDDING_MODEL`
- `TELEGRAM_BOT_TOKEN`

## Run with local PostgreSQL/Redis

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

In a second terminal:

```powershell
cd backend
.\venv\Scripts\activate
celery -A config worker -l info --pool=solo
```

Open web UI:
- `http://127.0.0.1:8000/login/`

## Run DB/Redis with Docker (optional)

If Docker is installed:

```powershell
docker compose up -d postgres redis
```

## Ollama (optional)

```bash
ollama pull qwen2.5:1.5b
```

If Ollama is unavailable, fallback logic is used and the project still works.

## Telegram bot

```powershell
cd backend
.\venv\Scripts\activate
python -m apps.telegram_bot.bot
```

## Tests

```powershell
cd backend
.\venv\Scripts\activate
python manage.py test
```
