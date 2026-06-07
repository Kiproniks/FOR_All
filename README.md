# FOR_All / Book Semantic Analyzer

## Описание

FOR_All — учебный Django-проект для интеллектуального анализа книг. Система позволяет:

- загружать книги FB2/PDF;
- проверять размер и валидность файла;
- разбирать структуру книги на главы, секции и logical blocks;
- запускать production-safe LLM-анализ в режиме `llm_fast_batched`;
- строить Semantic Theme Map;
- открывать конкретные logical blocks и видеть исходный фрагмент текста;
- смотреть concepts/subtopics, связанные с блоками;
- формировать конспект всей книги из уже готовых результатов анализа;
- экспортировать результат в PDF/TXT/CSV/JSON.

Проект сейчас работает как Python/Django web-приложение с server-rendered UI. React/Vite для запуска текущей версии не требуется.

## Возможности

- Загрузка FB2/PDF книг через веб-интерфейс.
- Глобальный кэш анализа по SHA-256 книги.
- Батчевый LLM-анализ `llm_fast_batched` с checkpoint/resume.
- Structure detection: отделение основного текста от front matter/service sections.
- Logical blocks: смысловые фрагменты книги с source text.
- Themes/Subtopics/ConceptMentions: крупные темы, подтемы и понятия.
- Semantic Theme Map: граф книг, тем и подтем с переходом к source block.
- Book Notes: конспект всей книги на основе уже сохранённых summary/themes/blocks/concepts.
- Quality audit и статус `ready_with_warnings` для non-fatal проблем.
- Telegram bot scaffold через aiogram.

## Стек

- Python / Django
- Django REST Framework
- PostgreSQL
- Redis
- Celery
- Ollama
- qwen2.5:1.5b
- lxml, pypdf, razdel, pymorphy3
- ChromaDB / sentence-transformers для embedding/RAG-части

## Требования

- Python 3.12+ recommended. Проект также проверялся локально на Python 3.13.
- PostgreSQL 16 или совместимый.
- Redis 7 или совместимый.
- Ollama для LLM-анализа.
- Git.
- Windows PowerShell для команд из quick start.

## Быстрый запуск

Подробная инструкция для Windows находится в [QUICK_START_WINDOWS.md](./QUICK_START_WINDOWS.md).

Минимально:

```powershell
git clone https://github.com/Kiproniks/FOR_All.git
cd FOR_All\backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example ..\.env
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 127.0.0.1:8000
```

Открыть:

```text
http://127.0.0.1:8000/login/
```

## Основные команды

### PostgreSQL/Redis через Docker

В корне проекта:

```powershell
docker compose up -d postgres redis
```

Важно: в `docker-compose.yml` PostgreSQL проброшен на `15432`, Redis на `16379`.
Для Docker-варианта в `.env` удобно использовать:

```env
DATABASE_URL=postgres://smart_user:smart_password@127.0.0.1:15432/smart_library
REDIS_URL=redis://127.0.0.1:16379/0
CELERY_BROKER_URL=redis://127.0.0.1:16379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:16379/0
```

### Миграции

```powershell
cd backend
.\venv\Scripts\activate
python manage.py migrate
```

### Django web

```powershell
cd backend
.\venv\Scripts\activate
python manage.py runserver 127.0.0.1:8000
```

### Celery worker

```powershell
cd backend
.\venv\Scripts\activate
celery -A config worker -l info --pool=solo
```

На Windows `--pool=solo` обычно стабильнее.

### Ollama

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

Если Ollama не запущена автоматически:

```powershell
ollama serve
```

## Структура проекта

```text
backend/
  manage.py
  config/
    settings.py
    urls.py
    celery.py
  apps/
    accounts/       # custom user/auth
    books/          # models, API, analysis services, management commands
    webui/          # server-rendered UI templates/static
    telegram_bot/   # aiogram bot scaffold
  requirements.txt

docker-compose.yml
.env.example
README.md
QUICK_START_WINDOWS.md
```

Ключевые сервисы анализа находятся в `backend/apps/books/services/`:

- `book_parser.py`, `fb2_parser.py`, `pdf_parser.py`
- `structure_detector.py`
- `content_filter.py`
- `logical_block_splitter.py`
- `llm_service.py`
- `llm_hierarchical_pipeline.py`
- `theme_hierarchy.py`
- `concept_map.py`
- `semantic_quality_v2.py`
- `study_notes.py`

Management commands находятся в `backend/apps/books/management/commands/`.

## Важные URL

- `/login/` — вход.
- `/register/` — регистрация.
- `/library/` — библиотека пользователя.
- `/library/map/` — Semantic Theme Map.
- `/library/books/<id>/` — страница книги.
- `/library/books/<id>/notes/` — конспект всей книги.
- `/library/books/<id>/blocks/<block_id>/` — конкретный logical block.
- `/library/concepts/` — общий список concepts.

## LLM-анализ

Основной production-safe режим: `llm_fast_batched`.

Он работает батчами, сохраняет промежуточные результаты в БД, умеет делать semantic audit и не должен запускаться одним огромным запросом.

### Полный batched-анализ книги

```powershell
cd backend
.\venv\Scripts\activate
python manage.py run_llm_fast_batched_analysis --file "C:\path\book.fb2" --batch-size 5
```

### Fresh run без использования старого LLM-кэша

```powershell
python manage.py run_llm_fast_batched_analysis --file "C:\path\book.fb2" --batch-size 5 --force-llm-refresh
```

### Semantic audit без нового анализа

```powershell
python manage.py run_llm_fast_batched_analysis --file "C:\path\book.fb2" --semantic-audit-only
```

### Repair fatal blocks only

```powershell
python manage.py run_llm_fast_batched_analysis --file "C:\path\book.fb2" --reanalyze-problem-blocks-only --fatal-only
```

### Segmentation mini-test без full analysis

```powershell
python manage.py test_book_segmentation --file "C:\path\book.fb2" --limit-main-sections 3 --output segmentation_mini_report
```

### Debug structure без LLM

```powershell
python manage.py analyze_book_debug --file "C:\path\book.fb2" --limit-sections 10 --mode debug_structure --show-quality --show-filtered --no-llm
```

## Статусы книги

- `ready` — анализ готов.
- `ready_with_warnings` — анализ готов, есть non-fatal warnings.
- `partial_ready` — частичный результат, fallback/timeout были существенными.
- `processing` / `llm_fast_batched_*` — анализ выполняется.
- `failed` / `failed_timeout` — анализ завершился ошибкой.

## Troubleshooting

### PostgreSQL connection refused

Проверьте, что PostgreSQL запущен и `.env` указывает правильный порт. Для Docker compose порт `15432`, не `5432`.

### Redis connection error

Проверьте Redis. Для Docker compose порт `16379`, не `6379`.

### Ollama model not found

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

### no such table

Выполните миграции:

```powershell
python manage.py migrate
```

### static files not loading

Для dev-сервера проверьте:

```powershell
python manage.py findstatic webui/css/library_ui.css
```

### port already in use

Найдите процесс:

```powershell
Get-NetTCPConnection -LocalPort 8000
```

### test database permission denied

Это проблема прав PostgreSQL на создание test database. Код менять не нужно: выдайте пользователю CREATEDB или запускайте тесты с пользователем, у которого есть такие права.

## Что не входит в репозиторий

- `.env` и любые реальные секреты.
- `venv/`.
- `backend/media/`.
- локальные базы данных и dumps.
- загруженные книги `.fb2/.pdf/.epub/.djvu`.
- LLM/audit/debug отчёты.
- `postgres_data/`, `run_logs/`, cache folders.

## Thought-chain analysis mode

Default book analysis mode is now `llm_thought_chain`.

This mode is intentionally sentence/thought based, not chapter-only:

1. The parsed book is split into source-aware sentences.
2. LLM extracts one main thought from each sentence.
3. The next thought is compared with the whole accumulated `current_block`, not only with the previous sentence.
4. If `same_block=true` and score is above `THOUGHT_SAME_BLOCK_THRESHOLD`, the thought is appended and the block main idea is updated.
5. If the new thought does not fit, the current sequential group is closed and a new group starts from the new thought.
6. Meaningful thoughts are compared pairwise by LLM and stored as `ThoughtRelation`.
7. Related thoughts are grouped into `GlobalLogicalThoughtBlock`; every thought receives a `ThoughtBlockMembership.relevance_score`.
8. New books compare their thoughts with existing global thought blocks before creating new blocks.

Legacy mode `llm_fast_batched` is kept and can still be launched manually, but it is no longer the default.

Dry-run without DB writes:

```powershell
cd backend
.\venv\Scripts\python.exe manage.py run_thought_chain_analysis --file "C:\Users\1\Desktop\НИКОЛАЕВ\Компьютерныее сети.fb2" --max-sentences 30 --dry-run --max-pairs 60
```

Persistent limited run:

```powershell
cd backend
.\venv\Scripts\python.exe manage.py run_thought_chain_analysis --file "C:\Users\1\Desktop\НИКОЛАЕВ\Компьютерныее сети.fb2" --max-sentences 30 --max-pairs 60 --force-refresh
```

Resume an existing book run:

```powershell
cd backend
.\venv\Scripts\python.exe manage.py run_thought_chain_analysis --book-id 10 --resume
```

Required Ollama check:

```powershell
ollama list
ollama pull qwen2.5:1.5b
```

Main environment variables:

```env
BOOK_ANALYSIS_MODE=llm_thought_chain
THOUGHT_SAME_BLOCK_THRESHOLD=0.65
THOUGHT_RELATION_THRESHOLD=0.65
THOUGHT_BLOCK_MEMBERSHIP_THRESHOLD=0.70
THOUGHT_CHAIN_DRY_RUN_MAX_PAIRS=60
```

---

## LLM Thought Chain: Greedy And Strict Modes

The current book-analysis pipeline includes a separate `llm_thought_chain` mode. It is designed to turn a book into a sequence of grounded thoughts and then into logical thought blocks.

### What The Project Does

The application lets a user upload FB2/PDF books, parse their text, split the text into meaningful units, run local LLM analysis through Ollama, store the result in Django/PostgreSQL, and inspect books, summaries, semantic maps, logical blocks, themes, concepts, and source text in the web UI.

### Sentence -> Thought -> Block

`llm_thought_chain` works in three main steps:

1. The book is parsed and split into sentences.
2. Each sentence is sent to the LLM and converted into a short grounded thought.
3. Thoughts are grouped into logical blocks.

The sequential step compares a new thought with the accumulated current block, not only with the previous sentence. If the new thought belongs to the current block, it is added and the block idea is updated. If not, the current block is closed and a new block starts.

### Strict Pairwise Mode

Strict mode is intended for small checks and teacher demonstrations. It compares every thought with every other thought through the LLM.

Use it only for small books or limited tests:

```powershell
.\run_all_thought_chain_books.ps1 -StrictPairwise
```

### Greedy Mode

Greedy mode is the default production mode for full runs. It is faster and avoids full pairwise explosion.

The logic is:

1. Take the first unused thought as a seed.
2. Compare each remaining candidate with the accumulated current block.
3. If the candidate belongs to the block, add it to that block.
4. Once a thought enters a block, remove it from future seed selection.
5. Continue from the next unused thought.
6. Merge blocks with identical or similar titles/main ideas.

Default full run:

```powershell
.\run_all_thought_chain_books.ps1
```

Default command mode used by the script:

```text
--full --strict --mode greedy --merge-same-title-blocks --resume
```

### Same-title Block Merge

After greedy block creation, the system merges blocks with identical or similar titles/main ideas. The old block is not lost: it is marked as merged and points to the surviving block through `merged_into`. Memberships are moved to the surviving block.

### Demo Files

The small deterministic demo book is committed here:

```text
test_books/demo/test.fb2
```

The already existing processed demo result is committed here:

```text
demo_results/test_fb2_greedy/
```

The demo analysis was not re-run during packaging. The result snapshot contains:

```text
thought_chain_analysis_report.md
thought_chain_analysis_report.json
quality_report.md
quality_report.json
SUMMARY.md
```

### Dry Plan

To see what would be processed without running analysis or writing DB data:

```powershell
.\run_all_thought_chain_books.ps1 -DryPlan
```

### Full Greedy Run For Local Books

Local real books are expected in:

```text
test_books/new/
```

Run all books through greedy mode:

```powershell
.\run_all_thought_chain_books.ps1
```

Results are written to:

```text
test_runs/thought_chain/full_auto/
```

### What Is Not Committed

The repository intentionally excludes real books and runtime artifacts:

- `test_books/new/`
- `test_books/old/`
- `.env`
- virtual environments
- PostgreSQL data
- media uploads
- runtime logs
- large generated reports under `test_runs/`
