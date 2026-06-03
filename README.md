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
