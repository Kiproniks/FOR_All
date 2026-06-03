# Быстрый запуск на Windows

Эта инструкция рассчитана на запуск проекта с нуля после клонирования репозитория.

## 1. Клонировать репозиторий

```powershell
git clone https://github.com/Kiproniks/FOR_All.git
cd FOR_All
```

## 2. Создать виртуальное окружение

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
```

Если PowerShell запрещает активацию скриптов:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\venv\Scripts\activate
```

## 3. Установить зависимости

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Подготовить PostgreSQL и Redis

### Вариант A: через Docker compose

Из корня проекта:

```powershell
cd ..
docker compose up -d postgres redis
cd backend
```

В этом варианте порты такие:

```text
PostgreSQL: 127.0.0.1:15432
Redis:      127.0.0.1:16379
```

### Вариант B: локально установленный PostgreSQL/Redis

Создайте базу и пользователя вручную. Пример:

```text
database: smart_library
user: smart_user
password: smart_password
host: 127.0.0.1
port: 5432
```

## 5. Создать `.env`

Из корня проекта:

```powershell
copy .env.example .env
```

Если используете Docker compose, проверьте в `.env`:

```env
DATABASE_URL=postgres://smart_user:smart_password@127.0.0.1:15432/smart_library
REDIS_URL=redis://127.0.0.1:16379/0
CELERY_BROKER_URL=redis://127.0.0.1:16379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:16379/0
```

Если используете локальный PostgreSQL на стандартном порту, можно использовать:

```env
DATABASE_URL=postgres://smart_user:smart_password@127.0.0.1:5432/smart_library
REDIS_URL=redis://127.0.0.1:6379/0
CELERY_BROKER_URL=redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0
```

## 6. Применить миграции

```powershell
cd backend
.\venv\Scripts\activate
python manage.py migrate
```

## 7. Создать пользователя

```powershell
python manage.py createsuperuser
```

Или зарегистрируйтесь через `/register/`, если регистрация открыта в UI.

## 8. Установить и запустить Ollama

Установите Ollama с официального сайта, затем скачайте модель:

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

Если сервер Ollama не стартует автоматически:

```powershell
ollama serve
```

## 9. Запустить Celery worker

Откройте отдельный PowerShell:

```powershell
cd C:\path\to\FOR_All\backend
.\venv\Scripts\activate
celery -A config worker -l info --pool=solo
```

На Windows параметр `--pool=solo` снижает риск проблем с multiprocessing.

## 10. Запустить Django

В отдельном PowerShell:

```powershell
cd C:\path\to\FOR_All\backend
.\venv\Scripts\activate
python manage.py runserver 127.0.0.1:8000
```

Открыть:

```text
http://127.0.0.1:8000/login/
```

## 11. Как пользоваться

1. Войти или зарегистрироваться.
2. Открыть `/library/`.
3. Загрузить FB2/PDF книгу.
4. Дождаться фонового анализа или запустить batched-анализ management command.
5. Открыть `/library/map/` для Semantic Theme Map.
6. Нажать на тему/subtopic и открыть source logical block.
7. Открыть `Конспект всей книги` на странице книги.

## 12. Команды анализа

### Structure/debug без LLM

```powershell
python manage.py analyze_book_debug --file "C:\path\book.fb2" --limit-sections 10 --mode debug_structure --show-quality --show-filtered --no-llm
```

### Mini-test сегментации

```powershell
python manage.py test_book_segmentation --file "C:\path\book.fb2" --limit-main-sections 3 --output segmentation_mini_report
```

### Section-level LLM preview

```powershell
python manage.py run_llm_section_preview --file "C:\path\book.fb2" --limit-main-sections 2 --output llm_section_preview_report
```

### Limited fast LLM run без записи full результата

```powershell
python manage.py run_limited_llm_full --file "C:\path\book.fb2" --sections 5 --mode fast --output limited_llm_full_report
```

### Production-safe batched full run

```powershell
python manage.py run_llm_fast_batched_analysis --file "C:\path\book.fb2" --batch-size 5
```

### Fresh run без старого LLM-кэша

```powershell
python manage.py run_llm_fast_batched_analysis --file "C:\path\book.fb2" --batch-size 5 --force-llm-refresh
```

### Semantic audit only

```powershell
python manage.py run_llm_fast_batched_analysis --file "C:\path\book.fb2" --semantic-audit-only
```

## 13. Проверка проекта

```powershell
python manage.py check
python manage.py findstatic webui/css/library_ui.css
```

Если есть права на создание test database:

```powershell
python manage.py test
```

## 14. Частые ошибки

### `connection refused PostgreSQL`

Проверьте порт и `DATABASE_URL`. Для Docker compose нужен `15432`.

### `Redis connection error`

Проверьте `REDIS_URL`. Для Docker compose нужен `16379`.

### `Ollama model not found`

```powershell
ollama pull qwen2.5:1.5b
```

### `no such table`

```powershell
python manage.py migrate
```

### CSS не загружается

```powershell
python manage.py findstatic webui/css/library_ui.css
```

### `port already in use`

```powershell
Get-NetTCPConnection -LocalPort 8000
```

### Тесты падают из-за PostgreSQL test database

Нужно выдать PostgreSQL-пользователю право `CREATEDB` или запускать тесты от пользователя с такими правами.
