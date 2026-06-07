import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [host for host in os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if host]


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework.authtoken",
    "corsheaders",
    "apps.accounts",
    "apps.books",
    "apps.telegram_bot",
    "apps.webui",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "config.wsgi.application"


USE_SQLITE = env_bool("USE_SQLITE", False)
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if DATABASE_URL and not USE_SQLITE:
    parsed = urlparse(DATABASE_URL)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": parsed.path.lstrip("/") or "smart_library",
            "USER": parsed.username or "smart_user",
            "PASSWORD": parsed.password or "smart_password",
            "HOST": parsed.hostname or "localhost",
            "PORT": str(parsed.port or "5432"),
        }
    }
elif USE_SQLITE:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "smart_library"),
            "USER": os.getenv("POSTGRES_USER", "smart_user"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", "smart_password"),
            "HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
        }
    }


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


LANGUAGE_CODE = "ru-ru"
TIME_ZONE = os.getenv("TIME_ZONE", "UTC")

USE_I18N = True
USE_TZ = True


STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
AUTH_USER_MODEL = "accounts.User"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.TokenAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
    ),
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": REDIS_URL,
    }
}

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", False)
CELERY_TASK_EAGER_PROPAGATES = True

MAX_FB2_FILE_SIZE = 50 * 1024 * 1024
DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", "noreply@local.local")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_MODEL_HIGH = os.getenv("OLLAMA_MODEL_HIGH", OLLAMA_MODEL)
OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", OLLAMA_MODEL)
OLLAMA_MODEL_FALLBACK = os.getenv("OLLAMA_MODEL_FALLBACK", OLLAMA_MODEL_FAST)
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
LLM_ENABLE_FALLBACK = env_bool("LLM_ENABLE_FALLBACK", True)
LLM_MAX_CALLS_PER_BOOK = int(os.getenv("LLM_MAX_CALLS_PER_BOOK", "220"))
LLM_MAX_CALLS_PER_CHAPTER = int(os.getenv("LLM_MAX_CALLS_PER_CHAPTER", "40"))
LLM_MAX_CHUNKS_PER_SECTION = int(os.getenv("LLM_MAX_CHUNKS_PER_SECTION", "4"))
LLM_MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "8000"))
LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "2592000"))
BOOK_ANALYSIS_MODE = os.getenv("BOOK_ANALYSIS_MODE", "llm_thought_chain")
THOUGHT_SAME_BLOCK_THRESHOLD = float(os.getenv("THOUGHT_SAME_BLOCK_THRESHOLD", "0.65"))
THOUGHT_RELATION_THRESHOLD = float(os.getenv("THOUGHT_RELATION_THRESHOLD", "0.65"))
THOUGHT_BLOCK_MEMBERSHIP_THRESHOLD = float(os.getenv("THOUGHT_BLOCK_MEMBERSHIP_THRESHOLD", "0.70"))
THOUGHT_CHAIN_DRY_RUN_MAX_PAIRS = int(os.getenv("THOUGHT_CHAIN_DRY_RUN_MAX_PAIRS", "60"))
THOUGHT_CHAIN_EXISTING_BLOCK_LIMIT = int(os.getenv("THOUGHT_CHAIN_EXISTING_BLOCK_LIMIT", "30"))
BOOK_STUCK_TIMEOUT_MINUTES = int(os.getenv("BOOK_STUCK_TIMEOUT_MINUTES", "90"))
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Semantic logical-block splitter tuning.
BLOCK_MIN_WORDS = int(os.getenv("BLOCK_MIN_WORDS", "260"))
BLOCK_TARGET_WORDS = int(os.getenv("BLOCK_TARGET_WORDS", "760"))
BLOCK_MAX_WORDS = int(os.getenv("BLOCK_MAX_WORDS", "1300"))
