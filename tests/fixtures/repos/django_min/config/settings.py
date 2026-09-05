"""Project settings."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("SECRET_KEY", "dev")
DEBUG = os.environ.get("DEBUG", "1") == "1"

INSTALLED_APPS = [
    "django.contrib.auth",
    "rest_framework",
    "authx",
    "billing",
]

MIDDLEWARE = ["django.middleware.common.CommonMiddleware"]
AUTH_USER_MODEL = "authx.User"


def database_url() -> str:
    """Resolve the database URL, defaulting to local sqlite."""
    return os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR}/db.sqlite3")
