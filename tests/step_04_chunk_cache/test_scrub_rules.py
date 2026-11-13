"""Secret scrubbing: the shapes it missed, and the fields it never looked at.

Two gaps, both reproduced before the fix:

* Common secrets passed straight through: a Django `SECRET_KEY` with symbols
  in it, a database URL with a password, a `DB_PASSWORD` literal, and any
  `sk-proj-`/`sk-ant-` key that was not inside quotes.
* Only `source_code` and `docstring` were scrubbed. A secret in a default
  argument or a decorator also lives in `signature` and `decorators`, which go
  into the chunk header, the embedding text, and the Neo4j node unredacted.
"""

from __future__ import annotations

import pytest

from adapters.python import PythonAdapter
from graph.writer import symbol_props
from index.chunker import REDACTED, chunk_file, scrub_secrets
from index.providers import ApproxCodeTokenizer
from retrieve.tokenize import build_search_text

SECRETS = {
    "django_secret_key": ('SECRET_KEY = "django-insecure-7#k!x2@vq_0&m3y^p8z(w)r+u1t=e5s*a9d"',
                          "7#k!x2@vq_0&m3y"),
    "django_default": ('SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-7#k!x2@vq_0&m3y^p8z")',
                       "7#k!x2@vq_0&m3y"),
    "openai_quoted": ('KEY = "sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4zAb7cD0eF3gH6iJ9k"', "Ab3dEf6hIj9k"),
    "openai_unquoted": ("# rotate sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4zAb7cD0eF3gH6iJ9k monthly",
                        "Ab3dEf6hIj9k"),
    "anthropic_unquoted": ("export ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf1234567890AbCdEf1234567890-XyZ",
                           "AbCdEf1234567890"),
    "database_url": ('DATABASE_URL = "postgres://user:pass@host/db"', "user:pass@"),
    "database_url_long": ("redis://default:Sup3rS3cret!2024@cache:6379/0", "Sup3rS3cret"),
    "db_password": ('DB_PASSWORD = "Sup3rS3cret!2024"', "Sup3rS3cret"),
    "password_kwarg": ('connect(password="Sup3rS3cret!2024")', "Sup3rS3cret"),
    "settings_dict": ("{'PASSWORD': 'hunter2-Prod!', 'HOST': 'db'}", "hunter2-Prod"),
}


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_common_secrets_are_redacted(name):
    text, secret_part = SECRETS[name]
    scrubbed = scrub_secrets(text)
    assert secret_part not in scrubbed, scrubbed
    assert REDACTED in scrubbed


def test_a_url_keeps_everything_but_the_password():
    scrubbed = scrub_secrets('DATABASE_URL = "postgres://user:pass@host/db"')
    assert scrubbed.startswith('DATABASE_URL = "postgres://user:')
    assert scrubbed.endswith('@host/db"')


@pytest.mark.parametrize(
    "code",
    [
        'password = request.POST["password"]',
        'PASSWORD_FIELD = "password"',
        'TOKEN_URL = "/api/token/refresh/"',
        'token_type = "Bearer"',
        'api_key = os.environ["API_KEY"]',
        'AUTH_PASSWORD_VALIDATORS = [{"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"}]',
        'DATABASE_URL = "postgres://localhost:5432/app"',
        'url = "https://example.com/docs/tokens"',
        "def check_password(raw_password, encoded):",
        'SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]',
    ],
)
def test_code_that_merely_mentions_a_secret_is_left_alone(code):
    """A rule that fires on ordinary code costs retrieval signal for nothing."""
    assert scrub_secrets(code) == code


SOURCE = b'''import requests


@retry(token="ghp_abcdefghijklmnopqrstuvwxyz0123456789")
def connect(dsn="postgres://admin:Sup3rS3cret!2024@db/prod", password="Sup3rS3cret!2024"):
    """Open a connection."""
    return requests.get(dsn)


class Client(Base, api_key="sk-ant-api03-AbCdEf1234567890AbCdEf1234567890-XyZ"):
    def ping(self):
        return 1
'''


@pytest.fixture
def chunked():
    parsed = PythonAdapter().parse("repo_scrub", "svc.py", SOURCE)
    chunk_file(parsed, ApproxCodeTokenizer())
    for sym in parsed.symbols:
        sym.search_text = build_search_text(sym)
    return {s.qualified_name: s for s in parsed.symbols}


LEAKS = ("Sup3rS3cret", "ghp_abcdefghij", "AbCdEf1234567890")


def leaked(value) -> list[str]:
    text = " ".join(value) if isinstance(value, list) else str(value)
    return [leak for leak in LEAKS if leak in text]


def test_no_field_that_is_stored_or_embedded_holds_a_secret(chunked):
    for qualified_name, sym in chunked.items():
        for field, value in symbol_props(sym).items():
            if field == "code_vec":
                continue
            assert not leaked(value), f"{qualified_name}.{field} leaks {leaked(value)}"


def test_the_header_and_embedding_text_are_scrubbed(chunked):
    """`chunk_text` is what is embedded; the header inside it carries the
    method's enclosing class signature."""
    ping = chunked["svc.Client.ping"]
    assert ping.enclosing_signature and not leaked(ping.enclosing_signature)
    assert not leaked(ping.chunk_text)
    assert not leaked(chunked["svc.connect"].signature)
    assert not leaked(chunked["svc.connect"].decorators)
