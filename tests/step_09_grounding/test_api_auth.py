"""The HTTP API answers only the person who started it.

Before: no authentication on any route, and no check of the `Host` header. A
web page open in the same browser could resolve its own domain to 127.0.0.1
(DNS rebinding) and read `/ask` — that is, the repository's source code —
and `--host 0.0.0.0` offered the same to the whole network.

Now `api.cli` serves every app with a per-run token, required on every route
but `/healthz`, and on a local bind accepts only a localhost `Host` header. A
non-local bind without a token chosen by the user is refused outright.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import InvestigationService, create_app
from api.cli import LOCAL_HOSTS, serve_security
from index.providers import ApproxCodeTokenizer, ScriptedLLM

TOKEN = "test-token-0123456789"


def app(**kwargs):
    service = InvestigationService(retriever=lambda q, r: ([], []), llm=ScriptedLLM("x"),
                                   tokenizer=ApproxCodeTokenizer())
    return create_app(service, **kwargs)


@pytest.fixture
def client():
    with TestClient(app(token=TOKEN, allowed_hosts=LOCAL_HOSTS),
                    base_url="http://127.0.0.1:8000") as c:
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.mark.parametrize("method, path", [
    ("get", "/status"), ("post", "/ask"), ("post", "/index/sync"),
    ("post", "/index/rename"), ("get", "/"),
])
def test_every_route_but_healthz_needs_the_token(client, method, path):
    reply = client.get(path) if method == "get" else client.post(path, json={})
    assert reply.status_code == 401


def test_healthz_needs_no_token(client):
    assert client.get("/healthz").status_code == 200


def test_the_token_opens_the_routes(client):
    assert client.get("/status", headers=AUTH).status_code == 200
    reply = client.post("/ask", json={"question": "what calls foo"}, headers=AUTH)
    assert reply.status_code == 200 and '"type": "done"' in reply.text


def test_a_wrong_token_is_refused(client):
    assert client.get("/status", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_the_console_page_takes_the_token_from_its_link(client):
    """The browser is opened at `/?token=…`; the page then sends the header."""
    page = client.get(f"/?token={TOKEN}")
    assert page.status_code == 200 and "Authorization" in page.text


def test_a_token_in_the_query_opens_only_the_page(client):
    """Query strings land in logs and history; the API wants the header."""
    assert client.get(f"/status?token={TOKEN}").status_code == 401


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "192.168.1.20:8000"])
def test_a_foreign_host_header_is_refused_even_with_the_token(client, host):
    """DNS rebinding arrives with the attacker's hostname in `Host`."""
    assert client.get("/status", headers={**AUTH, "Host": host}).status_code == 403


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000", "localhost"])
def test_localhost_names_are_accepted(client, host):
    assert client.get("/status", headers={**AUTH, "Host": host}).status_code == 200


# -- how the CLI configures it -------------------------------------------------------


def test_a_local_bind_gets_a_fresh_token_and_the_host_check():
    token, hosts = serve_security("127.0.0.1", env_token=None)
    again, _ = serve_security("127.0.0.1", env_token=None)
    assert len(token) >= 32 and token != again
    assert hosts == LOCAL_HOSTS


def test_a_token_from_the_environment_is_used():
    assert serve_security("127.0.0.1", env_token="from-env-0123456789abcdef")[0] == "from-env-0123456789abcdef"


def test_a_network_bind_without_a_chosen_token_is_refused():
    with pytest.raises(SystemExit) as stopped:
        serve_security("0.0.0.0", env_token=None)
    assert "SEI_API_TOKEN" in str(stopped.value)


def test_a_network_bind_with_a_token_skips_the_host_check():
    """Clients will use the machine's own name; the token is the protection."""
    token, hosts = serve_security("0.0.0.0", env_token="a-long-chosen-token-0123456789")
    assert token == "a-long-chosen-token-0123456789" and hosts is None


def test_a_short_chosen_token_is_refused():
    with pytest.raises(SystemExit):
        serve_security("0.0.0.0", env_token="short")
