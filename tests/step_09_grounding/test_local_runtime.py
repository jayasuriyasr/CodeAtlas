"""Running on one machine: which models answer, and what the reader is shown.

Decision 0003 put a local pair of models behind S2 next to the hosted pair.
That makes "which one is answering?" a question with consequences — cost, where
the source code goes, how much context fits — so the choice is made in one
place (`api.cli.choose_providers`), never silently, and everything downstream
reads it off the provider: the context budget, the cost, the answer format.
"""

from __future__ import annotations

import json

import pytest

import metrics
from api.app import AnswerCost, InvestigationService, create_app
from api.cli import choose_providers, context_budget
from api.prompt import ANSWER_FORMAT_REMINDER, build_prompt, build_retry_prompt
from config import AVAILABLE, LLM_MODEL, RESERVED_OUT, RESERVED_SYS
from index.ollama import OllamaEmbeddingProvider, OllamaLLM
from index.providers import (
    AnthropicProvider,
    ApproxCodeTokenizer,
    DisabledEmbeddingProvider,
    ScriptedLLM,
    UnavailableLLM,
    VoyageEmbeddingProvider,
)
from pack.packer import pack

from .test_pipeline_hardening import GOOD, UNCITED, block

BOTH = {"qwen2.5-coder:3b", "nomic-embed-text:latest"}


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    metrics.reset()
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "VOYAGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield
    metrics.reset()


# -- choosing ------------------------------------------------------------------


def test_off_means_no_model_of_either_kind():
    llm, provider, notes = choose_providers("off")
    assert isinstance(llm, UnavailableLLM) and isinstance(provider, DisabledEmbeddingProvider)
    assert notes


def test_auto_uses_the_local_models_when_nothing_hosted_is_configured():
    llm, provider, notes = choose_providers("auto", local_models=BOTH)
    assert isinstance(llm, OllamaLLM) and isinstance(provider, OllamaEmbeddingProvider)
    assert any("nothing leaves this machine" in note for note in notes)


def test_auto_prefers_a_hosted_model_the_user_configured(monkeypatch):
    """Setting a key is a decision to send code to that service (§12.0)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    llm, provider, _notes = choose_providers("auto", local_models=BOTH)
    assert isinstance(llm, AnthropicProvider) and isinstance(provider, VoyageEmbeddingProvider)


def test_the_two_halves_are_chosen_independently(monkeypatch):
    """One key without the other is a normal state, not an error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm, provider, _notes = choose_providers("auto", local_models=BOTH)
    assert isinstance(llm, AnthropicProvider) and isinstance(provider, OllamaEmbeddingProvider)


def test_no_server_and_no_keys_says_so_and_still_runs():
    llm, provider, notes = choose_providers("auto", local_models=None)
    assert isinstance(llm, UnavailableLLM) and "no Ollama server" in llm.reason
    assert isinstance(provider, DisabledEmbeddingProvider)
    assert len(notes) == 2, "one note per missing half"


def test_a_missing_model_names_the_command_that_fixes_it():
    llm, provider, _notes = choose_providers("auto", local_models={"nomic-embed-text:latest"})
    assert isinstance(llm, UnavailableLLM)
    assert "ollama pull qwen2.5-coder:3b" in llm.reason
    assert isinstance(provider, OllamaEmbeddingProvider)


def test_local_never_reaches_for_a_hosted_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    llm, provider, _notes = choose_providers("local", local_models=BOTH)
    assert isinstance(llm, OllamaLLM) and isinstance(provider, OllamaEmbeddingProvider)


def test_cloud_never_falls_back_to_local():
    llm, provider, notes = choose_providers("cloud", local_models=BOTH)
    assert isinstance(llm, AnthropicProvider)
    assert isinstance(provider, DisabledEmbeddingProvider)
    assert any("VOYAGE_API_KEY" in note for note in notes)


# -- the context budget --------------------------------------------------------


def test_a_hosted_model_keeps_the_specs_budget():
    assert context_budget(ScriptedLLM("x")) == AVAILABLE
    assert context_budget(AnthropicProvider()) == AVAILABLE


def test_a_local_models_window_is_a_capacity_not_a_preference():
    llm = OllamaLLM(context_tokens=8192)
    assert context_budget(llm) == 8192 - RESERVED_OUT - RESERVED_SYS
    assert context_budget(llm) < AVAILABLE


def test_a_huge_local_window_does_not_exceed_the_specs_budget():
    assert context_budget(OllamaLLM(context_tokens=1_000_000)) == AVAILABLE


def test_a_window_too_small_for_the_reserves_is_zero_not_negative():
    assert context_budget(OllamaLLM(context_tokens=1024)) == 0


def test_the_packer_stays_inside_a_local_budget():
    llm = OllamaLLM(context_tokens=4096)
    seeds = [block(f"s{i}", 1.0 - i / 10) for i in range(8)]
    seen = {}
    service = InvestigationService(
        retriever=lambda q, r: (seeds, []), llm=ScriptedLLM(GOOD),
        tokenizer=ApproxCodeTokenizer(), available=context_budget(llm),
    )
    result = service.answer("q")
    seen["packed"] = result.context.packed_tokens
    assert 0 < seen["packed"] <= context_budget(llm)


# -- the prompt ----------------------------------------------------------------


def test_the_default_prompt_is_unchanged(request):
    tok = ApproxCodeTokenizer()
    context = pack([block("charge_card", 1.0)], [], tok, available=4_000)
    assert "# How to answer" not in build_prompt("q", context)


def test_a_reminder_is_the_last_thing_in_the_prompt():
    tok = ApproxCodeTokenizer()
    context = pack([block("charge_card", 1.0)], [], tok, available=4_000)
    prompt = build_prompt("q", context, reminder=ANSWER_FORMAT_REMINDER)
    retry = build_retry_prompt("q", context, "why", reminder=ANSWER_FORMAT_REMINDER)
    assert prompt.endswith(ANSWER_FORMAT_REMINDER) and retry.endswith(ANSWER_FORMAT_REMINDER)


class FormatLLM(ScriptedLLM):
    answer_format = "# How to answer\nCite everything."


def test_the_service_passes_a_providers_answer_format_on_both_attempts():
    llm = FormatLLM(UNCITED, GOOD)
    InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=llm, tokenizer=ApproxCodeTokenizer(),
    ).answer("q")
    assert len(llm.calls) == 2
    assert all(prompt.endswith(FormatLLM.answer_format) for _system, prompt in llm.calls)


def test_a_provider_without_one_gets_the_plain_prompt():
    llm = ScriptedLLM(GOOD)
    InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=llm, tokenizer=ApproxCodeTokenizer(),
    ).answer("q")
    assert "# How to answer" not in llm.calls[0][1]


# -- cost ----------------------------------------------------------------------


class FreeLLM(ScriptedLLM):
    name = "free-local-model"
    rates = (0.0, 0.0)


def test_a_provider_with_its_own_rates_is_billed_at_them_under_its_own_name():
    result = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=FreeLLM(GOOD), tokenizer=ApproxCodeTokenizer(),
    ).answer("q")
    assert result.cost.usd == 0.0
    assert result.cost.model == "free-local-model"


def test_a_provider_without_rates_is_the_model_decision_0002_prices():
    result = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=ScriptedLLM(GOOD), tokenizer=ApproxCodeTokenizer(),
    ).answer("q")
    assert result.cost.model == LLM_MODEL and result.cost.usd > 0


def test_explicit_rates_override_the_rate_card():
    cost = AnswerCost(packed_tokens=1_000_000, output_tokens=1_000_000, rates=(2.0, 10.0))
    assert cost.usd == pytest.approx(12.0)


# -- what the stream carries ---------------------------------------------------


def events_of(service, question="q"):
    return [e for e, _r in service.stream(question)]


def test_the_context_event_carries_what_each_handle_points_at():
    """§12.4: the trust model rests on the badge. A badge the reader cannot
    check is a decoration."""
    service = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], [block("retry_failed", 0.5)]),
        llm=ScriptedLLM(GOOD), tokenizer=ApproxCodeTokenizer(),
    )
    events = events_of(service)
    context = next(e for e in events if isinstance(e, dict) and e["type"] == "context")

    assert [b["handle"] for b in context["blocks"]] == ["C1", "C2"]
    first = context["blocks"][0]
    assert first["qualified_name"] == "billing.charge_card.charge_card"
    assert first["rel_path"] == "billing/charge_card.py"
    assert "def charge_card(user):" in first["text"]
    assert first["level"] == "full" and context["blocks"][1]["level"] == "stub"
    assert context["retry"] is False and "Context:" in context["report"]


def test_the_context_arrives_before_the_first_word_of_the_answer():
    service = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], [], {"route": "SEMANTIC"}),
        llm=ScriptedLLM(GOOD), tokenizer=ApproxCodeTokenizer(),
    )
    kinds = [e["type"] if isinstance(e, dict) else e.type for e in events_of(service)]
    assert kinds[:2] == ["retrieval", "context"]
    assert kinds.index("context") < kinds.index("text")


def test_a_retry_sends_the_context_it_will_actually_use():
    service = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=ScriptedLLM(UNCITED, GOOD), tokenizer=ApproxCodeTokenizer(),
        expander=lambda uids: [block("refund_card", 0.9)],
    )
    contexts = [e for e in events_of(service) if isinstance(e, dict) and e["type"] == "context"]
    assert [c["retry"] for c in contexts] == [False, True]


def test_the_done_event_has_the_numbers_a_console_shows():
    done = events_of(InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], [block("retry_failed", 0.5)]),
        llm=ScriptedLLM(GOOD), tokenizer=ApproxCodeTokenizer(),
    ))[-1]
    assert done["type"] == "done"
    assert done["coverage"] == 1.0 and done["conflict_flags"] == 0
    assert done["model"] == LLM_MODEL and done["ttft_seconds"] is not None
    assert done["unknown_citations"] == 0, "GOOD cites C1 and C2, and both were shown"
    json.dumps(done)                              # every value is serialisable


def test_a_citation_of_a_block_that_was_never_shown_is_counted_beside_coverage():
    """Seen with the local 3B model: `[C0]` against seven blocks. §6.2's Coverage
    counts the sentence as cited — it has a handle — so the console read "100%
    cited" over an invented citation. T1's count travels with the number."""
    invented = "The `charge_card` function charges the card [C1]. It then retries [C9]."
    done = events_of(InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=ScriptedLLM(invented), tokenizer=ApproxCodeTokenizer(),
    ))[-1]
    assert done["coverage"] == 1.0
    assert done["unknown_citations"] == 1


def test_no_model_is_a_reason_not_a_stack_trace():
    service = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=UnavailableLLM("no Ollama server at http://127.0.0.1:11434"),
        tokenizer=ApproxCodeTokenizer(),
    )
    unavailable = next(e for e in events_of(service)
                       if isinstance(e, dict) and e["type"] == "synthesis_unavailable")
    assert "no Ollama server" in unavailable["reason"]
    assert unavailable["blocks"][0]["qualified_name"] == "billing.charge_card.charge_card"


# -- the HTTP surface ----------------------------------------------------------


def _app(**kwargs):
    service = InvestigationService(
        retriever=lambda q, r: ([block("charge_card", 1.0)], []),
        llm=ScriptedLLM(GOOD), tokenizer=ApproxCodeTokenizer(),
    )
    return create_app(service, **kwargs)


def test_the_console_page_is_served_and_is_self_contained():
    """No CDN, no build step: it has to work with the network cable out."""
    from fastapi.testclient import TestClient

    with TestClient(_app()) as client:
        response = client.get("/")
    page = response.text
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert "<title>SEI" in page and 'fetch("/ask"' in page and 'fetch("/status")' in page
    for external in ("https://", "http://", "//cdn", "<link rel=\"stylesheet\""):
        assert external not in page, f"the console pulls in {external!r}"


def test_status_reports_what_is_answering():
    from fastapi.testclient import TestClient

    status = lambda: {"repo": "r", "llm": "ollama:qwen2.5-coder:3b",     # noqa: E731
                      "vector_search": True, "graph": {"files": 3, "symbols": 9}}
    with TestClient(_app(repos=["r"], status=status)) as client:
        payload = client.get("/status").json()
    assert payload["ok"] is True and payload["repos"] == ["r"]
    assert payload["llm"] == "ollama:qwen2.5-coder:3b" and payload["graph"]["symbols"] == 9
    assert "watcher" not in payload


def test_status_works_with_nothing_configured():
    from fastapi.testclient import TestClient

    with TestClient(_app()) as client:
        assert client.get("/status").json() == {"ok": True, "repos": []}


def test_status_never_shows_a_finished_watcher_over_unfinished_figures():
    """`/status` is two reads at two instants: the route runs on a worker
    thread while the first index finishes on the event loop. Read in the wrong
    order, one response said "watching" beside `vector_search: false` — the
    figures were from before the index finished, the state from after. Seen
    once in 944 tests, as a failure of the whole-server test."""
    from fastapi.testclient import TestClient

    class Watcher:
        state, last_error, files_watched, last_change_at = "indexing", None, 0, None

        async def run(self, bootstrap=None):
            return None

        def stop(self):
            return None

    watcher = Watcher()

    def status():
        figures = {"vector_search": False, "graph": {"symbols": 0}}   # true while indexing
        watcher.state = "watching"                                    # …which then finishes
        return figures

    with TestClient(_app(watcher=watcher, status=status)) as client:
        first = client.get("/status").json()
        assert first["watcher"]["state"] == "indexing", (
            "the state may lag the figures; it must never lead them"
        )
        assert client.get("/status").json()["watcher"]["state"] == "watching"


def test_healthz_is_unchanged():
    from fastapi.testclient import TestClient

    with TestClient(_app()) as client:
        assert client.get("/healthz").json() == {"ok": True, "model": LLM_MODEL}


def test_the_watcher_runs_for_the_life_of_the_server_and_shows_in_status():
    from fastapi.testclient import TestClient

    class FakeWatcher:
        state, last_error, files_watched, last_change_at = "starting", None, 0, None

        def __init__(self):
            self.bootstrapped = False
            self.stopped = False

        async def run(self, bootstrap=None):
            import asyncio

            if bootstrap is not None:
                await bootstrap()
            self.state, self.files_watched = "watching", 12
            while not self.stopped:
                await asyncio.sleep(0.01)

        def stop(self):
            self.stopped = True

    watcher = FakeWatcher()

    async def bootstrap():
        watcher.bootstrapped = True

    with TestClient(_app(watcher=watcher, bootstrap=bootstrap)) as client:
        for _ in range(100):
            payload = client.get("/status").json()
            if payload["watcher"]["state"] == "watching":
                break
        assert payload["watcher"] == {"state": "watching", "error": None,
                                      "files_watched": 12, "last_change_at": None}
    assert watcher.bootstrapped and watcher.stopped
