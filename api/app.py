"""FastAPI + SSE — the query path, end to end (§7).

This is the step the plan calls "the first moment the system exists": a real
question returns a cited, streamed answer. Everything before it produced parts.

The pipeline follows §7's diagram: route -> resolve/retrieve -> expand -> pack
-> synthesize -> validate -> maybe retry once.
"""

# NOTE: deliberately no `from __future__ import annotations`.
# The request models below are defined inside `create_app`, and postponed
# annotations would leave FastAPI resolving the string 'Ask' against this
# module's globals, where it does not exist. It then treats the body as a
# query parameter and every request 422s. Python >= 3.11 (see pyproject)
# evaluates `str | None` and `list[Block]` natively, so nothing here needs it.

import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Collection, Iterator, Sequence

import metrics
from adapters.base import Symbol
from config import (
    AVAILABLE,
    COVERAGE_RETRY_THRESHOLD,
    LLM_INPUT_RATE_USD_PER_MTOK,
    LLM_MODEL,
    LLM_OUTPUT_RATE_USD_PER_MTOK,
    RESERVED_OUT,
)
from api.prompt import SYSTEM, build_prompt, build_retry_prompt
from api.stream import SentenceEvent, StreamingValidator, TextEvent
from ground.validate import GroundingReport, ground, plan_retry, should_retry
from index.chunker import Tokenizer
from pack.packer import PackedContext, pack
from pack.render import Block
from retrieve.router import QueryClass, QueryRouter, Routed


@dataclass
class AnswerCost:
    """§8.2 records `$/query` p50/p95 and p50 packed tokens alongside it.

    §5.4: "What determines actual cost is p50 packed tokens, not the ceiling."
    Both are recorded per answer so the p50 is a measurement rather than an
    estimate, and decision 0002's lever has data to be pulled against.

    `packed_tokens` is the context the kept answer was grounded on — §5.4's
    quantity. `input_tokens` and `output_tokens` are what was *billed*, summed
    over every generation: a §6.3 retry sends a second prompt and receives a
    second answer, and a `$/query` that counted only the one shown would
    understate exactly the queries that cost most. When the provider reports
    its own usage those are its numbers; otherwise they are counted locally.
    `input_tokens` of None falls back to `packed_tokens`, §5.4's approximation.
    """

    packed_tokens: int
    output_tokens: int
    model: str = LLM_MODEL
    input_tokens: int | None = None
    generations: int = 1
    #: `(input, output)` dollars per million tokens, when the provider states
    #: its own — a local model's are zero. None means decision 0002's rate card.
    rates: tuple[float, float] | None = None

    @property
    def usd(self) -> float:
        if self.rates is not None:
            rate_in, rate_out = self.rates
        else:
            rate_in = LLM_INPUT_RATE_USD_PER_MTOK or 0.0
            rate_out = LLM_OUTPUT_RATE_USD_PER_MTOK or 0.0
        billed_in = self.packed_tokens if self.input_tokens is None else self.input_tokens
        return (billed_in * rate_in + self.output_tokens * rate_out) / 1_000_000


@dataclass
class AnswerResult:
    question: str
    routed: Routed
    context: PackedContext
    text: str = ""
    report: GroundingReport | None = None
    retried: bool = False
    ttft_seconds: float | None = None
    total_seconds: float | None = None
    cost: AnswerCost | None = None
    events: list = field(default_factory=list)
    #: What retrieval wants the reader told: per-frame badges (§5.1), a
    #: degraded vector arm (§9.3), the route actually taken. Empty when the
    #: retriever has nothing to say.
    notes: dict = field(default_factory=dict)
    #: Billed totals across every generation, for `AnswerCost`.
    input_tokens: int = 0
    output_tokens: int = 0
    generations: int = 0


#: Retrieval is injected: `(question, routed) -> (seeds, neighbors)`, or
#: `(seeds, neighbors, notes)` when it has something to surface. The query path
#: is testable without a database this way, and `retrieve.pipeline` plugs the
#: real router-driven retrieval in behind the same signature.
Retriever = Callable[[str, Routed], "tuple"]

class SynthesisUnavailable(RuntimeError):
    """The LLM provider failed before or during an answer (§9.3)."""


#: §6.3's "bounded expansion around uncited/conflicting spans": uids in,
#: neighbour blocks out. Optional — without it a retry can only promote.
Expander = Callable[[Sequence[str]], Sequence[Block]]


@dataclass
class InvestigationService:
    """The orchestration §7 draws. One place, so the flow is readable."""

    retriever: Retriever
    llm: object
    tokenizer: Tokenizer
    router: QueryRouter = field(default_factory=QueryRouter)
    graph_runner: object | None = None
    repo_id: str = ""
    available: int = AVAILABLE
    expander: Expander | None = None

    def answer(self, question: str) -> AnswerResult:
        """Run the full path, collecting events rather than streaming them."""
        result = None
        for _event, result in self._run(question):
            pass
        assert result is not None
        return result

    def stream(self, question: str) -> Iterator[tuple[object, AnswerResult]]:
        yield from self._run(question)

    # ------------------------------------------------------------------

    def _run(self, question: str) -> Iterator[tuple[object, AnswerResult]]:
        started = time.perf_counter()
        routed = self.router.route(question)
        retrieved = self.retriever(question, routed)
        seeds, neighbors = retrieved[0], retrieved[1]
        notes = dict(retrieved[2]) if len(retrieved) > 2 and retrieved[2] else {}
        context = pack(seeds, neighbors, self.tokenizer, available=self.available)

        result = AnswerResult(
            question=question, routed=routed, context=context, notes=notes
        )
        if notes:
            event = {"type": "retrieval", **notes}
            result.events.append(event)
            yield (event, result)
        event = _context_event(context)
        result.events.append(event)
        yield (event, result)

        known_names = frozenset(
            p.block.symbol.qualified_name for p in context.blocks
        )

        # A provider may ask for its answer format to be restated after the
        # context — small local models need it (`api.prompt`); hosted ones do not.
        reminder = getattr(self.llm, "answer_format", "")
        prompt = build_prompt(question, context, reminder=reminder)
        try:
            yield from self._generate(result, SYSTEM, prompt, known_names, started)
        except SynthesisUnavailable as exc:
            yield from self._retrieval_only(result, exc, started)
            return

        # §6.3: one retry, bounded, never a loop.
        if should_retry(result.report):
            metrics.incr("ground.retry")
            plan = plan_retry(
                result.report, [b.uid for b in context.dropped], context.handles
            )
            promoted = _promote(seeds, neighbors, context, plan.promote)
            retry_neighbors = self._expand_for_retry(neighbors, plan.expand_around)
            retry_context = pack(
                promoted, retry_neighbors, self.tokenizer, available=self.available
            )
            added = _uids(retry_context) - _uids(context)
            result.context = retry_context
            result.retried = True
            known_names = frozenset(
                p.block.symbol.qualified_name for p in retry_context.blocks
            )
            event = {"type": "retry", "reason": plan.reason, "added_blocks": len(added)}
            result.events.append(event)
            yield (event, result)
            event = _context_event(retry_context, retry=True)
            result.events.append(event)
            yield (event, result)
            try:
                yield from self._generate(
                    result,
                    SYSTEM,
                    build_retry_prompt(
                        question, retry_context, plan.reason,
                        added_context=bool(added), reminder=reminder,
                    ),
                    known_names,
                    started,
                    retry_count=1,
                )
            except SynthesisUnavailable as exc:
                # The first answer stands, judged against the context it saw.
                metrics.incr("synthesis.unavailable")
                result.context = context
                event = {"type": "synthesis_unavailable", "during": "retry",
                         "reason": str(exc)}
                result.events.append(event)
                yield (event, result)

        result.total_seconds = time.perf_counter() - started
        # A provider that states its own rates is billed under its own name;
        # one that states none is the model decision 0002 prices.
        rates = getattr(self.llm, "rates", None)
        result.cost = AnswerCost(
            packed_tokens=result.context.packed_tokens,
            output_tokens=result.output_tokens,
            input_tokens=result.input_tokens,
            generations=result.generations,
            model=getattr(self.llm, "name", LLM_MODEL) if rates is not None else LLM_MODEL,
            rates=rates,
        )
        metrics.incr("answer.packed_tokens", result.context.packed_tokens)
        # §6.3's last line: a retry that still fails is surfaced honestly, not
        # hidden behind the second answer.
        needs_review = result.retried and (
            result.report.coverage < COVERAGE_RETRY_THRESHOLD
            or result.report.conflict_flags > 0
        )
        yield (
            {
                "type": "done",
                "report": result.report.render(),
                "context": result.context.report(),
                "retried": result.retried,
                "needs_review": needs_review,
                "coverage": result.report.coverage,
                "relation_precision": result.report.relation_precision,
                "conflict_flags": result.report.conflict_flags,
                # T1's count. Coverage (§6.2) counts a sentence as cited when
                # it carries a handle at all, so "100% cited" can sit beside a
                # citation of a block that was never shown; the console has to
                # be able to say so next to the number.
                "unknown_citations": len(result.report.handle_errors),
                # Cited claims naming code that is not in the blocks they cite.
                "unsupported_claims": len(result.report.unsupported),
                "ttft_seconds": result.ttft_seconds,
                "total_seconds": result.total_seconds,
                "model": result.cost.model,
                "usd": result.cost.usd,
                "input_tokens": result.cost.input_tokens,
                "output_tokens": result.cost.output_tokens,
            },
            result,
        )

    def _retrieval_only(
        self, result: AnswerResult, exc: Exception, started: float
    ) -> Iterator[tuple[object, AnswerResult]]:
        """§9.3, "LLM provider": retrieval-only — ranked code blocks, no prose.

        Whatever text arrived before the failure has already been shown, but it
        is not graded: a half-answer's Coverage measures where the stream broke,
        not how well the model cited.
        """
        metrics.incr("synthesis.unavailable")
        result.report = None
        result.total_seconds = time.perf_counter() - started
        blocks = [
            {
                "handle": p.handle,
                "uid": p.uid,
                "qualified_name": p.block.symbol.qualified_name,
                "rel_path": p.block.symbol.rel_path,
                "level": p.level,
            }
            for p in result.context.blocks
        ]
        event = {"type": "synthesis_unavailable", "during": "answer",
                 "reason": str(exc), "blocks": blocks}
        result.events.append(event)
        yield (event, result)
        yield (
            {
                "type": "done",
                "report": "Synthesis unavailable",
                "context": result.context.report(),
                "retried": False,
                "needs_review": True,
            },
            result,
        )

    def _expand_for_retry(
        self, neighbors: Sequence[Block], around: Sequence[str]
    ) -> list[Block]:
        """§6.3's bounded expansion, merged under the neighbours already held.

        Bounded twice over: §11.1's own limits cap what the expander returns,
        and everything it returns still has to fit the neighbour budget in
        `pack`. A block already present keeps its original score.
        """
        merged = list(neighbors)
        if not around or self.expander is None:
            return merged
        held = {b.uid for b in merged}
        for block in self.expander(list(around)):
            if block.uid not in held:
                merged.append(block)
                held.add(block.uid)
        metrics.incr("ground.retry_expanded", len(merged) - len(neighbors))
        return merged

    def _generate(
        self,
        result: AnswerResult,
        system: str,
        prompt: str,
        known_names: frozenset[str],
        started: float,
        retry_count: int = 0,
    ) -> Iterator[tuple[object, AnswerResult]]:
        validator = StreamingValidator(
            handle_map=result.context.handles, known_names=known_names
        )
        usage: dict = {}

        chunks = iter(self.llm.stream(system, prompt, max_tokens=RESERVED_OUT, usage=usage))
        while True:
            # Only the provider's own failures are §9.3's "LLM provider" row.
            # The validator runs outside this `try`, so a defect in it still
            # surfaces as the defect it is.
            try:
                chunk = next(chunks)
            except StopIteration:
                break
            except Exception as exc:               # noqa: BLE001 - typed below
                raise SynthesisUnavailable(f"{type(exc).__name__}: {exc}") from exc
            for event in validator.feed(chunk):
                # TTFT is the *first* token the reader sees. A retry's first
                # token arrives after a whole answer and is not a latency the
                # user perceives as waiting to start; overwriting would inflate
                # §8.2's p95 by exactly the retried queries.
                if result.ttft_seconds is None and isinstance(event, TextEvent):
                    result.ttft_seconds = time.perf_counter() - started
                result.events.append(event)
                yield (event, result)

        for event in validator.close():
            result.events.append(event)
            yield (event, result)

        result.text = validator.text
        result.generations += 1
        result.input_tokens += usage.get("input_tokens") or (
            self.tokenizer.count(system) + self.tokenizer.count(prompt)
        )
        result.output_tokens += usage.get("output_tokens") or self.tokenizer.count(
            validator.text
        )
        # T3 runs post-stream, on the whole answer (§6.1's tier table).
        result.report = ground(
            validator.text,
            result.context.handles,
            known_names=known_names,
            runner=self.graph_runner,
            repo_id=self.repo_id,
            retry_count=retry_count,
            block_text={p.handle: p.text for p in result.context.blocks if p.handle},
        )


def _promote(
    seeds: Sequence[Block],
    neighbors: Sequence[Block],
    context: PackedContext,
    promote: Sequence[str],
) -> list[Block]:
    """§6.3: promote dropped blocks before expanding.

    "Unseen evidence is a likelier explanation for low coverage than an
    insufficient neighborhood." Promotion is implemented as a score bump rather
    than a separate list, so the packer's ordering stays the single place that
    decides what gets in.
    """
    if not promote:
        return list(seeds)

    wanted = set(promote)
    by_uid = {b.uid: b for b in [*seeds, *neighbors, *context.dropped]}
    top = max((b.score for b in seeds), default=1.0)

    promoted = [
        Block(symbol=by_uid[uid].symbol, score=top + 1.0)
        for uid in promote
        if uid in by_uid
    ]
    return promoted + [b for b in seeds if b.uid not in wanted]


def _uids(context: PackedContext) -> set[str]:
    return {p.uid for p in context.blocks}


def _context_event(context: PackedContext, *, retry: bool = False) -> dict:
    """The packed blocks, exactly as the model is about to see them.

    §12.4 says the trust model rests on the citation badge. A badge the reader
    cannot check is a decoration, so the stream carries what each handle points
    at — the rendered text, the file and the lines — before the first token of
    the answer that cites it.
    """
    return {
        "type": "context",
        "retry": retry,
        "report": context.report(),
        "blocks": [
            {
                "handle": p.handle,
                "uid": p.uid,
                "qualified_name": p.block.symbol.qualified_name,
                "rel_path": p.block.symbol.rel_path,
                "start_line": p.block.symbol.start_line,
                "end_line": p.block.symbol.end_line,
                "kind": p.block.symbol.kind,
                "level": p.level,
                "text": p.text,
            }
            for p in context.blocks
        ],
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def sse(event: object) -> str:
    """One Server-Sent Event frame."""
    if isinstance(event, TextEvent):
        payload = {"type": "text", "text": event.text}
    elif isinstance(event, SentenceEvent):
        payload = {
            "type": "sentence",
            "text": event.text,
            "is_claim": event.is_claim,
            "cited": event.cited,
            "handles": list(event.handles),
            "badge": event.badge,
            "reason": event.reason,
            "amends": event.amends,
        }
    else:
        payload = dict(event)  # type: ignore[arg-type]
    return f"data: {json.dumps(payload)}\n\n"


def _host_name(host_header: str) -> str:
    """`localhost:8000` -> `localhost`, `[::1]:8000` -> `::1`, lower-cased."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _bearer(header: str) -> str | None:
    scheme, _, value = header.partition(" ")
    return value.strip() or None if scheme.lower() == "bearer" else None


#: The console: one self-contained page, no build step and no CDN, so it works
#: with the network cable out. Plan §9's cut ladder dropped the Next.js console
#: for the CLI; this is the smallest thing that puts §6.4's badges and §5.4's
#: packing report in front of someone who is not at a terminal.
CONSOLE_PAGE = Path(__file__).parent / "static" / "index.html"


def create_app(
    service: InvestigationService,
    indexer=None,
    repos: Sequence[str] = (),
    *,
    watcher=None,
    bootstrap: Callable[[], Awaitable[None]] | None = None,
    status: Callable[[], dict] | None = None,
    token: str | None = None,
    allowed_hosts: Collection[str] | None = None,
):
    """The FastAPI surface: the console, ask, rename ingress, Sync Project.

    The indexing routes are `async def` on purpose. `Indexer.on_rename` arms a
    debounce timer on the running event loop (§4.1); a plain `def` route runs in
    FastAPI's thread pool, where there is no loop, and every rename from a real
    editor would fail with `RuntimeError: no running event loop`.

    `repos` are the repositories whose `_indexed` set is seeded at startup
    (§4.1 `startup`) — without it every first flush pays for Signal B's git
    subprocess.

    `watcher` (an `index.watcher.TreeWatcher`) runs for the life of the server:
    `bootstrap` first, to bring the graph level with the tree, then polling.
    It runs as a task rather than in start-up proper so the server answers —
    and `/status` can say "indexing" — while a first index is still under way.
    `status` supplies what `/status` reports about providers and the graph.

    `token` and `allowed_hosts` are the access check, and `api.cli` always
    passes both. With a token, every route but `/healthz` needs
    `Authorization: Bearer <token>`; the console page alone may take it as
    `?token=` instead, because that is the link the browser is opened at.
    With `allowed_hosts`, a request whose `Host` header names anything else is
    refused before the token is looked at — the defence against a web page
    rebinding its own domain to 127.0.0.1. Left at None (as the tests that
    exercise the routes themselves do), the app is open.
    """
    import asyncio
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel

    @asynccontextmanager
    async def lifespan(_app):
        if indexer is not None and hasattr(indexer, "startup"):
            for repo in repos:
                await indexer.startup(repo)
        task = None
        if watcher is not None:
            task = asyncio.create_task(watcher.run(bootstrap))
        elif bootstrap is not None:
            task = asyncio.create_task(bootstrap())
        yield
        if watcher is not None:
            watcher.stop()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if indexer is not None and hasattr(indexer, "aclose"):
            await indexer.aclose()

    app = FastAPI(title="SEI", lifespan=lifespan)
    hosts = None if allowed_hosts is None else {h.lower() for h in allowed_hosts}

    @app.middleware("http")
    async def access_check(request, call_next):
        if hosts is not None and _host_name(request.headers.get("host", "")) not in hosts:
            return JSONResponse({"error": "this server answers on localhost only"}, status_code=403)
        if token is not None and request.url.path != "/healthz":
            supplied = _bearer(request.headers.get("authorization", ""))
            if supplied is None and request.method == "GET" and request.url.path == "/":
                supplied = request.query_params.get("token")
            if supplied is None or not secrets.compare_digest(supplied.encode(), token.encode()):
                return JSONResponse({"error": "missing or wrong token"}, status_code=401)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def console():
        # Read per request: the page is one small file, and an edit to it
        # should not need the server (and its warm model) restarted.
        return HTMLResponse(CONSOLE_PAGE.read_text(encoding="utf-8"))

    @app.get("/status")
    def get_status():
        """What is answering, from what, and how current the index is."""
        payload = {"ok": True, "repos": list(repos)}
        # The watcher is read *before* the rest. This route runs on a worker
        # thread while the first index finishes on the event loop, so the two
        # reads are two instants. In the other order a response could say
        # "watching" over figures taken while it was still indexing — seen as
        # `vector_search: false` beside a finished index. This way the state is
        # never ahead of the numbers it sits beside.
        watching = None
        if watcher is not None:
            watching = {
                "state": watcher.state,
                "error": watcher.last_error,
                "files_watched": watcher.files_watched,
                "last_change_at": watcher.last_change_at,
            }
        if status is not None:
            payload.update(status())
        if watching is not None:
            payload["watcher"] = watching
        return payload

    class Ask(BaseModel):
        question: str

    class Rename(BaseModel):
        repo: str
        old_path: str
        new_path: str

    class Sync(BaseModel):
        repo: str

    @app.post("/ask")
    def ask(body: Ask):
        def generate():
            for event, _result in service.stream(body.question):
                yield sse(event)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/index/rename")
    async def rename(body: Rename):
        """§4.4 Signal A's HTTP binding. The editor plugin posts here."""
        if indexer is None:
            return {"accepted": False, "reason": "no indexer configured"}
        if body.repo not in getattr(indexer, "roots", {body.repo: None}):
            # A flush for a repo with no root would fail reading the new path,
            # in a background task where nobody sees the error.
            return {"accepted": False, "reason": f"unknown repo {body.repo!r}"}
        try:
            indexer.on_rename(body.repo, body.old_path, body.new_path)
        except ValueError as exc:                  # UnsafePath: absolute, or outside the root
            return JSONResponse({"accepted": False, "reason": str(exc)}, status_code=400)
        return {"accepted": True}

    @app.post("/index/sync")
    async def sync(body: Sync):
        """§4.1's Sync Project button — the recovery path for undetected moves."""
        if indexer is None or not hasattr(indexer, "sync_project"):
            return {"accepted": False, "reason": "no indexer configured"}
        if body.repo not in getattr(indexer, "roots", {body.repo: None}):
            return {"accepted": False, "reason": f"unknown repo {body.repo!r}"}
        await indexer.sync_project(body.repo)
        return {"accepted": True}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": LLM_MODEL}

    return app
