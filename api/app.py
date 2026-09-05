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
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import metrics
from adapters.base import Symbol
from config import (
    AVAILABLE,
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
    """

    packed_tokens: int
    output_tokens: int
    model: str = LLM_MODEL

    @property
    def usd(self) -> float:
        rate_in = LLM_INPUT_RATE_USD_PER_MTOK or 0.0
        rate_out = LLM_OUTPUT_RATE_USD_PER_MTOK or 0.0
        return (
            self.packed_tokens * rate_in + self.output_tokens * rate_out
        ) / 1_000_000


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


#: Retrieval is injected: `(question, routed) -> (seeds, neighbors)`. The query
#: path is testable without a database this way, and step 6's retriever and
#: step 7's frame resolver plug in behind the same signature.
Retriever = Callable[[str, Routed], "tuple[Sequence[Block], Sequence[Block]]"]


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
        seeds, neighbors = self.retriever(question, routed)
        context = pack(seeds, neighbors, self.tokenizer, available=self.available)

        result = AnswerResult(question=question, routed=routed, context=context)
        known_names = frozenset(
            p.block.symbol.qualified_name for p in context.blocks
        )

        prompt = build_prompt(question, context)
        yield from self._generate(result, SYSTEM, prompt, known_names, started)

        # §6.3: one retry, bounded, never a loop.
        if should_retry(result.report):
            metrics.incr("ground.retry")
            plan = plan_retry(
                result.report, [b.uid for b in context.dropped], context.handles
            )
            promoted = _promote(seeds, neighbors, context, plan.promote)
            retry_context = pack(
                promoted, neighbors, self.tokenizer, available=self.available
            )
            result.context = retry_context
            result.retried = True
            known_names = frozenset(
                p.block.symbol.qualified_name for p in retry_context.blocks
            )
            yield from self._generate(
                result,
                SYSTEM,
                build_retry_prompt(question, retry_context, plan.reason),
                known_names,
                started,
                retry_count=1,
            )

        result.total_seconds = time.perf_counter() - started
        result.cost = AnswerCost(
            packed_tokens=result.context.packed_tokens,
            output_tokens=self.tokenizer.count(result.text),
        )
        metrics.incr("answer.packed_tokens", result.context.packed_tokens)
        yield ({"type": "done", "report": result.report.render()}, result)

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
        first_token = True

        for chunk in self.llm.stream(system, prompt, max_tokens=RESERVED_OUT):
            for event in validator.feed(chunk):
                if first_token and isinstance(event, TextEvent):
                    result.ttft_seconds = time.perf_counter() - started
                    first_token = False
                result.events.append(event)
                yield (event, result)

        for event in validator.close():
            result.events.append(event)
            yield (event, result)

        result.text = validator.text
        # T3 runs post-stream, on the whole answer (§6.1's tier table).
        result.report = ground(
            validator.text,
            result.context.handles,
            known_names=known_names,
            runner=self.graph_runner,
            repo_id=self.repo_id,
            retry_count=retry_count,
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
        }
    else:
        payload = dict(event)  # type: ignore[arg-type]
    return f"data: {json.dumps(payload)}\n\n"


def create_app(service: InvestigationService, indexer=None):
    """The FastAPI surface. Two routes: ask, and the §4.4 rename ingress."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="SEI")

    class Ask(BaseModel):
        question: str

    class Rename(BaseModel):
        repo: str
        old_path: str
        new_path: str

    @app.post("/ask")
    def ask(body: Ask):
        def generate():
            for event, _result in service.stream(body.question):
                yield sse(event)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/index/rename")
    def rename(body: Rename):
        """§4.4 Signal A's HTTP binding. The editor plugin posts here."""
        if indexer is None:
            return {"accepted": False, "reason": "no indexer configured"}
        indexer.on_rename(body.repo, body.old_path, body.new_path)
        return {"accepted": True}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": LLM_MODEL}

    return app
