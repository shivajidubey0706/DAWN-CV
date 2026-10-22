"""Ask SG-ERUP AI: Claude (when configured server-side) with retrieved context, else local fallback."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from .knowledge import KnowledgeBase
from .settings import settings

log = logging.getLogger("sgerup.ask")

SYSTEM_PROMPT = """You are "Ask SG-ERUP AI", the assistant inside a research prototype that implements SG-ERUP, \
a proposed spatially gated, saturation-aware extension of ERUP-YOLO for object detection in adverse weather.

You answer questions about the method (BPW and KBL filters, cue maps, the spatial gate, the saturation-aware loss, \
spatially heterogeneous augmentation), datasets (DAWN, VOC, RTTS, ExDark, BDD100K, ACDC, Foggy Cityscapes), metrics \
(mAP, precision, recall, NSR, BRB, brightness-stratified recall), experiment interpretation, limitations, and \
project setup or troubleshooting.

Ground rules:
- Use the context excerpts provided with each question. Label claims by provenance: [Paper] for what the \
paper documents, [Measured] for numbers from this project's result files, [Implementation] for how this codebase \
does something, [Hypothesis] for proposed but unverified effects.
- Never invent numbers, citations, dataset sizes or model status. If a result is not in the context, say it \
has not been measured. SG-ERUP is unvalidated: do not claim it improves detection unless measured results \
in the context show it, and point out that 2 seeds cannot establish significance.
- Mention reduced-protocol caveats (frozen detector, short training, YOLOv3u detector) when interpreting results.
- Be concise and direct; use short paragraphs or bullet lists; use plain-text math (e.g. I~ = G*B(I) + (1-G)*I)."""


def build_context(kb: KnowledgeBase, question: str) -> tuple[str, list[dict]]:
    hits = kb.search(question, k=7)
    measured = [c for c in kb.chunks if c.kind == "measured"][:2]
    seen, parts, sources = set(), [], []
    for _, c in hits + [(0, m) for m in measured]:
        key = (c.source, c.text[:60])
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"<excerpt source=\"{c.source}\" section=\"{c.title}\" kind=\"{c.kind}\">\n{c.text}\n</excerpt>")
        sources.append({"source": c.source, "section": c.title, "kind": c.kind})
    return "\n\n".join(parts), sources


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def fallback_answer(kb: KnowledgeBase, question: str) -> tuple[str, list[dict]]:
    hits = kb.search(question, k=4)
    if not hits:
        return ("**Local fallback mode** (no LLM configured). I found nothing in the paper, docs or result files "
                "matching your question. Try terms such as *spatial gate*, *L_sat*, *NSR*, *DAWN sand*, "
                "*KBL*, *training*, or *mAP*."), []
    lines = ["**Local fallback mode** — no LLM is configured on the server, so this is a ranked excerpt search "
             "over the paper, the project documentation and the measured result files (not a generated answer).", ""]
    sources = []
    for score, c in hits:
        tag = "Measured result" if c.kind == "measured" else ("Paper" if c.source == "paper" else "Project docs")
        text = c.text if len(c.text) < 900 else c.text[:900].rsplit(" ", 1)[0] + " …"
        lines += [f"**[{tag}] {c.title}** · _{c.source}_", "", text, ""]
        sources.append({"source": c.source, "section": c.title, "kind": c.kind, "score": round(score, 2)})
    return "\n".join(lines), sources


async def answer_stream(kb: KnowledgeBase, messages: list[dict]) -> AsyncIterator[str]:
    question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    ok, reason = settings.llm_configured()
    if not ok:
        text, sources = fallback_answer(kb, question)
        yield _sse({"type": "meta", "mode": "fallback", "reason": reason, "sources": sources})
        for i in range(0, len(text), 120):
            yield _sse({"type": "delta", "text": text[i:i + 120]})
        yield _sse({"type": "done"})
        return

    import anthropic

    context, sources = build_context(kb, question)
    yield _sse({"type": "meta", "mode": "llm", "model": settings.llm_model, "sources": sources})
    history = [{"role": m["role"], "content": m["content"]} for m in messages[-12:] if m.get("role") in ("user", "assistant")]
    if not history or history[-1]["role"] != "user":
        yield _sse({"type": "error", "message": "The last message must be from the user."})
        return
    history[-1] = {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}
    client = anthropic.AsyncAnthropic()
    try:
        async with client.beta.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=history,
            output_config={"effort": settings.llm_effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                yield _sse({"type": "delta", "text": text})
            final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            yield _sse({"type": "delta", "text": "\n\n_The model declined to answer this request._"})
        elif final.stop_reason == "max_tokens":
            yield _sse({"type": "delta", "text": "\n\n_(answer truncated at the token limit)_"})
        yield _sse({"type": "done", "stop_reason": final.stop_reason})
    except anthropic.AuthenticationError:
        log.warning("Ask AI: authentication failed")
        async for ev in _fallback_after_error(kb, question, "LLM authentication failed (check the server's API key)"):
            yield ev
    except anthropic.RateLimitError:
        async for ev in _fallback_after_error(kb, question, "LLM rate limit reached; try again shortly"):
            yield ev
    except anthropic.APIStatusError as e:
        log.warning("Ask AI: API error %s", e.status_code)
        async for ev in _fallback_after_error(kb, question, f"LLM API error (HTTP {e.status_code})"):
            yield ev
    except anthropic.APIConnectionError:
        async for ev in _fallback_after_error(kb, question, "could not reach the LLM API"):
            yield ev


async def _fallback_after_error(kb, question, reason):
    yield _sse({"type": "error", "message": reason, "recoverable": True})
    text, sources = fallback_answer(kb, question)
    yield _sse({"type": "meta", "mode": "fallback", "reason": reason, "sources": sources})
    yield _sse({"type": "delta", "text": text})
    yield _sse({"type": "done"})
