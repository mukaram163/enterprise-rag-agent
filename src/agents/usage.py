import threading
import time

from langchain_core.callbacks import BaseCallbackHandler

from src.utils.request_logger import log_request

_lock = threading.Lock()
_run = {"calls": 0, "tokens_in": 0, "tokens_out": 0}


def reset_run() -> None:
    with _lock:
        _run.update(calls=0, tokens_in=0, tokens_out=0)


def run_tokens() -> int:
    with _lock:
        return _run["tokens_in"] + _run["tokens_out"]


def run_summary() -> str:
    with _lock:
        total = _run["tokens_in"] + _run["tokens_out"]
        return f"{_run['calls']} model calls, {total:,} tokens ({_run['tokens_in']:,} in / {_run['tokens_out']:,} out)"


def _first_human(messages) -> str:
    for m in (messages[0] if messages else []):
        if getattr(m, "type", "") == "human":
            return str(m.content)[:120]
    return ""


class UsageCallback(BaseCallbackHandler):
    """Attached to every agent LLM, so no model call can skip logging or the token count."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._started = {}  # run_id -> (start time, short prompt text)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._started[run_id] = (time.perf_counter(), _first_human(messages))

    def on_llm_end(self, response, *, run_id, **kwargs):
        t0, prompt = self._started.pop(run_id, (None, ""))
        tin = tout = 0
        for gens in response.generations:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                tin += um.get("input_tokens", 0)
                tout += um.get("output_tokens", 0)
        if not (tin or tout):  # fallback for providers that report usage differently
            usage = (response.llm_output or {}).get("token_usage") or {}
            tin = usage.get("prompt_tokens", 0)
            tout = usage.get("completion_tokens", 0)
        with _lock:
            _run["calls"] += 1
            _run["tokens_in"] += tin
            _run["tokens_out"] += tout
        latency = (time.perf_counter() - t0) * 1000 if t0 is not None else None
        log_request(question=prompt, model_used=self.model_name, tokens_in=tin, tokens_out=tout,
                    latency_ms=latency, pages_cited=[], source="agent")
