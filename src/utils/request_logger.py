import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List

logger = logging.getLogger(__name__)

LOGS_DIR = "data/logs"
LOGS_FILE = os.path.join(LOGS_DIR, "requests.jsonl")


def log_request(
    question: str,
    model_used: str,
    tokens_in: Optional[int],
    tokens_out: Optional[int],
    latency_ms: Optional[float],
    pages_cited: List[int],
    estimated_cost: Optional[float] = None,
    tokens_thinking: Optional[int] = None,
    source: str = "api",
) -> None:
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "question": question,
            "model_used": model_used,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_thinking": tokens_thinking,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "estimated_cost": estimated_cost,
            "pages_cited": pages_cited,
        }
        with open(LOGS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Request log write failed (request not affected): {e}")