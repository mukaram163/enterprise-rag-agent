import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List
from src.utils.pricing import estimate_cost

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
    tier: Optional[str] = None,
    route_reasons: Optional[List[str]] = None,
) -> None:
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        if estimated_cost is None:
            estimated_cost = estimate_cost(model_used, tokens_in, tokens_out, tokens_thinking)
        
        would_cost_pro = None
        pro_model = os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
        if tier is not None:
            would_cost_pro = estimate_cost(pro_model, tokens_in, tokens_out, tokens_thinking)

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
            "route_mode": "dry_run",
            "route_tier": tier,
            "route_reasons": route_reasons,
            "would_cost_pro": would_cost_pro,
        }
        with open(LOGS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Request log write failed (request not affected): {e}")