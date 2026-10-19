import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG = "data/logs/requests.jsonl"


def total_spend(sources=("api", "eval"), since_hours: Optional[float] = None) -> float:
    """Sum estimated spend. since_hours=None means all time; otherwise only the last N hours."""
    total = 0.0
    cutoff = None
    if since_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    try:
        with open(LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("source") not in sources:
                    continue
                if cutoff is not None:
                    try:
                        when = datetime.fromisoformat(r["timestamp"])
                    except (KeyError, ValueError):
                        continue
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    if when < cutoff:
                        continue
                if r.get("route_tier") == "pro" and r.get("would_cost_pro") is not None:
                    total += r["would_cost_pro"]
                elif r.get("estimated_cost") is not None:
                    total += r["estimated_cost"]
    except FileNotFoundError:
        pass
    return total


def check_ceiling(tier: str) -> Tuple[str, Optional[str], bool]:
    """Return (tier, note, blocked). The ceiling is a rolling window (default 24 hours)."""
    soft = float(os.getenv("COST_CEILING_PRO_USD", "0.05"))
    hard = float(os.getenv("COST_CEILING_HARD_USD", "0.10"))
    window = float(os.getenv("COST_CEILING_WINDOW_HOURS", "24"))
    spent = total_spend(sources=("api", "eval", "agent"), since_hours=window)
    if spent >= hard:
        logger.warning(f"COST CEILING: spend ${spent:.4f} in last {window:g}h >= hard limit ${hard:.4f}; blocking request")
        return tier, "hard cost ceiling reached", True
    if spent >= soft and tier == "pro":
        logger.warning(f"COST CEILING: spend ${spent:.4f} in last {window:g}h >= soft limit ${soft:.4f}; Pro routing disabled")
        return "flash", "pro disabled: cost ceiling", False
    return tier, None, False
