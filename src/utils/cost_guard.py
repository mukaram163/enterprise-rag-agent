import os
import json
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG = "data/logs/requests.jsonl"


def total_spend(sources=("api", "eval")) -> float:
    total = 0.0
    try:
        with open(LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("source") not in sources:
                    continue
                if r.get("route_tier") == "pro" and r.get("would_cost_pro") is not None:
                    total += r["would_cost_pro"]
                elif r.get("estimated_cost") is not None:
                    total += r["estimated_cost"]
    except FileNotFoundError:
        pass
    return total


def check_ceiling(tier: str) -> Tuple[str, Optional[str], bool]:
    """Return (tier, note, blocked). note is added to the route reasons."""
    soft = float(os.getenv("COST_CEILING_PRO_USD", "0.05"))
    hard = float(os.getenv("COST_CEILING_HARD_USD", "0.10"))
    spent = total_spend()
    if spent >= hard:
        logger.warning(f"COST CEILING: spend ${spent:.4f} >= hard limit ${hard:.4f}; blocking request")
        return tier, "hard cost ceiling reached", True
    if spent >= soft and tier == "pro":
        logger.warning(f"COST CEILING: spend ${spent:.4f} >= soft limit ${soft:.4f}; Pro routing disabled")
        return "flash", "pro disabled: cost ceiling", False
    return tier, None, False