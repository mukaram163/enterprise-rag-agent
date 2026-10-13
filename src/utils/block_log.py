import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

BLOCKS_FILE = "data/logs/blocks.jsonl"


def log_block(kind: str, detail: str, session: Optional[str] = None, ip: Optional[str] = None) -> None:
    """kind: rate_limit | injection | pii | faithfulness | cost_ceiling. Never raises."""
    try:
        os.makedirs(os.path.dirname(BLOCKS_FILE), exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "detail": detail[:200],
            "session": session,
            "ip": ip,
        }
        with open(BLOCKS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Block log write failed: {e}")