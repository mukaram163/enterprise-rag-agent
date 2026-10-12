import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

PATTERNS = [
    (r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)", "ignore-instructions"),
    (r"disregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|system)", "disregard-instructions"),
    (r"(reveal|show|print|repeat|leak|display)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules)", "reveal-prompt"),
    (r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?)", "reveal-prompt"),
    (r"you\s+are\s+now\s+(a|an|in)\b", "role-override"),
    (r"\b(act|pretend|behave)\s+as\s+(if\s+you\s+(are|were)\s+)", "role-override"),
    (r"\b(developer|debug|dan|jailbreak)\s+mode\b", "mode-switch"),
    (r"(override|bypass|disable)\s+(your\s+|the\s+)?(safety|guardrails?|filters?|restrictions?)", "bypass-safety"),
]
COMPILED = [(re.compile(p, re.I), name) for p, name in PATTERNS]


def check_injection(text: str) -> Tuple[bool, List[str]]:
    hits = [name for rx, name in COMPILED if rx.search(text)]
    if hits:
        logger.warning(f"INJECTION ATTEMPT: rules={hits} text={text[:120]!r}")
    return bool(hits), hits