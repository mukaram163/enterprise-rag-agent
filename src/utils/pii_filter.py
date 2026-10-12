import re
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[EMAIL REDACTED]"),
    ("cnic", re.compile(r"\b\d{5}-\d{7}-\d\b"), "[ID REDACTED]"),
    ("phone", re.compile(r"(?<![\w/])(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)\d{3}[\s-]?\d{4}(?!\w)"), "[PHONE REDACTED]"),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[NUMBER REDACTED]"),
]


def redact(text: str) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for name, rx, repl in PATTERNS:
        text, n = rx.subn(repl, text)
        if n:
            counts[name] = n
    if counts:
        logger.warning(f"PII REDACTED: {counts}")
    return text, counts