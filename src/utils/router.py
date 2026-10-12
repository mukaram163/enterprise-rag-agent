import re
from typing import List, Tuple

HEAVY_WORDS = re.compile(
    r"\b(compare|comparison|difference|explain|summari[sz]e|breakdown|analy[sz]e|why)\b", re.I
)
SUBQUESTION = re.compile(r"(,|\band\b)\s+(how|what|which|why|when|who)\b", re.I)

LONG_QUESTION_WORDS = 20
PRO_THRESHOLD = 2


def classify(question: str) -> Tuple[str, List[str]]:
    """Return ('flash' | 'pro', reasons). Plain rules, no model."""
    reasons = []
    if len(question.split()) >= LONG_QUESTION_WORDS:
        reasons.append("long question")
    if SUBQUESTION.search(question) or question.count("?") > 1:
        reasons.append("multiple sub-questions")
    if HEAVY_WORDS.search(question):
        reasons.append("heavy keyword")
    tier = "pro" if len(reasons) >= PRO_THRESHOLD else "flash"
    return tier, reasons