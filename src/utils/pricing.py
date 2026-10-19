from datetime import date
from typing import Optional

# USD per 1M tokens: model -> list of (effective_from, input_rate, output_rate)
PRICING = {
    "gemini-3.8-flash": [
        (date(2026, 1, 1), 0.75, 3.75),
        (date(2027, 1, 1), 1.50, 7.50),
    ],
    "gemini-2.5-flash": [(date(2025, 1, 1), 0.30, 2.50)],
    "gemini-2.5-pro": [(date(2025, 1, 1), 1.25, 10.00)],
    "openai/gpt-oss-120b": [(date(2025, 1, 1), 0.15, 0.60)],  # Groq list price, Oct 2026
    "gemini-3.1-pro-preview": [(date(2025, 1, 1), 2.00, 12.00)],
}


def _rates(model: str, today: date):
    entries = PRICING.get(model)
    if not entries:
        return None
    current = None
    for start, rin, rout in sorted(entries):
        if today >= start:
            current = (rin, rout)
    return current


def estimate_cost(
    model: str,
    tokens_in: Optional[int],
    tokens_out: Optional[int],
    tokens_thinking: Optional[int] = None,
    today: Optional[date] = None,
) -> Optional[float]:
    """Estimated cost in USD at paid rates. None if unknown (never guess)."""
    if tokens_in is None or tokens_out is None:
        return None
    rates = _rates(model, today or date.today())
    if rates is None:
        return None
    rin, rout = rates
    # thinking tokens are billed at the output rate
    out_total = tokens_out + (tokens_thinking or 0)
    return round((tokens_in * rin + out_total * rout) / 1_000_000, 8)