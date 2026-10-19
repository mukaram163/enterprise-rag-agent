from src.utils.block_log import log_block
from src.utils.injection_guard import check_injection


def filter_chunks(chunks, session: str = "agent-tool"):
    """Drop retrieved chunks that contain instructions aimed at the AI (indirect injection)."""
    kept, removed = [], 0
    for c in chunks:
        flagged, rules = check_injection(c.text)
        if flagged:
            removed += 1
            log_block("injection", f"indirect: {c.file_name} p.{c.page_number} rules={rules}",
                      session=session, ip="local")
        else:
            kept.append(c)
    return kept, removed
