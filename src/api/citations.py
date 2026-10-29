"""Finds which source pages an answer actually cites."""
import re


def extract_used_citations(generated_text, chunks):
    valid = {}
    for chunk in chunks:
        fn = getattr(chunk, "file_name", "unknown.pdf")
        pn = getattr(chunk, "page_number", None)
        if pn is not None:
            valid[(fn, int(pn))] = True
    by_page = {}
    for (fn, pn) in valid:
        by_page.setdefault(pn, fn)

    used, seen = [], set()
    # [file.pdf p.21]  (file + page)
    for fn, pg in re.findall(r'\[([^\]\[]+?\.pdf)\s+p\.\s*(\d+)', generated_text):
        key = (fn.strip(), int(pg))
        if key in valid and key not in seen:
            seen.add(key)
            used.append({"file_name": key[0], "page_number": key[1]})
    # [p.21] or [p.21, p.25]  (page only: first file with that page)
    stripped = re.sub(r'\[[^\]\[]+?\.pdf\s+p\.\s*\d+[^\]]*\]', '', generated_text)
    for pg in re.findall(r'p\.\s*(\d+)', stripped):
        pn = int(pg)
        if pn in by_page and (by_page[pn], pn) not in seen:
            seen.add((by_page[pn], pn))
            used.append({"file_name": by_page[pn], "page_number": pn})
    return used
