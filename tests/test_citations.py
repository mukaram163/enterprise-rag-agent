from types import SimpleNamespace as C

from src.api.citations import extract_used_citations

CHUNKS = [C(file_name="a.pdf", page_number=3), C(file_name="a.pdf", page_number=5)]


def cite(text):
    return extract_used_citations(text, CHUNKS)


def test_file_and_page_form():
    assert cite("Yes [a.pdf p.3].") == [{"file_name": "a.pdf", "page_number": 3}]


def test_page_only_form_finds_the_file():
    assert cite("Yes [p.5].") == [{"file_name": "a.pdf", "page_number": 5}]


def test_page_not_in_retrieved_chunks_is_ignored():
    assert cite("Made up [a.pdf p.9] and [p.9].") == []


def test_duplicates_are_cited_once():
    assert len(cite("[a.pdf p.3] again [a.pdf p.3] and p.3")) == 1


def test_mixed_forms_keep_order():
    assert [c["page_number"] for c in cite("[a.pdf p.5] then [p.3]")] == [5, 3]


def test_no_citations():
    assert cite("No markers here.") == []
