import json
import sys
from pathlib import Path

# Resolve project root (data/golden_dataset.json)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "golden_dataset.json"

def validate():
    if not DATASET_PATH.exists():
        print(f"FAIL: Dataset file not found at {DATASET_PATH}")
        sys.exit(1)

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"FAIL: Unable to parse JSON: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("FAIL: Root JSON structure must be a list.")
        sys.exit(1)

    errors = []

    # 1. Total count check (15 - 20 items)
    if not (15 <= len(data) <= 20):
        errors.append(f"FAIL: Dataset must contain between 15 and 20 items. Found {len(data)} items.")

    required_keys = {"question", "expected_answer", "expected_page"}
    seen_questions = set()
    unanswerable_count = 0

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            errors.append(f"FAIL [Item {idx}]: Must be a JSON object.")
            continue

        # 2. Strict exact keys check
        item_keys = set(item.keys())
        if item_keys != required_keys:
            errors.append(f"FAIL [Item {idx}]: Keys must strictly match {required_keys}. Found: {item_keys}")

        q = item.get("question")
        ans = item.get("expected_answer")
        page = item.get("expected_page")

        # 3. Question check & Duplicate detection
        if not isinstance(q, str) or not q.strip():
            errors.append(f"FAIL [Item {idx}]: 'question' must be a non-empty string.")
        else:
            q_normalized = q.strip().lower()
            if q_normalized in seen_questions:
                errors.append(f"FAIL [Item {idx}]: Duplicate question detected -> \"{q.strip()}\"")
            seen_questions.add(q_normalized)

        # 4. Expected answer check
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"FAIL [Item {idx}]: 'expected_answer' must be a non-empty string.")

        # 5. Expected page range & Unanswerable counter
        if page is not None:
            if not isinstance(page, int) or isinstance(page, bool) or not (1 <= page <= 72):
                errors.append(f"FAIL [Item {idx}]: 'expected_page' must be an integer between 1 and 72, or null. Got: {page}")
        else:
            if isinstance(ans, str) and "not stated in the document" in ans.lower():
                unanswerable_count += 1

    # 6. Unanswerable items requirement check
    if unanswerable_count < 2:
        errors.append(f"FAIL: Must have at least 2 items with 'expected_page' set to null and 'expected_answer' containing 'not stated in the document'. Found {unanswerable_count}.")

    if errors:
        for err in errors:
            print(err)
        sys.exit(1)

    print(f"PASS: Golden dataset is fully valid with {len(data)} items.")

if __name__ == "__main__":
    validate()