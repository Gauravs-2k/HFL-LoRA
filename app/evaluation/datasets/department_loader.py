import json
from .formatters import normalize_text


def load_department_dataset(path, max_samples=None):
    """
    Loads a department dataset in JSONL format and unifies into:
        { "input": str, "target": str }

    Supports formats like:
        {"text": "...", "response": "..."}
        {"instruction": "...", "output": "..."}
        {"prompt": "...", "response": "..."}
    """
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Try all the input-style keys you might have
            inp = (
                obj.get("instruction")
                or obj.get("prompt")
                or obj.get("input")
                or obj.get("text")  # <- important for your *_dept.jsonl
            )

            # Try all the target-style keys
            out = (
                obj.get("output")
                or obj.get("response")
                or obj.get("label")
            )

            if not inp or not out:
                continue

            items.append(
                {
                    "input": normalize_text(str(inp)),
                    "target": normalize_text(str(out)),
                }
            )

            if max_samples and len(items) >= max_samples:
                break

    return items
