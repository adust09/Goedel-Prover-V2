import json
from pathlib import Path
from typing import Any, Iterable


def jload(path: str | Path) -> Any:
    p = Path(path)
    if p.suffix == ".jsonl":
        items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    else:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)


def jsave(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".jsonl":
        # Expect iterable of JSON-serializable items
        if not isinstance(obj, Iterable):
            raise TypeError("jsonl output expects an iterable of items")
        with p.open("w", encoding="utf-8") as f:
            for x in obj:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    else:
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

