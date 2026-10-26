from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class JsonlStore:
    """Append-only completed items; incomplete writes fail visibly on resume."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid checkpoint {path}:{line_no}; preserve and repair the incomplete line") from exc
                    key = row["key"]
                    if key in self.rows:
                        raise ValueError(f"Duplicate checkpoint key: {key}")
                    self.rows[key] = row

    def add(self, key: str, row: dict):
        if key in self.rows:
            raise ValueError(f"Checkpoint already exists: {key}")
        row = {**row, "key": key}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[key] = row
