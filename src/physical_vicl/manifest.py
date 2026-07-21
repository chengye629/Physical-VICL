"""Canonical manifest helpers shared by the model adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


REQUIRED_FIELDS = (
    "item_id",
    "condition",
    "case_id",
    "task_name",
    "init_frame",
    "prompt_key",
    "prompt",
    "gt_path",
)


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest, a JSON list/object, or one YAML item."""
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
    elif path.suffix == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
    for item in items:
        validate_item(item)
    return items


def validate_item(item: dict[str, Any], check_paths: bool = False) -> None:
    missing = [key for key in REQUIRED_FIELDS if key not in item]
    if missing:
        raise ValueError(f"manifest item is missing fields: {', '.join(missing)}")
    if item["condition"] == "no_demo" and item.get("demo_path") is not None:
        raise ValueError("no_demo item must have demo_path=null")
    if item["condition"] != "no_demo" and not item.get("demo_path"):
        raise ValueError("demo-conditioned item must provide demo_path")
    if check_paths:
        fields: Iterable[str] = ("init_frame", "gt_path", "demo_path")
        absent = [str(item[key]) for key in fields if item.get(key) and not Path(item[key]).is_file()]
        if absent:
            raise FileNotFoundError("missing manifest media: " + ", ".join(absent))


def select_item(path: str | Path, item_id: str | None = None, index: int = 0) -> dict[str, Any]:
    items = load_items(path)
    if item_id is not None:
        matches = [item for item in items if item["item_id"] == item_id]
        if len(matches) != 1:
            raise KeyError(f"expected one item_id={item_id!r}, found {len(matches)}")
        return matches[0]
    try:
        return items[index]
    except IndexError as exc:
        raise IndexError(f"manifest has {len(items)} items; index {index} is invalid") from exc
