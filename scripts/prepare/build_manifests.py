#!/usr/bin/env python3
"""Build condition-specific JSONL manifests from Physical-ICL summary.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_CONDITIONS = (
    "no_demo",
    "good_follow",
    "good_rule",
    "weak_typed",
    "opposite_typed",
    "irrelevant_follow",
    "bad_typed",
)


def resolve_media(dataset_root: Path, raw_path: str) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    candidates = [dataset_root / raw]
    parts = raw.parts
    if dataset_root.name in parts:
        candidates.insert(0, dataset_root.joinpath(*parts[parts.index(dataset_root.name) + 1 :]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def make_item(case: dict, condition: str, dataset_root: Path, demo: dict | None = None) -> dict:
    if condition == "no_demo":
        demo_path = None
        prompt_key = "prompt_no_demo"
        demo_name = "no_demo"
    else:
        demo_path = str(resolve_media(dataset_root, demo["demo_path"]))
        demo_name = Path(demo["demo_path"]).stem
        prompt_mode = condition.split("_", 1)[1]
        prompt_key = f"prompt_{demo_name}_{prompt_mode}"
    return {
        "item_id": f'{case["case_id"]}__{demo_name}__{condition}',
        "condition": condition,
        "case_id": case["case_id"],
        "task_name": case["task_name"],
        "episode_name": case["episode_name"],
        "init_frame": str(resolve_media(dataset_root, case["image"])),
        "demo_path": demo_path,
        "demo_type": None if demo is None else demo["demo_type"],
        "prompt_key": prompt_key,
        "prompt": case.get(prompt_key),
        "gt_path": str(resolve_media(dataset_root, case["gt_path"])),
        "duration": case.get("duration"),
        "physics_category": case.get("query_macro_group"),
        "event_tag": case.get("query_event_tag"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True, help="local data/physiq_prelim directory")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("manifests/physiq_prelim"))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    summary = args.summary or args.dataset_root / "summary.json"
    cases = json.loads(summary.read_text(encoding="utf-8"))
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    for condition in conditions:
        demo_type = None if condition == "no_demo" else condition.split("_", 1)[0]
        items = []
        for case in cases:
            demos = [None] if demo_type is None else [d for d in case.get("demos", []) if d["demo_type"] == demo_type]
            for demo in demos:
                item = make_item(case, condition, args.dataset_root, demo)
                if not item["prompt"]:
                    counts[f"{condition}:missing_prompt"] += 1
                    continue
                media = [item["init_frame"], item["gt_path"]] + ([item["demo_path"]] if item["demo_path"] else [])
                if not args.allow_missing and any(not Path(path).is_file() for path in media):
                    counts[f"{condition}:missing_media"] += 1
                    continue
                items.append(item)
        output = args.output_root / f"{condition}.jsonl"
        output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
        counts[f"{condition}:included"] = len(items)
        print(f"{condition}: {len(items)} items -> {output}")

    report = {"summary": str(summary), "dataset_root": str(args.dataset_root.resolve()), "counts": dict(sorted(counts.items()))}
    (args.output_root / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
