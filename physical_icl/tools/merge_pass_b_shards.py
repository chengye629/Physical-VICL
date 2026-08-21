#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    for worker in sorted(args.shards_root.glob("worker_*")):
        cards = worker / "cards"
        if not cards.is_dir():
            continue
        for source in sorted(cards.glob("*.json")):
            if source.name in seen:
                raise RuntimeError(f"duplicate card: {source.name}")
            seen.add(source.name)
            shutil.copy2(source, args.output_root / source.name)
            count += 1
    summary = {"cards": count}
    (args.output_root.parent / "merge_summary.json").write_text(
        json.dumps(summary, indent=2) + chr(10),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
