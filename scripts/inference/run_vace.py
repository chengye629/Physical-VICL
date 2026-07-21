#!/usr/bin/env python3
"""Launch VACE's official pipeline for a Physical-VICL manifest item."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from physical_vicl.manifest import select_item, validate_item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--item-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--vace-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/generations/vace"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="vace-14B")
    parser.add_argument("--size", default="720p")
    parser.add_argument("--animate-mode", default="salientbboxtrack")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    item = select_item(args.manifest, args.item_id, args.index)
    validate_item(item, check_paths=True)
    out_dir = (args.output_root / item["condition"] / item["task_name"] / item["item_id"] / f"seed_{args.seed}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "vace/vace_pipeline.py", "--base", "wan",
        "--task", "animate_anything" if item.get("demo_path") else "frameref",
        "--mode", args.animate_mode if item.get("demo_path") else "firstframe",
        "--image", item["init_frame"], "--prompt", item["prompt"],
        "--ckpt_dir", args.checkpoint, "--model_name", args.model_name,
        "--size", args.size, "--base_seed", str(args.seed),
        "--save_dir", str(out_dir), "--save_file", str(out_dir / "video.mp4"),
    ]
    if item.get("demo_path"):
        command.extend(["--video", item["demo_path"]])
    (out_dir / "launch.json").write_text(json.dumps({"command": command, "item": item}, indent=2), encoding="utf-8")
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=args.vace_root, check=True)


if __name__ == "__main__":
    main()
