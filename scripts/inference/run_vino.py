#!/usr/bin/env python3
"""Translate one canonical item to VINO JSON and launch the official runner."""

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
    parser.add_argument("--vino-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/generations/vino"))
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--guidance-scale-image", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    item = select_item(args.manifest, args.item_id, args.index)
    validate_item(item, check_paths=True)
    out_dir = (args.output_root / item["condition"] / item["task_name"] / item["item_id"] / f"seed_{args.seed}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "index": item["item_id"],
        "task": "tiv2v" if item.get("demo_path") else "i2v",
        "caption": item["prompt"],
        "ref_image_paths": [item["init_frame"]],
    }
    if item.get("demo_path"):
        job["ref_video_path"] = item["demo_path"]
    job_path = out_dir / "vino_job.json"
    job_path.write_text(json.dumps([job], indent=2, ensure_ascii=False), encoding="utf-8")
    command = [
        "torchrun", f"--nproc_per_node={args.nproc_per_node}", "inference.py",
        "--json_path", str(job_path), "--output_height", str(args.height),
        "--output_width", str(args.width), "--output_num_frames", str(args.num_frames),
        "--output_path", str(out_dir), "--seed", str(args.seed),
        "--negative_prompt_video", "", "--guidance_scale_image", str(args.guidance_scale_image),
    ]
    (out_dir / "launch.json").write_text(json.dumps({"command": command, "item": item}, indent=2), encoding="utf-8")
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=args.vino_root, check=True)


if __name__ == "__main__":
    main()
