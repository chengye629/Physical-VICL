#!/usr/bin/env python3
"""Run one canonical manifest item with Video-As-Prompt or its no-demo base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from physical_vicl.manifest import select_item, validate_item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--item-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--vap-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/generations/vap"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--ref-prompt", default="The reference video demonstrates a physical interaction. Transfer its motion and physical rule to the query scene without copying its objects or background.")
    args = parser.parse_args()

    item = select_item(args.manifest, args.item_id, args.index)
    validate_item(item, check_paths=True)
    sys.path.insert(0, str(args.vap_root / "infer"))
    from cog_vap import export_via_tmp, select_frames, set_global_seed
    from diffusers import (
        AutoencoderKLCogVideoX,
        CogVideoXImageToVideoMOTPipeline,
        CogVideoXImageToVideoPipeline,
        CogVideoXTransformer3DMOTModel,
    )
    from diffusers.utils import load_video

    set_global_seed(args.seed)
    image = Image.open(item["init_frame"]).convert("RGB")
    if item.get("demo_path"):
        vae = AutoencoderKLCogVideoX.from_pretrained(args.checkpoint, subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = CogVideoXTransformer3DMOTModel.from_pretrained(
            args.checkpoint, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = CogVideoXImageToVideoMOTPipeline.from_pretrained(
            args.checkpoint, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16
        )
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload() if args.cpu_offload else pipe.to("cuda")

    kwargs = dict(
        image=image,
        prompt=item["prompt"],
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        use_dynamic_cfg=True,
    )
    if item.get("demo_path"):
        reference = load_video(item["demo_path"])
        kwargs.update(
            ref_videos=[select_frames(reference, num=49, mode="evenly")],
            prompt_mot_ref=[args.ref_prompt],
            frames_selection="evenly",
        )

    out_dir = args.output_root / item["condition"] / item["task_name"] / item["item_id"] / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = pipe(**kwargs).frames[0]
    export_via_tmp(frames, str(out_dir / "video.mp4"), fps=args.fps)
    metadata = {**item, "model": "Video-As-Prompt-CogVideoX-5B", "checkpoint": args.checkpoint,
                "seed": args.seed, "num_frames": args.num_frames, "height": args.height,
                "width": args.width, "fps": args.fps, "cpu_offload": args.cpu_offload}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_dir / "video.mp4")


if __name__ == "__main__":
    main()
