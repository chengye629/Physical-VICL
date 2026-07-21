#!/usr/bin/env python3
"""Run one item through UniVideo's I2V or image-set in-context generation path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils import export_to_video

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from physical_vicl.manifest import select_item, validate_item


NEGATIVE_PROMPT = (
    "static, weak dynamics, distorted motion, unstable framing, morphing, identity drift, "
    "teleportation, flicker, low quality, blurry, watermark, text"
)


def sample_video(video: str, output_dir: Path, count: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", video],
        capture_output=True, text=True, check=True,
    )
    total = int(probe.stdout.strip())
    if total < count:
        raise ValueError(f"demo has {total} frames, fewer than requested sample count {count}")
    indices = [round(i * (total - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    paths = []
    for slot, frame_index in enumerate(indices):
        path = output_dir / f"demo_{slot:02d}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
             "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", str(path)],
            check=True,
        )
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--item-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--univideo-root", type=Path, required=True)
    parser.add_argument("--config", default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml")
    parser.add_argument("--transformer-checkpoint")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/generations/univideo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-frames", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=129)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    item = select_item(args.manifest, args.item_id, args.index)
    validate_item(item, check_paths=True)
    sys.path.insert(0, str(args.univideo_root))
    from pipeline_univideo import UniVideoPipeline
    from train.train_univideo import ModelArguments, get_torch_dtype, load_univideo_components
    from utils import pad_image_pil_to_square

    out_dir = args.output_root / item["condition"] / item["task_name"] / item["item_id"] / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = get_torch_dtype("bf16")
    model_args = ModelArguments(
        config=str((args.univideo_root / args.config).resolve()),
        transformer_ckpt_path=args.transformer_checkpoint,
        gradient_checkpointing=False,
        dtype="bf16",
    )
    transformer, vae, scheduler, mllm_encoder, pipe_cfg = load_univideo_components(
        model_args, device=torch.device("cuda"), dtype=dtype
    )
    pipe = UniVideoPipeline(
        transformer=transformer, vae=vae, scheduler=scheduler,
        mllm_encoder=mllm_encoder, univideo_config=pipe_cfg,
    ).to(device="cuda", dtype=dtype)

    kwargs = dict(
        prompts=[item["prompt"]], negative_prompt=NEGATIVE_PROMPT,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.steps, guidance_scale=5.0,
        image_guidance_scale=3.0, seed=args.seed, timestep_shift=7.0,
    )
    sampled = []
    if item.get("demo_path"):
        sampled = sample_video(item["demo_path"], out_dir / "inputs", args.demo_frames)
        refs = sampled + [Path(item["init_frame"])]
        kwargs.update(
            ref_images=[[pad_image_pil_to_square(Image.open(path).convert("RGB")) for path in refs]],
            task="multiid",
        )
    else:
        kwargs.update(cond_image_path=item["init_frame"], image_guidance_scale=1.0, task="i2v")

    frames = pipe(**kwargs).frames[0]
    export_to_video(frames, str(out_dir / "video.mp4"), fps=16)
    metadata = {**item, "model": "UniVideo", "config": args.config,
                "transformer_checkpoint": args.transformer_checkpoint, "seed": args.seed,
                "demo_frame_paths": [str(path) for path in sampled],
                "adapter_note": "with-demo uses sampled demo frames plus the query image in official multiid mode"}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_dir / "video.mp4")


if __name__ == "__main__":
    main()
