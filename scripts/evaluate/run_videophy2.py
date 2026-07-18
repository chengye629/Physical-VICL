#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", x)


def extract_prompt(raw_text: str, mode: str = "query") -> str:
    raw_text = (raw_text or "").strip()
    if mode == "full":
        return raw_text

    marker = "Query:"
    if marker in raw_text:
        q = raw_text.split(marker, 1)[1].strip()
        chunks = [x.strip() for x in q.split("\n\n") if x.strip()]
        if chunks:
            return chunks[0]
        return q

    return raw_text


def probe_video_duration(video_path: str):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0 and frames and frames > 0:
            return float(frames) / float(fps)
    except Exception:
        pass
    return None


def decide_num_frames(duration, target_fps: float, min_frames: int, max_frames: int, default_num_frames: int):
    if duration is None or duration <= 0:
        return int(default_num_frames)
    n = int(round(float(duration) * float(target_fps)))
    return max(int(min_frames), min(int(max_frames), n))


def collect_items(
    data_root: Path,
    exclude_keyword: str,
    include_raw_composite: bool,
    prompt_mode: str,
    only_types,
    target_fps: float,
    min_frames: int,
    max_frames: int,
    default_num_frames: int,
):
    items = []
    only_types = set(only_types or [])

    for type_dir in sorted(data_root.iterdir()):
        if not type_dir.is_dir():
            continue

        type_name = type_dir.name

        if only_types and type_name not in only_types:
            continue

        if exclude_keyword and exclude_keyword.lower() in type_name.lower():
            continue

        gen_root = type_dir / "generated_data"
        if not gen_root.exists():
            continue

        for mp4 in sorted(gen_root.glob("*/*/*/*/*.mp4")):
            if not include_raw_composite and mp4.name == "raw_composite.mp4":
                continue

            rel = mp4.relative_to(data_root)
            parts = rel.parts

            # type/generated_data/model/task/episode/run/file.mp4
            if len(parts) < 7 or parts[1] != "generated_data":
                continue

            model = parts[2]
            task = parts[3]
            episode = parts[4]
            run_id = parts[5]

            prompt_path = mp4.parent / "prompt" / "prompt.txt"
            if not prompt_path.exists():
                prompt = ""
                raw_prompt = ""
                err = f"missing prompt: {prompt_path}"
            else:
                raw_prompt = prompt_path.read_text(encoding="utf-8", errors="ignore")
                prompt = extract_prompt(raw_prompt, prompt_mode)
                err = ""

            duration = probe_video_duration(str(mp4))
            num_frames = decide_num_frames(
                duration=duration,
                target_fps=target_fps,
                min_frames=min_frames,
                max_frames=max_frames,
                default_num_frames=default_num_frames,
            )

            items.append({
                "id": "__".join(rel.parts),
                "type": type_name,
                "dataset": type_name,
                "model": model,
                "task": task,
                "episode": episode,
                "run": run_id,
                "video": str(mp4),
                "prompt_path": str(prompt_path),
                "prompt": prompt,
                "raw_prompt": raw_prompt,
                "duration_sec": duration,
                "num_frames": num_frames,
                "error": err,
            })

    return items


def write_input_csv(items, input_csv: Path, meta_jsonl: Path):
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["videopath", "caption"])
        writer.writeheader()
        for item in items:
            writer.writerow({
                "videopath": item["video"],
                "caption": item["prompt"],
            })

    with meta_jsonl.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_videophy2_task(videophy2_dir: Path, checkpoint: Path, input_csv: Path, output_csv: Path, task: str, num_frames: int):
    infer_py = videophy2_dir / "inference.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"VideoPhy2 inference.py not found: {infer_py}")

    cmd = [
        sys.executable,
        str(infer_py),
        "--input_csv", str(input_csv),
        "--checkpoint", str(checkpoint),
        "--output_csv", str(output_csv),
        "--task", task,
        "--num_frames", str(num_frames),
    ]

    print("[RUN]", " ".join(cmd), flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(videophy2_dir) + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        cmd,
        cwd=str(videophy2_dir),
        env=env,
        check=True,
    )

    print(f"[DONE] task={task}, num_frames={num_frames}, elapsed={time.time() - t0:.1f}s", flush=True)


def to_builtin(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    if pd.isna(x):
        return None
    return x


def clamp_score(x):
    if x is None:
        return None
    try:
        v = int(round(float(x)))
        return max(1, min(5, v))
    except Exception:
        return None


def infer_score_column(df: pd.DataFrame):
    for col in ["score", "Score", "pred_score", "prediction", "pred", "output", "rating"]:
        if col in df.columns:
            return col

    numeric_cols = []
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            numeric_cols.append(col)

    if numeric_cols:
        return numeric_cols[-1]

    raise RuntimeError(f"Cannot infer score column from columns: {list(df.columns)}")


def load_scores(csv_path: Path, expected_n: int):
    candidates = []

    try:
        df1 = pd.read_csv(csv_path)
        candidates.append(("header", df1))
    except Exception as e:
        print(f"[WARN] read header failed {csv_path}: {e}", flush=True)

    try:
        df2 = pd.read_csv(csv_path, header=None)
        candidates.append(("no_header", df2))
    except Exception as e:
        print(f"[WARN] read no-header failed {csv_path}: {e}", flush=True)

    if not candidates:
        raise RuntimeError(f"Cannot read CSV: {csv_path}")

    mode, df = candidates[0]
    for m, d in candidates:
        if len(d) == expected_n:
            mode, df = m, d
            break

    score_col = infer_score_column(df)
    print(f"[INFO] {csv_path.name} read_mode={mode} score_col={score_col}, rows={len(df)}, expected={expected_n}", flush=True)

    scores = []
    rows = []
    for _, row in df.iterrows():
        raw_row = {str(k): to_builtin(v) for k, v in row.to_dict().items()}
        rows.append(raw_row)
        scores.append(clamp_score(raw_row.get(str(score_col))))

    return scores, rows, str(score_col)


def append_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


def merge_group(items, sa_csv: Path, pc_csv: Path, out_path: Path, num_frames: int, target_fps: float):
    n = len(items)
    sa_scores, sa_rows, sa_col = load_scores(sa_csv, expected_n=n)
    pc_scores, pc_rows, pc_col = load_scores(pc_csv, expected_n=n)

    if len(sa_scores) != n or len(pc_scores) != n:
        raise RuntimeError(f"Row count mismatch: items={n}, sa={len(sa_scores)}, pc={len(pc_scores)}")

    records = []
    for i, item in enumerate(items):
        sa = sa_scores[i]
        pc = pc_scores[i]

        err = ""
        if item.get("error"):
            err = item["error"]
        elif sa is None or pc is None:
            err = "failed to parse VideoPhy2 SA/PC score"

        records.append({
            "id": item["id"],
            "type": item["type"],
            "dataset": item["dataset"],
            "model": item["model"],
            "task": item["task"],
            "episode": item["episode"],
            "run": item["run"],
            "video": item["video"],
            "prompt_path": item["prompt_path"],
            "prompt": item["prompt"],
            "physical_adherence": pc,
            "instruction_alignment": sa,
            "physical_reasoning": "Mapped from VideoPhy2 physical commonsense score.",
            "instruction_reasoning": "Mapped from VideoPhy2 semantic adherence score.",
            "raw": {
                "backend": "videophy2",
                "semantic_adherence": sa,
                "physical_commonsense": pc,
                "joint_pass": (sa is not None and pc is not None and sa >= 4 and pc >= 4),
                "sa_score_col": sa_col,
                "pc_score_col": pc_col,
                "sa_row": sa_rows[i],
                "pc_row": pc_rows[i],
                "settings": {
                    "sampling": "auto_num_frames_from_duration",
                    "target_fps": target_fps,
                    "num_frames": num_frames,
                    "duration_sec": item.get("duration_sec"),
                },
            },
            "error": err,
        })

    append_jsonl(out_path, records)
    print(f"[INFO] appended {len(records)} records -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/afs/L202500199/Physical-ICL-Video/data")
    ap.add_argument("--videophy2-dir", default="/mnt/afs/L202500199/mllm_as_embodied_world_judge/third_party/videophy/VIDEOPHY2")
    ap.add_argument("--checkpoint", default="/mnt/afs/L202500199/mllm_as_embodied_world_judge/checkpoints/videophy_2_auto")
    ap.add_argument("--out-dir", default="/mnt/afs/L202500199/Physical-ICL-Video/outputs/videophy2/by_type")
    ap.add_argument("--tmp-dir", default="/mnt/afs/L202500199/Physical-ICL-Video/tmp/videophy2_by_type")
    ap.add_argument("--exclude-keyword", default="opposite")
    ap.add_argument("--prompt-mode", choices=["query", "full"], default="query")
    ap.add_argument("--include-raw-composite", action="store_true")
    ap.add_argument("--only-types", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target-fps", type=float, default=4.0)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--default-num-frames", type=int, default=28)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    videophy2_dir = Path(args.videophy2_dir)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)

    only_types = None
    if args.only_types:
        only_types = [x.strip() for x in args.only_types.split(",") if x.strip()]

    print("===== Physical-ICL VideoPhy2 by type =====", flush=True)
    print("data_root:", data_root, flush=True)
    print("videophy2_dir:", videophy2_dir, flush=True)
    print("checkpoint:", checkpoint, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("exclude_keyword:", args.exclude_keyword, flush=True)
    print("prompt_mode:", args.prompt_mode, flush=True)
    print("only_types:", only_types, flush=True)

    items = collect_items(
        data_root=data_root,
        exclude_keyword=args.exclude_keyword,
        include_raw_composite=args.include_raw_composite,
        prompt_mode=args.prompt_mode,
        only_types=only_types,
        target_fps=args.target_fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        default_num_frames=args.default_num_frames,
    )

    print("collected items:", len(items), flush=True)

    if args.limit == 0:
        by_type = defaultdict(int)
        by_frames = defaultdict(int)
        for x in items:
            by_type[x["type"]] += 1
            by_frames[(x["type"], x["num_frames"])] += 1

        print("items by type:", flush=True)
        for k, v in sorted(by_type.items()):
            print(f"  {k}: {v}", flush=True)

        print("items by type/frame:", flush=True)
        for k, v in sorted(by_frames.items()):
            print(f"  {k}: {v}", flush=True)

        return

    if args.limit is not None:
        items = items[:args.limit]

    # type -> num_frames -> items
    groups = defaultdict(lambda: defaultdict(list))
    for item in items:
        groups[item["type"]][int(item["num_frames"])].append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous outputs for selected types.
    for type_name in groups:
        out_path = out_dir / f"judge_videophy2_{safe_name(type_name)}.jsonl"
        if out_path.exists():
            out_path.unlink()

    run_id = time.strftime("%Y%m%d_%H%M%S")

    for type_name in sorted(groups):
        out_path = out_dir / f"judge_videophy2_{safe_name(type_name)}.jsonl"

        print("\n" + "=" * 80, flush=True)
        print(f"[TYPE] {type_name}", flush=True)
        print("=" * 80, flush=True)

        for num_frames in sorted(groups[type_name]):
            group_items = groups[type_name][num_frames]
            tag = f"{run_id}_{safe_name(type_name)}_f{num_frames}"

            print("\n" + "-" * 80, flush=True)
            print(f"[GROUP] type={type_name}, num_frames={num_frames}, items={len(group_items)}", flush=True)
            print("-" * 80, flush=True)

            input_csv = tmp_dir / f"input_{tag}.csv"
            meta_jsonl = tmp_dir / f"meta_{tag}.jsonl"
            sa_csv = tmp_dir / f"output_sa_{tag}.csv"
            pc_csv = tmp_dir / f"output_pc_{tag}.csv"

            write_input_csv(group_items, input_csv, meta_jsonl)

            run_videophy2_task(videophy2_dir, checkpoint, input_csv, sa_csv, "sa", num_frames)
            run_videophy2_task(videophy2_dir, checkpoint, input_csv, pc_csv, "pc", num_frames)

            merge_group(
                group_items,
                sa_csv,
                pc_csv,
                out_path,
                num_frames=num_frames,
                target_fps=args.target_fps,
            )

    print("[DONE]", flush=True)
    print("outputs:", out_dir, flush=True)


if __name__ == "__main__":
    main()
