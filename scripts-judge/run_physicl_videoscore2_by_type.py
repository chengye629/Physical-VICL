#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", x)


def load_videoscore2_judge(judge_root: Path):
    sys.path.insert(0, str(judge_root))
    from judge.videoscore2_judge import VideoScore2Judge
    return VideoScore2Judge


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


def collect_items(data_root: Path, exclude_keyword: str, include_raw_composite: bool, prompt_mode: str, only_types=None):
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
                "error": err,
            })

    return items


def read_done_ids(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("id"):
                    done.add(r["id"])
            except Exception:
                pass
    return done


def write_record(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/afs/L202500199/Physical-ICL-Video/data")
    ap.add_argument("--judge-root", default="/mnt/afs/L202500199/mllm_as_embodied_world_judge")
    ap.add_argument("--checkpoint", default="/mnt/afs/L202500199/mllm_as_embodied_world_judge/checkpoints/VideoScore2")
    ap.add_argument("--out-dir", default="/mnt/afs/L202500199/Physical-ICL-Video/outputs/videoscore2/by_type")
    ap.add_argument("--exclude-keyword", default="opposite")
    ap.add_argument("--prompt-mode", choices=["query", "full"], default="query")
    ap.add_argument("--include-raw-composite", action="store_true")
    ap.add_argument("--only-types", default=None, help="comma-separated top-level type folders")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--infer-fps", type=float, default=4.0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    judge_root = Path(args.judge_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    only_types = None
    if args.only_types:
        only_types = [x.strip() for x in args.only_types.split(",") if x.strip()]

    print("===== Physical-ICL VideoScore2 by type =====", flush=True)
    print("data_root:", data_root, flush=True)
    print("judge_root:", judge_root, flush=True)
    print("checkpoint:", args.checkpoint, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("exclude_keyword:", args.exclude_keyword, flush=True)
    print("prompt_mode:", args.prompt_mode, flush=True)
    print("only_types:", only_types, flush=True)
    print("infer_fps:", args.infer_fps, flush=True)

    items = collect_items(
        data_root=data_root,
        exclude_keyword=args.exclude_keyword,
        include_raw_composite=args.include_raw_composite,
        prompt_mode=args.prompt_mode,
        only_types=only_types,
    )

    print("collected items:", len(items), flush=True)

    by_type = {}
    for item in items:
        by_type.setdefault(item["type"], 0)
        by_type[item["type"]] += 1

    print("items by type:", flush=True)
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}", flush=True)

    if args.limit == 0:
        print("[DONE] count only", flush=True)
        return

    if not args.resume:
        for type_name in by_type:
            p = out_dir / f"judge_videoscore2_{safe_name(type_name)}.jsonl"
            if p.exists():
                p.unlink()

    done_by_type = {}
    if args.resume:
        for type_name in by_type:
            p = out_dir / f"judge_videoscore2_{safe_name(type_name)}.jsonl"
            done_by_type[type_name] = read_done_ids(p)
    else:
        done_by_type = {k: set() for k in by_type}

    VideoScore2Judge = load_videoscore2_judge(judge_root)
    judge = VideoScore2Judge(
        model=args.checkpoint,
        infer_fps=args.infer_fps,
        use_num_frames=False,
        local_files_only=True,
    )

    n_run = 0

    for item in items:
        if args.limit is not None and n_run >= args.limit:
            break

        type_name = item["type"]
        out_path = out_dir / f"judge_videoscore2_{safe_name(type_name)}.jsonl"

        if item["id"] in done_by_type.get(type_name, set()):
            continue

        print(f"[{n_run + 1}] {type_name}/{item['model']}/{item['task']}/{item['episode']}/{item['run']}", flush=True)

        if item.get("error"):
            rec = {
                "id": item["id"],
                "type": type_name,
                "dataset": item["dataset"],
                "model": item["model"],
                "task": item["task"],
                "episode": item["episode"],
                "run": item["run"],
                "video": item["video"],
                "prompt_path": item["prompt_path"],
                "prompt": item.get("prompt", ""),
                "physical_adherence": None,
                "instruction_alignment": None,
                "physical_reasoning": "",
                "instruction_reasoning": "",
                "raw": {},
                "error": item["error"],
            }
            write_record(out_path, rec)
            n_run += 1
            continue

        result = judge.judge(
            video_path=item["video"],
            instruction=item["prompt"],
            init_frame_path=None,
        )

        rec = {
            "id": item["id"],
            "type": type_name,
            "dataset": item["dataset"],
            "model": item["model"],
            "task": item["task"],
            "episode": item["episode"],
            "run": item["run"],
            "video": item["video"],
            "prompt_path": item["prompt_path"],
            "prompt": item["prompt"],
            "physical_adherence": result.physical_adherence,
            "instruction_alignment": result.instruction_alignment,
            "physical_reasoning": result.physical_reasoning,
            "instruction_reasoning": result.instruction_reasoning,
            "raw": result.raw,
            "error": result.error or "",
        }

        write_record(out_path, rec)

        print(f"  -> PA={rec['physical_adherence']} IA={rec['instruction_alignment']} ERR={bool(rec['error'])}", flush=True)
        n_run += 1

    print("[DONE]", flush=True)
    print("new records:", n_run, flush=True)
    print("outputs:", out_dir, flush=True)


if __name__ == "__main__":
    main()
