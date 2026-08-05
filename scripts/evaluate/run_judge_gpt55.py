#!/usr/bin/env python3
"""Judge wan22_i2v_a14b baseline videos on Physical-ICL using GPT-5.5.

Downloads videos from HF, samples at 4 FPS, sends to GPT-5.5 via Azure proxy,
saves JSONL results with resume support.

Usage:
    python run_judge_gpt55.py --benchmark physiq_i2v720
    python run_judge_gpt55.py --benchmark mme_cof_pro_physical
    python run_judge_gpt55.py --benchmark vbvr_physics
    python run_judge_gpt55.py --all
"""

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.request

import cv2
from openai import AzureOpenAI

HF_BASE = "https://huggingface.co/datasets/Vincwng/Physical-ICL/resolve/main"
BENCHMARKS = ["physiq_i2v720", "mme_cof_pro_physical", "vbvr_physics"]
MODEL_DIR = "wan22_i2v_a14b"

JUDGE_MODEL = "gpt-5.5-2026-04-24"
JUDGE_REVISION = "gpt-5.5-2026-04-24-azure"
JUDGE_PROMPT_VERSION = "physical_vicl_v1"

AZURE_AK = "9BTUCgEM2zN0GmKnXAB4Y5pyscee6Ab8_GPT_AK"
AZURE_ENDPOINT = "https://search.bytedance.net/gpt/openapi/online/multimodal/crawl"

PROXY = "http://sys-proxy-rd-relay.byted.org:8118"

FPS = 4.0
MAX_FRAMES = 64
JPEG_QUALITY = 80
MAX_SIDE = 720

JUDGE_SYSTEM = ""
JUDGE_PROMPT_TEMPLATE = """You are a strict, calibrated evaluator of the PHYSICAL REALISM and EVENT REALIZATION of AI-generated videos.

You are given the generation prompt / task instruction and uniformly sampled frames from one generated video in temporal order. First identify the intended physical event from the prompt. Then judge whether the event visibly occurs, progresses in the correct causal order, and has a physically plausible outcome.

Score the observable video, not the likely real-world outcome. Do not give credit merely because the initial scene looks realistic or remains visually stable. If the required event never occurs, is replaced by a different event, or cannot be verified in the sampled frames, this must substantially lower the score. Do not penalize visual style, camera aesthetics, or minor appearance errors unless they obscure the event or create a physical inconsistency.

Generation prompt / task instruction:
{generation_prompt}

Task: Judge how completely and physically realistically the generated video realizes the intended event.

Criteria (address each criterion with concrete temporal evidence and note any other physical issue):

1. Event realization and completeness: the prompted physical event visibly starts, develops, and reaches the expected type of outcome. Required objects participate in the event, and lack of motion or an incomplete event is not treated as success.

2. Entity and material consistency: objects, bodies, and materials remain structurally coherent; they do not inexplicably melt, fuse, split, stretch, or deform.

3. Scene and object continuity: the environment and objects remain temporally stable, without unexplained flicker, teleportation, morphing, duplication, disappearance, or identity changes.

4. Interaction and causal realism: collisions, support, grasping, containment, cutting, tearing, pouring, and other contacts behave plausibly. Objects do not interpenetrate, pass through barriers, float without support, react before contact, or produce effects without a visible cause.

5. Motion and physical-law consistency: motion follows applicable gravity, inertia, momentum, friction, rigidity/deformation, and fluid behavior. Trajectories are continuous and the final outcome is plausible for the visible materials and interactions.

Score (integer 1-5):
1 = the intended event is absent/unrecognizable, or the video contains gross and pervasive physical violations;
2 = the event substantially fails or remains incomplete, or major violations break the physical outcome;
3 = the event is recognizable, but noticeable local inconsistencies or an uncertain/incomplete outcome remain;
4 = the event is complete and physically plausible overall, with only minor issues;
5 = the event is clearly complete, causally correct, temporally coherent, and has no visible physical violation.

Reason first, explicitly stating whether the intended event occurred and completed. Then score. Output JSON only with two keys: `reasoning` (string) and `physical_adherence` (integer 1-5)."""


def setup_proxy():
    os.environ.setdefault("http_proxy", PROXY)
    os.environ.setdefault("https_proxy", PROXY)
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    proxy_handler = urllib.request.ProxyHandler({
        "http": PROXY,
        "https": PROXY,
    })
    opener = urllib.request.build_opener(proxy_handler)
    urllib.request.install_opener(opener)


def get_client():
    return AzureOpenAI(
        api_key=AZURE_AK,
        api_version="2023-07-01-preview",
        azure_endpoint=AZURE_ENDPOINT,
        timeout=120,
    )


def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            if os.path.exists(dest):
                os.remove(dest)
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def sample_frames_4fps(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        return []
    duration = total / src_fps
    n_frames = min(MAX_FRAMES, max(1, int(math.ceil(duration * FPS))))
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            scale = min(MAX_SIDE / w, MAX_SIDE / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            frames_b64.append(base64.b64encode(buf).decode())
    cap.release()
    return frames_b64


def extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except:
        return None


def judge_video(client, frames_b64, generation_prompt):
    content = [{"type": "text", "text": JUDGE_PROMPT_TEMPLATE.format(generation_prompt=generation_prompt)}]
    for b64 in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=4096,
                seed=42,
            )
            raw = response.choices[0].message.content
            parsed = extract_json(raw)
            if parsed and "physical_adherence" in parsed:
                return {
                    "reasoning": parsed.get("reasoning", ""),
                    "physical_adherence": int(parsed["physical_adherence"]),
                    "raw_response": raw,
                }
            if attempt < 2:
                time.sleep(2)
                continue
            return {
                "reasoning": raw or "",
                "physical_adherence": None,
                "raw_response": raw,
                "parse_error": "no valid JSON with physical_adherence",
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return {
                "error": repr(e)[:300],
                "physical_adherence": None,
            }


def discover_items(benchmark, cache_dir):
    summary_url = f"{HF_BASE}/data/{benchmark}/summary.json"
    summary_path = os.path.join(cache_dir, benchmark, "summary.json")
    download_file(summary_url, summary_path)
    with open(summary_path) as f:
        cases = json.load(f)

    items = []
    for case in cases:
        task = case["task_name"]
        ep = case.get("episode_name", "episode_0001")
        video_rel = f"data/{benchmark}/generated_data/{MODEL_DIR}/{task}/{ep}/1/{task}_{ep}.mp4"
        prompt_rel = f"data/{benchmark}/generated_data/{MODEL_DIR}/{task}/{ep}/1/prompt/prompt.txt"
        items.append({
            "benchmark": benchmark,
            "case_id": case.get("case_id", ""),
            "task_name": task,
            "episode_name": ep,
            "video_url": f"{HF_BASE}/{video_rel}",
            "prompt_url": f"{HF_BASE}/{prompt_rel}",
            "video_rel": video_rel,
            "prompt_rel": prompt_rel,
            "gt_prompt": case.get("prompt", [""])[0] if case.get("prompt") else "",
        })
    return items


def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = f"{r.get('benchmark')}_{r.get('task_name')}"
                    if r.get("physical_adherence") is not None:
                        done.add(key)
                except:
                    pass
    return done


def run_benchmark(benchmark, cache_dir, out_dir):
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark}")
    print(f"{'='*60}")

    items = discover_items(benchmark, cache_dir)
    print(f"Discovered {len(items)} items")

    out_path = os.path.join(out_dir, f"judge_gpt55_{benchmark}.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    done = load_done(out_path)
    todo = [it for it in items if f"{it['benchmark']}_{it['task_name']}" not in done]
    print(f"Done: {len(done)}, Todo: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    client = get_client()
    n_ok = 0
    n_err = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for i, item in enumerate(todo):
            task = item["task_name"]
            # Download video
            video_local = os.path.join(cache_dir, benchmark, MODEL_DIR, f"{task}.mp4")
            try:
                download_file(item["video_url"], video_local)
            except Exception as e:
                rec = {**item, "error": f"download_video: {e}", "physical_adherence": None,
                       "judge_model": JUDGE_MODEL, "judge_revision": JUDGE_REVISION,
                       "judge_prompt_version": JUDGE_PROMPT_VERSION}
                del rec["video_url"], rec["prompt_url"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_err += 1
                print(f"[{i+1}/{len(todo)}] {task} DOWNLOAD_ERR")
                continue

            # Download/read prompt
            prompt_local = os.path.join(cache_dir, benchmark, MODEL_DIR, f"{task}_prompt.txt")
            try:
                download_file(item["prompt_url"], prompt_local)
                with open(prompt_local) as pf:
                    generation_prompt = pf.read().strip()
            except:
                generation_prompt = item.get("gt_prompt", "")

            if not generation_prompt:
                generation_prompt = item.get("gt_prompt", "No prompt available")

            # Sample frames
            frames = sample_frames_4fps(video_local)
            if not frames:
                rec = {**item, "error": "no_frames", "physical_adherence": None,
                       "judge_model": JUDGE_MODEL, "judge_revision": JUDGE_REVISION,
                       "judge_prompt_version": JUDGE_PROMPT_VERSION}
                del rec["video_url"], rec["prompt_url"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_err += 1
                print(f"[{i+1}/{len(todo)}] {task} NO_FRAMES")
                continue

            # Judge
            t0 = time.time()
            result = judge_video(client, frames, generation_prompt)
            dt = time.time() - t0

            rec = {
                "benchmark": benchmark,
                "case_id": item["case_id"],
                "task_name": task,
                "episode_name": item["episode_name"],
                "video_path": item["video_rel"],
                "prompt_path": item["prompt_rel"],
                "generation_prompt": generation_prompt,
                "judge_model": JUDGE_MODEL,
                "judge_revision": JUDGE_REVISION,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "n_frames": len(frames),
                "physical_adherence": result.get("physical_adherence"),
                "reasoning": result.get("reasoning", ""),
                "error": result.get("error", "") or result.get("parse_error", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            if rec["physical_adherence"] is not None:
                n_ok += 1
                tag = f"PA={rec['physical_adherence']}"
            else:
                n_err += 1
                tag = f"ERR: {rec.get('error', '')[:40]}"

            print(f"[{i+1}/{len(todo)}] {dt:.1f}s {tag} | {task}")

    print(f"\nDone {benchmark}: ok={n_ok} err={n_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=BENCHMARKS, help="Single benchmark to judge")
    ap.add_argument("--all", action="store_true", help="Judge all benchmarks")
    ap.add_argument("--cache_dir", default="/tmp/physical_vicl_cache")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    if not args.benchmark and not args.all:
        ap.error("Specify --benchmark or --all")

    setup_proxy()

    benchmarks = BENCHMARKS if args.all else [args.benchmark]
    for bm in benchmarks:
        run_benchmark(bm, args.cache_dir, args.out_dir)

    print("\n\nAll done!")


if __name__ == "__main__":
    main()
