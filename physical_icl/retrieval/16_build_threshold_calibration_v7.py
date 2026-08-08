#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_v7_clean_common import (
    build_all_idfs,
    build_similarity_matrices,
    compact_card,
    load_embeddings,
    query_mode,
    score_pair,
    top_language_candidates,
    write_json,
    write_jsonl,
)


def bin3(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def compact_line(values: list[str]) -> str:
    return ", ".join(str(x) for x in values) if values else "<none>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--embedding-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--language-top-k", type=int, default=500)
    ap.add_argument("--review-count", type=int, default=180)
    ap.add_argument("--seed", type=int, default=20260807)
    args = ap.parse_args()

    rows, emb = load_embeddings(args.index, args.embedding_root)
    idfs = build_all_idfs(rows)
    lang_mat, _ = build_similarity_matrices(emb)

    # Store pair references compactly: one row per candidate in the semantic top-k.
    qi_list: list[int] = []
    di_list: list[int] = []
    lang_list: list[float] = []
    proc_list: list[float] = []
    phys_list: list[float] = []
    rank_list: list[float] = []
    mode_list: list[int] = []  # 1 canonical, 0 open_vocab

    for qi, query in enumerate(rows):
        candidates = top_language_candidates(lang_mat, qi, args.language_top_k)
        for rank, di in enumerate(candidates, start=1):
            item = score_pair(rows, emb, idfs, lang_mat, qi, di, rank)
            qi_list.append(qi)
            di_list.append(di)
            lang_list.append(item["language"])
            proc_list.append(item["dims"]["process"])
            phys_list.append(item["physical"])
            rank_list.append(item["rank_score"])
            mode_list.append(1 if query_mode(query) == "canonical" else 0)

    qi_arr = np.asarray(qi_list, dtype=np.int32)
    di_arr = np.asarray(di_list, dtype=np.int32)
    lang_arr = np.asarray(lang_list, dtype=np.float32)
    proc_arr = np.asarray(proc_list, dtype=np.float32)
    phys_arr = np.asarray(phys_list, dtype=np.float32)
    rank_arr = np.asarray(rank_list, dtype=np.float32)
    mode_arr = np.asarray(mode_list, dtype=np.int8)

    rng = random.Random(args.seed)
    chosen: list[int] = []
    used_queries: set[int] = set()
    calibration_meta: dict[str, Any] = {"modes": {}}

    # Balance canonical/open-vocab review examples so threshold calibration does not
    # ignore the smaller open-vocabulary subset. Thresholds themselves remain global.
    mode_targets = {1: args.review_count // 2, 0: args.review_count - args.review_count // 2}

    for mode_value, target in mode_targets.items():
        mode_name = "canonical" if mode_value == 1 else "open_vocab"
        idx_mode = np.flatnonzero(mode_arr == mode_value)
        if len(idx_mode) == 0:
            calibration_meta["modes"][mode_name] = {"pair_count": 0}
            continue

        lq = np.quantile(lang_arr[idx_mode], [1 / 3, 2 / 3]).tolist()
        pq = np.quantile(proc_arr[idx_mode], [1 / 3, 2 / 3]).tolist()
        hq = np.quantile(phys_arr[idx_mode], [1 / 3, 2 / 3]).tolist()
        calibration_meta["modes"][mode_name] = {
            "pair_count": int(len(idx_mode)),
            "language_tertiles": lq,
            "process_tertiles": pq,
            "physical_tertiles": hq,
        }

        cells: dict[str, list[int]] = defaultdict(list)
        for idx in idx_mode.tolist():
            cell = "/".join([
                f"L:{bin3(float(lang_arr[idx]), lq[0], lq[1])}",
                f"P:{bin3(float(proc_arr[idx]), pq[0], pq[1])}",
                f"H:{bin3(float(phys_arr[idx]), hq[0], hq[1])}",
            ])
            cells[cell].append(idx)

        nonempty = sorted(cells)
        per_cell = max(1, target // max(1, len(nonempty)))
        mode_chosen: list[int] = []

        # First pass: broad score-region coverage with query diversity.
        for cell in nonempty:
            candidates = cells[cell].copy()
            rng.shuffle(candidates)
            taken = 0
            for idx in candidates:
                qi = int(qi_arr[idx])
                if qi in used_queries:
                    continue
                mode_chosen.append(idx)
                used_queries.add(qi)
                taken += 1
                if taken >= per_cell or len(mode_chosen) >= target:
                    break
            if len(mode_chosen) >= target:
                break

        # Second pass: fill target while still preferring unseen queries.
        remaining = idx_mode.tolist()
        rng.shuffle(remaining)
        for idx in remaining:
            if len(mode_chosen) >= target:
                break
            if idx in mode_chosen:
                continue
            qi = int(qi_arr[idx])
            if qi in used_queries:
                continue
            mode_chosen.append(idx)
            used_queries.add(qi)

        # Final fill if unique-query constraint is exhausted.
        if len(mode_chosen) < target:
            remaining = [idx for idx in idx_mode.tolist() if idx not in mode_chosen]
            rng.shuffle(remaining)
            mode_chosen.extend(remaining[: target - len(mode_chosen)])

        chosen.extend(mode_chosen[:target])

    rng.shuffle(chosen)
    chosen = chosen[: args.review_count]

    review_rows: list[dict[str, Any]] = []
    txt_lines: list[str] = [
        "PHYSICS-CARD DEMO RETRIEVAL THRESHOLD CALIBRATION",
        "",
        "Please judge each Query-Demo pair as GOOD / ACCEPTABLE / BAD.",
        "GOOD: strongly useful physical demonstration; the dominant interaction/evolution is closely aligned.",
        "ACCEPTABLE: physically relevant and potentially useful, but object/context or secondary details differ noticeably.",
        "BAD: dominant process/causal structure is different or the pair would plausibly mislead training.",
        "",
        "Scores are shown only for threshold calibration; do not judge by the score itself.",
        "",
    ]

    csv_rows: list[dict[str, Any]] = []

    for number, idx in enumerate(chosen, start=1):
        qi, di = int(qi_arr[idx]), int(di_arr[idx])
        q, d = rows[qi], rows[di]
        qmode = query_mode(q)
        mode_meta = calibration_meta["modes"][qmode]
        lq = mode_meta["language_tertiles"]
        pq = mode_meta["process_tertiles"]
        hq = mode_meta["physical_tertiles"]
        region = "/".join([
            f"L:{bin3(float(lang_arr[idx]), lq[0], lq[1])}",
            f"P:{bin3(float(proc_arr[idx]), pq[0], pq[1])}",
            f"H:{bin3(float(phys_arr[idx]), hq[0], hq[1])}",
        ])

        item = score_pair(rows, emb, idfs, lang_mat, qi, di, None)
        qcard, dcard = compact_card(q), compact_card(d)
        record = {
            "review_id": f"pair_{number:04d}",
            "query_mode": qmode,
            "score_region": region,
            "scores": {
                "language": float(lang_arr[idx]),
                "process": float(proc_arr[idx]),
                "physical": float(phys_arr[idx]),
                "rank": float(rank_arr[idx]),
                "object": item["dims"]["object"],
                "impact": item["dims"]["impact"],
                "mechanism": item["dims"]["mechanism"],
            },
            "query": qcard,
            "demo": dcard,
            "assistant_label": "",
            "assistant_reason": "",
        }
        review_rows.append(record)

        txt_lines.extend([
            "=" * 120,
            f"PAIR {number:04d} | mode={qmode} | region={region}",
            f"SCORES: language={lang_arr[idx]:.4f} process={proc_arr[idx]:.4f} physical={phys_arr[idx]:.4f} rank={rank_arr[idx]:.4f}",
            "",
            "QUERY",
            f"  id: {qcard['sample_id']}",
            f"  family/type: {qcard['family']} / {qcard['type']}",
            f"  summary: {qcard['summary']}",
            f"  objects: {compact_line(qcard['objects'])}",
            f"  actions: {compact_line(qcard['actions'])}",
            f"  impact axes: {compact_line(qcard['impact_axes'])}",
            f"  mechanisms: {compact_line(qcard['mechanism_families'])}",
            f"  process raw: {qcard['process_raw']}",
            f"  impact raw: {qcard['impact_raw']}",
            "",
            "DEMO",
            f"  id: {dcard['sample_id']}",
            f"  family/type: {dcard['family']} / {dcard['type']}",
            f"  summary: {dcard['summary']}",
            f"  objects: {compact_line(dcard['objects'])}",
            f"  actions: {compact_line(dcard['actions'])}",
            f"  impact axes: {compact_line(dcard['impact_axes'])}",
            f"  mechanisms: {compact_line(dcard['mechanism_families'])}",
            f"  process raw: {dcard['process_raw']}",
            f"  impact raw: {dcard['impact_raw']}",
            "",
            "ASSISTANT LABEL: <leave blank; upload this file back to ChatGPT>",
            "ASSISTANT REASON: <leave blank>",
            "",
        ])

        csv_rows.append({
            "review_id": record["review_id"],
            "query_mode": qmode,
            "score_region": region,
            "language_score": float(lang_arr[idx]),
            "process_score": float(proc_arr[idx]),
            "physical_score": float(phys_arr[idx]),
            "rank_score": float(rank_arr[idx]),
            "query_id": qcard["sample_id"],
            "query_family": qcard["family"],
            "query_type": qcard["type"],
            "query_summary": qcard["summary"],
            "demo_id": dcard["sample_id"],
            "demo_family": dcard["family"],
            "demo_type": dcard["type"],
            "demo_summary": dcard["summary"],
            "assistant_label": "",
            "assistant_reason": "",
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "threshold_calibration_review.jsonl", review_rows)
    (args.output_root / "threshold_calibration_review.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    with (args.output_root / "threshold_calibration_review.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    calibration_meta.update({
        "review_count_requested": args.review_count,
        "review_count_written": len(review_rows),
        "language_top_k": args.language_top_k,
        "seed": args.seed,
        "sampling": "balanced canonical/open-vocab; stratified by within-mode tertiles of language/process/physical scores; prefers unique queries",
        "threshold_policy": "Thresholds are global. Query resolution changes only structured-vs-semantic reliability in Process scoring.",
    })
    write_json(args.output_root / "threshold_calibration_summary.json", calibration_meta)
    print(json.dumps(calibration_meta, ensure_ascii=False, indent=2))
    print(f"[DONE] Review TXT : {args.output_root / 'threshold_calibration_review.txt'}")
    print(f"[DONE] Review JSONL: {args.output_root / 'threshold_calibration_review.jsonl'}")
    print(f"[DONE] Review CSV : {args.output_root / 'threshold_calibration_review.csv'}")


if __name__ == "__main__":
    main()
