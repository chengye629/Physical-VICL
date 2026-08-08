#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_v7_clean_common import (
    DIMENSION_WEIGHTS,
    FINAL_RANK_WEIGHTS,
    HYBRID_WEIGHTS,
    build_all_idfs,
    build_similarity_matrices,
    compact_card,
    load_embeddings,
    mmr_select,
    passes_thresholds,
    query_mode,
    score_pair,
    top_language_candidates,
    write_json,
    write_jsonl,
)


def optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "off"}:
        return None
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be in [0,1] or 'none'")
    return x


def summarize_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"pair_count": 0}
    known = [x for x in items if x["query_type_known"] and x["demo_type_known"]]
    return {
        "pair_count": len(items),
        "same_family_rate": float(np.mean([x["same_family"] for x in items])),
        "exact_type_rate_all": float(np.mean([x["exact_type"] for x in items])),
        "exact_type_rate_when_both_known": float(np.mean([x["exact_type"] for x in known])) if known else None,
        "mean_language_score": float(np.mean([x["language"] for x in items])),
        "mean_process_score": float(np.mean([x["process"] for x in items])),
        "mean_physical_score": float(np.mean([x["physical"] for x in items])),
        "mean_impact_axis_overlap": float(np.mean([x["impact_axis_overlap"] for x in items])),
        "mean_mechanism_family_overlap": float(np.mean([x["mechanism_family_overlap"] for x in items])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--embedding-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--language-top-k", type=int, default=500)
    ap.add_argument("--demo-count", type=int, default=10)
    ap.add_argument("--mmr-lambda", type=float, default=0.85)
    ap.add_argument("--language-threshold", type=optional_float, default=None)
    ap.add_argument("--process-threshold", type=optional_float, default=None)
    ap.add_argument("--physical-threshold", type=optional_float, default=None)
    ap.add_argument("--example-count", type=int, default=60)
    args = ap.parse_args()

    rows, emb = load_embeddings(args.index, args.embedding_root)
    idfs = build_all_idfs(rows)
    lang_mat, redundancy_mat = build_similarity_matrices(emb)

    results: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    selected_metrics: list[dict[str, Any]] = []
    selected_metrics_by_mode: dict[str, list[dict[str, Any]]] = {"canonical": [], "open_vocab": []}
    rejection_reason_counts: Counter[str] = Counter()
    candidate_count_before: list[int] = []
    candidate_count_after: list[int] = []

    for qi, query in enumerate(rows):
        candidate_indices = top_language_candidates(lang_mat, qi, args.language_top_k)
        lang_rank = {idx: r for r, idx in enumerate(candidate_indices, start=1)}
        scored = [score_pair(rows, emb, idfs, lang_mat, qi, di, lang_rank[di]) for di in candidate_indices]
        candidate_count_before.append(len(scored))

        accepted: list[dict[str, Any]] = []
        for item in scored:
            ok, reasons = passes_thresholds(
                item,
                args.language_threshold,
                args.process_threshold,
                args.physical_threshold,
            )
            if ok:
                accepted.append(item)
            else:
                for reason in reasons:
                    rejection_reason_counts[reason] += 1
        candidate_count_after.append(len(accepted))

        selected = mmr_select(accepted, redundancy_mat, args.demo_count, args.mmr_lambda)
        qmode = query_mode(query)

        result = {
            "query_id": query["sample_id"],
            "query_mode": qmode,
            "query_family": query["process_family"],
            "query_type": query.get("process_type"),
            "query_summary": query.get("event_summary", ""),
            "candidate_count_before_filter": len(scored),
            "candidate_count_after_filter": len(accepted),
            "selected_demo_count": len(selected),
            "demos": [],
        }

        for item in selected:
            demo = rows[item["index"]]
            saved = {k: v for k, v in item.items() if k != "index"}
            saved.update({
                "demo_family": demo["process_family"],
                "demo_type": demo.get("process_type"),
                "demo_summary": demo.get("event_summary", ""),
                "demo_object_names": demo.get("object_names", []),
                "source": demo.get("source", {}),
            })
            result["demos"].append(saved)
            pair_rows.append({"query_id": query["sample_id"], **saved})
            metric = {
                "same_family": saved["same_family"],
                "exact_type": saved["exact_type"],
                "query_type_known": bool(query.get("process_type")),
                "demo_type_known": bool(demo.get("process_type")),
                "language": saved["language"],
                "process": saved["dims"]["process"],
                "physical": saved["physical"],
                "impact_axis_overlap": saved["impact_axis_overlap"],
                "mechanism_family_overlap": saved["mechanism_family_overlap"],
            }
            selected_metrics.append(metric)
            selected_metrics_by_mode[qmode].append(metric)

        results.append(result)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "retrievals_clean_v21.jsonl", results)
    write_jsonl(args.output_root / "pairs_clean_v21.jsonl", pair_rows)

    demo_counts = [r["selected_demo_count"] for r in results]
    summary = {
        "method": "semantic_recall -> reliability_aware_physics_matching -> confidence_filtering -> MMR",
        "query_count": len(results),
        "language_top_k": args.language_top_k,
        "demo_count_max": args.demo_count,
        "mmr_lambda": args.mmr_lambda,
        "thresholds": {
            "language": args.language_threshold,
            "process": args.process_threshold,
            "physical": args.physical_threshold,
        },
        "weights": {
            "dimension": DIMENSION_WEIGHTS,
            "structured_semantic": HYBRID_WEIGHTS,
            "final_rank": FINAL_RANK_WEIGHTS,
        },
        "selection": {
            "queries_with_any_demo": sum(c > 0 for c in demo_counts),
            "queries_with_full_demo_count": sum(c == args.demo_count for c in demo_counts),
            "queries_with_zero_demo": sum(c == 0 for c in demo_counts),
            "mean_demo_count": float(np.mean(demo_counts)),
            "median_demo_count": float(np.median(demo_counts)),
            "min_demo_count": int(min(demo_counts)),
            "max_demo_count": int(max(demo_counts)),
            "mean_candidates_before_filter": float(np.mean(candidate_count_before)),
            "mean_candidates_after_filter": float(np.mean(candidate_count_after)),
            "rejection_reason_counts": dict(rejection_reason_counts),
        },
        "selected_pair_metrics": summarize_pairs(selected_metrics),
        "selected_pair_metrics_by_query_mode": {
            mode: summarize_pairs(items) for mode, items in selected_metrics_by_mode.items()
        },
        "notes": [
            "No hard process-family gate is used.",
            "No exact-type bonus, family bonus, same-family augmentation, or exact-type augmentation is used.",
            "Canonical/open-vocabulary status only controls the reliability interpolation between structured and semantic Process similarity.",
            "A query may return fewer than K demos, including zero, when candidates fail calibrated confidence thresholds.",
        ],
    }
    write_json(args.output_root / "retrieval_summary_clean_v21.json", summary)

    # Deterministic qualitative examples: queries with the fewest accepted candidates first,
    # then stable query id. This makes threshold effects easy to inspect.
    ordered = sorted(results, key=lambda r: (r["candidate_count_after_filter"], r["query_id"]))[: args.example_count]
    lines: list[str] = []
    for i, r in enumerate(ordered, start=1):
        lines.extend([
            "=" * 120,
            f"EXAMPLE {i:03d}",
            f"QUERY {r['query_id']}",
            f"Mode: {r['query_mode']} | Family/type: {r['query_family']} / {r.get('query_type')}",
            f"Summary: {r['query_summary']}",
            f"Candidates: {r['candidate_count_before_filter']} -> {r['candidate_count_after_filter']} | selected={r['selected_demo_count']}",
            "",
        ])
        for d in r["demos"][:5]:
            lines.append(
                f"  #{d['rank']} {d['sample_id']} {d['demo_family']}/{d.get('demo_type')} "
                f"rank={d['rank_score']:.4f} lang={d['language']:.4f} phys={d['physical']:.4f} "
                f"O/P/I/M={d['dims']['object']:.3f}/{d['dims']['process']:.3f}/{d['dims']['impact']:.3f}/{d['dims']['mechanism']:.3f}"
            )
            lines.append(f"     {d['demo_summary']}")
        lines.append("")
    (args.output_root / "retrieval_examples_clean_v21.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
