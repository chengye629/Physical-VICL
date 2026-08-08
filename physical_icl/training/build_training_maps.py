#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                if isinstance(x, dict):
                    yield x


def source_video_path(row: dict[str, Any]) -> str:
    raw = row.get("video_path")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    src = row.get("source")
    if isinstance(src, dict):
        raw = src.get("video_path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raise KeyError(f"video_path missing for sample_id={row.get('sample_id')}")


def portable_relpath(raw: str, project_root: Path) -> str:
    p = Path(raw)
    raw_posix = raw.replace('\\', '/')
    cross_platform_abs = raw_posix.startswith('/') or p.is_absolute() or PureWindowsPath(raw).is_absolute()
    if not cross_platform_abs:
        return PurePosixPath(raw_posix).as_posix()
    try:
        return p.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        # Historical manifests may point into another checkout. Preserve only a
        # recognizable project-relative data suffix when present.
        parts = [x for x in PurePosixPath(raw_posix).parts if x not in {'/', '\\'}]
        if "data" in parts:
            i = parts.index("data")
            return PurePosixPath(*parts[i:]).as_posix()
        raise ValueError(
            f"Cannot make absolute path portable: {raw}. "
            "Run on the original project checkout or provide a normalized manifest."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--retrievals", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    index = {str(r["sample_id"]): r for r in read_jsonl(args.index)}
    retrievals = list(read_jsonl(args.retrievals))
    raw_pairs = list(read_jsonl(args.pairs))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flat_path = args.output_dir / "training_pairs_1demo.jsonl"
    query_path = args.output_dir / "query_demo_map.jsonl"
    lookup_path = args.output_dir / "video_lookup.jsonl"

    rebuilt_edges: set[tuple[str, str]] = set()
    with flat_path.open("w", encoding="utf-8") as fout:
        for p in raw_pairs:
            qid = str(p["query_id"])
            did = str(p["sample_id"])
            q = index[qid]
            d = index[did]
            edge = (qid, did)
            if edge in rebuilt_edges:
                raise RuntimeError(f"duplicate directed pair: {edge}")
            rebuilt_edges.add(edge)
            out = {
                "pair_id": f"{qid}__{did}",
                "query_id": qid,
                "demo_id": did,
                "query_video_relpath": portable_relpath(source_video_path(q), args.project_root),
                "demo_video_relpath": portable_relpath(source_video_path(d), args.project_root),
                "query_physics": {
                    "mode": q.get("process_resolution_level"),
                    "family": q.get("process_family"),
                    "type": q.get("process_type"),
                },
                "demo_physics": {
                    "mode": d.get("process_resolution_level"),
                    "family": d.get("process_family"),
                    "type": d.get("process_type"),
                },
                "selected_rank": p.get("rank"),
                "semantic_recall_rank": p.get("language_rank"),
                "scores": {
                    "rank_score": p.get("rank_score"),
                    "language": p.get("language"),
                    "physical": p.get("physical"),
                    "dimensions": p.get("dims"),
                    "structured_dimensions": p.get("structured_dims"),
                    "semantic_dimensions": p.get("feature_dims"),
                    "impact_axis_overlap": p.get("impact_axis_overlap"),
                    "mechanism_family_overlap": p.get("mechanism_family_overlap"),
                    "mmr_score": p.get("mmr_score"),
                },
                "copy_risk_status": "pending",
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    expanded_edges: set[tuple[str, str]] = set()
    with query_path.open("w", encoding="utf-8") as fout:
        for r in retrievals:
            qid = str(r["query_id"])
            q = index[qid]
            demos = []
            for drec in r.get("demos", []):
                did = str(drec["sample_id"])
                d = index[did]
                expanded_edges.add((qid, did))
                demos.append({
                    "demo_id": did,
                    "demo_video_relpath": portable_relpath(source_video_path(d), args.project_root),
                    "demo_mode": d.get("process_resolution_level"),
                    "demo_family": d.get("process_family"),
                    "demo_type": d.get("process_type"),
                    "rank": drec.get("rank"),
                    "language_rank": drec.get("language_rank"),
                    "rank_score": drec.get("rank_score"),
                    "language_score": drec.get("language"),
                    "physical_score": drec.get("physical"),
                    "dimension_scores": drec.get("dims"),
                    "structured_dimension_scores": drec.get("structured_dims"),
                    "semantic_dimension_scores": drec.get("feature_dims"),
                    "impact_axis_overlap": drec.get("impact_axis_overlap"),
                    "mechanism_family_overlap": drec.get("mechanism_family_overlap"),
                    "mmr_score": drec.get("mmr_score"),
                    "copy_risk_status": "pending",
                })
            out = {
                "query_id": qid,
                "query_video_relpath": portable_relpath(source_video_path(q), args.project_root),
                "query_mode": q.get("process_resolution_level"),
                "query_family": q.get("process_family"),
                "query_type": q.get("process_type"),
                "query_summary": r.get("query_summary"),
                "candidate_count_before_filter": r.get("candidate_count_before_filter"),
                "candidate_count_after_filter": r.get("candidate_count_after_filter"),
                "selected_demo_count": len(demos),
                "demos": demos,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    raw_edges = {(str(p["query_id"]), str(p["sample_id"])) for p in raw_pairs}
    if raw_edges != rebuilt_edges or raw_edges != expanded_edges:
        raise RuntimeError(
            f"edge mismatch raw={len(raw_edges)} flat={len(rebuilt_edges)} query={len(expanded_edges)}"
        )

    with lookup_path.open("w", encoding="utf-8") as fout:
        for sid, row in sorted(index.items()):
            fout.write(json.dumps({
                "sample_id": sid,
                "video_relpath": portable_relpath(source_video_path(row), args.project_root),
                "process_resolution_level": row.get("process_resolution_level"),
                "process_family": row.get("process_family"),
                "process_type": row.get("process_type"),
            }, ensure_ascii=False) + "\n")

    summary = {
        "eligible_videos": len(index),
        "query_rows": len(retrievals),
        "directed_pairs": len(raw_edges),
        "mode_source": "retrieval_index.process_resolution_level for each endpoint",
        "paths": "project-relative only",
        "copy_risk_status": "pending",
    }
    (args.output_dir / "mapping_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
