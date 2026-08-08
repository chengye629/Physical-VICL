#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DIMENSION_WEIGHTS = {
    "object": 0.20,
    "process": 0.45,
    "impact": 0.30,
    "mechanism": 0.05,
}

# Reliability-aware interpolation between discrete ontology matching and
# continuous semantic matching. Process reliability depends on whether the
# query has a canonical subtype or only an open-vocabulary description.
HYBRID_WEIGHTS = {
    "object": (0.60, 0.40),
    "process_canonical": (0.75, 0.25),
    "process_open_vocab": (0.40, 0.60),
    "impact": (0.80, 0.20),
    "mechanism": (0.80, 0.20),
}

FINAL_RANK_WEIGHTS = {"physical": 0.90, "language": 0.10}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_words(s: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]+", str(s or "").lower())
        if len(w) > 1
        and w not in {"the", "a", "an", "and", "or", "of", "to", "is", "are", "in", "on", "with", "for"}
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_idf(rows: list[dict[str, Any]], token_key: str) -> dict[str, float]:
    n = len(rows)
    df: Counter[str] = Counter()
    for row in rows:
        df.update(set(row.get("tokens", {}).get(token_key, [])))
    return {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}


def build_all_idfs(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    token_keys = [
        "object_role_kind",
        "object_phase",
        "object_property",
        "object_initial",
        "action",
        "secondary_process",
        "temporal",
        "impact_axis",
        "impact_type",
        "mechanism_family",
        "mechanism_type",
    ]
    return {key: build_idf(rows, key) for key in token_keys}


def weighted_jaccard(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = a | b
    inter = a & b
    denom = sum(idf.get(token, 1.0) for token in union)
    return sum(idf.get(token, 1.0) for token in inter) / denom if denom else 0.0


def active_average(parts: list[tuple[float, float, bool]]) -> float:
    active = [(value, weight) for value, weight, use in parts if use]
    if not active:
        return 0.0
    denom = sum(weight for _, weight in active)
    return sum(value * weight for value, weight in active) / denom


def cosine01(value: float) -> float:
    return max(0.0, min(1.0, (float(value) + 1.0) * 0.5))


def query_mode(row: dict[str, Any]) -> str:
    if row.get("process_type") and row.get("process_resolution_level") == "canonical_type":
        return "canonical"
    return "open_vocab"


def structured_scores(
    query: dict[str, Any],
    demo: dict[str, Any],
    idfs: dict[str, dict[str, float]],
) -> dict[str, float]:
    qt, dt = query["tokens"], demo["tokens"]

    role_kind = weighted_jaccard(set(qt["object_role_kind"]), set(dt["object_role_kind"]), idfs["object_role_kind"])
    phase = weighted_jaccard(set(qt["object_phase"]), set(dt["object_phase"]), idfs["object_phase"])
    prop = weighted_jaccard(set(qt["object_property"]), set(dt["object_property"]), idfs["object_property"])
    initial = weighted_jaccard(set(qt["object_initial"]), set(dt["object_initial"]), idfs["object_initial"])
    obj_lex = jaccard(norm_words(query["texts"]["object"]), norm_words(demo["texts"]["object"]))
    obj = active_average([
        (role_kind, 0.45, bool(qt["object_role_kind"] or dt["object_role_kind"])),
        (phase, 0.15, bool(qt["object_phase"] or dt["object_phase"])),
        (prop, 0.15, bool(qt["object_property"] or dt["object_property"])),
        (initial, 0.10, bool(qt["object_initial"] or dt["object_initial"])),
        (obj_lex, 0.15, True),
    ])

    family = 1.0 if query["process_family"] == demo["process_family"] else 0.0
    qtype, dtype = query.get("process_type"), demo.get("process_type")
    type_score = 1.0 if qtype and dtype and qtype == dtype else 0.0
    action = weighted_jaccard(set(qt["action"]), set(dt["action"]), idfs["action"])
    secondary = weighted_jaccard(set(qt["secondary_process"]), set(dt["secondary_process"]), idfs["secondary_process"])
    temporal = weighted_jaccard(set(qt["temporal"]), set(dt["temporal"]), idfs["temporal"])
    proc_lex = jaccard(norm_words(query["texts"]["process_raw"]), norm_words(demo["texts"]["process_raw"]))
    proc = active_average([
        (family, 0.45, True),
        (type_score, 0.20, bool(qtype and dtype)),
        (action, 0.10, bool(qt["action"] or dt["action"])),
        (secondary, 0.05, bool(qt["secondary_process"] or dt["secondary_process"])),
        (temporal, 0.05, True),
        (proc_lex, 0.15, True),
    ])

    axes = weighted_jaccard(set(qt["impact_axis"]), set(dt["impact_axis"]), idfs["impact_axis"])
    types = weighted_jaccard(set(qt["impact_type"]), set(dt["impact_type"]), idfs["impact_type"])
    impact_lex = jaccard(norm_words(query["texts"]["impact_raw"]), norm_words(demo["texts"]["impact_raw"]))
    impact = active_average([
        (axes, 0.55, bool(qt["impact_axis"] or dt["impact_axis"])),
        (types, 0.20, bool(qt["impact_type"] and dt["impact_type"])),
        (impact_lex, 0.25, True),
    ])

    mech_family = weighted_jaccard(
        set(qt["mechanism_family"]), set(dt["mechanism_family"]), idfs["mechanism_family"]
    )
    mech_type = weighted_jaccard(
        set(qt["mechanism_type"]), set(dt["mechanism_type"]), idfs["mechanism_type"]
    )
    mech_lex = jaccard(norm_words(query["texts"]["mechanism_raw"]), norm_words(demo["texts"]["mechanism_raw"]))
    mechanism = active_average([
        (mech_family, 0.45, bool(qt["mechanism_family"] or dt["mechanism_family"])),
        (mech_type, 0.25, bool(qt["mechanism_type"] and dt["mechanism_type"])),
        (mech_lex, 0.30, True),
    ])

    return {
        "object": obj,
        "process": proc,
        "impact": impact,
        "mechanism": mechanism,
        "process_family_exact": family,
        "process_type_exact": type_score,
        "impact_axis": axes,
        "mechanism_family": mech_family,
    }


def feature_scores(
    emb: dict[str, np.ndarray],
    qi: int,
    di: int,
) -> dict[str, float]:
    return {
        dim: cosine01(float(emb[dim][qi] @ emb[dim][di]))
        for dim in ["object", "process", "impact", "mechanism"]
    }


def clean_hybrid_scores(
    query: dict[str, Any],
    structured: dict[str, float],
    feature: dict[str, float],
) -> tuple[dict[str, float], float, str]:
    mode = query_mode(query)

    os_, of_ = HYBRID_WEIGHTS["object"]
    is_, if_ = HYBRID_WEIGHTS["impact"]
    ms_, mf_ = HYBRID_WEIGHTS["mechanism"]
    ps_, pf_ = HYBRID_WEIGHTS["process_canonical" if mode == "canonical" else "process_open_vocab"]

    dims = {
        "object": os_ * structured["object"] + of_ * feature["object"],
        "process": ps_ * structured["process"] + pf_ * feature["process"],
        "impact": is_ * structured["impact"] + if_ * feature["impact"],
        "mechanism": ms_ * structured["mechanism"] + mf_ * feature["mechanism"],
    }
    physical = sum(DIMENSION_WEIGHTS[k] * dims[k] for k in DIMENSION_WEIGHTS)
    return dims, physical, mode


def load_embeddings(index_path: Path, embedding_root: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = read_jsonl(index_path)
    ids = json.loads((embedding_root / "sample_ids.json").read_text(encoding="utf-8"))
    expected_ids = [row["sample_id"] for row in rows]
    if ids != expected_ids:
        raise SystemExit("embedding sample order does not match retrieval index")

    emb = {
        key: np.asarray(np.load(embedding_root / f"{key}.npy"), dtype=np.float32)
        for key in ["language", "object", "process", "impact", "mechanism"]
    }
    n = len(rows)
    for key, array in emb.items():
        if array.shape[0] != n:
            raise SystemExit(f"{key} embedding count mismatch")
    return rows, emb


def build_similarity_matrices(emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lang_mat = emb["language"] @ emb["language"].T
    np.fill_diagonal(lang_mat, -np.inf)
    process_sim = emb["process"] @ emb["process"].T
    redundancy_mat = 0.5 * lang_mat + 0.5 * process_sim
    return lang_mat, redundancy_mat


def top_language_candidates(lang_mat: np.ndarray, qi: int, k: int) -> list[int]:
    n = lang_mat.shape[0]
    k = min(k, n - 1)
    idx = np.argpartition(-lang_mat[qi], k - 1)[:k]
    idx = idx[np.argsort(-lang_mat[qi, idx])]
    return [int(x) for x in idx]


def score_pair(
    rows: list[dict[str, Any]],
    emb: dict[str, np.ndarray],
    idfs: dict[str, dict[str, float]],
    lang_mat: np.ndarray,
    qi: int,
    di: int,
    language_rank: int | None = None,
) -> dict[str, Any]:
    query, demo = rows[qi], rows[di]
    structured = structured_scores(query, demo, idfs)
    feature = feature_scores(emb, qi, di)
    dims, physical, mode = clean_hybrid_scores(query, structured, feature)
    language = cosine01(float(lang_mat[qi, di]))
    rank_score = FINAL_RANK_WEIGHTS["physical"] * physical + FINAL_RANK_WEIGHTS["language"] * language
    return {
        "index": di,
        "sample_id": demo["sample_id"],
        "language": language,
        "language_rank": language_rank,
        "physical": physical,
        "rank_score": rank_score,
        "dims": dims,
        "mode": mode,
        "same_family": bool(structured["process_family_exact"]),
        "exact_type": bool(structured["process_type_exact"]),
        "impact_axis_overlap": structured["impact_axis"],
        "mechanism_family_overlap": structured["mechanism_family"],
        "structured_dims": {k: structured[k] for k in ["object", "process", "impact", "mechanism"]},
        "feature_dims": feature,
    }


def passes_thresholds(
    item: dict[str, Any],
    language_threshold: float | None,
    process_threshold: float | None,
    physical_threshold: float | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if language_threshold is not None and item["language"] < language_threshold:
        reasons.append("language")
    if process_threshold is not None and item["dims"]["process"] < process_threshold:
        reasons.append("process")
    if physical_threshold is not None and item["physical"] < physical_threshold:
        reasons.append("physical")
    return not reasons, reasons


def mmr_select(
    pool: list[dict[str, Any]],
    redundancy_mat: np.ndarray,
    demo_count: int,
    mmr_lambda: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = pool.copy()
    while remaining and len(selected) < demo_count:
        best = None
        best_value = -1e9
        for item in remaining:
            if not selected:
                value = float(item["rank_score"])
            else:
                redundancy = max(
                    cosine01(float(redundancy_mat[item["index"], chosen["index"]]))
                    for chosen in selected
                )
                value = mmr_lambda * float(item["rank_score"]) - (1.0 - mmr_lambda) * redundancy
            if value > best_value:
                best_value = value
                best = item
        assert best is not None
        chosen = dict(best)
        chosen["mmr_score"] = best_value
        chosen["rank"] = len(selected) + 1
        selected.append(chosen)
        remaining = [item for item in remaining if item["index"] != best["index"]]
    return selected


def compact_card(row: dict[str, Any]) -> dict[str, Any]:
    tokens = row.get("tokens", {})
    texts = row.get("texts", {})
    return {
        "sample_id": row.get("sample_id"),
        "summary": row.get("event_summary", ""),
        "family": row.get("process_family"),
        "type": row.get("process_type"),
        "resolution": row.get("process_resolution_level"),
        "objects": row.get("object_names", []),
        "actions": tokens.get("action", []),
        "impact_axes": tokens.get("impact_axis", []),
        "impact_types": tokens.get("impact_type", []),
        "mechanism_families": tokens.get("mechanism_family", []),
        "mechanism_types": tokens.get("mechanism_type", []),
        "process_raw": texts.get("process_raw", ""),
        "impact_raw": texts.get("impact_raw", ""),
        "mechanism_raw": texts.get("mechanism_raw", ""),
        "source": row.get("source", {}),
    }
