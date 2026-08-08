#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def card(payload):
    return payload.get("physics_card", {}) if isinstance(payload, dict) and payload.get("status") == "success" else {}


def tok(x):
    return str(x or "").strip().lower().replace("-", "_")


def p_family(c):
    return tok(c.get("process", {}).get("primary_process", {}).get("family"))


def p_type(c):
    return tok(c.get("process", {}).get("primary_process", {}).get("type"))


def impact_axes(c):
    out = set()
    for imp in c.get("impacts", []) or []:
        if not isinstance(imp, dict):
            continue
        for tr in imp.get("state_transitions", []) or []:
            if isinstance(tr, dict) and tok(tr.get("axis")):
                out.add(tok(tr.get("axis")))
    return out


def mechanisms(c):
    return {tok(x.get("family")) for x in (c.get("mechanisms", []) or []) if isinstance(x, dict) and tok(x.get("family"))}


def jac(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-cards", type=Path, required=True)
    ap.add_argument("--new-cards", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    ids = sorted({p.stem for p in args.reference_cards.glob("*.json")} & {p.stem for p in args.new_cards.glob("*.json")})
    rows = []
    for sid in ids:
        a = card(load(args.reference_cards / f"{sid}.json") or {})
        b = card(load(args.new_cards / f"{sid}.json") or {})
        if not a or not b:
            continue
        rows.append({
            "sample_id": sid,
            "family_equal": p_family(a) == p_family(b),
            "type_equal": p_type(a) == p_type(b),
            "reference_family": p_family(a), "new_family": p_family(b),
            "reference_type": p_type(a), "new_type": p_type(b),
            "impact_jaccard": jac(impact_axes(a), impact_axes(b)),
            "mechanism_jaccard": jac(mechanisms(a), mechanisms(b)),
        })

    known = [x for x in rows if x["reference_type"] and x["new_type"]]
    summary = {
        "paired_successful_cards": len(rows),
        "process_family_agreement": float(np.mean([x["family_equal"] for x in rows])) if rows else None,
        "process_type_agreement_when_both_known": float(np.mean([x["type_equal"] for x in known])) if known else None,
        "mean_impact_axis_jaccard": float(np.mean([x["impact_jaccard"] for x in rows])) if rows else None,
        "mean_mechanism_family_jaccard": float(np.mean([x["mechanism_jaccard"] for x in rows])) if rows else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
