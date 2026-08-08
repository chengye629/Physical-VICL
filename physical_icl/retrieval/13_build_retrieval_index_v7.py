#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def safe_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def tok(v: Any) -> str:
    s = str(v or "").strip().lower().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def text(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "").strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def role_order(roles: list[str]) -> int:
    priority = {
        "target": 0,
        "actor": 1,
        "tool": 2,
        "source": 3,
        "product": 4,
        "support": 5,
        "container": 6,
        "medium": 7,
    }
    return min((priority.get(r, 99) for r in roles), default=99)


def build_record(sample_id: str, payload: dict[str, Any], manifest_row: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("status") != "success":
        return None
    card = safe_dict(payload.get("physics_card"))
    if not card:
        return None
    process = safe_dict(card.get("process"))
    primary = safe_dict(process.get("primary_process"))
    family = tok(primary.get("family"))
    ptype = tok(primary.get("type")) or None
    scope = tok(process.get("scope"))

    # Structured physical demos exclude no-dominant-process and camera-only samples.
    eligible = bool(family and family != "special" and ptype != "none" and scope != "camera_only")

    objects = [x for x in safe_list(card.get("objects")) if isinstance(x, dict)]
    objects.sort(key=lambda x: role_order([tok(r) for r in safe_list(x.get("event_roles"))]))
    object_by_id = {str(x.get("id")): x for x in objects if x.get("id")}

    object_lines = []
    role_kind_tokens, material_phase_tokens, property_tokens, initial_state_tokens = set(), set(), set(), set()
    object_names = []
    for obj in objects:
        oid = str(obj.get("id") or "")
        name = text(obj.get("name"))
        kind = tok(obj.get("entity_kind"))
        roles = [tok(r) for r in safe_list(obj.get("event_roles")) if tok(r)]
        material = safe_dict(obj.get("material"))
        phase = tok(material.get("phase"))
        props = [tok(p) for p in safe_list(material.get("canonical_properties")) if tok(p)]
        raw_props = [text(p) for p in safe_list(material.get("raw_properties")) if text(p)]
        initial = safe_dict(obj.get("initial_state"))
        motion = tok(initial.get("motion"))
        integrity = tok(initial.get("integrity"))
        if name:
            object_names.append(name.lower())
        for role in roles or ["unspecified"]:
            if kind:
                role_kind_tokens.add(f"{role}:{kind}")
        if phase and phase not in {"unknown", "not_applicable"}:
            material_phase_tokens.add(phase)
        property_tokens.update(props)
        if motion and motion != "unknown":
            initial_state_tokens.add(f"motion:{motion}")
        if integrity and integrity not in {"unknown", "not_applicable"}:
            initial_state_tokens.add(f"integrity:{integrity}")
        object_lines.append(
            "; ".join(
                x for x in [
                    f"object {name}" if name else "",
                    f"kind {kind}" if kind else "",
                    f"roles {', '.join(roles)}" if roles else "",
                    f"phase {phase}" if phase else "",
                    f"properties {', '.join(props + raw_props)}" if (props or raw_props) else "",
                    f"initial motion {motion}" if motion else "",
                    f"initial integrity {integrity}" if integrity else "",
                ] if x
            )
        )

    actions = [x for x in safe_list(process.get("actions")) if isinstance(x, dict)]
    action_tokens, action_phrases = set(), []
    for a in actions:
        af = tok(a.get("family")); at = tok(a.get("type")); raw = text(a.get("raw_action"))
        if af:
            action_tokens.add(f"family:{af}")
        if at:
            action_tokens.add(f"type:{at}")
        if raw:
            action_phrases.append(raw)

    secondary = [x for x in safe_list(process.get("secondary_processes")) if isinstance(x, dict)]
    secondary_tokens, secondary_phrases = set(), []
    for s in secondary:
        sf = tok(s.get("family")); st = tok(s.get("type")); sr = text(s.get("raw_type"))
        if sf:
            secondary_tokens.add(f"family:{sf}")
        if st:
            secondary_tokens.add(f"type:{st}")
        if sr:
            secondary_phrases.append(sr)

    temporal = safe_dict(process.get("temporal"))
    temporal_tokens = {
        f"extent:{tok(temporal.get('extent'))}",
        f"structure:{tok(temporal.get('structure'))}",
        f"profile:{tok(temporal.get('change_profile'))}",
    }
    temporal_tokens = {x for x in temporal_tokens if not x.endswith(":")}

    process_text = "; ".join(x for x in [
        f"primary family {family}",
        f"primary type {ptype}" if ptype else "",
        f"primary process {text(primary.get('raw_type'))}",
        f"process description {text(primary.get('description'))}",
        f"actions {', '.join(action_phrases)}" if action_phrases else "",
        f"secondary processes {', '.join(secondary_phrases)}" if secondary_phrases else "",
        f"temporal {tok(temporal.get('extent'))}, {tok(temporal.get('structure'))}, {tok(temporal.get('change_profile'))}",
    ] if x)

    impacts = [x for x in safe_list(card.get("impacts")) if isinstance(x, dict)]
    impact_axes, impact_types, impact_raw_phrases, impact_lines = set(), set(), [], []
    for imp in impacts:
        oid = str(imp.get("object_id") or "")
        obj = object_by_id.get(oid, {})
        obj_name = text(obj.get("name")) or oid
        transitions = [x for x in safe_list(imp.get("state_transitions")) if isinstance(x, dict)]
        tparts = []
        for tr in transitions:
            axis = tok(tr.get("axis")); tt = tok(tr.get("type")); raw = text(tr.get("raw_transition"))
            if axis:
                impact_axes.add(axis)
            if tt:
                impact_types.add(tt)
            if raw:
                impact_raw_phrases.append(raw)
            tparts.append(" / ".join(x for x in [axis, tt or "open", raw] if x))
        response = text(imp.get("response_description")); final = text(imp.get("final_state_description"))
        if response:
            impact_raw_phrases.append(response)
        if final:
            impact_raw_phrases.append(final)
        impact_lines.append(
            f"affected object {obj_name}; response {response}; transitions {' | '.join(tparts)}; final {final}"
        )

    mechanisms = [x for x in safe_list(card.get("mechanisms")) if isinstance(x, dict)]
    mechanism_families, mechanism_types, mechanism_raw_phrases, mechanism_lines = set(), set(), [], []
    for m in mechanisms:
        mf = tok(m.get("family")); mt = tok(m.get("type")); raw = text(m.get("raw_mechanism")); desc = text(m.get("description"))
        if mf:
            mechanism_families.add(mf)
        if mt:
            mechanism_types.add(mt)
        if raw:
            mechanism_raw_phrases.append(raw)
        if desc:
            mechanism_raw_phrases.append(desc)
        mechanism_lines.append("; ".join(x for x in [f"family {mf}" if mf else "", f"type {mt}" if mt else "", raw, desc] if x))

    object_text = " | ".join(object_lines)
    impact_text = " | ".join(impact_lines)
    mechanism_text = " | ".join(mechanism_lines) if mechanism_lines else "no explicit mechanism"

    # Holistic detailed-card representation for semantic recall.
    # This intentionally contains both closed ontology anchors and open-vocabulary
    # fine descriptions; precise O/P/I/M matching is performed separately later.
    language_text = " | ".join(
        x for x in [
            f"event summary {text(card.get('event_summary'))}" if text(card.get('event_summary')) else "",
            f"event description {text(card.get('raw_event_description'))}" if text(card.get('raw_event_description')) else "",
            f"objects {object_text}" if object_text else "",
            f"process {process_text}" if process_text else "",
            f"impacts {impact_text}" if impact_text else "",
            f"mechanisms {mechanism_text}" if mechanism_text else "",
        ] if x
    )

    # Useful raw lexical text for open-vocabulary comparison.
    process_raw_text = " ".join(x for x in [text(primary.get("raw_type")), text(primary.get("description")), *action_phrases, *secondary_phrases] if x)
    impact_raw_text = " ".join(impact_raw_phrases)
    mechanism_raw_text = " ".join(mechanism_raw_phrases)

    pass_a = safe_dict(payload.get("pass_a_source"))
    first = safe_dict(pass_a.get("first_frame"))

    source_fields = {}
    for key in ["source_zip", "source_shard", "label", "category", "video_path", "dataset_index"]:
        if key in manifest_row:
            source_fields[key] = manifest_row[key]

    return {
        "sample_id": sample_id,
        "eligible": eligible,
        "event_summary": text(card.get("event_summary")),
        "raw_event_description": text(card.get("raw_event_description")),
        "process_family": family,
        "process_type": ptype,
        "process_resolution_level": tok(primary.get("resolution_level")),
        "process_scope": scope,
        "first_frame_visibility": tok(first.get("key_object_visibility")),
        "first_frame_setup": tok(first.get("setup_clarity")),
        "texts": {
            "language": language_text,
            "object": object_text,
            "process": process_text,
            "impact": impact_text,
            "mechanism": mechanism_text,
            "process_raw": process_raw_text,
            "impact_raw": impact_raw_text,
            "mechanism_raw": mechanism_raw_text,
        },
        "tokens": {
            "object_role_kind": sorted(role_kind_tokens),
            "object_phase": sorted(material_phase_tokens),
            "object_property": sorted(property_tokens),
            "object_initial": sorted(initial_state_tokens),
            "action": sorted(action_tokens),
            "secondary_process": sorted(secondary_tokens),
            "temporal": sorted(temporal_tokens),
            "impact_axis": sorted(impact_axes),
            "impact_type": sorted(impact_types),
            "mechanism_family": sorted(mechanism_families),
            "mechanism_type": sorted(mechanism_types),
        },
        "object_names": sorted(set(object_names)),
        "source": source_fields,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pass-b-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.manifest)
    manifest_map = {str(r.get("sample_id")): r for r in rows}
    records = []
    failures = Counter()
    for row in rows:
        sid = str(row.get("sample_id"))
        path = args.pass_b_root / f"{sid}.json"
        if not path.is_file():
            failures["missing_card_file"] += 1
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            failures["invalid_json"] += 1
            continue
        rec = build_record(sid, payload, manifest_map.get(sid, {}))
        if rec is None:
            failures[str(payload.get("failure_reason") or "not_success")] += 1
            continue
        records.append(rec)

    eligible = [r for r in records if r["eligible"]]
    family_counts = Counter(r["process_family"] for r in eligible)
    type_counts = Counter(r["process_type"] or "<open>" for r in eligible)
    scope_counts = Counter(r["process_scope"] for r in records)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "retrieval_index_all.jsonl", records)
    write_jsonl(args.output_root / "retrieval_index_eligible.jsonl", eligible)
    summary = {
        "manifest_count": len(rows),
        "successful_card_count": len(records),
        "eligible_count": len(eligible),
        "excluded_count": len(records) - len(eligible),
        "failure_counts": dict(failures),
        "eligible_family_counts": dict(family_counts.most_common()),
        "eligible_type_counts": dict(type_counts.most_common()),
        "all_scope_counts": dict(scope_counts.most_common()),
        "eligibility_rule": "pass_b success AND primary family != special AND primary type != none AND scope != camera_only",
    }
    write_json(args.output_root / "index_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
