#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common_v7 import (
    collect_evidence_refs,
    load_ontology,
    read_jsonl,
    safe_dict,
    safe_list,
    token,
    valid_evidence_ids,
    write_json,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def family_names(ontology: dict[str, Any], section: str, key: str) -> set[str]:
    return {
        token(item.get("label"))
        for item in safe_list(safe_dict(ontology.get(section)).get(key))
        if isinstance(item, dict) and token(item.get("label"))
    }


def semantic_issues(card: dict[str, Any]) -> list[str]:
    process = safe_dict(card.get("process"))
    scope = token(process.get("scope"))
    primary = safe_dict(process.get("primary_process"))
    family = token(primary.get("family"))
    subtype = token(primary.get("type"))
    temporal = safe_dict(process.get("temporal"))
    extent = token(temporal.get("extent"))
    structure = token(temporal.get("structure"))
    profile = token(temporal.get("change_profile"))

    axes = {
        token(transition.get("axis"))
        for impact in safe_list(card.get("impacts"))
        if isinstance(impact, dict)
        for transition in safe_list(impact.get("state_transitions"))
        if isinstance(transition, dict) and token(transition.get("axis"))
    }

    issues: list[str] = []
    if extent == "static" and structure != "not_applicable":
        issues.append("static_structure_inconsistent")
    if extent == "static" and profile != "none":
        issues.append("static_profile_inconsistent")
    if structure == "not_applicable" and extent != "static":
        issues.append("nonstatic_structure_not_applicable")
    if scope == "camera_only" and not (family == "special" and subtype == "none"):
        issues.append("camera_only_primary_inconsistent")
    if scope == "camera_only" and card.get("impacts"):
        issues.append("camera_only_has_impacts")

    expected_axis = {
        "deformation": "geometry",
        "fracture_separation": "integrity",
        "phase_transition": "phase",
        "relation_reconfiguration": "relation",
    }.get(family)
    if expected_axis and expected_axis not in axes:
        issues.append(f"{family}_missing_{expected_axis}_impact")
    if family == "combustion" and not (
        {"thermal_reaction", "emission_transport", "optical_visibility"} & axes
    ):
        issues.append("combustion_missing_reaction_emission_or_optical_impact")
    if family == "explosive_release" and not (
        {"integrity", "emission_transport", "motion", "optical_visibility"} & axes
    ):
        issues.append("explosive_release_missing_observable_impact")
    if family == "special" and subtype == "none" and card.get("impacts"):
        issues.append("none_primary_has_impacts")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pass-a-root", type=Path, required=True)
    parser.add_argument("--pass-b-root", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--example-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    ontology = load_ontology(args.ontology)

    valid_process_families = family_names(ontology, "process", "primary_families")
    valid_impact_axes = family_names(ontology, "impact", "transition_axes")
    valid_mechanism_families = family_names(ontology, "mechanism", "families")

    pass_a_status = Counter()
    pass_b_status = Counter()
    pass_a_failures = Counter()
    pass_b_failures = Counter()
    pass_b_repairs = Counter()
    pass_b_notes = Counter()
    review_reasons = Counter()
    semantic_issue_counts = Counter()

    observed_scope = Counter()
    process_scope = Counter()
    primary_families = Counter()
    primary_types = Counter()
    primary_resolution = Counter()
    temporal_extent = Counter()
    temporal_structure = Counter()
    change_profile = Counter()

    entity_kinds = Counter()
    event_roles = Counter()
    material_phases = Counter()
    canonical_properties = Counter()
    raw_property_count = 0

    action_families = Counter()
    action_types = Counter()
    action_resolution = Counter()
    transition_axes = Counter()
    transition_types = Counter()
    transition_resolution = Counter()
    mechanism_families = Counter()
    mechanism_types = Counter()
    mechanism_resolution = Counter()

    evidence_counts: list[float] = []
    object_counts: list[float] = []
    timeline_counts: list[float] = []
    action_counts: list[float] = []
    impact_counts: list[float] = []
    transition_counts: list[float] = []
    mechanism_counts: list[float] = []
    technical_scores: list[float] = []

    invalid_evidence_refs = 0
    object_id_mismatches = 0
    invalid_process_family = 0
    invalid_impact_axis = 0
    invalid_mechanism_family = 0
    temporal_consistency_errors = 0
    semantic_issue_samples = 0

    unknown_entity_kind = 0
    unknown_material_phase = 0
    core_objects_without_roles = 0
    primary_family_unresolved = 0
    primary_canonical_type_resolved = 0
    scene_process_without_impact = 0
    scene_process_without_transition = 0
    samples_without_mechanism = 0

    audit_rows: list[dict[str, Any]] = []
    family_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for manifest_row in rows:
        sample_id = str(manifest_row["sample_id"])
        pass_a_path = args.pass_a_root / f"{sample_id}.json"
        pass_b_path = args.pass_b_root / f"{sample_id}.json"
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "video_path": manifest_row.get("video_path", ""),
        }
        risk_reasons: list[str] = []
        risk_score = 0

        pass_a_payload = read_json(pass_a_path) if pass_a_path.is_file() else {}
        pass_b_payload = read_json(pass_b_path) if pass_b_path.is_file() else {}
        a_status = str(pass_a_payload.get("status", "missing"))
        b_status = str(pass_b_payload.get("status", "missing"))
        pass_a_status[a_status] += 1
        pass_b_status[b_status] += 1
        row["pass_a_status"] = a_status
        row["pass_b_status"] = b_status

        if a_status != "success":
            reason = str(pass_a_payload.get("failure_reason", "missing_file"))
            pass_a_failures[reason] += 1
            risk_score += 100
            risk_reasons.append(f"pass_a:{reason}")
        if b_status != "success":
            reason = str(pass_b_payload.get("failure_reason", "missing_file"))
            pass_b_failures[reason] += 1
            risk_score += 100
            risk_reasons.append(f"pass_b:{reason}")

        pass_a = safe_dict(pass_a_payload.get("observable_record"))
        card = safe_dict(pass_b_payload.get("physics_card"))
        derived = safe_dict(pass_b_payload.get("derived"))

        for repair in safe_list(pass_b_payload.get("normalization_repairs")):
            pass_b_repairs[str(repair)] += 1
        for note in safe_list(pass_b_payload.get("normalization_notes")):
            pass_b_notes[str(note)] += 1

        if pass_a:
            evidence = safe_list(pass_a.get("evidence_bank"))
            objects = safe_list(pass_a.get("objects"))
            timeline = safe_list(pass_a.get("timeline"))
            evidence_counts.append(float(len(evidence)))
            object_counts.append(float(len(objects)))
            timeline_counts.append(float(len(timeline)))
            observed_scope[token(pass_a.get("observed_scope")) or "missing"] += 1
            row["event_summary"] = str(pass_a.get("event_summary") or "")
            row["raw_event_description"] = str(
                pass_a.get("raw_event_description") or ""
            )
            row["observed_scope"] = token(pass_a.get("observed_scope"))

        if card:
            valid_refs = valid_evidence_ids(pass_a)
            invalid_refs = sorted(
                {ref for ref in collect_evidence_refs(card) if ref not in valid_refs}
            )
            invalid_evidence_refs += len(invalid_refs)
            if invalid_refs:
                risk_score += 20
                risk_reasons.append("invalid_evidence_refs")

            pass_a_ids = {
                str(item.get("id"))
                for item in safe_list(pass_a.get("objects"))
                if isinstance(item, dict)
            }
            card_ids = {
                str(item.get("id"))
                for item in safe_list(card.get("objects"))
                if isinstance(item, dict)
            }
            if pass_a_ids != card_ids:
                object_id_mismatches += 1
                risk_score += 20
                risk_reasons.append("object_id_mismatch")

            role_by_id: dict[str, list[str]] = {}
            for item in safe_list(card.get("objects")):
                if not isinstance(item, dict):
                    continue
                object_id = str(item.get("id") or "")
                kind = token(item.get("entity_kind")) or "missing"
                entity_kinds[kind] += 1
                if kind == "unknown":
                    unknown_entity_kind += 1
                roles = [
                    token(value)
                    for value in safe_list(item.get("event_roles"))
                    if token(value)
                ]
                role_by_id[object_id] = roles
                for value in roles:
                    event_roles[value] += 1
                material = safe_dict(item.get("material"))
                phase = token(material.get("phase")) or "missing"
                material_phases[phase] += 1
                if phase == "unknown":
                    unknown_material_phase += 1
                for value in safe_list(material.get("canonical_properties")):
                    if token(value):
                        canonical_properties[token(value)] += 1
                raw_property_count += len(safe_list(material.get("raw_properties")))

            process = safe_dict(card.get("process"))
            scope = token(process.get("scope")) or "missing"
            process_scope[scope] += 1
            primary = safe_dict(process.get("primary_process"))
            primary_family = token(primary.get("family")) or "missing"
            primary_type = token(primary.get("type"))
            primary_level = token(primary.get("resolution_level")) or "missing"
            primary_families[primary_family] += 1
            primary_types[primary_type or "null"] += 1
            primary_resolution[primary_level] += 1
            row["process_scope"] = scope
            row["primary_family"] = primary_family
            row["primary_type"] = primary_type or ""
            row["primary_raw_type"] = str(primary.get("raw_type") or "")
            row["primary_resolution"] = primary_level

            if primary_family not in valid_process_families:
                invalid_process_family += 1
                risk_score += 20
                risk_reasons.append("invalid_process_family")

            family_is_resolved = bool(
                primary_family in valid_process_families
                and (
                    primary_family != "special"
                    or primary_type == "none"
                )
            )
            if not family_is_resolved or primary_level == "unresolved":
                primary_family_unresolved += 1
                risk_score += 8
                risk_reasons.append("primary_family_unresolved")
            if primary_type:
                primary_canonical_type_resolved += 1

            actions = [
                item
                for item in safe_list(process.get("actions"))
                if isinstance(item, dict)
            ]
            action_counts.append(float(len(actions)))
            for item in actions:
                action_families[token(item.get("family")) or "missing"] += 1
                action_types[token(item.get("type")) or "null"] += 1
                action_resolution[
                    token(item.get("resolution_level")) or "missing"
                ] += 1

            temporal = safe_dict(process.get("temporal"))
            temporal_extent[token(temporal.get("extent")) or "missing"] += 1
            temporal_structure[
                token(temporal.get("structure")) or "missing"
            ] += 1
            change_profile[
                token(temporal.get("change_profile")) or "missing"
            ] += 1

            impacts = [
                item
                for item in safe_list(card.get("impacts"))
                if isinstance(item, dict)
            ]
            impact_counts.append(float(len(impacts)))
            transition_count = 0
            impact_object_ids = {str(item.get("object_id") or "") for item in impacts}
            primary_subject_ids = {
                str(value) for value in safe_list(primary.get("subject_ids"))
            }
            for core_id in impact_object_ids | primary_subject_ids:
                if core_id and not role_by_id.get(core_id):
                    core_objects_without_roles += 1

            for impact in impacts:
                for transition in safe_list(impact.get("state_transitions")):
                    if not isinstance(transition, dict):
                        continue
                    transition_count += 1
                    axis = token(transition.get("axis")) or "missing"
                    subtype = token(transition.get("type"))
                    level = token(transition.get("resolution_level")) or "missing"
                    transition_axes[axis] += 1
                    transition_types[subtype or "null"] += 1
                    transition_resolution[level] += 1
                    if axis not in valid_impact_axes:
                        invalid_impact_axis += 1
                        risk_score += 10
                        risk_reasons.append("invalid_impact_axis")
            transition_counts.append(float(transition_count))

            if (
                scope == "scene_physics"
                and primary_family != "special"
                and not impacts
            ):
                scene_process_without_impact += 1
                risk_score += 8
                risk_reasons.append("scene_process_without_impact")
            if (
                scope == "scene_physics"
                and primary_family != "special"
                and transition_count == 0
            ):
                scene_process_without_transition += 1
                risk_score += 5
                risk_reasons.append("scene_process_without_transition")

            mechanisms = [
                item
                for item in safe_list(card.get("mechanisms"))
                if isinstance(item, dict)
            ]
            mechanism_counts.append(float(len(mechanisms)))
            if not mechanisms:
                samples_without_mechanism += 1
            for item in mechanisms:
                family = token(item.get("family")) or "missing"
                subtype = token(item.get("type"))
                level = token(item.get("resolution_level")) or "missing"
                mechanism_families[family] += 1
                mechanism_types[subtype or "null"] += 1
                mechanism_resolution[level] += 1
                if family not in valid_mechanism_families:
                    invalid_mechanism_family += 1
                    risk_score += 10
                    risk_reasons.append("invalid_mechanism_family")

            issues = semantic_issues(card)
            if issues:
                semantic_issue_samples += 1
            for issue in issues:
                semantic_issue_counts[issue] += 1
                if issue.startswith("static_") or issue.startswith("camera_only_"):
                    temporal_consistency_errors += 1
                    risk_score += 8
                else:
                    risk_score += 3
                risk_reasons.append(issue)

            score = derived.get("technical_quality_score")
            if isinstance(score, (int, float)):
                technical_scores.append(float(score))
            for reason in safe_list(derived.get("review_reasons")):
                review_reasons[str(reason)] += 1

        row["risk_score"] = risk_score
        row["risk_reasons"] = " | ".join(sorted(set(risk_reasons)))
        row["needs_review"] = bool(derived.get("needs_review"))
        audit_rows.append(row)
        if row.get("primary_family"):
            family_examples[row["primary_family"]].append(row)

    successful_cards = pass_b_status.get("success", 0)
    total_transitions = sum(transition_resolution.values())
    total_mechanisms = sum(mechanism_resolution.values())

    summary = {
        "manifest_count": len(rows),
        "pass_a_status": dict(pass_a_status),
        "pass_b_status": dict(pass_b_status),
        "pass_a_failure_reasons": dict(pass_a_failures),
        "pass_b_failure_reasons": dict(pass_b_failures),
        "pass_a_statistics": {
            "evidence_count": summarize(evidence_counts),
            "object_count": summarize(object_counts),
            "timeline_count": summarize(timeline_counts),
            "observed_scope": dict(observed_scope),
        },
        "object_statistics": {
            "entity_kinds": dict(entity_kinds.most_common()),
            "event_roles": dict(event_roles.most_common()),
            "material_phases": dict(material_phases.most_common()),
            "canonical_properties": dict(canonical_properties.most_common()),
            "raw_property_count": raw_property_count,
            "unknown_entity_kind_count": unknown_entity_kind,
            "unknown_material_phase_count": unknown_material_phase,
            "core_objects_without_roles_count": core_objects_without_roles,
        },
        "process_statistics": {
            "scope": dict(process_scope),
            "primary_families": dict(primary_families.most_common()),
            "primary_types": dict(primary_types.most_common()),
            "primary_resolution_levels": dict(primary_resolution.most_common()),
            "primary_family_unresolved_count": primary_family_unresolved,
            "primary_family_resolved_rate": ratio(
                successful_cards - primary_family_unresolved, successful_cards
            ),
            "primary_canonical_type_count": primary_canonical_type_resolved,
            "primary_canonical_type_rate": ratio(
                primary_canonical_type_resolved, successful_cards
            ),
            "action_families": dict(action_families.most_common()),
            "action_types": dict(action_types.most_common()),
            "action_resolution_levels": dict(action_resolution.most_common()),
            "action_count": summarize(action_counts),
            "temporal_extent": dict(temporal_extent),
            "temporal_structure": dict(temporal_structure),
            "change_profile": dict(change_profile),
        },
        "impact_statistics": {
            "impact_object_count": summarize(impact_counts),
            "transition_count": summarize(transition_counts),
            "transition_axes": dict(transition_axes.most_common()),
            "transition_types": dict(transition_types.most_common()),
            "transition_resolution_levels": dict(transition_resolution.most_common()),
            "canonical_transition_type_rate": ratio(
                transition_resolution.get("canonical_type", 0), total_transitions
            ),
            "axis_resolved_rate": ratio(
                total_transitions - transition_resolution.get("unresolved", 0),
                total_transitions,
            ),
            "scene_process_without_impact_count": scene_process_without_impact,
            "scene_process_without_transition_count": scene_process_without_transition,
        },
        "mechanism_statistics": {
            "mechanism_count": summarize(mechanism_counts),
            "families": dict(mechanism_families.most_common()),
            "types": dict(mechanism_types.most_common()),
            "resolution_levels": dict(mechanism_resolution.most_common()),
            "canonical_type_rate": ratio(
                mechanism_resolution.get("canonical_type", 0), total_mechanisms
            ),
            "family_resolved_rate": ratio(
                total_mechanisms - mechanism_resolution.get("unresolved", 0),
                total_mechanisms,
            ),
            "samples_without_mechanism_count": samples_without_mechanism,
        },
        "consistency_checks": {
            "invalid_evidence_reference_count": invalid_evidence_refs,
            "object_id_mismatch_count": object_id_mismatches,
            "invalid_process_family_count": invalid_process_family,
            "invalid_impact_axis_count": invalid_impact_axis,
            "invalid_mechanism_family_count": invalid_mechanism_family,
            "temporal_consistency_error_count": temporal_consistency_errors,
            "semantic_issue_sample_count": semantic_issue_samples,
            "semantic_issue_counts": dict(semantic_issue_counts.most_common()),
        },
        "technical_quality": summarize(technical_scores),
        "review_reasons": dict(review_reasons.most_common()),
        "normalization": {
            "repairs": dict(pass_b_repairs.most_common()),
            "notes": dict(pass_b_notes.most_common()),
        },
    }

    write_json(args.output_root / "audit_summary.json", summary)
    write_csv(args.output_root / "audit_rows.csv", audit_rows)

    rng = random.Random(args.seed)
    high_risk = sorted(
        audit_rows,
        key=lambda item: (-int(item.get("risk_score", 0)), item["sample_id"]),
    )[: args.example_count // 2]
    selected_ids = {item["sample_id"] for item in high_risk}
    representatives: list[dict[str, Any]] = []
    for family in sorted(family_examples):
        candidates = [
            item
            for item in family_examples[family]
            if item["sample_id"] not in selected_ids
            and item.get("pass_b_status") == "success"
        ]
        if candidates:
            rng.shuffle(candidates)
            representatives.append(candidates[0])
            selected_ids.add(candidates[0]["sample_id"])
        if len(high_risk) + len(representatives) >= args.example_count:
            break

    lines: list[str] = []
    for index, item in enumerate(high_risk + representatives, start=1):
        lines.extend(
            [
                "=" * 110,
                f"EXAMPLE {index:02d}",
                "=" * 110,
                f"Sample ID: {item['sample_id']}",
                f"Pass A / Pass B: {item.get('pass_a_status')} / {item.get('pass_b_status')}",
                f"Risk score: {item.get('risk_score')}",
                f"Risk reasons: {item.get('risk_reasons') or 'none'}",
                f"Observed scope: {item.get('observed_scope', '')}",
                f"Process scope: {item.get('process_scope', '')}",
                f"Primary: {item.get('primary_family', '')} / {item.get('primary_type', '')}",
                f"Resolution: {item.get('primary_resolution', '')}",
                f"Raw type: {item.get('primary_raw_type', '')}",
                f"Event summary: {item.get('event_summary', '')}",
                f"Raw event: {item.get('raw_event_description', '')}",
                "",
            ]
        )
    (args.output_root / "audit_examples.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
