from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


CONFIDENCE_KEYS = {
    "confidence",
    "confidence_score",
    "probability",
    "certainty",
    "likelihood",
    "mapping_confidence",
}


def token(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in safe_list(value) if str(item).strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(stripped[start : index + 1])
                    return value if isinstance(value, dict) else None
                except Exception:
                    return None
    return None


def walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


def confidence_paths(value: Any) -> list[str]:
    found: list[str] = []
    for path, current in walk(value):
        if not isinstance(current, dict):
            continue
        for key in current:
            if token(key) in CONFIDENCE_KEYS:
                found.append(f"{path}.{key}")
    return sorted(set(found))


def load_ontology(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Ontology must be a dictionary")
    return value


def axis_labels(ontology: dict[str, Any], section: str, axis_name: str) -> set[str]:
    for axis in safe_list(safe_dict(ontology.get(section)).get("axes")):
        if not isinstance(axis, dict) or axis.get("name") != axis_name:
            continue
        return {token(item.get("label")) for item in safe_list(axis.get("labels")) if isinstance(item, dict)}
    return set()


def temporal_axis_labels(ontology: dict[str, Any], axis_name: str) -> set[str]:
    for axis in safe_list(safe_dict(ontology.get("process")).get("temporal_axes")):
        if not isinstance(axis, dict) or axis.get("name") != axis_name:
            continue
        return {token(item.get("label")) for item in safe_list(axis.get("labels")) if isinstance(item, dict)}
    return set()


def family_type_map(ontology: dict[str, Any], path: tuple[str, str]) -> dict[str, str]:
    section = safe_dict(ontology.get(path[0]))
    families = safe_list(section.get(path[1]))
    result: dict[str, str] = {}
    for family in families:
        if not isinstance(family, dict):
            continue
        family_name = token(family.get("label"))
        for item in safe_list(family.get("types")):
            if isinstance(item, dict):
                result[token(item.get("label"))] = family_name
    return result


def impact_type_map(ontology: dict[str, Any]) -> dict[str, str]:
    return family_type_map(ontology, ("impact", "transition_axes"))


def compact_ontology_for_prompt(ontology: dict[str, Any]) -> dict[str, Any]:
    def compact_axis(axis: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": axis.get("name"),
            "cardinality": axis.get("cardinality"),
            "labels": [
                {"label": item.get("label"), "definition": item.get("definition")}
                for item in safe_list(axis.get("labels"))
                if isinstance(item, dict)
            ],
        }

    def compact_families(families: list[Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for family in families:
            if not isinstance(family, dict):
                continue
            output.append(
                {
                    "family": family.get("label"),
                    "definition": family.get("definition"),
                    "types": [
                        {"label": item.get("label"), "definition": item.get("definition")}
                        for item in safe_list(family.get("types"))
                        if isinstance(item, dict)
                    ],
                }
            )
        return output

    return {
        "object_axes": [compact_axis(axis) for axis in safe_list(safe_dict(ontology.get("object")).get("axes")) if isinstance(axis, dict)],
        "action_families": compact_families(safe_list(safe_dict(ontology.get("process")).get("action_families"))),
        "primary_process_families": compact_families(safe_list(safe_dict(ontology.get("process")).get("primary_families"))),
        "temporal_axes": [compact_axis(axis) for axis in safe_list(safe_dict(ontology.get("process")).get("temporal_axes")) if isinstance(axis, dict)],
        "impact_transition_axes": compact_families(safe_list(safe_dict(ontology.get("impact")).get("transition_axes"))),
        "mechanism_families": compact_families(safe_list(safe_dict(ontology.get("mechanism")).get("families"))),
        "mechanism_basis_values": safe_list(safe_dict(ontology.get("mechanism")).get("basis_values")),
    }


def normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = token(value)
    return normalized if normalized in allowed else default


def normalize_multi(value: Any, allowed: set[str], *, maximum: int | None = None) -> list[str]:
    output: list[str] = []
    for item in safe_list(value):
        normalized = token(item)
        if normalized in allowed and normalized not in output:
            output.append(normalized)
        if maximum is not None and len(output) >= maximum:
            break
    return output


def collect_evidence_refs(value: Any) -> list[str]:
    refs: list[str] = []
    for _, current in walk(value):
        if not isinstance(current, dict):
            continue
        for item in string_list(current.get("evidence_refs")):
            normalized = token(item)
            if normalized:
                refs.append(normalized)
    return refs


def valid_evidence_ids(pass_a: dict[str, Any]) -> set[str]:
    return {
        token(item.get("evidence_id"))
        for item in safe_list(pass_a.get("evidence_bank"))
        if isinstance(item, dict) and token(item.get("evidence_id"))
    }


def normalize_evidence_refs(value: Any, valid_ids: set[str], defaults: list[str] | None = None) -> list[str]:
    output: list[str] = []
    for item in string_list(value):
        normalized = token(item)
        if normalized in valid_ids and normalized not in output:
            output.append(normalized)
    if not output and defaults:
        output = [item for item in defaults if item in valid_ids]
    return output


def technical_quality_score(pass_a: dict[str, Any]) -> float:
    technical = safe_dict(pass_a.get("technical_quality"))
    maps = {
        "temporal_coverage": {"complete": 1.0, "partial": 0.5, "unclear": 0.0},
        "event_clarity": {"clear": 1.0, "partially_clear": 0.5, "unclear": 0.0},
        "entity_trackability": {"good": 1.0, "intermittent": 0.5, "poor": 0.0, "unclear": 0.0},
        "occlusion": {"none": 1.0, "partial": 0.5, "severe": 0.0, "unclear": 0.0},
        "scene_cut": {"no": 1.0, "yes": 0.0, "unclear": 0.25},
        "camera_motion": {"low": 1.0, "medium": 0.7, "high": 0.3, "unclear": 0.5},
    }
    values = [mapping.get(token(technical.get(field)), 0.0) for field, mapping in maps.items()]
    return round(sum(values) / len(values), 4)
