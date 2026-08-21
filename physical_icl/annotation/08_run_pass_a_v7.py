#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

from common_v7 import (  # noqa: E402
    confidence_paths,
    normalize_choice,
    parse_json_object,
    read_jsonl,
    safe_dict,
    safe_list,
    string_list,
    token,
    write_json,
)
from qwen3vl_runner import VideoRunner  # noqa: E402


STAGES = {"early", "middle", "late", "continuous", "unclear"}
SCOPE_VALUES = {"scene_physics", "camera_only", "mixed", "unclear"}
FIRST_FRAME_PHASES = {"pre_event", "onset", "mid_event", "post_event", "continuous", "unclear"}
VISIBILITY_VALUES = {"clear", "partial", "poor", "unclear"}
SETUP_VALUES = {"clear", "partial", "unclear"}
CAMERA_MOTION_VALUES = {"static", "pan", "tilt", "zoom", "tracking", "handheld", "shake", "mixed", "unknown"}
FOCUS_VALUES = {"none", "defocus", "refocus", "variable", "unknown"}
TECHNICAL_VALUES = {
    "temporal_coverage": {"complete", "partial", "unclear"},
    "event_clarity": {"clear", "partially_clear", "unclear"},
    "entity_trackability": {"good", "intermittent", "poor", "unclear"},
    "occlusion": {"none", "partial", "severe", "unclear"},
    "scene_cut": {"yes", "no", "unclear"},
    "camera_motion": {"low", "medium", "high", "unclear"},
}


def build_prompt(previous_errors: list[str] | None) -> str:
    correction = ""
    if previous_errors:
        correction = "\nThe previous response failed structural validation. Return the complete corrected JSON and fix:\n- " + "\n- ".join(previous_errors)
    return f"""
You are creating the observable evidence record for Physics Card v7.

Inspect the full video once. Before writing, identify the core visible event: the interaction or state change that best explains the clip as a whole. Center the event summary, description, objects, evidence, and timeline on that core event; include brief, background, preparatory, or incidental events only when they affect or contextualize it.

Record only visible facts. Do not normalize them into a physics ontology and do not infer hidden mechanisms such as gravity, pressure, friction, elasticity, heat transfer, or chemical energy release.

Do not use captions, filenames, dataset labels, or external metadata.
Do not output confidence, probability, certainty, likelihood, or confidence scores.

Separate scene physics from camera-only changes. Camera motion, zoom, and focus changes are not object translation, deformation, or optical scene physics unless a scene entity independently changes.

Return exactly one JSON object:

{{
  "schema_version": "observable_record_v7",
  "event_summary": "one concise factual sentence",
  "raw_event_description": "specific free-text description of the visible event and its temporal order",
  "observed_scope": "scene_physics | camera_only | mixed | unclear",
  "first_frame": {{
    "phase": "pre_event | onset | mid_event | post_event | continuous | unclear",
    "key_object_visibility": "clear | partial | poor | unclear",
    "setup_clarity": "clear | partial | unclear",
    "reasons": []
  }},
  "objects": [
    {{
      "id": "obj_1",
      "name": "short visible object name",
      "visible_form": "free-text physical form",
      "material_description": "only directly visible material cues",
      "visible_properties": [],
      "initial_motion_description": "visible motion before the main event",
      "initial_integrity_description": "visible integrity before the main event",
      "event_role_hints": []
    }}
  ],
  "initial_relation_observations": [
    {{
      "subject_id": "obj_1",
      "relation_description": "directly visible relation relevant to the event",
      "object_id": "obj_2",
      "evidence_refs": ["e1"]
    }}
  ],
  "evidence_bank": [
    {{
      "evidence_id": "e1",
      "stage": "early | middle | late | continuous | unclear",
      "observation": "one atomic directly visible observation"
    }}
  ],
  "timeline": [
    {{
      "stage": "early | middle | late | continuous | unclear",
      "evidence_refs": ["e1"],
      "observations": [],
      "object_events": [
        {{
          "object_id": "obj_1",
          "observation": "visible action, motion, response, emission, or state change"
        }}
      ]
    }}
  ],
  "camera_behavior": {{
    "motion_type": "static | pan | tilt | zoom | tracking | handheld | shake | mixed | unknown",
    "focus_change": "none | defocus | refocus | variable | unknown",
    "evidence_refs": []
  }},
  "technical_quality": {{
    "temporal_coverage": "complete | partial | unclear",
    "event_clarity": "clear | partially_clear | unclear",
    "entity_trackability": "good | intermittent | poor | unclear",
    "occlusion": "none | partial | severe | unclear",
    "scene_cut": "yes | no | unclear",
    "camera_motion": "low | medium | high | unclear"
  }},
  "unresolved_observations": []
}}

Rules:
1. Use stable object IDs obj_1, obj_2, and so on.
2. Evidence-bank observations must be atomic and visually grounded.
3. Timeline may use one continuous stage for a persistent process.
4. Do not create a causal chain or mechanism labels in Pass A.
5. Do not treat camera motion as scene-object motion.
6. Return JSON only.
{correction}
""".strip()


def normalize_output(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repairs: list[str] = []

    raw_evidence = [item for item in safe_list(raw.get("evidence_bank")) if isinstance(item, dict)]
    evidence_bank: list[dict[str, str]] = []
    evidence_map: dict[str, str] = {}
    for item in raw_evidence:
        observation = str(item.get("observation") or "").strip()
        if not observation:
            continue
        new_id = f"e{len(evidence_bank) + 1}"
        old_id = token(item.get("evidence_id"))
        if old_id:
            evidence_map[old_id] = new_id
        evidence_bank.append(
            {
                "evidence_id": new_id,
                "stage": normalize_choice(item.get("stage"), STAGES, "unclear"),
                "observation": observation,
            }
        )
    valid_evidence = {item["evidence_id"] for item in evidence_bank}

    raw_objects = [item for item in safe_list(raw.get("objects")) if isinstance(item, dict)]
    objects: list[dict[str, Any]] = []
    object_map: dict[str, str] = {}
    for item in raw_objects:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        new_id = f"obj_{len(objects) + 1}"
        old_id = str(item.get("id") or "").strip()
        if old_id:
            object_map[old_id] = new_id
        objects.append(
            {
                "id": new_id,
                "name": name,
                "visible_form": str(item.get("visible_form") or "").strip(),
                "material_description": str(item.get("material_description") or "").strip(),
                "visible_properties": string_list(item.get("visible_properties")),
                "initial_motion_description": str(item.get("initial_motion_description") or "").strip(),
                "initial_integrity_description": str(item.get("initial_integrity_description") or "").strip(),
                "event_role_hints": string_list(item.get("event_role_hints")),
            }
        )
    valid_objects = {item["id"] for item in objects}

    def map_object(value: Any) -> str:
        text = str(value or "").strip()
        if text in object_map:
            return object_map[text]
        return text if text in valid_objects else ""

    def map_refs(value: Any, defaults: list[str] | None = None) -> list[str]:
        output: list[str] = []
        for item in string_list(value):
            normalized = evidence_map.get(token(item), token(item))
            if normalized in valid_evidence and normalized not in output:
                output.append(normalized)
        if not output and defaults:
            output = [item for item in defaults if item in valid_evidence]
        return output

    relations: list[dict[str, Any]] = []
    for item in safe_list(raw.get("initial_relation_observations")):
        if not isinstance(item, dict):
            continue
        subject_id = map_object(item.get("subject_id"))
        object_id = map_object(item.get("object_id"))
        description = str(item.get("relation_description") or "").strip()
        if not subject_id or not object_id or not description:
            repairs.append("invalid_initial_relation_removed")
            continue
        relations.append(
            {
                "subject_id": subject_id,
                "relation_description": description,
                "object_id": object_id,
                "evidence_refs": map_refs(item.get("evidence_refs")),
            }
        )

    timeline: list[dict[str, Any]] = []
    for item in safe_list(raw.get("timeline")):
        if not isinstance(item, dict):
            continue
        stage = normalize_choice(item.get("stage"), STAGES, "unclear")
        stage_defaults = [e["evidence_id"] for e in evidence_bank if e["stage"] == stage]
        object_events: list[dict[str, str]] = []
        for event in safe_list(item.get("object_events")):
            if not isinstance(event, dict):
                continue
            object_id = map_object(event.get("object_id"))
            observation = str(event.get("observation") or "").strip()
            if object_id and observation:
                object_events.append({"object_id": object_id, "observation": observation})
        timeline.append(
            {
                "stage": stage,
                "evidence_refs": map_refs(item.get("evidence_refs"), stage_defaults),
                "observations": string_list(item.get("observations")),
                "object_events": object_events,
            }
        )

    first = safe_dict(raw.get("first_frame"))
    camera = safe_dict(raw.get("camera_behavior"))
    technical = safe_dict(raw.get("technical_quality"))

    normalized = {
        "schema_version": "observable_record_v7",
        "event_summary": str(raw.get("event_summary") or "").strip(),
        "raw_event_description": str(raw.get("raw_event_description") or raw.get("event_summary") or "").strip(),
        "observed_scope": normalize_choice(raw.get("observed_scope"), SCOPE_VALUES, "unclear"),
        "first_frame": {
            "phase": normalize_choice(first.get("phase"), FIRST_FRAME_PHASES, "unclear"),
            "key_object_visibility": normalize_choice(first.get("key_object_visibility"), VISIBILITY_VALUES, "unclear"),
            "setup_clarity": normalize_choice(first.get("setup_clarity"), SETUP_VALUES, "unclear"),
            "reasons": string_list(first.get("reasons")),
        },
        "objects": objects,
        "initial_relation_observations": relations,
        "evidence_bank": evidence_bank,
        "timeline": timeline,
        "camera_behavior": {
            "motion_type": normalize_choice(camera.get("motion_type"), CAMERA_MOTION_VALUES, "unknown"),
            "focus_change": normalize_choice(camera.get("focus_change"), FOCUS_VALUES, "unknown"),
            "evidence_refs": map_refs(camera.get("evidence_refs")),
        },
        "technical_quality": {
            field: normalize_choice(technical.get(field), allowed, "unclear")
            for field, allowed in TECHNICAL_VALUES.items()
        },
        "unresolved_observations": string_list(raw.get("unresolved_observations")),
        "prohibited_fields_detected": confidence_paths(raw),
    }

    if not timeline and evidence_bank:
        normalized["timeline"] = [
            {
                "stage": "continuous",
                "evidence_refs": [item["evidence_id"] for item in evidence_bank],
                "observations": [item["observation"] for item in evidence_bank],
                "object_events": [],
            }
        ]
        repairs.append("timeline_derived_from_evidence")

    return normalized, repairs


def validate_output(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != "observable_record_v7":
        errors.append("schema_version invalid")
    if not data.get("event_summary"):
        errors.append("event_summary empty")
    if not data.get("raw_event_description"):
        errors.append("raw_event_description empty")
    if not data.get("objects"):
        errors.append("objects must be non-empty")
    if not data.get("evidence_bank"):
        errors.append("evidence_bank must be non-empty")
    if not data.get("timeline"):
        errors.append("timeline must be non-empty")
    if data.get("prohibited_fields_detected"):
        errors.append("confidence-like fields are forbidden")
    valid_objects = {item.get("id") for item in safe_list(data.get("objects")) if isinstance(item, dict)}
    valid_evidence = {item.get("evidence_id") for item in safe_list(data.get("evidence_bank")) if isinstance(item, dict)}
    for index, relation in enumerate(safe_list(data.get("initial_relation_observations"))):
        if not isinstance(relation, dict):
            errors.append(f"initial_relation_observations[{index}] invalid")
            continue
        if relation.get("subject_id") not in valid_objects or relation.get("object_id") not in valid_objects:
            errors.append(f"initial_relation_observations[{index}] references unknown object")
    for index, item in enumerate(safe_list(data.get("timeline"))):
        if not isinstance(item, dict):
            errors.append(f"timeline[{index}] invalid")
            continue
        if not item.get("observations") and not item.get("object_events"):
            errors.append(f"timeline[{index}] has no content")
        invalid_refs = [ref for ref in string_list(item.get("evidence_refs")) if ref not in valid_evidence]
        if invalid_refs:
            errors.append(f"timeline[{index}] has invalid evidence refs")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-frames", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=3200)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    output_dir = args.output_root / "pass_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = VideoRunner(args.model_path)
    summary_records: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="Pass A v7"):
        sample_id = str(row["sample_id"])
        video_path = Path(row["video_path"])
        output_path = output_dir / f"{sample_id}.json"
        if output_path.is_file() and not args.overwrite:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                summary_records.append({"sample_id": sample_id, "status": "skipped_existing"})
                continue

        payload: dict[str, Any] = {
            "sample_id": sample_id,
            "video_path": str(video_path),
            "status": "failed",
            "attempts": [],
        }
        previous_errors: list[str] | None = None
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            for attempt_index in range(args.retries + 1):
                raw_response, sampling = runner.generate(
                    video_path=video_path,
                    prompt=build_prompt(previous_errors),
                    fps=args.fps,
                    min_frames=args.min_frames,
                    max_frames=args.max_frames,
                    max_new_tokens=args.max_new_tokens,
                )
                parsed = parse_json_object(raw_response)
                if parsed is None:
                    normalized = None
                    repairs: list[str] = []
                    errors = ["response is not JSON"]
                else:
                    normalized, repairs = normalize_output(parsed)
                    errors = validate_output(normalized)
                payload["attempts"].append(
                    {
                        "attempt": attempt_index + 1,
                        "raw_response": raw_response,
                        "parse_ok": parsed is not None,
                        "normalization_repairs": repairs,
                        "validation_errors": errors,
                        "sampling": sampling,
                    }
                )
                if normalized is not None and not errors:
                    payload.update(
                        {
                            "status": "success",
                            "schema_valid": True,
                            "observable_record": normalized,
                            "normalization_repairs": repairs,
                            "sampling": sampling,
                        }
                    )
                    break
                previous_errors = errors
            if payload["status"] != "success":
                payload["failure_reason"] = "schema_validation_failed"
        except Exception as error:
            payload["failure_reason"] = "runtime_error"
            payload["error"] = repr(error)
            payload["traceback"] = traceback.format_exc()

        write_json(output_path, payload)
        summary_records.append(
            {
                "sample_id": sample_id,
                "status": payload["status"],
                "failure_reason": payload.get("failure_reason", ""),
            }
        )
        write_json(
            args.output_root / "run_summary.json",
            {
                "records": summary_records,
                "counts": dict(Counter(item["status"] for item in summary_records)),
                "failure_reasons": dict(Counter(item["failure_reason"] for item in summary_records if item.get("failure_reason"))),
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(Counter(item["status"] for item in summary_records))


if __name__ == "__main__":
    main()
