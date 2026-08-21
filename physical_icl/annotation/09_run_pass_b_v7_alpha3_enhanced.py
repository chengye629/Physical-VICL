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

PROTOCOL_VERSION = "pass_b_v7_alpha3_enhanced_v1"
PRIMARY_CONFIDENCE_VALUES = {"high", "medium", "low", "unknown"}
MECHANISM_CONFIDENCE_VALUES = {"high", "medium", "low", "unknown"}
MECHANISM_SUPPORT_LEVELS = {
    "strongly_supported",
    "supported_inference",
    "tentative",
}


def prohibited_confidence_paths(value: Any) -> list[str]:
    allowed_primary = "$.process.primary_process.confidence"
    return [
        path
        for path in confidence_paths(value)
        if path != allowed_primary
        and not (
            path.startswith("$.mechanisms[")
            and path.endswith("].confidence")
        )
    ]

from common_v7 import (  # noqa: E402
    axis_labels,
    collect_evidence_refs,
    compact_ontology_for_prompt,
    confidence_paths,
    family_type_map,
    impact_type_map,
    load_ontology,
    normalize_choice,
    normalize_evidence_refs,
    normalize_multi,
    parse_json_object,
    read_jsonl,
    safe_dict,
    safe_list,
    string_list,
    technical_quality_score,
    temporal_axis_labels,
    token,
    valid_evidence_ids,
    write_json,
)
from qwen3vl_runner import TextRunner  # noqa: E402


def family_sets(ontology: dict[str, Any], path: tuple[str, str]) -> dict[str, set[str]]:
    section = safe_dict(ontology.get(path[0]))
    output: dict[str, set[str]] = {}
    for family in safe_list(section.get(path[1])):
        if not isinstance(family, dict):
            continue
        family_name = token(family.get("label"))
        if not family_name:
            continue
        output[family_name] = {
            token(item.get("label"))
            for item in safe_list(family.get("types"))
            if isinstance(item, dict) and token(item.get("label"))
        }
    return output


def alias_map(ontology: dict[str, Any], name: str) -> dict[str, str]:
    aliases = safe_dict(ontology.get("aliases"))
    raw = safe_dict(aliases.get(name))
    return {token(key): token(value) for key, value in raw.items() if token(key) and token(value)}


def text_value(value: Any) -> str:
    return str(value or "").strip()


def normalized_string_list(value: Any, maximum: int | None = None) -> list[str]:
    output: list[str] = []
    for item in safe_list(value):
        text = text_value(item)
        if text and text not in output:
            output.append(text)
        if maximum is not None and len(output) >= maximum:
            break
    return output


def normalize_primary_legacy_fields(
    raw_family: Any,
    raw_type: Any,
    raw_text: Any,
) -> tuple[Any, Any]:
    """Handle a few alpha-2 family boundaries without enumerating phrases."""
    family = token(raw_family)
    subtype = token(raw_type)
    combined = " ".join(
        part
        for part in (
            text_value(raw_family),
            text_value(raw_type),
            text_value(raw_text),
        )
        if part
    ).lower()

    if family == "combustion_energetic":
        if any(word in combined for word in ("explos", "deton", "blast", "burst")):
            family = "explosive_release"
        else:
            family = "combustion"

    if family == "static_persistent":
        if any(word in combined for word in ("smoke", "steam", "plume", "vapor", "emission")):
            family = "gas_particulate_motion"
        elif any(word in combined for word in ("light", "illumination", "glow", "sunset", "shadow")):
            family = "optical_interaction"
        else:
            family = "special"
            subtype = subtype or "none"

    if family == "thermal_transfer":
        optical_words = ("sunset", "illumination", "brightness", "sky color", "shadow", "light change")
        thermal_words = ("heat", "temperature", "melt", "boil", "warm", "cool", "char")
        if any(word in combined for word in optical_words) and not any(
            word in combined for word in thermal_words
        ):
            family = "optical_interaction"
            subtype = "illumination_change"

    return family or raw_family, subtype or raw_type


def derive_event_roles(
    objects: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    primary: dict[str, Any],
    impacts: list[dict[str, Any]],
) -> list[str]:
    """Complete roles from explicit structural references; return repair notes."""
    notes: list[str] = []
    by_id = {str(item.get("id")): item for item in objects if item.get("id")}

    def add_role(object_id: str, role: str, source: str) -> None:
        item = by_id.get(object_id)
        if item is None:
            return
        roles = item.setdefault("event_roles", [])
        if role not in roles:
            roles.append(role)
            notes.append(f"derived_role:{role}:{source}")

    for action in actions:
        for object_id in safe_list(action.get("actor_ids")):
            add_role(str(object_id), "actor", "action_actor")
        for object_id in safe_list(action.get("target_ids")):
            add_role(str(object_id), "target", "action_target")

    for impact in impacts:
        add_role(str(impact.get("object_id") or ""), "target", "impact_object")

    for object_id in safe_list(primary.get("subject_ids")):
        object_id = str(object_id)
        item = by_id.get(object_id)
        if item is not None and not item.get("event_roles"):
            add_role(object_id, "target", "primary_subject")

    return notes


def resolve_hierarchical_label(
    *,
    raw_family: Any,
    raw_type: Any,
    open_text: Any = None,
    families: dict[str, set[str]],
    family_aliases: dict[str, str],
    type_aliases: dict[str, str],
    fallback_family: str | None = None,
    fallback_type: str | None = None,
) -> tuple[str | None, str | None, str, str, list[str]]:
    """Resolve a closed family and optional canonical subtype.

    Returns: family, canonical_type, raw_text, resolution_level, notes.
    resolution_level is one of canonical_type, family_only, open_vocab, unresolved.
    """

    notes: list[str] = []
    family_token = family_aliases.get(token(raw_family), token(raw_family))
    type_token_original = token(raw_type)
    type_token = type_aliases.get(type_token_original, type_token_original)
    raw_text = text_value(open_text) or text_value(raw_type) or text_value(raw_family)

    type_to_family = {
        subtype: family
        for family, subtypes in families.items()
        for subtype in subtypes
    }

    # A canonical type is the strongest signal and determines its parent family.
    if type_token in type_to_family:
        derived_family = type_to_family[type_token]
        if family_token in families and family_token != derived_family:
            notes.append("family_overridden_by_canonical_type")
        if type_token != type_token_original and type_token_original:
            notes.append("type_alias_applied")
        return derived_family, type_token, raw_text or type_token, "canonical_type", notes

    # Models often put a family label in the type slot.
    if type_token in families:
        if family_token in families and family_token != type_token:
            notes.append("family_overridden_by_type_family")
        return type_token, None, raw_text or type_token, "family_only", notes

    if family_token in families:
        if token(raw_family) and family_token != token(raw_family):
            notes.append("family_alias_applied")
        if not type_token or type_token == family_token:
            level = "open_vocab" if text_value(open_text) and token(open_text) != family_token else "family_only"
            return family_token, None, raw_text or family_token, level, notes
        return family_token, None, raw_text, "open_vocab", notes

    if fallback_family is not None:
        return fallback_family, fallback_type, raw_text, "unresolved", notes
    return None, None, raw_text, "unresolved", notes


def build_prompt(
    pass_a: dict[str, Any],
    ontology: dict[str, Any],
    previous_errors: list[str] | None,
) -> str:
    correction = ""
    if previous_errors:
        correction = (
            "\nThe previous response failed structural validation. Return the complete corrected JSON and fix:\n- "
            + "\n- ".join(previous_errors)
        )

    compact = compact_ontology_for_prompt(ontology)
    process_families = [item["family"] for item in compact["primary_process_families"]]
    action_families = [item["family"] for item in compact["action_families"]]
    impact_axes = [item["family"] for item in compact["impact_transition_axes"]]
    mechanism_families = [item["family"] for item in compact["mechanism_families"]]

    return f"""
You are normalizing an observable video record into Physics Card v7 alpha-3.

The card has four dimensions only: Object, Process, Impact, and Mechanism.
The representation uses CLOSED coarse families/axes and OPTIONAL canonical subtypes.
Do not force a leaf subtype when only the coarse family or axis is reliable.

Use only the observable record and ontology below. Do not use dataset labels, filenames, or outside metadata.
Keep three levels distinct:
- Observation is a directly visible fact already grounded in Pass A.
- Physical abstraction maps visible patterns to a standard process or transition.
- Physical inference proposes a mechanism that explains observed dynamics.

Categorical confidence is allowed only for primary-process selection and mechanisms, using high, medium, low, or unknown. Do not output numeric scores, probabilities, certainty, likelihood, or confidence fields elsewhere.

STRICT RULES:
1. Primary process.family is required and must be one of: {json.dumps(process_families)}.
2. Primary process.type may be an exact canonical subtype from that family or null.
3. raw_type is required and should describe the process in a concise open-vocabulary phrase.
4. Action, Process, Impact, and Mechanism must remain distinct:
   - Action: external intervention, such as press, drop, stir, or ignite.
   - Process: physical event that occurs, such as deformation, flow, collision, or combustion.
   - Impact: object-specific observable state change.
   - Mechanism: explanatory physical driver or material response.
5. Impact transition.axis is required and must be one of: {json.dumps(impact_axes)}.
   Transition type is optional; raw_transition is required.
6. Mechanism family must be one of: {json.dumps(mechanism_families)}.
   Mechanism type is optional; raw_mechanism is required. Mechanisms may be empty.
7. Action family must be one of: {json.dumps(action_families)}. Action type is optional.
8. If uncertain about a subtype, set type=null instead of copying the family into type.
9. A camera-only video must use scope="camera_only", primary family="special", type="none", and no scene impacts.
10. Use exact object IDs and evidence IDs from Pass A.
11. Do not output object relation triples. Use event_roles such as support, container, and medium instead.
12. Choose the primary process from the core event described by Pass A, prioritizing the event summary and its supporting temporal evidence. Do not promote a brief, background, preparatory, or incidental event merely because it maps cleanly to an ontology label. Use secondary processes only for distinct co-occurring processes that materially affect the core event.
13. multi_stage means two or more qualitatively different important physical stages, not ordinary before/during/after.
14. Static is not a process family. If no scene physics occurs, use primary special/none and temporal static/not_applicable/none.
15. Separate combustion from explosive release:
    - combustion: ignition or sustained burning;
    - explosive_release: rapid stored-energy or pressure release, with or without flame.
16. Use thermal_transfer/radiative_heating only when a target visibly heats or transforms. Sunset, shadows, and ambient brightness changes belong to optical_interaction/illumination_change.
17. Primary-process selection must follow the dominant physical evolution, not raw visual salience. Prefer the process that explains the central state transition, especially an irreversible transition, and the causal-temporal organization of the clip.
18. selection_rationale must briefly compare plausible candidates when necessary and explain why the chosen primary process is central. Primary confidence expresses confidence in this selection, not confidence that the visible observations occurred.
19. Mechanism inference is allowed and encouraged when Pass A dynamics provide meaningful support. A mechanism need not be directly visible, but its evidence_refs must cite the observations from which it is inferred.
20. Include a mechanism only when it helps explain an important observed motion, deformation, or state transition. Do not list background forces or material laws merely because they normally exist; omit mechanisms with no meaningful evidence.
21. strongly_supported means the dynamics strongly favor the mechanism with few reasonable alternatives; supported_inference means it is useful and plausible but not uniquely determined; tentative means there is real but limited evidence. Do not mark every inferred mechanism tentative by default.

OBSERVABLE RECORD:
{json.dumps(pass_a, ensure_ascii=False, indent=2)}

V7 ALPHA-3 ONTOLOGY:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Return exactly one JSON object:
{{
  "schema_version": "physics_card_v7_alpha3",
  "annotation_protocol_version": "pass_b_v7_alpha3_enhanced_v1",
  "objects": [
    {{
      "id": "exact Pass A object ID",
      "name": "short object name",
      "entity_kind": "exact ontology label",
      "event_roles": [],
      "material": {{
        "phase": "exact ontology label",
        "canonical_properties": [],
        "raw_properties": [],
        "description": "free-text visible material description"
      }},
      "initial_state": {{
        "motion": "exact ontology label",
        "integrity": "exact ontology label"
      }},
      "evidence_refs": ["e1"]
    }}
  ],
  "process": {{
    "scope": "scene_physics | camera_only | mixed | unclear",
    "actions": [
      {{
        "family": "closed action family",
        "type": "canonical action subtype or null",
        "raw_action": "concise open-vocabulary action phrase",
        "actor_ids": [],
        "target_ids": [],
        "evidence_refs": ["e1"]
      }}
    ],
    "primary_process": {{
      "family": "closed primary process family",
      "type": "canonical subtype or null",
      "raw_type": "concise open-vocabulary process phrase",
      "subject_ids": [],
      "description": "specific process instance description",
      "selection_rationale": "brief reason this is the dominant physical event",
      "confidence": "high | medium | low | unknown",
      "evidence_refs": ["e1"]
    }},
    "secondary_processes": [
      {{
        "family": "closed primary process family",
        "type": "canonical subtype or null",
        "raw_type": "concise open-vocabulary process phrase",
        "subject_ids": [],
        "description": "specific secondary process",
        "evidence_refs": ["e1"]
      }}
    ],
    "temporal": {{
      "extent": "static | brief | extended",
      "structure": "single | repeated | multi_stage | not_applicable",
      "change_profile": "none | abrupt | gradual | steady | mixed"
    }}
  }},
  "impacts": [
    {{
      "object_id": "exact Pass A object ID",
      "response_description": "observable object response",
      "state_transitions": [
        {{
          "axis": "closed impact axis",
          "type": "canonical transition subtype or null",
          "raw_transition": "concise open-vocabulary state change",
          "from_state": "short free text",
          "to_state": "short free text",
          "recoverability": "recoverable | persistent | unknown | not_applicable",
          "evidence_refs": ["e1"]
        }}
      ],
      "final_state_description": "observable final state",
      "evidence_refs": ["e1"]
    }}
  ],
  "mechanisms": [
    {{
      "family": "closed mechanism family",
      "type": "canonical mechanism subtype or null",
      "raw_mechanism": "concise open-vocabulary mechanism phrase",
      "description": "brief explanation linking process to impact",
      "linked_process_families": [],
      "object_ids": [],
      "support_level": "strongly_supported | supported_inference | tentative",
      "confidence": "high | medium | low | unknown",
      "evidence_refs": ["e1"]
    }}
  ],
  "unresolved_notes": []
}}

Return JSON only.
{correction}
""".strip()


def normalize_output(
    raw: dict[str, Any],
    pass_a: dict[str, Any],
    ontology: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    repairs: list[str] = []
    notes: list[str] = []
    valid_evidence = valid_evidence_ids(pass_a)
    pass_a_objects = [
        item
        for item in safe_list(pass_a.get("objects"))
        if isinstance(item, dict) and item.get("id")
    ]
    object_ids = {str(item["id"]) for item in pass_a_objects}

    entity_kinds = axis_labels(ontology, "object", "entity_kind")
    event_roles = axis_labels(ontology, "object", "event_roles")
    material_phases = axis_labels(ontology, "object", "material_phase")
    properties = axis_labels(ontology, "object", "physical_properties")
    motion_states = axis_labels(ontology, "object", "initial_motion_state")
    integrity_states = axis_labels(ontology, "object", "initial_integrity_state")

    action_families = family_sets(ontology, ("process", "action_families"))
    process_families = family_sets(ontology, ("process", "primary_families"))
    transition_families = family_sets(ontology, ("impact", "transition_axes"))
    mechanism_families = family_sets(ontology, ("mechanism", "families"))

    scope_values = temporal_axis_labels(ontology, "process_scope")
    extent_values = temporal_axis_labels(ontology, "temporal_extent")
    structure_values = temporal_axis_labels(ontology, "temporal_structure")
    profile_values = temporal_axis_labels(ontology, "change_profile")

    raw_object_map = {
        str(item.get("id")): item
        for item in safe_list(raw.get("objects"))
        if isinstance(item, dict) and str(item.get("id")) in object_ids
    }

    objects: list[dict[str, Any]] = []
    for source in pass_a_objects:
        object_id = str(source["id"])
        item = safe_dict(raw_object_map.get(object_id))
        material = safe_dict(item.get("material"))
        initial = safe_dict(item.get("initial_state"))

        raw_property_values = safe_list(material.get("raw_properties"))
        canonical_input = safe_list(
            material.get("canonical_properties", material.get("properties"))
        )
        canonical_properties: list[str] = []
        for value in canonical_input:
            normalized = token(value)
            if normalized in properties and normalized not in canonical_properties:
                canonical_properties.append(normalized)
            elif text_value(value):
                raw_property_values.append(text_value(value))

        objects.append(
            {
                "id": object_id,
                "name": text_value(item.get("name") or source.get("name") or "unknown object"),
                "entity_kind": normalize_choice(
                    item.get("entity_kind"), entity_kinds, "unknown"
                ),
                "event_roles": normalize_multi(
                    item.get("event_roles"), event_roles, maximum=3
                ),
                "material": {
                    "phase": normalize_choice(
                        material.get("phase"), material_phases, "unknown"
                    ),
                    "canonical_properties": canonical_properties[:5],
                    "raw_properties": normalized_string_list(raw_property_values, maximum=8),
                    "description": text_value(
                        material.get("description")
                        or source.get("material_description")
                    ),
                },
                "initial_state": {
                    "motion": normalize_choice(
                        initial.get("motion"), motion_states, "unknown"
                    ),
                    "integrity": normalize_choice(
                        initial.get("integrity"), integrity_states, "unknown"
                    ),
                },
                "evidence_refs": normalize_evidence_refs(
                    item.get("evidence_refs"), valid_evidence
                ),
            }
        )

    process_raw = safe_dict(raw.get("process"))
    scope = normalize_choice(process_raw.get("scope"), scope_values, "unclear")

    actions: list[dict[str, Any]] = []
    for item in safe_list(process_raw.get("actions"))[:3]:
        if not isinstance(item, dict):
            continue
        family, subtype, raw_action, level, item_notes = resolve_hierarchical_label(
            raw_family=item.get("family"),
            raw_type=item.get("type"),
            open_text=item.get("raw_action"),
            families=action_families,
            family_aliases=alias_map(ontology, "action_family"),
            type_aliases=alias_map(ontology, "action_type"),
            fallback_family="special",
            fallback_type="other",
        )
        notes.extend(f"action:{value}" for value in item_notes)
        if not raw_action and family == "special" and subtype == "other":
            continue
        actions.append(
            {
                "family": family,
                "type": subtype,
                "raw_action": raw_action or text_value(item.get("raw_action")) or family,
                "resolution_level": level,
                "actor_ids": [
                    str(value)
                    for value in safe_list(item.get("actor_ids"))
                    if str(value) in object_ids
                ],
                "target_ids": [
                    str(value)
                    for value in safe_list(item.get("target_ids"))
                    if str(value) in object_ids
                ],
                "evidence_refs": normalize_evidence_refs(
                    item.get("evidence_refs"), valid_evidence
                ),
            }
        )

    primary_raw = safe_dict(process_raw.get("primary_process"))
    legacy_family, legacy_type = normalize_primary_legacy_fields(
        primary_raw.get("family"),
        primary_raw.get("type"),
        primary_raw.get("raw_type"),
    )
    primary_family, primary_type, raw_type, primary_level, primary_notes = (
        resolve_hierarchical_label(
            raw_family=legacy_family,
            raw_type=legacy_type,
            open_text=primary_raw.get("raw_type"),
            families=process_families,
            family_aliases=alias_map(ontology, "process_family"),
            type_aliases=alias_map(ontology, "process_type"),
            fallback_family="special",
            fallback_type="unknown",
        )
    )
    notes.extend(f"primary:{value}" for value in primary_notes)

    if scope == "camera_only":
        if primary_family != "special" or primary_type != "none":
            repairs.append("camera_only_primary_forced_to_none")
        primary_family, primary_type, primary_level = "special", "none", "canonical_type"
        raw_type = raw_type or "camera-only visual change"

    primary = {
        "family": primary_family,
        "type": primary_type,
        "raw_type": text_value(primary_raw.get("raw_type")) or raw_type or primary_family,
        "resolution_level": primary_level,
        "subject_ids": [
            str(value)
            for value in safe_list(primary_raw.get("subject_ids"))
            if str(value) in object_ids
        ],
        "description": text_value(
            primary_raw.get("description")
            or pass_a.get("raw_event_description")
        ),
        "selection_rationale": text_value(
            primary_raw.get("selection_rationale")
        ),
        "confidence": normalize_choice(
            primary_raw.get("confidence"),
            PRIMARY_CONFIDENCE_VALUES,
            "unknown",
        ),
        "evidence_refs": normalize_evidence_refs(
            primary_raw.get("evidence_refs"), valid_evidence
        ),
        "abstain": primary_level == "unresolved",
        "abstain_reason": text_value(primary_raw.get("abstain_reason")),
    }

    secondary: list[dict[str, Any]] = []
    for item in safe_list(process_raw.get("secondary_processes")):
        if not isinstance(item, dict) or len(secondary) >= 3:
            continue
        family, subtype, raw_secondary, level, item_notes = resolve_hierarchical_label(
            raw_family=item.get("family"),
            raw_type=item.get("type"),
            open_text=item.get("raw_type"),
            families=process_families,
            family_aliases=alias_map(ontology, "process_family"),
            type_aliases=alias_map(ontology, "process_type"),
        )
        notes.extend(f"secondary:{value}" for value in item_notes)
        if family is None or family == "special":
            repairs.append("unresolved_secondary_process_removed")
            continue
        if family == primary_family and subtype is not None and subtype == primary_type:
            continue
        secondary.append(
            {
                "family": family,
                "type": subtype,
                "raw_type": text_value(item.get("raw_type")) or raw_secondary or family,
                "resolution_level": level,
                "subject_ids": [
                    str(value)
                    for value in safe_list(item.get("subject_ids"))
                    if str(value) in object_ids
                ],
                "description": text_value(item.get("description")),
                "evidence_refs": normalize_evidence_refs(
                    item.get("evidence_refs"), valid_evidence
                ),
            }
        )

    temporal_raw = safe_dict(process_raw.get("temporal"))
    temporal = {
        "extent": normalize_choice(
            temporal_raw.get("extent"), extent_values, "extended"
        ),
        "structure": normalize_choice(
            temporal_raw.get("structure"), structure_values, "single"
        ),
        "change_profile": normalize_choice(
            temporal_raw.get("change_profile"), profile_values, "steady"
        ),
    }

    if scope == "camera_only" or (
        primary_family == "special" and primary_type == "none"
    ):
        temporal = {
            "extent": "static",
            "structure": "not_applicable",
            "change_profile": "none",
        }
    else:
        if temporal["extent"] == "static":
            repairs.append("scene_process_extent_forced_to_extended")
            temporal["extent"] = "extended"
        if temporal["structure"] == "not_applicable":
            repairs.append("nonstatic_structure_forced_to_single")
            temporal["structure"] = "single"
        if temporal["change_profile"] == "none":
            repairs.append("scene_process_profile_forced_to_steady")
            temporal["change_profile"] = "steady"

    impacts: list[dict[str, Any]] = []
    for item in safe_list(raw.get("impacts")):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if object_id not in object_ids:
            repairs.append("invalid_impact_object_removed")
            continue

        transitions: list[dict[str, Any]] = []
        for transition in safe_list(item.get("state_transitions")):
            if not isinstance(transition, dict):
                continue
            axis, subtype, raw_transition, level, item_notes = resolve_hierarchical_label(
                raw_family=transition.get("axis"),
                raw_type=transition.get("type"),
                open_text=transition.get("raw_transition"),
                families=transition_families,
                family_aliases=alias_map(ontology, "impact_axis"),
                type_aliases=alias_map(ontology, "impact_type"),
            )
            notes.extend(f"transition:{value}" for value in item_notes)
            if axis is None:
                repairs.append("unresolved_transition_axis_removed")
                continue
            recoverability = normalize_choice(
                transition.get("recoverability"),
                {"recoverable", "persistent", "unknown", "not_applicable"},
                "not_applicable",
            )
            transitions.append(
                {
                    "axis": axis,
                    "type": subtype,
                    "raw_transition": text_value(transition.get("raw_transition"))
                    or raw_transition
                    or axis,
                    "resolution_level": level,
                    "from_state": text_value(transition.get("from_state")),
                    "to_state": text_value(transition.get("to_state")),
                    "recoverability": recoverability,
                    "evidence_refs": normalize_evidence_refs(
                        transition.get("evidence_refs"), valid_evidence
                    ),
                }
            )

        impacts.append(
            {
                "object_id": object_id,
                "response_description": text_value(item.get("response_description")),
                "state_transitions": transitions,
                "final_state_description": text_value(
                    item.get("final_state_description")
                ),
                "evidence_refs": normalize_evidence_refs(
                    item.get("evidence_refs"), valid_evidence
                ),
            }
        )

    if scope == "camera_only":
        if impacts:
            repairs.append("camera_only_impacts_removed")
        impacts = []

    mechanisms: list[dict[str, Any]] = []
    for item in safe_list(raw.get("mechanisms")):
        if not isinstance(item, dict) or len(mechanisms) >= 4:
            continue
        family, subtype, raw_mechanism, level, item_notes = resolve_hierarchical_label(
            raw_family=item.get("family"),
            raw_type=item.get("type"),
            open_text=item.get("raw_mechanism"),
            families=mechanism_families,
            family_aliases=alias_map(ontology, "mechanism_family"),
            type_aliases=alias_map(ontology, "mechanism_type"),
        )
        notes.extend(f"mechanism:{value}" for value in item_notes)
        if family is None:
            repairs.append("unresolved_mechanism_family_removed")
            continue
        mechanisms.append(
            {
                "family": family,
                "type": subtype,
                "raw_mechanism": text_value(item.get("raw_mechanism"))
                or raw_mechanism
                or family,
                "resolution_level": level,
                "description": text_value(item.get("description")),
                "linked_process_families": normalize_multi(
                    item.get("linked_process_families"),
                    set(process_families),
                    maximum=4,
                ),
                "object_ids": [
                    str(value)
                    for value in safe_list(item.get("object_ids"))
                    if str(value) in object_ids
                ],
                "support_level": normalize_choice(
                    item.get("support_level"),
                    MECHANISM_SUPPORT_LEVELS,
                    "invalid",
                ),
                "confidence": normalize_choice(
                    item.get("confidence"),
                    MECHANISM_CONFIDENCE_VALUES,
                    "unknown",
                ),
                "evidence_refs": normalize_evidence_refs(
                    item.get("evidence_refs"), valid_evidence
                ),
            }
        )

    role_notes = derive_event_roles(objects, actions, primary, impacts)
    notes.extend(role_notes)

    card = {
        "schema_version": "physics_card_v7_alpha3",
        "annotation_protocol_version": PROTOCOL_VERSION,
        "ontology_version": ontology.get("ontology_version", "unknown"),
        "event_summary": text_value(pass_a.get("event_summary")),
        "raw_event_description": text_value(pass_a.get("raw_event_description")),
        "objects": objects,
        "process": {
            "scope": scope,
            "actions": actions,
            "primary_process": primary,
            "secondary_processes": secondary,
            "temporal": temporal,
        },
        "impacts": impacts,
        "mechanisms": mechanisms,
        "unresolved_notes": string_list(raw.get("unresolved_notes")),
        "prohibited_fields_detected": prohibited_confidence_paths(raw),
    }
    return card, repairs, notes


def validate_output(
    card: dict[str, Any],
    pass_a: dict[str, Any],
    ontology: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if card.get("schema_version") != "physics_card_v7_alpha3":
        errors.append("schema_version invalid")
    if card.get("annotation_protocol_version") != PROTOCOL_VERSION:
        errors.append("annotation_protocol_version invalid")
    if not card.get("objects"):
        errors.append("objects must be non-empty")

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
        errors.append("Pass A and Pass B object IDs differ")

    action_families = family_sets(ontology, ("process", "action_families"))
    process_families = family_sets(ontology, ("process", "primary_families"))
    transition_families = family_sets(ontology, ("impact", "transition_axes"))
    mechanism_families = family_sets(ontology, ("mechanism", "families"))
    valid_refs = valid_evidence_ids(pass_a)

    process = safe_dict(card.get("process"))
    primary = safe_dict(process.get("primary_process"))
    primary_family = token(primary.get("family"))
    primary_type = token(primary.get("type"))
    if primary_family not in process_families:
        errors.append("primary process family invalid")
    if primary_type and primary_type not in process_families.get(primary_family, set()):
        errors.append("primary process type invalid for family")
    if process.get("scope") == "camera_only" and not (
        primary_family == "special" and primary_type == "none"
    ):
        errors.append("camera_only scope requires special/none primary")
    if not primary.get("selection_rationale"):
        errors.append("primary selection_rationale empty")
    if token(primary.get("confidence")) not in PRIMARY_CONFIDENCE_VALUES:
        errors.append("primary confidence invalid")
    if valid_refs and not string_list(primary.get("evidence_refs")):
        errors.append("primary process requires evidence_refs")

    for action in safe_list(process.get("actions")):
        if not isinstance(action, dict):
            continue
        family = token(action.get("family"))
        subtype = token(action.get("type"))
        if family not in action_families:
            errors.append("action family invalid")
        elif subtype and subtype not in action_families[family]:
            errors.append("action type invalid for family")
        if valid_refs and not string_list(action.get("evidence_refs")):
            errors.append("action requires evidence_refs")

    for secondary in safe_list(process.get("secondary_processes")):
        if not isinstance(secondary, dict):
            continue
        family = token(secondary.get("family"))
        subtype = token(secondary.get("type"))
        if family not in process_families:
            errors.append("secondary process family invalid")
        elif subtype and subtype not in process_families[family]:
            errors.append("secondary process type invalid for family")
        if valid_refs and not string_list(secondary.get("evidence_refs")):
            errors.append("secondary process requires evidence_refs")

    for impact in safe_list(card.get("impacts")):
        if not isinstance(impact, dict):
            continue
        if impact.get("object_id") not in card_ids:
            errors.append("impact references unknown object")
        if valid_refs and not string_list(impact.get("evidence_refs")):
            errors.append("impact requires evidence_refs")
        for transition in safe_list(impact.get("state_transitions")):
            if not isinstance(transition, dict):
                continue
            axis = token(transition.get("axis"))
            subtype = token(transition.get("type"))
            if axis not in transition_families:
                errors.append("impact transition axis invalid")
            elif subtype and subtype not in transition_families[axis]:
                errors.append("impact transition type invalid for axis")
            if valid_refs and not string_list(transition.get("evidence_refs")):
                errors.append("impact transition requires evidence_refs")

    for item in safe_list(card.get("mechanisms")):
        if not isinstance(item, dict):
            continue
        family = token(item.get("family"))
        subtype = token(item.get("type"))
        if family not in mechanism_families:
            errors.append("mechanism family invalid")
        elif subtype and subtype not in mechanism_families[family]:
            errors.append("mechanism type invalid for family")
        linked_families = {
            token(value)
            for value in safe_list(item.get("linked_process_families"))
        }
        if not linked_families:
            errors.append("mechanism requires linked_process_families")
        elif not linked_families.issubset(process_families):
            errors.append("mechanism linked process family invalid")
        mechanism_object_ids = {
            str(value) for value in safe_list(item.get("object_ids"))
        }
        if not mechanism_object_ids:
            errors.append("mechanism requires object_ids")
        elif not mechanism_object_ids.issubset(card_ids):
            errors.append("mechanism references unknown object")
        support_level = token(item.get("support_level"))
        confidence = token(item.get("confidence"))
        if support_level not in MECHANISM_SUPPORT_LEVELS:
            errors.append("mechanism support_level invalid")
        if confidence not in MECHANISM_CONFIDENCE_VALUES:
            errors.append("mechanism confidence invalid")
        if support_level == "strongly_supported" and confidence != "high":
            errors.append("strongly_supported mechanism requires high confidence")
        if support_level == "supported_inference" and confidence not in {"high", "medium"}:
            errors.append("supported_inference mechanism requires high or medium confidence")
        if support_level == "tentative" and confidence not in {"medium", "low", "unknown"}:
            errors.append("tentative mechanism confidence invalid")
        if valid_refs and not string_list(item.get("evidence_refs")):
            errors.append("mechanism requires evidence_refs")

    if card.get("prohibited_fields_detected"):
        errors.append("unsupported confidence-like fields are forbidden")

    invalid_refs = sorted(
        {ref for ref in collect_evidence_refs(card) if ref not in valid_refs}
    )
    if invalid_refs:
        errors.append(f"invalid evidence refs: {invalid_refs}")

    return sorted(set(errors))


def derive_fields(
    card: dict[str, Any],
    pass_a: dict[str, Any],
    repairs: list[str],
) -> dict[str, Any]:
    first = safe_dict(pass_a.get("first_frame"))
    technical = safe_dict(pass_a.get("technical_quality"))
    process = safe_dict(card.get("process"))
    primary = safe_dict(process.get("primary_process"))
    scope = token(process.get("scope"))
    primary_family = token(primary.get("family"))
    primary_type = token(primary.get("type"))
    primary_level = token(primary.get("resolution_level"))

    eligible_as_query = bool(
        first.get("key_object_visibility") in {"clear", "partial"}
        and first.get("setup_clarity") in {"clear", "partial"}
        and technical.get("event_clarity") in {"clear", "partially_clear"}
        and technical.get("scene_cut") == "no"
        and scope != "camera_only"
    )
    eligible_as_demo = bool(
        technical.get("temporal_coverage") in {"complete", "partial"}
        and technical.get("event_clarity") in {"clear", "partially_clear"}
        and technical.get("entity_trackability") in {"good", "intermittent"}
        and technical.get("scene_cut") == "no"
    )

    object_core_resolved = sum(
        token(item.get("entity_kind")) != "unknown"
        for item in safe_list(card.get("objects"))
        if isinstance(item, dict)
    )
    impact_transition_count = sum(
        len(safe_list(item.get("state_transitions")))
        for item in safe_list(card.get("impacts"))
        if isinstance(item, dict)
    )
    canonical_transition_count = sum(
        token(transition.get("resolution_level")) == "canonical_type"
        for impact in safe_list(card.get("impacts"))
        if isinstance(impact, dict)
        for transition in safe_list(impact.get("state_transitions"))
        if isinstance(transition, dict)
    )

    family_resolved = bool(
        primary_family
        and primary_family != "special"
        or (primary_family == "special" and primary_type == "none")
    )

    review_reasons: list[str] = []
    if not family_resolved or primary_level == "unresolved":
        review_reasons.append("primary_family_unresolved")
    if (
        scope == "scene_physics"
        and primary_family != "special"
        and not card.get("impacts")
    ):
        review_reasons.append("scene_process_without_impact")
    if object_core_resolved == 0:
        review_reasons.append("all_object_kinds_unknown")

    structural_repairs = {
        "invalid_impact_object_removed",
        "unresolved_transition_axis_removed",
        "camera_only_impacts_removed",
    }
    if any(item in structural_repairs for item in repairs):
        review_reasons.append("structural_normalization_repairs")

    return {
        "eligible_as_query": eligible_as_query,
        "eligible_as_demo": eligible_as_demo,
        "technical_quality_score": technical_quality_score(pass_a),
        "needs_review": bool(review_reasons),
        "review_reasons": sorted(set(review_reasons)),
        "core_coverage": {
            "object_count": len(safe_list(card.get("objects"))),
            "resolved_object_kind_count": object_core_resolved,
            "action_count": len(safe_list(process.get("actions"))),
            "primary_family_resolved": family_resolved,
            "primary_canonical_type_resolved": bool(primary_type),
            "primary_resolution_level": primary_level,
            "secondary_process_count": len(
                safe_list(process.get("secondary_processes"))
            ),
            "impact_object_count": len(safe_list(card.get("impacts"))),
            "impact_transition_count": impact_transition_count,
            "canonical_transition_type_count": canonical_transition_count,
            "mechanism_count": len(safe_list(card.get("mechanisms"))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pass-a-root", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=3800)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    ontology = load_ontology(args.ontology)
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol_marker = args.output_root / "PASS_B_PROTOCOL"
    if protocol_marker.is_file():
        existing_protocol = protocol_marker.read_text(encoding="utf-8").strip()
        if existing_protocol != PROTOCOL_VERSION:
            raise RuntimeError(
                f"output root belongs to {existing_protocol!r}, expected {PROTOCOL_VERSION!r}"
            )
    elif any(args.output_root.iterdir()):
        raise RuntimeError(
            "refusing to use a non-empty output root without a PASS_B_PROTOCOL marker"
        )
    protocol_marker.write_text(PROTOCOL_VERSION + "\n", encoding="utf-8")
    output_dir = args.output_root / "cards"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = TextRunner(args.model_path)
    summary_records: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="Pass B v7 alpha3 enhanced"):
        sample_id = str(row["sample_id"])
        pass_a_path = args.pass_a_root / f"{sample_id}.json"
        output_path = output_dir / f"{sample_id}.json"
        if output_path.is_file() and not args.overwrite:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                existing_protocol = (
                    existing.get("annotation_protocol_version")
                    or safe_dict(existing.get("physics_card")).get(
                        "annotation_protocol_version"
                    )
                )
                if existing_protocol != PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"existing card {output_path} belongs to protocol "
                        f"{existing_protocol!r}, expected {PROTOCOL_VERSION!r}"
                    )
                summary_records.append(
                    {"sample_id": sample_id, "status": "skipped_existing"}
                )
                continue

        payload: dict[str, Any] = {
            "sample_id": sample_id,
            "video_path": row.get("video_path"),
            "annotation_protocol_version": PROTOCOL_VERSION,
            "status": "failed",
            "attempts": [],
        }
        try:
            if not pass_a_path.is_file():
                payload["failure_reason"] = "missing_pass_a"
            else:
                pass_a_payload = json.loads(pass_a_path.read_text(encoding="utf-8"))
                if pass_a_payload.get("status") != "success":
                    payload["failure_reason"] = "pass_a_not_successful"
                else:
                    pass_a = safe_dict(pass_a_payload.get("observable_record"))
                    previous_errors: list[str] | None = None
                    for attempt_index in range(args.retries + 1):
                        raw_response = runner.generate(
                            build_prompt(pass_a, ontology, previous_errors),
                            args.max_new_tokens,
                        )
                        parsed = parse_json_object(raw_response)
                        if parsed is None:
                            card = None
                            repairs: list[str] = []
                            notes: list[str] = []
                            errors = ["response is not JSON"]
                        else:
                            card, repairs, notes = normalize_output(
                                parsed, pass_a, ontology
                            )
                            errors = validate_output(card, pass_a, ontology)
                        payload["attempts"].append(
                            {
                                "attempt": attempt_index + 1,
                                "raw_response": raw_response,
                                "parse_ok": parsed is not None,
                                "normalization_repairs": repairs,
                                "normalization_notes": notes,
                                "validation_errors": errors,
                            }
                        )
                        if card is not None and not errors:
                            payload.update(
                                {
                                    "status": "success",
                                    "schema_valid": True,
                                    "physics_card": card,
                                    "derived": derive_fields(card, pass_a, repairs),
                                    "normalization_repairs": repairs,
                                    "normalization_notes": notes,
                                    "pass_a_source": pass_a,
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
                "annotation_protocol_version": PROTOCOL_VERSION,
                "records": summary_records,
                "counts": dict(Counter(item["status"] for item in summary_records)),
                "failure_reasons": dict(
                    Counter(
                        item["failure_reason"]
                        for item in summary_records
                        if item.get("failure_reason")
                    )
                ),
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(Counter(item["status"] for item in summary_records))


if __name__ == "__main__":
    main()
