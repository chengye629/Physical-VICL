#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ALPHA2_BUILDER = SCRIPT_DIR / "00_build_ontology_v7_alpha2.py"


def load_alpha2_builder():
    if not ALPHA2_BUILDER.is_file():
        raise FileNotFoundError(
            f"Missing {ALPHA2_BUILDER}. Keep this script beside 00_build_ontology_v7_alpha2.py."
        )
    spec = importlib.util.spec_from_file_location("ontology_v7_alpha2", ALPHA2_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ALPHA2_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pop_family(families: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for index, item in enumerate(families):
        if item.get("label") == name:
            return families.pop(index)
    raise KeyError(f"Family not found: {name}")


def find_family(families: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for item in families:
        if item.get("label") == name:
            return item
    raise KeyError(f"Family not found: {name}")


def remove_type(family: dict[str, Any], name: str) -> dict[str, Any] | None:
    types = family.setdefault("types", [])
    for index, item in enumerate(types):
        if item.get("label") == name:
            return types.pop(index)
    return None


def add_type(family: dict[str, Any], item: dict[str, Any]) -> None:
    types = family.setdefault("types", [])
    if not any(existing.get("label") == item.get("label") for existing in types):
        types.append(item)


def build_ontology() -> dict[str, Any]:
    alpha2 = load_alpha2_builder()
    ontology = alpha2.build_ontology()
    base = alpha2.load_alpha1_builder()

    # ------------------------------------------------------------------
    # Object: remove the unreliable explicit relation-triple axis.
    # Environment-like roles remain available through support/container/medium.
    # ------------------------------------------------------------------
    object_axes = ontology["object"]["axes"]
    ontology["object"]["axes"] = [
        axis for axis in object_axes if axis.get("name") != "initial_relation_type"
    ]

    # ------------------------------------------------------------------
    # Process: static is temporal/special, not a physical-process family.
    # Split combustion from explosive release.
    # ------------------------------------------------------------------
    families = ontology["process"]["primary_families"]
    pop_family(families, "static_persistent")
    old_combustion = pop_family(families, "combustion_energetic")

    ignition = next(
        item for item in old_combustion.get("types", []) if item.get("label") == "ignition"
    )
    sustained = base.label(
        "sustained_burning",
        "Combustion continues over an observable interval after ignition.",
        include_when=["visible flame or burning front persists"],
        exclude_when=["a brief non-combustive pressure burst", "light emission without burning"],
        confusable_with=["ignition", "explosion"],
    )
    explosion = next(
        item for item in old_combustion.get("types", []) if item.get("label") == "explosion"
    )

    insertion_index = next(
        (i for i, item in enumerate(families) if item.get("label") == "optical_interaction"),
        len(families),
    )
    families.insert(
        insertion_index,
        base.family(
            "combustion",
            "Ignition and sustained exothermic burning processes.",
            [ignition, sustained],
        ),
    )
    families.insert(
        insertion_index + 1,
        base.family(
            "explosive_release",
            "A rapid release of stored energy or pressure that produces abrupt expansion, propulsion, fragmentation, or a blast-like event; combustion is not implied.",
            [explosion],
        ),
    )

    thermal = find_family(families, "thermal_transfer")
    remove_type(thermal, "thermal_radiation")
    add_type(
        thermal,
        base.label(
            "radiative_heating",
            "Radiative energy visibly heats a target or contributes to a thermal, phase, or reaction change.",
            include_when=["a target shows a visible thermal or phase response"],
            exclude_when=[
                "sunset or ambient illumination change only",
                "visible light beams without target heating",
            ],
            confusable_with=["illumination_change", "light_emission"],
        ),
    )

    optical = find_family(families, "optical_interaction")
    add_type(
        optical,
        base.label(
            "illumination_change",
            "Visible illumination, brightness, shadow, or scene color changes without evidence of target heating.",
            include_when=["sunset, shadow motion, changing artificial illumination"],
            exclude_when=["radiation visibly heats or transforms a target"],
            confusable_with=["light_emission", "radiative_heating"],
        ),
    )

    # ------------------------------------------------------------------
    # Aliases: only high-confidence normalization. Fine-grained text stays open.
    # ------------------------------------------------------------------
    aliases = ontology.setdefault("aliases", {})
    process_family_aliases = aliases.setdefault("process_family", {})
    process_family_aliases.update(
        {
            "combustion_energetic": "combustion",
            "burning": "combustion",
            "explosive": "explosive_release",
            "explosion": "explosive_release",
            "static": "special",
            "static_persistent": "special",
        }
    )
    process_type_aliases = aliases.setdefault("process_type", {})
    process_type_aliases.update(
        {
            "combustion": "sustained_burning",
            "burning": "sustained_burning",
            "sustained_combustion": "sustained_burning",
            "thermal_radiation": "radiative_heating",
            "radiation_heating": "radiative_heating",
            "lighting_change": "illumination_change",
            "light_change": "illumination_change",
            "brightness_change": "illumination_change",
            "sunset": "illumination_change",
            "static_configuration": "none",
            "persistent_optical_effect": "illumination_change",
            "persistent_emission": "gas_emission",
        }
    )

    ontology["ontology_version"] = "physics_ontology_v7_alpha3"
    ontology["design"] = {
        "core_dimensions": ["object", "process", "impact", "mechanism"],
        "representation": "closed_families_open_subtypes",
        "principles": [
            "Object, Process, Impact, and Mechanism remain semantically distinct.",
            "Coarse process family, impact axis, and mechanism family are closed.",
            "Canonical subtype is optional and open-vocabulary text is always preserved.",
            "Static is represented by special/none plus temporal fields, not a process family.",
            "Combustion and explosive release are distinct process families.",
            "Object roles are completed deterministically from action, process, and impact references.",
            "Explicit object relation triples are excluded from the Physics Card because they were unstable.",
            "Camera-only changes are separated from scene physics by deterministic rules.",
        ],
    }
    ontology["matching_policy"]["strict_fields"] = [
        "object.entity_kind",
        "object.event_roles",
        "process.primary_process.family",
        "impact.state_transitions.axis",
        "mechanism.family",
        "process.scope",
    ]
    ontology["matching_policy"]["removed_fields"] = [
        "object.initial_state.relations",
        "mechanism.basis",
    ]

    ontology["boundary_examples"] = [
        {
            "scenario": "A soda can bursts and sprays liquid without visible burning.",
            "primary_process": "explosive_release/explosion",
            "impact": ["integrity/ruptured", "emission_transport/liquid_emission_onset"],
            "mechanism": ["mechanical_forcing/pressure_gradient"],
        },
        {
            "scenario": "Dry grass burns and the flame front continues to propagate.",
            "primary_process": "combustion/sustained_burning",
            "impact": ["thermal_reaction/unburned_to_burning", "emission_transport/smoke_emission_onset"],
            "mechanism": ["energy_reaction/chemical_energy_release"],
        },
        {
            "scenario": "The sky darkens during sunset with no visible target heating.",
            "primary_process": "optical_interaction/illumination_change",
            "impact": ["optical_visibility/intensity_decrease", "optical_visibility/color_change"],
            "mechanism": [],
        },
        {
            "scenario": "A stationary forest scene shows no object or scene-state change.",
            "primary_process": "special/none",
            "temporal": "static/not_applicable/none",
            "impact": [],
        },
    ]

    return ontology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ontology = build_ontology()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(
            ontology,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
