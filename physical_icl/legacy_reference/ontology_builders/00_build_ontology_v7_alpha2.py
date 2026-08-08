#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_BUILDER = SCRIPT_DIR / "00_build_ontology_v7.py"


def load_alpha1_builder():
    if not BASE_BUILDER.is_file():
        raise FileNotFoundError(
            f"Missing {BASE_BUILDER}. Keep this script beside 00_build_ontology_v7.py."
        )
    spec = importlib.util.spec_from_file_location("ontology_v7_alpha1", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_type(
    families: list[dict[str, Any]],
    family_name: str,
    item: dict[str, Any],
) -> None:
    for family in families:
        if family.get("label") != family_name:
            continue
        types = family.setdefault("types", [])
        if not any(entry.get("label") == item.get("label") for entry in types):
            types.append(item)
        return
    raise KeyError(f"Family not found: {family_name}")


def build_ontology() -> dict[str, Any]:
    base = load_alpha1_builder()
    ontology = base.build_ontology()

    # A few high-yield additions from the 500-sample pilot. The design remains
    # intentionally small: families/axes are closed, while subtypes may remain open.
    add_type(
        ontology["process"]["primary_families"],
        "rigid_motion",
        base.label(
            "bouncing",
            "Repeated or dominant contact-and-rebound motion of an object, such as a dribbled ball.",
            confusable_with=["oscillation", "collision"],
        ),
    )
    add_type(
        ontology["process"]["primary_families"],
        "fracture_separation",
        base.label(
            "rupture",
            "A membrane, shell, soft enclosure, or pressurized object loses integrity and opens or bursts, often releasing contents.",
            confusable_with=["breaking", "fragmentation", "explosion"],
        ),
    )
    add_type(
        ontology["process"]["primary_families"],
        "phase_transition",
        base.label(
            "boiling",
            "Vapor bubbles form within a liquid and rise or escape, indicating liquid-to-gas transition.",
            confusable_with=["vaporization", "gas_emission"],
        ),
    )
    add_type(
        ontology["process"]["primary_families"],
        "phase_transition",
        base.label(
            "crystallization",
            "An ordered solid or crystalline structure visibly forms or grows from another phase or solution.",
            confusable_with=["solidification", "deposition"],
        ),
    )
    add_type(
        ontology["impact"]["transition_axes"],
        "integrity",
        base.label(
            "ruptured",
            "A shell, membrane, soft enclosure, or pressurized object opens or bursts and loses containment.",
            confusable_with=["broken", "fragmented"],
        ),
    )
    add_type(
        ontology["impact"]["transition_axes"],
        "phase",
        base.label(
            "crystallized",
            "A visibly ordered solid or crystalline structure forms or grows.",
            confusable_with=["liquid_to_solid"],
        ),
    )
    add_type(
        ontology["mechanism"]["families"],
        "mechanical_forcing",
        base.label(
            "airflow_forcing",
            "Moving air exerts distributed aerodynamic force that drives or perturbs visible motion.",
            confusable_with=["drag", "pressure_gradient"],
        ),
    )
    add_type(
        ontology["mechanism"]["families"],
        "optical_electromagnetic",
        base.label(
            "light_transmission",
            "Light passes through a material or opening without reflection, refraction, or scattering being the sole dominant explanation.",
            confusable_with=["refraction_at_interface", "volumetric_scattering", "absorption"],
        ),
    )

    ontology["ontology_version"] = "physics_ontology_v7_alpha2"
    ontology["design"] = {
        "core_dimensions": ["object", "process", "impact", "mechanism"],
        "representation": "closed_families_open_subtypes",
        "principles": [
            "Object, Process, Impact, and Mechanism remain semantically distinct.",
            "Process family, Impact axis, and Mechanism family are closed and must be selected explicitly.",
            "Canonical subtype is optional; open-vocabulary subtype text is always preserved.",
            "One primary process family is required; a leaf type is not required.",
            "Impact is object-specific and may contain multiple transition axes.",
            "Mechanism is optional and is treated as an explanatory inference.",
            "Camera-only changes are separated from scene physics by deterministic rules.",
        ],
    }

    ontology["matching_policy"] = {
        "strict_fields": [
            "object.entity_kind",
            "process.primary_process.family",
            "impact.state_transitions.axis",
            "mechanism.family",
            "process.scope",
        ],
        "optional_canonical_fields": [
            "process.primary_process.type",
            "process.secondary_processes.type",
            "impact.state_transitions.type",
            "mechanism.type",
        ],
        "open_text_fields": [
            "process.primary_process.raw_type",
            "process.primary_process.description",
            "impact.state_transitions.raw_transition",
            "impact.response_description",
            "mechanism.raw_mechanism",
            "mechanism.description",
        ],
    }

    # These are deliberately compact, high-precision mappings rather than an
    # attempt to enumerate every phrase that a model may generate.
    ontology["aliases"] = {
        "process_family": {
            "rigid_body_motion": "rigid_motion",
            "contact_interaction": "contact_loading",
            "fracture": "fracture_separation",
            "fluid_dynamics": "fluid_motion",
            "gas_motion": "gas_particulate_motion",
            "particulate_motion": "gas_particulate_motion",
            "thermal": "thermal_transfer",
            "phase_change": "phase_transition",
            "combustion": "combustion_energetic",
            "optical": "optical_interaction",
            "relation_change": "relation_reconfiguration",
            "static": "static_persistent",
        },
        "process_type": {
            "freefall": "free_fall",
            "bounce": "bouncing",
            "bouncing_motion": "bouncing",
            "dribbling": "bouncing",
            "burst": "rupture",
            "bursting": "rupture",
            "popping": "rupture",
            "rupturing": "rupture",
            "stirring": "mixing",
            "freeze": "solidification",
            "freezing": "solidification",
            "boil": "boiling",
            "boiling_water": "boiling",
            "crystal_growth": "crystallization",
            "frost_formation": "crystallization",
            "burning": "combustion",
            "explode": "explosion",
        },
        "action_family": {
            "manipulation": "transport",
            "fluid_manipulation": "fluid_handling",
            "heat_input": "thermal",
        },
        "action_type": {
            "rotate_object": "rotate",
            "move_object": "move",
            "compress": "press",
            "puncture": "pierce",
            "mix": "stir",
        },
        "impact_axis": {
            "contact": "relation",
            "spatial_relation": "relation",
            "shape": "geometry",
            "structural_integrity": "integrity",
            "optical": "optical_visibility",
            "visibility": "optical_visibility",
            "emission": "emission_transport",
            "transport": "emission_transport",
        },
        "impact_type": {
            "fragmentation": "fragmented",
            "rupture": "ruptured",
            "burst": "ruptured",
            "bursting": "ruptured",
            "expansion": "expanded",
            "compression": "compressed",
            "bending": "bent",
            "stretching": "stretched",
            "free_surface_motion": "free_surface_change",
            "attachment": "attachment_gain",
            "detachment": "attachment_loss",
            "crystallization": "crystallized",
        },
        "mechanism_family": {
            "mechanical": "mechanical_forcing",
            "material": "material_response",
            "energy_transfer": "energy_reaction",
            "optical_interaction": "optical_electromagnetic",
            "electromagnetic": "optical_electromagnetic",
        },
        "mechanism_type": {
            "reflection": "surface_reflection",
            "refraction": "refraction_at_interface",
            "thermal_radiation": "radiative_transfer",
            "air_resistance": "drag",
            "wind": "airflow_forcing",
            "air_currents": "airflow_forcing",
            "atmospheric_scattering": "volumetric_scattering",
            "transmission": "light_transmission",
        },
    }

    ontology["process"]["open_subtype_allowed"] = True
    ontology["impact"]["open_subtype_allowed"] = True
    ontology["mechanism"]["open_subtype_allowed"] = True
    ontology["mechanism"]["basis_values"] = []
    ontology["mechanism"]["basis_deprecated"] = True

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
    print(args.output)


if __name__ == "__main__":
    main()
