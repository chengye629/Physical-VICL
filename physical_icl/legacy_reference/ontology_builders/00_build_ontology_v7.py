#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def label(
    name: str,
    definition: str,
    *,
    include_when: list[str] | None = None,
    exclude_when: list[str] | None = None,
    confusable_with: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "label": name,
        "definition": definition,
        "include_when": include_when or [],
        "exclude_when": exclude_when or [],
        "confusable_with": confusable_with or [],
    }


def family(name: str, definition: str, types: list[dict[str, Any]]) -> dict[str, Any]:
    return {"label": name, "definition": definition, "types": types}


def axis(
    name: str,
    definition: str,
    cardinality: str,
    labels: list[dict[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "definition": definition,
        "cardinality": cardinality,
        "required": required,
        "labels": labels,
    }


def build_ontology() -> dict[str, Any]:
    entity_kind = axis(
        "entity_kind",
        "Physical form of an entity; event function belongs to event_roles instead.",
        "single",
        [
            label("agent", "Person, animal, or autonomous robot that can initiate an action."),
            label("rigid_solid", "Solid object with no visually significant deformation at the video scale."),
            label("deformable_solid", "Solid object that visibly changes shape continuously."),
            label("articulated_object", "Object composed of rigid parts connected by joints."),
            label("liquid_body", "Continuous liquid volume with flow or a free surface."),
            label("gas_or_vapor", "Continuous gaseous substance or vapor."),
            label("aerosol_or_smoke", "Visible particles suspended in a gas, including smoke, mist, and spray clouds."),
            label("granular_material", "Collection of many discrete grains or particles such as sand, powder, or snow."),
            label("surface_or_region", "Surface or spatial region mainly serving as support, boundary, or interaction area."),
            label("unknown", "Physical form cannot be determined reliably."),
        ],
        required=True,
    )

    event_roles = axis(
        "event_roles",
        "Functional roles of an entity in the annotated event. Multiple roles are allowed.",
        "multi",
        [
            label("actor", "Entity that visibly performs or initiates an intervention."),
            label("target", "Entity receiving the main action or undergoing the main state change."),
            label("tool", "Intermediate object used by an actor to apply an action."),
            label("support", "Entity or surface that supports another entity or provides an interaction surface."),
            label("container", "Entity that spatially contains or bounds another object or material."),
            label("medium", "Material or region through which another entity moves or in which the process occurs."),
            label("source", "Entity from which material, light, heat, gas, particles, or energy is released."),
            label("product", "Entity or material newly formed, separated, or emitted by the event."),
        ],
        required=True,
    )

    material_phase = axis(
        "material_phase",
        "Coarse material phase visible or strongly supported by the video.",
        "single",
        [
            label("solid", "Material retains a persistent shape at the observed scale."),
            label("liquid", "Material flows and conforms to boundaries while maintaining a free surface or volume."),
            label("gas", "Material is gaseous or vapor-like."),
            label("granular", "Material consists of many discrete grains."),
            label("mixed", "Entity contains multiple materially distinct phases that cannot be separated for this event."),
            label("not_applicable", "Material phase is not meaningful for this entity, such as a spatial region."),
            label("unknown", "Material phase cannot be determined reliably."),
        ],
        required=False,
    )

    physical_properties = axis(
        "physical_properties",
        "Behavioral or optical properties supported by visible evidence; do not infer unsupported properties.",
        "multi",
        [
            label("rigid", "Shows negligible visible deformation under the observed interaction."),
            label("flexible", "Can bend or fold visibly; recoverability is not implied."),
            label("elastic", "Visible deformation is substantially recovered after unloading."),
            label("plastically_deformable", "Visible deformation persists after loading is removed."),
            label("brittle", "Fractures with little visible plastic deformation."),
            label("viscous", "Flows slowly or resists shape change under sustained loading."),
            label("adhesive", "Forms persistent adhesion with another surface."),
            label("transparent", "Transmits light with clear visibility through the material."),
            label("translucent", "Transmits light but obscures detailed visibility."),
            label("reflective", "Produces clear or strong reflected light patterns."),
            label("emissive", "Acts as a visible source of light or thermal glow."),
            label("combustible", "Visible evidence shows that the material can burn in this event."),
        ],
        required=False,
    )

    initial_motion = axis(
        "initial_motion_state",
        "Motion state before the primary process begins.",
        "single",
        [
            label("stationary", "No discernible object motion before the event."),
            label("translating", "Object position changes without rotation being dominant."),
            label("rotating", "Angular motion is dominant."),
            label("oscillating", "Object repeatedly moves around an equilibrium state."),
            label("falling", "Object is already moving downward without support."),
            label("flowing", "Continuous material is already flowing."),
            label("dispersing", "Gas, aerosol, or particles are already spreading."),
            label("unknown", "Initial motion cannot be determined."),
        ],
        required=True,
    )

    integrity_state = axis(
        "initial_integrity_state",
        "Structural integrity before the primary process begins.",
        "single",
        [
            label("intact", "Object is whole with no visible major fracture."),
            label("deformed", "Object already has a visible shape deformation."),
            label("cracked", "Object contains visible cracks but remains largely connected."),
            label("fragmented", "Object already consists of multiple fragments."),
            label("separated", "Previously connected parts are visibly separate."),
            label("not_applicable", "Integrity is not meaningful for this entity."),
            label("unknown", "Integrity cannot be determined."),
        ],
        required=False,
    )

    relation_type = axis(
        "initial_relation_type",
        "Restricted relations that are directly relevant to the main physical event.",
        "multi",
        [
            label("contact_with", "Two entities are visibly touching."),
            label("supported_by", "One entity is held against gravity by another entity or surface."),
            label("contained_in", "Entity lies within the spatial boundary of a container."),
            label("attached_to", "Entities are persistently connected before the event."),
            label("submerged_in", "Entity is partly or fully below the surface of a liquid or granular medium."),
            label("suspended_from", "Entity hangs from another entity or connection."),
            label("above", "Entity is vertically above another relevant entity without contact."),
        ],
        required=False,
    )

    actions = [
        family("release", "Actions that remove support or impart free motion.", [
            label("release", "Remove restraint or support without a clearly intentional downward placement."),
            label("drop", "Release an object so that it begins downward free motion.", confusable_with=["release", "throw"]),
            label("throw", "Impart an initial velocity and release the object."),
        ]),
        family("transport", "Actions that move or reorient an object.", [
            label("push", "Apply force away from the actor."),
            label("pull", "Apply force toward the actor."),
            label("lift", "Move an object upward while supporting it."),
            label("lower", "Move an object downward while maintaining support."),
            label("move", "Move an object when a more specific transport action is not supported."),
            label("rotate", "Actively turn an object around an axis."),
        ]),
        family("loading", "Actions that apply mechanical loading.", [
            label("strike", "Apply a brief high-rate contact action.", confusable_with=["press"]),
            label("press", "Apply sustained predominantly one-directional loading.", confusable_with=["squeeze"]),
            label("squeeze", "Apply opposing compressive actions from two sides."),
        ]),
        family("shape_manipulation", "Actions intended to change shape.", [
            label("bend", "Apply an action that changes curvature."),
            label("twist", "Apply relative rotation to different sections."),
            label("stretch", "Apply tensile action that increases length."),
        ]),
        family("separation", "Actions intended to separate material.", [
            label("cut", "Use a sharp edge to separate material."),
            label("pierce", "Drive a narrow object through a surface."),
            label("tear", "Pull flexible material apart along a propagating tear."),
            label("shred", "Repeatedly cut or tear material into many small strips or pieces."),
        ]),
        family("assembly", "Actions that establish or remove spatial connections.", [
            label("insert", "Place one entity into another object or opening."),
            label("remove", "Take an entity out of a container, support, or assembly."),
            label("attach", "Establish a persistent connection."),
            label("detach", "Break a persistent connection."),
            label("place", "Set an entity at a target location or onto a support."),
        ]),
        family("fluid_handling", "Actions that manipulate fluids or granular material.", [
            label("pour", "Tilt or position a source to transfer material under gravity."),
            label("stir", "Move a tool through material to induce mixing or circulation."),
            label("spray", "Expel material as a dispersed stream or droplets."),
            label("dispense", "Release a controlled quantity from a source or device."),
            label("immerse", "Move an object into a liquid or granular medium."),
        ]),
        family("thermal", "Actions that add or remove thermal or radiative energy.", [
            label("heat", "Apply a visible or explicit heat source."),
            label("cool", "Apply a visible or explicit cooling source."),
            label("irradiate", "Expose an object to visible radiation as an intervention."),
        ]),
        family("ignition", "Actions that initiate combustion.", [
            label("ignite", "Apply a flame, spark, or ignition source to begin burning."),
        ]),
        family("special", "Fallback action labels.", [
            label("none", "No visible external intervention."),
            label("other", "A visible intervention exists but is outside the current ontology."),
            label("unknown", "A visible intervention may exist but cannot be classified reliably."),
        ]),
    ]

    primary_processes = [
        family("rigid_motion", "Macroscopic motion of a largely shape-preserving object.", [
            label("translation", "Object position changes while rotation is not dominant.", exclude_when=["Only camera motion changes the view."], confusable_with=["camera_only"]),
            label("rotation", "Angular motion around an axis is the dominant process."),
            label("rolling", "Translation and rotation are coupled by contact with a surface.", confusable_with=["sliding"]),
            label("sliding", "Object translates along a support while contact points slip."),
            label("free_fall", "Unsupported downward motion dominated by gravity.", exclude_when=["The object is actively lowered or supported."], confusable_with=["translation"]),
            label("projectile_motion", "Free motion after release with a visible curved or ballistic trajectory."),
            label("oscillation", "Repeated motion around an equilibrium configuration."),
            label("toppling", "Loss of stability followed by rotation around a support edge or base."),
        ]),
        family("contact_loading", "Mechanical contact and loading between entities.", [
            label("collision", "Previously separated entities undergo brief contact causing a visible response.", confusable_with=["sustained_contact", "compression_loading"]),
            label("sustained_contact", "Contact persists and is central, without a brief collision dominating."),
            label("compression_loading", "Predominant loading reduces separation along the load direction."),
            label("tension_loading", "Predominant loading pulls parts apart along a tensile direction."),
            label("shear_loading", "Tangential loading drives relative motion between neighboring regions."),
            label("penetration", "One entity enters another object or material interior."),
        ]),
        family("deformation", "Continuous visible change of object geometry.", [
            label("compression_deformation", "Object dimension decreases along a compressive direction."),
            label("bending_deformation", "Object centerline or surface curvature changes."),
            label("stretching_deformation", "Object length increases under tension."),
            label("twisting_deformation", "Different sections undergo relative angular displacement."),
            label("shear_deformation", "Material layers undergo tangential displacement."),
            label("buckling", "Compressive instability produces lateral bending or collapse."),
        ]),
        family("fracture_separation", "Loss of structural integrity or material separation.", [
            label("cracking", "Visible crack forms while the object remains largely connected."),
            label("breaking", "Object loses overall integrity and separates into a small number of parts."),
            label("shattering", "Brittle object rapidly separates into many fragments.", confusable_with=["fragmentation"]),
            label("fragmentation", "Object forms multiple fragments without requiring brittle behavior."),
            label("tearing", "Flexible material separates along a propagating tear."),
            label("shredding", "Repeated cutting or tearing creates many small strips or pieces."),
            label("cutting_separation", "A cutting action visibly separates material."),
        ]),
        family("fluid_motion", "Motion or redistribution of a continuous liquid.", [
            label("bulk_flow", "Liquid undergoes sustained macroscopic flow without a more specific dominant mode."),
            label("pouring", "Liquid transfers from a source under gravity, usually from a container edge."),
            label("dripping", "Discrete droplets form and detach intermittently."),
            label("jetting", "Directed, relatively coherent high-speed liquid stream."),
            label("spraying", "Liquid disperses into many small droplets over an angular region."),
            label("splashing", "Impact or disturbance ejects liquid outward or upward."),
            label("droplet_motion", "Motion of distinct droplets is the main process."),
            label("free_surface_motion", "Waves, ripples, foam, or surface variation without dominant transfer."),
            label("viscous_spreading", "High-viscosity liquid slowly spreads, sags, or levels."),
            label("mixing", "Distinct liquid regions or constituents visibly interpenetrate or homogenize."),
        ]),
        family("gas_particulate_motion", "Motion, emission, or redistribution of gas, aerosol, or grains.", [
            label("gas_flow", "Macroscopic gas motion without a distinct source plume dominating."),
            label("plume_motion", "An already formed gas, smoke, or aerosol plume moves or evolves."),
            label("gas_emission", "Gas, vapor, smoke, or aerosol is visibly produced from a localized source."),
            label("aerosol_dispersion", "Visible suspended particles spread through a gas."),
            label("granular_flow", "Granular material flows collectively without a more specific mode."),
            label("granular_pouring", "Granular material transfers in a gravity-driven stream."),
            label("granular_scattering", "Particles are ejected or dispersed in multiple directions."),
            label("granular_rearrangement", "Particles locally rearrange without strong outward scattering."),
            label("deposition", "Moving gas-borne or granular material settles or accumulates on a surface."),
        ]),
        family("thermal_transfer", "Visible process dominated by heating, cooling, or thermal radiation.", [
            label("heating", "Object or material visibly undergoes heating-related change without phase change dominating."),
            label("cooling", "Object or material visibly undergoes cooling-related change without phase change dominating."),
            label("thermal_radiation", "Thermal radiation or glow is the principal visible physical process."),
        ]),
        family("phase_transition", "Change between macroscopic material phases.", [
            label("melting", "Solid becomes liquid."),
            label("solidification", "Liquid becomes solid, including freezing."),
            label("vaporization", "Liquid becomes gas or vapor."),
            label("condensation", "Gas or vapor becomes liquid."),
        ]),
        family("combustion_energetic", "Combustion or rapid energetic release.", [
            label("ignition", "Transition from non-burning to burning is the primary event."),
            label("combustion", "Sustained burning or chemical reaction is the primary process."),
            label("explosion", "Rapid energy release produces abrupt expansion, debris, shock-like motion, or intense emission."),
        ]),
        family("optical_interaction", "Scene-level optical interaction rather than camera focus behavior.", [
            label("reflection", "Light redirects from a surface and produces a visible reflected pattern or image."),
            label("refraction", "Light changes direction at an interface or through an optical element."),
            label("scattering", "Light redirects from particles or irregular structures into multiple directions."),
            label("diffraction", "Wave spreading or interference occurs around apertures or structures."),
            label("absorption", "Visible attenuation is dominated by material absorption."),
            label("light_emission", "A scene entity visibly produces light."),
        ]),
        family("relation_reconfiguration", "The main event establishes, removes, or changes spatial relations.", [
            label("attachment", "A persistent connection is established."),
            label("detachment", "A persistent connection is removed."),
            label("insertion", "One entity enters a container, opening, or assembly."),
            label("removal", "One entity exits a container, opening, support, or assembly."),
            label("support_change", "Support is gained, lost, or transferred."),
            label("containment_transfer", "Material or object changes which container bounds it."),
            label("assembly", "Multiple parts are combined into a persistent structure."),
            label("rearrangement", "Relative arrangement changes without a more specific process dominating."),
        ]),
        family("static_persistent", "No substantial scene-level state change, or a persistent effect dominates.", [
            label("static_configuration", "Scene entities remain physically unchanged; camera motion alone does not count."),
            label("persistent_emission", "An emission remains present with no clear onset or major change."),
            label("persistent_optical_effect", "A stable or slowly varying optical effect dominates without another process."),
        ]),
        family("special", "Fallback primary process labels.", [
            label("none", "No scene-level physical process is present, including camera-only videos."),
            label("other", "A clear physical process exists but is not represented in the current ontology."),
            label("unknown", "A physical process exists but cannot be classified reliably."),
        ]),
    ]

    temporal_axes = [
        axis("temporal_extent", "How much of the video is occupied by the primary process.", "single", [
            label("static", "No discernible scene-level physical process."),
            label("brief", "Primary process is localized to a relatively short interval."),
            label("extended", "Primary process persists over a substantial portion of the video."),
        ], required=True),
        axis("temporal_structure", "Event organization over time.", "single", [
            label("single", "One main event or one main process chain."),
            label("repeated", "The same main process repeats or cycles multiple times."),
            label("multi_stage", "Two or more qualitatively different important stages occur in sequence."),
            label("not_applicable", "No scene-level process is present."),
        ], required=True),
        axis("change_profile", "Temporal profile of visible change.", "single", [
            label("none", "No visible scene-level state change."),
            label("abrupt", "Major change occurs within a short interval."),
            label("gradual", "State accumulates or evolves progressively."),
            label("steady", "Process continues with approximately stable character or rate."),
            label("mixed", "Distinct stages have different change profiles."),
        ], required=True),
        axis("process_scope", "Whether observed changes belong to scene physics or only to image formation/camera motion.", "single", [
            label("scene_physics", "Independent scene entities or materials undergo a physical process."),
            label("camera_only", "Only camera motion, zoom, or focus changes; no scene-level process is visible."),
            label("mixed", "Both scene physics and camera/image changes are important."),
            label("unclear", "Scope cannot be determined reliably."),
        ], required=True),
    ]

    transition_axes = [
        family("motion", "Changes in translational or rotational motion.", [
            label("motion_onset", "Entity changes from stationary to moving."),
            label("motion_arrest", "Entity changes from moving to stationary."),
            label("speed_increase", "Entity speed visibly increases."),
            label("speed_decrease", "Entity speed visibly decreases."),
            label("direction_change", "Entity motion direction changes."),
            label("rebound", "After contact, relative motion reverses from approach to separation."),
            label("rotation_onset", "Entity begins rotating."),
            label("rotation_arrest", "Entity stops rotating."),
            label("toppling", "Entity transitions from stable support to falling rotation."),
            label("settling", "Distributed material or object motion decays into a stable configuration."),
        ]),
        family("relation", "Changes in contact, support, attachment, or containment.", [
            label("contact_gain", "A new contact is established."),
            label("contact_loss", "An existing contact is removed."),
            label("support_gain", "Entity becomes supported against gravity."),
            label("support_loss", "Entity loses support."),
            label("attachment_gain", "A persistent attachment is established."),
            label("attachment_loss", "A persistent attachment is removed."),
            label("containment_gain", "Entity becomes contained in a boundary or container."),
            label("containment_loss", "Entity leaves a containing boundary or container."),
        ]),
        family("geometry", "Changes in size or shape without necessarily losing integrity.", [
            label("compressed", "Dimension decreases along a loading direction."),
            label("stretched", "Length increases along a tensile direction."),
            label("bent", "Curvature increases or changes."),
            label("twisted", "Sections rotate relative to one another."),
            label("flattened", "Geometry becomes substantially flatter."),
            label("expanded", "Overall occupied size or volume visibly increases."),
            label("contracted", "Overall occupied size or volume visibly decreases."),
        ]),
        family("integrity", "Changes in structural connectedness or damage.", [
            label("cracked", "A crack forms while the object remains largely connected."),
            label("broken", "Object separates into a small number of parts."),
            label("fragmented", "Object becomes multiple fragments."),
            label("torn", "Flexible material separates along a tear."),
            label("shredded", "Material becomes many small strips or pieces."),
            label("penetrated", "A new opening or penetration path is formed."),
        ]),
        family("phase", "Changes between material phases.", [
            label("solid_to_liquid", "Solid becomes liquid."),
            label("liquid_to_solid", "Liquid becomes solid."),
            label("liquid_to_gas", "Liquid becomes gas or vapor."),
            label("gas_to_liquid", "Gas or vapor becomes liquid."),
        ]),
        family("thermal_reaction", "Visible thermal or reaction-state changes.", [
            label("temperature_increase", "Visible evidence supports increasing temperature."),
            label("temperature_decrease", "Visible evidence supports decreasing temperature."),
            label("unburned_to_burning", "Material begins burning."),
            label("burning_to_extinguished", "Burning stops."),
            label("material_to_charred", "Material visibly chars or blackens due to reaction."),
        ]),
        family("optical_visibility", "Changes in illumination, color, visibility, or scene optical state.", [
            label("illumination_gain", "Previously unilluminated region becomes illuminated."),
            label("illumination_loss", "Illumination disappears from a region."),
            label("intensity_increase", "Visible optical intensity increases."),
            label("intensity_decrease", "Visible optical intensity decreases."),
            label("color_change", "Object or scene color visibly changes."),
            label("visibility_increase", "An entity or region becomes more visible."),
            label("visibility_decrease", "An entity or region becomes more obscured."),
        ]),
        family("emission_transport", "Onset or redistribution of emitted or transported matter or light.", [
            label("light_emission_onset", "Visible light emission begins."),
            label("smoke_emission_onset", "Smoke emission begins."),
            label("spark_emission_onset", "Spark emission begins."),
            label("particle_emission_onset", "Solid or granular particle emission begins."),
            label("liquid_emission_onset", "Liquid emission begins."),
            label("material_transfer", "Material moves from one container, region, or entity to another."),
            label("mixing", "Previously distinct materials visibly mix."),
            label("dispersion", "Material spreads over a larger region."),
            label("deposition", "Moving material settles or adheres to a surface."),
            label("accumulation", "Material amount visibly increases in a region."),
            label("free_surface_change", "Liquid free-surface geometry changes."),
        ]),
    ]

    mechanisms = [
        family("mechanical_forcing", "Mechanical drivers connecting process and impact.", [
            label("gravity", "Gravity drives unsupported or buoyancy-relative motion."),
            label("contact_impulse", "Brief contact transfers momentum and changes velocity or direction."),
            label("sustained_pressure", "Sustained distributed contact force drives deformation or flow."),
            label("tension", "Tensile force pulls material or parts apart."),
            label("shear", "Tangential force drives relative displacement between neighboring regions."),
            label("friction", "Contact resistance opposes relative tangential motion."),
            label("torque", "A moment of force drives or changes angular motion."),
            label("buoyancy", "Fluid pressure produces an upward resultant force."),
            label("drag", "Fluid resistance opposes relative motion."),
            label("pressure_gradient", "Spatial pressure differences drive fluid or gas motion."),
        ]),
        family("material_response", "Material properties that determine response.", [
            label("elasticity", "Stored deformation energy supports recovery or rebound."),
            label("plasticity", "Material yields and retains permanent deformation."),
            label("brittleness", "Material fractures with little plastic deformation."),
            label("viscosity", "Internal resistance controls fluid deformation and flow rate."),
            label("surface_tension", "Interface energy controls droplets, bubbles, and free surfaces."),
            label("adhesion", "Interfacial attraction produces persistent attachment."),
        ]),
        family("energy_reaction", "Energy transfer or release mechanisms.", [
            label("heat_transfer", "Thermal energy transfer drives temperature or phase change."),
            label("radiative_transfer", "Radiation transfers energy to or from an object."),
            label("chemical_energy_release", "Chemical reaction releases energy, as in combustion or explosion."),
            label("mass_transport", "Material transport carries species or mass between regions."),
            label("diffusion", "Random molecular or particulate transport reduces concentration gradients."),
        ]),
        family("optical_electromagnetic", "Optical or electromagnetic explanatory mechanisms.", [
            label("surface_reflection", "Light reflects from a surface according to its orientation and optical properties."),
            label("refraction_at_interface", "Light direction changes across an interface."),
            label("volumetric_scattering", "Particles or inhomogeneities scatter light through a volume."),
            label("diffraction", "Wave propagation around structures produces spreading or interference."),
            label("absorption", "Material absorbs incident light or radiation."),
            label("light_emission", "Material or a device produces visible light."),
            label("magnetic_attraction", "Magnetic interaction produces attraction or alignment."),
            label("electrostatic_interaction", "Electric charge interaction produces attraction, repulsion, or alignment."),
        ]),
    ]

    return {
        "ontology_version": "physics_ontology_v7_alpha1",
        "design": {
            "core_dimensions": ["object", "process", "impact", "mechanism"],
            "principles": [
                "One unique primary process per video, or explicit abstention.",
                "Action, primary process, impact, and mechanism must remain semantically distinct.",
                "Impact is object-specific and may contain multiple atomic state transitions.",
                "Mechanisms are optional inferences and must include a basis label.",
                "Camera-only changes are separated from scene physics.",
            ],
        },
        "object": {
            "axes": [entity_kind, event_roles, material_phase, physical_properties, initial_motion, integrity_state, relation_type],
        },
        "process": {
            "action_families": actions,
            "primary_families": primary_processes,
            "temporal_axes": temporal_axes,
            "secondary_process_uses_primary_ontology": True,
            "max_actions": 3,
            "max_secondary_processes": 3,
        },
        "impact": {
            "transition_axes": transition_axes,
            "response_description_is_free_text": True,
        },
        "mechanism": {
            "families": mechanisms,
            "basis_values": ["directly_supported", "process_outcome_inference", "default_assumption", "unknown"],
            "max_mechanisms": 4,
        },
        "boundary_examples": [
            {
                "scenario": "A person presses a sponge and it recovers.",
                "action": "press or squeeze",
                "primary_process": "compression_deformation",
                "impact": ["compressed"],
                "mechanism": ["sustained_pressure", "elasticity"],
            },
            {
                "scenario": "A ball is dropped and rebounds from the floor.",
                "action": "drop",
                "primary_process": "collision",
                "secondary_processes": ["free_fall"],
                "impact": ["contact_gain", "rebound", "contact_loss"],
                "mechanism": ["gravity", "contact_impulse", "elasticity"],
            },
            {
                "scenario": "The camera moves forward through a corridor; scene objects do not move.",
                "process_scope": "camera_only",
                "primary_process": "none",
                "impact": [],
                "mechanism": [],
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ontology = build_ontology()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(ontology, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
