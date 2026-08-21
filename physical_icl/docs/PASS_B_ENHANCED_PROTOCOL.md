# Pass B enhanced annotation protocol

This protocol is implemented by:

- `annotation/09_run_pass_b_v7_alpha3_enhanced.py`

The existing `09_run_pass_b_v7_alpha3.py` remains in place and is not overwritten by this enhanced protocol.

## Compatibility

Enhanced cards retain:

- `schema_version: physics_card_v7_alpha3`;
- the existing Object, Process, Impact, and Mechanism structure;
- all existing canonical family/type labels and open-text descriptions;
- the existing ontology file without taxonomy changes.

They add `annotation_protocol_version: pass_b_v7_alpha3_enhanced_v1`. Existing consumers that ignore unknown fields can continue reading the common V7 alpha-3 fields. The enhanced validator is intended for new enhanced outputs; it does not retroactively require the new fields from historical cards.

## Unchanged fields

Objects retain identity, entity kind, event roles, material details, and initial state. Actions remain separate from physical processes. Primary and secondary processes, temporal structure, object-level impacts, state transitions, mechanisms, unresolved notes, canonical labels, raw descriptions, resolution levels, and Pass A evidence references remain in place.

Object records additionally accept `evidence_refs`, but these are not required because object identity may be established across several Pass A observations.

## Primary-process enhancement

The primary process is the single process that best explains the clip's dominant physical evolution. Selection prioritizes:

1. the central state transition, especially an irreversible one;
2. causal and temporal centrality;
3. coverage of the clip as a whole.

Raw visual size, color, or motion salience is not sufficient. Brief, background, preparatory, and incidental processes may be secondary when they materially affect the core event.

The primary process adds:

- `selection_rationale`: a short explanation of why it is primary, including comparison with plausible alternatives when useful;
- `confidence`: `high`, `medium`, `low`, or `unknown`, referring only to confidence in the primary-process choice.

## Mechanism enhancement

Mechanism inference is allowed when it explains important observed dynamics and is grounded in Pass A evidence. A force or material law is not included merely because it normally exists in the scene. Mechanisms with no meaningful evidence should be omitted rather than listed speculatively.

Each mechanism adds:

- `linked_process_families`: existing process families explained by the mechanism;
- `object_ids`: existing Pass A/Pass B objects involved in the explanation;
- `support_level`: `strongly_supported`, `supported_inference`, or `tentative`;
- `confidence`: `high`, `medium`, `low`, or `unknown`.

Support levels mean:

- `strongly_supported`: observed dynamics strongly favor the mechanism and leave few reasonable alternatives;
- `supported_inference`: the mechanism is meaningful and plausible but not uniquely determined;
- `tentative`: there is real but limited evidence.

The validator enforces consistent combinations: strongly supported pairs with high confidence; supported inference pairs with high or medium confidence; tentative pairs with medium, low, or unknown confidence.

## Evidence grounding and validation

The enhanced validator:

- preserves all existing family/type and object-ID checks;
- requires a primary selection rationale and primary evidence;
- requires evidence for emitted actions, secondary processes, impacts, transitions, and mechanisms;
- validates mechanism-to-process and mechanism-to-object links;
- permits categorical confidence only at the primary process and mechanism levels;
- continues rejecting numeric or unrelated confidence-like fields;
- treats uncertainty as valid output rather than a failure when represented with the allowed categorical values or existing null/unknown conventions.

No WISA caption, filename-derived label, retrieval result, or other external metadata is supplied to Pass B.
