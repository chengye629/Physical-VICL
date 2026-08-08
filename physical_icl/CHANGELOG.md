# v2 changes after handoff audit

- corrected query/demo `mode` derivation in training mappings;
- added the mapping builder as source code;
- exported training/video paths as project-relative paths only;
- made Pass A/B self-contained with a local Qwen3-VL runner instead of an external v6.3 runtime dependency;
- documented the true two-model-stage annotation logic: VLM observation -> LLM physical abstraction -> deterministic normalization/validation;
- added configurable 32B 4-GPU launcher with no hard-coded machine paths;
- froze the alpha-3 ontology as a static YAML in the normal path and moved ontology builders to legacy reference;
- added environment recording and run provenance;
- added threshold-calibration evidence and limitations;
- added explicit split-leakage policy;
- added portable server-side data bundle builder and strict validator;
- fixed checksum generation so `SHA256SUMS` does not include itself;
- copy-risk remains explicitly pending rather than being implied by `clean_v21`.

- removed the experimental metadata-only LLM pre-screen; the frozen first-stage screening is the existing Qwen3-Embedding semantic recall inside retrieval.
