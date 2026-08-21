# v2.2 annotation protocol versioning

- replaced the old `wisa_test100` / 5,000-pool task instructions with the staged `wisa_test100_v2` gate followed by pinned full-shard annotation;
- replaced the old task block in the existing `physical_icl/README.md` with the authoritative campaign runbook, including optional extra-shard and delta-campaign rules;
- added a v2-specific manifest builder that joins repeated `video.mp4` filenames through `metadata.jsonl.sample_id + gt_path` and verifies the frozen 100-ID list;
- added an optional legacy alpha-3 Pass B-only launcher that reuses an enhanced run's combined Pass A while keeping protocols and output roots separate;
- added `pass_b_v7_alpha3_enhanced_v1` as a backward-compatible Physics Card V7 alpha-3 field superset;
- kept the existing Pass B script and launcher as explicit legacy entry points;
- added separate enhanced full-run and overlap launchers;
- added a deterministic WISA metadata/video joiner that produces the runtime annotation manifest, portable caption metadata, duplicate report, and input hashes;
- added output protocol markers and refusal checks to prevent legacy/enhanced directory mixing;
- recorded package and annotation protocol versions in enhanced provenance;
- documented the exact scripts, launchers, output identifiers, and recommended commands in the README.

# v2.1 changes after handoff audit

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
