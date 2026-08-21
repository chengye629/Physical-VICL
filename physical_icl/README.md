# Physical-ICL Portable Handoff v2

> **Current annotation task:** follow the **Current WISA annotation task**
> section in this README. It replaces the old `wisa_test100`, 5,000-pool,
> and training-pair instructions as guidance for the new
> `wisa_test100_v2` to full-WISA campaign.

This package is the portable handoff for the frozen **Physics Card v7 / alpha-3** data pipeline and the current clean demo-retrieval pipeline.

It is intentionally organized around a few scientifically meaningful stages instead of preserving every historical experiment script.

## What was simplified

The core scientific stages are **not** merged into one monolithic program because they have different semantics and should remain independently auditable:

1. Pass A video observation;
2. Pass B physical abstraction into Physics Card v7;
3. deterministic normalization/validation + audit;
4. Physics Card indexing and **embedding semantic recall**;
5. reliability-aware physics matching + confidence filtering + MMR;
6. training mapping derivation.

The package does simplify historical implementation details:

- the frozen alpha-3 ontology is shipped directly as `annotation/physics_ontology_v7_alpha3.yaml`; ontology-builder chains are moved to `legacy_reference/` and are not needed for normal use;
- Pass A/B no longer depend on an external `scripts/annotation/v6_3/` directory; Qwen3-VL model I/O is centralized in `annotation/qwen3vl_runner.py`;
- machine-specific `/mnt/...` paths are removed from launchers and mappings;
- the main workflow uses two main entry points: 4-GPU annotation and retrieval;
- the training mapping builder is included and derives query/demo mode from each endpoint's own retrieval-index row.

## Annotation protocol versions — read before running

This package intentionally keeps two Pass B protocols. They share the frozen Physics Card V7 alpha-3 ontology and common card fields, but they are different annotation protocols and must not share an output directory.

| Protocol | Pass B script | Launcher | Output identification | Intended use |
| --- | --- | --- | --- | --- |
| Legacy V7 alpha-3 | `annotation/09_run_pass_b_v7_alpha3.py` | `annotation/run_annotation_4gpu.sh` | no `annotation_protocol_version` field | Reproduce or continue an explicitly legacy run only |
| Enhanced V7 alpha-3 v1 | `annotation/09_run_pass_b_v7_alpha3_enhanced.py` | `annotation/run_annotation_4gpu_enhanced.sh` | `annotation_protocol_version: pass_b_v7_alpha3_enhanced_v1` | **Recommended for all new annotation runs** |

The package release is recorded in `VERSION`; the card schema remains `physics_card_v7_alpha3`; the annotation behavior is identified separately by `annotation_protocol_version`. Do not infer the annotation protocol from the schema version alone.

For enhanced runs:

- use a new `ANNOTATION_ROOT`, preferably ending in `annotation_v7_alpha3_enhanced_v1`;
- never point the enhanced launcher at a legacy or unmarked annotation directory;
- keep the generated `PASS_B_PROTOCOL` and `PROVENANCE.json` with the output;
- verify that every successful card contains `annotation_protocol_version: pass_b_v7_alpha3_enhanced_v1`.

The enhanced launcher refuses to use a non-empty unmarked output directory or a directory marked with a different protocol. Detailed field semantics and compatibility rules are in `docs/PASS_B_ENHANCED_PROTOCOL.md`.

## Current WISA annotation task

This is a new annotation task. It replaces the old `wisa_test100` pilot, historical 5,000-video status, and existing training/Query-Demo files as instructions for what to annotate.

Run the task in this order:

1. annotate `data/wisa_test100_v2`;
2. download and annotate the selected official WISA ZIP shards;
3. optionally download more official ZIP shards and annotate them using the same preparation rules.

The recommended protocol for all new outputs is `pass_b_v7_alpha3_enhanced_v1`.

### Step 1 — annotate `wisa_test100_v2`

The current test set is frozen at:

| Field | Value |
| --- | --- |
| Repository | `Vincwng/Physical-ICL` |
| Revision | `64a357158a3eae3be1fbc1056904f48684e5d8cc` |
| Directory | `data/wisa_test100_v2` |

Download it from the `physical_icl` directory:

```bash
TEST_REV=64a357158a3eae3be1fbc1056904f48684e5d8cc

HF_ENDPOINT=https://hf-mirror.com hf download \
  Vincwng/Physical-ICL \
  --repo-type dataset \
  --revision "$TEST_REV" \
  --include "data/wisa_test100_v2/**" \
  --local-dir hf_download/physical_icl_test_v2
```

All 100 task videos are named `video.mp4`, so their IDs cannot be derived from the local filename. Use the provided metadata join:

```text
sample_id = metadata.jsonl["sample_id"]
video_path = repo_root / metadata.jsonl["gt_path"]
```

Build the annotation manifest:

```bash
python tools/build_wisa_test_v2_manifest.py \
  --metadata hf_download/physical_icl_test_v2/data/wisa_test100_v2/metadata.jsonl \
  --ids-file hf_download/physical_icl_test_v2/data/wisa_test100_v2/test100_v2_ids.txt \
  --repo-root hf_download/physical_icl_test_v2 \
  --output-dir data/campaign_wisa_test100_v2
```

The builder requires exactly the frozen 100 IDs and writes `annotation_manifest.jsonl`, `test_v2_metadata.jsonl`, `build_report.json`, and `build_summary.json`. Confirm that `manifest_rows=100`, `count_matches=true`, and `id_set_matches=true`.

Create a config with a new output root:

```bash
cp config/example.env config/test_v2.enhanced.env
```

```text
PROJECT_ROOT=/absolute/path/to/your/workspace
QWEN_MODEL=/absolute/path/to/Qwen3-VL-32B-Instruct
MANIFEST=/absolute/path/to/physical_icl/data/campaign_wisa_test100_v2/annotation_manifest.jsonl
ANNOTATION_ROOT=/absolute/path/to/outputs/wisa_test100_v2_enhanced_v1
GPU_LIST="0 1 2 3"
```

Run the test_v2 annotation:

```bash
bash annotation/run_annotation_4gpu_enhanced.sh config/test_v2.enhanced.env
```

### Optional — also run legacy alpha-3 on the same Pass A

The legacy alpha-3 Pass B can reuse the completed test_v2 Pass A, so the video does not need to be processed twice:

```bash
bash annotation/run_legacy_pass_b_4gpu_from_pass_a.sh \
  config/test_v2.enhanced.env \
  /absolute/path/to/outputs/wisa_test100_v2_enhanced_v1/combined/pass_a \
  /absolute/path/to/outputs/wisa_test100_v2_legacy_alpha3
```

This runs only the original `09_run_pass_b_v7_alpha3.py` and writes a separate legacy result. Legacy and enhanced cards must use separate output roots. They may be part of the same test_v2 experiment, but do not run both Pass B jobs concurrently on the same GPUs.

This comparison is optional. For the full pool, use enhanced only unless a full legacy comparison is explicitly requested, because running both protocols approximately doubles Pass B inference.

### Step 2 — download the initial full WISA shards

Use the historical official WISA snapshot:

```text
repository: qihoo360/WISA-80K
revision: 8fbd4a1d1a83bdd9e1f58187d1974c3fbb3a0d37
metadata: data/wisa-80k.json
shards: 7 18 29 43 54 67 76 80 91 98 101 102 103 104 105 106
```

Download the metadata:

```bash
WISA_REV=8fbd4a1d1a83bdd9e1f58187d1974c3fbb3a0d37

HF_ENDPOINT=https://hf-mirror.com hf download \
  qihoo360/WISA-80K data/wisa-80k.json \
  --repo-type dataset \
  --revision "$WISA_REV" \
  --local-dir hf_download/wisa80k_8fbd4a1
```

Download and extract the initial ZIPs:

```bash
WISA_REV=8fbd4a1d1a83bdd9e1f58187d1974c3fbb3a0d37
SHARDS=(7 18 29 43 54 67 76 80 91 98 101 102 103 104 105 106)
mkdir -p hf_download/wisa80k_8fbd4a1/archives
mkdir -p hf_download/wisa80k_8fbd4a1/extracted

for SHARD in "${SHARDS[@]}"; do
  aria2c -c -x 8 -s 8 \
    -d hf_download/wisa80k_8fbd4a1/archives \
    -o "${SHARD}.zip" \
    "https://hf-mirror.com/datasets/qihoo360/WISA-80K/resolve/${WISA_REV}/data/videos/${SHARD}.zip"
  mkdir -p "hf_download/wisa80k_8fbd4a1/extracted/${SHARD}"
  unzip -q "hf_download/wisa80k_8fbd4a1/archives/${SHARD}.zip" \
    -d "hf_download/wisa80k_8fbd4a1/extracted/${SHARD}"
done
```

Keep every shard in its own extracted directory. Do not flatten them: a small number of identical filenames occur across official shards.

### Step 3 — join videos with metadata and run full annotation

For official WISA archives, the stable join is:

```text
sample_id = Path(metadata.video_name).stem
sample_id = Path(local_video_path).stem
instruction = first non-empty caption in metadata.captions
```

Build the full annotation manifest:

```bash
python tools/build_wisa_annotation_manifest.py \
  --metadata hf_download/wisa80k_8fbd4a1/data/wisa-80k.json \
  --videos-root hf_download/wisa80k_8fbd4a1/extracted \
  --output-dir data/campaign_wisa_full_initial
```

The outputs are:

| File | Purpose |
| --- | --- |
| `annotation_manifest.jsonl` | Runtime `sample_id` and absolute local `video_path` used by Pass A |
| `wisa_metadata.jsonl` | Portable `sample_id`, original caption, filename, and source-shard join |
| `duplicate_report.json` | Identical/conflicting filenames, missing videos, and orphan videos |
| `build_summary.json` | Counts and SHA256 hashes |

The runtime manifest deliberately excludes captions, so annotation remains based on video evidence. Duplicate IDs are checked by SHA256: identical bytes keep one path; different bytes under the same ID are excluded.

Create a separate full-run config and output root:

```bash
cp config/example.env config/wisa_full.enhanced.env
```

```text
MANIFEST=/absolute/path/to/physical_icl/data/campaign_wisa_full_initial/annotation_manifest.jsonl
ANNOTATION_ROOT=/absolute/path/to/outputs/wisa_full_initial_enhanced_v1
```

Then run:

```bash
bash annotation/run_annotation_4gpu_enhanced.sh config/wisa_full.enhanced.env
```

Keep the manifest, build reports, exact shard list, Hugging Face revisions, `PASS_B_PROTOCOL`, `PROVENANCE.json`, merged Pass A/cards, and audit outputs together.

### Downloading more ZIPs

More official WISA ZIPs can be annotated. Pin the same WISA revision and extract every added ZIP into its own shard directory.

If the full run has not started, add the new shard directories and rebuild the full manifest before running. If annotation has already started, do not change its frozen manifest or reuse its output root; build a delta manifest from only the newly added shard directories and use a new `ANNOTATION_ROOT`.

Record the added shard numbers and new manifest SHA256.

### Relation to Query-Demo pair files

`sample_id` is the common key across WISA videos, original captions, Physics Cards, and later Query-Demo mappings:

```text
query_id = query video's sample_id
demo_id  = demo video's sample_id
```

The older raw `pairs_clean_v21.jsonl` stores the demo endpoint under `sample_id`; final exported pair files normalize it to `demo_id`. Pair files are downstream products and are not inputs to this annotation task.

## Pipeline

### Stage 1 — Pass A: observable video evidence

Pass A uses a VLM to inspect the full video and produce an `observable_record_v7`.

It records visible objects, temporal evidence, camera behavior, and directly observable state changes. It is deliberately instructed **not** to normalize into the physics ontology and not to infer hidden mechanisms.

### Stage 2 — Pass B: physical abstraction

Pass B is **not** a deterministic-only stage.

It is:

```text
Pass A observable evidence
    -> LLM physical abstraction
    -> Physics Card v7
    -> deterministic normalization
    -> deterministic validation
```

The Physics Card has four dimensions:

- Object
- Process
- Impact
- Mechanism

The frozen schema combines closed coarse families/axes with optional canonical subtypes and open-vocabulary fine descriptions.

For new runs, use the enhanced v1 protocol. It preserves the existing schema while adding an explicit primary-process selection rationale and evidence-calibrated mechanism metadata. The legacy script remains available and is not silently redirected.

### Qwen 32B protocol

For the main 32B validation experiment, use the **same Qwen3-VL 32B checkpoint for Pass A and Pass B** unless there is a deliberate ablation.

The launcher supports independent models:

```bash
PASS_A_MODEL=/path/to/model_A
PASS_B_MODEL=/path/to/model_B
```

If these are unset, both default to `QWEN_MODEL`.

Before large-scale 32B labeling, build a 500-sample overlap subset with `tools/make_overlap_manifest.py`, run `annotation/run_overlap_32b_enhanced.sh`, and compare against the existing 8B cards with `tools/compare_32b_annotations.py` plus manual review.

### Stage 3 — audit

Both launchers merge worker outputs and invoke `10_audit_v7_alpha3.py` automatically. The enhanced launcher additionally records and checks the Pass B protocol marker.

### Stage 4 — retrieval

The **existing lightweight first-stage screening** in retrieval is embedding-based semantic recall, not an additional LLM screening stage. Each detailed Physics Card is encoded with Qwen3-Embedding-8B, and the top-K semantic neighbors are recalled before structured physics matching.

The frozen retrieval flow is:

```text
Detailed Physics Card semantic recall
    -> reliability-aware O/P/I/M matching
    -> global confidence filtering
    -> MMR diversity selection
```

Current settings:

- language top-K: 500
- max demos/query: 10
- MMR lambda: 0.85
- language threshold: 0.84
- process threshold: 0.55
- physical threshold: 0.55

There is no hard process-family gate, exact-type bonus, family bonus, same-family augmentation, or exact-type augmentation.

### Stage 5 — training mappings

The canonical source files are:

1. `retrieval_index_eligible.jsonl`
2. `retrievals_clean_v21.jsonl`
3. `pairs_clean_v21.jsonl`

`training/build_training_maps.py` derives portable convenience mappings without modifying those source files.

It outputs project-relative paths only and reads endpoint mode from:

```text
retrieval_index.process_resolution_level
```

for the query and demo separately.

## Configuration

For a new enhanced annotation run, create a dedicated config:

```bash
cp config/example.env config/local.enhanced.env
```

Edit the paths for the current machine and set a fresh output root, for example:

```bash
ANNOTATION_ROOT=/path/to/output/annotation_v7_alpha3_enhanced_v1
```

No launcher assumes the original author's filesystem.

Activate your Python/conda environment yourself, then run the recommended enhanced protocol:

```bash
bash annotation/run_annotation_4gpu_enhanced.sh config/local.enhanced.env
```

The legacy command remains available only when an explicitly legacy run is required:

```bash
bash annotation/run_annotation_4gpu.sh config/local.env
```

Do not use the same `ANNOTATION_ROOT` for these two commands.

For retrieval, set `CARDS_ROOT`, `EMBED_MODEL`, and `RETRIEVAL_ROOT` in the environment/config, then run:

```bash
bash retrieval/run_retrieval.sh config/local.env
```

## 4-GPU annotation

The recommended enhanced annotation launcher performs independent data-parallel annotation workers. With:

```text
GPU_LIST="0 1 2 3"
```

it splits the manifest into four shards. Each worker runs Pass A and then Pass B. No keepalive process is used; workers exit naturally when their shard completes.

A 32B model may require substantial VRAM. The package does not assume a particular GPU type. If one 32B copy does not fit on a single visible GPU, change the deployment strategy on that machine rather than silently changing the annotation schema.

## Environment and provenance

`environment/requirements.txt` lists the minimal package dependencies, but historical exact library versions were not recorded in every card and are not fabricated here.

Run:

```bash
bash environment/record_environment.sh environment_snapshot.txt
```

for every new annotation run.

Both launchers record model references, sampling settings, and code/ontology hashes in `PROVENANCE.json`. The enhanced launcher additionally records `annotation_protocol`, the package version, and the hash of `09_run_pass_b_v7_alpha3_enhanced.py`.

## Building a full server-side handoff

The code zip does not contain the user's 100+ MB pair/card payload. On the original project server, after extracting this package, run:

```bash
bash tools/build_server_handoff.sh /path/to/Physical-ICL-Video physical_icl_handoff_5000_v7_portable.tar.gz
```

The builder:

- copies the canonical 5000 manifest and retrieval results;
- rewrites exported video paths to project-relative paths;
- rebuilds corrected training mappings;
- includes the 5000 Physics Card payloads when found;
- exports annotation-failure counts;
- includes existing 5000-row audit artifacts when available;
- includes threshold-calibration evidence;
- includes this portable code package;
- generates `SHA256SUMS` without hashing the checksum file itself.

## Important split rule

Do **not** randomly split the 39260 pair rows into train/val/test.

See `docs/SPLIT_POLICY.md`. Retrieval-threshold evidence and limitations are summarized in `docs/CALIBRATION.md`.

## Copy risk

Near-duplicate/copy-risk annotation remains an independent data-hygiene stage. Physics retrieval answers whether two events are physically analogous; copy-risk answers whether they are so visually/temporally similar that the demo leaks the answer.

Keep the raw 39260 relations unchanged and append copy-risk metadata later.

## Calibration evidence provenance

The included 180-pair qualitative threshold review is LLM-assisted: its rows
are explicitly recorded as `ASSISTANT LABEL` / `ASSISTANT REASON`. It is not a
human-annotated calibration set, and the historical assistant model/revision
was not recorded. Treat it as heuristic development evidence; do not claim
human accuracy or inter-annotator agreement from it.
