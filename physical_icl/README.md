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
- the historical local-model implementation no longer depends on an external `scripts/annotation/v6_3/` directory; the current WISA campaign is run by the colleague's GPT-5.5 Agent;
- machine-specific `/mnt/...` paths are removed from launchers and mappings;
- the current annotation workflow uses the checked-in Pass A/Pass B protocol code as the GPT-5.5 Agent's specification;
- the training mapping builder is included and derives query/demo mode from each endpoint's own retrieval-index row.

## Annotation protocol versions — read before running

This package intentionally keeps two Pass B protocols. They share the frozen Physics Card V7 alpha-3 ontology and common card fields, but they are different annotation protocols and must not share an output directory.

| Protocol | Pass B protocol source | Output identification | Intended use |
| --- | --- | --- | --- |
| Legacy V7 alpha-3 | `annotation/09_run_pass_b_v7_alpha3.py` | no `annotation_protocol_version` field | Optional GPT-5.5 comparison only |
| Enhanced V7 alpha-3 v1 | `annotation/09_run_pass_b_v7_alpha3_enhanced.py` | `annotation_protocol_version: pass_b_v7_alpha3_enhanced_v1` | **Current GPT-5.5 annotation protocol** |

The package release is recorded in `VERSION`; the card schema remains `physics_card_v7_alpha3`; the annotation behavior is identified separately by `annotation_protocol_version`. Do not infer the annotation protocol from the schema version alone.

For current GPT-5.5 enhanced runs:

- use separate result directories for legacy and enhanced cards;
- keep `PROVENANCE.json` with the output;
- verify that every successful card contains `annotation_protocol_version: pass_b_v7_alpha3_enhanced_v1`.

Detailed field semantics and compatibility rules are in `docs/PASS_B_ENHANCED_PROTOCOL.md`.

## Current WISA annotation task

This is a new **GPT-5.5 annotation task**. It replaces the old `wisa_test100` pilot, historical 5,000-video status, existing training/Query-Demo files, and local Qwen experiments as instructions for what to annotate.

Run the task in this order:

1. annotate `data/wisa_test100_v2` with GPT-5.5;
2. download and annotate the selected official WISA ZIP shards with GPT-5.5;
3. optionally download more official ZIP shards and annotate them using the same preparation rules.

The annotation protocol is `pass_b_v7_alpha3_enhanced_v1`. The model is GPT-5.5 for both Pass A and Pass B. Record the exact GPT-5.5 model/revision exposed by the colleague's Agent in the run provenance.

### Step 1 — prepare `wisa_test100_v2`

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

### Step 2 — annotate test_v2 with GPT-5.5

The colleague's GPT-5.5 Agent should read this README and the protocol code, then process every row in `annotation_manifest.jsonl`:

1. inspect the video at `video_path`;
2. produce Pass A using the prompt and `observable_record_v7` schema in `annotation/08_run_pass_a_v7.py`;
3. produce Pass B from that Pass A using `annotation/09_run_pass_b_v7_alpha3_enhanced.py` and `annotation/physics_ontology_v7_alpha3.yaml`;
4. write one JSON file per `sample_id`.

Use:

```text
results/annotation_gpt55_wisa_test_v2/
├── pass_a/<sample_id>.json
├── pass_b/<sample_id>.json
└── PROVENANCE.json
```

The Agent must use GPT-5.5, not the local Qwen runner or a Qwen 4-GPU launcher. Captions are not annotation input. Pass A and Pass B must focus on the video's central physical event rather than incidental secondary motion.

Each enhanced Pass B result must contain:

```text
schema_version: physics_card_v7_alpha3
annotation_protocol_version: pass_b_v7_alpha3_enhanced_v1
```

The existing `results/annotation_gpt55_wisa/` directory is an older GPT-5.5 output-format example, not the input set for this new task.

### Optional — also produce legacy alpha-3 Pass B

The Agent can reuse the completed GPT-5.5 Pass A files and apply the original Pass B protocol in `annotation/09_run_pass_b_v7_alpha3.py`. The videos do not need to be inspected again.

Write this optional comparison separately:

```text
results/annotation_gpt55_wisa_test_v2/pass_b_legacy_alpha3/<sample_id>.json
```

Do not mix legacy cards with enhanced `pass_b/` cards. For the full pool, produce enhanced Pass B only unless a full legacy comparison is explicitly requested.

### Step 3 — download the initial full WISA shards

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

### Step 4 — join full-pool videos with metadata

For official WISA archives, the stable join is:

```text
sample_id = Path(metadata.video_name).stem
sample_id = Path(local_video_path).stem
instruction = first non-empty caption in metadata.captions
```

Build the annotation-ready manifest:

```bash
python tools/build_wisa_annotation_manifest.py \
  --metadata hf_download/wisa80k_8fbd4a1/data/wisa-80k.json \
  --videos-root hf_download/wisa80k_8fbd4a1/extracted \
  --output-dir data/campaign_wisa_full_initial
```

The outputs are:

| File | Purpose |
| --- | --- |
| `annotation_manifest.jsonl` | Runtime `sample_id` and absolute local `video_path` |
| `wisa_metadata.jsonl` | Portable `sample_id`, original caption, filename, and source-shard join |
| `duplicate_report.json` | Identical/conflicting filenames, missing videos, and orphan videos |
| `build_summary.json` | Counts and SHA256 hashes |

The runtime manifest deliberately excludes captions, so GPT-5.5 annotation remains based on video evidence. Duplicate IDs are checked by SHA256: identical bytes keep one path; different bytes under the same ID are excluded.

### Step 5 — annotate the full manifest with GPT-5.5

Run the same GPT-5.5 Agent procedure used for test_v2 over every full-manifest row and write:

```text
results/annotation_gpt55_wisa_full/
├── pass_a/<sample_id>.json
├── pass_b/<sample_id>.json
└── PROVENANCE.json
```

Keep the manifest, build reports, exact shard list, Hugging Face revisions, GPT-5.5 model revision, protocol version, and annotation outputs together.

### Downloading more ZIPs

More official WISA ZIPs can be annotated. Pin the same WISA revision and extract every added ZIP into its own shard directory.

If the full run has not started, add the new shard directories and rebuild the full manifest before running. If annotation has already started, do not silently change its frozen manifest; build a delta manifest from only the newly added shard directories and write its GPT-5.5 outputs to a new result directory.

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

### Current annotator

The current WISA task uses GPT-5.5 for both Pass A and Pass B. The Python files in `annotation/` define the prompts, schemas, normalization, and validation behavior; their local Qwen command-line path is historical tooling and is not the current campaign command.

### Stage 3 — audit

Use the deterministic checks in `10_audit_v7_alpha3.py` when auditing exported GPT-5.5 cards.

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

The current WISA annotation task does not require a local-model config file. The colleague's Agent uses GPT-5.5 and follows the current-task instructions above. Record the exact GPT-5.5 model revision and output paths in `PROVENANCE.json`.

`config/example.env` and the local GPU launchers are retained only for historical local-model reproducibility; do not use them for the current GPT-5.5 campaign.

For retrieval, set `CARDS_ROOT`, `EMBED_MODEL`, and `RETRIEVAL_ROOT` in the environment/config, then run:

```bash
bash retrieval/run_retrieval.sh config/local.env
```

## Historical local-model tooling

The scripts `annotation/run_annotation_4gpu.sh` and `annotation/qwen3vl_runner.py` are retained to reproduce historical local-model runs. They are not the execution path for the current GPT-5.5 task.

## Environment and provenance

`environment/requirements.txt` lists the minimal package dependencies, but historical exact library versions were not recorded in every card and are not fabricated here.

Run:

```bash
bash environment/record_environment.sh environment_snapshot.txt
```

for every new annotation run.

For the current campaign, `PROVENANCE.json` must record GPT-5.5 as the annotator, its exact available revision, the manifest SHA256, the Pass A/Pass B protocol versions, and the code/ontology hashes.

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
