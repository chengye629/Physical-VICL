# Physical-ICL Portable Handoff v2

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

## Current data status

The 5000-target run used for the current pair pool has:

- manifest targets: **5000**
- successful Physics Cards: **4861**
- failed annotations: **139**
- retrieval-eligible videos: **4151**
- directed Query -> Demo relations: **39260**

`clean_v21` means the retrieval method has been simplified and confidence-filtered. It does **not** mean visual near-duplicates have already been removed.

Copy-risk / near-duplicate labeling is still pending. Therefore the current pair pool is suitable for retrieval analysis and training-pipeline smoke tests, but should not yet be used to claim final generalization results or to construct a leakage-sensitive validation split.

## WISA data preparation and pair resolution

### Step 1 — Pilot annotation on `wisa_test100`

Before downloading and processing the full WISA-80K data pool, first read the videos in [`data/wisa_test100`](https://huggingface.co/datasets/Vincwng/Physical-ICL/tree/main/data/wisa_test100) and use them for pilot annotation.

### WISA-80K Data Download

The WISA-80K videos used in this project are downloaded from the **historical snapshot of the official Hugging Face repository `qihoo360/WISA-80K`**, since the current `main` branch no longer contains the video archives. We pin revision `8fbd4a1d1a83bdd9e1f58187d1974c3fbb3a0d37`. The original metadata is `data/wisa-80k.json`, where `video_name` is used to derive the sample ID and `captions` provides the original TI2V instruction. Our current data pool uses video shards `7, 18, 29, 43, 54, 67, 76, 80, 91, 98, 101–106`. Large ZIP files are downloaded with `aria2c` from the pinned revision, e.g.:

```bash
REV=8fbd4a1d1a83bdd9e1f58187d1974c3fbb3a0d37

# metadata
HF_ENDPOINT=https://hf-mirror.com hf download \
  qihoo360/WISA-80K data/wisa-80k.json \
  --repo-type dataset \
  --revision "$REV" \
  --local-dir hf_download/wisa80k_8fbd4a1

# example video shard
aria2c -c -x 8 -s 8 \
  -o 101.zip \
  "https://hf-mirror.com/datasets/qihoo360/WISA-80K/resolve/${REV}/data/videos/101.zip"
```

When extracting multiple shards, keep them in separate shard directories rather than merging them into one flat directory. A small number of identical filenames occur across official shards; therefore, dataset construction checks duplicate `sample_id`s by SHA256, keeps one copy when the bytes are identical, and excludes IDs whose corresponding files have different contents.

### Mapping WISA Videos to Query–Demo Pairs

All mappings use the WISA video filename as the stable identifier. For each WISA metadata entry,

```text
sample_id = Path(video_name).stem
```

For example,

```text
video_name:
1a4303595cd4c801f8ed99abeb75dd5acf58328a22950f120ec047fd3209189d.mp4

sample_id:
1a4303595cd4c801f8ed99abeb75dd5acf58328a22950f120ec047fd3209189d
```

The final pair files reference videos only by these IDs:

```text
query_id -> query video's sample_id
demo_id  -> demo video's sample_id
```

Therefore, to resolve a pair, first build a local lookup from the downloaded WISA videos:

```python
sample_id_to_path = {
    Path(video_path).stem: video_path
}
```

and then retrieve the two videos with:

```python
query_video = sample_id_to_path[row["query_id"]]
demo_video  = sample_id_to_path[row["demo_id"]]
```

The corresponding original TI2V instruction is obtained by joining the same `sample_id` with `wisa-80k.json`, where `video_name` provides the ID and `captions` provides the original caption.

In the provided processed metadata, `video_metadata.jsonl` already performs this join:

```text
sample_id
instruction          # original WISA caption
physics_card         # full Physics Card v7
```

while `train_pairs.jsonl` and `test100_1demo.jsonl` contain the Query–Demo relations:

```text
query_id
demo_id
retrieval scores / split metadata
```

Thus the common key across **WISA videos, original captions, Physics Cards, and Query–Demo mappings is `sample_id`**. No machine-specific video path is required.

Note: in the older raw retrieval file `pairs_clean_v21.jsonl`, the demo endpoint is stored under `sample_id` rather than `demo_id`. In the final exported training/test data, this has been normalized to `demo_id`.

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

### Qwen 32B protocol

For the main 32B validation experiment, use the **same Qwen3-VL 32B checkpoint for Pass A and Pass B** unless there is a deliberate ablation.

The launcher supports independent models:

```bash
PASS_A_MODEL=/path/to/model_A
PASS_B_MODEL=/path/to/model_B
```

If these are unset, both default to `QWEN_MODEL`.

Before large-scale 32B labeling, build a 500-sample overlap subset with `tools/make_overlap_manifest.py`, run the 32B pipeline, and compare against the existing 8B cards with `tools/compare_32b_annotations.py` plus manual review.

### Stage 3 — audit

`annotation/run_annotation_4gpu.sh` merges worker outputs and invokes `10_audit_v7_alpha3.py` automatically.

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

Copy:

```bash
cp config/example.env config/local.env
```

and edit the paths for the current machine.

No launcher assumes the original author's filesystem.

Activate your Python/conda environment yourself, then run:

```bash
bash annotation/run_annotation_4gpu.sh config/local.env
```

For retrieval, set `CARDS_ROOT`, `EMBED_MODEL`, and `RETRIEVAL_ROOT` in the environment/config, then run:

```bash
bash retrieval/run_retrieval.sh config/local.env
```

## 4-GPU annotation

The annotation launcher performs independent data-parallel annotation workers. With:

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

`annotation/run_annotation_4gpu.sh` also records model references, sampling settings, and code/ontology hashes in `PROVENANCE.json`.

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
