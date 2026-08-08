# Qwen 32B annotation validation

The goal of the 32B experiment is to improve fine-grained Physics Card accuracy while keeping the schema fixed.

## Main protocol

Use the same 32B Qwen3-VL checkpoint in both stages:

```text
video -> 32B Pass A observation -> 32B Pass B physical abstraction -> deterministic normalization/validation
```

## Suggested first run

Build a 500-video overlap subset from the current eligible index:

```bash
python tools/make_overlap_manifest.py \
  --manifest data/manifest/scale_5000_alpha3.jsonl \
  --eligible-index data/raw_retrieval/retrieval_index_eligible.jsonl \
  --output work/overlap_500.jsonl \
  --count 500
```

Relocate the manifest to the colleague's local video root if needed, then run the 4-GPU annotation launcher.

Compare 32B cards with existing 8B reference cards using:

```bash
python tools/compare_32b_annotations.py \
  --reference-cards /path/to/8b/cards \
  --new-cards /path/to/32b/cards \
  --output work/compare_8b_32b.json
```

Structural agreement metrics are diagnostics only. Because the 32B run is intended to correct 8B mistakes, manual review of disagreements is required before declaring one annotator better.

## Optional ablation

A smaller ablation can use 32B only for Pass A and the old/smaller model for Pass B. This isolates whether improvements come mostly from video observation or from physical abstraction.
