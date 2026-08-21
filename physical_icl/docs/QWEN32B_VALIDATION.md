# Qwen 32B annotation validation

> This document describes the historical 500-item overlap experiment. It is not
> the current annotation task. For the `wisa_test100_v2` gate and subsequent
> full WISA annotation campaign, follow the Current WISA annotation task
> section in [`README.md`](../README.md).

The goal of the 32B experiment is to improve fine-grained Physics Card accuracy while keeping the schema fixed.

## Main protocol

For all new validation runs, use annotation protocol `pass_b_v7_alpha3_enhanced_v1`. Do not reuse a legacy annotation output directory.

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

Relocate the manifest to the colleague's local video root if needed. Create `config/local.enhanced.env`, set `ANNOTATION_ROOT` to a fresh directory such as `annotation_v7_alpha3_enhanced_v1`, and run:

```bash
bash annotation/run_overlap_32b_enhanced.sh config/local.enhanced.env
```

The legacy `run_overlap_32b.sh` invokes the legacy Pass B protocol and must not be used for a new enhanced validation run.

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
