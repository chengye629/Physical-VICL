# Data preparation

Convert `physiq_prelim/summary.json` into one JSONL manifest per experiment condition:

```bash
python scripts/prepare/build_manifests.py \
  --dataset-root /data/physiq_prelim \
  --output-root manifests/physiq_prelim
```

The default matrix is `no_demo`, `good_follow`, `good_rule`, `weak_typed`,
`opposite_typed`, `irrelevant_follow`, and `bad_typed`. Missing demos or prompts are
skipped; paths are checked before an item is emitted. Use `--allow-missing` only when
building manifests on a machine that does not hold the media.
