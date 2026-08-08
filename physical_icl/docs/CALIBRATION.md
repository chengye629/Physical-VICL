# Retrieval threshold calibration

The current frozen thresholds are:

```text
language >= 0.84
process  >= 0.55
physical >= 0.55
```

The package includes an earlier 180-pair score-stratified **LLM-assisted qualitative review** and the final 5000-pool retrieval summary under `evidence/`. The artifact records `ASSISTANT LABEL` / `ASSISTANT REASON`; it is not a human-annotated calibration set.

The threshold history matters:

1. a stratified 180-pair LLM-assisted review suggested that a global conjunctive filter was preferable to family-specific rules;
2. the first conservative operating point used a lower language/physical floor;
3. full-pool edge-case audits showed residual false positives concentrated near the language boundary, leading to `language=0.84`;
4. a second full-pool audit showed low-physical-score false positives, leading to `physical=0.55` while keeping `process=0.55`;
5. the 5000-pool run kept these thresholds fixed and improved naturally as the candidate pool grew.

Under the current frozen thresholds (`0.84 / 0.55 / 0.55`), 8 of those 180 pairs are retained: 3 labeled GOOD, 5 ACCEPTABLE, and 0 BAD by the assistant reviewer.

The assistant model/revision used for the historical review was not recorded. Treat this artifact as heuristic threshold-development evidence, not human ground truth. No human accuracy or inter-annotator agreement should be claimed from it. A human-reviewed subset should be added before publication-quality calibration claims are made.
