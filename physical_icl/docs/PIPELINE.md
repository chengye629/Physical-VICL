# Pipeline logic

The package keeps stages separate because each stage has a different epistemic role.

## 1. Pass A — observation

Purpose: record what is visibly supported in the video.

No dataset labels/captions are passed to the prompt. No hidden mechanism is inferred.

## 2. Pass B — abstraction

Purpose: map Pass A evidence into Object / Process / Impact / Mechanism using the frozen v7 alpha-3 ontology.

Pass B uses an LLM, then deterministic code normalizes and validates the generated card.

## 3. Audit

Purpose: expose schema failures, unresolved fields, inconsistent primary process choices, and technical-quality issues.

## 4. Retrieval representation and embedding recall

Each eligible Physics Card is represented in two complementary ways:

- structured ontology tokens;
- continuous semantic embeddings.

The language recall embedding is holistic and includes detailed objects, actions, process, impacts, and mechanisms.

This embedding stage is the original lightweight first-stage screen: top-K semantic recall narrows the candidate pool before O/P/I/M physics matching. No additional metadata-only or pairwise LLM screening is part of the frozen pipeline.

## 5. Retrieval matching

Object / Process / Impact / Mechanism similarities are combined into a physical score. Canonical/open-vocabulary Process status changes only the structured-vs-semantic reliability interpolation.

## 6. Confidence filtering

Candidates must pass global thresholds for language, process, and physical agreement. Thresholds are global, not family-specific.

## 7. Diversity selection

MMR selects up to K accepted demos; the selector never forces every query to have K demos.

## 8. Copy-risk annotation

Independent from retrieval. It should tag near-duplicate relations without deleting the canonical raw retrieval output.

## 9. Split-safe training construction

Split videos / near-duplicate clusters first, then construct allowed directed edges.
