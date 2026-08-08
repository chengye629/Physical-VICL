#!/usr/bin/env bash
set -Eeuo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PKG_ROOT/config/local.env}"
if [[ -f "$CONFIG" ]]; then set -a; source "$CONFIG"; set +a; fi

if [[ -n "${PROJECT_ROOT:-}" ]]; then cd "$PROJECT_ROOT"; fi

: "${MANIFEST:?Set MANIFEST}"
: "${CARDS_ROOT:?Set CARDS_ROOT to the directory containing <sample_id>.json Physics Cards}"
: "${EMBED_MODEL:?Set EMBED_MODEL}"
: "${RETRIEVAL_ROOT:?Set RETRIEVAL_ROOT}"

GPU="${GPU:-0}"
LANGUAGE_TOP_K="${LANGUAGE_TOP_K:-500}"
DEMO_COUNT="${DEMO_COUNT:-10}"
MMR_LAMBDA="${MMR_LAMBDA:-0.85}"
TAU_LANGUAGE="${TAU_LANGUAGE:-0.84}"
TAU_PROCESS="${TAU_PROCESS:-0.55}"
TAU_PHYSICAL="${TAU_PHYSICAL:-0.55}"
BATCH_SIZE="${BATCH_SIZE:-8}"

INDEX_ROOT="$RETRIEVAL_ROOT/index"
EMB_ROOT="$RETRIEVAL_ROOT/embeddings"
RESULT_ROOT="$RETRIEVAL_ROOT/results"
mkdir -p "$INDEX_ROOT" "$EMB_ROOT" "$RESULT_ROOT"

python -u "$PKG_ROOT/retrieval/13_build_retrieval_index_v7.py" \
  --manifest "$MANIFEST" \
  --pass-b-root "$CARDS_ROOT" \
  --output-root "$INDEX_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
python -u "$PKG_ROOT/retrieval/14_encode_retrieval_v7.py" \
  --index "$INDEX_ROOT/retrieval_index_eligible.jsonl" \
  --model-path "$EMBED_MODEL" \
  --output-root "$EMB_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --device cuda

python -u "$PKG_ROOT/retrieval/15_retrieve_demos_v7_clean.py" \
  --index "$INDEX_ROOT/retrieval_index_eligible.jsonl" \
  --embedding-root "$EMB_ROOT" \
  --output-root "$RESULT_ROOT" \
  --language-top-k "$LANGUAGE_TOP_K" \
  --demo-count "$DEMO_COUNT" \
  --mmr-lambda "$MMR_LAMBDA" \
  --language-threshold "$TAU_LANGUAGE" \
  --process-threshold "$TAU_PROCESS" \
  --physical-threshold "$TAU_PHYSICAL" \
  --example-count 100

echo "[DONE] $RESULT_ROOT"
