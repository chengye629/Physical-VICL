#!/usr/bin/env bash
set -Eeuo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PKG_ROOT/config/local.env}"
if [[ -f "$CONFIG" ]]; then set -a; source "$CONFIG"; set +a; fi
: "${RETRIEVAL_ROOT:?Set RETRIEVAL_ROOT}"
python -u "$PKG_ROOT/retrieval/16_build_threshold_calibration_v7.py" \
  --index "$RETRIEVAL_ROOT/index/retrieval_index_eligible.jsonl" \
  --embedding-root "$RETRIEVAL_ROOT/embeddings" \
  --output-root "$RETRIEVAL_ROOT/calibration" \
  --review-count "${CALIBRATION_COUNT:-180}" \
  --language-top-k "${LANGUAGE_TOP_K:-500}" \
  --seed "${CALIBRATION_SEED:-20260807}"
