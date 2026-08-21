#!/usr/bin/env bash
# Optional comparison runner: reuse an existing Pass A and run legacy alpha-3 Pass B only.
set -Eeuo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?Usage: $0 CONFIG PASS_A_ROOT OUTPUT_ROOT}"
PASS_A_ROOT="${2:?Usage: $0 CONFIG PASS_A_ROOT OUTPUT_ROOT}"
OUTPUT_ROOT="${3:?Usage: $0 CONFIG PASS_A_ROOT OUTPUT_ROOT}"
if [[ -f "$CONFIG" ]]; then
  set -a; source "$CONFIG"; set +a
fi

: "${PROJECT_ROOT:?Set PROJECT_ROOT in the config or environment}"
: "${MANIFEST:?Set MANIFEST}"
: "${QWEN_MODEL:?Set QWEN_MODEL}"

PASS_B_MODEL="${PASS_B_MODEL:-$QWEN_MODEL}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
MAX_NEW_TOKENS="${PASS_B_MAX_NEW_TOKENS:-3800}"
RETRIES="${PASS_B_RETRIES:-0}"
PROTOCOL="legacy_v7_alpha3_pass_b_only"
MARKER="$OUTPUT_ROOT/PASS_B_PROTOCOL"

if [[ ! -d "$PASS_A_ROOT" ]]; then
  echo "[ERROR] Pass A root does not exist: $PASS_A_ROOT"
  exit 1
fi
if [[ -f "$MARKER" ]]; then
  existing_protocol="$(tr -d '[:space:]' < "$MARKER")"
  if [[ "$existing_protocol" != "$PROTOCOL" ]]; then
    echo "[ERROR] OUTPUT_ROOT belongs to '$existing_protocol', expected '$PROTOCOL'"
    exit 1
  fi
elif [[ -d "$OUTPUT_ROOT" ]] && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] refusing to use non-empty OUTPUT_ROOT without a PASS_B_PROTOCOL marker"
  exit 1
fi

cd "$PROJECT_ROOT"
read -r -a GPUS <<< "$GPU_LIST"
PARTS="${#GPUS[@]}"
[[ "$PARTS" -gt 0 ]]

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/workers"
printf '%s\n' "$PROTOCOL" > "$MARKER"
SHARD_DIR="$OUTPUT_ROOT/_manifest_shards"
rm -rf "$SHARD_DIR"
python "$PKG_ROOT/tools/split_manifest.py" \
  --input "$MANIFEST" \
  --output-dir "$SHARD_DIR" \
  --parts "$PARTS"

ONTOLOGY="$OUTPUT_ROOT/physics_ontology_v7_alpha3.yaml"
cp "$PKG_ROOT/annotation/physics_ontology_v7_alpha3.yaml" "$ONTOLOGY"

pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  part="$(printf '%02d' "$i")"
  shard="$SHARD_DIR/part_${part}.jsonl"
  worker="$OUTPUT_ROOT/workers/worker_${part}"
  log="$OUTPUT_ROOT/logs/worker_${part}.log"
  mkdir -p "$worker"
  (
    set -Eeuo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[worker $part] legacy Pass B model=$PASS_B_MODEL GPU=$gpu"
    python -u "$PKG_ROOT/annotation/09_run_pass_b_v7_alpha3.py" \
      --manifest "$shard" \
      --pass-a-root "$PASS_A_ROOT" \
      --ontology "$ONTOLOGY" \
      --model-path "$PASS_B_MODEL" \
      --output-root "$worker" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --retries "$RETRIES"
  ) > >(tee "$log") 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[ERROR] at least one legacy Pass B worker failed"
  exit 1
fi

COMBINED="$OUTPUT_ROOT/combined"
rm -rf "$COMBINED"
python "$PKG_ROOT/tools/merge_pass_b_shards.py" \
  --shards-root "$OUTPUT_ROOT/workers" \
  --output-root "$COMBINED/cards"

AUDIT="$OUTPUT_ROOT/audit"
rm -rf "$AUDIT"
python "$PKG_ROOT/annotation/10_audit_v7_alpha3.py" \
  --manifest "$MANIFEST" \
  --pass-a-root "$PASS_A_ROOT" \
  --pass-b-root "$COMBINED/cards" \
  --ontology "$ONTOLOGY" \
  --output-root "$AUDIT"

python - <<PY
import hashlib, json, platform
from pathlib import Path
pkg=Path(r'''$PKG_ROOT''')
out=Path(r'''$OUTPUT_ROOT''')
def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
provenance={
  'pipeline':'Physics Card v7 alpha-3',
  'annotation_protocol':'legacy_v7_alpha3_pass_b_only',
  'package_version':(pkg/'VERSION').read_text().strip(),
  'pass_a_root':r'''$PASS_A_ROOT''',
  'pass_b_model':r'''$PASS_B_MODEL''',
  'manifest':r'''$MANIFEST''',
  'manifest_sha256':sha(Path(r'''$MANIFEST''')),
  'gpu_list':r'''$GPU_LIST'''.split(),
  'python':platform.python_version(),
  'script_sha256':{
    'pass_b':sha(pkg/'annotation/09_run_pass_b_v7_alpha3.py'),
    'common':sha(pkg/'annotation/common_v7.py'),
    'ontology':sha(out/'physics_ontology_v7_alpha3.yaml'),
  },
}
(out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+chr(10))
print(json.dumps(provenance,indent=2))
PY

echo "[DONE] protocol=$PROTOCOL"
echo "[DONE] cards=$COMBINED/cards"
echo "[DONE] audit=$AUDIT"
