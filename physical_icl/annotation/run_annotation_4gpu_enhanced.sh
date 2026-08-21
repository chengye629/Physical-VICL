#!/usr/bin/env bash
# Annotation protocol: pass_b_v7_alpha3_enhanced_v1.
# This launcher intentionally does not replace run_annotation_4gpu.sh.
set -Eeuo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$PKG_ROOT/config/local.enhanced.env}"
if [[ -f "$CONFIG" ]]; then
  set -a; source "$CONFIG"; set +a
fi

: "${PROJECT_ROOT:?Set PROJECT_ROOT in the enhanced config or environment}"
: "${MANIFEST:?Set MANIFEST}"
: "${ANNOTATION_ROOT:?Set ANNOTATION_ROOT}"
: "${QWEN_MODEL:?Set QWEN_MODEL}"

cd "$PROJECT_ROOT"

PASS_A_MODEL="${PASS_A_MODEL:-$QWEN_MODEL}"
PASS_B_MODEL="${PASS_B_MODEL:-$QWEN_MODEL}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
FPS="${FPS:-8}"
MIN_FRAMES="${MIN_FRAMES:-32}"
MAX_FRAMES="${MAX_FRAMES:-64}"
export PHYSICL_LOCAL_FILES_ONLY="${PHYSICL_LOCAL_FILES_ONLY:-1}"

PASS_B_PROTOCOL="pass_b_v7_alpha3_enhanced_v1"
PROTOCOL_MARKER="$ANNOTATION_ROOT/PASS_B_PROTOCOL"
if [[ -f "$PROTOCOL_MARKER" ]]; then
  existing_protocol="$(tr -d '[:space:]' < "$PROTOCOL_MARKER")"
  if [[ "$existing_protocol" != "$PASS_B_PROTOCOL" ]]; then
    echo "[ERROR] ANNOTATION_ROOT belongs to protocol '$existing_protocol', expected '$PASS_B_PROTOCOL'"
    exit 1
  fi
elif [[ -d "$ANNOTATION_ROOT" ]] && [[ -n "$(find "$ANNOTATION_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] refusing to use non-empty ANNOTATION_ROOT without a PASS_B_PROTOCOL marker"
  echo "[ERROR] choose a new directory, for example annotation_v7_alpha3_enhanced_v1"
  exit 1
fi

read -r -a GPUS <<< "$GPU_LIST"
PARTS="${#GPUS[@]}"
[[ "$PARTS" -gt 0 ]]

mkdir -p "$ANNOTATION_ROOT" "$ANNOTATION_ROOT/logs"
printf '%s\n' "$PASS_B_PROTOCOL" > "$PROTOCOL_MARKER"
SHARD_DIR="$ANNOTATION_ROOT/_manifest_shards"
rm -rf "$SHARD_DIR"
python "$PKG_ROOT/tools/split_manifest.py" --input "$MANIFEST" --output-dir "$SHARD_DIR" --parts "$PARTS"

ONTOLOGY="$ANNOTATION_ROOT/physics_ontology_v7_alpha3.yaml"
cp "$PKG_ROOT/annotation/physics_ontology_v7_alpha3.yaml" "$ONTOLOGY"

pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  part=$(printf '%02d' "$i")
  shard="$SHARD_DIR/part_${part}.jsonl"
  worker="$ANNOTATION_ROOT/workers/worker_${part}"
  mkdir -p "$worker/pass_a" "$worker/pass_b"
  log="$ANNOTATION_ROOT/logs/worker_${part}.log"
  (
    set -Eeuo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[worker $part] GPU=$gpu"
    echo "[worker $part] Pass A model=$PASS_A_MODEL"
    python -u "$PKG_ROOT/annotation/08_run_pass_a_v7.py" \
      --manifest "$shard" \
      --model-path "$PASS_A_MODEL" \
      --output-root "$worker/pass_a" \
      --fps "$FPS" --min-frames "$MIN_FRAMES" --max-frames "$MAX_FRAMES"
    echo "[worker $part] Pass B model=$PASS_B_MODEL"
    python -u "$PKG_ROOT/annotation/09_run_pass_b_v7_alpha3_enhanced.py" \
      --manifest "$shard" \
      --pass-a-root "$worker/pass_a/pass_a" \
      --ontology "$ONTOLOGY" \
      --model-path "$PASS_B_MODEL" \
      --output-root "$worker/pass_b"
  ) > >(tee "$log") 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[ERROR] at least one annotation worker failed"
  exit 1
fi

COMBINED="$ANNOTATION_ROOT/combined"
rm -rf "$COMBINED"
python "$PKG_ROOT/tools/merge_annotation_shards.py" \
  --shards-root "$ANNOTATION_ROOT/workers" \
  --output-root "$COMBINED"

AUDIT="$ANNOTATION_ROOT/audit"
rm -rf "$AUDIT"
python "$PKG_ROOT/annotation/10_audit_v7_alpha3.py" \
  --manifest "$MANIFEST" \
  --pass-a-root "$COMBINED/pass_a" \
  --pass-b-root "$COMBINED/cards" \
  --ontology "$ONTOLOGY" \
  --output-root "$AUDIT"

python - <<PY
import hashlib, json, os, platform, subprocess
from pathlib import Path
pkg=Path(r'''$PKG_ROOT''')
out=Path(r'''$ANNOTATION_ROOT''')
def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
prov={
  'pipeline':'Physics Card v7 alpha-3',
  'annotation_protocol':'pass_b_v7_alpha3_enhanced_v1',
  'package_version':(pkg/'VERSION').read_text().strip(),
  'pass_a_model':r'''$PASS_A_MODEL''',
  'pass_b_model':r'''$PASS_B_MODEL''',
  'manifest':r'''$MANIFEST''',
  'manifest_sha256':sha(Path(r'''$MANIFEST''')),
  'fps':float(r'''$FPS'''), 'min_frames':int(r'''$MIN_FRAMES'''), 'max_frames':int(r'''$MAX_FRAMES'''),
  'gpu_list':r'''$GPU_LIST'''.split(),
  'python':platform.python_version(),
  'script_sha256':{
    'pass_a':sha(pkg/'annotation/08_run_pass_a_v7.py'),
    'pass_b':sha(pkg/'annotation/09_run_pass_b_v7_alpha3_enhanced.py'),
    'common':sha(pkg/'annotation/common_v7.py'),
    'ontology':sha(out/'physics_ontology_v7_alpha3.yaml'),
  },
}
(out/'PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps(prov,indent=2))
PY

echo "[DONE] annotation=$ANNOTATION_ROOT"
echo "[DONE] protocol=$PASS_B_PROTOCOL"
echo "[DONE] cards=$COMBINED/cards"
echo "[DONE] audit=$AUDIT"
