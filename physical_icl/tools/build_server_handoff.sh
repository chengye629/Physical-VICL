#!/usr/bin/env bash
set -Eeuo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${1:-.}"
OUTPUT_TAR="${2:-physical_icl_handoff_5000_v7_portable.tar.gz}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
if [[ "$OUTPUT_TAR" != /* ]]; then OUTPUT_TAR="$PROJECT_ROOT/$OUTPUT_TAR"; fi

SCALE="$PROJECT_ROOT/data/wisa80k/v7/scale_5000_alpha3_4gpu"
RET="$SCALE/demo_retrieval_v2_1_clean"
RESULTS="$RET/results"
INDEX="$RET/index/retrieval_index_eligible.jsonl"
MANIFEST="$PROJECT_ROOT/data/wisa80k/v7/manifests/scale_5000_alpha3.jsonl"
RETRIEVALS="$RESULTS/retrievals_clean_v21.jsonl"
PAIRS="$RESULTS/pairs_clean_v21.jsonl"

for f in "$MANIFEST" "$INDEX" "$RETRIEVALS" "$PAIRS"; do
  [[ -s "$f" ]] || { echo "[ERROR] missing $f"; exit 1; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
OUT="$TMP/physical_icl_handoff_5000_v7_portable"
mkdir -p "$OUT"/{code,data/{manifest,raw_retrieval,training_mapping,physics_cards,annotation_failures,audit,calibration},provenance}

# Code package: portable main workflow + legacy reference, no machine-specific config.
cp -a "$PKG_ROOT" "$OUT/code/physical_icl_pipeline"
rm -f "$OUT/code/physical_icl_pipeline/config/local.env" 2>/dev/null || true

# Sanitize all paths before exporting data.
python "$PKG_ROOT/tools/sanitize_paths.py" --project-root "$PROJECT_ROOT" --input "$MANIFEST" --output "$OUT/data/manifest/scale_5000_alpha3.jsonl"
python "$PKG_ROOT/tools/sanitize_paths.py" --project-root "$PROJECT_ROOT" --input "$INDEX" --output "$OUT/data/raw_retrieval/retrieval_index_eligible.jsonl"
python "$PKG_ROOT/tools/sanitize_paths.py" --project-root "$PROJECT_ROOT" --input "$RETRIEVALS" --output "$OUT/data/raw_retrieval/retrievals_clean_v21.jsonl"
python "$PKG_ROOT/tools/sanitize_paths.py" --project-root "$PROJECT_ROOT" --input "$PAIRS" --output "$OUT/data/raw_retrieval/pairs_clean_v21.jsonl"
cp -a "$RESULTS/retrieval_summary_clean_v21.json" "$OUT/data/raw_retrieval/"
[[ -f "$RESULTS/retrieval_examples_clean_v21.txt" ]] && cp -a "$RESULTS/retrieval_examples_clean_v21.txt" "$OUT/data/raw_retrieval/"
[[ -f "$RET/index/index_summary.json" ]] && cp -a "$RET/index/index_summary.json" "$OUT/data/raw_retrieval/"

# Build fixed, portable convenience mappings. Endpoint modes come from each endpoint's index row.
python "$PKG_ROOT/training/build_training_maps.py" \
  --project-root "$PROJECT_ROOT" \
  --index "$OUT/data/raw_retrieval/retrieval_index_eligible.jsonl" \
  --retrievals "$OUT/data/raw_retrieval/retrievals_clean_v21.jsonl" \
  --pairs "$OUT/data/raw_retrieval/pairs_clean_v21.jsonl" \
  --output-dir "$OUT/data/training_mapping"

# Locate the merged 5000-card directory by count, then sanitize cards.
CARDS="$(python - <<PY
from pathlib import Path
root=Path(r'''$SCALE''')
c=[]
for d in root.rglob('cards'):
    if d.is_dir():
        n=sum(1 for _ in d.glob('*.json'))
        if n>=4800: c.append((n,str(d)))
if not c: raise SystemExit('')
print(sorted(c,reverse=True)[0][1])
PY
)"
if [[ -n "$CARDS" && -d "$CARDS" ]]; then
  echo "[INFO] cards=$CARDS"
  python "$PKG_ROOT/tools/sanitize_paths.py" --project-root "$PROJECT_ROOT" --input "$CARDS" --output "$OUT/data/physics_cards"
else
  echo "[ERROR] could not locate merged Physics Cards"
  exit 1
fi

# Compute actual annotation counts from exported card payloads.
python - <<PY
import json
from pathlib import Path
cards=Path(r'''$OUT/data/physics_cards''')
rows=[]
for p in cards.glob('*.json'):
    try: x=json.loads(p.read_text())
    except Exception: continue
    rows.append(x)
succ=sum(x.get('status')=='success' for x in rows); fail=sum(x.get('status')!='success' for x in rows)
fails=[{'sample_id':x.get('sample_id'),'failure_reason':x.get('failure_reason','unknown')} for x in rows if x.get('status')!='success']
Path(r'''$OUT/data/annotation_failures/failed_cards.jsonl''').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in fails))
summary={'manifest_rows':5000,'card_files':len(rows),'successful_cards':succ,'failed_annotations':fail,'retrieval_eligible':sum(1 for _ in open(r'''$OUT/data/raw_retrieval/retrieval_index_eligible.jsonl''')),'directed_pairs':sum(1 for _ in open(r'''$OUT/data/raw_retrieval/pairs_clean_v21.jsonl'''))}
Path(r'''$OUT/DATA_COUNTS.json''').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY

# Try to preserve Pass A payloads for failed annotations when they still exist on the source server.
python - <<PY
import json, shutil
from pathlib import Path
scale=Path(r'''$SCALE''')
out=Path(r'''$OUT/data/annotation_failures/pass_a''')
out.mkdir(parents=True, exist_ok=True)
failed=[]
for line in Path(r'''$OUT/data/annotation_failures/failed_cards.jsonl''').open():
    if line.strip(): failed.append(str(json.loads(line).get('sample_id')))
failed=set(x for x in failed if x and x!='None')
found={}
for pth in scale.rglob('*.json'):
    if pth.stem in failed and pth.parent.name=='pass_a' and pth.stem not in found:
        found[pth.stem]=pth
for sid,pth in found.items(): shutil.copy2(pth,out/f'{sid}.json')
(out.parent/'pass_a_failure_export_summary.json').write_text(json.dumps({'failed_ids':len(failed),'pass_a_payloads_found':len(found),'missing_pass_a_payloads':sorted(failed-set(found))},indent=2)+'\n')
print('[INFO] failed Pass A payloads found',len(found),'of',len(failed))
PY

# Preserve raw failure diagnostics without preserving machine-local paths.
if [[ -d $OUT/data/annotation_failures/pass_a ]]; then
  RAW_FAILURES=$TMP/pass_a_failures_raw
  mv $OUT/data/annotation_failures/pass_a $RAW_FAILURES
  python $PKG_ROOT/tools/sanitize_paths.py --project-root $PROJECT_ROOT --input $RAW_FAILURES --output $OUT/data/annotation_failures/pass_a
fi

# Collect existing audit artifacts when they really correspond to 5000 rows.
python - <<PY
import json, shutil
from pathlib import Path
root=Path(r'''$SCALE'''); out=Path(r'''$OUT/data/audit''')
chosen=None
for p in root.rglob('audit_summary.json'):
    try: x=json.loads(p.read_text())
    except Exception: continue
    if x.get('manifest_count')==5000:
        chosen=p.parent; break
if chosen:
    for p in chosen.iterdir():
        if p.is_file(): shutil.copy2(p,out/p.name)
    print('[INFO] audit=',chosen)
else:
    (out/'README.txt').write_text('No 5000-row audit artifact was found during packaging. Run annotation/10_audit_v7_alpha3.py with the original Pass A/Card roots if needed.\n')
    print('[WARN] no 5000-row audit artifact found')
PY

# Audits may include CSV path columns, so sanitize the copied directory too.
if [[ -d $OUT/data/audit ]]; then
  RAW_AUDIT=$TMP/audit_raw
  mv $OUT/data/audit $RAW_AUDIT
  python $PKG_ROOT/tools/sanitize_paths.py --project-root $PROJECT_ROOT --input $RAW_AUDIT --output $OUT/data/audit
fi

echo
echo "============================================================"
echo "[PACK 6/9] Copy calibration evidence"
echo "============================================================"

cp -a "$PKG_ROOT/evidence/threshold_calibration_report.txt" "$OUT/data/calibration/"
cp -a "$PKG_ROOT/evidence/threshold_calibration_review_labeled.txt" "$OUT/data/calibration/"
cp -a "$PKG_ROOT/evidence/retrieval_summary_scale5000.json" "$OUT/data/calibration/"

echo
echo "============================================================"
echo "[PACK 7/9] Record lightweight packaging environment"
echo "============================================================"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$(python --version 2>&1 || true)"
  echo "python_executable=$(command -v python || true)"
  echo "pip=$(python -m pip --version 2>&1 || true)"
  echo "hostname=$(hostname || true)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "[nvidia-smi]"
    nvidia-smi --query-gpu=name,driver_version,memory.total       --format=csv,noheader 2>&1 || true
  fi
} > "$OUT/provenance/packaging_environment.txt"

echo "[OK] lightweight environment recorded"
python - <<PY
import hashlib,json
from pathlib import Path
pkg=Path(r'''$PKG_ROOT'''); out=Path(r'''$OUT/provenance/PIPELINE_PROVENANCE.json''')
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
x={
 'schema':'Physics Card v7 alpha-3',
 'historical_primary_annotator':'Qwen3-VL-8B-Instruct',
 'historical_model_revision':'not recorded in card payloads',
 'current_recommended_32b_protocol':'same 32B Qwen3-VL checkpoint for Pass A and Pass B; optional Pass-A-only ablation',
 'retrieval_thresholds':{'language':0.84,'process':0.55,'physical':0.55},
 'retrieval':{'language_top_k':500,'demo_max':10,'mmr_lambda':0.85},
 'copy_risk':'pending; not included in clean_v21 semantics',
 'split_policy':'do not random-split pair rows; split video/near-duplicate clusters first',
 'code_sha256':{
  'pass_a':sha(pkg/'annotation/08_run_pass_a_v7.py'),
  'pass_b':sha(pkg/'annotation/09_run_pass_b_v7_alpha3.py'),
  'ontology':sha(pkg/'annotation/physics_ontology_v7_alpha3.yaml'),
  'retrieval':sha(pkg/'retrieval/15_retrieve_demos_v7_clean.py'),
  'mapping_builder':sha(pkg/'training/build_training_maps.py'),
 }
}
out.write_text(json.dumps(x,indent=2)+'\n')
PY

# Root README is the portable handoff README.
cp "$PKG_ROOT/README.md" "$OUT/README.md"
cp "$PKG_ROOT/docs/SPLIT_POLICY.md" "$OUT/SPLIT_POLICY.md"

echo
echo "============================================================"
echo "[PACK 8/9] Validate complete handoff payload"
echo "============================================================"

python "$PKG_ROOT/tools/validate_handoff.py"   --root "$OUT"   | tee "$OUT/VALIDATION_REPORT.json"

echo
echo "============================================================"
echo "[PACK 9/9] Checksums and tar.gz"
echo "============================================================"

# Checksums exclude the checksum file itself.
(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

rm -f "$OUTPUT_TAR" "$OUTPUT_TAR.sha256"
if command -v pigz >/dev/null 2>&1; then
  tar -I 'pigz -1' -cf "$OUTPUT_TAR" -C "$TMP" "$(basename "$OUT")"
else
  tar -czf "$OUTPUT_TAR" -C "$TMP" "$(basename "$OUT")"
fi
sha256sum "$OUTPUT_TAR" > "$OUTPUT_TAR.sha256"
echo "[DONE] $OUTPUT_TAR"
echo "[DONE] $OUTPUT_TAR.sha256"
ls -lh "$OUTPUT_TAR" "$OUTPUT_TAR.sha256"
