#!/usr/bin/env bash
set -Eeuo pipefail
OUT="${1:-environment_snapshot.txt}"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$(python --version 2>&1)"
  python - <<'PY'
mods=['torch','transformers','qwen_vl_utils','accelerate','sentence_transformers','numpy','pandas','pyarrow','decord','yaml']
for m in mods:
    try:
        x=__import__(m)
        print(f"{m}={getattr(x,'__version__','unknown')}")
    except Exception as e:
        print(f"{m}=MISSING ({e!r})")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  fi
} > "$OUT"
echo "[OK] wrote $OUT"
