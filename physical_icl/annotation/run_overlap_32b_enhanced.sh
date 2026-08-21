#!/usr/bin/env bash
# Convenience alias for enhanced Pass B: Pass A and Pass B use the same checkpoint by default.
set -Eeuo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PKG_ROOT/annotation/run_annotation_4gpu_enhanced.sh" "${1:-$PKG_ROOT/config/local.enhanced.env}"
