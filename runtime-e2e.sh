#!/usr/bin/env bash
# Linux VM thin wrapper over the shared runtime acceptance engine.
# Usage: ./runtime-e2e.sh [--base-url ...] [--project-id ...] [--accept-destructive ...]
set -euo pipefail
cd "$(dirname "$0")"
exec python3 runtime-e2e.py "$@"
