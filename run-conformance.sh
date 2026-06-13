#!/usr/bin/env bash
# Run the shared JTF conformance suite (Python + JS) against every golden vector.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/conformance/run.py" "$@"
