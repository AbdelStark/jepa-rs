#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/run_parity_suite.sh /path/to/ijepa-reference-fixture.json" >&2
  exit 2
fi

fixture_path="$1"

if [[ ! -f "$fixture_path" ]]; then
  echo "fixture not found: $fixture_path" >&2
  exit 2
fi

python3 - "$fixture_path" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
with fixture_path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

required = {
    "raw_input",
    "mask",
    "target_positions",
    "context",
    "target",
    "predicted",
    "energy",
}
missing = sorted(required.difference(payload))
if missing:
    raise SystemExit(f"fixture is missing required keys: {', '.join(missing)}")
PY

echo "Fixture schema validated: $fixture_path"
echo "Reference-backed numeric comparison is intentionally fixture-driven."
echo "Wire the exported fixture into the parity command described in docs/QUALITY_GATES.md."
