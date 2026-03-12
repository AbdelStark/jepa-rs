#!/usr/bin/env bash

set -euo pipefail

default_fixture="specs/differential/ijepa_strict_tiny_fixture.json"

if [[ $# -gt 1 ]]; then
  echo "usage: scripts/run_parity_suite.sh [/path/to/ijepa-reference-fixture.json]" >&2
  exit 2
fi

fixture_path="${1:-$default_fixture}"

if [[ ! -f "$fixture_path" ]]; then
  echo "fixture not found: $fixture_path" >&2
  exit 2
fi

fixture_path="$(python3 - "$fixture_path" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
)"

echo "Running strict I-JEPA parity against fixture: $fixture_path"
JEPA_PARITY_FIXTURE="$fixture_path" cargo test \
  -p jepa-vision \
  --test integration \
  test_ijepa_strict_fixture_parity \
  -- --ignored --exact --nocapture
