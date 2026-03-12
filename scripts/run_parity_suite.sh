#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "usage: scripts/run_parity_suite.sh [fixture.json|fixture_dir]" >&2
}

resolve_path() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
}

fixture_summary() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

metadata = payload.get("metadata", {})
name = metadata.get("name", "unnamed-fixture")
abs_tolerance = metadata.get("abs_tolerance", "unknown")
rel_tolerance = metadata.get("rel_tolerance", "unknown")
print(f"{name} (abs={abs_tolerance}, rel={rel_tolerance})")
PY
}

discover_fixtures() {
  find "$1" -maxdepth 1 -type f -name '*_fixture.json' | sort
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

declare -a fixtures=()

if [[ $# -eq 0 ]]; then
  while IFS= read -r fixture; do
    fixtures+=("$fixture")
  done < <(discover_fixtures "specs/differential")
else
  input_path="$1"
  if [[ -d "$input_path" ]]; then
    while IFS= read -r fixture; do
      fixtures+=("$fixture")
    done < <(discover_fixtures "$input_path")
  elif [[ -f "$input_path" ]]; then
    fixtures+=("$input_path")
  else
    echo "fixture path not found: $input_path" >&2
    exit 2
  fi
fi

if [[ ${#fixtures[@]} -eq 0 ]]; then
  echo "no parity fixtures found" >&2
  exit 2
fi

echo "Running strict I-JEPA parity across ${#fixtures[@]} fixture(s)"

for fixture in "${fixtures[@]}"; do
  fixture_path="$(resolve_path "$fixture")"
  summary="$(fixture_summary "$fixture_path")"
  echo "==> $fixture_path"
  echo "    $summary"
  JEPA_PARITY_FIXTURE="$fixture_path" cargo test \
    -p jepa-vision \
    --test integration \
    test_ijepa_strict_fixture_parity \
    -- --ignored --exact --nocapture
done
