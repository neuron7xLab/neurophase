#!/usr/bin/env bash
# scripts/run_layer_d_acceptance.sh
#
# One-command hardware acceptance runner for NeuroPhase Layer-D.
# This script does NOT run in CI. It is the operator-side formal
# gate that decides whether a given hardware session (real Polar H10
# + real BT adapter + real chest strap contact) passes the Layer-D
# contract defined in docs/LIVE_SESSION_PROTOCOL.md.
#
# Output is deterministic: exit 0 = PASS, any non-zero = FAIL.
# Every step's stdout/stderr is teed to a dated artifacts directory
# so a reviewer can reconstruct the run without re-running.
#
# Usage:
#     scripts/run_layer_d_acceptance.sh [--session-label <name>]
#
# Default label is today's date + "acceptance".

set -euo pipefail

# -----------------------------------------------------------------
# arg parse
# -----------------------------------------------------------------
SESSION_LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session-label)
            SESSION_LABEL="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--session-label <name>]

Runs the full Layer-D acceptance chain on real hardware:
  * tools/polar_producer.py --self-test                (parser sanity)
  * pytest -m hardware tests/test_hardware_polar.py    (LSL + BLE path)
  * pytest -m acceptance tests/test_acceptance_chain.py (fresh-clone chain)

Artifacts land in artifacts/live/<label>/. Every step must exit 0
for the runner to return 0.

Requires: a powered BT adapter AND a reachable Polar H10 worn with
skin contact. Without those, TestLayerDLiveChain fails as designed.
EOF
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

DATE_UTC="$(date -u +%Y-%m-%d)"
if [[ -z "${SESSION_LABEL}" ]]; then
    SESSION_LABEL="${DATE_UTC}-acceptance"
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="${REPO_ROOT}/artifacts/live/${SESSION_LABEL}"
mkdir -p "${ART_DIR}"

# stdout log
exec > >(tee -a "${ART_DIR}/runner.log") 2>&1

echo "============================================================"
echo "  NeuroPhase Layer-D acceptance runner"
echo "  session label : ${SESSION_LABEL}"
echo "  repo root     : ${REPO_ROOT}"
echo "  artifacts dir : ${ART_DIR}"
echo "  UTC timestamp : $(date -u +%FT%TZ)"
echo "============================================================"

# -----------------------------------------------------------------
# Step 1 — Parser self-test (no hardware, no LSL)
# -----------------------------------------------------------------
echo
echo ">>> [1/3] parser self-test (polar_producer --self-test)"
cd "${REPO_ROOT}"
python tools/polar_producer.py --self-test \
    2> "${ART_DIR}/step1_selftest.stderr" \
    |  tee    "${ART_DIR}/step1_selftest.stdout"
STEP1_RC=${PIPESTATUS[0]}
echo "    exit code: ${STEP1_RC}"
if [[ ${STEP1_RC} -ne 0 ]]; then
    echo "FAIL: parser self-test exit ${STEP1_RC}" >&2
    exit 1
fi

# -----------------------------------------------------------------
# Step 2 — Hardware pytest suite
# -----------------------------------------------------------------
echo
echo ">>> [2/3] pytest -m hardware tests/test_hardware_polar.py"
pytest -q -m hardware tests/test_hardware_polar.py \
    --junit-xml="${ART_DIR}/step2_hardware_junit.xml" \
    2> "${ART_DIR}/step2_hardware.stderr" \
    |  tee    "${ART_DIR}/step2_hardware.stdout"
STEP2_RC=${PIPESTATUS[0]}
echo "    exit code: ${STEP2_RC}"
if [[ ${STEP2_RC} -ne 0 ]]; then
    echo "FAIL: hardware suite exit ${STEP2_RC}" >&2
    exit 2
fi

# -----------------------------------------------------------------
# Step 3 — Fresh-clone acceptance suite (LSL loopback + replay parity)
# -----------------------------------------------------------------
echo
echo ">>> [3/3] pytest -m acceptance tests/test_acceptance_chain.py"
pytest -q -m acceptance tests/test_acceptance_chain.py --no-cov \
    --junit-xml="${ART_DIR}/step3_acceptance_junit.xml" \
    2> "${ART_DIR}/step3_acceptance.stderr" \
    |  tee    "${ART_DIR}/step3_acceptance.stdout"
STEP3_RC=${PIPESTATUS[0]}
echo "    exit code: ${STEP3_RC}"
if [[ ${STEP3_RC} -ne 0 ]]; then
    echo "FAIL: acceptance suite exit ${STEP3_RC}" >&2
    exit 3
fi

# -----------------------------------------------------------------
# Summary
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "  PASS: Layer-D acceptance chain completed"
echo "  artifacts: ${ART_DIR}"
echo "============================================================"

# Emit a machine-readable verdict file.
cat > "${ART_DIR}/VERDICT.json" <<EOF
{
  "session_label": "${SESSION_LABEL}",
  "utc_timestamp": "$(date -u +%FT%TZ)",
  "verdict": "PASS",
  "steps": {
    "parser_self_test": ${STEP1_RC},
    "hardware_suite":   ${STEP2_RC},
    "acceptance_suite": ${STEP3_RC}
  },
  "artifacts_dir": "${ART_DIR}"
}
EOF

exit 0
