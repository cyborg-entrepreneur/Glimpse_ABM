#!/bin/bash
#
# RESUME ROBUSTNESS SWEEP v2: After cliffs_delta fix
# Re-runs stats for high_cost, then runs remaining 6 configurations
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PACKAGE_DIR")"

cd "$PARENT_DIR"

OUTPUT_BASE="${1:-glimpse_robustness_sweep}"

echo "=============================================================="
echo "RESUME ROBUSTNESS SWEEP v2 (with fixed cliffs_delta)"
echo "=============================================================="
echo "Working directory: $(pwd)"
echo "Output directory: $OUTPUT_BASE"
echo "Resume time: $(date)"
echo ""

run_config() {
    local name=$1
    local description=$2
    local override_file=$3

    echo "--------------------------------------------------------------"
    echo "[$name] $description"
    echo "--------------------------------------------------------------"

    if [ -z "$override_file" ]; then
        python3 -m glimpse_abm.cli \
            --task fixed \
            --calibration-profile minimal_causal \
            --results-dir "$OUTPUT_BASE/$name" \
            --skip-visualizations \
            --fast-stats
    else
        python3 -m glimpse_abm.cli \
            --task fixed \
            --calibration-profile minimal_causal \
            --calibration-file "$override_file" \
            --results-dir "$OUTPUT_BASE/$name" \
            --skip-visualizations \
            --fast-stats
    fi

    echo "[$name] Completed at $(date)"
    echo ""
}

echo "=============================================================="
echo "RUNNING 7 CONFIGURATIONS (high_cost + 6 remaining)"
echo "=============================================================="
echo ""

# Re-run high_cost (simulations complete, just needs stats with fixed cliffs_delta)
run_config "high_cost" \
    "High operational cost (BASE_OPERATIONAL_COST=90000) - RE-RUNNING STATS" \
    "/tmp/robustness_high_cost.json"

run_config "low_cost" \
    "Low operational cost (BASE_OPERATIONAL_COST=40000)" \
    "/tmp/robustness_low_cost.json"

run_config "high_threshold" \
    "High survival threshold (SURVIVAL_CAPITAL_RATIO=0.55)" \
    "/tmp/robustness_high_threshold.json"

run_config "low_threshold" \
    "Low survival threshold (SURVIVAL_CAPITAL_RATIO=0.25)" \
    "/tmp/robustness_low_threshold.json"

run_config "high_noise" \
    "High return noise (RETURN_NOISE_SCALE=0.35)" \
    "/tmp/robustness_high_noise.json"

run_config "low_noise" \
    "Low return noise (RETURN_NOISE_SCALE=0.10)" \
    "/tmp/robustness_low_noise.json"

run_config "multi_sector" \
    "Multi-sector model (4 sectors for ecological validity)" \
    "/tmp/robustness_multi_sector.json"

echo "=============================================================="
echo "ROBUSTNESS SWEEP COMPLETE"
echo "=============================================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "To analyze results, run:"
echo "  python3 -m glimpse_abm.scripts.analyze_robustness '$OUTPUT_BASE'"
echo ""
