#!/bin/bash
#
# RESUME ROBUSTNESS SWEEP: Continues from where it stopped
# Skips baseline (already completed) and runs remaining 7 configurations
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PACKAGE_DIR")"

cd "$PARENT_DIR"

OUTPUT_BASE="${1:-glimpse_robustness_sweep}"

echo "=============================================================="
echo "RESUME ROBUSTNESS SWEEP (skipping baseline)"
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

# Create override files
echo "Creating parameter override files..."

cat > /tmp/robustness_high_cost.json << 'EOF'
{"overrides": {"BASE_OPERATIONAL_COST": 90000}}
EOF

cat > /tmp/robustness_low_cost.json << 'EOF'
{"overrides": {"BASE_OPERATIONAL_COST": 40000}}
EOF

cat > /tmp/robustness_high_threshold.json << 'EOF'
{"overrides": {"SURVIVAL_CAPITAL_RATIO": 0.55}}
EOF

cat > /tmp/robustness_low_threshold.json << 'EOF'
{"overrides": {"SURVIVAL_CAPITAL_RATIO": 0.25}}
EOF

cat > /tmp/robustness_high_noise.json << 'EOF'
{"overrides": {"RETURN_NOISE_SCALE": 0.35}}
EOF

cat > /tmp/robustness_low_noise.json << 'EOF'
{"overrides": {"RETURN_NOISE_SCALE": 0.10}}
EOF

cat > /tmp/robustness_multi_sector.json << 'EOF'
{"overrides": {"SECTOR_PROFILES": {
    "tech": {
        "return_range": [1.2, 2.5],
        "return_log_mu": 0.5,
        "return_log_sigma": 0.3,
        "return_volatility_range": [0.2, 0.3],
        "failure_range": [0.25, 0.40],
        "failure_volatility_range": [0.05, 0.10],
        "capital_range": [200000, 800000],
        "maturity_range": [12, 24],
        "gross_margin_range": [0.5, 0.7],
        "operating_margin_range": [0.1, 0.2]
    },
    "retail": {
        "return_range": [1.15, 2.1],
        "return_log_mu": 0.35,
        "return_log_sigma": 0.25,
        "return_volatility_range": [0.15, 0.25],
        "failure_range": [0.20, 0.38],
        "failure_volatility_range": [0.04, 0.08],
        "capital_range": [150000, 600000],
        "maturity_range": [8, 18],
        "gross_margin_range": [0.3, 0.5],
        "operating_margin_range": [0.05, 0.15]
    },
    "service": {
        "return_range": [1.25, 2.2],
        "return_log_mu": 0.4,
        "return_log_sigma": 0.28,
        "return_volatility_range": [0.12, 0.22],
        "failure_range": [0.10, 0.28],
        "failure_volatility_range": [0.03, 0.07],
        "capital_range": [100000, 400000],
        "maturity_range": [6, 15],
        "gross_margin_range": [0.55, 0.75],
        "operating_margin_range": [0.15, 0.25]
    },
    "manufacturing": {
        "return_range": [1.3, 2.65],
        "return_log_mu": 0.45,
        "return_log_sigma": 0.32,
        "return_volatility_range": [0.18, 0.28],
        "failure_range": [0.25, 0.42],
        "failure_volatility_range": [0.05, 0.09],
        "capital_range": [300000, 1000000],
        "maturity_range": [15, 30],
        "gross_margin_range": [0.35, 0.55],
        "operating_margin_range": [0.08, 0.18]
    }
}}}
EOF

echo "Override files created."
echo ""

# Skip baseline (already completed), run remaining 7 configurations
echo "=============================================================="
echo "RUNNING 7 REMAINING CONFIGURATIONS (skipping baseline)"
echo "=============================================================="
echo ""

run_config "high_cost" \
    "High operational cost (BASE_OPERATIONAL_COST=90000)" \
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
echo "ROBUSTNESS SWEEP RESUMED AND COMPLETE"
echo "=============================================================="
echo "End time: $(date)"
echo ""
