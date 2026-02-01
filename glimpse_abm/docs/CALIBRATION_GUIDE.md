# Calibration & Validation Guide

This guide provides comprehensive instructions for calibrating the GLIMPSE ABM to empirical benchmarks and validating simulation outputs for academic publication. All parameters are calibrated for **monthly cadence** (120 rounds = 10 years).

---

## Table of Contents

1. [Calibration Philosophy](#1-calibration-philosophy)
2. [Empirical Targets](#2-empirical-targets)
3. [Calibration Profiles](#3-calibration-profiles)
4. [Running Calibration](#4-running-calibration)
5. [Interpreting Outputs](#5-interpreting-outputs)
6. [Parameter Tuning](#6-parameter-tuning)
7. [Sector-Specific Calibration](#7-sector-specific-calibration)
8. [AI Paradox Calibration](#8-ai-paradox-calibration)
9. [Robustness Validation](#9-robustness-validation)
10. [Documentation for Publication](#10-documentation-for-publication)

---

## 1. Calibration Philosophy

The GLIMPSE ABM is calibrated to reproduce stylized facts from entrepreneurship research rather than match specific time-series data. This approach:

- **Validates mechanisms** rather than fitting curves
- **Prioritizes behavioral realism** over predictive accuracy
- **Enables causal identification** through controlled experiments

### Key Calibration Principles

1. **Monthly Cadence**: All parameters are expressed as monthly rates (120 rounds = 10 years)
2. **Mechanism Fidelity**: The AI paradox must emerge from competitive crowding, not parameter forcing
3. **Empirical Anchoring**: Survival rates, ROI distributions, and innovation rates match published benchmarks
4. **Sensitivity Stability**: Results should be robust to ±20% parameter variation

---

## 2. Empirical Targets

### Primary Targets (Must Match)

| Metric | Target | Tolerance | Source |
|--------|--------|-----------|--------|
| 10-year survival rate | 45-55% | ±8% | BLS Business Employment Dynamics |
| Mean investment ROI | 1.08-1.15× | ±0.10 | PitchBook Series A-D data |
| Innovation share of returns | 35-45% | ±10% | NVCA 2024 estimates |
| Monthly failure rate | 0.4-0.6% | ±0.2% | BLS establishment exit data |

### Secondary Targets (Should Match)

| Metric | Target | Tolerance | Source |
|--------|--------|-----------|--------|
| Unicorn rate (top 5%) | 3-7% | ±2% | CB Insights unicorn data |
| Tech sector survival premium | +5-10 pp | ±5% | Kauffman Foundation |
| Capital concentration (Gini) | 0.55-0.70 | ±0.10 | Federal Reserve SCF |

### AI Paradox Targets (Critical for Paper)

| Metric | Target | Tolerance | Mechanism |
|--------|--------|-----------|-----------|
| Premium vs None survival gap | -8 to -14 pp | ±3% | Competitive crowding |
| Premium competition ratio | 2-3× None | ±0.5× | Convergent decisions |
| Premium unicorn deficit | ~50% fewer | ±15% | Homogenized outcomes |

---

## 3. Calibration Profiles

### `minimal_causal` (Recommended for Academic Research)

Optimized for clean causal identification of AI effects:

```python
# Key settings
N_AGENTS = 1000
N_ROUNDS = 120          # 10 years monthly
N_RUNS = 50             # Per AI tier
SECTORS = ["tech"]      # Single sector eliminates confounds
AI_ASSIGNMENT = "fixed" # Exogenous tier assignment
```

**Use for**: Primary analysis, ATE estimation, mechanism tests

### `venture_baseline_2024`

Full-scale configuration anchored to 2024 empirical benchmarks:

```python
# Key settings
N_AGENTS = 1000
N_ROUNDS = 120
N_RUNS = 30
SECTORS = ["tech", "retail", "service", "manufacturing"]
INITIAL_CAPITAL = 5_000_000
SURVIVAL_THRESHOLD = 230_000
```

**Use for**: Ecological validity, sector comparisons, robustness checks

### `deeptech_capital_constrained`

High-risk/high-reward configuration for deep tech ventures:

```python
# Key overrides
SURVIVAL_CAPITAL_RATIO = 0.55    # Harsher survival pressure
INNOVATION_PROBABILITY = 0.18    # Higher innovation rate
BASE_OPERATIONAL_COST = 25_000   # Higher monthly burn
```

**Use for**: Alternative specification tests, boundary conditions

---

## 4. Running Calibration

### Python CLI

```bash
# Basic calibration run
python3 -m glimpse_abm.cli --task master \
  --calibration-profile venture_baseline_2024 \
  --runs 30 \
  --results-dir ./calibration_runs/baseline

# Fixed-tier causal analysis (recommended)
python3 -m glimpse_abm.cli --task fixed \
  --calibration-profile minimal_causal \
  --results-dir ./calibration_runs/fixed_tier

# Quick smoke test
python3 -m glimpse_abm.cli --task master \
  --smoke \
  --results-dir ./smoke_test
```

### Julia (High-Performance)

```julia
using GlimpseABM

# Configure for calibration
config = EmergentConfig(
    N_AGENTS = 1000,
    N_ROUNDS = 120,
    INITIAL_CAPITAL = 5_000_000.0,
    SURVIVAL_THRESHOLD = 230_000.0,
    BASE_OPERATIONAL_COST = 16_667.0,
    RANDOM_SEED = 42
)

# Run fixed-tier experiments
for tier in ["none", "basic", "advanced", "premium"]
    for run in 1:50
        tier_dist = Dict(t => (t == tier ? 1.0 : 0.0)
                         for t in ["none","basic","advanced","premium"])
        sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)
        for r in 1:120
            GlimpseABM.step!(sim, r)
        end
        # Record results...
    end
end
```

---

## 5. Interpreting Outputs

### Key Output Files

| File | Purpose | Key Metrics |
|------|---------|-------------|
| `tables/ai_stage_performance.csv` | AI tier comparison | survival_rate, mean_roic, action_shares |
| `tables/ai_uncertainty_paradox_by_ai.csv` | Uncertainty decomposition | actor_ignorance, competitive_recursion |
| `tables/ai_paradox_signal.csv` | Paradox measurement | confidence_roi_gap, competition_ratio |
| `tables/matured_outcomes_by_ai.csv` | Investment outcomes | realized_multiples, failure_rates |
| `config_snapshot.json` | Full configuration | All 150+ parameters |
| `analysis_metadata.json` | Version tracking | analysis_version, table_paths |

### Reading the Paradox Signal

The AI paradox is confirmed when:

```
Premium survival < None survival (by 8-14 pp)
AND
Premium competition_ratio > None competition_ratio (by 2-3×)
AND
Correlation(competition_ratio, survival) < 0 (within Premium tier)
```

### Sample Analysis Code

```python
import pandas as pd

# Load results
perf = pd.read_csv("./results/tables/ai_stage_performance.csv")
paradox = pd.read_csv("./results/tables/ai_paradox_signal.csv")

# Check survival gap
none_surv = perf[perf['primary_ai_level'] == 'none']['survival_rate'].values[0]
prem_surv = perf[perf['primary_ai_level'] == 'premium']['survival_rate'].values[0]
gap = none_surv - prem_surv
print(f"Survival gap (None - Premium): {gap:.1%}")

# Check competition ratio
none_cr = paradox[paradox['ai_tier'] == 'none']['mean_competition_ratio'].values[0]
prem_cr = paradox[paradox['ai_tier'] == 'premium']['mean_competition_ratio'].values[0]
print(f"Competition ratio (Premium/None): {prem_cr/none_cr:.2f}×")
```

---

## 6. Parameter Tuning

### Monthly Rate Parameters

All rate parameters are calibrated for monthly resolution:

| Parameter | Monthly Value | Annual Equivalent | Tuning Range |
|-----------|---------------|-------------------|--------------|
| `BASE_OPERATIONAL_COST` | $16,667 | $200,000 | ±30% |
| `DISCOVERY_PROBABILITY` | 0.10 | ~70%/year | ±30% |
| `INNOVATION_PROBABILITY` | 0.14 | ~82%/year | ±30% |
| `KNOWLEDGE_DECAY_RATE` | 0.025 | ~26%/year | ±50% |
| `BLACK_SWAN_PROBABILITY` | 0.017 | ~18%/year | ±50% |
| `MARKET_SHIFT_PROBABILITY` | 0.03 | ~30%/year | ±30% |

### Competition Parameters (Critical for Paradox)

These parameters directly affect the paradox magnitude:

| Parameter | Default | Effect on Paradox | Tuning Notes |
|-----------|---------|-------------------|--------------|
| `OPPORTUNITY_COMPETITION_PENALTY` | 0.5 | ↑ increases gap | Core crowding mechanism |
| `CAPACITY_PENALTY_MAX` | 0.4 | ↑ increases gap | Maximum return dilution |
| `CAPACITY_PENALTY_START` | 0.7 | ↓ increases gap | Earlier penalty onset |
| `OPPORTUNITY_BASE_CAPACITY` | 500,000 | ↓ increases gap | Smaller capacity = more crowding |

### Custom Calibration File

Create a JSON file for custom parameter overrides:

```json
{
  "name": "my_calibration_v1",
  "description": "Tuned for 2024 tech sector benchmarks",
  "monthly_cadence": true,
  "overrides": {
    "BASE_OPERATIONAL_COST": 18000,
    "SURVIVAL_CAPITAL_RATIO": 0.40,
    "OPPORTUNITY_COMPETITION_PENALTY": 0.55,
    "CAPACITY_PENALTY_MAX": 0.45
  },
  "target_metrics": {
    "survival_rate_10yr": {"target": 0.50, "tolerance": 0.05},
    "mean_investment_roi": {"target": 1.12, "tolerance": 0.08},
    "paradox_gap_pp": {"target": 0.12, "tolerance": 0.03}
  }
}
```

Run with:
```bash
python3 -m glimpse_abm.cli --task master --calibration-file ./my_calibration_v1.json
```

---

## 7. Sector-Specific Calibration

### Sector Profiles (Monthly Cadence)

| Parameter | Tech | Retail | Service | Manufacturing | Source |
|-----------|------|--------|---------|---------------|--------|
| `initial_capital_range` | $3M-$6M | $2.2M-$4M | $1.4M-$2.5M | $4M-$7.5M | NVCA 2024 |
| `operational_cost_range` | $20k-$30k/mo | $13k-$23k/mo | $8k-$15k/mo | $27k-$40k/mo | BLS QCEW |
| `survival_threshold` | $150,000 | $130,000 | $70,000 | $200,000 | BLS BED |
| `innovation_probability` | 0.16/mo | 0.11/mo | 0.13/mo | 0.17/mo | NSF BRDIS |
| `knowledge_decay_rate` | 0.04/mo | 0.023/mo | 0.017/mo | 0.01/mo | Skill research |
| `competition_intensity` | 1.2 | 0.7 | 0.9 | 1.4 | Census HHI |

### Sector Weights

From NVCA 2024 Deal Flow:
- **Tech**: 60%
- **Service**: 15%
- **Manufacturing**: 15%
- **Retail**: 10%

### Sector-Specific Innovation Returns

| Sector | Return Multiplier Range | Rationale |
|--------|------------------------|-----------|
| Tech | (2.0×, 4.0×) | High upside, venture-scale |
| Manufacturing | (1.5×, 2.8×) | Incremental improvements |
| Service | (1.6×, 2.5×) | Moderate returns |
| Retail | (1.6×, 2.5×) | Moderate returns |

---

## 8. AI Paradox Calibration

### Mechanism Verification

The paradox must emerge from the **convergence → crowding → dilution** mechanism, not from:
- Direct survival penalties on Premium agents
- Artificially high Premium costs
- Reduced Premium information quality

### Verification Tests

1. **Disable competition test**: Set `DISABLE_COMPETITION_DYNAMICS = true`
   - Expected: Paradox disappears (Premium outperforms None)

2. **Equal capacity test**: Set very high `OPPORTUNITY_BASE_CAPACITY`
   - Expected: Paradox diminishes (less crowding)

3. **Random assignment test**: Shuffle tier assignments post-hoc
   - Expected: Paradox disappears (placebo check)

### Competition Ratio Calibration

Target competition ratios by tier:

| AI Tier | Target Mean CR | Target SD | Mechanism |
|---------|---------------|-----------|-----------|
| None | 3-5 | 2-3 | Diverse, dispersed choices |
| Basic | 4-7 | 2-4 | Moderate convergence |
| Advanced | 6-10 | 3-5 | High convergence |
| Premium | 10-15 | 4-6 | Maximum convergence |

### Tuning for Paradox Magnitude

If paradox is **too weak** (gap < 8pp):
- Increase `OPPORTUNITY_COMPETITION_PENALTY`
- Decrease `OPPORTUNITY_BASE_CAPACITY`
- Increase `CAPACITY_PENALTY_MAX`

If paradox is **too strong** (gap > 14pp):
- Decrease `OPPORTUNITY_COMPETITION_PENALTY`
- Increase `OPPORTUNITY_BASE_CAPACITY`
- Decrease `CAPACITY_PENALTY_MAX`

---

## 9. Robustness Validation

### Standard Robustness Battery

1. **Sensitivity Analysis** (`--task sensitivity`)
   - ±10%, ±20% variation on key parameters
   - Report coefficient of variation across runs

2. **Latin Hypercube Sampling** (`--task lhs`)
   - Simultaneous variation across parameter space
   - 100+ samples recommended

3. **Bootstrap Confidence Intervals**
   - 1000 bootstrap resamples
   - Report 95% CI for all key metrics

4. **Placebo Tests**
   - Shuffle AI tier assignments
   - Verify paradox disappears

5. **Balanced Design**
   - Equal agents per tier (250 each)
   - Verify results hold

### Robustness Configurations

```bash
# Sensitivity sweep
python3 -m glimpse_abm.cli --task sensitivity \
  --results-dir ./robustness/sensitivity

# LHS sweep
python3 -m glimpse_abm.cli --task lhs \
  --results-dir ./robustness/lhs \
  --lhs-samples 100
```

### Key Robustness Outputs

| File | Content |
|------|---------|
| `sensitivity_summary.csv` | Parameter sensitivity coefficients |
| `sensitivity_effects.csv` | Effect sizes per parameter |
| `lhs_summary.csv` | LHS sample results |
| `bootstrap_ci.csv` | Bootstrap confidence intervals |

### Acceptable Robustness Ranges

The calibration is considered robust if:

| Metric | Acceptable Range | Notes |
|--------|------------------|-------|
| Survival rate | 45-60% | Across all LHS samples |
| Paradox gap | 5-18 pp | Must remain negative |
| Competition ratio (Premium) | 8-20 | Must exceed None |
| Effect size (Cohen's d) | 0.3-0.8 | Medium-to-large effect |

---

## 10. Documentation for Publication

### Methods Appendix Checklist

1. **Empirical Targets**
   - [ ] List all target metrics with sources
   - [ ] Document tolerance ranges
   - [ ] Justify target selection

2. **Calibration Process**
   - [ ] Describe baseline profile used
   - [ ] List all parameter overrides
   - [ ] Report iteration count and convergence

3. **Calibration Error**
   - [ ] Report error per target metric
   - [ ] Include confidence intervals
   - [ ] Note any targets outside tolerance

4. **Robustness Evidence**
   - [ ] Sensitivity analysis summary
   - [ ] LHS sample distributions
   - [ ] Placebo test results
   - [ ] Bootstrap confidence intervals

5. **Reproducibility**
   - [ ] Include `config_snapshot.json`
   - [ ] Record random seeds
   - [ ] Provide replication code

### Sample Methods Text

> "The simulation was calibrated to match empirical benchmarks from the U.S. venture capital ecosystem. We targeted a 10-year survival rate of 50% (±8%), consistent with BLS Business Employment Dynamics data, and a mean investment ROI of 1.12× (±0.10), consistent with PitchBook Series A-D returns. Parameters were initialized from the `venture_baseline_2024` profile and iteratively refined over 12 calibration runs until all targets fell within tolerance. The AI Information Paradox emerged organically from the competitive crowding mechanism, with Premium AI agents exhibiting 12.3 percentage points lower survival than None agents (95% CI: [9.8, 14.7]). This effect was robust across 100 Latin Hypercube samples (range: 8.1-16.2 pp) and disappeared in placebo tests where AI tier assignments were shuffled (gap: 0.2 pp, p > 0.8)."

### Replication Package Contents

```
replication/
├── config_snapshot.json      # Full parameter configuration
├── calibration_log.md        # Calibration iteration history
├── random_seeds.txt          # All seeds used
├── run_replication.sh        # Single command to reproduce
├── tables/                   # All output tables
│   ├── ai_stage_performance.csv
│   ├── ai_paradox_signal.csv
│   └── ...
├── robustness/               # Robustness analysis
│   ├── sensitivity_summary.csv
│   ├── lhs_summary.csv
│   └── bootstrap_ci.csv
└── figures/                  # Generated visualizations
    ├── survival_by_tier.png
    ├── competition_ratio_dist.png
    └── ...
```

---

## Quick Reference: Monthly Conversion

When adapting quarterly parameters to monthly cadence:

| Parameter Type | Conversion | Example |
|---------------|------------|---------|
| Probabilities | ÷3 | 0.30/quarter → 0.10/month |
| Costs | ÷3 | $50,000/quarter → $16,667/month |
| Decay rates | ÷3 | 0.075/quarter → 0.025/month |
| Window sizes | ×3 | 15 quarters → 45 months |
| Regime transitions | `p_mo = 1-(1-p_q)/3` | See transition matrix |

---

## References

BLS Business Employment Dynamics (2024). Establishment birth and death statistics.

NVCA (2024). NVCA 2024 Yearbook. National Venture Capital Association.

NSF BRDIS (2023). Business R&D and Innovation Survey. National Science Foundation.

PitchBook (2024). Venture capital performance benchmarks.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025). Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.
