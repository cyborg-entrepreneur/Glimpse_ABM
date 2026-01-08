## Calibration & Validation Guide

This guide outlines how to align Glimpse ABM with empirical benchmarks (venture survival, ROIC, AI adoption, etc.) so that your research submission can document a reproducible calibration.

### 1. Choose a Baseline

Start from one of the bundled profiles:

* `venture_baseline_2024` – targets ~55% five-year survival and 1.1× ROIC.
* `deeptech_capital_constrained` – harsher survival (~35%) with higher innovation payoffs.

Apply it via the CLI:

```bash
python3 -m glimpse_abm.cli --task master --calibration-profile venture_baseline_2024
```

### 2. Inspect Targets

After the run:

* `tables/ai_stage_performance.csv` – check survival, ROIC, action shares by AI tier.
* `tables/matured_outcomes_by_ai.csv` – inspect realized multiples per tier.
* `tables/ai_uncertainty_paradox_by_ai.csv` – shows the four uncertainty components and paradox signal by AI tier.
* `tables/ai_paradox_signal*.csv` – raw and cohort paradox-gap measures (decision confidence vs realized ROI).
* `analysis_metadata.json` – note the `analysis_version` and table paths for your appendix.

Compare these to your empirical targets (e.g., BLS survival, NVCA ROI). Record the calibration error (absolute or relative) for each target.

### 3. Refine Parameters

If the gap is large, create a custom calibration file:

```json
{
  "name": "venture_tuned_v1",
  "description": "Tweaked opex and survival ratio to match 2019 cohort.",
  "overrides": {
    "BASE_OPERATIONAL_COST": 68000,
    "SURVIVAL_CAPITAL_RATIO": 0.53,
    "RETURN_OVERSUPPLY_PENALTY": 0.5
  },
  "target_metrics": {
    "survival_rate_round250": {"target": 0.55, "tolerance": 0.05},
    "mean_investment_roi": {"target": 1.12, "tolerance": 0.1}
  }
}
```

Run:

```bash
python3 -m glimpse_abm.cli --task master --calibration-file ./calibration/venture_tuned_v1.json
```

Iterate until the simulated metrics land within your tolerances.

Common knobs in the current build:

* **Downside pressure:** `DOWNSIDE_OVERSUPPLY_WEIGHT` (higher → deeper losses), `RETURN_LOWER_BOUND` (e.g., -1.5 to allow heavier tails).
* **AI novelty uplift:** `AI_NOVELTY_UPLIFT` (higher → more exploration/novelty from AI).
* **Recursion weights:** `RECURSION_WEIGHTS.crowd_weight`, `.ai_herd_weight`, `.volatility_weight`, `.premium_reuse_weight` (control how crowding/volatility/AI herding shape competitive recursion).
* **Capital resilience:** `SURVIVAL_CAPITAL_RATIO`, `BASE_OPERATIONAL_COST`.

### 4. Validate Robustness

Use the built-in sweeps to show the model is stable:

* **Grid:** `--task sensitivity` with a few +/-10% variations around your calibrated values.
* **Latin Hypercube:** `--task lhs` for broader coverage.

Include the resulting `sensitivity_summary.csv`, `sensitivity_effects.csv`, and `lhs_summary.csv` in your replication package. Summarize the ranges in your appendix (e.g., “survival rate stayed between 47–61% across the LHS samples”).

### 5. Document the Process

When writing the methods appendix:

1. List the empirical targets and data sources (BLS, NVCA, etc.).
2. Describe the calibration steps (baseline profile → custom overrides → iteration count).
3. Report the calibration error per target.
4. Reference the robustness sweeps and attach the CSVs/plots.

### 6. Recommended Automation

* Keep a notebook (e.g., `notebooks/calibration.ipynb`) that runs the CLI, parses the key tables, and displays error bars.
* Store calibration files under `calibration/` with descriptive names and comments.

Following this workflow ensures reviewers can reproduce your parameter choices and verify that the ABM matches the empirical phenomena you cite. Feel free to expand this guide as you add new targets or scenarios.
