## Calibration Baseline (Workbook Stub)

Use this notebook (or convert it to `.ipynb`) to document your calibration runs. Recommended workflow:

1. **Run the baseline profile.**
   ```bash
   python3 -m glimpse_abm.cli --task master --calibration-profile venture_baseline_2024 --results-dir ./calibration_runs/baseline
   ```

2. **Load the outputs in Python / pandas** to compare against empirical targets:
   ```python
   import pandas as pd
   summary = pd.read_csv("./calibration_runs/baseline/tables/ai_stage_performance.csv")
   print(summary[['primary_ai_level', 'survival_rate', 'mean_roic_invest']])
   ```

3. **Record calibration errors** relative to your empirical benchmarks (BLS survival, NVCA ROI, etc.).

4. **Iterate with custom overrides** (`--calibration-file`) until your targets fall within tolerance.

5. **Attach final metrics** (e.g., tables, error calculations) to your appendix/replication package.

> Replace this markdown stub with a full Jupyter notebook if you prefer interactive plots.

