#!/usr/bin/env python3
"""
Analyze robustness sweep results and generate publication-ready summary.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def analyze(results_base: str = "glimpse_robustness_sweep"):
    """Analyze all robustness configurations and generate summary tables."""

    results_dir = Path(results_base)

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found.")
        return

    configs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not configs:
        print(f"Error: No configuration directories found in '{results_dir}'")
        return

    print("=" * 80)
    print("ROBUSTNESS ANALYSIS: Fixed-Tier Causal Effects Across Parameter Space")
    print("=" * 80)
    print(f"\nFound {len(configs)} configurations to analyze.")

    # Collect results from all configurations
    all_results = []

    for config_dir in sorted(configs):
        config_name = config_dir.name

        # Load agent data from all runs
        run_dirs = list(config_dir.glob('Fixed_AI_Level_*'))

        if not run_dirs:
            print(f"  Warning: No runs found in {config_name}")
            continue

        agents_list = []
        for run_dir in run_dirs:
            agents_file = run_dir / 'final_agents.pkl'
            if agents_file.exists():
                with open(agents_file, 'rb') as f:
                    agents_df = pickle.load(f)
                tier = run_dir.name.split('_')[3]
                agents_df['ai_tier'] = tier
                agents_df['config'] = config_name
                agents_list.append(agents_df)

        if not agents_list:
            continue

        agents_df = pd.concat(agents_list, ignore_index=True)

        # Compute survival by tier
        baseline = agents_df[agents_df['ai_tier'] == 'none']['survived']
        baseline_mean = baseline.mean()
        baseline_std = baseline.std()

        for tier in ['basic', 'advanced', 'premium']:
            treatment = agents_df[agents_df['ai_tier'] == tier]['survived']

            if len(treatment) == 0:
                continue

            ate = treatment.mean() - baseline_mean
            pooled_std = np.sqrt((baseline_std**2 + treatment.std()**2) / 2)
            cohens_d = ate / pooled_std if pooled_std > 0 else 0

            t_stat, p_value = stats.ttest_ind(treatment, baseline)
            se = np.sqrt(baseline.var()/len(baseline) + treatment.var()/len(treatment))
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se

            all_results.append({
                'config': config_name,
                'comparison': f'{tier}_vs_none',
                'baseline_survival': baseline_mean,
                'treatment_survival': treatment.mean(),
                'ate': ate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'cohens_d': cohens_d,
                'p_value': p_value,
                'n_treatment': len(treatment),
                'n_control': len(baseline),
            })

        print(f"  Processed: {config_name} ({len(run_dirs)} runs)")

    if not all_results:
        print("Error: No results to analyze.")
        return

    results_df = pd.DataFrame(all_results)

    # Print summary table
    print("\n" + "=" * 80)
    print("TABLE: Robustness of Causal Effects Across Parameter Configurations")
    print("=" * 80)

    # Pivot for display
    summary = results_df.pivot_table(
        index='config',
        columns='comparison',
        values=['ate', 'cohens_d'],
        aggfunc='first'
    )

    print("\nAverage Treatment Effects (ATE):")
    print("-" * 60)
    ate_table = results_df.pivot(index='config', columns='comparison', values='ate')
    print(ate_table.round(3).to_string())

    print("\n\nCohen's d Effect Sizes:")
    print("-" * 60)
    d_table = results_df.pivot(index='config', columns='comparison', values='cohens_d')
    print(d_table.round(3).to_string())

    # Robustness bounds
    print("\n" + "=" * 80)
    print("ROBUSTNESS BOUNDS (Range of Effects Across All Configurations)")
    print("=" * 80)

    for comparison in ['basic_vs_none', 'advanced_vs_none', 'premium_vs_none']:
        subset = results_df[results_df['comparison'] == comparison]

        print(f"\n{comparison}:")
        print(f"  ATE Range:      [{subset['ate'].min():.3f}, {subset['ate'].max():.3f}]")
        print(f"  ATE Mean ± SD:  {subset['ate'].mean():.3f} ± {subset['ate'].std():.3f}")
        print(f"  Cohen's d Range: [{subset['cohens_d'].min():.3f}, {subset['cohens_d'].max():.3f}]")

        # Check if effect direction is consistent
        all_negative = (subset['ate'] < 0).all()
        print(f"  Direction consistent: {'YES (all negative)' if all_negative else 'NO (some positive)'}")

    # Save results
    output_file = results_dir / 'robustness_summary.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")

    # Create publication table
    pub_table = []
    for config in results_df['config'].unique():
        config_data = results_df[results_df['config'] == config]
        row = {'Configuration': config}
        for _, r in config_data.iterrows():
            comp = r['comparison'].replace('_vs_none', '')
            row[f'{comp}_ATE'] = f"{r['ate']:.3f}"
            row[f'{comp}_d'] = f"{r['cohens_d']:.2f}"
        pub_table.append(row)

    pub_df = pd.DataFrame(pub_table)
    pub_file = results_dir / 'robustness_publication_table.csv'
    pub_df.to_csv(pub_file, index=False)
    print(f"Publication table saved to: {pub_file}")

    return results_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        analyze()
