#!/usr/bin/env python3
"""
Run-Level Statistical Analysis for GlimpseABM

This module implements proper unit of analysis handling for nested data structure
where agents are clustered within simulation runs. Addresses AMJ reviewer concerns
about independence assumptions.

Key Functions:
- run_level_anova: Test differences across AI tiers using runs as unit
- calculate_icc: Quantify within-run clustering
- bootstrap_run_ci: Proper confidence intervals via run-level resampling
- pairwise_run_tests: All comparisons with FDR correction

Theoretical Foundation:
Agents within runs share market conditions, opportunity distributions, and
competitive dynamics → violates independence assumption. Must aggregate to
run level or use hierarchical models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
import warnings

# ============================================================================
# RUN-LEVEL AGGREGATION
# ============================================================================

def aggregate_to_run_level(results_dir: Path) -> pd.DataFrame:
    """
    Aggregate agent-level data to run-level statistics.

    This is the PRIMARY unit of analysis for causal inference. Each simulation
    run is treated as one independent observation.

    Parameters:
    -----------
    results_dir : Path
        Directory containing simulation results (e.g., 'test_causal_quick/')

    Returns:
    --------
    run_df : DataFrame
        Run-level data with columns:
        - run_id: Unique identifier
        - ai_tier: Treatment assignment
        - n_agents: Total agents in run
        - survival_rate: Proportion survived
        - mean_capital: Average capital (all agents)
        - median_capital: Median capital (all agents)
        - mean_capital_survivors: Average capital (survivors only)
        - mean_actor_ignorance: Average uncertainty dimension
        - mean_practical_indeterminism: Average uncertainty dimension
        - mean_agentic_novelty: Average uncertainty dimension
        - mean_competitive_recursion: Average uncertainty dimension
    """
    run_data = []

    # Find all agent data files - try multiple naming patterns
    agent_files = list(results_dir.glob("**/final_agents.pkl"))
    if not agent_files:
        agent_files = list(results_dir.glob("**/final_agents_*.pkl"))
    if not agent_files:
        agent_files = list(results_dir.glob("**/agents.pkl"))
    if not agent_files:
        agent_files = list(results_dir.glob("**/agents_*.pkl"))
    if not agent_files:
        raise FileNotFoundError(f"No agent data files found in {results_dir}")

    for agent_file in agent_files:
        # Load agent data
        agents = pd.read_pickle(agent_file)

        # Extract run metadata (use parent directory name if file is just "final_agents.pkl")
        if agent_file.stem == 'final_agents' or agent_file.stem == 'agents':
            run_id = agent_file.parent.name
        else:
            run_id = agent_file.stem

        # Get AI tier (should be same for all agents in fixed-tier design)
        if 'primary_ai_level' in agents.columns:
            ai_tier = agents['primary_ai_level'].iloc[0]
            # Normalize naming
            if ai_tier == 'human':
                ai_tier = 'none'
            elif ai_tier == 'premium_ai':
                ai_tier = 'premium'
        elif 'ai_tier' in agents.columns:
            ai_tier = agents['ai_tier'].iloc[0]
        else:
            warnings.warn(f"No AI tier column found in {agent_file}")
            continue

        # Compute run-level statistics
        run_stats = {
            'run_id': run_id,
            'ai_tier': ai_tier,
            'n_agents': len(agents),
            'survival_rate': agents['survived'].mean() if 'survived' in agents.columns else agents['alive'].mean(),
        }

        # Capital metrics
        if 'final_capital' in agents.columns:
            run_stats['mean_capital'] = agents['final_capital'].mean()
            run_stats['median_capital'] = agents['final_capital'].median()
            survivors = agents[agents['survived']]
            run_stats['mean_capital_survivors'] = survivors['final_capital'].mean() if len(survivors) > 0 else np.nan

        # Uncertainty dimensions (if available)
        uncertainty_cols = ['actor_ignorance', 'practical_indeterminism',
                           'agentic_novelty', 'competitive_recursion']
        for col in uncertainty_cols:
            if col in agents.columns:
                run_stats[f'mean_{col}'] = agents[col].mean()

        run_data.append(run_stats)

    run_df = pd.DataFrame(run_data)

    print(f"\n✓ Aggregated {len(run_df)} runs to run-level statistics")
    print(f"  AI tiers: {run_df['ai_tier'].value_counts().to_dict()}")

    return run_df


# ============================================================================
# RUN-LEVEL HYPOTHESIS TESTS
# ============================================================================

def run_level_anova(run_df: pd.DataFrame, outcome: str = 'survival_rate') -> Dict:
    """
    Test for differences across AI tiers using ANOVA at run level.

    This is the PRIMARY test for AI effects. Treats each run as one
    independent observation, avoiding inflated significance from
    agent-level analysis.

    Parameters:
    -----------
    run_df : DataFrame
        Run-level data from aggregate_to_run_level()
    outcome : str
        Outcome variable to test (e.g., 'survival_rate', 'mean_capital')

    Returns:
    --------
    results : dict
        - f_statistic: F-test statistic
        - p_value: Probability under null
        - eta_squared: Effect size (proportion variance explained)
        - power: Statistical power (if detectable)
        - group_means: Mean for each AI tier
        - group_sds: SD for each AI tier
        - n_runs_per_group: Sample sizes
    """
    # Extract groups
    groups = {}
    for tier in run_df['ai_tier'].unique():
        tier_data = run_df[run_df['ai_tier'] == tier][outcome].dropna()
        groups[tier] = tier_data.values

    # ANOVA
    f_stat, p_value = f_oneway(*groups.values())

    # Effect size (eta-squared)
    grand_mean = run_df[outcome].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups.values())
    ss_total = sum((run_df[outcome] - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Interpret effect size
    if eta_squared < 0.01:
        effect_interpretation = "negligible"
    elif eta_squared < 0.06:
        effect_interpretation = "small"
    elif eta_squared < 0.14:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    results = {
        'test': 'One-Way ANOVA (Run-Level)',
        'outcome': outcome,
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'effect_interpretation': effect_interpretation,
        'df_between': len(groups) - 1,
        'df_within': len(run_df) - len(groups),
        'n_runs': len(run_df),
        'group_means': {tier: data.mean() for tier, data in groups.items()},
        'group_sds': {tier: data.std() for tier, data in groups.items()},
        'n_runs_per_group': {tier: len(data) for tier, data in groups.items()},
    }

    return results


def pairwise_run_tests(run_df: pd.DataFrame, outcome: str = 'survival_rate',
                       correction: str = 'fdr_bh') -> pd.DataFrame:
    """
    All pairwise comparisons between AI tiers at run level with FDR correction.

    Parameters:
    -----------
    run_df : DataFrame
        Run-level data
    outcome : str
        Outcome to compare
    correction : str
        Multiple testing correction method ('fdr_bh', 'bonferroni', 'none')

    Returns:
    --------
    results_df : DataFrame
        Pairwise comparison results with columns:
        - comparison: "tier1 vs tier2"
        - mean_diff: Difference in means
        - t_statistic: t-test statistic
        - p_value: Uncorrected p-value
        - p_adjusted: FDR-corrected p-value
        - ci_lower, ci_upper: 95% confidence interval for difference
        - cohens_d: Effect size
    """
    tiers = sorted(run_df['ai_tier'].unique())
    results = []

    for i, tier1 in enumerate(tiers):
        for tier2 in tiers[i+1:]:
            data1 = run_df[run_df['ai_tier'] == tier1][outcome].dropna()
            data2 = run_df[run_df['ai_tier'] == tier2][outcome].dropna()

            # t-test
            t_stat, p_val = ttest_ind(data1, data2)

            # Mean difference
            mean_diff = data2.mean() - data1.mean()

            # Cohen's d
            pooled_std = np.sqrt((data1.var() + data2.var()) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            # 95% CI for difference
            se_diff = np.sqrt(data1.var()/len(data1) + data2.var()/len(data2))
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff

            results.append({
                'comparison': f"{tier1} vs {tier2}",
                'tier1': tier1,
                'tier2': tier2,
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'cohens_d': cohens_d,
                'n1': len(data1),
                'n2': len(data2),
            })

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if correction == 'fdr_bh':
        try:
            from statsmodels.stats.multitest import multipletests
            _, p_adjusted, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
            results_df['p_adjusted'] = p_adjusted
            results_df['correction'] = 'FDR (Benjamini-Hochberg)'
        except ImportError:
            warnings.warn("statsmodels not available, using Bonferroni correction instead")
            results_df['p_adjusted'] = results_df['p_value'] * len(results_df)
            results_df['p_adjusted'] = results_df['p_adjusted'].clip(upper=1.0)
            results_df['correction'] = 'Bonferroni (statsmodels not available)'
    elif correction == 'bonferroni':
        results_df['p_adjusted'] = results_df['p_value'] * len(results_df)
        results_df['p_adjusted'] = results_df['p_adjusted'].clip(upper=1.0)
        results_df['correction'] = 'Bonferroni'
    else:
        results_df['p_adjusted'] = results_df['p_value']
        results_df['correction'] = 'None'

    return results_df


# ============================================================================
# INTRACLASS CORRELATION (ICC)
# ============================================================================

def calculate_icc(agent_df: pd.DataFrame, outcome: str = 'survived',
                  run_id_col: str = 'run_id', treatment_col: str = None) -> Dict:
    """
    Calculate intraclass correlation coefficient (ICC) to quantify clustering.

    For fixed-tier designs, computes BOTH:
    1. Overall ICC (includes treatment effect - inflated)
    2. Within-treatment ICC (true clustering - more accurate)

    Parameters:
    -----------
    agent_df : DataFrame
        Agent-level data with run_id column
    outcome : str
        Outcome variable
    run_id_col : str
        Column identifying runs
    treatment_col : str, optional
        Column identifying treatment assignment (e.g., 'ai_tier', 'primary_ai_level')
        If provided, computes within-treatment ICC

    Returns:
    --------
    icc_results : dict
        - icc_overall: Overall ICC (includes treatment variance)
        - icc_within_treatment: ICC within treatment groups (if treatment_col provided)
        - within_treatment_iccs: Dict of ICC per treatment level
        - interpretation: Verbal description
        - design_effect_overall: Design effect from overall ICC
        - design_effect_within: Design effect from within-treatment ICC
    """
    # Ensure run_id exists
    if run_id_col not in agent_df.columns:
        raise ValueError(f"Column '{run_id_col}' not found in agent data")

    # Convert boolean to numeric if needed
    df = agent_df.copy()
    if df[outcome].dtype == bool:
        df[outcome] = df[outcome].astype(float)

    # Remove missing values
    df = df[[run_id_col, outcome] + ([treatment_col] if treatment_col else [])].dropna()

    def compute_icc(data, run_col, outcome_col):
        """Helper to compute ICC for a subset of data."""
        y = data[outcome_col]
        grand_mean = y.mean()

        # Between-run variance
        run_means = data.groupby(run_col)[outcome_col].mean()
        run_sizes = data.groupby(run_col).size()
        if len(run_means) <= 1:
            return 0.0, 0.0, 0.0

        between_var = ((run_means - grand_mean)**2 * run_sizes).sum() / (len(run_means) - 1)

        # Within-run variance
        within_var = 0
        for run_id in data[run_col].unique():
            run_data = data[data[run_col] == run_id][outcome_col]
            if len(run_data) > 1:
                run_mean = run_data.mean()
                within_var += ((run_data - run_mean)**2).sum()

        dof = len(data) - len(run_means)
        within_var = within_var / dof if dof > 0 else 0

        # ICC
        total_var = between_var + within_var
        icc = between_var / total_var if total_var > 0 else 0

        return icc, between_var, within_var

    # Overall ICC (includes treatment effect)
    icc_overall, between_var, within_var = compute_icc(df, run_id_col, outcome)
    avg_cluster_size = len(df) / df[run_id_col].nunique()
    design_effect_overall = 1 + (avg_cluster_size - 1) * icc_overall

    results = {
        'icc_overall': icc_overall,
        'between_run_variance': between_var,
        'within_run_variance': within_var,
        'total_variance': between_var + within_var,
        'n_runs': df[run_id_col].nunique(),
        'mean_agents_per_run': avg_cluster_size,
        'design_effect_overall': design_effect_overall,
    }

    # Within-treatment ICC (true clustering, no treatment effect)
    if treatment_col and treatment_col in df.columns:
        within_treatment_iccs = {}
        for treatment in df[treatment_col].unique():
            treatment_data = df[df[treatment_col] == treatment]
            if len(treatment_data) > 0 and treatment_data[run_id_col].nunique() > 1:
                icc_treat, _, _ = compute_icc(treatment_data, run_id_col, outcome)
                within_treatment_iccs[treatment] = icc_treat

        # Average within-treatment ICC
        icc_within = np.mean(list(within_treatment_iccs.values()))
        design_effect_within = 1 + (avg_cluster_size - 1) * icc_within

        results['icc_within_treatment'] = icc_within
        results['within_treatment_iccs'] = within_treatment_iccs
        results['design_effect_within'] = design_effect_within

        # Interpretation based on within-treatment ICC
        if icc_within < 0.05:
            interpretation = "negligible within-treatment clustering (agent-level with clustered SEs acceptable)"
        elif icc_within < 0.10:
            interpretation = "small within-treatment clustering (run-level or clustered SEs recommended)"
        elif icc_within < 0.20:
            interpretation = "moderate within-treatment clustering (run-level analysis recommended)"
        else:
            interpretation = "substantial within-treatment clustering (run-level analysis required)"

        results['interpretation'] = interpretation
        results['note'] = (
            f"Overall ICC={icc_overall:.3f} (includes treatment effect). "
            f"Within-treatment ICC={icc_within:.3f} (true clustering). "
            f"For fixed-tier designs, within-treatment ICC is more meaningful."
        )
    else:
        # No treatment column - interpret overall ICC
        if icc_overall < 0.05:
            interpretation = "negligible clustering (agent-level analysis may be acceptable)"
        elif icc_overall < 0.10:
            interpretation = "small clustering (run-level analysis recommended)"
        elif icc_overall < 0.20:
            interpretation = "moderate clustering (run-level analysis strongly recommended)"
        else:
            interpretation = "substantial clustering (run-level analysis required)"

        results['interpretation'] = interpretation
        results['note'] = 'ICC > 0.05 suggests run-level analysis is more appropriate than agent-level'

    return results


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS (RUN-LEVEL)
# ============================================================================

def bootstrap_run_ci(run_df: pd.DataFrame, tier1: str, tier2: str,
                     outcome: str = 'survival_rate', n_bootstrap: int = 10000,
                     confidence_level: float = 0.95) -> Dict:
    """
    Bootstrap confidence interval for difference in means, resampling RUNS.

    This is the correct way to compute CIs given nested data structure.
    Resamples runs (not agents) to respect clustering.

    Parameters:
    -----------
    run_df : DataFrame
        Run-level data
    tier1, tier2 : str
        AI tiers to compare
    outcome : str
        Outcome variable
    n_bootstrap : int
        Number of bootstrap resamples
    confidence_level : float
        CI level (default 0.95 for 95% CI)

    Returns:
    --------
    results : dict
        - mean_diff: Observed difference (tier2 - tier1)
        - ci_lower, ci_upper: Bootstrap CI bounds
        - se_boot: Bootstrap standard error
        - n_runs_tier1, n_runs_tier2: Sample sizes
        - bootstrap_distribution: Array of bootstrap estimates
    """
    # Extract run-level data
    data1 = run_df[run_df['ai_tier'] == tier1][outcome].dropna().values
    data2 = run_df[run_df['ai_tier'] == tier2][outcome].dropna().values

    # Observed difference
    observed_diff = data2.mean() - data1.mean()

    # Bootstrap resampling (resample RUNS, not agents)
    boot_diffs = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Resample runs with replacement
        boot1 = rng.choice(data1, size=len(data1), replace=True)
        boot2 = rng.choice(data2, size=len(data2), replace=True)
        boot_diffs.append(boot2.mean() - boot1.mean())

    boot_diffs = np.array(boot_diffs)

    # CI from percentiles
    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_diffs, 100 * alpha/2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha/2))

    # Bootstrap SE
    se_boot = boot_diffs.std()

    return {
        'comparison': f"{tier1} vs {tier2}",
        'outcome': outcome,
        'mean_diff': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se_boot': se_boot,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'n_runs_tier1': len(data1),
        'n_runs_tier2': len(data2),
        'bootstrap_distribution': boot_diffs,
    }


# ============================================================================
# AGENT-LEVEL ANALYSIS WITH CLUSTERED STANDARD ERRORS
# ============================================================================

def agent_level_analysis_clustered(agent_df: pd.DataFrame, outcome: str = 'survived',
                                   treatment_col: str = 'ai_tier', run_id_col: str = 'run_id') -> Dict:
    """
    Agent-level analysis with clustered standard errors (supplementary analysis).

    This treats agents as the unit of analysis but adjusts standard errors
    for clustering within runs. Appropriate when within-treatment ICC < 0.05.

    Parameters:
    -----------
    agent_df : DataFrame
        Agent-level data
    outcome : str
        Outcome variable (binary or continuous)
    treatment_col : str
        Treatment assignment column
    run_id_col : str
        Run identifier column

    Returns:
    --------
    results : dict
        - means: Mean outcome by treatment
        - t_tests: Pairwise t-tests with clustered SEs
        - note: Interpretation guidance
    """
    results = {}

    # Compute means by treatment
    means = agent_df.groupby(treatment_col)[outcome].agg(['mean', 'std', 'count'])
    results['means'] = means.to_dict('index')

    # Pairwise t-tests with clustered SEs
    # Note: Proper clustered SEs require statsmodels or linearmodels
    # For now, we'll report regular SEs with a warning

    treatments = sorted(agent_df[treatment_col].unique())
    pairwise = []

    for i, treat1 in enumerate(treatments):
        for treat2 in treatments[i+1:]:
            data1 = agent_df[agent_df[treatment_col] == treat1][outcome]
            data2 = agent_df[agent_df[treatment_col] == treat2][outcome]

            # Regular t-test (SEs NOT clustered - needs statsmodels)
            t_stat, p_val = ttest_ind(data1, data2)
            mean_diff = data2.mean() - data1.mean()

            # Naive SE (not adjusted for clustering)
            se_naive = np.sqrt(data1.var()/len(data1) + data2.var()/len(data2))

            pairwise.append({
                'comparison': f"{treat1} vs {treat2}",
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_val,
                'se_naive': se_naive,
                'warning': 'SE not adjusted for clustering (use run-level analysis as primary)'
            })

    results['pairwise'] = pd.DataFrame(pairwise)
    results['note'] = (
        "⚠️ SUPPLEMENTARY ANALYSIS: Agent-level results do not adjust for clustering. "
        "Use run-level analysis as primary. This analysis is provided for comparison only. "
        "For proper clustered SEs, see run-level bootstrap CIs."
    )

    return results


# ============================================================================
# COMPLETE RUN-LEVEL ANALYSIS
# ============================================================================

def run_complete_run_level_analysis(results_dir: Path, output_dir: Optional[Path] = None) -> Dict:
    """
    Execute complete run-level analysis pipeline.

    This is the MAIN function to call for publication-quality analysis.

    Steps:
    1. Aggregate agents to run level
    2. Run-level ANOVA for primary outcomes
    3. Pairwise comparisons with FDR correction
    4. Calculate ICCs to justify run-level approach
    5. Bootstrap CIs for key comparisons
    6. Generate summary tables

    Parameters:
    -----------
    results_dir : Path
        Directory with simulation results
    output_dir : Path, optional
        Where to save output tables (defaults to results_dir/tables_run_level/)

    Returns:
    --------
    all_results : dict
        Complete results dictionary with all analyses
    """
    print("\n" + "="*70)
    print("RUN-LEVEL STATISTICAL ANALYSIS")
    print("="*70)
    print("\nThis analysis treats simulation RUNS (not agents) as the unit of analysis,")
    print("properly accounting for clustering of agents within runs.\n")

    # Set up output directory
    if output_dir is None:
        output_dir = results_dir / "tables_run_level"
    output_dir.mkdir(exist_ok=True, parents=True)

    all_results = {}

    # Step 1: Aggregate to run level
    print("\n[1/5] Aggregating to run level...")
    run_df = aggregate_to_run_level(results_dir)
    run_df.to_csv(output_dir / "run_level_data.csv", index=False)
    all_results['run_data'] = run_df

    # Step 2: Run-level ANOVA for key outcomes
    print("\n[2/5] Run-level ANOVA tests...")
    outcomes = ['survival_rate']
    if 'mean_capital' in run_df.columns:
        outcomes.append('mean_capital')

    uncertainty_outcomes = [col for col in run_df.columns if col.startswith('mean_') and
                           any(dim in col for dim in ['ignorance', 'indeterminism', 'novelty', 'recursion'])]
    outcomes.extend(uncertainty_outcomes)

    anova_results = []
    for outcome in outcomes:
        if outcome in run_df.columns and run_df[outcome].notna().sum() > 0:
            result = run_level_anova(run_df, outcome)
            anova_results.append(result)
            print(f"  ✓ {outcome}: F={result['f_statistic']:.2f}, p={result['p_value']:.4f}, η²={result['eta_squared']:.3f} ({result['effect_interpretation']})")

    all_results['anova'] = anova_results
    pd.DataFrame(anova_results).to_csv(output_dir / "run_level_anova.csv", index=False)

    # Step 3: Pairwise comparisons
    print("\n[3/5] Pairwise comparisons with FDR correction...")
    pairwise_results = {}
    for outcome in outcomes[:3]:  # Primary outcomes only
        if outcome in run_df.columns:
            pw_df = pairwise_run_tests(run_df, outcome)
            pairwise_results[outcome] = pw_df
            print(f"  ✓ {outcome}: {len(pw_df)} comparisons")
            pw_df.to_csv(output_dir / f"pairwise_{outcome}.csv", index=False)

    all_results['pairwise'] = pairwise_results

    # Step 4: ICC calculations (both overall and within-treatment)
    print("\n[4/5] Calculating intraclass correlations (ICC)...")
    # Need agent-level data for ICC - reuse the agent_files found earlier
    agent_files_for_icc = list(results_dir.glob("**/final_agents.pkl"))
    if not agent_files_for_icc:
        agent_files_for_icc = list(results_dir.glob("**/final_agents_*.pkl"))
    if agent_files_for_icc:
        # Combine all agents with run_id
        all_agents = []
        for agent_file in agent_files_for_icc[:10]:  # Sample for efficiency
            agents = pd.read_pickle(agent_file)
            # Use parent directory name as run_id
            if agent_file.stem == 'final_agents' or agent_file.stem == 'agents':
                agents['run_id'] = agent_file.parent.name
            else:
                agents['run_id'] = agent_file.stem
            all_agents.append(agents)

        combined_agents = pd.concat(all_agents, ignore_index=True)

        # Detect treatment column
        treatment_col = None
        for col in ['ai_tier', 'primary_ai_level']:
            if col in combined_agents.columns:
                treatment_col = col
                break

        icc_results = []
        for outcome in ['survived', 'final_capital'] if 'final_capital' in combined_agents.columns else ['survived']:
            if outcome in combined_agents.columns:
                icc = calculate_icc(combined_agents, outcome, treatment_col=treatment_col)
                icc_results.append({**{'outcome': outcome}, **icc})

                # Print both ICCs if available
                if 'icc_within_treatment' in icc:
                    print(f"  ✓ {outcome}:")
                    print(f"    Overall ICC={icc['icc_overall']:.3f} (includes treatment effect)")
                    print(f"    Within-treatment ICC={icc['icc_within_treatment']:.3f} - {icc['interpretation']}")
                    if 'within_treatment_iccs' in icc:
                        print(f"    By tier: {icc['within_treatment_iccs']}")
                else:
                    print(f"  ✓ {outcome}: ICC={icc['icc_overall']:.3f} - {icc['interpretation']}")

        all_results['icc'] = icc_results
        pd.DataFrame(icc_results).to_csv(output_dir / "icc_analysis.csv", index=False)

        # Step 4b: Agent-level analysis (supplementary)
        print("\n[4b] Agent-level analysis with clustering note (supplementary)...")
        if treatment_col:
            agent_analysis = agent_level_analysis_clustered(
                combined_agents, 'survived', treatment_col=treatment_col
            )
            all_results['agent_level'] = agent_analysis
            agent_analysis['pairwise'].to_csv(output_dir / "agent_level_supplementary.csv", index=False)
            print(f"  ✓ Agent-level comparisons computed (use run-level as primary)")
            print(f"  ℹ️  {agent_analysis['note']}")
    else:
        print("  ⚠ Could not calculate ICC (agent-level data not found)")
        all_results['icc'] = []

    # Step 5: Bootstrap CIs for key comparisons
    print("\n[5/5] Bootstrap confidence intervals (run-level resampling)...")
    bootstrap_results = []

    # Key comparisons
    comparisons = [
        ('none', 'premium'),
        ('basic', 'none'),
        ('basic', 'premium'),
    ]

    for tier1, tier2 in comparisons:
        if tier1 in run_df['ai_tier'].values and tier2 in run_df['ai_tier'].values:
            boot_result = bootstrap_run_ci(run_df, tier1, tier2, 'survival_rate')
            bootstrap_results.append(boot_result)
            print(f"  ✓ {tier1} vs {tier2}: Δ={boot_result['mean_diff']:.3f}, 95% CI=[{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]")

    all_results['bootstrap'] = bootstrap_results

    # Save bootstrap summary (without full distributions)
    boot_summary = [{k: v for k, v in br.items() if k != 'bootstrap_distribution'}
                    for br in bootstrap_results]
    pd.DataFrame(boot_summary).to_csv(output_dir / "bootstrap_cis.csv", index=False)

    print("\n" + "="*70)
    print("✅ Run-level analysis complete!")
    print(f"   Results saved to: {output_dir}")
    print("="*70)

    return all_results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_level_analysis.py <results_dir>")
        print("\nExample:")
        print("  python run_level_analysis.py glimpse_abm/test_causal_quick")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    # Run analysis
    results = run_complete_run_level_analysis(results_dir)

    print("\n✓ Analysis complete. See tables_run_level/ directory for results.")
