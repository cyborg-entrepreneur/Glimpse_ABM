"""
Publication Output Generator for Glimpse ABM.

This module generates publication-ready outputs including:
- LaTeX tables with proper formatting, significance stars, and notes
- High-resolution figures suitable for journal submission
- PDF reports combining all analyses

Supports the empirical investigation of:
    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.
"""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def _format_p_value(p: float, include_stars: bool = True) -> str:
    """Format p-value with significance stars."""
    if pd.isna(p):
        return ""

    stars = ""
    if include_stars:
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        elif p < 0.10:
            stars = "†"

    if p < 0.001:
        return f"$<$.001{stars}"
    elif p < 0.01:
        return f"{p:.3f}{stars}"
    else:
        return f"{p:.3f}{stars}"


def _format_number(x: float, decimals: int = 2, thousands_sep: bool = True) -> str:
    """Format number for LaTeX table."""
    if pd.isna(x):
        return "—"

    if thousands_sep and abs(x) >= 1000:
        return f"{x:,.{decimals}f}"
    return f"{x:.{decimals}f}"


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        return str(text)

    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class LaTeXTableGenerator:
    """Generate publication-ready LaTeX tables."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables_generated: List[str] = []

    def generate_descriptive_stats_table(
        self,
        agent_df: pd.DataFrame,
        filename: str = "table_descriptive_stats.tex",
        caption: str = "Descriptive Statistics by AI Tier",
        label: str = "tab:descriptive"
    ) -> Path:
        """
        Generate Table 1: Descriptive statistics by AI tier.

        Parameters
        ----------
        agent_df : pd.DataFrame
            Agent-level data with columns: ai_tier, final_capital, survived,
            successful_investments, failed_investments, etc.
        """
        # Group by AI tier
        tier_order = ['none', 'basic', 'advanced', 'premium']
        available_tiers = [t for t in tier_order if t in agent_df['ai_tier'].unique()]

        rows = []
        for tier in available_tiers:
            tier_data = agent_df[agent_df['ai_tier'] == tier]
            n = len(tier_data)

            # Survival rate
            survived = tier_data['survived'].mean() if 'survived' in tier_data.columns else np.nan

            # Capital statistics
            cap_mean = tier_data['final_capital'].mean() if 'final_capital' in tier_data.columns else np.nan
            cap_sd = tier_data['final_capital'].std() if 'final_capital' in tier_data.columns else np.nan

            # Investment outcomes
            if 'successful_investments' in tier_data.columns:
                success_rate = (tier_data['successful_investments'] /
                               (tier_data['successful_investments'] + tier_data['failed_investments'] + 0.001)).mean()
            else:
                success_rate = np.nan

            rows.append({
                'AI Tier': tier.capitalize(),
                'N': n,
                'Survival Rate': survived,
                'Mean Capital': cap_mean,
                'SD Capital': cap_sd,
                'Investment Success': success_rate
            })

        df = pd.DataFrame(rows)

        # Build LaTeX
        latex = self._build_table_header(caption, label, len(df.columns))
        latex += r"\toprule" + "\n"
        latex += r"AI Tier & $N$ & Survival & \multicolumn{2}{c}{Final Capital} & Investment \\" + "\n"
        latex += r" & & Rate & Mean & SD & Success Rate \\" + "\n"
        latex += r"\midrule" + "\n"

        for _, row in df.iterrows():
            latex += f"{row['AI Tier']} & {row['N']:,} & {_format_number(row['Survival Rate'], 3)} & "
            latex += f"\\${_format_number(row['Mean Capital']/1000, 1)}K & "
            latex += f"\\${_format_number(row['SD Capital']/1000, 1)}K & "
            latex += f"{_format_number(row['Investment Success'], 3)} \\\\\n"

        latex += r"\bottomrule" + "\n"
        latex += self._build_table_footer([
            "Note: Statistics computed across all simulation runs.",
            "Capital values in thousands of dollars."
        ])

        output_path = self.output_dir / filename
        output_path.write_text(latex)
        self.tables_generated.append(filename)
        return output_path

    def generate_hypothesis_tests_table(
        self,
        test_results: List[Dict[str, Any]],
        filename: str = "table_hypothesis_tests.tex",
        caption: str = "Statistical Tests of AI Tier Effects",
        label: str = "tab:hypothesis"
    ) -> Path:
        """
        Generate Table 2: Hypothesis test results with effect sizes.

        Parameters
        ----------
        test_results : List[Dict]
            Each dict should have: test_name, statistic, p_value, effect_size,
            effect_type, ci_lower, ci_upper, interpretation
        """
        latex = self._build_table_header(caption, label, 6)
        latex += r"\toprule" + "\n"
        latex += r"Hypothesis & Test & Statistic & $p$-value & Effect Size & 95\% CI \\" + "\n"
        latex += r"\midrule" + "\n"

        for result in test_results:
            test_name = _escape_latex(result.get('test_name', ''))
            statistic = result.get('statistic', np.nan)
            p_val = result.get('p_value', np.nan)
            effect = result.get('effect_size', np.nan)
            effect_type = result.get('effect_type', "Cohen's $d$")
            ci_lower = result.get('ci_lower', np.nan)
            ci_upper = result.get('ci_upper', np.nan)

            latex += f"{test_name} & {result.get('test_type', '')} & "
            latex += f"{_format_number(statistic, 2)} & {_format_p_value(p_val)} & "
            latex += f"{_format_number(effect, 2)} & [{_format_number(ci_lower, 2)}, {_format_number(ci_upper, 2)}] \\\\\n"

        latex += r"\bottomrule" + "\n"
        latex += self._build_table_footer([
            f"Note: Effect sizes reported as {effect_type}.",
            "†$p < .10$; *$p < .05$; **$p < .01$; ***$p < .001$"
        ])

        output_path = self.output_dir / filename
        output_path.write_text(latex)
        self.tables_generated.append(filename)
        return output_path

    def generate_uncertainty_transformation_table(
        self,
        uncertainty_df: pd.DataFrame,
        filename: str = "table_uncertainty_transformation.tex",
        caption: str = "Uncertainty Transformation by AI Tier",
        label: str = "tab:uncertainty"
    ) -> Path:
        """
        Generate Table 3: Uncertainty dimension changes by AI tier.

        Shows how each dimension of Knightian uncertainty is affected by AI adoption,
        including the paradox effects (AI deltas).
        """
        tier_order = ['none', 'basic', 'advanced', 'premium']

        dimensions = [
            ('actor_ignorance', 'Actor Ignorance'),
            ('practical_indeterminism', 'Practical Indeterminism'),
            ('agentic_novelty', 'Agentic Novelty'),
            ('competitive_recursion', 'Competitive Recursion')
        ]

        latex = self._build_table_header(caption, label, 6)
        latex += r"\toprule" + "\n"
        latex += r"Uncertainty & \multicolumn{4}{c}{AI Tier} & AI Effect \\" + "\n"
        latex += r"\cmidrule(lr){2-5}" + "\n"
        latex += r"Dimension & None & Basic & Advanced & Premium & (Premium - None) \\" + "\n"
        latex += r"\midrule" + "\n"

        for dim_key, dim_label in dimensions:
            level_col = f"{dim_key}_level"

            if level_col not in uncertainty_df.columns:
                continue

            values = []
            for tier in tier_order:
                tier_data = uncertainty_df[uncertainty_df['ai_tier'] == tier] if 'ai_tier' in uncertainty_df.columns else uncertainty_df
                if len(tier_data) > 0 and level_col in tier_data.columns:
                    values.append(tier_data[level_col].mean())
                else:
                    values.append(np.nan)

            # Calculate effect (Premium - None)
            effect = values[3] - values[0] if len(values) == 4 else np.nan
            effect_str = f"+{effect:.3f}" if effect > 0 else f"{effect:.3f}"

            latex += f"{dim_label} & "
            latex += " & ".join([_format_number(v, 3) for v in values])
            latex += f" & {effect_str} \\\\\n"

        latex += r"\bottomrule" + "\n"
        latex += self._build_table_footer([
            "Note: Values represent mean uncertainty levels across all rounds.",
            "AI Effect shows the difference between Premium and None tiers.",
            "Positive effects indicate AI increases uncertainty in that dimension."
        ])

        output_path = self.output_dir / filename
        output_path.write_text(latex)
        self.tables_generated.append(filename)
        return output_path

    def generate_survival_analysis_table(
        self,
        survival_results: Dict[str, Any],
        filename: str = "table_survival_analysis.tex",
        caption: str = "Cox Proportional Hazards Model: AI Effects on Survival",
        label: str = "tab:survival"
    ) -> Path:
        """
        Generate Table 4: Survival analysis results.
        """
        latex = self._build_table_header(caption, label, 5)
        latex += r"\toprule" + "\n"
        latex += r"Variable & Hazard Ratio & 95\% CI & $z$ & $p$-value \\" + "\n"
        latex += r"\midrule" + "\n"

        # AI tier effects (reference: none)
        tiers = ['basic', 'advanced', 'premium']
        for tier in tiers:
            hr = survival_results.get(f'hr_{tier}', np.nan)
            ci_lower = survival_results.get(f'ci_lower_{tier}', np.nan)
            ci_upper = survival_results.get(f'ci_upper_{tier}', np.nan)
            z = survival_results.get(f'z_{tier}', np.nan)
            p = survival_results.get(f'p_{tier}', np.nan)

            latex += f"AI Tier: {tier.capitalize()} & {_format_number(hr, 3)} & "
            latex += f"[{_format_number(ci_lower, 3)}, {_format_number(ci_upper, 3)}] & "
            latex += f"{_format_number(z, 2)} & {_format_p_value(p)} \\\\\n"

        latex += r"\midrule" + "\n"

        # Model fit statistics
        n = survival_results.get('n_observations', 0)
        events = survival_results.get('n_events', 0)
        concordance = survival_results.get('concordance', np.nan)

        latex += f"\\multicolumn{{5}}{{l}}{{Model Statistics}} \\\\\n"
        latex += f"\\quad $N$ (observations) & \\multicolumn{{4}}{{l}}{{{n:,}}} \\\\\n"
        latex += f"\\quad Events (failures) & \\multicolumn{{4}}{{l}}{{{events:,}}} \\\\\n"
        latex += f"\\quad Concordance & \\multicolumn{{4}}{{l}}{{{_format_number(concordance, 3)}}} \\\\\n"

        latex += r"\bottomrule" + "\n"
        latex += self._build_table_footer([
            "Note: Reference category is AI Tier = None.",
            "Hazard ratios < 1 indicate reduced failure risk (protective effect).",
            "†$p < .10$; *$p < .05$; **$p < .01$; ***$p < .001$"
        ])

        output_path = self.output_dir / filename
        output_path.write_text(latex)
        self.tables_generated.append(filename)
        return output_path

    def generate_pairwise_comparisons_table(
        self,
        pairwise_results: List[Dict[str, Any]],
        filename: str = "table_pairwise_comparisons.tex",
        caption: str = "Pairwise Comparisons of AI Tiers",
        label: str = "tab:pairwise"
    ) -> Path:
        """
        Generate Table 5: Pairwise Mann-Whitney U tests with corrections.
        """
        latex = self._build_table_header(caption, label, 6)
        latex += r"\toprule" + "\n"
        latex += r"Comparison & $U$ & $p$ (raw) & $p$ (adj.) & Cohen's $d$ & Interpretation \\" + "\n"
        latex += r"\midrule" + "\n"

        for result in pairwise_results:
            comparison = _escape_latex(result.get('comparison', ''))
            u_stat = result.get('u_statistic', np.nan)
            p_raw = result.get('p_value_raw', np.nan)
            p_adj = result.get('p_value_adjusted', np.nan)
            cohens_d = result.get('cohens_d', np.nan)
            interp = result.get('interpretation', '')

            latex += f"{comparison} & {_format_number(u_stat, 0)} & "
            latex += f"{_format_p_value(p_raw, include_stars=False)} & "
            latex += f"{_format_p_value(p_adj)} & "
            latex += f"{_format_number(cohens_d, 2)} & {interp} \\\\\n"

        latex += r"\bottomrule" + "\n"
        latex += self._build_table_footer([
            "Note: $p$-values adjusted using Benjamini-Hochberg FDR correction.",
            "Effect size interpretation: |$d$| < 0.2 = negligible, 0.2-0.5 = small,",
            "0.5-0.8 = medium, > 0.8 = large.",
            "†$p < .10$; *$p < .05$; **$p < .01$; ***$p < .001$"
        ])

        output_path = self.output_dir / filename
        output_path.write_text(latex)
        self.tables_generated.append(filename)
        return output_path

    def generate_construct_operationalization_table(
        self,
        filename: str = "table_constructs.tex",
        caption: str = "Operationalization of Knightian Uncertainty Constructs",
        label: str = "tab:constructs"
    ) -> Path:
        """
        Generate Table 6: Mapping theoretical constructs to simulation mechanisms.
        """
        constructs = [
            {
                'construct': 'Actor Ignorance',
                'definition': 'Information gaps and knowledge deficits that limit decision quality',
                'mechanism': 'Knowledge decay, incomplete market signals, bounded information processing',
                'parameters': 'KNOWLEDGE\\_DECAY\\_RATE, INFO\\_QUALITY by AI tier',
                'measures': 'knowledge\\_level, info\\_gap, decision\\_confidence'
            },
            {
                'construct': 'Practical Indeterminism',
                'definition': 'Execution uncertainty and timing criticality in implementation',
                'mechanism': 'Investment maturation variance, market timing sensitivity',
                'parameters': 'INVESTMENT\\_MATURATION\\_TIME, execution\\_volatility',
                'measures': 'timing\\_accuracy, execution\\_success\\_rate'
            },
            {
                'construct': 'Agentic Novelty',
                'definition': 'Unpredictability arising from genuine innovation and creativity',
                'mechanism': 'Innovation attempts with stochastic outcomes, disruption events',
                'parameters': 'INNOVATION\\_PROBABILITY, novelty\\_premium',
                'measures': 'innovation\\_rate, novelty\\_score, disruption\\_impact'
            },
            {
                'construct': 'Competitive Recursion',
                'definition': 'Strategic interdependence where actions affect others\' option spaces',
                'mechanism': 'Herding penalties, strategic opacity, crowding effects',
                'parameters': 'COMPETITION\\_INTENSITY, herding\\_penalty',
                'measures': 'combo\\_rate, strategy\\_uniqueness, crowding\\_penalty'
            }
        ]

        latex = self._build_table_header(caption, label, 4, longtable=True)
        latex += r"\toprule" + "\n"
        latex += r"Construct & Definition & Simulation Mechanism & Key Measures \\" + "\n"
        latex += r"\midrule" + "\n"
        latex += r"\endhead" + "\n"  # For longtable

        for c in constructs:
            latex += f"\\textbf{{{c['construct']}}} & {c['definition']} & "
            latex += f"{c['mechanism']} & {c['measures']} \\\\\n"
            latex += r"\addlinespace" + "\n"

        latex += r"\bottomrule" + "\n"
        latex += r"\end{longtable}" + "\n"

        output_path = self.output_dir / filename
        output_path.write_text(latex)
        self.tables_generated.append(filename)
        return output_path

    def _build_table_header(
        self,
        caption: str,
        label: str,
        n_cols: int,
        longtable: bool = False
    ) -> str:
        """Build LaTeX table header."""
        if longtable:
            header = r"\begin{longtable}{" + "l" + "p{3cm}" * (n_cols - 1) + "}\n"
            header += f"\\caption{{{caption}}} \\label{{{label}}} \\\\\n"
        else:
            header = r"\begin{table}[htbp]" + "\n"
            header += r"\centering" + "\n"
            header += f"\\caption{{{caption}}}\n"
            header += f"\\label{{{label}}}\n"
            header += r"\small" + "\n"
            header += r"\begin{tabular}{" + "l" * n_cols + "}\n"
        return header

    def _build_table_footer(self, notes: List[str]) -> str:
        """Build LaTeX table footer with notes."""
        footer = r"\end{tabular}" + "\n"
        footer += r"\begin{tablenotes}" + "\n"
        footer += r"\small" + "\n"
        for note in notes:
            footer += f"\\item {note}\n"
        footer += r"\end{tablenotes}" + "\n"
        footer += r"\end{table}" + "\n"
        return footer


# =============================================================================
# UNCERTAINTY TRANSFORMATION ANALYSIS
# =============================================================================

@dataclass
class UncertaintyTransformationResult:
    """Results from uncertainty transformation analysis."""
    dimension: str
    ai_tier: str
    mean_level: float
    std_level: float
    mean_delta: float  # AI-induced change
    trend_slope: float  # Over time
    perception_gap: float  # Objective - perceived
    n_observations: int


class UncertaintyTransformationAnalyzer:
    """
    Analyze how AI transforms uncertainty across the four Knightian dimensions.

    This directly addresses the paper's core question: Does AI reduce uncertainty
    or transform it into different forms?
    """

    DIMENSIONS = [
        'actor_ignorance',
        'practical_indeterminism',
        'agentic_novelty',
        'competitive_recursion'
    ]

    def __init__(
        self,
        uncertainty_df: pd.DataFrame,
        decision_df: Optional[pd.DataFrame] = None
    ):
        self.uncertainty_df = uncertainty_df
        self.decision_df = decision_df
        self.results: List[UncertaintyTransformationResult] = []

    def analyze_all_dimensions(self) -> pd.DataFrame:
        """
        Comprehensive analysis of uncertainty transformation across all dimensions.

        Returns
        -------
        pd.DataFrame
            Summary statistics for each dimension × AI tier combination.
        """
        tier_order = ['none', 'basic', 'advanced', 'premium']
        results = []

        for dim in self.DIMENSIONS:
            level_col = f"{dim}_level"
            delta_col = f"ai_{dim.split('_')[0]}_delta" if dim != 'competitive_recursion' else 'ai_recursion_delta'

            for tier in tier_order:
                if 'ai_tier' in self.uncertainty_df.columns:
                    tier_data = self.uncertainty_df[self.uncertainty_df['ai_tier'] == tier]
                else:
                    tier_data = self.uncertainty_df

                if len(tier_data) == 0:
                    continue

                # Basic statistics
                mean_level = tier_data[level_col].mean() if level_col in tier_data.columns else np.nan
                std_level = tier_data[level_col].std() if level_col in tier_data.columns else np.nan

                # AI delta (transformation effect)
                mean_delta = tier_data[delta_col].mean() if delta_col in tier_data.columns else np.nan

                # Temporal trend
                if 'round' in tier_data.columns and level_col in tier_data.columns:
                    try:
                        slope, _, _, _, _ = stats.linregress(
                            tier_data['round'].astype(float),
                            tier_data[level_col].astype(float)
                        )
                    except:
                        slope = np.nan
                else:
                    slope = np.nan

                # Perception gap (if decision data available)
                perception_gap = self._compute_perception_gap(dim, tier)

                result = UncertaintyTransformationResult(
                    dimension=dim,
                    ai_tier=tier,
                    mean_level=mean_level,
                    std_level=std_level,
                    mean_delta=mean_delta,
                    trend_slope=slope,
                    perception_gap=perception_gap,
                    n_observations=len(tier_data)
                )
                self.results.append(result)
                results.append(result.__dict__)

        return pd.DataFrame(results)

    def compute_paradox_effects(self) -> Dict[str, Dict[str, float]]:
        """
        Compute the "Paradox of Future Knowledge" effects.

        For each dimension, measure how AI simultaneously reduces some aspects
        of uncertainty while amplifying others.

        Returns
        -------
        Dict with dimension -> {reduction, amplification, net_effect}
        """
        paradox = {}

        for dim in self.DIMENSIONS:
            level_col = f"{dim}_level"

            if 'ai_tier' not in self.uncertainty_df.columns:
                continue

            # Compare no-AI vs premium AI
            no_ai = self.uncertainty_df[self.uncertainty_df['ai_tier'] == 'none']
            premium = self.uncertainty_df[self.uncertainty_df['ai_tier'] == 'premium']

            if len(no_ai) == 0 or len(premium) == 0:
                continue

            no_ai_mean = no_ai[level_col].mean() if level_col in no_ai.columns else np.nan
            premium_mean = premium[level_col].mean() if level_col in premium.columns else np.nan

            # Net effect
            net_effect = premium_mean - no_ai_mean

            # Decompose into reduction/amplification phases
            # (using early vs late rounds)
            if 'round' in self.uncertainty_df.columns:
                max_round = self.uncertainty_df['round'].max()
                early = self.uncertainty_df['round'] <= max_round * 0.3
                late = self.uncertainty_df['round'] >= max_round * 0.7

                early_effect = (
                    premium[early & (premium.index.isin(premium.index))][level_col].mean() -
                    no_ai[early & (no_ai.index.isin(no_ai.index))][level_col].mean()
                ) if level_col in premium.columns else np.nan

                late_effect = (
                    premium[late & (premium.index.isin(premium.index))][level_col].mean() -
                    no_ai[late & (no_ai.index.isin(no_ai.index))][level_col].mean()
                ) if level_col in premium.columns else np.nan
            else:
                early_effect = np.nan
                late_effect = np.nan

            paradox[dim] = {
                'no_ai_level': no_ai_mean,
                'premium_level': premium_mean,
                'net_effect': net_effect,
                'early_effect': early_effect,
                'late_effect': late_effect,
                'effect_reversal': late_effect - early_effect if not (pd.isna(early_effect) or pd.isna(late_effect)) else np.nan
            }

        return paradox

    def compute_effect_sizes(self) -> pd.DataFrame:
        """
        Compute Cohen's d effect sizes for AI tier comparisons on each dimension.
        """
        results = []
        tier_pairs = [
            ('none', 'basic'),
            ('none', 'advanced'),
            ('none', 'premium'),
            ('basic', 'advanced'),
            ('basic', 'premium'),
            ('advanced', 'premium')
        ]

        for dim in self.DIMENSIONS:
            level_col = f"{dim}_level"

            if level_col not in self.uncertainty_df.columns:
                continue

            for tier1, tier2 in tier_pairs:
                if 'ai_tier' not in self.uncertainty_df.columns:
                    continue

                data1 = self.uncertainty_df[self.uncertainty_df['ai_tier'] == tier1][level_col].dropna()
                data2 = self.uncertainty_df[self.uncertainty_df['ai_tier'] == tier2][level_col].dropna()

                if len(data1) < 2 or len(data2) < 2:
                    continue

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(data1) - 1) * data1.std()**2 + (len(data2) - 1) * data2.std()**2) /
                    (len(data1) + len(data2) - 2)
                )

                if pooled_std > 0:
                    cohens_d = (data2.mean() - data1.mean()) / pooled_std
                else:
                    cohens_d = 0.0

                # Bootstrap CI for effect size
                ci_lower, ci_upper = self._bootstrap_effect_size_ci(data1, data2)

                # Interpretation
                abs_d = abs(cohens_d)
                if abs_d < 0.2:
                    interp = 'Negligible'
                elif abs_d < 0.5:
                    interp = 'Small'
                elif abs_d < 0.8:
                    interp = 'Medium'
                else:
                    interp = 'Large'

                results.append({
                    'dimension': dim,
                    'comparison': f"{tier1} vs {tier2}",
                    'tier1': tier1,
                    'tier2': tier2,
                    'cohens_d': cohens_d,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'interpretation': interp,
                    'n1': len(data1),
                    'n2': len(data2)
                })

        return pd.DataFrame(results)

    def _compute_perception_gap(self, dimension: str, tier: str) -> float:
        """Compute gap between objective uncertainty and agent perception."""
        if self.decision_df is None:
            return np.nan

        # Perception columns in decision_df
        perceived_col = f"{dimension}_perception"

        if perceived_col not in self.decision_df.columns:
            return np.nan

        if 'ai_tier' not in self.decision_df.columns:
            return np.nan

        tier_decisions = self.decision_df[self.decision_df['ai_tier'] == tier]
        if len(tier_decisions) == 0:
            return np.nan

        # Get objective level from uncertainty_df
        level_col = f"{dimension}_level"
        if 'ai_tier' in self.uncertainty_df.columns:
            objective = self.uncertainty_df[self.uncertainty_df['ai_tier'] == tier][level_col].mean()
        else:
            objective = self.uncertainty_df[level_col].mean()

        perceived = tier_decisions[perceived_col].mean()

        return objective - perceived

    def _bootstrap_effect_size_ci(
        self,
        data1: pd.Series,
        data2: pd.Series,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for Cohen's d."""
        d_samples = []

        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1.values, size=len(data1), replace=True)
            sample2 = np.random.choice(data2.values, size=len(data2), replace=True)

            pooled_std = np.sqrt(
                ((len(sample1) - 1) * sample1.std()**2 + (len(sample2) - 1) * sample2.std()**2) /
                (len(sample1) + len(sample2) - 2)
            )

            if pooled_std > 0:
                d = (sample2.mean() - sample1.mean()) / pooled_std
                d_samples.append(d)

        if len(d_samples) == 0:
            return (np.nan, np.nan)

        return (
            np.percentile(d_samples, 100 * alpha / 2),
            np.percentile(d_samples, 100 * (1 - alpha / 2))
        )


# =============================================================================
# PUBLICATION FIGURE GENERATOR
# =============================================================================

class PublicationFigureGenerator:
    """Generate publication-quality figures for journal submission."""

    # Journal-quality settings
    FIGURE_DPI = 300
    FONT_SIZE = 12
    TITLE_SIZE = 14
    TICK_SIZE = 10

    # Color palette for AI tiers
    TIER_COLORS = {
        'none': '#808080',      # Gray
        'basic': '#3498db',     # Blue
        'advanced': '#e67e22',  # Orange
        'premium': '#e74c3c'    # Red
    }

    TIER_LABELS = {
        'none': 'No AI',
        'basic': 'Basic AI',
        'advanced': 'Advanced AI',
        'premium': 'Premium AI'
    }

    def __init__(self, output_dir: Union[str, Path]):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib and seaborn required for figure generation")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_generated: List[str] = []

        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': self.FONT_SIZE,
            'axes.titlesize': self.TITLE_SIZE,
            'axes.labelsize': self.FONT_SIZE,
            'xtick.labelsize': self.TICK_SIZE,
            'ytick.labelsize': self.TICK_SIZE,
            'legend.fontsize': self.TICK_SIZE,
            'figure.dpi': 150,
            'savefig.dpi': self.FIGURE_DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

    def generate_survival_by_tier(
        self,
        agent_df: pd.DataFrame,
        filename: str = "figure_survival_by_tier.pdf"
    ) -> Path:
        """
        Generate Figure 1: Survival rates by AI tier with confidence intervals.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        tier_order = ['none', 'basic', 'advanced', 'premium']
        survival_stats = []

        for tier in tier_order:
            tier_data = agent_df[agent_df['ai_tier'] == tier]
            if len(tier_data) == 0:
                continue

            survived = tier_data['survived'].values if 'survived' in tier_data.columns else np.array([])
            if len(survived) == 0:
                continue

            rate = survived.mean()
            # Wilson score interval
            n = len(survived)
            z = 1.96
            denominator = 1 + z**2/n
            centre = (rate + z**2/(2*n)) / denominator
            margin = z * np.sqrt((rate*(1-rate) + z**2/(4*n))/n) / denominator

            survival_stats.append({
                'tier': tier,
                'rate': rate,
                'ci_lower': max(0, centre - margin),
                'ci_upper': min(1, centre + margin),
                'n': n
            })

        if not survival_stats:
            plt.close()
            return None

        x_pos = np.arange(len(survival_stats))
        rates = [s['rate'] for s in survival_stats]
        errors_lower = [s['rate'] - s['ci_lower'] for s in survival_stats]
        errors_upper = [s['ci_upper'] - s['rate'] for s in survival_stats]
        colors = [self.TIER_COLORS[s['tier']] for s in survival_stats]

        bars = ax.bar(x_pos, rates, color=colors, edgecolor='black', linewidth=1)
        ax.errorbar(x_pos, rates, yerr=[errors_lower, errors_upper],
                   fmt='none', color='black', capsize=5, capthick=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.TIER_LABELS[s['tier']] for s in survival_stats])
        ax.set_ylabel('Survival Rate')
        ax.set_xlabel('AI Tier')
        ax.set_ylim(0, 1)
        ax.set_title('Survival Rates by AI Tier')

        # Add sample sizes
        for i, s in enumerate(survival_stats):
            ax.annotate(f"n={s['n']}", xy=(i, 0.02), ha='center', fontsize=9)

        output_path = self.output_dir / filename
        fig.savefig(output_path)
        plt.close()
        self.figures_generated.append(filename)
        return output_path

    def generate_effect_size_forest_plot(
        self,
        effect_sizes_df: pd.DataFrame,
        filename: str = "figure_effect_sizes.pdf"
    ) -> Path:
        """
        Generate Figure 2: Forest plot of effect sizes by uncertainty dimension.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Filter to key comparisons (none vs others)
        key_comparisons = effect_sizes_df[effect_sizes_df['tier1'] == 'none'].copy()

        if len(key_comparisons) == 0:
            plt.close()
            return None

        # Organize by dimension
        dimensions = key_comparisons['dimension'].unique()
        y_positions = []
        current_y = 0

        for dim in dimensions:
            dim_data = key_comparisons[key_comparisons['dimension'] == dim]
            for _, row in dim_data.iterrows():
                y_positions.append(current_y)

                # Plot point estimate
                color = self.TIER_COLORS.get(row['tier2'], 'black')
                ax.plot(row['cohens_d'], current_y, 'o', color=color, markersize=8)

                # Plot CI
                ax.hlines(current_y, row['ci_lower'], row['ci_upper'],
                         colors=color, linewidth=2)

                # Label
                label = f"{dim.replace('_', ' ').title()}: {row['tier2'].capitalize()}"
                ax.annotate(label, xy=(ax.get_xlim()[0], current_y),
                           xytext=(-5, 0), textcoords='offset points',
                           ha='right', va='center', fontsize=9)

                current_y += 1

            current_y += 0.5  # Gap between dimensions

        # Add vertical reference line at 0
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        # Add effect size interpretation zones
        ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligible')
        ax.axvspan(0.2, 0.5, alpha=0.1, color='blue', label='Small')
        ax.axvspan(-0.5, -0.2, alpha=0.1, color='blue')
        ax.axvspan(0.5, 0.8, alpha=0.1, color='orange', label='Medium')
        ax.axvspan(-0.8, -0.5, alpha=0.1, color='orange')

        ax.set_xlabel("Cohen's d Effect Size")
        ax.set_title("Effect of AI Tier on Uncertainty Dimensions")
        ax.set_yticks([])

        # Legend
        handles = [
            mpatches.Patch(color=self.TIER_COLORS['basic'], label='Basic AI'),
            mpatches.Patch(color=self.TIER_COLORS['advanced'], label='Advanced AI'),
            mpatches.Patch(color=self.TIER_COLORS['premium'], label='Premium AI'),
        ]
        ax.legend(handles=handles, loc='lower right')

        output_path = self.output_dir / filename
        fig.savefig(output_path)
        plt.close()
        self.figures_generated.append(filename)
        return output_path

    def generate_uncertainty_dynamics(
        self,
        uncertainty_df: pd.DataFrame,
        filename: str = "figure_uncertainty_dynamics.pdf"
    ) -> Path:
        """
        Generate Figure 3: 4-panel figure showing uncertainty dynamics over time.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        dimensions = [
            ('actor_ignorance_level', 'Actor Ignorance'),
            ('practical_indeterminism_level', 'Practical Indeterminism'),
            ('agentic_novelty_level', 'Agentic Novelty'),
            ('competitive_recursion_level', 'Competitive Recursion')
        ]

        tier_order = ['none', 'basic', 'advanced', 'premium']

        for ax, (col, title) in zip(axes, dimensions):
            if col not in uncertainty_df.columns or 'round' not in uncertainty_df.columns:
                ax.set_visible(False)
                continue

            for tier in tier_order:
                if 'ai_tier' not in uncertainty_df.columns:
                    tier_data = uncertainty_df
                else:
                    tier_data = uncertainty_df[uncertainty_df['ai_tier'] == tier]

                if len(tier_data) == 0:
                    continue

                # Group by round and compute mean
                round_means = tier_data.groupby('round')[col].mean()

                ax.plot(round_means.index, round_means.values,
                       label=self.TIER_LABELS.get(tier, tier),
                       color=self.TIER_COLORS.get(tier, 'black'),
                       linewidth=2)

            ax.set_xlabel('Round')
            ax.set_ylabel('Uncertainty Level')
            ax.set_title(title)
            ax.legend(loc='best', fontsize=8)

        plt.tight_layout()

        output_path = self.output_dir / filename
        fig.savefig(output_path)
        plt.close()
        self.figures_generated.append(filename)
        return output_path

    def generate_paradox_figure(
        self,
        paradox_effects: Dict[str, Dict[str, float]],
        filename: str = "figure_paradox_effects.pdf"
    ) -> Path:
        """
        Generate Figure 4: The Paradox of Future Knowledge visualization.

        Shows how AI simultaneously reduces and amplifies different aspects
        of uncertainty.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        dimensions = list(paradox_effects.keys())
        x_pos = np.arange(len(dimensions))

        # Extract effects
        no_ai_levels = [paradox_effects[d]['no_ai_level'] for d in dimensions]
        premium_levels = [paradox_effects[d]['premium_level'] for d in dimensions]
        net_effects = [paradox_effects[d]['net_effect'] for d in dimensions]

        width = 0.35

        bars1 = ax.bar(x_pos - width/2, no_ai_levels, width,
                       label='No AI', color=self.TIER_COLORS['none'], edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, premium_levels, width,
                       label='Premium AI', color=self.TIER_COLORS['premium'], edgecolor='black')

        # Add net effect arrows
        for i, (x, net) in enumerate(zip(x_pos, net_effects)):
            if pd.isna(net):
                continue

            arrow_color = 'green' if net < 0 else 'red'
            direction = '↓' if net < 0 else '↑'
            ax.annotate(f"{direction} {abs(net):.2f}",
                       xy=(x + 0.5, max(no_ai_levels[i], premium_levels[i]) + 0.02),
                       ha='center', fontsize=10, color=arrow_color, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([d.replace('_', ' ').title() for d in dimensions], rotation=15)
        ax.set_ylabel('Uncertainty Level')
        ax.set_title('The Paradox of Future Knowledge: AI Effects on Uncertainty')
        ax.legend(loc='upper right')

        # Add explanatory note
        ax.annotate('Green arrows = AI reduces uncertainty\nRed arrows = AI amplifies uncertainty',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        output_path = self.output_dir / filename
        fig.savefig(output_path)
        plt.close()
        self.figures_generated.append(filename)
        return output_path

    def generate_kaplan_meier_curves(
        self,
        agent_df: pd.DataFrame,
        filename: str = "figure_kaplan_meier.pdf"
    ) -> Path:
        """
        Generate Figure 5: Kaplan-Meier survival curves by AI tier.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        tier_order = ['none', 'basic', 'advanced', 'premium']

        for tier in tier_order:
            tier_data = agent_df[agent_df['ai_tier'] == tier]

            if len(tier_data) == 0:
                continue

            # Get survival time (rounds survived) and event indicator
            if 'rounds_survived' not in tier_data.columns:
                if 'final_round' in tier_data.columns:
                    times = tier_data['final_round'].values
                else:
                    continue
            else:
                times = tier_data['rounds_survived'].values

            if 'survived' in tier_data.columns:
                # Event = 1 means failure (did not survive)
                events = (~tier_data['survived']).astype(int).values
            else:
                events = np.ones(len(times))

            # Compute Kaplan-Meier estimate
            unique_times = np.sort(np.unique(times))
            survival_probs = []
            current_prob = 1.0

            n_at_risk = len(times)
            for t in unique_times:
                n_events = np.sum((times == t) & (events == 1))
                if n_at_risk > 0:
                    current_prob *= (1 - n_events / n_at_risk)
                n_at_risk -= np.sum(times == t)
                survival_probs.append(current_prob)

            # Step plot
            ax.step(unique_times, survival_probs, where='post',
                   label=self.TIER_LABELS[tier],
                   color=self.TIER_COLORS[tier],
                   linewidth=2)

        ax.set_xlabel('Round')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Kaplan-Meier Survival Curves by AI Tier')
        ax.legend(loc='lower left')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, None)

        output_path = self.output_dir / filename
        fig.savefig(output_path)
        plt.close()
        self.figures_generated.append(filename)
        return output_path


# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

class PDFReportGenerator:
    """
    Generate a comprehensive PDF report combining all tables and figures.

    Uses LaTeX compilation to produce publication-quality output.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        title: str = "Glimpse ABM: Empirical Analysis Results",
        author: str = "",
        tables: List[str] = None,
        figures: List[str] = None,
        include_methods: bool = True
    ) -> Optional[Path]:
        """
        Generate a complete PDF report.

        Parameters
        ----------
        title : str
            Report title
        author : str
            Author name(s)
        tables : List[str]
            List of .tex table files to include
        figures : List[str]
            List of figure files to include
        include_methods : bool
            Whether to include a methods section

        Returns
        -------
        Path to generated PDF, or None if compilation failed
        """
        tables = tables or []
        figures = figures or []

        # Build LaTeX document
        latex = self._build_preamble(title, author)
        latex += r"\begin{document}" + "\n"
        latex += r"\maketitle" + "\n"

        # Abstract
        latex += self._build_abstract()

        # Methods section
        if include_methods:
            latex += self._build_methods_section()

        # Results section
        latex += self._build_results_section(tables, figures)

        # End document
        latex += r"\end{document}" + "\n"

        # Write and compile
        tex_path = self.output_dir / "analysis_report.tex"
        tex_path.write_text(latex)

        return self._compile_latex(tex_path)

    def _build_preamble(self, title: str, author: str) -> str:
        """Build LaTeX preamble."""
        return f"""\\documentclass[12pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{hyperref}}
\\usepackage{{longtable}}
\\usepackage{{threeparttable}}
\\usepackage{{caption}}

\\title{{{title}}}
\\author{{{author}}}
\\date{{\\today}}

"""

    def _build_abstract(self) -> str:
        """Build abstract section."""
        return r"""
\begin{abstract}
This report presents empirical analysis results from the Glimpse Agent-Based Model (ABM),
examining how artificial intelligence affects entrepreneurial decision-making under
Knightian uncertainty. The analysis tests propositions from Townsend et al. (2025)
regarding the ``paradox of future knowledge'' -- the finding that AI tools may
simultaneously reduce some aspects of uncertainty while amplifying others.
Key findings include survival rate differentials across AI tiers, effect sizes
for uncertainty dimension transformations, and causal estimates of AI adoption effects.
\end{abstract}

"""

    def _build_methods_section(self) -> str:
        """Build methods section."""
        return r"""
\section{Methods}

\subsection{Agent-Based Model}
The Glimpse ABM simulates entrepreneurial agents making investment and innovation
decisions under varying levels of AI augmentation. Agents are assigned to one of
four AI tiers: None, Basic, Advanced, or Premium, each providing different levels
of information quality and decision support.

\subsection{Uncertainty Dimensions}
Following Townsend et al. (2025), we operationalize Knightian uncertainty across
four dimensions:
\begin{itemize}
    \item \textbf{Actor Ignorance}: Information gaps and knowledge deficits
    \item \textbf{Practical Indeterminism}: Execution uncertainty and timing criticality
    \item \textbf{Agentic Novelty}: Unpredictability from genuine innovation
    \item \textbf{Competitive Recursion}: Strategic interdependence effects
\end{itemize}

\subsection{Statistical Analysis}
All analyses employ appropriate statistical methods for the data structure:
\begin{itemize}
    \item Non-parametric tests (Kruskal-Wallis, Mann-Whitney U) for ordinal comparisons
    \item Effect sizes (Cohen's $d$) with bootstrap confidence intervals
    \item Multiple comparison corrections (Benjamini-Hochberg FDR)
    \item Run-level clustering for proper inference
\end{itemize}

"""

    def _build_results_section(self, tables: List[str], figures: List[str]) -> str:
        """Build results section with tables and figures."""
        results = r"\section{Results}" + "\n\n"

        # Include tables
        if tables:
            results += r"\subsection{Summary Statistics and Tests}" + "\n\n"
            for table_file in tables:
                if Path(table_file).exists():
                    results += f"\\input{{{table_file}}}\n\n"
                else:
                    # Try relative path
                    results += f"\\input{{{Path(table_file).name}}}\n\n"

        # Include figures
        if figures:
            results += r"\subsection{Visualizations}" + "\n\n"
            for fig_file in figures:
                fig_path = Path(fig_file)
                results += f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{fig_path.name}}}
\\caption{{{fig_path.stem.replace('_', ' ').title()}}}
\\end{{figure}}

"""

        return results

    def _compile_latex(self, tex_path: Path) -> Optional[Path]:
        """Compile LaTeX to PDF."""
        try:
            # Try pdflatex
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_path.name],
                cwd=self.output_dir,
                capture_output=True,
                timeout=60
            )

            # Run twice for references
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_path.name],
                cwd=self.output_dir,
                capture_output=True,
                timeout=60
            )

            pdf_path = self.output_dir / tex_path.with_suffix('.pdf').name
            if pdf_path.exists():
                print(f"PDF generated: {pdf_path}")
                return pdf_path
            else:
                print("PDF compilation may have failed. Check .log file for details.")
                return None

        except FileNotFoundError:
            print("pdflatex not found. LaTeX files generated but not compiled.")
            print(f"To compile manually: cd {self.output_dir} && pdflatex {tex_path.name}")
            return None
        except subprocess.TimeoutExpired:
            print("LaTeX compilation timed out.")
            return None


# =============================================================================
# UNIFIED PUBLICATION PIPELINE
# =============================================================================

class PublicationPipeline:
    """
    Unified pipeline for generating all publication outputs.

    Usage
    -----
    >>> pipeline = PublicationPipeline(results_dir, output_dir)
    >>> pipeline.run_full_pipeline()
    """

    def __init__(
        self,
        results_dir: Union[str, Path],
        output_dir: Union[str, Path],
        agent_df: Optional[pd.DataFrame] = None,
        decision_df: Optional[pd.DataFrame] = None,
        uncertainty_df: Optional[pd.DataFrame] = None,
        matured_df: Optional[pd.DataFrame] = None
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data frames (load from results_dir if not provided)
        self.agent_df = agent_df
        self.decision_df = decision_df
        self.uncertainty_df = uncertainty_df
        self.matured_df = matured_df

        # Component generators
        self.table_generator = LaTeXTableGenerator(self.output_dir / 'tables')
        self.uncertainty_analyzer = None
        self.figure_generator = None
        self.pdf_generator = PDFReportGenerator(self.output_dir)

        # Track outputs
        self.tables_generated: List[Path] = []
        self.figures_generated: List[Path] = []

    def load_data(self) -> bool:
        """Load data from results directory if not already provided."""
        try:
            import pickle

            # Try to load compiled dataframes
            compiled_path = self.results_dir / 'compiled_data.pkl'
            if compiled_path.exists():
                with open(compiled_path, 'rb') as f:
                    data = pickle.load(f)
                    self.agent_df = data.get('agent_df', self.agent_df)
                    self.decision_df = data.get('decision_df', self.decision_df)
                    self.uncertainty_df = data.get('uncertainty_df', self.uncertainty_df)
                    self.matured_df = data.get('matured_df', self.matured_df)
                return True

            # Try CSV files
            for name, attr in [
                ('agents', 'agent_df'),
                ('decisions', 'decision_df'),
                ('uncertainty', 'uncertainty_df'),
                ('matured', 'matured_df')
            ]:
                csv_path = self.results_dir / f'{name}.csv'
                if csv_path.exists() and getattr(self, attr) is None:
                    setattr(self, attr, pd.read_csv(csv_path))

            return self.agent_df is not None

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def run_full_pipeline(
        self,
        include_tables: bool = True,
        include_figures: bool = True,
        include_pdf: bool = True,
        author: str = ""
    ) -> Dict[str, Any]:
        """
        Run the complete publication pipeline.

        Returns
        -------
        Dict with paths to all generated outputs
        """
        results = {
            'tables': [],
            'figures': [],
            'pdf': None,
            'uncertainty_analysis': None
        }

        # Ensure data is loaded
        if self.agent_df is None:
            if not self.load_data():
                print("Warning: Could not load data. Some outputs may be incomplete.")

        # Initialize components
        if self.uncertainty_df is not None:
            self.uncertainty_analyzer = UncertaintyTransformationAnalyzer(
                self.uncertainty_df,
                self.decision_df
            )

        if HAS_MATPLOTLIB:
            self.figure_generator = PublicationFigureGenerator(self.output_dir / 'figures')

        print("\n" + "="*60)
        print("PUBLICATION OUTPUT PIPELINE")
        print("="*60)

        # 1. Generate tables
        if include_tables:
            print("\n[1/4] Generating LaTeX tables...")
            results['tables'] = self._generate_all_tables()

        # 2. Run uncertainty analysis
        if self.uncertainty_analyzer is not None:
            print("\n[2/4] Running uncertainty transformation analysis...")
            results['uncertainty_analysis'] = self._run_uncertainty_analysis()

        # 3. Generate figures
        if include_figures and self.figure_generator is not None:
            print("\n[3/4] Generating publication figures...")
            results['figures'] = self._generate_all_figures()

        # 4. Generate PDF report
        if include_pdf:
            print("\n[4/4] Generating PDF report...")
            results['pdf'] = self._generate_pdf_report(author)

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"  Tables: {len(results['tables'])} files")
        print(f"  Figures: {len(results['figures'])} files")
        print(f"  PDF: {'Generated' if results['pdf'] else 'Not generated'}")

        return results

    def _generate_all_tables(self) -> List[Path]:
        """Generate all publication tables."""
        tables = []

        # Table 1: Descriptive statistics
        if self.agent_df is not None:
            try:
                path = self.table_generator.generate_descriptive_stats_table(self.agent_df)
                tables.append(path)
                print(f"  ✓ Descriptive statistics table")
            except Exception as e:
                print(f"  ✗ Descriptive statistics: {e}")

        # Table 3: Uncertainty transformation
        if self.uncertainty_df is not None:
            try:
                path = self.table_generator.generate_uncertainty_transformation_table(
                    self.uncertainty_df
                )
                tables.append(path)
                print(f"  ✓ Uncertainty transformation table")
            except Exception as e:
                print(f"  ✗ Uncertainty transformation: {e}")

        # Table 6: Construct operationalization
        try:
            path = self.table_generator.generate_construct_operationalization_table()
            tables.append(path)
            print(f"  ✓ Construct operationalization table")
        except Exception as e:
            print(f"  ✗ Construct operationalization: {e}")

        self.tables_generated = tables
        return tables

    def _run_uncertainty_analysis(self) -> Dict[str, Any]:
        """Run comprehensive uncertainty transformation analysis."""
        results = {}

        # Analyze all dimensions
        try:
            summary_df = self.uncertainty_analyzer.analyze_all_dimensions()
            summary_path = self.output_dir / 'uncertainty_transformation_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            results['summary'] = summary_path
            print(f"  ✓ Dimension analysis summary")
        except Exception as e:
            print(f"  ✗ Dimension analysis: {e}")

        # Compute paradox effects
        try:
            paradox = self.uncertainty_analyzer.compute_paradox_effects()
            results['paradox_effects'] = paradox
            print(f"  ✓ Paradox effects computed")
        except Exception as e:
            print(f"  ✗ Paradox effects: {e}")

        # Compute effect sizes
        try:
            effect_sizes = self.uncertainty_analyzer.compute_effect_sizes()
            effect_path = self.output_dir / 'uncertainty_effect_sizes.csv'
            effect_sizes.to_csv(effect_path, index=False)
            results['effect_sizes'] = effect_sizes
            results['effect_sizes_path'] = effect_path
            print(f"  ✓ Effect sizes computed")
        except Exception as e:
            print(f"  ✗ Effect sizes: {e}")

        return results

    def _generate_all_figures(self) -> List[Path]:
        """Generate all publication figures."""
        figures = []

        # Figure 1: Survival by tier
        if self.agent_df is not None:
            try:
                path = self.figure_generator.generate_survival_by_tier(self.agent_df)
                if path:
                    figures.append(path)
                    print(f"  ✓ Survival by tier figure")
            except Exception as e:
                print(f"  ✗ Survival by tier: {e}")

        # Figure 2: Effect size forest plot
        if hasattr(self, '_uncertainty_analysis_results'):
            effect_sizes = self._uncertainty_analysis_results.get('effect_sizes')
            if effect_sizes is not None:
                try:
                    path = self.figure_generator.generate_effect_size_forest_plot(effect_sizes)
                    if path:
                        figures.append(path)
                        print(f"  ✓ Effect size forest plot")
                except Exception as e:
                    print(f"  ✗ Effect size forest plot: {e}")

        # Figure 3: Uncertainty dynamics
        if self.uncertainty_df is not None:
            try:
                path = self.figure_generator.generate_uncertainty_dynamics(self.uncertainty_df)
                if path:
                    figures.append(path)
                    print(f"  ✓ Uncertainty dynamics figure")
            except Exception as e:
                print(f"  ✗ Uncertainty dynamics: {e}")

        # Figure 4: Paradox effects
        if self.uncertainty_analyzer is not None:
            try:
                paradox = self.uncertainty_analyzer.compute_paradox_effects()
                path = self.figure_generator.generate_paradox_figure(paradox)
                if path:
                    figures.append(path)
                    print(f"  ✓ Paradox effects figure")
            except Exception as e:
                print(f"  ✗ Paradox effects figure: {e}")

        # Figure 5: Kaplan-Meier curves
        if self.agent_df is not None:
            try:
                path = self.figure_generator.generate_kaplan_meier_curves(self.agent_df)
                if path:
                    figures.append(path)
                    print(f"  ✓ Kaplan-Meier curves")
            except Exception as e:
                print(f"  ✗ Kaplan-Meier curves: {e}")

        self.figures_generated = figures
        return figures

    def _generate_pdf_report(self, author: str = "") -> Optional[Path]:
        """Generate the final PDF report."""
        try:
            table_files = [str(t) for t in self.tables_generated]
            figure_files = [str(f) for f in self.figures_generated]

            pdf_path = self.pdf_generator.generate_report(
                title="Glimpse ABM: Empirical Analysis of AI Effects on Entrepreneurial Decision-Making",
                author=author,
                tables=table_files,
                figures=figure_files
            )

            if pdf_path:
                print(f"  ✓ PDF report generated: {pdf_path}")
            return pdf_path

        except Exception as e:
            print(f"  ✗ PDF generation: {e}")
            return None


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def run_publication_pipeline(
    results_dir: str,
    output_dir: str = None,
    author: str = ""
) -> Dict[str, Any]:
    """
    Convenience function to run the full publication pipeline.

    Parameters
    ----------
    results_dir : str
        Directory containing simulation results
    output_dir : str, optional
        Output directory for publication files (defaults to results_dir/publication)
    author : str
        Author name for the report

    Returns
    -------
    Dict with paths to all generated outputs
    """
    if output_dir is None:
        output_dir = str(Path(results_dir) / 'publication')

    pipeline = PublicationPipeline(results_dir, output_dir)
    return pipeline.run_full_pipeline(author=author)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication outputs from Glimpse ABM results")
    parser.add_argument("results_dir", help="Directory containing simulation results")
    parser.add_argument("--output-dir", "-o", help="Output directory for publication files")
    parser.add_argument("--author", "-a", default="", help="Author name for report")

    args = parser.parse_args()

    run_publication_pipeline(
        args.results_dir,
        args.output_dir,
        args.author
    )
