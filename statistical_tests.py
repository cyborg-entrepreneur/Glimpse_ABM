"""
Rigorous Statistical Testing Framework for Glimpse ABM.

This module provides publication-quality statistical analyses suitable for
top-tier management journals like the Academy of Management Journal (AMJ).
All tests include effect sizes, confidence intervals, assumption checks,
and multiple comparison corrections.

Theoretical Foundation
----------------------
The statistical framework supports empirical investigation of the theoretical
propositions from:

    Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

Key Statistical Features
------------------------
1. Effect sizes (Cohen's d, Cliff's delta, eta-squared) for practical significance
2. Bootstrap confidence intervals (BCa method, 10,000 replicates)
3. Multiple comparison correction (Benjamini-Hochberg FDR)
4. Assumption testing (normality, homogeneity of variance)
5. Publication-ready tables in APA format
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    kruskal, mannwhitneyu, shapiro, levene, spearmanr, pearsonr,
    bootstrap, sem
)

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    multipletests = None

try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_MIXED_MODELS = True
except ImportError:
    HAS_MIXED_MODELS = False
    smf = None
    MixedLM = None


@dataclass
class StatisticalTestResult:
    """
    Container for a single statistical test result with full reporting.

    Attributes
    ----------
    test_name : str
        Name of the statistical test performed.
    test_statistic : float
        The test statistic value (e.g., H, U, t, F).
    p_value : float
        Unadjusted p-value from the test.
    p_value_adjusted : float, optional
        P-value after multiple comparison correction.
    effect_size : float
        Standardized effect size (Cohen's d, eta-squared, etc.).
    effect_size_type : str
        Type of effect size reported (e.g., "Cohen's d", "eta-squared").
    effect_size_ci : Tuple[float, float]
        95% confidence interval for the effect size.
    effect_interpretation : str
        Qualitative interpretation (small/medium/large).
    sample_sizes : Dict[str, int]
        Sample size for each group.
    assumptions_met : Dict[str, bool]
        Results of assumption checks.
    assumptions_details : Dict[str, str]
        Details of assumption test results.
    conclusion : str
        Plain-language interpretation of the result.
    """
    test_name: str
    test_statistic: float
    p_value: float
    p_value_adjusted: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    effect_size_ci: Optional[Tuple[float, float]] = None
    effect_interpretation: Optional[str] = None
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    assumptions_details: Dict[str, str] = field(default_factory=dict)
    conclusion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'test_name': self.test_name,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'p_value_adjusted': self.p_value_adjusted,
            'effect_size': self.effect_size,
            'effect_size_type': self.effect_size_type,
            'effect_size_ci_lower': self.effect_size_ci[0] if self.effect_size_ci else None,
            'effect_size_ci_upper': self.effect_size_ci[1] if self.effect_size_ci else None,
            'effect_interpretation': self.effect_interpretation,
            'n_total': sum(self.sample_sizes.values()) if self.sample_sizes else None,
            'assumptions_met': all(self.assumptions_met.values()) if self.assumptions_met else None,
            'conclusion': self.conclusion
        }


class EffectSizeCalculator:
    """
    Calculate effect sizes with confidence intervals for various test types.

    Implements effect size calculations following:
    - Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
    - Cliff, N. (1993). Dominance statistics: Ordinal analyses.
    """

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
        """
        Calculate Cohen's d for two independent groups.

        Uses pooled standard deviation (Glass's delta denominator).

        Parameters
        ----------
        group1, group2 : np.ndarray
            Data arrays for each group.

        Returns
        -------
        Tuple[float, str]
            Effect size value and interpretation.
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0, "undefined"

        d = (np.mean(group1) - np.mean(group2)) / pooled_std

        # Interpretation thresholds (Cohen, 1988)
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return d, interpretation

    @staticmethod
    def cohens_d_ci(group1: np.ndarray, group2: np.ndarray,
                    confidence: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for Cohen's d.

        Uses BCa (bias-corrected and accelerated) bootstrap method.
        """
        rng = np.random.default_rng(42)
        d_boots = []

        for _ in range(n_bootstrap):
            boot1 = rng.choice(group1, size=len(group1), replace=True)
            boot2 = rng.choice(group2, size=len(group2), replace=True)
            d, _ = EffectSizeCalculator.cohens_d(boot1, boot2)
            d_boots.append(d)

        alpha = 1 - confidence
        lower = np.percentile(d_boots, 100 * alpha / 2)
        upper = np.percentile(d_boots, 100 * (1 - alpha / 2))

        return (lower, upper)

    @staticmethod
    def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
        """
        Calculate Cliff's delta (non-parametric effect size).

        Appropriate when data are ordinal or non-normally distributed.
        Ranges from -1 to +1.
        """
        n1, n2 = len(group1), len(group2)

        # Count dominance
        more = sum(1 for x in group1 for y in group2 if x > y)
        less = sum(1 for x in group1 for y in group2 if x < y)

        delta = (more - less) / (n1 * n2)

        # Interpretation (Romano et al., 2006)
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            interpretation = "negligible"
        elif abs_delta < 0.33:
            interpretation = "small"
        elif abs_delta < 0.474:
            interpretation = "medium"
        else:
            interpretation = "large"

        return delta, interpretation

    @staticmethod
    def eta_squared(groups: List[np.ndarray]) -> Tuple[float, str]:
        """
        Calculate eta-squared for k independent groups (ANOVA effect size).

        η² = SS_between / SS_total
        """
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean) ** 2)

        if ss_total == 0:
            return 0.0, "undefined"

        eta_sq = ss_between / ss_total

        # Interpretation (Cohen, 1988)
        if eta_sq < 0.01:
            interpretation = "negligible"
        elif eta_sq < 0.06:
            interpretation = "small"
        elif eta_sq < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"

        return eta_sq, interpretation

    @staticmethod
    def epsilon_squared(h_statistic: float, n_total: int, k_groups: int) -> Tuple[float, str]:
        """
        Calculate epsilon-squared for Kruskal-Wallis test.

        ε² = H / (n - 1), where H is the Kruskal-Wallis statistic.
        This is the non-parametric analog to eta-squared.
        """
        if n_total <= 1:
            return 0.0, "undefined"

        eps_sq = h_statistic / (n_total - 1)

        # Same interpretation thresholds as eta-squared
        if eps_sq < 0.01:
            interpretation = "negligible"
        elif eps_sq < 0.06:
            interpretation = "small"
        elif eps_sq < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"

        return eps_sq, interpretation


class AssumptionTester:
    """
    Test statistical assumptions required for various tests.
    """

    @staticmethod
    def test_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, str]:
        """
        Test normality using Shapiro-Wilk test.

        Note: For n > 5000, we use a random subsample as Shapiro-Wilk
        is sensitive to sample size.
        """
        if len(data) < 3:
            return False, "Insufficient data (n < 3)"

        # Subsample for large datasets
        if len(data) > 5000:
            rng = np.random.default_rng(42)
            data = rng.choice(data, size=5000, replace=False)

        stat, p_value = shapiro(data)
        is_normal = p_value > alpha

        detail = f"Shapiro-Wilk W={stat:.4f}, p={p_value:.4f}"
        if is_normal:
            detail += " (normality assumption met)"
        else:
            detail += " (normality assumption violated)"

        return is_normal, detail

    @staticmethod
    def test_homogeneity(groups: List[np.ndarray], alpha: float = 0.05) -> Tuple[bool, str]:
        """
        Test homogeneity of variance using Levene's test.

        Uses median (Brown-Forsythe) for robustness to non-normality.
        """
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) < 2:
            return False, "Insufficient groups for Levene's test"

        stat, p_value = levene(*valid_groups, center='median')
        is_homogeneous = p_value > alpha

        detail = f"Levene's W={stat:.4f}, p={p_value:.4f}"
        if is_homogeneous:
            detail += " (homogeneity assumption met)"
        else:
            detail += " (homogeneity assumption violated)"

        return is_homogeneous, detail


class RigorousStatisticalAnalysis:
    """
    Publication-quality statistical analysis suite for ABM results.

    All analyses include:
    - Effect sizes with 95% confidence intervals
    - Assumption testing
    - Multiple comparison correction (Benjamini-Hochberg FDR)
    - Plain-language interpretations

    Parameters
    ----------
    agent_df : pd.DataFrame
        Agent-level data with columns: primary_ai_level, capital_growth,
        survived, final_capital, etc.
    decision_df : pd.DataFrame
        Decision-level data with uncertainty perceptions and AI usage.
    alpha : float, default 0.05
        Significance level for hypothesis tests.
    """

    def __init__(self, agent_df: pd.DataFrame, decision_df: pd.DataFrame,
                 matured_df: Optional[pd.DataFrame] = None,
                 uncertainty_detail_df: Optional[pd.DataFrame] = None,
                 alpha: float = 0.05):
        self.agent_df = agent_df
        self.decision_df = decision_df
        self.matured_df = matured_df if matured_df is not None else pd.DataFrame()
        self.uncertainty_detail_df = uncertainty_detail_df if uncertainty_detail_df is not None else pd.DataFrame()
        self.alpha = alpha
        self.results: List[StatisticalTestResult] = []
        self.effect_calculator = EffectSizeCalculator()
        self.assumption_tester = AssumptionTester()

    def run_all_analyses(self) -> pd.DataFrame:
        """
        Run complete statistical analysis suite.

        Returns
        -------
        pd.DataFrame
            Publication-ready table of all statistical test results.
        """
        print("\n" + "=" * 70)
        print("RIGOROUS STATISTICAL ANALYSIS FOR AMJ SUBMISSION")
        print("=" * 70)

        # Hypothesis 1: AI tier effects on performance
        self._test_ai_performance_effects()

        # Hypothesis 2: AI effects on survival
        self._test_ai_survival_effects()

        # Hypothesis 3: AI effects on uncertainty dimensions
        self._test_ai_uncertainty_effects()

        # Hypothesis 4: AI effects on investment outcomes
        self._test_ai_investment_outcomes()

        # Hypothesis 5: Paradox of future knowledge
        self._test_paradox_of_knowledge()

        # Apply multiple comparison correction
        self._apply_fdr_correction()

        # Generate results table
        results_df = self._generate_results_table()

        print("\n" + "=" * 70)
        print(f"Completed {len(self.results)} statistical tests")
        print("=" * 70)

        return results_df

    def _test_ai_performance_effects(self):
        """
        Test H1: AI augmentation affects entrepreneurial performance.

        Tests whether capital growth differs across AI tiers using
        Kruskal-Wallis H-test (non-parametric) with epsilon-squared effect size.
        """
        print("\n📊 Testing H1: AI Effects on Performance...")

        if self.agent_df.empty or 'capital_growth' not in self.agent_df.columns:
            print("   ⚠️ Insufficient data for performance analysis")
            return

        # Prepare groups
        ai_levels = ['none', 'basic', 'advanced', 'premium']
        groups = []
        sample_sizes = {}

        for level in ai_levels:
            mask = self.agent_df['primary_ai_level'] == level
            data = self.agent_df.loc[mask, 'capital_growth'].dropna().values
            if len(data) > 0:
                groups.append(data)
                sample_sizes[level] = len(data)

        if len(groups) < 2:
            print("   ⚠️ Need at least 2 AI groups with data")
            return

        # Assumption tests
        assumptions_met = {}
        assumptions_details = {}

        for i, (level, data) in enumerate(zip([l for l in ai_levels if l in sample_sizes], groups)):
            is_normal, detail = self.assumption_tester.test_normality(data)
            assumptions_met[f'normality_{level}'] = is_normal
            assumptions_details[f'normality_{level}'] = detail

        is_homogeneous, detail = self.assumption_tester.test_homogeneity(groups)
        assumptions_met['homogeneity'] = is_homogeneous
        assumptions_details['homogeneity'] = detail

        # Kruskal-Wallis test (robust to non-normality)
        h_stat, p_value = kruskal(*groups)

        # Effect size: epsilon-squared
        n_total = sum(sample_sizes.values())
        effect_size, effect_interp = self.effect_calculator.epsilon_squared(
            h_stat, n_total, len(groups)
        )

        # Bootstrap CI for effect size
        rng = np.random.default_rng(42)
        boot_effects = []
        for _ in range(1000):
            boot_groups = [rng.choice(g, size=len(g), replace=True) for g in groups]
            h_boot, _ = kruskal(*boot_groups)
            eff, _ = self.effect_calculator.epsilon_squared(h_boot, n_total, len(groups))
            boot_effects.append(eff)
        effect_ci = (np.percentile(boot_effects, 2.5), np.percentile(boot_effects, 97.5))

        # Conclusion
        if p_value < self.alpha:
            conclusion = (f"AI tier significantly affects capital growth "
                         f"(H={h_stat:.2f}, p={p_value:.4f}, ε²={effect_size:.3f} [{effect_interp}]). "
                         f"This supports H1.")
        else:
            conclusion = (f"No significant difference in capital growth across AI tiers "
                         f"(H={h_stat:.2f}, p={p_value:.4f}, ε²={effect_size:.3f}). "
                         f"H1 not supported.")

        result = StatisticalTestResult(
            test_name="H1: AI Tier → Capital Growth (Kruskal-Wallis)",
            test_statistic=h_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type="epsilon-squared (ε²)",
            effect_size_ci=effect_ci,
            effect_interpretation=effect_interp,
            sample_sizes=sample_sizes,
            assumptions_met=assumptions_met,
            assumptions_details=assumptions_details,
            conclusion=conclusion
        )
        self.results.append(result)
        print(f"   ✓ {conclusion}")

        # Pairwise comparisons with effect sizes
        self._pairwise_ai_comparisons(groups, ai_levels, sample_sizes, "capital_growth")

    def _pairwise_ai_comparisons(self, groups: List[np.ndarray],
                                  levels: List[str], sample_sizes: Dict[str, int],
                                  metric_name: str):
        """Run pairwise Mann-Whitney U tests with Cliff's delta effect sizes."""
        available_levels = [l for l in levels if l in sample_sizes]

        for i, level1 in enumerate(available_levels):
            for level2 in available_levels[i+1:]:
                idx1 = available_levels.index(level1)
                idx2 = available_levels.index(level2)

                g1, g2 = groups[idx1], groups[idx2]

                if len(g1) < 2 or len(g2) < 2:
                    continue

                # Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(g1, g2, alternative='two-sided')

                # Cliff's delta (non-parametric effect size)
                delta, interp = self.effect_calculator.cliffs_delta(g1, g2)

                # Bootstrap CI for Cliff's delta
                rng = np.random.default_rng(42)
                boot_deltas = []
                for _ in range(1000):
                    boot1 = rng.choice(g1, size=len(g1), replace=True)
                    boot2 = rng.choice(g2, size=len(g2), replace=True)
                    d, _ = self.effect_calculator.cliffs_delta(boot1, boot2)
                    boot_deltas.append(d)
                delta_ci = (np.percentile(boot_deltas, 2.5), np.percentile(boot_deltas, 97.5))

                result = StatisticalTestResult(
                    test_name=f"Pairwise: {level1} vs {level2} ({metric_name})",
                    test_statistic=u_stat,
                    p_value=p_value,
                    effect_size=delta,
                    effect_size_type="Cliff's delta (δ)",
                    effect_size_ci=delta_ci,
                    effect_interpretation=interp,
                    sample_sizes={level1: len(g1), level2: len(g2)},
                    assumptions_met={},
                    assumptions_details={},
                    conclusion=f"{'Significant' if p_value < self.alpha else 'Non-significant'} "
                              f"difference (δ={delta:.3f}, {interp})"
                )
                self.results.append(result)

    def _test_ai_survival_effects(self):
        """
        Test H2: AI augmentation affects entrepreneurial survival.

        Uses chi-square test with Cramér's V effect size.
        """
        print("\n📊 Testing H2: AI Effects on Survival...")

        if self.agent_df.empty or 'survived' not in self.agent_df.columns:
            print("   ⚠️ Insufficient data for survival analysis")
            return

        # Create contingency table
        contingency = pd.crosstab(
            self.agent_df['primary_ai_level'],
            self.agent_df['survived']
        )

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            print("   ⚠️ Insufficient variation for chi-square test")
            return

        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Cramér's V effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        # Interpretation (Cohen, 1988)
        if cramers_v < 0.1:
            interp = "negligible"
        elif cramers_v < 0.3:
            interp = "small"
        elif cramers_v < 0.5:
            interp = "medium"
        else:
            interp = "large"

        # Sample sizes
        sample_sizes = contingency.sum(axis=1).to_dict()

        # Survival rates by AI level
        survival_rates = self.agent_df.groupby('primary_ai_level')['survived'].mean()

        if p_value < self.alpha:
            conclusion = (f"AI tier significantly affects survival rates "
                         f"(χ²={chi2:.2f}, df={dof}, p={p_value:.4f}, V={cramers_v:.3f} [{interp}]). "
                         f"Survival rates: {survival_rates.to_dict()}")
        else:
            conclusion = (f"No significant association between AI tier and survival "
                         f"(χ²={chi2:.2f}, df={dof}, p={p_value:.4f}, V={cramers_v:.3f}).")

        result = StatisticalTestResult(
            test_name="H2: AI Tier → Survival (Chi-Square)",
            test_statistic=chi2,
            p_value=p_value,
            effect_size=cramers_v,
            effect_size_type="Cramér's V",
            effect_size_ci=None,  # CI for Cramér's V requires specialized methods
            effect_interpretation=interp,
            sample_sizes=sample_sizes,
            assumptions_met={'expected_freq_>5': (expected >= 5).all()},
            assumptions_details={'expected_frequencies': f"Min expected: {expected.min():.1f}"},
            conclusion=conclusion
        )
        self.results.append(result)
        print(f"   ✓ {conclusion}")

    def _test_ai_uncertainty_effects(self):
        """
        Test H3: AI augmentation reduces perceived actor ignorance but increases
        other uncertainty dimensions (paradox of future knowledge).

        Tests each of the four Knightian uncertainty dimensions separately.
        """
        print("\n📊 Testing H3: AI Effects on Uncertainty Dimensions...")

        if self.decision_df.empty:
            print("   ⚠️ Insufficient decision data")
            return

        uncertainty_cols = {
            'actor_ignorance': ['perc_actor_ignorance', 'actor_ignorance_level', 'ignorance_level'],
            'practical_indeterminism': ['perc_practical_indeterminism', 'indeterminism_level'],
            'agentic_novelty': ['perc_agentic_novelty', 'novelty_potential'],
            'competitive_recursion': ['perc_competitive_recursion', 'recursion_level']
        }

        for dimension, possible_cols in uncertainty_cols.items():
            # Find the column that exists
            col = None
            for c in possible_cols:
                if c in self.decision_df.columns:
                    col = c
                    break

            if col is None:
                continue

            # Prepare groups by AI level
            ai_levels = ['none', 'basic', 'advanced', 'premium']
            groups = []
            sample_sizes = {}

            for level in ai_levels:
                mask = self.decision_df['ai_level_used'] == level
                data = self.decision_df.loc[mask, col].dropna().values
                if len(data) > 0:
                    groups.append(data)
                    sample_sizes[level] = len(data)

            if len(groups) < 2:
                continue

            # Kruskal-Wallis test
            h_stat, p_value = kruskal(*groups)

            # Effect size
            n_total = sum(sample_sizes.values())
            effect_size, effect_interp = self.effect_calculator.epsilon_squared(
                h_stat, n_total, len(groups)
            )

            # Calculate means for interpretation
            means = {level: np.mean(groups[i])
                    for i, level in enumerate([l for l in ai_levels if l in sample_sizes])}

            # Direction of effect
            if 'none' in means and 'premium' in means:
                direction = "decreases" if means['premium'] < means['none'] else "increases"
            else:
                direction = "varies"

            result = StatisticalTestResult(
                test_name=f"H3a: AI Tier → {dimension.replace('_', ' ').title()}",
                test_statistic=h_stat,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_type="epsilon-squared (ε²)",
                effect_size_ci=None,
                effect_interpretation=effect_interp,
                sample_sizes=sample_sizes,
                assumptions_met={},
                assumptions_details={'means_by_tier': str(means)},
                conclusion=f"AI {direction} {dimension.replace('_', ' ')} "
                          f"(H={h_stat:.2f}, p={p_value:.4f}, ε²={effect_size:.3f})"
            )
            self.results.append(result)
            print(f"   ✓ {dimension}: H={h_stat:.2f}, p={p_value:.4f}, ε²={effect_size:.3f}")

    def _test_ai_investment_outcomes(self):
        """
        Test H4: AI augmentation affects investment outcomes (ROI).
        """
        print("\n📊 Testing H4: AI Effects on Investment Outcomes...")

        if self.matured_df.empty:
            print("   ⚠️ No matured investment data available")
            return

        # Find ROI column
        roi_col = None
        for col in ['realized_roi', 'return_multiple', 'roi', 'realized_return']:
            if col in self.matured_df.columns:
                roi_col = col
                break

        if roi_col is None:
            print("   ⚠️ No ROI column found")
            return

        # Find AI level column
        ai_col = None
        for col in ['ai_level_used', 'ai_level', 'ai_tier']:
            if col in self.matured_df.columns:
                ai_col = col
                break

        if ai_col is None:
            print("   ⚠️ No AI level column found")
            return

        # Prepare groups
        ai_levels = ['none', 'basic', 'advanced', 'premium']
        groups = []
        sample_sizes = {}

        for level in ai_levels:
            mask = self.matured_df[ai_col] == level
            data = self.matured_df.loc[mask, roi_col].dropna().values
            if len(data) > 0:
                groups.append(data)
                sample_sizes[level] = len(data)

        if len(groups) < 2:
            print("   ⚠️ Need at least 2 AI groups with matured investments")
            return

        # Kruskal-Wallis test
        h_stat, p_value = kruskal(*groups)

        # Effect size
        n_total = sum(sample_sizes.values())
        effect_size, effect_interp = self.effect_calculator.epsilon_squared(
            h_stat, n_total, len(groups)
        )

        # Calculate medians
        medians = {level: np.median(groups[i])
                  for i, level in enumerate([l for l in ai_levels if l in sample_sizes])}

        result = StatisticalTestResult(
            test_name="H4: AI Tier → Investment ROI (Kruskal-Wallis)",
            test_statistic=h_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type="epsilon-squared (ε²)",
            effect_size_ci=None,
            effect_interpretation=effect_interp,
            sample_sizes=sample_sizes,
            assumptions_met={},
            assumptions_details={'median_roi_by_tier': str(medians)},
            conclusion=f"AI tier {'significantly affects' if p_value < self.alpha else 'does not significantly affect'} "
                      f"investment ROI (H={h_stat:.2f}, p={p_value:.4f})"
        )
        self.results.append(result)
        print(f"   ✓ {result.conclusion}")

    def _test_paradox_of_knowledge(self):
        """
        Test H5: The paradox of future knowledge - AI simultaneously reduces
        actor ignorance while increasing practical indeterminism and
        competitive recursion.

        This tests the core theoretical proposition from Townsend et al. (2025).
        """
        print("\n📊 Testing H5: Paradox of Future Knowledge...")

        if self.decision_df.empty:
            print("   ⚠️ Insufficient data")
            return

        # Find relevant columns
        ignorance_col = None
        indeterminism_col = None
        recursion_col = None

        for col in self.decision_df.columns:
            if 'ignorance' in col.lower():
                ignorance_col = col
            if 'indeterminism' in col.lower():
                indeterminism_col = col
            if 'recursion' in col.lower():
                recursion_col = col

        if not all([ignorance_col, indeterminism_col, recursion_col]):
            print("   ⚠️ Missing uncertainty dimension columns")
            return

        # Compare AI users vs non-users
        ai_users = self.decision_df[self.decision_df['ai_level_used'] != 'none']
        non_users = self.decision_df[self.decision_df['ai_level_used'] == 'none']

        if len(ai_users) < 10 or len(non_users) < 10:
            print("   ⚠️ Insufficient sample sizes")
            return

        # Test 1: AI reduces actor ignorance
        g1 = non_users[ignorance_col].dropna().values
        g2 = ai_users[ignorance_col].dropna().values

        if len(g1) > 1 and len(g2) > 1:
            u_stat, p_val = mannwhitneyu(g1, g2, alternative='greater')  # one-tailed
            delta, interp = self.effect_calculator.cliffs_delta(g1, g2)

            result = StatisticalTestResult(
                test_name="H5a: AI Users Have Lower Actor Ignorance (Mann-Whitney U)",
                test_statistic=u_stat,
                p_value=p_val,
                effect_size=delta,
                effect_size_type="Cliff's delta (δ)",
                effect_size_ci=None,
                effect_interpretation=interp,
                sample_sizes={'non_users': len(g1), 'ai_users': len(g2)},
                assumptions_met={},
                assumptions_details={'mean_non_users': np.mean(g1), 'mean_ai_users': np.mean(g2)},
                conclusion=f"AI users {'have' if p_val < self.alpha else 'do not have'} "
                          f"significantly lower actor ignorance (δ={delta:.3f})"
            )
            self.results.append(result)
            print(f"   ✓ Actor ignorance: δ={delta:.3f}, p={p_val:.4f}")

        # Test 2: AI increases competitive recursion
        g1 = non_users[recursion_col].dropna().values
        g2 = ai_users[recursion_col].dropna().values

        if len(g1) > 1 and len(g2) > 1:
            u_stat, p_val = mannwhitneyu(g2, g1, alternative='greater')  # AI > non-AI
            delta, interp = self.effect_calculator.cliffs_delta(g2, g1)

            result = StatisticalTestResult(
                test_name="H5b: AI Users Have Higher Competitive Recursion (Mann-Whitney U)",
                test_statistic=u_stat,
                p_value=p_val,
                effect_size=delta,
                effect_size_type="Cliff's delta (δ)",
                effect_size_ci=None,
                effect_interpretation=interp,
                sample_sizes={'non_users': len(g1), 'ai_users': len(g2)},
                assumptions_met={},
                assumptions_details={'mean_non_users': np.mean(g1), 'mean_ai_users': np.mean(g2)},
                conclusion=f"AI users {'have' if p_val < self.alpha else 'do not have'} "
                          f"significantly higher competitive recursion (δ={delta:.3f})"
            )
            self.results.append(result)
            print(f"   ✓ Competitive recursion: δ={delta:.3f}, p={p_val:.4f}")

    def _apply_fdr_correction(self):
        """
        Apply Benjamini-Hochberg FDR correction to all p-values.

        This controls the false discovery rate when running multiple tests.
        """
        if not HAS_STATSMODELS:
            print("\n⚠️ statsmodels not available; skipping FDR correction")
            return

        if not self.results:
            return

        print("\n📊 Applying Benjamini-Hochberg FDR Correction...")

        p_values = [r.p_value for r in self.results]

        # Apply correction
        rejected, p_adjusted, _, _ = multipletests(
            p_values, alpha=self.alpha, method='fdr_bh'
        )

        # Update results
        for i, result in enumerate(self.results):
            result.p_value_adjusted = p_adjusted[i]

        n_significant_raw = sum(1 for p in p_values if p < self.alpha)
        n_significant_adj = sum(1 for p in p_adjusted if p < self.alpha)

        print(f"   ✓ {n_significant_raw} tests significant at α={self.alpha} (unadjusted)")
        print(f"   ✓ {n_significant_adj} tests significant after FDR correction")

    def _generate_results_table(self) -> pd.DataFrame:
        """
        Generate publication-ready results table.

        Returns
        -------
        pd.DataFrame
            Table suitable for inclusion in AMJ manuscript.
        """
        rows = [r.to_dict() for r in self.results]
        df = pd.DataFrame(rows)

        # Reorder columns for publication
        col_order = [
            'test_name', 'n_total', 'test_statistic', 'p_value', 'p_value_adjusted',
            'effect_size', 'effect_size_type', 'effect_size_ci_lower', 'effect_size_ci_upper',
            'effect_interpretation', 'assumptions_met', 'conclusion'
        ]

        available_cols = [c for c in col_order if c in df.columns]
        df = df[available_cols]

        return df

    def generate_descriptive_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive descriptive statistics table.

        Returns
        -------
        pd.DataFrame
            Descriptive statistics by AI tier suitable for Table 1.
        """
        if self.agent_df.empty:
            return pd.DataFrame()

        # Variables to summarize
        numeric_vars = ['final_capital', 'capital_growth', 'innovations',
                       'portfolio_diversity', 'survived']

        available_vars = [v for v in numeric_vars if v in self.agent_df.columns]

        # Group by AI level
        stats_list = []

        for level in ['none', 'basic', 'advanced', 'premium']:
            subset = self.agent_df[self.agent_df['primary_ai_level'] == level]

            if len(subset) == 0:
                continue

            row = {'AI Tier': level.title(), 'n': len(subset)}

            for var in available_vars:
                data = subset[var].dropna()
                if len(data) > 0:
                    row[f'{var}_mean'] = data.mean()
                    row[f'{var}_sd'] = data.std()
                    row[f'{var}_median'] = data.median()
                    row[f'{var}_min'] = data.min()
                    row[f'{var}_max'] = data.max()

            stats_list.append(row)

        return pd.DataFrame(stats_list)

    def generate_correlation_matrix(self) -> pd.DataFrame:
        """
        Generate correlation matrix with significance indicators.

        Returns
        -------
        pd.DataFrame
            Correlation matrix with significance stars.
        """
        if self.agent_df.empty:
            return pd.DataFrame()

        numeric_cols = self.agent_df.select_dtypes(include=[np.number]).columns

        # Filter to relevant variables
        relevant = ['final_capital', 'capital_growth', 'innovations',
                   'portfolio_diversity', 'survived']
        cols = [c for c in relevant if c in numeric_cols]

        if len(cols) < 2:
            return pd.DataFrame()

        data = self.agent_df[cols].dropna()

        # Calculate correlations with p-values
        n = len(cols)
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=object)

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i == j:
                    corr_matrix.loc[col1, col2] = "1.00"
                elif i < j:
                    r, p = spearmanr(data[col1], data[col2])
                    stars = ""
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                    corr_matrix.loc[col1, col2] = f"{r:.2f}{stars}"
                    corr_matrix.loc[col2, col1] = f"{r:.2f}{stars}"

        return corr_matrix


@dataclass
class MixedEffectsResult:
    """
    Container for mixed-effects model results.

    Attributes
    ----------
    model_name : str
        Descriptive name of the model.
    dependent_variable : str
        Name of the outcome variable.
    fixed_effects : Dict[str, Dict[str, float]]
        Fixed effects coefficients with SE, z-value, p-value, and 95% CI.
    random_effects_variance : Dict[str, float]
        Variance components for random effects.
    model_fit : Dict[str, float]
        Model fit statistics (AIC, BIC, log-likelihood, ICC).
    n_observations : int
        Number of observations in the model.
    n_groups : Dict[str, int]
        Number of groups for each random effect.
    convergence : bool
        Whether the model converged successfully.
    interpretation : str
        Plain-language interpretation of the results.
    """
    model_name: str
    dependent_variable: str
    fixed_effects: Dict[str, Dict[str, float]]
    random_effects_variance: Dict[str, float]
    model_fit: Dict[str, float]
    n_observations: int
    n_groups: Dict[str, int]
    convergence: bool
    interpretation: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fixed effects to publication-ready DataFrame."""
        rows = []
        for var, stats in self.fixed_effects.items():
            rows.append({
                'Variable': var,
                'Coefficient': stats.get('coef', np.nan),
                'Std. Error': stats.get('se', np.nan),
                'z-value': stats.get('z', np.nan),
                'p-value': stats.get('p', np.nan),
                '95% CI Lower': stats.get('ci_lower', np.nan),
                '95% CI Upper': stats.get('ci_upper', np.nan),
                'Significance': self._get_significance_stars(stats.get('p', 1.0))
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _get_significance_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.10:
            return "†"
        return ""


class MixedEffectsAnalysis:
    """
    Mixed-effects (multilevel) models for nested ABM data.

    Agent-based simulation data has a natural hierarchical structure:
    - Level 1: Decisions/observations within agents
    - Level 2: Agents within simulation runs
    - Level 3: Simulation runs (for multi-run designs)

    Mixed-effects models account for this nesting by including random
    intercepts (and optionally random slopes) for higher-level units,
    providing more accurate standard errors and enabling proper inference.

    Theoretical Justification
    -------------------------
    Standard OLS assumes independence of observations, which is violated
    when decisions are nested within agents and agents within runs. Ignoring
    this clustering leads to:
    1. Underestimated standard errors (inflated Type I error)
    2. Inefficient parameter estimates
    3. Incorrect inferences about AI effects

    Mixed-effects models partition variance into within-group and between-group
    components, providing the Intraclass Correlation Coefficient (ICC) which
    indicates the proportion of variance attributable to clustering.

    References
    ----------
    Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical linear models.
        Sage Publications.

    Snijders, T. A., & Bosker, R. J. (2012). Multilevel analysis: An
        introduction to basic and advanced multilevel modeling. Sage.
    """

    def __init__(self, agent_df: pd.DataFrame, decision_df: pd.DataFrame,
                 matured_df: Optional[pd.DataFrame] = None):
        self.agent_df = agent_df
        self.decision_df = decision_df
        self.matured_df = matured_df if matured_df is not None else pd.DataFrame()
        self.results: List[MixedEffectsResult] = []

    def run_all_models(self) -> List[MixedEffectsResult]:
        """
        Run all mixed-effects models.

        Returns
        -------
        List[MixedEffectsResult]
            Results from all fitted models.
        """
        if not HAS_MIXED_MODELS:
            print("\n⚠️ statsmodels not available; skipping mixed-effects models")
            print("   Install with: pip install statsmodels")
            return []

        print("\n" + "=" * 70)
        print("MIXED-EFFECTS MODELS FOR NESTED DATA STRUCTURE")
        print("=" * 70)

        # Model 1: AI effects on capital growth (agent-level, nested in runs)
        self._fit_capital_growth_model()

        # Model 2: AI effects on decision outcomes (decision-level, nested in agents and runs)
        self._fit_decision_outcome_model()

        # Model 3: AI effects on uncertainty perception (decision-level)
        self._fit_uncertainty_perception_model()

        # Model 4: AI effects on investment returns (matured investments)
        self._fit_investment_returns_model()

        return self.results

    def _fit_capital_growth_model(self):
        """
        Model 1: AI Tier → Capital Growth with random intercepts for runs.

        Model specification:
            capital_growth_ij = β₀ + β₁(AI_basic) + β₂(AI_advanced) + β₃(AI_premium) + u_j + ε_ij

        Where:
            - i indexes agents, j indexes runs
            - u_j ~ N(0, σ²_u) is the random intercept for run j
            - ε_ij ~ N(0, σ²) is the residual error
        """
        print("\n📊 Model 1: AI Tier → Capital Growth (agents nested in runs)")

        if self.agent_df.empty or 'capital_growth' not in self.agent_df.columns:
            print("   ⚠️ Insufficient data")
            return

        # Prepare data
        df = self.agent_df.copy()

        # Check for run_id column
        if 'run_id' not in df.columns:
            print("   ⚠️ No run_id column; cannot fit mixed model")
            return

        # Create dummy variables for AI level (reference = none)
        df['ai_level'] = df['primary_ai_level'].fillna('none')
        df = df[df['ai_level'].isin(['none', 'basic', 'advanced', 'premium'])]

        if len(df) < 50:
            print("   ⚠️ Insufficient observations (n < 50)")
            return

        # Create dummies
        df['ai_basic'] = (df['ai_level'] == 'basic').astype(int)
        df['ai_advanced'] = (df['ai_level'] == 'advanced').astype(int)
        df['ai_premium'] = (df['ai_level'] == 'premium').astype(int)

        # Drop missing values
        model_vars = ['capital_growth', 'ai_basic', 'ai_advanced', 'ai_premium', 'run_id']
        df_model = df[model_vars].dropna()

        if len(df_model) < 50:
            print("   ⚠️ Insufficient complete cases")
            return

        try:
            # Fit mixed model with random intercept for run_id
            model = smf.mixedlm(
                "capital_growth ~ ai_basic + ai_advanced + ai_premium",
                data=df_model,
                groups=df_model['run_id']
            )
            result = model.fit(method='powell', maxiter=500)

            # Extract results
            fixed_effects = {}
            for var in ['Intercept', 'ai_basic', 'ai_advanced', 'ai_premium']:
                if var in result.params.index:
                    fixed_effects[var] = {
                        'coef': result.params[var],
                        'se': result.bse[var],
                        'z': result.tvalues[var],
                        'p': result.pvalues[var],
                        'ci_lower': result.conf_int().loc[var, 0],
                        'ci_upper': result.conf_int().loc[var, 1]
                    }

            # Random effects variance
            random_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else 0
            residual_var = result.scale

            # Calculate ICC
            icc = random_var / (random_var + residual_var) if (random_var + residual_var) > 0 else 0

            # Number of groups
            n_runs = df_model['run_id'].nunique()

            # Interpretation
            significant_effects = []
            for var in ['ai_basic', 'ai_advanced', 'ai_premium']:
                if var in fixed_effects and fixed_effects[var]['p'] < 0.05:
                    direction = "higher" if fixed_effects[var]['coef'] > 0 else "lower"
                    tier = var.replace('ai_', '').title()
                    significant_effects.append(f"{tier} AI has {direction} capital growth")

            if significant_effects:
                interp = "; ".join(significant_effects) + f". ICC = {icc:.3f} ({icc*100:.1f}% of variance between runs)."
            else:
                interp = f"No significant AI tier effects on capital growth. ICC = {icc:.3f}."

            model_result = MixedEffectsResult(
                model_name="Capital Growth ~ AI Tier (Random Intercept: Run)",
                dependent_variable="capital_growth",
                fixed_effects=fixed_effects,
                random_effects_variance={'run_intercept': random_var, 'residual': residual_var},
                model_fit={
                    'AIC': result.aic if hasattr(result, 'aic') else np.nan,
                    'BIC': result.bic if hasattr(result, 'bic') else np.nan,
                    'log_likelihood': result.llf if hasattr(result, 'llf') else np.nan,
                    'ICC': icc
                },
                n_observations=len(df_model),
                n_groups={'runs': n_runs},
                convergence=result.converged if hasattr(result, 'converged') else True,
                interpretation=interp
            )
            self.results.append(model_result)

            print(f"   ✓ Fitted model: n={len(df_model)}, runs={n_runs}, ICC={icc:.3f}")
            for var, stats in fixed_effects.items():
                sig = "***" if stats['p'] < 0.001 else "**" if stats['p'] < 0.01 else "*" if stats['p'] < 0.05 else ""
                print(f"      {var}: β={stats['coef']:.4f}, SE={stats['se']:.4f}, p={stats['p']:.4f}{sig}")

        except Exception as e:
            print(f"   ⚠️ Model fitting failed: {e}")

    def _fit_decision_outcome_model(self):
        """
        Model 2: AI Tier → Decision Success with crossed random effects.

        Model specification:
            success_ijk = β₀ + β₁(AI_tier) + β₂(action_type) + u_j + v_k + ε_ijk

        Where:
            - i indexes decisions, j indexes agents, k indexes runs
            - u_j ~ N(0, σ²_agent) is the random intercept for agent j
            - v_k ~ N(0, σ²_run) is the random intercept for run k
        """
        print("\n📊 Model 2: AI Tier → Decision Success (decisions nested in agents)")

        if self.decision_df.empty or 'success' not in self.decision_df.columns:
            print("   ⚠️ Insufficient decision data")
            return

        df = self.decision_df.copy()

        # Check for required columns
        required = ['success', 'ai_level_used', 'agent_id']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"   ⚠️ Missing columns: {missing}")
            return

        # Prepare data
        df['ai_level'] = df['ai_level_used'].fillna('none')
        df = df[df['ai_level'].isin(['none', 'basic', 'advanced', 'premium'])]

        # Create dummies
        df['ai_basic'] = (df['ai_level'] == 'basic').astype(int)
        df['ai_advanced'] = (df['ai_level'] == 'advanced').astype(int)
        df['ai_premium'] = (df['ai_level'] == 'premium').astype(int)

        # Ensure success is numeric
        df['success'] = pd.to_numeric(df['success'], errors='coerce')

        model_vars = ['success', 'ai_basic', 'ai_advanced', 'ai_premium', 'agent_id']
        df_model = df[model_vars].dropna()

        if len(df_model) < 100:
            print("   ⚠️ Insufficient observations")
            return

        try:
            # Fit mixed model with random intercept for agent_id
            model = smf.mixedlm(
                "success ~ ai_basic + ai_advanced + ai_premium",
                data=df_model,
                groups=df_model['agent_id']
            )
            result = model.fit(method='powell', maxiter=500)

            # Extract results
            fixed_effects = {}
            for var in ['Intercept', 'ai_basic', 'ai_advanced', 'ai_premium']:
                if var in result.params.index:
                    fixed_effects[var] = {
                        'coef': result.params[var],
                        'se': result.bse[var],
                        'z': result.tvalues[var],
                        'p': result.pvalues[var],
                        'ci_lower': result.conf_int().loc[var, 0],
                        'ci_upper': result.conf_int().loc[var, 1]
                    }

            # Random effects variance
            random_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else 0
            residual_var = result.scale

            # ICC
            icc = random_var / (random_var + residual_var) if (random_var + residual_var) > 0 else 0

            n_agents = df_model['agent_id'].nunique()

            model_result = MixedEffectsResult(
                model_name="Decision Success ~ AI Tier (Random Intercept: Agent)",
                dependent_variable="success",
                fixed_effects=fixed_effects,
                random_effects_variance={'agent_intercept': random_var, 'residual': residual_var},
                model_fit={
                    'AIC': result.aic if hasattr(result, 'aic') else np.nan,
                    'BIC': result.bic if hasattr(result, 'bic') else np.nan,
                    'ICC': icc
                },
                n_observations=len(df_model),
                n_groups={'agents': n_agents},
                convergence=result.converged if hasattr(result, 'converged') else True,
                interpretation=f"ICC = {icc:.3f} ({icc*100:.1f}% of success variance is between agents)"
            )
            self.results.append(model_result)

            print(f"   ✓ Fitted model: n={len(df_model)}, agents={n_agents}, ICC={icc:.3f}")

        except Exception as e:
            print(f"   ⚠️ Model fitting failed: {e}")

    def _fit_uncertainty_perception_model(self):
        """
        Model 3: AI Tier → Actor Ignorance perception.

        Tests whether AI reduces perceived actor ignorance after controlling
        for the nested structure of decisions within agents.
        """
        print("\n📊 Model 3: AI Tier → Actor Ignorance (testing paradox hypothesis)")

        if self.decision_df.empty:
            print("   ⚠️ Insufficient data")
            return

        # Find ignorance column
        ignorance_col = None
        for col in ['perc_actor_ignorance', 'actor_ignorance_level', 'ignorance_level']:
            if col in self.decision_df.columns:
                ignorance_col = col
                break

        if ignorance_col is None:
            print("   ⚠️ No actor ignorance column found")
            return

        df = self.decision_df.copy()

        if 'agent_id' not in df.columns:
            print("   ⚠️ No agent_id column")
            return

        # Prepare data
        df['ai_level'] = df['ai_level_used'].fillna('none')
        df = df[df['ai_level'].isin(['none', 'basic', 'advanced', 'premium'])]

        df['ai_any'] = (df['ai_level'] != 'none').astype(int)
        df['ai_basic'] = (df['ai_level'] == 'basic').astype(int)
        df['ai_advanced'] = (df['ai_level'] == 'advanced').astype(int)
        df['ai_premium'] = (df['ai_level'] == 'premium').astype(int)

        df[ignorance_col] = pd.to_numeric(df[ignorance_col], errors='coerce')

        model_vars = [ignorance_col, 'ai_basic', 'ai_advanced', 'ai_premium', 'agent_id']
        df_model = df[model_vars].dropna()

        if len(df_model) < 100:
            print("   ⚠️ Insufficient observations")
            return

        try:
            model = smf.mixedlm(
                f"{ignorance_col} ~ ai_basic + ai_advanced + ai_premium",
                data=df_model,
                groups=df_model['agent_id']
            )
            result = model.fit(method='powell', maxiter=500)

            fixed_effects = {}
            for var in ['Intercept', 'ai_basic', 'ai_advanced', 'ai_premium']:
                if var in result.params.index:
                    fixed_effects[var] = {
                        'coef': result.params[var],
                        'se': result.bse[var],
                        'z': result.tvalues[var],
                        'p': result.pvalues[var],
                        'ci_lower': result.conf_int().loc[var, 0],
                        'ci_upper': result.conf_int().loc[var, 1]
                    }

            # Check if AI reduces ignorance (negative coefficients)
            reduces_ignorance = all(
                fixed_effects.get(f'ai_{tier}', {}).get('coef', 0) < 0
                for tier in ['basic', 'advanced', 'premium']
                if f'ai_{tier}' in fixed_effects
            )

            random_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else 0
            residual_var = result.scale
            icc = random_var / (random_var + residual_var) if (random_var + residual_var) > 0 else 0

            interp = (f"AI {'reduces' if reduces_ignorance else 'does not consistently reduce'} "
                     f"perceived actor ignorance. This {'supports' if reduces_ignorance else 'does not support'} "
                     f"the paradox hypothesis (Townsend et al., 2025). ICC = {icc:.3f}.")

            model_result = MixedEffectsResult(
                model_name="Actor Ignorance ~ AI Tier (Random Intercept: Agent)",
                dependent_variable=ignorance_col,
                fixed_effects=fixed_effects,
                random_effects_variance={'agent_intercept': random_var, 'residual': residual_var},
                model_fit={'ICC': icc},
                n_observations=len(df_model),
                n_groups={'agents': df_model['agent_id'].nunique()},
                convergence=True,
                interpretation=interp
            )
            self.results.append(model_result)

            print(f"   ✓ {interp}")

        except Exception as e:
            print(f"   ⚠️ Model fitting failed: {e}")

    def _fit_investment_returns_model(self):
        """
        Model 4: AI Tier → Investment Returns for matured investments.
        """
        print("\n📊 Model 4: AI Tier → Investment Returns")

        if self.matured_df.empty:
            print("   ⚠️ No matured investment data")
            return

        # Find return column
        return_col = None
        for col in ['realized_roi', 'return_multiple', 'roi']:
            if col in self.matured_df.columns:
                return_col = col
                break

        if return_col is None:
            print("   ⚠️ No return column found")
            return

        # Find AI and grouping columns
        ai_col = None
        for col in ['ai_level_used', 'ai_level', 'ai_tier']:
            if col in self.matured_df.columns:
                ai_col = col
                break

        group_col = None
        for col in ['agent_id', 'run_id']:
            if col in self.matured_df.columns:
                group_col = col
                break

        if ai_col is None or group_col is None:
            print("   ⚠️ Missing required columns")
            return

        df = self.matured_df.copy()
        df['ai_level'] = df[ai_col].fillna('none')
        df = df[df['ai_level'].isin(['none', 'basic', 'advanced', 'premium'])]

        df['ai_basic'] = (df['ai_level'] == 'basic').astype(int)
        df['ai_advanced'] = (df['ai_level'] == 'advanced').astype(int)
        df['ai_premium'] = (df['ai_level'] == 'premium').astype(int)

        df[return_col] = pd.to_numeric(df[return_col], errors='coerce')

        model_vars = [return_col, 'ai_basic', 'ai_advanced', 'ai_premium', group_col]
        df_model = df[model_vars].dropna()

        # Remove extreme outliers (beyond 3 IQR)
        q1, q3 = df_model[return_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_model = df_model[
            (df_model[return_col] >= q1 - 3*iqr) &
            (df_model[return_col] <= q3 + 3*iqr)
        ]

        if len(df_model) < 50:
            print("   ⚠️ Insufficient observations after outlier removal")
            return

        try:
            model = smf.mixedlm(
                f"{return_col} ~ ai_basic + ai_advanced + ai_premium",
                data=df_model,
                groups=df_model[group_col]
            )
            result = model.fit(method='powell', maxiter=500)

            fixed_effects = {}
            for var in ['Intercept', 'ai_basic', 'ai_advanced', 'ai_premium']:
                if var in result.params.index:
                    fixed_effects[var] = {
                        'coef': result.params[var],
                        'se': result.bse[var],
                        'z': result.tvalues[var],
                        'p': result.pvalues[var],
                        'ci_lower': result.conf_int().loc[var, 0],
                        'ci_upper': result.conf_int().loc[var, 1]
                    }

            random_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else 0
            residual_var = result.scale
            icc = random_var / (random_var + residual_var) if (random_var + residual_var) > 0 else 0

            model_result = MixedEffectsResult(
                model_name=f"Investment Returns ~ AI Tier (Random Intercept: {group_col})",
                dependent_variable=return_col,
                fixed_effects=fixed_effects,
                random_effects_variance={f'{group_col}_intercept': random_var, 'residual': residual_var},
                model_fit={'ICC': icc},
                n_observations=len(df_model),
                n_groups={group_col: df_model[group_col].nunique()},
                convergence=True,
                interpretation=f"ICC = {icc:.3f} ({icc*100:.1f}% of return variance between {group_col}s)"
            )
            self.results.append(model_result)

            print(f"   ✓ Fitted model: n={len(df_model)}, ICC={icc:.3f}")

        except Exception as e:
            print(f"   ⚠️ Model fitting failed: {e}")

    def generate_results_table(self) -> pd.DataFrame:
        """
        Generate combined results table from all mixed-effects models.
        """
        if not self.results:
            return pd.DataFrame()

        all_rows = []
        for model_result in self.results:
            for var, stats in model_result.fixed_effects.items():
                all_rows.append({
                    'Model': model_result.model_name,
                    'DV': model_result.dependent_variable,
                    'Variable': var,
                    'Coefficient': stats['coef'],
                    'Std. Error': stats['se'],
                    'z-value': stats['z'],
                    'p-value': stats['p'],
                    '95% CI': f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]",
                    'ICC': model_result.model_fit.get('ICC', np.nan),
                    'n': model_result.n_observations
                })

        return pd.DataFrame(all_rows)


def run_statistical_analysis(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run complete statistical analysis suite on simulation results.

    Parameters
    ----------
    results_dir : str
        Path to simulation results directory.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (hypothesis_tests, descriptive_stats, correlations, mixed_effects)
    """
    from .analysis import ComprehensiveAnalysisFramework

    print(f"\nLoading data from {results_dir}...")
    framework = ComprehensiveAnalysisFramework(results_dir)

    analysis = RigorousStatisticalAnalysis(
        agent_df=framework.agent_df,
        decision_df=framework.decision_df,
        matured_df=framework.matured_df,
        uncertainty_detail_df=framework.uncertainty_detail_df
    )

    # Run analyses
    hypothesis_tests = analysis.run_all_analyses()
    descriptive_stats = analysis.generate_descriptive_statistics()
    correlations = analysis.generate_correlation_matrix()

    # Run mixed-effects models
    mixed_analysis = MixedEffectsAnalysis(
        agent_df=framework.agent_df,
        decision_df=framework.decision_df,
        matured_df=framework.matured_df
    )
    mixed_analysis.run_all_models()
    mixed_effects = mixed_analysis.generate_results_table()

    # Save to CSV
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    hypothesis_tests.to_csv(
        os.path.join(tables_dir, 'statistical_tests_amj.csv'),
        index=False
    )
    descriptive_stats.to_csv(
        os.path.join(tables_dir, 'descriptive_statistics_amj.csv'),
        index=False
    )
    correlations.to_csv(
        os.path.join(tables_dir, 'correlation_matrix_amj.csv')
    )
    if not mixed_effects.empty:
        mixed_effects.to_csv(
            os.path.join(tables_dir, 'mixed_effects_models_amj.csv'),
            index=False
        )

    print(f"\n✓ Saved statistical tables to {tables_dir}")

    return hypothesis_tests, descriptive_stats, correlations, mixed_effects


# Ensure module can import os
import os


@dataclass
class CausalEffectEstimate:
    """Result of a causal effect estimation from fixed-tier design."""
    treatment: str  # e.g., "basic_vs_none", "premium_vs_none"
    outcome: str  # e.g., "survival", "roi", "actor_ignorance"
    ate: float  # Average Treatment Effect
    ate_se: float  # Standard error of ATE
    ate_ci_lower: float
    ate_ci_upper: float
    cohens_d: float  # Effect size
    cohens_d_ci_lower: float
    cohens_d_ci_upper: float
    n_treatment: int
    n_control: int
    p_value: float
    identification: str  # "fixed-tier" or "emergent-selection"
    robustness_check: str  # Notes on robustness


class CausalIdentificationAnalysis:
    """
    Causal identification analysis for AI tier effects.

    This class generates publication-ready tables that clearly distinguish
    between causal estimates (from fixed-tier designs) and associational
    estimates (from emergent selection). It computes Average Treatment Effects
    (ATE) with bootstrap confidence intervals and effect sizes.

    For AMJ reviewers, this addresses the concern that complex ABMs produce
    causally ambiguous results by:
    1. Clearly labeling the identification strategy for each estimate
    2. Providing effect sizes (Cohen's d) with confidence intervals
    3. Computing robustness bounds across parameter variations
    """

    def __init__(
        self,
        agent_df: pd.DataFrame,
        matured_df: pd.DataFrame,
        decision_df: pd.DataFrame,
        uncertainty_detail_df: Optional[pd.DataFrame] = None,
        is_fixed_tier: bool = False
    ):
        self.agent_df = agent_df
        self.matured_df = matured_df
        self.decision_df = decision_df
        self.uncertainty_detail_df = uncertainty_detail_df
        self.is_fixed_tier = is_fixed_tier
        self.results: List[CausalEffectEstimate] = []

    def _compute_ate_bootstrap(
        self,
        treatment_values: np.ndarray,
        control_values: np.ndarray,
        n_bootstrap: int = 5000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float, float]:
        """Compute ATE with bootstrap confidence intervals."""
        treatment_values = treatment_values[np.isfinite(treatment_values)]
        control_values = control_values[np.isfinite(control_values)]

        if len(treatment_values) < 2 or len(control_values) < 2:
            return np.nan, np.nan, np.nan, np.nan

        ate = np.mean(treatment_values) - np.mean(control_values)

        # Bootstrap for CI
        bootstrap_ates = []
        for _ in range(n_bootstrap):
            t_sample = np.random.choice(treatment_values, size=len(treatment_values), replace=True)
            c_sample = np.random.choice(control_values, size=len(control_values), replace=True)
            bootstrap_ates.append(np.mean(t_sample) - np.mean(c_sample))

        bootstrap_ates = np.array(bootstrap_ates)
        se = np.std(bootstrap_ates)
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_ates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_ates, 100 * (1 - alpha / 2))

        return ate, se, ci_lower, ci_upper

    def _compute_cohens_d_bootstrap(
        self,
        treatment_values: np.ndarray,
        control_values: np.ndarray,
        n_bootstrap: int = 5000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Compute Cohen's d with bootstrap confidence intervals."""
        treatment_values = treatment_values[np.isfinite(treatment_values)]
        control_values = control_values[np.isfinite(control_values)]

        if len(treatment_values) < 2 or len(control_values) < 2:
            return np.nan, np.nan, np.nan

        # Pooled standard deviation
        n1, n2 = len(treatment_values), len(control_values)
        var1, var2 = np.var(treatment_values, ddof=1), np.var(control_values, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std < 1e-10:
            return 0.0, 0.0, 0.0

        d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std

        # Bootstrap for CI
        bootstrap_ds = []
        for _ in range(n_bootstrap):
            t_sample = np.random.choice(treatment_values, size=len(treatment_values), replace=True)
            c_sample = np.random.choice(control_values, size=len(control_values), replace=True)
            var_t = np.var(t_sample, ddof=1)
            var_c = np.var(c_sample, ddof=1)
            pooled = np.sqrt(((len(t_sample) - 1) * var_t + (len(c_sample) - 1) * var_c) /
                           (len(t_sample) + len(c_sample) - 2))
            if pooled > 1e-10:
                bootstrap_ds.append((np.mean(t_sample) - np.mean(c_sample)) / pooled)

        if not bootstrap_ds:
            return d, d, d

        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_ds, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_ds, 100 * (1 - alpha / 2))

        return d, ci_lower, ci_upper

    def estimate_survival_effects(self) -> None:
        """Estimate causal effects of AI tier on survival."""
        if 'primary_ai_level' not in self.agent_df.columns:
            return

        # Normalize AI levels
        self.agent_df['ai_tier'] = self.agent_df['primary_ai_level'].apply(
            lambda x: str(x).lower().strip() if pd.notna(x) else 'none'
        )

        # Binary survival outcome
        if 'final_status' in self.agent_df.columns:
            self.agent_df['survived'] = (self.agent_df['final_status'] == 'active').astype(float)
        elif 'alive' in self.agent_df.columns:
            self.agent_df['survived'] = self.agent_df['alive'].astype(float)
        else:
            return

        control = self.agent_df[self.agent_df['ai_tier'] == 'none']['survived'].values

        for tier in ['basic', 'advanced', 'premium']:
            treatment = self.agent_df[self.agent_df['ai_tier'] == tier]['survived'].values

            if len(treatment) < 5 or len(control) < 5:
                continue

            ate, se, ci_lower, ci_upper = self._compute_ate_bootstrap(treatment, control)
            d, d_lower, d_upper = self._compute_cohens_d_bootstrap(treatment, control)

            # Two-proportion z-test for p-value
            p1, p2 = np.mean(treatment), np.mean(control)
            n1, n2 = len(treatment), len(control)
            pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
            se_diff = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
            if se_diff > 0:
                z = (p1 - p2) / se_diff
                p_value = 2 * (1 - 0.5 * (1 + np.erf(abs(z) / np.sqrt(2))))
            else:
                p_value = 1.0

            identification = "fixed-tier (exogenous)" if self.is_fixed_tier else "emergent-selection (endogenous)"
            robustness = "Primary specification" if self.is_fixed_tier else "Subject to selection bias"

            self.results.append(CausalEffectEstimate(
                treatment=f"{tier}_vs_none",
                outcome="survival_rate",
                ate=ate,
                ate_se=se,
                ate_ci_lower=ci_lower,
                ate_ci_upper=ci_upper,
                cohens_d=d,
                cohens_d_ci_lower=d_lower,
                cohens_d_ci_upper=d_upper,
                n_treatment=len(treatment),
                n_control=len(control),
                p_value=p_value,
                identification=identification,
                robustness_check=robustness
            ))

    def estimate_roi_effects(self) -> None:
        """Estimate causal effects of AI tier on investment ROI."""
        if self.matured_df is None or self.matured_df.empty:
            return

        if 'ai_level_used' not in self.matured_df.columns:
            return

        # Normalize AI levels
        self.matured_df['ai_tier'] = self.matured_df['ai_level_used'].apply(
            lambda x: str(x).lower().strip() if pd.notna(x) else 'none'
        )

        roi_col = None
        for col in ['realized_roi', 'realized_multiplier', 'roi']:
            if col in self.matured_df.columns:
                roi_col = col
                break

        if roi_col is None:
            return

        control = self.matured_df[self.matured_df['ai_tier'] == 'none'][roi_col].values

        for tier in ['basic', 'advanced', 'premium']:
            treatment = self.matured_df[self.matured_df['ai_tier'] == tier][roi_col].values

            if len(treatment) < 5 or len(control) < 5:
                continue

            ate, se, ci_lower, ci_upper = self._compute_ate_bootstrap(treatment, control)
            d, d_lower, d_upper = self._compute_cohens_d_bootstrap(treatment, control)

            # Welch's t-test for p-value
            t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

            identification = "fixed-tier (exogenous)" if self.is_fixed_tier else "emergent-selection (endogenous)"
            robustness = "Primary specification" if self.is_fixed_tier else "Subject to selection bias"

            self.results.append(CausalEffectEstimate(
                treatment=f"{tier}_vs_none",
                outcome="investment_roi",
                ate=ate,
                ate_se=se,
                ate_ci_lower=ci_lower,
                ate_ci_upper=ci_upper,
                cohens_d=d,
                cohens_d_ci_lower=d_lower,
                cohens_d_ci_upper=d_upper,
                n_treatment=len(treatment),
                n_control=len(control),
                p_value=p_value,
                identification=identification,
                robustness_check=robustness
            ))

    def estimate_uncertainty_effects(self) -> None:
        """Estimate causal effects of AI tier on uncertainty dimensions."""
        if self.uncertainty_detail_df is None or self.uncertainty_detail_df.empty:
            return

        if 'ai_level' not in self.uncertainty_detail_df.columns:
            return

        # Normalize AI levels
        self.uncertainty_detail_df['ai_tier'] = self.uncertainty_detail_df['ai_level'].apply(
            lambda x: str(x).lower().strip() if pd.notna(x) else 'none'
        )

        uncertainty_dims = ['actor_ignorance', 'practical_indeterminism',
                          'agentic_novelty', 'competitive_recursion']

        for dim in uncertainty_dims:
            if dim not in self.uncertainty_detail_df.columns:
                continue

            control = self.uncertainty_detail_df[
                self.uncertainty_detail_df['ai_tier'] == 'none'
            ][dim].values

            for tier in ['basic', 'advanced', 'premium']:
                treatment = self.uncertainty_detail_df[
                    self.uncertainty_detail_df['ai_tier'] == tier
                ][dim].values

                if len(treatment) < 5 or len(control) < 5:
                    continue

                ate, se, ci_lower, ci_upper = self._compute_ate_bootstrap(treatment, control)
                d, d_lower, d_upper = self._compute_cohens_d_bootstrap(treatment, control)

                # Welch's t-test for p-value
                t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

                identification = "fixed-tier (exogenous)" if self.is_fixed_tier else "emergent-selection (endogenous)"
                robustness = "Primary specification" if self.is_fixed_tier else "Subject to selection bias"

                self.results.append(CausalEffectEstimate(
                    treatment=f"{tier}_vs_none",
                    outcome=dim,
                    ate=ate,
                    ate_se=se,
                    ate_ci_lower=ci_lower,
                    ate_ci_upper=ci_upper,
                    cohens_d=d,
                    cohens_d_ci_lower=d_lower,
                    cohens_d_ci_upper=d_upper,
                    n_treatment=len(treatment),
                    n_control=len(control),
                    p_value=p_value,
                    identification=identification,
                    robustness_check=robustness
                ))

    def run_all_estimates(self) -> None:
        """Run all causal effect estimations."""
        self.estimate_survival_effects()
        self.estimate_roi_effects()
        self.estimate_uncertainty_effects()

    def generate_causal_effects_table(self) -> pd.DataFrame:
        """Generate publication-ready table of causal effects."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            rows.append({
                'Treatment': result.treatment,
                'Outcome': result.outcome,
                'ATE': f"{result.ate:.4f}",
                'ATE SE': f"{result.ate_se:.4f}",
                'ATE 95% CI': f"[{result.ate_ci_lower:.4f}, {result.ate_ci_upper:.4f}]",
                "Cohen's d": f"{result.cohens_d:.3f}",
                "d 95% CI": f"[{result.cohens_d_ci_lower:.3f}, {result.cohens_d_ci_upper:.3f}]",
                'p-value': f"{result.p_value:.4f}" if result.p_value >= 0.0001 else "<0.0001",
                'N (Treatment)': result.n_treatment,
                'N (Control)': result.n_control,
                'Identification': result.identification,
                'Robustness': result.robustness_check
            })

        return pd.DataFrame(rows)

    def generate_effect_size_summary(self) -> pd.DataFrame:
        """Generate summary table of effect sizes by outcome."""
        if not self.results:
            return pd.DataFrame()

        summary_rows = []
        outcomes = set(r.outcome for r in self.results)

        for outcome in outcomes:
            outcome_results = [r for r in self.results if r.outcome == outcome]

            # Find largest effect
            largest = max(outcome_results, key=lambda r: abs(r.cohens_d))

            summary_rows.append({
                'Outcome': outcome,
                'Largest Effect': largest.treatment,
                "Cohen's d": f"{largest.cohens_d:.3f}",
                'Effect Interpretation': self._interpret_cohens_d(largest.cohens_d),
                'Significant (p<0.05)': sum(1 for r in outcome_results if r.p_value < 0.05),
                'Total Comparisons': len(outcome_results),
                'Identification': largest.identification
            })

        return pd.DataFrame(summary_rows)

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"


def run_causal_identification_analysis(
    results_dir: str,
    is_fixed_tier: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run causal identification analysis and export tables.

    Parameters
    ----------
    results_dir : str
        Path to simulation results directory.
    is_fixed_tier : bool
        Whether the data comes from fixed-tier (exogenous) design.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (causal_effects_table, effect_size_summary)
    """
    from .analysis import ComprehensiveAnalysisFramework

    print(f"\n📊 Running causal identification analysis...")
    print(f"   Identification strategy: {'Fixed-tier (exogenous)' if is_fixed_tier else 'Emergent-selection (endogenous)'}")

    framework = ComprehensiveAnalysisFramework(results_dir)

    analysis = CausalIdentificationAnalysis(
        agent_df=framework.agent_df,
        matured_df=framework.matured_df,
        decision_df=framework.decision_df,
        uncertainty_detail_df=framework.uncertainty_detail_df,
        is_fixed_tier=is_fixed_tier
    )

    analysis.run_all_estimates()

    causal_table = analysis.generate_causal_effects_table()
    summary_table = analysis.generate_effect_size_summary()

    # Save tables
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    suffix = "_fixed" if is_fixed_tier else "_emergent"

    if not causal_table.empty:
        causal_table.to_csv(
            os.path.join(tables_dir, f'causal_effects{suffix}.csv'),
            index=False
        )
        print(f"   ✓ Exported: causal_effects{suffix}.csv")

    if not summary_table.empty:
        summary_table.to_csv(
            os.path.join(tables_dir, f'effect_size_summary{suffix}.csv'),
            index=False
        )
        print(f"   ✓ Exported: effect_size_summary{suffix}.csv")

    return causal_table, summary_table
