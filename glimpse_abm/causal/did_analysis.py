"""
Difference-in-Differences Analysis for Glimpse ABM.

DiD compares changes in outcomes over time between a treatment group
(AI adopters) and a control group (non-adopters).

References
----------
Angrist, J. D., & Pischke, J. S. (2009). Mostly harmless econometrics.
    Princeton University Press.

Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with
    multiple time periods. Journal of Econometrics, 225(2), 200-230.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DiDResult:
    """Results from Difference-in-Differences analysis."""
    outcome: str
    att: float  # Average Treatment Effect on Treated
    att_se: float
    att_ci_lower: float
    att_ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    n_periods_pre: int
    n_periods_post: int
    parallel_trends_test: Optional[Dict[str, float]]
    interpretation: str


class DifferenceInDifferencesAnalysis:
    """
    Difference-in-Differences Analysis for AI Adoption Effects.

    DiD compares changes in outcomes over time between a treatment group
    (AI adopters) and a control group (non-adopters). The key identifying
    assumption is that, absent treatment, both groups would have followed
    parallel trends.

    Design Variants
    ---------------
    1. Classic 2x2 DiD: One pre-period, one post-period, binary treatment
    2. Staggered DiD: Different units adopt at different times
    3. Event Study: Dynamic effects before and after adoption

    In the ABM Context
    ------------------
    - Treatment: AI adoption (by tier)
    - Treatment timing: Step when agent first uses AI
    - Pre-period: Steps before adoption
    - Post-period: Steps after adoption
    - Control: Agents who never adopt AI

    Key Assumptions
    ---------------
    1. Parallel trends: E[Y(0)_t | D=1] - E[Y(0)_t-1 | D=1] =
                        E[Y(0)_t | D=0] - E[Y(0)_t-1 | D=0]
    2. No anticipation: Treatment doesn't affect outcomes before it occurs
    3. SUTVA: No spillovers between treatment and control
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,  # Long-format panel: agent × time
        agent_df: Optional[pd.DataFrame] = None
    ):
        self.panel_df = panel_df.copy()
        self.agent_df = agent_df.copy() if agent_df is not None else None
        self.results: List[DiDResult] = []

    def prepare_did_data(
        self,
        agent_id_col: str = 'agent_id',
        time_col: str = 'step',
        treatment_col: str = 'uses_ai',
        adoption_time_col: Optional[str] = 'first_ai_step'
    ) -> pd.DataFrame:
        """
        Prepare panel data for DiD analysis.

        Creates:
        - post: indicator for post-treatment period
        - treated: indicator for treatment group
        - did_term: interaction (treated × post)
        """
        df = self.panel_df.copy()

        if agent_id_col not in df.columns or time_col not in df.columns:
            print(f"   ⚠️ Missing columns: {agent_id_col} or {time_col}")
            return pd.DataFrame()

        # Identify adoption timing
        if adoption_time_col and adoption_time_col in df.columns:
            # Use existing adoption time column
            pass
        elif treatment_col in df.columns:
            # Compute first treatment time per agent
            treated_obs = df[df[treatment_col] == 1]
            if len(treated_obs) > 0:
                first_treat = treated_obs.groupby(agent_id_col)[time_col].min()
                df[adoption_time_col] = df[agent_id_col].map(first_treat)
            else:
                df[adoption_time_col] = np.nan
        else:
            print("   ⚠️ Cannot determine treatment timing")
            return pd.DataFrame()

        # Create DiD terms
        df['treated'] = df[adoption_time_col].notna().astype(int)
        df['post'] = (df[time_col] >= df[adoption_time_col]).fillna(False).astype(int)
        df['did_term'] = df['treated'] * df['post']

        # Event time (relative to adoption)
        df['event_time'] = df[time_col] - df[adoption_time_col]

        return df

    def estimate_twfe_did(
        self,
        outcome_col: str,
        agent_id_col: str = 'agent_id',
        time_col: str = 'step',
        covariates: Optional[List[str]] = None
    ) -> Optional[DiDResult]:
        """
        Estimate DiD using Two-Way Fixed Effects (TWFE) regression.

        Model: Y_it = α_i + λ_t + τ*DiD_it + β*X_it + ε_it

        Where:
        - α_i: Agent fixed effects
        - λ_t: Time fixed effects
        - τ: DiD treatment effect (ATT)
        - X_it: Time-varying covariates

        Note: TWFE can be biased with staggered adoption and heterogeneous
        effects. For robustness, also run event study specification.
        """
        print(f"\n📊 Estimating Two-Way Fixed Effects DiD for {outcome_col}...")

        df = self.prepare_did_data(agent_id_col, time_col)
        if df.empty:
            return None

        if outcome_col not in df.columns:
            print(f"   ⚠️ Outcome column {outcome_col} not found")
            return None

        # Subset to complete cases
        model_cols = [outcome_col, agent_id_col, time_col, 'treated', 'post', 'did_term']
        if covariates:
            model_cols.extend([c for c in covariates if c in df.columns])

        df_model = df[model_cols].dropna()

        if len(df_model) < 100:
            print("   ⚠️ Insufficient observations")
            return None

        n_treated = df_model[df_model['treated'] == 1][agent_id_col].nunique()
        n_control = df_model[df_model['treated'] == 0][agent_id_col].nunique()

        if n_treated < 5 or n_control < 5:
            print(f"   ⚠️ Insufficient treated ({n_treated}) or control ({n_control}) units")
            return None

        try:
            # Demean for fixed effects (within transformation)
            # Agent-specific means
            agent_means = df_model.groupby(agent_id_col)[outcome_col].transform('mean')
            # Time-specific means
            time_means = df_model.groupby(time_col)[outcome_col].transform('mean')
            # Grand mean
            grand_mean = df_model[outcome_col].mean()

            # Within transformation
            df_model['Y_within'] = df_model[outcome_col] - agent_means - time_means + grand_mean

            # Similarly for regressors
            for col in ['did_term', 'post']:
                agent_m = df_model.groupby(agent_id_col)[col].transform('mean')
                time_m = df_model.groupby(time_col)[col].transform('mean')
                grand_m = df_model[col].mean()
                df_model[f'{col}_within'] = df_model[col] - agent_m - time_m + grand_m

            # OLS on transformed data
            X = df_model['did_term_within'].values.reshape(-1, 1)
            y = df_model['Y_within'].values

            # Add constant if needed (should be ~0 after demeaning)
            X = np.column_stack([np.ones(len(X)), X])

            # OLS
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            att = beta[1]  # DiD coefficient

            # Clustered standard errors (by agent)
            residuals = y - X @ beta
            df_model['resid'] = residuals

            # Cluster-robust variance
            clusters = df_model[agent_id_col].values
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters)

            # Meat of sandwich estimator
            meat = np.zeros((2, 2))
            for c in unique_clusters:
                mask = clusters == c
                Xc = X[mask]
                rc = residuals[mask]
                meat += (Xc.T @ np.outer(rc, rc) @ Xc)

            # Bread
            bread = np.linalg.inv(X.T @ X)

            # Cluster-robust variance
            var_robust = bread @ meat @ bread * (n_clusters / (n_clusters - 1))
            se_att = np.sqrt(var_robust[1, 1])

            # CI and p-value
            ci_lower = att - 1.96 * se_att
            ci_upper = att + 1.96 * se_att
            t_stat = att / se_att
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_clusters - 1))

            # Count pre/post periods
            n_pre = df_model[df_model['post'] == 0][time_col].nunique()
            n_post = df_model[df_model['post'] == 1][time_col].nunique()

            # Interpretation
            if p_value < 0.05:
                direction = "increases" if att > 0 else "decreases"
                interpretation = (
                    f"AI adoption {direction} {outcome_col} by {abs(att):.4f} "
                    f"(ATT={att:.4f}, SE={se_att:.4f}, p={p_value:.4f}). "
                    f"Identification: DiD with {n_treated} treated, {n_control} control agents."
                )
            else:
                interpretation = (
                    f"No significant effect of AI adoption on {outcome_col} "
                    f"(ATT={att:.4f}, p={p_value:.4f})."
                )

            result = DiDResult(
                outcome=outcome_col,
                att=att,
                att_se=se_att,
                att_ci_lower=ci_lower,
                att_ci_upper=ci_upper,
                p_value=p_value,
                n_treated=n_treated,
                n_control=n_control,
                n_periods_pre=n_pre,
                n_periods_post=n_post,
                parallel_trends_test=None,  # Would need pre-trend test
                interpretation=interpretation
            )

            self.results.append(result)

            print(f"   ✓ DiD estimate: ATT = {att:.4f} (SE = {se_att:.4f})")
            print(f"   ✓ 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"   ✓ p-value: {p_value:.4f} (clustered by agent, {n_clusters} clusters)")
            print(f"   ✓ Units: {n_treated} treated, {n_control} control")
            print(f"   ✓ Periods: {n_pre} pre, {n_post} post")
            print(f"\n   {interpretation}")

            return result

        except Exception as e:
            print(f"   ⚠️ TWFE DiD estimation failed: {e}")
            return None

    def event_study(
        self,
        outcome_col: str,
        agent_id_col: str = 'agent_id',
        time_col: str = 'step',
        event_window: Tuple[int, int] = (-5, 10)
    ) -> pd.DataFrame:
        """
        Estimate event study (dynamic DiD) specification.

        Model: Y_it = α_i + λ_t + Σ_k τ_k * 1{t - E_i = k} + ε_it

        Where E_i is the adoption time for unit i, and τ_k are the
        dynamic treatment effects at each event time k.

        Returns DataFrame of coefficients by event time.
        """
        print(f"\n📊 Estimating Event Study for {outcome_col}...")

        df = self.prepare_did_data(agent_id_col, time_col)
        if df.empty or outcome_col not in df.columns:
            return pd.DataFrame()

        # Restrict to event window
        min_k, max_k = event_window
        df_es = df[
            (df['event_time'] >= min_k) &
            (df['event_time'] <= max_k) &
            df['event_time'].notna()
        ].copy()

        if len(df_es) < 50:
            print("   ⚠️ Insufficient observations in event window")
            return pd.DataFrame()

        # Reference period: k = -1 (period just before treatment)
        ref_period = -1

        # Create event time dummies
        event_times = sorted(df_es['event_time'].unique())
        event_times = [k for k in event_times if k != ref_period]

        results = []

        for k in event_times:
            # Simple comparison at each event time
            # (In full implementation, would use full regression with all dummies)
            at_k = df_es[df_es['event_time'] == k]
            at_ref = df_es[df_es['event_time'] == ref_period]

            if len(at_k) < 10 or len(at_ref) < 10:
                continue

            # Difference in means (simplified)
            coef = at_k[outcome_col].mean() - at_ref[outcome_col].mean()
            se = np.sqrt(
                at_k[outcome_col].var() / len(at_k) +
                at_ref[outcome_col].var() / len(at_ref)
            )

            results.append({
                'event_time': k,
                'coefficient': coef,
                'se': se,
                'ci_lower': coef - 1.96 * se,
                'ci_upper': coef + 1.96 * se,
                'n': len(at_k)
            })

        es_df = pd.DataFrame(results)

        if not es_df.empty:
            # Check pre-trends (coefficients before t=0 should be ~0)
            pre_coefs = es_df[es_df['event_time'] < 0]['coefficient']
            if len(pre_coefs) > 0:
                pre_trend_test = stats.ttest_1samp(pre_coefs, 0)
                print(f"   Pre-trend test: mean={pre_coefs.mean():.4f}, p={pre_trend_test.pvalue:.4f}")
                if pre_trend_test.pvalue < 0.05:
                    print("   ⚠️ Warning: Pre-trends detected (parallel trends assumption may be violated)")
                else:
                    print("   ✓ No significant pre-trends (parallel trends plausible)")

        return es_df

    def generate_results_table(self) -> pd.DataFrame:
        """Generate publication-ready DiD results table."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            rows.append({
                'Outcome': r.outcome,
                'ATT': f"{r.att:.4f}",
                'Std. Error': f"{r.att_se:.4f}",
                '95% CI': f"[{r.att_ci_lower:.4f}, {r.att_ci_upper:.4f}]",
                'p-value': f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<0.0001",
                'N (Treated)': r.n_treated,
                'N (Control)': r.n_control,
                'Pre-periods': r.n_periods_pre,
                'Post-periods': r.n_periods_post
            })

        return pd.DataFrame(rows)
