"""
Propensity Score Methods for Causal Inference in Glimpse ABM.

This module implements propensity score methods to address selection bias
when agents self-select into AI adoption.

References
----------
Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies for causal effects.
    Biometrika, 70(1), 41-55.

Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing
    data and causal inference models. Biometrics, 61(4), 962-973.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# Import bootstrap iteration helper from parent module
def get_bootstrap_iterations(full_iterations: int = 5000) -> int:
    """Get bootstrap iterations based on current mode."""
    try:
        from ..statistical_tests import get_bootstrap_iterations as _get_bootstrap
        return _get_bootstrap(full_iterations)
    except ImportError:
        return min(500, full_iterations)  # Default to fast mode


@dataclass
class PropensityScoreResult:
    """Results from propensity score analysis."""
    method: str  # 'matching', 'ipw', 'aipw'
    outcome: str
    ate: float  # Average Treatment Effect
    att: float  # Average Treatment Effect on Treated
    ate_se: float
    att_se: float
    ate_ci: Tuple[float, float]
    att_ci: Tuple[float, float]
    p_value_ate: float
    p_value_att: float
    n_treated: int
    n_control: int
    n_matched: Optional[int]  # For matching methods
    balance_before: Dict[str, float]  # SMD before adjustment
    balance_after: Dict[str, float]  # SMD after adjustment
    overlap_summary: Dict[str, float]  # Propensity score distribution stats
    interpretation: str


class PropensityScoreAnalysis:
    """
    Propensity Score Methods for Causal Inference in Emergent AI Adoption.

    This class implements propensity score methods to address selection bias
    when agents self-select into AI adoption. These methods are appropriate
    for the emergent design where AI adoption is endogenous.

    Theoretical Foundation
    ----------------------
    The propensity score e(X) = P(T=1|X) is a balancing score: conditional
    on e(X), the distribution of X is the same for treated and control units.

    Under the assumptions of:
    1. Unconfoundedness: Y(0), Y(1) ⊥ T | X
    2. Positivity: 0 < P(T=1|X) < 1 for all X
    3. SUTVA: No interference between units

    We can identify causal effects by adjusting for the propensity score.

    Methods Implemented
    -------------------
    1. Propensity Score Matching (PSM): Match treated to control units
    2. Inverse Probability Weighting (IPW): Weight by inverse propensity
    3. Augmented IPW (AIPW): Doubly-robust combination of IPW + outcome model
    """

    def __init__(
        self,
        agent_df: pd.DataFrame,
        treatment_col: str = 'uses_ai',
        covariate_cols: Optional[List[str]] = None
    ):
        self.agent_df = agent_df.copy()
        self.treatment_col = treatment_col
        self.covariate_cols = covariate_cols
        self.propensity_scores: Optional[np.ndarray] = None
        self.propensity_model = None
        self.results: List[PropensityScoreResult] = []

    def _identify_covariates(self) -> List[str]:
        """Identify covariates for propensity score model."""
        if self.covariate_cols is not None:
            return [c for c in self.covariate_cols if c in self.agent_df.columns]

        # Auto-identify reasonable covariates
        potential_covariates = [
            'initial_capital', 'log_initial_capital',
            'sector', 'risk_tolerance', 'innovation_propensity',
            'starting_capital', 'capital_at_start',
            'age', 'experience', 'education_level'
        ]

        available = [c for c in potential_covariates if c in self.agent_df.columns]

        # Add any numeric columns that look like pre-treatment characteristics
        for col in self.agent_df.columns:
            if col not in available and col != self.treatment_col:
                if self.agent_df[col].dtype in [np.float64, np.int64]:
                    if 'initial' in col.lower() or 'start' in col.lower():
                        available.append(col)

        return available

    def _prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for propensity score analysis."""
        df = self.agent_df.copy()

        # Ensure treatment is binary
        if self.treatment_col not in df.columns:
            # Try to create from AI level
            if 'primary_ai_canonical' in df.columns:
                df[self.treatment_col] = (df['primary_ai_canonical'] != 'none').astype(int)
            elif 'ai_level' in df.columns:
                df[self.treatment_col] = (df['ai_level'] != 'none').astype(int)
            else:
                raise ValueError(f"Treatment column '{self.treatment_col}' not found")

        covariates = self._identify_covariates()

        if not covariates:
            raise ValueError("No covariates identified for propensity score model")

        # Handle categorical variables
        for col in covariates.copy():
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Create dummies
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                covariates.remove(col)
                covariates.extend(dummies.columns.tolist())

        # Add log capital if not present
        if 'initial_capital' in df.columns and 'log_initial_capital' not in covariates:
            df['log_initial_capital'] = np.log1p(df['initial_capital'].clip(lower=1))
            if 'log_initial_capital' not in covariates:
                covariates.append('log_initial_capital')

        return df, covariates

    def estimate_propensity_scores(
        self,
        method: str = 'logistic'
    ) -> np.ndarray:
        """
        Estimate propensity scores using logistic regression.

        Parameters
        ----------
        method : str
            Estimation method: 'logistic' (default) or 'gbm' (gradient boosting)

        Returns
        -------
        np.ndarray
            Estimated propensity scores for each observation.
        """
        print("\n📊 Estimating Propensity Scores...")

        df, covariates = self._prepare_data()

        # Prepare model data
        model_cols = [self.treatment_col] + covariates
        df_model = df[model_cols].dropna()

        if len(df_model) < 50:
            raise ValueError("Insufficient observations for propensity score estimation")

        X = df_model[covariates].values
        T = df_model[self.treatment_col].values

        print(f"   Covariates: {covariates}")
        print(f"   N = {len(df_model)}, Treated = {T.sum()}, Control = {len(T) - T.sum()}")

        if method == 'logistic':
            # Logistic regression
            from scipy.special import expit

            # Add intercept
            X_design = np.column_stack([np.ones(len(X)), X])

            # Fit via iteratively reweighted least squares
            beta = np.zeros(X_design.shape[1])

            for _ in range(25):  # Max iterations
                p = expit(X_design @ beta)
                p = np.clip(p, 1e-10, 1 - 1e-10)  # Numerical stability
                W = np.diag(p * (1 - p))
                z = X_design @ beta + (T - p) / (p * (1 - p) + 1e-10)

                try:
                    beta_new = np.linalg.solve(X_design.T @ W @ X_design, X_design.T @ W @ z)
                    if np.max(np.abs(beta_new - beta)) < 1e-6:
                        break
                    beta = beta_new
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse if singular
                    beta = np.linalg.lstsq(X_design.T @ W @ X_design, X_design.T @ W @ z, rcond=None)[0]
                    break

            propensity_scores = expit(X_design @ beta)
            self.propensity_model = {'method': 'logistic', 'coefficients': beta, 'covariates': covariates}

        else:
            raise ValueError(f"Unknown method: {method}")

        # Store scores
        self.propensity_scores = propensity_scores
        self.agent_df = self.agent_df.loc[df_model.index].copy()
        self.agent_df['propensity_score'] = propensity_scores
        self.agent_df['treatment'] = T

        # Summary statistics
        ps_treated = propensity_scores[T == 1]
        ps_control = propensity_scores[T == 0]

        print(f"\n   Propensity Score Distribution:")
        print(f"      Treated:  mean={ps_treated.mean():.3f}, std={ps_treated.std():.3f}, "
              f"range=[{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
        print(f"      Control:  mean={ps_control.mean():.3f}, std={ps_control.std():.3f}, "
              f"range=[{ps_control.min():.3f}, {ps_control.max():.3f}]")

        # Check overlap
        overlap_min = max(ps_treated.min(), ps_control.min())
        overlap_max = min(ps_treated.max(), ps_control.max())
        in_overlap = ((propensity_scores >= overlap_min) & (propensity_scores <= overlap_max)).mean()

        print(f"\n   Overlap region: [{overlap_min:.3f}, {overlap_max:.3f}]")
        print(f"   Observations in overlap: {in_overlap*100:.1f}%")

        if in_overlap < 0.5:
            print("   ⚠️ Warning: Limited overlap - positivity assumption may be violated")

        return propensity_scores

    def compute_balance_statistics(
        self,
        weights: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute covariate balance statistics (standardized mean differences).

        SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)

        SMD < 0.1 is typically considered good balance.
        """
        df, covariates = self._prepare_data()
        df = df.loc[self.agent_df.index]

        T = self.agent_df['treatment'].values
        treated_mask = T == 1
        control_mask = T == 0

        balance_stats = []

        for cov in covariates:
            if cov not in df.columns:
                continue

            x = df[cov].values

            if weights is None:
                mean_t = x[treated_mask].mean()
                mean_c = x[control_mask].mean()
                var_t = x[treated_mask].var()
                var_c = x[control_mask].var()
            else:
                w_t = weights[treated_mask]
                w_c = weights[control_mask]
                mean_t = np.average(x[treated_mask], weights=w_t)
                mean_c = np.average(x[control_mask], weights=w_c)
                var_t = np.average((x[treated_mask] - mean_t)**2, weights=w_t)
                var_c = np.average((x[control_mask] - mean_c)**2, weights=w_c)

            pooled_sd = np.sqrt((var_t + var_c) / 2)

            if pooled_sd > 0:
                smd = (mean_t - mean_c) / pooled_sd
            else:
                smd = 0.0

            balance_stats.append({
                'covariate': cov,
                'mean_treated': mean_t,
                'mean_control': mean_c,
                'smd': smd,
                'abs_smd': abs(smd),
                'balanced': abs(smd) < 0.1
            })

        return pd.DataFrame(balance_stats)

    def nearest_neighbor_matching(
        self,
        outcome_col: str,
        n_neighbors: int = 1,
        caliper: Optional[float] = 0.2,
        with_replacement: bool = True
    ) -> Optional[PropensityScoreResult]:
        """
        Propensity Score Matching using nearest neighbor.

        Parameters
        ----------
        outcome_col : str
            Column name for the outcome variable.
        n_neighbors : int
            Number of control matches per treated unit.
        caliper : float, optional
            Maximum distance for a valid match (in SD of propensity score).
        with_replacement : bool
            Whether control units can be matched multiple times.
        """
        print(f"\n📊 Propensity Score Matching for {outcome_col}...")

        if self.propensity_scores is None:
            self.estimate_propensity_scores()

        df = self.agent_df.copy()

        if outcome_col not in df.columns:
            print(f"   ⚠️ Outcome column '{outcome_col}' not found")
            return None

        T = df['treatment'].values
        ps = df['propensity_score'].values
        Y = df[outcome_col].values

        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        if len(treated_idx) < 5 or len(control_idx) < 5:
            print("   ⚠️ Insufficient treated or control units")
            return None

        # Caliper in SD units
        if caliper is not None:
            caliper_abs = caliper * ps.std()
        else:
            caliper_abs = np.inf

        # Perform matching
        matched_treated = []
        matched_control = []
        control_used = set() if not with_replacement else None

        for t_idx in treated_idx:
            ps_t = ps[t_idx]
            distances = np.abs(ps[control_idx] - ps_t)
            valid = distances <= caliper_abs

            if not with_replacement:
                valid = valid & np.array([c not in control_used for c in control_idx])

            valid_idx = np.where(valid)[0]

            if len(valid_idx) == 0:
                continue

            sorted_idx = valid_idx[np.argsort(distances[valid_idx])]
            matches = sorted_idx[:n_neighbors]

            for m in matches:
                matched_treated.append(t_idx)
                matched_control.append(control_idx[m])
                if not with_replacement:
                    control_used.add(control_idx[m])

        if len(matched_treated) < 10:
            print(f"   ⚠️ Only {len(matched_treated)} matches found")
            return None

        matched_treated = np.array(matched_treated)
        matched_control = np.array(matched_control)

        # Compute ATT
        Y_treated = Y[matched_treated]
        Y_control = Y[matched_control]
        att = np.mean(Y_treated - Y_control)

        # Bootstrap SE
        n_boot = get_bootstrap_iterations(1000)
        boot_atts = []
        for _ in range(n_boot):
            boot_idx = np.random.choice(len(matched_treated), size=len(matched_treated), replace=True)
            boot_att = np.mean(Y_treated[boot_idx] - Y_control[boot_idx])
            boot_atts.append(boot_att)

        att_se = np.std(boot_atts)
        att_ci = (np.percentile(boot_atts, 2.5), np.percentile(boot_atts, 97.5))

        t_stat = att / att_se if att_se > 0 else 0
        p_value_att = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        ate = att
        ate_se = att_se
        ate_ci = att_ci
        p_value_ate = p_value_att

        balance_before = self.compute_balance_statistics()

        matched_weights = np.zeros(len(df))
        for t_idx, c_idx in zip(matched_treated, matched_control):
            matched_weights[t_idx] = 1.0
            matched_weights[c_idx] += 1.0

        balance_after = self.compute_balance_statistics(weights=matched_weights)

        n_unique_treated = len(np.unique(matched_treated))
        n_unique_control = len(np.unique(matched_control))

        if p_value_att < 0.05:
            direction = "increases" if att > 0 else "decreases"
            interpretation = (
                f"AI adoption {direction} {outcome_col} by {abs(att):.4f} "
                f"(ATT={att:.4f}, SE={att_se:.4f}, p={p_value_att:.4f}). "
                f"Matched {n_unique_treated} treated to {n_unique_control} control units."
            )
        else:
            interpretation = (
                f"No significant effect of AI adoption on {outcome_col} "
                f"(ATT={att:.4f}, p={p_value_att:.4f})."
            )

        ps_treated = ps[T == 1]
        ps_control = ps[T == 0]
        overlap_summary = {
            'ps_mean_treated': ps_treated.mean(),
            'ps_mean_control': ps_control.mean(),
            'ps_overlap_pct': ((ps >= max(ps_treated.min(), ps_control.min())) &
                              (ps <= min(ps_treated.max(), ps_control.max()))).mean()
        }

        result = PropensityScoreResult(
            method='nearest_neighbor_matching',
            outcome=outcome_col,
            ate=ate,
            att=att,
            ate_se=ate_se,
            att_se=att_se,
            ate_ci=ate_ci,
            att_ci=att_ci,
            p_value_ate=p_value_ate,
            p_value_att=p_value_att,
            n_treated=n_unique_treated,
            n_control=n_unique_control,
            n_matched=len(matched_treated),
            balance_before={row['covariate']: row['smd'] for _, row in balance_before.iterrows()},
            balance_after={row['covariate']: row['smd'] for _, row in balance_after.iterrows()},
            overlap_summary=overlap_summary,
            interpretation=interpretation
        )

        self.results.append(result)

        print(f"   ✓ Matched {n_unique_treated} treated to {n_unique_control} control units")
        print(f"   ✓ ATT = {att:.4f} (SE = {att_se:.4f})")
        print(f"   ✓ 95% CI: [{att_ci[0]:.4f}, {att_ci[1]:.4f}]")
        print(f"   ✓ p-value: {p_value_att:.4f}")

        smd_before = balance_before['abs_smd'].mean()
        smd_after = balance_after['abs_smd'].mean()
        print(f"\n   Balance (mean |SMD|): {smd_before:.3f} → {smd_after:.3f}")

        if smd_after > 0.1:
            print("   ⚠️ Warning: Residual imbalance after matching (mean |SMD| > 0.1)")

        print(f"\n   {interpretation}")

        return result

    def inverse_probability_weighting(
        self,
        outcome_col: str,
        trim_weights: bool = True,
        trim_quantile: float = 0.99
    ) -> Optional[PropensityScoreResult]:
        """
        Inverse Probability Weighting (IPW) estimator.

        Weights:
        - Treated: w = 1 / e(X)
        - Control: w = 1 / (1 - e(X))

        For ATT:
        - Treated: w = 1
        - Control: w = e(X) / (1 - e(X))
        """
        print(f"\n📊 Inverse Probability Weighting for {outcome_col}...")

        if self.propensity_scores is None:
            self.estimate_propensity_scores()

        df = self.agent_df.copy()

        if outcome_col not in df.columns:
            print(f"   ⚠️ Outcome column '{outcome_col}' not found")
            return None

        T = df['treatment'].values
        ps = df['propensity_score'].values
        Y = df[outcome_col].values

        valid = (ps > 0.01) & (ps < 0.99)
        T, ps, Y = T[valid], ps[valid], Y[valid]

        if len(T) < 50:
            print("   ⚠️ Insufficient observations after trimming extreme propensities")
            return None

        weights_ate = T / ps + (1 - T) / (1 - ps)
        weights_att_treated = np.ones_like(ps)
        weights_att_control = ps / (1 - ps)
        weights_att = T * weights_att_treated + (1 - T) * weights_att_control

        if trim_weights:
            for w in [weights_ate, weights_att]:
                threshold = np.quantile(w, trim_quantile)
                w[w > threshold] = threshold

        weights_ate = weights_ate / weights_ate.sum() * len(weights_ate)
        weights_att = weights_att / weights_att.sum() * len(weights_att)

        ate = np.sum(weights_ate * Y * T) / np.sum(weights_ate * T) - \
              np.sum(weights_ate * Y * (1 - T)) / np.sum(weights_ate * (1 - T))

        att = np.mean(Y[T == 1]) - np.sum(weights_att[T == 0] * Y[T == 0]) / np.sum(weights_att[T == 0])

        n_boot = get_bootstrap_iterations(1000)
        boot_ates = []
        boot_atts = []

        for _ in range(n_boot):
            boot_idx = np.random.choice(len(T), size=len(T), replace=True)
            T_b, ps_b, Y_b = T[boot_idx], ps[boot_idx], Y[boot_idx]

            w_ate_b = T_b / ps_b + (1 - T_b) / (1 - ps_b)
            w_att_b = T_b + (1 - T_b) * ps_b / (1 - ps_b)

            if trim_weights:
                w_ate_b = np.clip(w_ate_b, 0, np.quantile(weights_ate, trim_quantile))
                w_att_b = np.clip(w_att_b, 0, np.quantile(weights_att, trim_quantile))

            w_ate_b = w_ate_b / w_ate_b.sum() * len(w_ate_b)
            w_att_b = w_att_b / w_att_b.sum() * len(w_att_b)

            ate_b = np.sum(w_ate_b * Y_b * T_b) / np.sum(w_ate_b * T_b) - \
                    np.sum(w_ate_b * Y_b * (1 - T_b)) / np.sum(w_ate_b * (1 - T_b))
            boot_ates.append(ate_b)

            if np.sum(w_att_b[T_b == 0]) > 0:
                att_b = np.mean(Y_b[T_b == 1]) - np.sum(w_att_b[T_b == 0] * Y_b[T_b == 0]) / np.sum(w_att_b[T_b == 0])
                boot_atts.append(att_b)

        ate_se = np.std(boot_ates)
        att_se = np.std(boot_atts) if boot_atts else ate_se

        ate_ci = (np.percentile(boot_ates, 2.5), np.percentile(boot_ates, 97.5))
        att_ci = (np.percentile(boot_atts, 2.5), np.percentile(boot_atts, 97.5)) if boot_atts else ate_ci

        p_value_ate = 2 * (1 - stats.norm.cdf(abs(ate / ate_se))) if ate_se > 0 else 1.0
        p_value_att = 2 * (1 - stats.norm.cdf(abs(att / att_se))) if att_se > 0 else 1.0

        balance_before = self.compute_balance_statistics()

        full_weights = np.zeros(len(self.agent_df))
        full_weights[valid] = weights_ate
        balance_after = self.compute_balance_statistics(weights=full_weights)

        n_treated = T.sum()
        n_control = len(T) - n_treated

        if p_value_ate < 0.05:
            direction = "increases" if ate > 0 else "decreases"
            interpretation = (
                f"AI adoption {direction} {outcome_col} by {abs(ate):.4f} "
                f"(ATE={ate:.4f}, SE={ate_se:.4f}, p={p_value_ate:.4f}). "
                f"IPW with {n_treated} treated, {n_control} control units."
            )
        else:
            interpretation = (
                f"No significant effect of AI adoption on {outcome_col} "
                f"(ATE={ate:.4f}, p={p_value_ate:.4f})."
            )

        overlap_summary = {
            'ps_mean_treated': ps[T == 1].mean(),
            'ps_mean_control': ps[T == 0].mean(),
            'max_weight': weights_ate.max(),
            'effective_sample_size': len(weights_ate)**2 / np.sum(weights_ate**2)
        }

        result = PropensityScoreResult(
            method='inverse_probability_weighting',
            outcome=outcome_col,
            ate=ate,
            att=att,
            ate_se=ate_se,
            att_se=att_se,
            ate_ci=ate_ci,
            att_ci=att_ci,
            p_value_ate=p_value_ate,
            p_value_att=p_value_att,
            n_treated=int(n_treated),
            n_control=int(n_control),
            n_matched=None,
            balance_before={row['covariate']: row['smd'] for _, row in balance_before.iterrows()},
            balance_after={row['covariate']: row['smd'] for _, row in balance_after.iterrows()},
            overlap_summary=overlap_summary,
            interpretation=interpretation
        )

        self.results.append(result)

        print(f"   ✓ N = {len(T)} (Treated: {n_treated}, Control: {n_control})")
        print(f"   ✓ ATE = {ate:.4f} (SE = {ate_se:.4f}), p = {p_value_ate:.4f}")
        print(f"   ✓ ATT = {att:.4f} (SE = {att_se:.4f}), p = {p_value_att:.4f}")
        print(f"   ✓ Effective sample size: {overlap_summary['effective_sample_size']:.1f}")

        smd_before = balance_before['abs_smd'].mean()
        smd_after = balance_after['abs_smd'].mean()
        print(f"\n   Balance (mean |SMD|): {smd_before:.3f} → {smd_after:.3f}")

        print(f"\n   {interpretation}")

        return result

    def augmented_ipw(
        self,
        outcome_col: str,
        outcome_model: str = 'linear'
    ) -> Optional[PropensityScoreResult]:
        """
        Augmented Inverse Probability Weighting (AIPW) - Doubly Robust Estimator.

        AIPW combines IPW with an outcome regression model. It is consistent
        if EITHER the propensity score model OR the outcome model is correctly
        specified (doubly robust property).
        """
        print(f"\n📊 Augmented IPW (Doubly Robust) for {outcome_col}...")

        if self.propensity_scores is None:
            self.estimate_propensity_scores()

        df = self.agent_df.copy()

        if outcome_col not in df.columns:
            print(f"   ⚠️ Outcome column '{outcome_col}' not found")
            return None

        _, covariates = self._prepare_data()
        df = df.loc[self.agent_df.index]

        T = df['treatment'].values
        ps = df['propensity_score'].values
        Y = df[outcome_col].values

        X_cols = [c for c in covariates if c in df.columns]
        X = df[X_cols].values

        valid = ~(np.isnan(Y) | np.isnan(ps) | np.any(np.isnan(X), axis=1))
        T, ps, Y, X = T[valid], ps[valid], Y[valid], X[valid]

        ps = np.clip(ps, 0.01, 0.99)

        if len(T) < 50:
            print("   ⚠️ Insufficient observations")
            return None

        X_design = np.column_stack([np.ones(len(X)), X])

        treated_mask = T == 1
        if treated_mask.sum() > 10:
            beta_1 = np.linalg.lstsq(X_design[treated_mask], Y[treated_mask], rcond=None)[0]
            mu_1 = X_design @ beta_1
        else:
            mu_1 = np.full(len(Y), Y[treated_mask].mean())

        control_mask = T == 0
        if control_mask.sum() > 10:
            beta_0 = np.linalg.lstsq(X_design[control_mask], Y[control_mask], rcond=None)[0]
            mu_0 = X_design @ beta_0
        else:
            mu_0 = np.full(len(Y), Y[control_mask].mean())

        psi_1 = T * (Y - mu_1) / ps + mu_1
        psi_0 = (1 - T) * (Y - mu_0) / (1 - ps) + mu_0

        ate = np.mean(psi_1) - np.mean(psi_0)

        att_num = np.mean(T * (Y - mu_0) - (1 - T) * ps / (1 - ps) * (Y - mu_0))
        att_denom = np.mean(T)
        att = att_num / att_denom if att_denom > 0 else ate

        n_boot = get_bootstrap_iterations(1000)
        boot_ates = []
        boot_atts = []

        for _ in range(n_boot):
            boot_idx = np.random.choice(len(T), size=len(T), replace=True)
            T_b = T[boot_idx]
            ps_b = ps[boot_idx]
            Y_b = Y[boot_idx]
            mu_1_b = mu_1[boot_idx]
            mu_0_b = mu_0[boot_idx]

            psi_1_b = T_b * (Y_b - mu_1_b) / ps_b + mu_1_b
            psi_0_b = (1 - T_b) * (Y_b - mu_0_b) / (1 - ps_b) + mu_0_b

            ate_b = np.mean(psi_1_b) - np.mean(psi_0_b)
            boot_ates.append(ate_b)

            att_num_b = np.mean(T_b * (Y_b - mu_0_b) - (1 - T_b) * ps_b / (1 - ps_b) * (Y_b - mu_0_b))
            att_denom_b = np.mean(T_b)
            if att_denom_b > 0:
                boot_atts.append(att_num_b / att_denom_b)

        ate_se = np.std(boot_ates)
        att_se = np.std(boot_atts) if boot_atts else ate_se

        ate_ci = (np.percentile(boot_ates, 2.5), np.percentile(boot_ates, 97.5))
        att_ci = (np.percentile(boot_atts, 2.5), np.percentile(boot_atts, 97.5)) if boot_atts else ate_ci

        p_value_ate = 2 * (1 - stats.norm.cdf(abs(ate / ate_se))) if ate_se > 0 else 1.0
        p_value_att = 2 * (1 - stats.norm.cdf(abs(att / att_se))) if att_se > 0 else 1.0

        balance_before = self.compute_balance_statistics()
        balance_after = balance_before.copy()

        n_treated = T.sum()
        n_control = len(T) - n_treated

        if p_value_ate < 0.05:
            direction = "increases" if ate > 0 else "decreases"
            interpretation = (
                f"AI adoption {direction} {outcome_col} by {abs(ate):.4f} "
                f"(ATE={ate:.4f}, SE={ate_se:.4f}, p={p_value_ate:.4f}). "
                f"Doubly-robust AIPW estimator with {n_treated} treated, {n_control} control."
            )
        else:
            interpretation = (
                f"No significant effect of AI adoption on {outcome_col} "
                f"(ATE={ate:.4f}, p={p_value_ate:.4f}). Doubly-robust AIPW estimator."
            )

        overlap_summary = {
            'ps_mean_treated': ps[T == 1].mean(),
            'ps_mean_control': ps[T == 0].mean(),
            'outcome_model_r2_treated': 1 - np.var(Y[T == 1] - mu_1[T == 1]) / np.var(Y[T == 1]) if np.var(Y[T == 1]) > 0 else 0,
            'outcome_model_r2_control': 1 - np.var(Y[T == 0] - mu_0[T == 0]) / np.var(Y[T == 0]) if np.var(Y[T == 0]) > 0 else 0
        }

        result = PropensityScoreResult(
            method='augmented_ipw',
            outcome=outcome_col,
            ate=ate,
            att=att,
            ate_se=ate_se,
            att_se=att_se,
            ate_ci=ate_ci,
            att_ci=att_ci,
            p_value_ate=p_value_ate,
            p_value_att=p_value_att,
            n_treated=int(n_treated),
            n_control=int(n_control),
            n_matched=None,
            balance_before={row['covariate']: row['smd'] for _, row in balance_before.iterrows()},
            balance_after={row['covariate']: row['smd'] for _, row in balance_after.iterrows()},
            overlap_summary=overlap_summary,
            interpretation=interpretation
        )

        self.results.append(result)

        print(f"   ✓ N = {len(T)} (Treated: {n_treated}, Control: {n_control})")
        print(f"   ✓ ATE = {ate:.4f} (SE = {ate_se:.4f}), p = {p_value_ate:.4f}")
        print(f"   ✓ ATT = {att:.4f} (SE = {att_se:.4f}), p = {p_value_att:.4f}")
        print(f"   ✓ Outcome model R² (treated): {overlap_summary['outcome_model_r2_treated']:.3f}")
        print(f"   ✓ Outcome model R² (control): {overlap_summary['outcome_model_r2_control']:.3f}")

        print(f"\n   {interpretation}")

        return result

    def run_all_methods(
        self,
        outcome_col: str
    ) -> List[PropensityScoreResult]:
        """Run all propensity score methods for a given outcome."""
        print("\n" + "=" * 70)
        print(f"PROPENSITY SCORE ANALYSIS FOR: {outcome_col}")
        print("=" * 70)

        self.estimate_propensity_scores()

        results = []

        print("\n" + "-" * 70)
        print("Method 1: Nearest Neighbor Matching")
        print("-" * 70)
        result = self.nearest_neighbor_matching(outcome_col)
        if result:
            results.append(result)

        print("\n" + "-" * 70)
        print("Method 2: Inverse Probability Weighting")
        print("-" * 70)
        result = self.inverse_probability_weighting(outcome_col)
        if result:
            results.append(result)

        print("\n" + "-" * 70)
        print("Method 3: Augmented IPW (Doubly Robust)")
        print("-" * 70)
        result = self.augmented_ipw(outcome_col)
        if result:
            results.append(result)

        return results

    def generate_results_table(self) -> pd.DataFrame:
        """Generate publication-ready propensity score results table."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            rows.append({
                'Method': r.method,
                'Outcome': r.outcome,
                'ATE': f"{r.ate:.4f}",
                'ATE SE': f"{r.ate_se:.4f}",
                'ATE 95% CI': f"[{r.ate_ci[0]:.4f}, {r.ate_ci[1]:.4f}]",
                'ATE p-value': f"{r.p_value_ate:.4f}" if r.p_value_ate >= 0.0001 else "<0.0001",
                'ATT': f"{r.att:.4f}",
                'ATT SE': f"{r.att_se:.4f}",
                'ATT 95% CI': f"[{r.att_ci[0]:.4f}, {r.att_ci[1]:.4f}]",
                'ATT p-value': f"{r.p_value_att:.4f}" if r.p_value_att >= 0.0001 else "<0.0001",
                'N Treated': r.n_treated,
                'N Control': r.n_control,
                'N Matched': r.n_matched if r.n_matched else 'N/A'
            })

        return pd.DataFrame(rows)

    def generate_balance_table(self) -> pd.DataFrame:
        """Generate covariate balance table before/after adjustment."""
        if not self.results:
            return pd.DataFrame()

        result = self.results[-1]

        rows = []
        for cov in result.balance_before.keys():
            rows.append({
                'Covariate': cov,
                'SMD Before': f"{result.balance_before[cov]:.3f}",
                'SMD After': f"{result.balance_after.get(cov, np.nan):.3f}",
                'Balanced Before': '✓' if abs(result.balance_before[cov]) < 0.1 else '✗',
                'Balanced After': '✓' if abs(result.balance_after.get(cov, 1)) < 0.1 else '✗'
            })

        return pd.DataFrame(rows)

    def generate_overlap_diagnostics(self) -> Dict[str, Any]:
        """Generate overlap diagnostic plots data."""
        if self.propensity_scores is None:
            return {}

        T = self.agent_df['treatment'].values
        ps = self.agent_df['propensity_score'].values

        return {
            'ps_treated': ps[T == 1].tolist(),
            'ps_control': ps[T == 0].tolist(),
            'overlap_region': (
                max(ps[T == 1].min(), ps[T == 0].min()),
                min(ps[T == 1].max(), ps[T == 0].max())
            ),
            'pct_in_overlap': ((ps >= max(ps[T == 1].min(), ps[T == 0].min())) &
                              (ps <= min(ps[T == 1].max(), ps[T == 0].max()))).mean()
        }
