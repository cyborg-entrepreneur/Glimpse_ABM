"""
Regression Discontinuity Analysis for Glimpse ABM.

RD exploits discontinuities in treatment assignment based on a continuous
"running variable" crossing a threshold.

References
----------
Imbens, G., & Lemieux, T. (2008). Regression discontinuity designs:
    A guide to practice. Journal of Econometrics, 142(2), 615-635.

Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in
    economics. Journal of Economic Literature, 48(2), 281-355.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RDResult:
    """Results from Regression Discontinuity analysis."""
    running_variable: str
    cutoff: float
    bandwidth: float
    treatment_effect: float
    effect_se: float
    effect_ci_lower: float
    effect_ci_upper: float
    p_value: float
    n_left: int  # Observations left of cutoff
    n_right: int  # Observations right of cutoff
    polynomial_order: int
    kernel: str
    interpretation: str
    mccrary_test: Optional[Dict[str, float]]  # Manipulation test


class RegressionDiscontinuityAnalysis:
    """
    Regression Discontinuity Design for AI Adoption Effects.

    RD exploits discontinuities in treatment assignment based on a continuous
    "running variable" crossing a threshold. In the ABM context, this could be:

    1. Capital threshold: Agents adopt AI when capital exceeds $X
    2. Performance threshold: Agents upgrade AI at certain growth rates
    3. Time threshold: AI becomes available at simulation step T

    Identification Strategy
    -----------------------
    If agents just below the cutoff are comparable to those just above
    (local randomization), the difference in outcomes at the cutoff
    identifies the causal effect of treatment.

    Key Assumptions
    ---------------
    1. No precise manipulation of the running variable around the cutoff
    2. Continuity of potential outcomes at the cutoff
    3. Treatment assignment is deterministic at the cutoff (sharp RD)
       or probabilistic (fuzzy RD)
    """

    def __init__(
        self,
        agent_df: pd.DataFrame,
        decision_df: Optional[pd.DataFrame] = None
    ):
        self.agent_df = agent_df.copy()
        self.decision_df = decision_df.copy() if decision_df is not None else None
        self.results: List[RDResult] = []

    def identify_discontinuity(
        self,
        running_var: str,
        treatment_var: str,
        potential_cutoffs: Optional[List[float]] = None
    ) -> Optional[Tuple[float, str]]:
        """
        Identify a valid discontinuity in treatment assignment.

        Parameters
        ----------
        running_var : str
            Column name for the running variable (e.g., 'capital_at_decision')
        treatment_var : str
            Column name for treatment indicator (e.g., 'adopted_ai')
        potential_cutoffs : list, optional
            Candidate cutoff values to test

        Returns
        -------
        Tuple of (cutoff, type) or None if no valid discontinuity found.
        Type is 'sharp' or 'fuzzy'.
        """
        df = self.agent_df if self.decision_df is None else self.decision_df

        if running_var not in df.columns or treatment_var not in df.columns:
            return None

        data = df[[running_var, treatment_var]].dropna()

        if len(data) < 100:
            return None

        # If no cutoffs specified, try quartiles
        if potential_cutoffs is None:
            potential_cutoffs = data[running_var].quantile([0.25, 0.5, 0.75]).tolist()

        best_cutoff = None
        best_discontinuity = 0

        for cutoff in potential_cutoffs:
            below = data[data[running_var] < cutoff][treatment_var].mean()
            above = data[data[running_var] >= cutoff][treatment_var].mean()
            discontinuity = abs(above - below)

            if discontinuity > best_discontinuity:
                best_discontinuity = discontinuity
                best_cutoff = cutoff

        if best_cutoff is None or best_discontinuity < 0.1:
            return None

        # Determine if sharp or fuzzy
        below = data[data[running_var] < best_cutoff][treatment_var]
        above = data[data[running_var] >= best_cutoff][treatment_var]

        # Sharp RD: treatment jumps from 0 to 1 at cutoff
        if below.mean() < 0.1 and above.mean() > 0.9:
            rd_type = 'sharp'
        else:
            rd_type = 'fuzzy'

        return (best_cutoff, rd_type)

    def estimate_rd_effect(
        self,
        running_var: str,
        outcome_var: str,
        cutoff: float,
        bandwidth: Optional[float] = None,
        polynomial_order: int = 1,
        kernel: str = 'triangular'
    ) -> Optional[RDResult]:
        """
        Estimate the RD treatment effect using local polynomial regression.

        Parameters
        ----------
        running_var : str
            The running variable determining treatment.
        outcome_var : str
            The outcome of interest.
        cutoff : float
            The threshold value.
        bandwidth : float, optional
            Width of window around cutoff. If None, uses IK optimal bandwidth.
        polynomial_order : int
            Degree of polynomial (1 = local linear, 2 = local quadratic).
        kernel : str
            Kernel for weighting ('triangular', 'uniform', 'epanechnikov').

        Returns
        -------
        RDResult or None if estimation fails.
        """
        print(f"\n📊 Estimating RD Effect at cutoff = {cutoff}...")

        df = self.agent_df if self.decision_df is None else self.decision_df

        if running_var not in df.columns or outcome_var not in df.columns:
            print(f"   ⚠️ Missing columns: {running_var} or {outcome_var}")
            return None

        data = df[[running_var, outcome_var]].dropna()

        if len(data) < 50:
            print("   ⚠️ Insufficient observations")
            return None

        # Center running variable at cutoff
        data['X_centered'] = data[running_var] - cutoff
        data['treated'] = (data[running_var] >= cutoff).astype(int)

        # Compute IK-style bandwidth if not specified
        if bandwidth is None:
            # Simple rule-of-thumb bandwidth (Silverman)
            std_x = data['X_centered'].std()
            n = len(data)
            bandwidth = 1.06 * std_x * (n ** (-1/5)) * 2  # Double Silverman

        # Restrict to bandwidth window
        data_bw = data[abs(data['X_centered']) <= bandwidth].copy()

        n_left = (data_bw['X_centered'] < 0).sum()
        n_right = (data_bw['X_centered'] >= 0).sum()

        if n_left < 20 or n_right < 20:
            print(f"   ⚠️ Insufficient observations in bandwidth window (left={n_left}, right={n_right})")
            return None

        # Compute kernel weights
        if kernel == 'triangular':
            data_bw['weight'] = (1 - abs(data_bw['X_centered']) / bandwidth).clip(lower=0)
        elif kernel == 'uniform':
            data_bw['weight'] = 1.0
        elif kernel == 'epanechnikov':
            u = data_bw['X_centered'] / bandwidth
            data_bw['weight'] = (0.75 * (1 - u**2)).clip(lower=0)
        else:
            data_bw['weight'] = 1.0

        # Local polynomial regression
        # Y = α + τ*D + β₁*X + β₂*D*X + ... + ε
        # where D = 1{X >= 0} is treatment indicator

        # Create polynomial terms
        for p in range(1, polynomial_order + 1):
            data_bw[f'X_{p}'] = data_bw['X_centered'] ** p
            data_bw[f'X_{p}_D'] = data_bw[f'X_{p}'] * data_bw['treated']

        # Build design matrix
        X_cols = ['treated']
        for p in range(1, polynomial_order + 1):
            X_cols.extend([f'X_{p}', f'X_{p}_D'])

        X = data_bw[X_cols].values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept
        y = data_bw[outcome_var].values
        w = data_bw['weight'].values

        # Weighted least squares
        try:
            W = np.diag(w)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            beta = np.linalg.solve(XtWX, XtWy)

            # Treatment effect is coefficient on 'treated' (index 1)
            tau = beta[1]

            # Standard error via sandwich estimator
            residuals = y - X @ beta
            sigma2 = np.sum(w * residuals**2) / (np.sum(w) - len(beta))
            var_beta = sigma2 * np.linalg.inv(XtWX)
            se_tau = np.sqrt(var_beta[1, 1])

            # CI and p-value
            ci_lower = tau - 1.96 * se_tau
            ci_upper = tau + 1.96 * se_tau
            z_stat = tau / se_tau
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            # Interpretation
            if p_value < 0.05:
                direction = "increases" if tau > 0 else "decreases"
                interpretation = (
                    f"Crossing the {running_var} threshold of {cutoff:.2f} {direction} "
                    f"{outcome_var} by {abs(tau):.4f} (p={p_value:.4f}). "
                    f"This effect is identified locally at the discontinuity."
                )
            else:
                interpretation = (
                    f"No significant discontinuity in {outcome_var} at {running_var}={cutoff:.2f} "
                    f"(τ={tau:.4f}, p={p_value:.4f})."
                )

            result = RDResult(
                running_variable=running_var,
                cutoff=cutoff,
                bandwidth=bandwidth,
                treatment_effect=tau,
                effect_se=se_tau,
                effect_ci_lower=ci_lower,
                effect_ci_upper=ci_upper,
                p_value=p_value,
                n_left=n_left,
                n_right=n_right,
                polynomial_order=polynomial_order,
                kernel=kernel,
                interpretation=interpretation,
                mccrary_test=None  # Would require density estimation
            )

            self.results.append(result)

            print(f"   ✓ RD estimate: τ = {tau:.4f} (SE = {se_tau:.4f})")
            print(f"   ✓ 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"   ✓ p-value: {p_value:.4f}")
            print(f"   ✓ Bandwidth: {bandwidth:.4f}, n_left={n_left}, n_right={n_right}")
            print(f"\n   {interpretation}")

            return result

        except Exception as e:
            print(f"   ⚠️ RD estimation failed: {e}")
            return None

    def robustness_bandwidths(
        self,
        running_var: str,
        outcome_var: str,
        cutoff: float,
        bandwidths: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Check robustness of RD estimates across different bandwidths.

        Returns DataFrame with estimates for each bandwidth.
        """
        if bandwidths is None:
            # Default: 50%, 75%, 100%, 125%, 150% of optimal
            df = self.agent_df if self.decision_df is None else self.decision_df
            if running_var not in df.columns:
                return pd.DataFrame()
            std_x = df[running_var].std()
            n = len(df)
            h_opt = 1.06 * std_x * (n ** (-1/5)) * 2
            bandwidths = [h_opt * m for m in [0.5, 0.75, 1.0, 1.25, 1.5]]

        results = []
        for bw in bandwidths:
            result = self.estimate_rd_effect(
                running_var, outcome_var, cutoff,
                bandwidth=bw, polynomial_order=1
            )
            if result:
                results.append({
                    'bandwidth': bw,
                    'effect': result.treatment_effect,
                    'se': result.effect_se,
                    'ci_lower': result.effect_ci_lower,
                    'ci_upper': result.effect_ci_upper,
                    'p_value': result.p_value,
                    'n_left': result.n_left,
                    'n_right': result.n_right
                })

        return pd.DataFrame(results)

    def generate_results_table(self) -> pd.DataFrame:
        """Generate publication-ready RD results table."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            rows.append({
                'Running Variable': r.running_variable,
                'Cutoff': f"{r.cutoff:.2f}",
                'Treatment Effect': f"{r.treatment_effect:.4f}",
                'Std. Error': f"{r.effect_se:.4f}",
                '95% CI': f"[{r.effect_ci_lower:.4f}, {r.effect_ci_upper:.4f}]",
                'p-value': f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<0.0001",
                'Bandwidth': f"{r.bandwidth:.4f}",
                'N (left)': r.n_left,
                'N (right)': r.n_right,
                'Polynomial': r.polynomial_order,
                'Kernel': r.kernel
            })

        return pd.DataFrame(rows)
