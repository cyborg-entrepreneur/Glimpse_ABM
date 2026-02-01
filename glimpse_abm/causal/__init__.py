"""
Causal Inference Methods for Glimpse ABM.

This subpackage provides advanced causal inference methods for analyzing
the effects of AI adoption on entrepreneurial outcomes.

Modules
-------
survival
    Cox Proportional Hazards survival analysis
rd_analysis
    Regression Discontinuity Design
did_analysis
    Difference-in-Differences analysis
propensity
    Propensity Score Methods (PSM, IPW, AIPW)
"""

from .survival import CoxRegressionResult, CoxSurvivalAnalysis
from .rd_analysis import RDResult, RegressionDiscontinuityAnalysis
from .did_analysis import DiDResult, DifferenceInDifferencesAnalysis
from .propensity import PropensityScoreResult, PropensityScoreAnalysis

__all__ = [
    # Survival Analysis
    'CoxRegressionResult',
    'CoxSurvivalAnalysis',
    # Regression Discontinuity
    'RDResult',
    'RegressionDiscontinuityAnalysis',
    # Difference-in-Differences
    'DiDResult',
    'DifferenceInDifferencesAnalysis',
    # Propensity Score Methods
    'PropensityScoreResult',
    'PropensityScoreAnalysis',
]
