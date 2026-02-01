"""Public API for the Glimpse ABM package.

Agent-based simulation for investigating Knightian uncertainty under AI augmentation.
Based on Townsend et al. (2025, The Academy of Management Review).
"""

__version__ = "1.0.0"
__author__ = "Townsend, Hunt, Rady, Manocha, & Jin"

from .analysis import (
    ComprehensiveAnalysisFramework,
    ComprehensiveVisualizationSuite,
    InformationParadoxAnalyzer,
    StatisticalAnalysisSuite,
)
from .statistical_tests import (
    RigorousStatisticalAnalysis,
    StatisticalTestResult,
    EffectSizeCalculator,
    AssumptionTester,
    MixedEffectsAnalysis,
    MixedEffectsResult,
    CausalIdentificationAnalysis,
    CausalEffectEstimate,
    run_causal_identification_analysis,
)
from .cli import (
    run_agent_scaling_analysis,
    run_cli,
    run_fixed_level_uncertainty_batch,
    run_master_launcher,
    run_uncertainty_scenario_sweep,
)
from .config import (
    EmergentConfig,
    CalibrationProfile,
    apply_calibration_profile,
    get_calibration_profile,
    list_calibration_profiles,
    load_calibration_profile,
)
from .innovation import CombinationTracker, Innovation, InnovationEngine
from .knowledge import KnowledgeBase
from .market import MarketEnvironment
from .uncertainty import KnightianUncertaintyEnvironment
from .simulation import EmergentSimulation
from .utils import (
    safe_mean,
    safe_exp,
    stable_sigmoid,
    normalize_ai_label,
    canonical_to_display,
)

__all__ = [
    "__version__",
    "__author__",
    "ComprehensiveAnalysisFramework",
    "ComprehensiveVisualizationSuite",
    "InformationParadoxAnalyzer",
    "StatisticalAnalysisSuite",
    "RigorousStatisticalAnalysis",
    "StatisticalTestResult",
    "EffectSizeCalculator",
    "AssumptionTester",
    "MixedEffectsAnalysis",
    "MixedEffectsResult",
    "CausalIdentificationAnalysis",
    "CausalEffectEstimate",
    "run_causal_identification_analysis",
    "run_cli",
    "run_master_launcher",
    "run_fixed_level_uncertainty_batch",
    "run_uncertainty_scenario_sweep",
    "run_agent_scaling_analysis",
    "EmergentConfig",
    "CalibrationProfile",
    "apply_calibration_profile",
    "get_calibration_profile",
    "list_calibration_profiles",
    "load_calibration_profile",
    "CombinationTracker",
    "Innovation",
    "InnovationEngine",
    "KnowledgeBase",
    "MarketEnvironment",
    "KnightianUncertaintyEnvironment",
    "EmergentSimulation",
    "safe_mean",
    "safe_exp",
    "stable_sigmoid",
    "normalize_ai_label",
    "canonical_to_display",
]
