"""Analysis and visualization utilities for Glimpse ABM."""

from __future__ import annotations

ANALYSIS_VERSION = "2025.11.14"

import glob
import os
import warnings
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional plotting dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:  # pragma: no cover - optional plotting dependency
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover
    sns = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy, mannwhitneyu, kruskal
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import EmergentConfig
from .utils import canonical_to_display, normalize_ai_label, safe_mean

try:
    from lifelines import CoxPHFitter  # type: ignore
    HAS_LIFELINES = True
except ImportError:  # pragma: no cover - optional dependency
    CoxPHFitter = None  # type: ignore
    HAS_LIFELINES = False

warnings.filterwarnings('ignore')

HAS_PLOTTING_LIBRARIES = plt is not None and sns is not None

if HAS_PLOTTING_LIBRARIES:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def _apply_standard_transforms(
    df: pd.DataFrame,
    rename_map: Optional[Dict[str, str]] = None,
    numeric_cols: Optional[List[str]] = None,
    ensure_round: bool = True,
    ensure_run_id: bool = True,
) -> pd.DataFrame:
    """
    Apply common DataFrame standardization transforms.

    Args:
        df: DataFrame to transform
        rename_map: Column rename mapping {old_name: new_name}
        numeric_cols: Columns to convert to numeric
        ensure_round: If True, ensure 'round' column exists as Int64
        ensure_run_id: If True, ensure 'run_id' column exists as str

    Returns:
        Transformed DataFrame copy
    """
    if df.empty:
        return df
    df = df.copy()

    # Apply column renames
    if rename_map:
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

    # Ensure round column
    if ensure_round:
        if 'round' in df.columns:
            df['round'] = pd.to_numeric(df['round'], errors='coerce').astype('Int64')
        else:
            df['round'] = np.arange(len(df))

    # Ensure run_id column
    if ensure_run_id:
        if 'run_id' in df.columns:
            df['run_id'] = df['run_id'].astype(str)
        else:
            df['run_id'] = 'default_run'

    # Convert numeric columns
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


class ComprehensiveAnalysisFramework:
    """
    Analyzes ABM results by reading data directly from disk to conserve memory.
    Refactored to read .pkl files instead of .parquet.
    """
    def __init__(self, results_directory: str, config: Optional[EmergentConfig] = None):
        self.results_dir = results_directory
        self.config = config or EmergentConfig()
        (self.agent_df, self.decision_df, self.market_df,
        self.uncertainty_df, self.innovation_df, self.knowledge_df,
        self.summary_df, self.matured_df, self.uncertainty_detail_df) = self._compile_dataframes()
        self._standardize_all_dataframes()
        self._create_emergent_behavioral_groups()

        self.figure_output_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(self.figure_output_dir, exist_ok=True)

        self.analyses = {}

    def load_data(self):
        """Load all necessary data from the results directory."""
        # 1. Look for the .pkl file instead of the .csv file
        final_agents_path = os.path.join(self.results_dir, 'final_agents.pkl')

        if os.path.exists(final_agents_path):
            # 2. Use pd.read_pickle to load the data
            self.agent_df = pd.read_pickle(final_agents_path)
            print(f"✅ Successfully loaded final agent data from {final_agents_path}")
        else:
            print(f"❌ Error: Final agent data file not found at {final_agents_path}")
            # Initialize an empty DataFrame with expected columns if the file is missing
            # This prevents other parts of the code from failing.
            self.agent_df = pd.DataFrame()

    def _standardize_all_dataframes(self):
        """Renaming and harmonizing columns to ensure downstream visualizations have data."""
        self.agent_df = self._standardize_agent_df(self.agent_df)
        self.decision_df = self._standardize_decision_df(self.decision_df)
        self.market_df = self._standardize_market_df(self.market_df)
        self.uncertainty_df = self._standardize_uncertainty_df(self.uncertainty_df)
        self.innovation_df = self._standardize_innovation_df(self.innovation_df)
        self.knowledge_df = self._standardize_knowledge_df(self.knowledge_df)
        self.summary_df = self._standardize_summary_df(self.summary_df)
        self.matured_df = self._standardize_matured_df(self.matured_df)
        self.uncertainty_detail_df = self._standardize_uncertainty_detail_df(self.uncertainty_detail_df)

    def _standardize_agent_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()

        rename_map = {
            'primary_ai': 'primary_ai_level',
            'primary_ai_category': 'primary_ai_level',
            'primary_ai_label': 'primary_ai_level',
            'primary_ai_tier': 'primary_ai_level',
            'agent': 'agent_id',
            'id': 'agent_id'
        }
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        if 'survived' not in df.columns and 'survival' in df.columns:
            df.rename(columns={'survival': 'survived'}, inplace=True)

        numeric_cols = ['final_capital', 'capital_growth', 'survived', 'innovations', 'portfolio_diversity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'survived' in df.columns:
            df['survived'] = df['survived'].fillna(0)

        if 'behavioral_group' not in df.columns:
            df['behavioral_group'] = 'Unknown'
        else:
            df['behavioral_group'] = df['behavioral_group'].fillna('Unknown')

        if 'agent_id' not in df.columns:
            df['agent_id'] = np.arange(len(df))

        if 'run_id' in df.columns:
            df['run_id'] = df['run_id'].astype(str)
        else:
            df['run_id'] = 'default_run'

        if 'primary_ai_level' not in df.columns:
            df['primary_ai_level'] = 'human'

        df['primary_ai_canonical'] = df['primary_ai_level'].apply(normalize_ai_label)
        df['primary_ai_display'] = df['primary_ai_canonical'].apply(canonical_to_display)
        # Maintain backward compatibility with earlier plots
        df['primary_ai_level'] = df['primary_ai_display']

        return df

    def _standardize_decision_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()

        rename_map = {
            'ai_level': 'ai_level_used',
            'chosen_ai_level': 'ai_level_used',
            'decision_round': 'round',
            'round_idx': 'round',
            'step': 'round',
            'time': 'round',
            'action_type': 'action',
            'opportunity': 'opportunity_id',
            'agent': 'agent_id',
            'actor_id': 'agent_id'
        }
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        if 'round' in df.columns:
            df['round'] = pd.to_numeric(df['round'], errors='coerce').astype('Int64')
        else:
            df['round'] = np.arange(len(df))

        if 'run_id' in df.columns:
            df['run_id'] = df['run_id'].astype(str)
        else:
            df['run_id'] = 'default_run'

        if 'agent_id' not in df.columns:
            df['agent_id'] = np.arange(len(df))

        if 'ai_level_used' not in df.columns:
            df['ai_level_used'] = 'none'
        df['ai_level_used'] = df['ai_level_used'].apply(normalize_ai_label)

        df['ai_used'] = df['ai_level_used'].ne('none')

        if 'success' in df.columns:
            df['success'] = pd.to_numeric(df['success'], errors='coerce')
        else:
            df['success'] = np.nan

        return df

    def _standardize_market_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return _apply_standard_transforms(
            df,
            rename_map={
                'round_idx': 'round',
                'step': 'round',
                'time': 'round',
                'market_regime': 'regime'
            }
        )

    def _standardize_uncertainty_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return _apply_standard_transforms(
            df,
            rename_map={
                'round_idx': 'round',
                'time': 'round',
                'actor_ignorance': 'actor_ignorance_level',
                'practical_indeterminism': 'practical_indeterminism_level',
                'agentic_novelty': 'agentic_novelty_level',
                'competitive_recursion': 'competitive_recursion_level'
            }
        )

    def _standardize_innovation_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return _apply_standard_transforms(
            df,
            numeric_cols=['quality', 'novelty', 'market_impact']
        )

    def _standardize_knowledge_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return _apply_standard_transforms(df)

    def _standardize_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()

        rename_map = {
            'round_idx': 'round',
            'step': 'round',
            'time': 'round'
        }
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        if 'round' in df.columns:
            df['round'] = pd.to_numeric(df['round'], errors='coerce').astype('Int64')
        if 'run_id' in df.columns:
            df['run_id'] = df['run_id'].astype(str)

        adoption_map = {
            'ai_share_human': 'ai_share_none',
            'ai_share_basic_ai': 'ai_share_basic',
            'ai_share_advanced_ai': 'ai_share_advanced',
            'ai_share_premium_ai': 'ai_share_premium'
        }
        for src, dst in adoption_map.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        required_shares = ['ai_share_none', 'ai_share_basic', 'ai_share_advanced', 'ai_share_premium']
        for col in required_shares:
            if col not in df.columns:
                df[col] = 0.0

        numeric_cols = [
            'mean_capital', 'median_capital', 'capital_std',
            'mean_ai_trust', 'ai_trust_std',
            'mean_portfolio_diversity', 'portfolio_diversity_std',
            'mean_roe', 'mean_roic_invest', 'mean_roic_innovate', 'mean_roic_explore',
            'innovation_success_rate', 'top_sector_share', 'mean_confidence_invest'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fallback_numeric = [
            'mean_portfolio_diversity', 'portfolio_diversity_std',
            'mean_ai_trust', 'ai_trust_std',
            'innovation_success_rate', 'top_sector_share',
            'mean_confidence_invest', 'mean_capital', 'capital_std'
        ]
        for col in fallback_numeric:
            if col not in df.columns:
                df[col] = 0.0

        return df
    
    def _standardize_matured_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        numeric_cols = [
            'round',
            'agent_id',
            'entry_round',
            'maturation_round',
            'time_to_maturity',
            'investment_amount',
            'capital_returned',
            'net_return',
            'ai_estimated_return',
            'ai_estimated_uncertainty',
            'ai_confidence',
            'ai_actual_accuracy',
            'ai_overconfidence_factor',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'success' in df.columns:
            df['success'] = df['success'].astype(float)
        if 'run_id' in df.columns:
            df['run_id'] = df['run_id'].astype(str)
        if 'ai_level_used' in df.columns:
            df['ai_level_used'] = df['ai_level_used'].apply(normalize_ai_label)
        return df

    def _standardize_uncertainty_detail_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        numeric_cols = [
            'round',
            'agent_id',
            'ai_switch_count',
            'ai_trust',
            'decision_confidence',
            'actor_ignorance_level',
            'actor_ignorance_info_sufficiency',
            'practical_indeterminism_level',
            'practical_indeterminism_regime_stability',
            'agentic_novelty_potential',
            'agentic_novelty_creative_confidence',
            'competitive_recursion_level',
            'competitive_recursion_herding_awareness',
            'ai_estimated_return',
            'ai_estimated_uncertainty',
            'ai_confidence',
            'ai_actual_accuracy',
            'ai_overconfidence_factor',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'run_id' in df.columns:
            df['run_id'] = df['run_id'].astype(str)
        if 'ai_level_used' in df.columns:
            df['ai_level_used'] = df['ai_level_used'].apply(normalize_ai_label)
        if 'strategy_mode' in df.columns:
            df['strategy_mode'] = df['strategy_mode'].fillna('unknown')
        return df
    

    def _create_emergent_behavioral_groups(self):
        """
        Analyzes decision data to categorize agents by their emergent AI usage patterns.
        This should only be run on data from 'emergent' simulations.
        """
        # Use the correct attribute name: self.decision_df
        if self.decision_df.empty or self.agent_df.empty:
            self.agent_df['behavioral_group'] = 'Unknown'
            return

        # Use the correct attribute name: self.decision_df
        emergent_decisions = self.decision_df[self.decision_df['run_id'].str.contains('emergent')]
        if emergent_decisions.empty:
            self.agent_df['behavioral_group'] = self.agent_df['primary_ai_level']
            return

        # Calculate AI usage proportions for each agent
        usage_counts = emergent_decisions.groupby(['run_id', 'agent_id'])['ai_level_used'].value_counts(normalize=True)
        usage_df = usage_counts.unstack(level='ai_level_used', fill_value=0).reset_index()

        # Define classification rules
        def classify_agent(row):
            if row.get('none', 0) > 0.8: return 'AI Skeptic'
            if row.get('premium', 0) > 0.6: return 'AI Devotee'
            if row.get('advanced', 0) > 0.6: return 'AI Devotee'
            if row.get('basic', 0) > 0.7: return 'Cautious Adopter'
            levels_used = sum(1 for level in ['none', 'basic', 'advanced', 'premium'] if row.get(level, 0) > 0.1)
            if levels_used >= 3: return 'Adaptive User'
            return 'Standard User'

        # Apply classification
        usage_df['behavioral_group'] = usage_df.apply(classify_agent, axis=1)

        # Merge the new behavioral group back into the main agent dataframe
        merged_base = self.agent_df.drop(columns=['behavioral_group'], errors='ignore')
        self.agent_df = pd.merge(
            merged_base,
            usage_df[['run_id', 'agent_id', 'behavioral_group']],
            on=['run_id', 'agent_id'],
            how='left'
        )
        if 'behavioral_group_y' in self.agent_df.columns:
            # pandas added suffixes; consolidate them
            self.agent_df['behavioral_group'] = self.agent_df['behavioral_group_y']
            self.agent_df.drop(columns=[col for col in ['behavioral_group_x', 'behavioral_group_y'] if col in self.agent_df.columns], inplace=True)
        elif 'behavioral_group' not in self.agent_df.columns and 'behavioral_group_x' in self.agent_df.columns:
            self.agent_df.rename(columns={'behavioral_group_x': 'behavioral_group'}, inplace=True)
        if 'behavioral_group' not in self.agent_df.columns:
            self.agent_df['behavioral_group'] = np.nan

        # Fill any agents who had no decisions with their primary type as a fallback
        self.agent_df['behavioral_group'].fillna(self.agent_df['primary_ai_level'], inplace=True)
    
    def run_full_analysis(self):
        """
        Run all analysis helpers, populate the self.analyses dictionary,
        and return the results.
        """
        print("\n🔬 EXECUTING COMPREHENSIVE ANALYSIS FRAMEWORK")
        print(f"   Reading data from: {self.results_dir}")
        print("=" * 70)

        # Check if data is available before proceeding
        if self.agent_df.empty and self.decision_df.empty:
            print("⚠️ Agent and decision data not found. Skipping analysis.")
            return {} 

        try:
            # Run all the private analysis methods and store results
            self.analyses['performance_outcomes'] = self._analyze_performance_outcomes()
            self.analyses['ai_augmentation_effects'] = self._analyze_ai_augmentation_effects()
            self.analyses['ai_vs_uncertainty'] = self._analyze_ai_vs_uncertainty()
            self.analyses['knightian_dynamics'] = self._analyze_knightian_uncertainty_dynamics()
            self.analyses['emergent_behaviors'] = self._analyze_emergent_behaviors()
            self.analyses['uncertainty_detail'] = self._analyze_uncertainty_detail()
            self.analyses['innovation_equilibrium'] = self._analyze_innovation_equilibrium_trap()

        except Exception as e:
            print(f"Analysis failed with error: {e}")
            # Return any partial results that were successful
            return self.analyses

        return self.analyses


    def _analyze_performance_outcomes(self):
        results = {}
        if not self.agent_df.empty:
            level_col = 'primary_ai_canonical' if 'primary_ai_canonical' in self.agent_df.columns else 'primary_ai_level'
            survival_series = self.agent_df.groupby(level_col)['survived'].mean().dropna()
            results['survival_rate_by_ai'] = {normalize_ai_label(level): float(value) for level, value in survival_series.items()}
            if 'behavioral_group' in self.agent_df.columns:
                results['survival_rate_by_behavioral_group'] = self.agent_df.groupby('behavioral_group')['survived'].mean().to_dict()
            wealth_df = self.agent_df.groupby(level_col)['final_capital'].agg(['mean', 'std', 'median']).fillna(0)
            results['wealth_distribution_by_ai'] = {normalize_ai_label(idx): wealth_df.loc[idx].to_dict() for idx in wealth_df.index}
            if 'behavioral_group' in self.agent_df.columns:
                results['wealth_distribution_by_behavioral_group'] = self.agent_df.groupby('behavioral_group')['final_capital'].agg(['mean', 'std', 'median']).fillna(0).to_dict('index')
            corr_cols = [col for col in ['capital_growth', 'uncertainty_tolerance', 'innovativeness', 'exploration_tendency', 'ai_trust', 'innovations', 'portfolio_diversity'] if col in self.agent_df.columns]
            if 'capital_growth' in corr_cols:
                corr_cols.remove('capital_growth')
            if corr_cols:
                corr_df = self.agent_df[['capital_growth'] + corr_cols].copy()
                corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
                results['performance_drivers_correlation'] = (corr_df.corr()['capital_growth']
                                                               .drop('capital_growth', errors='ignore')
                                                               .dropna()
                                                               .to_dict())
        return results
    
    def _analyze_ai_augmentation_effects(self):
        results = {}
        if not self.agent_df.empty:
            ai_levels = ['none', 'basic', 'advanced', 'premium']
            metrics = ['capital_growth', 'final_capital', 'innovations', 'portfolio_diversity']
            level_col = 'primary_ai_canonical' if 'primary_ai_canonical' in self.agent_df.columns else 'primary_ai_level'
            for metric in metrics:
                if metric in self.agent_df.columns:
                    values = {}
                    for level in ai_levels:
                        if level_col == 'primary_ai_canonical':
                            mask = self.agent_df[level_col] == level
                        else:
                            display_label = canonical_to_display(level, level)
                            mask = self.agent_df[level_col].astype(str).str.contains(display_label, case=False, na=False)
                        if mask.any():
                            values[level] = float(self.agent_df.loc[mask, metric].mean())
                    if values:
                        results[metric] = values
        if not self.decision_df.empty:
            for metric in ['perc_actor_ignorance_level', 'perc_practical_indeterminism_level', 'perc_agentic_novelty_potential', 'perc_competitive_recursion_level']:
                if metric in self.decision_df.columns:
                    results[metric] = self.decision_df.groupby('ai_level_used')[metric].mean().dropna().to_dict()
        if not self.matured_df.empty:
            matured = self.matured_df[self.matured_df['investment_amount'] > 0].copy()
            if not matured.empty:
                matured['return_multiple'] = matured['capital_returned'] / matured['investment_amount']
                results['investment_success_rate_by_ai'] = (matured.groupby('ai_level_used')['success']
                                                            .mean()
                                                            .dropna()
                                                            .to_dict())
                results['investment_return_multiple_by_ai'] = (matured.groupby('ai_level_used')['return_multiple']
                                                               .mean()
                                                               .dropna()
                                                               .to_dict())
        if not self.uncertainty_detail_df.empty and 'decision_confidence' in self.uncertainty_detail_df.columns:
            results['decision_confidence_by_ai'] = (self.uncertainty_detail_df.groupby('ai_level_used')['decision_confidence']
                                                    .mean()
                                                    .dropna()
                                                    .to_dict())
        return results
    
    def _analyze_ai_vs_uncertainty(self):
        results = {}
        if self.decision_df.empty:
            return results
        uncertainty_columns = [col for col in self.decision_df.columns if col.startswith('perc_')]
        for level in ['none', 'basic', 'advanced', 'premium']:
            level_df = self.decision_df[self.decision_df['ai_level_used'] == level]
            if level_df.empty:
                continue
            level_results = {}
            for col in uncertainty_columns:
                if col.endswith('_level') or col.endswith('_confidence') or col.endswith('_awareness'):
                    cleaned = col[len('perc_'):]
                    parts = cleaned.split('_')
                    if len(parts) >= 2:
                        main_key = '_'.join(parts[:2])
                        sub_key = '_'.join(parts[2:]) if len(parts) > 2 else 'value'
                    else:
                        main_key = parts[0]
                        sub_key = 'value'
                    level_results.setdefault(main_key, {})[sub_key] = float(level_df[col].mean())
            if level_results:
                results[level] = level_results
        return results
    
    def _analyze_knightian_uncertainty_dynamics(self):
        results = {}
        if not self.uncertainty_df.empty:
            evolution_cols = [col for col in self.uncertainty_df.columns if col.endswith('_level')]
            if evolution_cols:
                tmp_unc = self.uncertainty_df[['round'] + evolution_cols].copy()
                tmp_unc[evolution_cols] = tmp_unc[evolution_cols].apply(pd.to_numeric, errors='coerce')
                results['uncertainty_evolution'] = (self.uncertainty_df.groupby('round')[evolution_cols]
                                                    .mean()
                                                    .fillna(0)
                                                    .to_dict('index'))
        if not self.market_df.empty and 'regime' in self.market_df.columns:
            results['market_regime_frequency'] = self.market_df['regime'].value_counts(normalize=True).to_dict()
        return results
    
    def _analyze_uncertainty_detail(self):
        results = {}
        df = self.uncertainty_detail_df
        if df.empty:
            return results

        if 'ai_level_used' in df.columns and 'actor_ignorance_level' in df.columns:
            results['ignorance_by_ai'] = (df.groupby('ai_level_used')['actor_ignorance_level']
                                          .mean()
                                          .dropna()
                                          .to_dict())

        if 'strategy_mode' in df.columns and 'actor_ignorance_level' in df.columns:
            results['ignorance_by_strategy'] = (df.groupby('strategy_mode')['actor_ignorance_level']
                                                .mean()
                                                .dropna()
                                                .to_dict())

        if {'ai_level_used', 'ai_contains_hallucination'}.issubset(df.columns):
            results['hallucination_rate_by_ai'] = (df.groupby('ai_level_used')['ai_contains_hallucination']
                                                   .mean()
                                                   .dropna()
                                                   .to_dict())

        if {'ai_switch_count', 'decision_confidence'}.issubset(df.columns):
            corr = df[['ai_switch_count', 'decision_confidence']].apply(pd.to_numeric, errors='coerce').corr()
            if not corr.empty:
                results['switch_confidence_correlation'] = float(corr.loc['ai_switch_count', 'decision_confidence'])

        if {'ai_estimated_return', 'actor_ignorance_level'}.issubset(df.columns):
            subset = df[['ai_level_used', 'ai_estimated_return', 'actor_ignorance_level']].copy()
            subset[['ai_estimated_return', 'actor_ignorance_level']] = subset[['ai_estimated_return', 'actor_ignorance_level']].apply(pd.to_numeric, errors='coerce')
            pivot = (subset.groupby('ai_level_used')[['ai_estimated_return', 'actor_ignorance_level']]
                     .mean()
                     .dropna())
            if not pivot.empty:
                results['ai_return_vs_ignorance'] = pivot.to_dict('index')

        return results
    
    def _analyze_innovation_equilibrium_trap(self):
        results = {}
        matured = self.matured_df
        if not matured.empty:
            matured = matured.copy()
            matured = matured[matured['investment_amount'] > 0]
            if not matured.empty:
                matured['return_multiple'] = matured['capital_returned'] / matured['investment_amount']
                matured['success'] = pd.to_numeric(matured['success'], errors='coerce')
                results['matured_success_rate_by_ai'] = (matured.groupby('ai_level_used')['success']
                                                         .mean()
                                                         .dropna()
                                                         .to_dict())
                results['matured_return_multiple_by_ai'] = (matured.groupby('ai_level_used')['return_multiple']
                                                            .mean()
                                                            .dropna()
                                                            .to_dict())
                sector_success = (matured.groupby(['sector', 'ai_level_used'])['success']
                                  .mean()
                                  .unstack(fill_value=0))
                results['sector_success_matrix'] = {sector: sector_success.loc[sector].to_dict()
                                                    for sector in sector_success.index}

        if not self.summary_df.empty and 'innovation_success_rate' in self.summary_df.columns:
            tmp = self.summary_df[['round', 'innovation_success_rate']].copy()
            tmp['innovation_success_rate'] = pd.to_numeric(tmp['innovation_success_rate'], errors='coerce')
            results['innovation_success_trend'] = (tmp.groupby('round')['innovation_success_rate']
                                                   .mean()
                                                   .dropna()
                                                   .to_dict())

        if not self.summary_df.empty and 'ai_share_premium' in self.summary_df.columns:
            tmp = self.summary_df[['round', 'ai_share_premium']].copy()
            tmp['ai_share_premium'] = pd.to_numeric(tmp['ai_share_premium'], errors='coerce')
            results['premium_ai_share_trend'] = (tmp.groupby('round')['ai_share_premium']
                                                 .mean()
                                                 .dropna()
                                                 .to_dict())

        if not self.market_df.empty and 'market_saturation' in self.market_df.columns:
            tmp = self.market_df[['round', 'market_saturation']].copy()
            tmp['market_saturation'] = pd.to_numeric(tmp['market_saturation'], errors='coerce')
            results['market_saturation_trend'] = (tmp.groupby('round')['market_saturation']
                                                  .mean()
                                                  .dropna()
                                                  .to_dict())

        return results
    
    def _analyze_emergent_behaviors(self):
        results = {}
        if not self.summary_df.empty:
            summary_df = self.summary_df.copy()
            adoption_cols = ['round', 'ai_share_none', 'ai_share_basic', 'ai_share_advanced', 'ai_share_premium']
            action_cols = ['round', 'action_share_invest', 'action_share_innovate', 'action_share_explore', 'action_share_maintain']

            summary_df[adoption_cols[1:]] = summary_df[adoption_cols[1:]].apply(pd.to_numeric, errors='coerce')
            adoption_df = (summary_df[adoption_cols]
                           .groupby('round', as_index=False)
                           .mean())
            results['ai_adoption_trends'] = adoption_df.set_index('round').to_dict('index')

            summary_df[action_cols[1:]] = summary_df[action_cols[1:]].apply(pd.to_numeric, errors='coerce')
            action_df = (summary_df[action_cols]
                         .groupby('round', as_index=False)
                         .mean())
            results['action_share_trends'] = action_df.set_index('round').to_dict('index')
        if not self.decision_df.empty:
            results['action_counts'] = self.decision_df.groupby('action')['agent_id'].count().to_dict()
        return results

    def plot_survival_rate(self):
        """Plots the survival rate."""
        if 'survived' in self.agent_df.columns:
            survival_rate = self.agent_df['survived'].mean() * 100
            print(f"Overall Survival Rate: {survival_rate:.2f}%")
            # Visualization code would go here...
        else:
            print("'survived' column not found. Cannot plot survival rate.")
    
    def plot_final_capital_distribution(self):
        """Plots the distribution of final capital for surviving agents."""
        if not HAS_PLOTTING_LIBRARIES:
            print("Matplotlib/seaborn unavailable; skipping capital distribution plot.")
            return
        if 'final_capital' in self.agent_df.columns and 'survived' in self.agent_df.columns:
            survivors = self.agent_df[self.agent_df['survived'] == 1]
            if not survivors.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(survivors['final_capital'], bins=20, kde=True)
                plt.title('Final Capital Distribution of Survivors')
                plt.xlabel('Final Capital')
                plt.ylabel('Number of Agents')
                plt.show()
            else:
                print("No survivors to plot.")
        else:
            print("'final_capital' or 'survived' column not found. Cannot plot capital distribution.")
    
    def _compile_dataframes(self):
        """
        Reads all data from the simulation output directories using pickle.
        (Handles both single and multi-run directories)
        """
        chunk_row_target = int(getattr(self.config, 'ANALYSIS_CHUNK_ROW_TARGET', 500_000) or 500_000)
        chunk_row_target = max(50_000, chunk_row_target)

        def _concat_pickles(file_list: List[str]) -> pd.DataFrame:
            if not file_list:
                return pd.DataFrame()
            combined: Optional[pd.DataFrame] = None
            buffer: List[pd.DataFrame] = []
            buffer_rows = 0

            for file_path in file_list:
                try:
                    df = pd.read_pickle(file_path)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"Warning: failed to load {file_path}: {exc}")
                    continue
                if df is None or df.empty:
                    continue
                buffer.append(df)
                buffer_rows += len(df)
                if buffer_rows >= chunk_row_target:
                    chunk = pd.concat(buffer, ignore_index=True)
                    buffer.clear()
                    buffer_rows = 0
                    if combined is None:
                        combined = chunk
                    else:
                        combined = pd.concat([combined, chunk], ignore_index=True)
                    gc.collect()
            if buffer:
                chunk = pd.concat(buffer, ignore_index=True)
                if combined is None:
                    combined = chunk
                else:
                    combined = pd.concat([combined, chunk], ignore_index=True)
            if combined is None:
                return pd.DataFrame()
            combined.reset_index(drop=True, inplace=True)
            return combined

        def read_partitioned_data(data_type: str):
            single_run_pattern = os.path.join(self.results_dir, data_type, '*.pkl')
            multi_run_pattern = os.path.join(self.results_dir, '*', data_type, '*.pkl')
            files = sorted(glob.glob(single_run_pattern))
            if not files:
                files = sorted(glob.glob(multi_run_pattern))
            if not files:
                if data_type not in ['innovations', 'knowledge', 'summary', 'matured', 'uncertainty_details']:
                    print(f"Warning: No .pkl files found for data type '{data_type}'")
                return pd.DataFrame()
            return _concat_pickles(files)

        single_file_path = os.path.join(self.results_dir, 'final_agents.pkl')
        multi_run_pattern = os.path.join(self.results_dir, '*', 'final_agents.pkl')

        agent_df = pd.DataFrame()
        if os.path.exists(single_file_path):
            agent_df = pd.read_pickle(single_file_path)
        else:
            agent_files = sorted(glob.glob(multi_run_pattern))
            if agent_files:
                agent_df = _concat_pickles(agent_files)

        decision_df = read_partitioned_data('decisions')
        market_df = read_partitioned_data('market')
        uncertainty_df = read_partitioned_data('uncertainty')
        innovation_df = read_partitioned_data('innovations')
        knowledge_df = read_partitioned_data('knowledge')
        summary_df = read_partitioned_data('summary')
        matured_df = read_partitioned_data('matured')
        uncertainty_detail_df = read_partitioned_data('uncertainty_details')

        return agent_df, decision_df, market_df, uncertainty_df, innovation_df, knowledge_df, summary_df, matured_df, uncertainty_detail_df

    def export_research_tables(
        self,
        output_dir: str,
        stage_bounds: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Dict[str, str]:
        """
        Export tables supporting the Knightian uncertainty research questions.
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, Path] = {}

        def _write_table(df: Optional[pd.DataFrame], filename: str) -> Optional[Path]:
            if df is None or df.empty:
                return None
            path = output_path / filename
            df.to_csv(path, index=False)
            return path

        decisions = self.decision_df.dropna(subset=['round']).copy()
        if not decisions.empty:
            decisions['round'] = pd.to_numeric(decisions['round'], errors='coerce')
            decisions = decisions.dropna(subset=['round'])
            decisions['round'] = decisions['round'].astype(int)

        base_uncertainty_cols = [
            'actor_ignorance_level',
            'practical_indeterminism_level',
            'agentic_novelty_level',
            'competitive_recursion_level',
            'agentic_component_scarcity',
            'agentic_disruption_avg',
            'agentic_combo_rate',
            'agentic_reuse_pressure',
            'agentic_adoption_rate',
            'agentic_new_possibility_rate',
        ]
        extra_uncertainty_cols = [
            'actor_ignorance_std',
            'practical_indeterminism_std',
            'agentic_novelty_std',
            'competitive_recursion_std',
            'actor_ignorance_delta',
            'practical_indeterminism_delta',
            'agentic_novelty_delta',
            'competitive_recursion_delta',
        ]
        base_uncertainty_set = set(base_uncertainty_cols + extra_uncertainty_cols)
        tier_uncertainty_cols = [
            col
            for col in self.uncertainty_df.columns
            if col.startswith('competitive_recursion_') and col not in base_uncertainty_set
        ]
        uncertainty_cols = [
            col for col in base_uncertainty_cols + extra_uncertainty_cols + tier_uncertainty_cols
            if col in self.uncertainty_df.columns
        ]

        merged_uncertainty = None
        if uncertainty_cols and not decisions.empty:
            uncertainty_subset = self.uncertainty_df.dropna(subset=['round']).copy()
            uncertainty_subset['round'] = pd.to_numeric(
                uncertainty_subset['round'], errors='coerce'
            )
            uncertainty_subset = uncertainty_subset.dropna(subset=['round'])
            uncertainty_subset['round'] = uncertainty_subset['round'].astype(int)
            uncertainty_subset = uncertainty_subset[['run_id', 'round'] + uncertainty_cols]
            merged_uncertainty = decisions.merge(
                uncertainty_subset, on=['run_id', 'round'], how='left'
            )
            table_uncertainty = (
                merged_uncertainty.groupby(['round', 'ai_level_used'])[uncertainty_cols]
                .mean()
                .reset_index()
            )
            path = _write_table(table_uncertainty, 'uncertainty_by_ai.csv')
            if path:
                exported['uncertainty_by_ai'] = path

        if not self.matured_df.empty:
            matured_export = self.matured_df[self.matured_df['investment_amount'] > 0].copy()
            if not matured_export.empty:
                matured_export['return_multiple'] = matured_export['capital_returned'] / matured_export['investment_amount']
                if 'realized_roi' not in matured_export.columns:
                    matured_export['realized_roi'] = (
                        (matured_export['capital_returned'] - matured_export['investment_amount'])
                        / matured_export['investment_amount']
                    )
                metrics_cols = ['success', 'return_multiple', 'realized_roi']
                matured_export[metrics_cols] = matured_export[metrics_cols].apply(pd.to_numeric, errors='coerce')
                ai_table = (matured_export.groupby('ai_level_used')[metrics_cols]
                            .mean()
                            .reset_index())
                ai_table.rename(columns={'success': 'success_rate', 'return_multiple': 'mean_return_multiple'}, inplace=True)
                path = _write_table(ai_table, 'matured_outcomes_by_ai.csv')
                if path:
                    exported['matured_outcomes_by_ai'] = path

                sector_table = (matured_export.groupby(['sector', 'ai_level_used'])[metrics_cols]
                                .mean()
                                .reset_index())
                sector_table.rename(columns={'success': 'success_rate', 'return_multiple': 'mean_return_multiple'}, inplace=True)
                path = _write_table(sector_table, 'matured_outcomes_by_sector_ai.csv')
                if path:
                    exported['matured_outcomes_by_sector_ai'] = path

        if not self.uncertainty_detail_df.empty:
            detail_export = self.uncertainty_detail_df.copy()
            detail_table = (detail_export.groupby(['ai_level_used', 'strategy_mode'])[
                ['actor_ignorance_level', 'decision_confidence', 'ai_switch_count', 'ai_trust']
            ].mean().reset_index())
            path = _write_table(detail_table, 'uncertainty_detail_by_strategy.csv')
            if path:
                exported['uncertainty_detail_by_strategy'] = path

        total_rounds = None
        if not self.summary_df.empty and 'round' in self.summary_df.columns:
            total_rounds = (
                pd.to_numeric(self.summary_df['round'], errors='coerce').dropna().max()
            )
            if pd.notna(total_rounds):
                total_rounds = int(total_rounds) + 1
        if not total_rounds:
            total_rounds = int(getattr(self.config, 'N_ROUNDS', 0)) or 1

        stage_bounds = stage_bounds or {
            'early': (0, max(1, total_rounds // 3)),
            'middle': (max(1, total_rounds // 3), max(1, (2 * total_rounds) // 3)),
            'late': (max(1, (2 * total_rounds) // 3), total_rounds),
        }

        summary_round = None
        if not self.summary_df.empty and {'run_id', 'round'}.issubset(self.summary_df.columns):
            summary_round = self.summary_df[['run_id', 'round']].copy()
            summary_round['round'] = pd.to_numeric(summary_round['round'], errors='coerce')
            summary_round = summary_round.dropna(subset=['round'])
            summary_round['round'] = summary_round['round'].astype(int)
            available_summary_cols = [
                col
                for col in [
                    'mean_capital',
                    'innovation_success_rate',
                    'mean_agent_knowledge_count',
                    'mean_sector_knowledge',
                ]
                if col in self.summary_df.columns
            ]
            for col in available_summary_cols:
                summary_round[col] = self.summary_df[col]

        knowledge_round = None
        if (
            not self.knowledge_df.empty
            and {'run_id', 'round', 'knowledge_count'}.issubset(self.knowledge_df.columns)
        ):
            knowledge_round = self.knowledge_df[['run_id', 'round', 'knowledge_count']].copy()
            knowledge_round['round'] = pd.to_numeric(
                knowledge_round['round'], errors='coerce'
            )
            knowledge_round = knowledge_round.dropna(subset=['round'])
            knowledge_round['round'] = knowledge_round['round'].astype(int)
            knowledge_round = (
                knowledge_round.groupby(['run_id', 'round'])['knowledge_count']
                .mean()
                .reset_index(name='mean_knowledge_count')
            )

        stage_tables = []
        stage_base = decisions.copy()
        if not stage_base.empty and 'capital_before_action' in stage_base.columns:
            stage_base['agent_capital_at_decision'] = pd.to_numeric(
                stage_base['capital_before_action'], errors='coerce'
            )
        if not stage_base.empty and summary_round is not None:
            stage_base = stage_base.merge(summary_round, on=['run_id', 'round'], how='left')
        if not stage_base.empty and knowledge_round is not None:
            stage_base = stage_base.merge(knowledge_round, on=['run_id', 'round'], how='left')
        if (
            not stage_base.empty
            and merged_uncertainty is not None
        ):
            cols = [
                'agentic_novelty_level',
                'agentic_component_scarcity',
                'agentic_disruption_avg',
                'competitive_recursion_level',
                'competitive_recursion_none',
                'competitive_recursion_basic',
                'competitive_recursion_advanced',
                'competitive_recursion_premium',
            ]
            available = [col for col in cols if col in merged_uncertainty.columns]
            if available:
                novelty_subset = merged_uncertainty[['run_id', 'round'] + available].drop_duplicates()
                stage_base = stage_base.merge(
                    novelty_subset, on=['run_id', 'round'], how='left'
                )
        if not stage_base.empty:
            stage_base = stage_base.loc[:, ~stage_base.columns.duplicated()]

        if not stage_base.empty:
            for stage, (start, end) in stage_bounds.items():
                stage_df = stage_base[
                    (stage_base['round'] >= start) & (stage_base['round'] < end)
                ]
                if stage_df.empty:
                    continue
                value_cols: List[str] = []
                for col in [
                    'agent_capital_at_decision',
                    'success',
                    'mean_knowledge_count',
                    'agentic_novelty_level',
                    'agentic_component_scarcity',
                    'agentic_disruption_avg',
                    'competitive_recursion_level',
                    'agentic_combo_rate',
                    'agentic_reuse_pressure',
                    'agentic_adoption_rate',
                    'agentic_new_possibility_rate',
                    'net_capital_flow_invest',
                ]:
                    if col in stage_df.columns:
                        value_cols.append(col)
                for tier in [
                    'competitive_recursion_none',
                    'competitive_recursion_basic',
                    'competitive_recursion_advanced',
                    'competitive_recursion_premium',
                ]:
                    if tier in stage_df.columns:
                        value_cols.append(tier)
                if not value_cols:
                    continue
                subset = stage_df[['ai_level_used'] + value_cols].copy()
                for col in value_cols:
                    col_data = subset[col]
                    if isinstance(col_data, pd.DataFrame):
                        col_series = col_data.iloc[:, 0]
                    else:
                        col_series = col_data
                    subset[col] = pd.to_numeric(col_series, errors='coerce')
                agg = (
                    subset.groupby('ai_level_used', dropna=False)[value_cols]
                    .mean()
                    .reset_index()
                )
                agg.insert(0, 'stage', stage)
                if 'success' in agg.columns:
                    agg.rename(columns={'success': 'mean_success_rate'}, inplace=True)
                if 'agent_capital_at_decision' in agg.columns:
                    agg.rename(columns={'agent_capital_at_decision': 'mean_capital_active'}, inplace=True)
                rename_map = {
                    'agentic_novelty_level': 'agentic_novelty_level',
                    'agentic_component_scarcity': 'component_scarcity',
                    'agentic_disruption_avg': 'disruption_potential',
                }
                agg.rename(columns={k: v for k, v in rename_map.items() if k in agg.columns}, inplace=True)
                stage_tables.append(agg)

        stage_table = pd.concat(stage_tables, ignore_index=True) if stage_tables else pd.DataFrame()
        path = _write_table(stage_table, 'ai_stage_performance.csv')
        if path:
            exported['ai_stage_performance'] = path

        performance_trend = None
        if not stage_table.empty and 'mean_capital' in stage_table.columns:
            pivot = stage_table.pivot_table(
                index='ai_level_used', columns='stage', values='mean_capital'
            )
            if 'early' in pivot.columns and 'late' in pivot.columns:
                pivot = pivot.reset_index()
                pivot['performance_trend'] = pivot.get('late') - pivot.get('early')
                performance_trend = pivot[['ai_level_used', 'performance_trend']]
                path = _write_table(pivot, 'ai_paradox_metric.csv')
                if path:
                    exported['ai_paradox_metric'] = path

        raw_paradox_path = None
        if not self.summary_df.empty and {'run_id', 'round', 'mean_confidence_invest', 'mean_capital'}.issubset(self.summary_df.columns):
            paradox_df = self.summary_df[['run_id', 'round', 'mean_confidence_invest', 'mean_capital']].copy()
            paradox_df.sort_values(['run_id', 'round'], inplace=True)
            paradox_df['confidence_trend'] = (
                paradox_df.groupby('run_id')['mean_confidence_invest']
                .transform(lambda s: s.rolling(window=6, min_periods=3).mean())
            )
            paradox_df['capital_trend'] = (
                paradox_df.groupby('run_id')['mean_capital']
                .transform(lambda s: s.rolling(window=6, min_periods=3).mean())
            )
            paradox_df['confidence_delta'] = paradox_df.groupby('run_id')['confidence_trend'].diff()
            paradox_df['capital_delta'] = paradox_df.groupby('run_id')['capital_trend'].diff()
            capital_norm = paradox_df['capital_trend'].replace(0, np.nan).abs().fillna(1.0)
            paradox_df['capital_delta_ratio'] = paradox_df['capital_delta'] / capital_norm
            base_condition = (
                (paradox_df['confidence_delta'] > 0.008)
                & (
                    (paradox_df['capital_delta_ratio'] < -0.004)
                    | (paradox_df['capital_delta'] < -25000)
                )
            )
            rolling_flags = (
                paradox_df.assign(_base=base_condition.astype(int))
                .groupby('run_id')['_base']
                .transform(lambda s: s.rolling(window=3, min_periods=2).sum() >= 2)
            )
            paradox_df['paradox_flag'] = rolling_flags.astype(bool)
            raw_paradox = paradox_df.dropna(subset=['confidence_delta', 'capital_delta'])
            raw_paradox_path = _write_table(raw_paradox, 'ai_paradox_signal_raw.csv')
            if raw_paradox_path:
                exported['ai_paradox_signal_raw'] = raw_paradox_path

        if (
            not self.uncertainty_detail_df.empty
            and not self.matured_df.empty
            and {'run_id', 'round', 'ai_level_used', 'decision_confidence'}.issubset(self.uncertainty_detail_df.columns)
        ):
            confidence_base = (
                self.uncertainty_detail_df[['run_id', 'round', 'ai_level_used', 'decision_confidence']]
                .dropna(subset=['round'])
                .copy()
            )
            confidence_stats = (
                confidence_base.groupby(['run_id', 'round', 'ai_level_used'])['decision_confidence']
                .mean()
                .reset_index()
            )

            matured_base = self.matured_df[self.matured_df['investment_amount'] > 0].copy()
            if not matured_base.empty:
                if 'realized_roi' not in matured_base.columns:
                    matured_base['realized_roi'] = (
                        (matured_base['capital_returned'] - matured_base['investment_amount'])
                        / matured_base['investment_amount'].replace(0, np.nan)
                    )
                roi_stats = (
                    matured_base.dropna(subset=['realized_roi'])
                    .groupby(['run_id', 'round', 'ai_level_used'])['realized_roi']
                    .mean()
                    .reset_index()
                )
                if not roi_stats.empty:
                    roi_stats['rolling_roi'] = (
                        roi_stats.groupby(['run_id', 'ai_level_used'])['realized_roi']
                        .transform(lambda s: s.rolling(window=8, min_periods=3).mean())
                    )
                    cohort_df = confidence_stats.merge(
                        roi_stats, on=['run_id', 'round', 'ai_level_used'], how='left'
                    )
                    cohort_df.sort_values(['run_id', 'ai_level_used', 'round'], inplace=True)
                    cohort_df['rolling_roi'] = cohort_df.groupby(['run_id', 'ai_level_used'])['rolling_roi'].ffill()
                    cohort_df = cohort_df.dropna(subset=['rolling_roi'])
                    cohort_df['confidence_trend'] = (
                        cohort_df.groupby(['run_id', 'ai_level_used'])['decision_confidence']
                        .transform(lambda s: s.rolling(window=6, min_periods=3).mean())
                    )
                    cohort_df['confidence_delta'] = cohort_df.groupby(['run_id', 'ai_level_used'])['confidence_trend'].diff()
                    cohort_df['rolling_roi_delta'] = cohort_df.groupby(['run_id', 'ai_level_used'])['rolling_roi'].diff()
                    roi_condition = (
                        (cohort_df['rolling_roi'] < -0.02)
                        | (cohort_df['rolling_roi_delta'] < -0.012)
                    ).fillna(False)
                    base_condition = (cohort_df['confidence_delta'] > 0.01) & roi_condition
                    rolling_flags = (
                        cohort_df.assign(_base=base_condition.astype(int))
                        .groupby(['run_id', 'ai_level_used'])['_base']
                        .transform(lambda s: s.rolling(window=3, min_periods=2).sum() >= 2)
                    )
                    cohort_df['paradox_flag'] = rolling_flags.astype(bool)
                    cohort_df = cohort_df.dropna(subset=['confidence_delta'])
                    cohort_path = _write_table(cohort_df, 'ai_paradox_signal_cohort.csv')
                    if cohort_path:
                        exported['ai_paradox_signal_cohort'] = cohort_path
                    aggregate = (
                        cohort_df.groupby(['run_id', 'round'])[
                            ['decision_confidence', 'rolling_roi', 'confidence_delta', 'rolling_roi_delta', 'paradox_flag']
                        ]
                        .agg({
                            'decision_confidence': 'mean',
                            'rolling_roi': 'mean',
                            'confidence_delta': 'mean',
                            'rolling_roi_delta': 'mean',
                            'paradox_flag': 'max',
                        })
                        .reset_index()
                    )
                    aggregate['paradox_flag'] = aggregate['paradox_flag'].astype(bool)
                    combined_path = _write_table(aggregate, 'ai_paradox_signal.csv')
                    if combined_path:
                        exported['ai_paradox_signal'] = combined_path
            elif raw_paradox_path:
                exported['ai_paradox_signal'] = raw_paradox_path
        if 'ai_paradox_signal' not in exported and raw_paradox_path:
            exported['ai_paradox_signal'] = raw_paradox_path

        # Paradox + uncertainty components by AI tier (action-level)
        if not self.uncertainty_detail_df.empty:
            detail = self.uncertainty_detail_df.copy()
            detail['ai_level_used'] = detail['ai_level_used'].fillna('none')
            cols_map = {
                'perc_actor_ignorance_level': 'actor_ignorance',
                'perc_practical_indeterminism_level': 'practical_indeterminism',
                'perc_agentic_novelty_potential': 'agentic_novelty',
                'perc_competitive_recursion_level': 'competitive_recursion',
            }
            available = {v: k for k, v in cols_map.items() if k in detail.columns}
            agg_cols = list(available.values())
            base_cols = ['paradox_gap', 'paradox_signal']
            use_cols = [c for c in base_cols if c in detail.columns] + agg_cols
            if use_cols:
                grouped = detail.groupby('ai_level_used')[use_cols].mean().reset_index()
                grouped.rename(columns=available, inplace=True)
                path = _write_table(grouped, 'ai_uncertainty_paradox_by_ai.csv')
                if path:
                    exported['ai_uncertainty_paradox_by_ai'] = path

        # Table 3: Uncertainty shift and survival metrics
        table3: Optional[pd.DataFrame] = None
        if merged_uncertainty is not None and not merged_uncertainty.empty:
            early_start, early_end = stage_bounds.get('early', (0, total_rounds // 3))
            late_start, late_end = stage_bounds.get('late', (2 * total_rounds // 3, total_rounds))
            early_mask = (merged_uncertainty['round'] >= early_start) & (
                merged_uncertainty['round'] < early_end
            )
            late_mask = (merged_uncertainty['round'] >= late_start) & (
                merged_uncertainty['round'] < late_end
            )
            early_unc = merged_uncertainty[early_mask]
            late_unc = merged_uncertainty[late_mask]
            if not early_unc.empty and not late_unc.empty:
                early_means = early_unc.groupby('ai_level_used')[uncertainty_cols].mean()
                late_means = late_unc.groupby('ai_level_used')[uncertainty_cols].mean()
                delta_unc = (late_means - early_means).reset_index()
                for col in uncertainty_cols:
                    delta_unc.rename(columns={col: f'delta_{col}'}, inplace=True)
                table3 = delta_unc

        def _merge_table(
            base: Optional[pd.DataFrame], df: Optional[pd.DataFrame]
        ) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return base
            if base is None or base.empty:
                return df.copy()
            return base.merge(df, on='ai_level_used', how='left')

        overall_success = None
        if 'success' in self.decision_df.columns:
            overall_success = (
                self.decision_df.groupby('ai_level_used')['success']
                .mean()
                .reset_index(name='overall_success_rate')
            )

        innovation_success = None
        innov_decisions = self.decision_df[self.decision_df['action'] == 'innovate']
        if not innov_decisions.empty and 'success' in innov_decisions.columns:
            innovation_success = (
                innov_decisions.groupby('ai_level_used')['success']
                .mean()
                .reset_index(name='innovation_success_rate')
            )

        survival_stats = None
        if not self.agent_df.empty and 'primary_ai_canonical' in self.agent_df.columns:
            survival_stats = (
                self.agent_df.groupby('primary_ai_canonical')
                .agg(
                    survival_rate=('survived', 'mean'),
                    survivors=('survived', 'sum'),
                    initial_agents=('survived', 'size'),
                    mean_capital_growth=('capital_growth', 'mean'),
                )
                .reset_index()
                .rename(columns={'primary_ai_canonical': 'ai_level_used'})
            )

        table3 = _merge_table(table3, performance_trend)
        table3 = _merge_table(table3, survival_stats)
        table3 = _merge_table(table3, overall_success)
        table3 = _merge_table(table3, innovation_success)

        path = _write_table(table3, 'ai_uncertainty_survival.csv')
        if path:
            exported['ai_uncertainty_survival'] = path

        # Run rigorous statistical analyses
        try:
            from .statistical_tests import RigorousStatisticalAnalysis, MixedEffectsAnalysis
            print("\n📊 Running rigorous statistical analysis...")

            stat_analysis = RigorousStatisticalAnalysis(
                agent_df=self.agent_df,
                decision_df=self.decision_df,
                matured_df=self.matured_df,
                uncertainty_detail_df=self.uncertainty_detail_df
            )

            # Run all analyses with effect sizes and corrections
            hypothesis_tests = stat_analysis.run_all_analyses()
            descriptive_stats = stat_analysis.generate_descriptive_statistics()
            correlations = stat_analysis.generate_correlation_matrix()

            # Export tables
            if not hypothesis_tests.empty:
                path = output_path / 'statistical_tests_amj.csv'
                hypothesis_tests.to_csv(path, index=False)
                exported['statistical_tests_amj'] = path
                print(f"   ✓ Exported hypothesis tests: {path}")

            if not descriptive_stats.empty:
                path = output_path / 'descriptive_statistics_amj.csv'
                descriptive_stats.to_csv(path, index=False)
                exported['descriptive_statistics_amj'] = path
                print(f"   ✓ Exported descriptive statistics: {path}")

            if not correlations.empty:
                path = output_path / 'correlation_matrix_amj.csv'
                correlations.to_csv(path, index=True)
                exported['correlation_matrix_amj'] = path
                print(f"   ✓ Exported correlation matrix: {path}")

            # Run mixed-effects models for nested data structure
            mixed_analysis = MixedEffectsAnalysis(
                agent_df=self.agent_df,
                decision_df=self.decision_df,
                matured_df=self.matured_df
            )
            mixed_analysis.run_all_models()
            mixed_effects = mixed_analysis.generate_results_table()

            if not mixed_effects.empty:
                path = output_path / 'mixed_effects_models_amj.csv'
                mixed_effects.to_csv(path, index=False)
                exported['mixed_effects_models_amj'] = path
                print(f"   ✓ Exported mixed-effects models: {path}")

        except ImportError as e:
            print(f"   ⚠️ Could not run statistical analysis: {e}")
        except Exception as e:
            print(f"   ⚠️ Statistical analysis error: {e}")

        return {name: str(path) for name, path in exported.items()}

class ComprehensiveVisualizationSuite:
    """Advanced visualization suite for the analysis results."""

    def __init__(self, analysis_framework: ComprehensiveAnalysisFramework):
        if not HAS_PLOTTING_LIBRARIES:
            raise RuntimeError("Matplotlib and seaborn are required for visualization.")
        self.framework = analysis_framework
        self.results = analysis_framework.analyses
        if not self.results:
            self.results = self.framework.run_full_analysis()
        self.agent_df = analysis_framework.agent_df
        self.market_df = analysis_framework.market_df
        self.uncertainty_df = analysis_framework.uncertainty_df
        self.innovation_df = analysis_framework.innovation_df
        self.summary_df = analysis_framework.summary_df
        self.decision_df = analysis_framework.decision_df
        self.knowledge_df = analysis_framework.knowledge_df
        self.matured_df = analysis_framework.matured_df
        self.uncertainty_detail_df = analysis_framework.uncertainty_detail_df
        self.figure_output_dir = getattr(analysis_framework, "figure_output_dir", "")
        self.stats_results = self.framework.analyses.get('statistical_summary', {})
        self.saved_figures = []
        self._summary_by_round_cache = None
        self.color_palette = {
            # Original AI Levels
            'human': '#95a5a6', 'none': '#95a5a6',
            'basic_ai': '#3498db', 'basic': '#3498db',
            'advanced_ai': '#f39c12', 'advanced': '#f39c12',
            'premium_ai': '#e74c3c', 'premium': '#e74c3c', 

            # NEW Behavioral Groups
            'AI Skeptic': '#95a5a6',      # Gray (same as human/none)
            'Cautious Adopter': '#3498db', # Blue (same as basic)
            'Standard User': '#f39c12',     # Orange (same as advanced)
            'Adaptive User': '#2ecc71',      # Green (distinct color for dynamic users)
            'AI Devotee': '#e74c3c',       # Red (same as premium)
            'Unknown': '#7f8c8d',          # A darker gray for any fallbacks

            # Uncertainty colors
            'actor_ignorance': '#9b59b6',
            'practical_indeterminism': '#34495e',
            'agentic_novelty': '#27ae60',
            'competitive_recursion': '#d35400'
        }

    def create_all_visualizations(self):
        """Create the full suite of dashboards."""
        print("\n📊 CREATING COMPREHENSIVE VISUALIZATION SUITE")
        print("=" * 70)
        if self.agent_df.empty:
            print("⚠️ Agent data not found. Skipping visualizations.")
            return
        self.create_perception_storyboard()
        self.create_decision_storyboard()
        self.create_performance_dashboard()
        self.create_market_storyboard()
        # legacy dashboards for backward compatibility
        self.create_temporal_dynamics_dashboard()
        self.create_innovation_dashboard()
        self.create_concentration_dashboard()
        self.create_ai_uncertainty_dashboard()
        self.create_uncertainty_and_market_dashboard()
        print("\n✅ Visualization suite complete!")

    def _save_current_figure(self, fig, name):
        if not self.figure_output_dir:
            return
        safe_name = name.replace(" ", "_").lower()
        path = os.path.join(self.figure_output_dir, f"{safe_name}.png")
        fig.savefig(path, dpi=300)
        self.saved_figures.append(path)

    def _get_summary_by_round(self) -> pd.DataFrame:
        if self._summary_by_round_cache is not None:
            return self._summary_by_round_cache

        summary_numeric = pd.DataFrame()
        if not self.summary_df.empty and {'round', 'run_id'}.issubset(self.summary_df.columns):
            summary_numeric = (self.summary_df.dropna(subset=['round'])
                               .groupby(['run_id', 'round'], as_index=False)
                               .mean(numeric_only=True))

        fallback_summary = self._build_round_metrics_from_decisions()

        if summary_numeric.empty:
            summary_numeric = fallback_summary
        elif not fallback_summary.empty:
            merged = pd.merge(summary_numeric, fallback_summary, on=['run_id', 'round'], how='outer', suffixes=('', '_fallback'))
            for col in fallback_summary.columns:
                if col in ['round', 'run_id']:
                    continue
                fallback_col = f'{col}_fallback'
                if fallback_col in merged.columns:
                    if col not in merged.columns or merged[col].isna().all():
                        merged[col] = merged[fallback_col]
                    else:
                        merged[col] = merged[col].fillna(merged[fallback_col])
                    merged.drop(columns=fallback_col, inplace=True)
            summary_numeric = merged

        if summary_numeric is None:
            summary_numeric = pd.DataFrame()

        if not summary_numeric.empty and {'round', 'run_id'}.issubset(summary_numeric.columns):
            summary_numeric = summary_numeric.sort_values(['run_id', 'round']).reset_index(drop=True)

        self._summary_by_round_cache = summary_numeric
        return summary_numeric

    def _build_round_metrics_from_decisions(self) -> pd.DataFrame:
        if self.decision_df.empty or not {'round', 'run_id'}.issubset(self.decision_df.columns):
            return pd.DataFrame()

        decisions = self.decision_df.dropna(subset=['round']).copy()
        if decisions.empty:
            return pd.DataFrame()

        decisions['round'] = decisions['round'].astype(int)

        rows = []
        for (run_id, round_num), group in decisions.groupby(['run_id', 'round']):
            total_actions = len(group)
            if total_actions == 0:
                continue

            action_share = group['action'].value_counts(normalize=True)
            ai_share = group['ai_level_used'].value_counts(normalize=True)

            hhi = np.nan
            if 'opportunity_id' in group.columns:
                opp_counts = group['opportunity_id'].dropna().value_counts()
                total = opp_counts.sum()
                if total > 0:
                    hhi = float(((opp_counts / total) ** 2).sum())

            top_sector_share = np.nan
            if 'opportunity_sector' in group.columns:
                sector_counts = group['opportunity_sector'].dropna().value_counts()
                total_sector = sector_counts.sum()
                if total_sector > 0:
                    top_sector_share = float(sector_counts.iloc[0] / total_sector)

            innovation_success_rate = np.nan
            innov_mask = group['action'] == 'innovate'
            if innov_mask.any():
                successes = group.loc[innov_mask, 'success'].mean()
                innovation_success_rate = float(successes) if not np.isnan(successes) else np.nan

            row = {
                'run_id': run_id,
                'round': round_num,
                'overall_hhi': hhi,
                'top_sector_share': top_sector_share,
                'innovation_success_rate': innovation_success_rate
            }

            for action in ['invest', 'innovate', 'explore', 'maintain']:
                row[f'action_share_{action}'] = float(action_share.get(action, 0.0))

            for level in ['none', 'basic', 'advanced', 'premium']:
                row[f'ai_share_{level}'] = float(ai_share.get(level, 0.0))

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.sort_values(['run_id', 'round'], inplace=True)
        return df.reset_index(drop=True)

    def create_performance_dashboard(self):
        print("   - Generating Performance Dashboard...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
        fig.suptitle('Performance and Outcomes Analysis', fontsize=24, fontweight='bold')
        self._plot_wealth_distribution(axes[0, 0])
        self._plot_survival_rates(axes[0, 1])
        self._plot_success_factors(axes[1, 0])
        self._plot_performance_vs_innovations(axes[1, 1])
        self._save_current_figure(fig, "All Visualizations")
        self._save_current_figure(fig, "Performance Dashboard")
        plt.show()

    def create_temporal_dynamics_dashboard(self):
        print("   - Generating Temporal Dynamics Dashboard...")
        summary_numeric = self._get_summary_by_round()
        if summary_numeric.empty or 'round' not in summary_numeric.columns:
            print("   ⚠️ Summary data not found. Skipping temporal dashboard.")
            return

        summary_mean = summary_numeric.groupby('round', as_index=False).mean(numeric_only=True)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
        fig.suptitle('Temporal Dynamics of AI-Augmented Entrepreneurship', fontsize=22, fontweight='bold')

        if 'mean_capital' in summary_mean.columns:
            sns.lineplot(data=summary_mean, x='round', y='mean_capital', ax=axes[0, 0], label='Average')
            if 'run_id' in summary_numeric.columns and summary_numeric['run_id'].nunique() > 1:
                sns.lineplot(data=summary_numeric, x='round', y='mean_capital', hue='run_id', ax=axes[0, 0], alpha=0.2, legend=False)
        axes[0, 0].set_title('Average Capital Over Time')
        axes[0, 0].set_ylabel('Mean Capital')

        adoption_cols = [col for col in ['ai_share_none', 'ai_share_basic', 'ai_share_advanced', 'ai_share_premium'] if col in summary_mean.columns]
        if adoption_cols:
            adoption_df = summary_mean[['round'] + adoption_cols].melt(id_vars='round', var_name='ai_level', value_name='share')
            adoption_df['ai_level'] = adoption_df['ai_level'].map({
                'ai_share_none': 'none',
                'ai_share_basic': 'basic',
                'ai_share_advanced': 'advanced',
                'ai_share_premium': 'premium'
            })
            adoption_df = adoption_df.dropna(subset=['share'])
            adoption_palette = {level: self.color_palette.get(level, '#7f8c8d') for level in ['none', 'basic', 'advanced', 'premium']}
            if not adoption_df.empty:
                sns.lineplot(data=adoption_df, x='round', y='share', hue='ai_level', palette=adoption_palette, ax=axes[0, 1])
                if 'run_id' in summary_numeric.columns and summary_numeric['run_id'].nunique() > 1:
                    per_run = summary_numeric[['run_id', 'round'] + adoption_cols].melt(id_vars=['run_id', 'round'], var_name='ai_level', value_name='share')
                    per_run['ai_level'] = per_run['ai_level'].map({
                        'ai_share_none': 'none',
                        'ai_share_basic': 'basic',
                        'ai_share_advanced': 'advanced',
                        'ai_share_premium': 'premium'
                    })
                    sns.lineplot(data=per_run, x='round', y='share', hue='ai_level', ax=axes[0, 1], alpha=0.08, legend=False)
                axes[0, 1].set_ylim(0, 1)
                axes[0, 1].set_title('AI Adoption Share Over Time')
                axes[0, 1].set_ylabel('Share of Active Agents')
                axes[0, 1].legend(title='AI Level')
            else:
                axes[0, 1].text(0.5, 0.5, 'No AI adoption data', ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, 'No AI adoption data', ha='center', va='center', transform=axes[0, 1].transAxes)

        if not self.decision_df.empty and 'action' in self.decision_df.columns:
            action_trends = (self.decision_df.groupby(['run_id', 'round'])['action']
                                             .value_counts(normalize=True)
                                             .unstack(fill_value=0.0)
                                             .reset_index())
            action_mean = action_trends.groupby('round', as_index=False).mean(numeric_only=True)
            action_melt = action_mean.melt(id_vars='round', var_name='action', value_name='share')
            sns.lineplot(data=action_melt, x='round', y='share', hue='action', ax=axes[1, 0])
            if action_trends['run_id'].nunique() > 1:
                per_run_action = action_trends.melt(id_vars=['run_id', 'round'], var_name='action', value_name='share')
                sns.lineplot(data=per_run_action, x='round', y='share', hue='action', ax=axes[1, 0], alpha=0.08, legend=False)
            axes[1, 0].set_title('Action Mix Over Time')
            axes[1, 0].set_ylabel('Share of Actions')
            axes[1, 0].legend(title='Action')
        else:
            axes[1, 0].text(0.5, 0.5, 'No decision data available', ha='center', va='center', transform=axes[1, 0].transAxes)

        if 'innovation_success_rate' in summary_mean.columns:
            sns.lineplot(data=summary_mean, x='round', y='innovation_success_rate', ax=axes[1, 1], label='Average')
            if 'run_id' in summary_numeric.columns and summary_numeric['run_id'].nunique() > 1:
                sns.lineplot(data=summary_numeric, x='round', y='innovation_success_rate', hue='run_id',
                             ax=axes[1, 1], alpha=0.2, legend=False)
            axes[1, 1].set_title('Innovation Success Rate Over Time')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1)
        else:
            axes[1, 1].text(0.5, 0.5, 'Innovation success rate unavailable', ha='center', va='center', transform=axes[1, 1].transAxes)

        self._save_current_figure(fig, "Temporal Dynamics Dashboard")
        plt.show()

    def create_innovation_dashboard(self):
        print("   - Generating Innovation Dynamics Dashboard...")
        summary_numeric = self._get_summary_by_round()
        if summary_numeric.empty and self.innovation_df.empty:
            print("   ⚠️ No innovation data found. Skipping innovation dashboard.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
        fig.suptitle('Innovation Dynamics and Knowledge Diversity', fontsize=22, fontweight='bold')

        if not self.innovation_df.empty:
            impact_series = None
            impact_title = 'Distribution of Innovation Impact'
            impact_label = 'Market Impact'
            if 'market_impact' in self.innovation_df.columns and self.innovation_df['market_impact'].notna().any():
                impact_series = self.innovation_df['market_impact'].dropna()
            else:
                impact_series = (self.innovation_df['quality'] * self.innovation_df['novelty']).rename('impact_proxy')
                impact_title = 'Innovation Impact Proxy (quality × novelty)'
                impact_label = 'Impact Proxy'
            sns.histplot(impact_series, bins=30, ax=axes[0, 0], color='#2ecc71')
            axes[0, 0].set_title(impact_title)
            axes[0, 0].set_xlabel(impact_label)

            if 'type' in self.innovation_df.columns:
                sns.countplot(data=self.innovation_df, x='type', palette='viridis', ax=axes[0, 1])
                axes[0, 1].set_title('Innovation Type Frequency')
                axes[0, 1].set_xlabel('Innovation Type')
            else:
                axes[0, 1].text(0.5, 0.5, 'No innovation type data', ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 0].text(0.5, 0.5, 'No innovation data', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 1].text(0.5, 0.5, 'No innovation data', ha='center', va='center', transform=axes[0, 1].transAxes)

        summary_mean = summary_numeric.groupby('round', as_index=False).mean(numeric_only=True)
        if not summary_mean.empty and 'mean_portfolio_diversity' in summary_mean.columns:
            sns.lineplot(data=summary_mean, x='round', y='mean_portfolio_diversity', ax=axes[1, 0], color='#9b59b6')
            axes[1, 0].set_title('Average Portfolio Diversity Over Time')
            axes[1, 0].set_ylabel('Diversification Score')
        else:
            axes[1, 0].text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=axes[1, 0].transAxes)

        if not summary_mean.empty and 'mean_ai_trust' in summary_mean.columns:
            sns.lineplot(data=summary_mean, x='round', y='mean_ai_trust', ax=axes[1, 1], color='#e74c3c')
            axes[1, 1].set_title('Average AI Trust Over Time')
            axes[1, 1].set_ylabel('Mean Trust')
        else:
            axes[1, 0].text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=axes[1, 1].transAxes)

        self._save_current_figure(fig, "Innovation Dashboard")
        plt.show()

    def create_concentration_dashboard(self):
        print("   - Generating Concentration & Fragility Dashboard...")
        summary_numeric = self._get_summary_by_round()
        has_summary = not summary_numeric.empty
        has_market = not self.market_df.empty and 'round' in self.market_df.columns
        has_decisions = not self.decision_df.empty
        if not (has_summary or has_market or has_decisions):
            print("   ⚠️ Insufficient data for concentration dashboard.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
        fig.suptitle('Concentration and Fragility Metrics', fontsize=22, fontweight='bold')

        summary_mean = summary_numeric.groupby('round', as_index=False).mean(numeric_only=True) if has_summary else pd.DataFrame()

        axes[0, 0].set_title('Investment Concentration (HHI) Over Time')
        axes[0, 0].set_ylabel('HHI')
        if not summary_mean.empty and 'overall_hhi' in summary_mean.columns:
            sns.lineplot(data=summary_mean, x='round', y='overall_hhi', ax=axes[0, 0], color='#34495e', label='Average')
            if 'run_id' in summary_numeric.columns and summary_numeric['run_id'].nunique() > 1:
                sns.lineplot(data=summary_numeric, x='round', y='overall_hhi', hue='run_id', ax=axes[0, 0], alpha=0.15, legend=False)
        else:
            axes[0, 0].text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=axes[0, 0].transAxes)

        if not summary_mean.empty and 'top_sector_share' in summary_mean.columns:
            sns.lineplot(data=summary_mean, x='round', y='top_sector_share', ax=axes[0, 1], color='#f39c12', label='Average')
            axes[0, 1].set_title('Top Sector Market Share Over Time')
            axes[0, 1].set_ylabel('Share')
            axes[0, 1].set_ylim(0, 1)
            if 'run_id' in summary_numeric.columns and summary_numeric['run_id'].nunique() > 1:
                sns.lineplot(data=summary_numeric, x='round', y='top_sector_share', hue='run_id', ax=axes[0, 1], alpha=0.15, legend=False)
        else:
            axes[0, 1].text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=axes[0, 1].transAxes)

        if has_market and 'volatility' in self.market_df.columns:
            market_numeric = (self.market_df.dropna(subset=['round'])
                              .groupby('round', as_index=False)
                              .mean(numeric_only=True)
                              .sort_values('round'))
            sns.lineplot(data=market_numeric, x='round', y='volatility', ax=axes[1, 0])
            axes[1, 0].set_title('Market Volatility Over Time')
            axes[1, 0].set_ylabel('Volatility')
        else:
            axes[1, 0].text(0.5, 0.5, 'No market volatility data', ha='center', va='center', transform=axes[1, 0].transAxes)

        if not self.decision_df.empty and 'opportunity_id' in self.decision_df.columns:
            premium_decisions = self.decision_df[self.decision_df['ai_level_used'] == 'premium']
            if not premium_decisions.empty:
                premium_herding = (premium_decisions.groupby('round')['opportunity_id'].nunique() /
                                   premium_decisions.groupby('round')['opportunity_id'].count()).reset_index(name='unique_share')
                premium_herding = premium_herding.replace([np.inf, -np.inf], np.nan).dropna()
                if not premium_herding.empty:
                    sns.lineplot(data=premium_herding, x='round', y='unique_share', ax=axes[1, 1], color='#e74c3c')
                    axes[1, 1].set_title('Premium AI Opportunity Diversity')
                    axes[1, 1].set_ylabel('Unique Opps / Total')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Insufficient premium AI data', ha='center', va='center', transform=axes[1, 1].transAxes)
            else:
                axes[1, 1].text(0.5, 0.5, 'No premium AI decisions', ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'No decision data', ha='center', va='center', transform=axes[1, 1].transAxes)

        self._save_current_figure(fig, "Concentration Dashboard")
        plt.show()

    def create_ai_uncertainty_dashboard(self):
        print("   - Generating AI vs. Uncertainty Dashboard...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
        fig.suptitle('AI Augmentation vs. Perceived Knightian Uncertainty', fontsize=24, fontweight='bold')
        uncertainty_data = self.results.get('ai_vs_uncertainty', {})

        if not uncertainty_data or not any(uncertainty_data.values()):
            fig.text(0.5, 0.5, 'AI vs. Uncertainty data not available.', ha='center', va='center', fontsize=18, color='red')
        else:
            self._plot_ai_vs_uncertainty_subplot(axes[0, 0], uncertainty_data, 'actor_ignorance', ['level', 'info_sufficiency'])
            self._plot_ai_vs_uncertainty_subplot(axes[0, 1], uncertainty_data, 'practical_indeterminism', ['level', 'regime_stability'])
            self._plot_ai_vs_uncertainty_subplot(axes[1, 0], uncertainty_data, 'agentic_novelty', ['potential', 'creative_confidence'])
            self._plot_ai_vs_uncertainty_subplot(axes[1, 1], uncertainty_data, 'competitive_recursion', ['level', 'herding_awareness'])

        self._save_current_figure(fig, "Ai Uncertainty Dashboard")
        plt.show()

        self._plot_uncertainty_by_ai_tier()
        self._plot_uncertainty_rate_of_change()

    def _plot_action_distribution_over_time(self, ax):
        ax.set_title('Agent Action Distribution Over Time')
        if self.framework.decision_df.empty:
            ax.text(0.5, 0.5, "Decision data missing.", ha='center', va='center', transform=ax.transAxes)
            return

        action_trends = self.framework.decision_df.groupby('round')['action'].value_counts(normalize=True).unstack().fillna(0)

        action_trends.plot(
            kind='area',
            stacked=True,
            ax=ax,
            linewidth=0,
            cmap='viridis'
        )

        ax.set_xlabel('Simulation Round')
        ax.set_ylabel('Proportion of Actions')
        ax.legend(title='Action')
        ax.set_ylim(0, 1)

    def create_perception_storyboard(self):
        print("   - Generating Perception Storyboard...")
        data = self._build_uncertainty_by_ai()
        if data.empty:
            print("      • Skipping perception storyboard (insufficient uncertainty detail).")
            return
        metrics = [
            ("actor_ignorance_level", "Actor Ignorance"),
            ("practical_indeterminism_level", "Practical Indeterminism"),
            ("agentic_novelty_level", "Agentic Novelty"),
            ("competitive_recursion_level", "Competitive Recursion"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, constrained_layout=True)
        palette = {lvl: self.color_palette.get(lvl, "#7f8c8d") for lvl in ["none", "basic", "advanced", "premium"]}
        for ax, (col, title) in zip(axes.flat, metrics):
            for lvl, color in palette.items():
                subset = data[data["ai_level_used"] == lvl].dropna(subset=[col])
                if subset.empty:
                    continue
                subset = subset.sort_values("round")
                disp = canonical_to_display(lvl)
                ax.plot(subset["round"], subset[col], label=disp, color=color, linewidth=2)
                std_col = f"{col}_std"
                if std_col in subset.columns:
                    std_vals = subset[std_col].fillna(0.0).values
                    ax.fill_between(
                        subset["round"],
                        subset[col] - std_vals,
                        subset[col] + std_vals,
                        color=color,
                        alpha=0.15,
                    )
            ax.set_title(title)
            ax.set_ylabel("Perceived Level")
        axes[-1, 0].set_xlabel("Simulation Round")
        axes[-1, 1].set_xlabel("Simulation Round")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, title="AI Level")
        self._save_current_figure(fig, "perception_storyboard")
        plt.show()

    def _plot_ai_adoption_trend(self, ax):
        summary = self._get_summary_by_round()
        adoption_cols = [
            col for col in ["ai_share_none", "ai_share_basic", "ai_share_advanced", "ai_share_premium"]
            if col in summary.columns
        ]
        if summary.empty or not adoption_cols:
            ax.text(0.5, 0.5, "AI adoption data unavailable.", ha="center", va="center", transform=ax.transAxes)
            return
        for col in adoption_cols:
            lvl = col.split("_")[-1]
            disp = canonical_to_display(lvl)
            ax.plot(
                summary["round"],
                summary[col],
                label=disp,
                color=self.color_palette.get(lvl, "#7f8c8d"),
                linewidth=2,
            )
        ax.set_title("AI Adoption Share Over Time")
        ax.set_xlabel("Simulation Round")
        ax.set_ylabel("Share of Active Decisions")
        ax.set_ylim(0, 1)
        ax.legend()

    def _plot_decision_mix_by_tier(self, ax):
        if self.decision_df.empty:
            ax.text(0.5, 0.5, "Decision data unavailable.", ha="center", va="center", transform=ax.transAxes)
            return
        mix = (
            self.decision_df.groupby(["ai_level_used", "action"]).size()
            .unstack(fill_value=0)
            .apply(lambda col: col / col.sum(), axis=1)
        )
        if mix.empty:
            ax.text(0.5, 0.5, "Decision data unavailable.", ha="center", va="center", transform=ax.transAxes)
            return
        display_levels = [lvl for lvl in ["none", "basic", "advanced", "premium"] if lvl in mix.index]
        mix.loc[display_levels].plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=["#1abc9c", "#f39c12", "#8e44ad", "#34495e"],
        )
        ax.set_title("Cumulative Decision Mix by AI Level")
        ax.set_xlabel("AI Level")
        ax.set_ylabel("Decision Share")
        ax.legend(title="Action", bbox_to_anchor=(1.05, 1), loc="upper left")

    def _plot_knowledge_trend(self, ax):
        summary = self._get_summary_by_round()
        if not summary.empty and "mean_knowledge_count" in summary.columns:
            trend = summary.groupby("round")["mean_knowledge_count"].mean()
        elif not self.knowledge_df.empty and {"round", "knowledge_count"}.issubset(self.knowledge_df.columns):
            tmp = (
                self.knowledge_df.dropna(subset=["round"])
                .groupby("round")["knowledge_count"]
                .mean()
            )
            trend = tmp
        else:
            ax.text(0.5, 0.5, "Knowledge tracking unavailable.", ha="center", va="center", transform=ax.transAxes)
            return
        if trend.empty:
            ax.text(0.5, 0.5, "Knowledge tracking unavailable.", ha="center", va="center", transform=ax.transAxes)
            return
        ax.plot(trend.index, trend.values, color="#2ecc71", linewidth=2)
        ax.set_title("Average Knowledge Count")
        ax.set_xlabel("Simulation Round")
        ax.set_ylabel("Mean Knowledge Pieces")

    def create_decision_storyboard(self):
        print("   - Generating Decision Storyboard...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
        self._plot_action_distribution_over_time(axes[0, 0])
        self._plot_ai_adoption_trend(axes[0, 1])
        self._plot_decision_mix_by_tier(axes[1, 0])
        self._plot_knowledge_trend(axes[1, 1])
        axes[0, 0].set_title("Action Distribution Over Time")
        axes[0, 1].set_title("AI Adoption Over Time")
        axes[1, 0].set_title("Decision Mix by AI Level")
        axes[1, 1].set_title("Knowledge Accumulation")
        self._save_current_figure(fig, "decision_storyboard")
        plt.show()

    def create_market_storyboard(self):
        print("   - Generating Market Dynamics Storyboard...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
        fig.suptitle("Market & Uncertainty Storyboard", fontsize=24, fontweight="bold")
        self._plot_uncertainty_evolution(axes[0, 0])
        self._plot_market_regime_frequency(axes[0, 1])
        self._plot_herding_evolution(axes[1, 0])
        self._plot_action_distribution_over_time(axes[1, 1])
        self._save_current_figure(fig, "market_storyboard")
        plt.show()

    def create_uncertainty_and_market_dashboard(self):
        print("   - Generating Market & Uncertainty Dashboard (legacy)...")
        self.create_market_storyboard()

    def _plot_wealth_distribution(self, ax):
        ax.set_title('Final Capital Distribution by Behavioral Group')

        if self.agent_df.empty:
            ax.text(0.5, 0.5, 'No agent data to plot.',
                    ha='center', va='center', transform=ax.transAxes)
            return

        sns.violinplot(data=self.agent_df, x='behavioral_group', y='final_capital',
                       ax=ax, palette=self.color_palette,
                       order=sorted(self.agent_df['behavioral_group'].unique()), cut=0)

        ax.set_ylabel('Final Capital')
        ax.set_xlabel('Emergent Behavioral Group')

    def _plot_survival_by_behavioral_group(self, ax):
        ax.set_title('Agent Survival Rate by Behavioral Group')

        survival_data = self.results.get('performance_outcomes', {}).get('survival_rate_by_behavioral_group', {})

        if not survival_data:
            ax.text(0.5, 0.5, "Data not available.", ha='center', va='center')
            return

        plot_df = pd.DataFrame(list(survival_data.items()), columns=['Emergent Behavioral Group', 'Survival Rate'])
        order = list(survival_data.keys())

        sns.barplot(data=plot_df, x='Emergent Behavioral Group', y='Survival Rate', ax=ax, palette=self.color_palette, order=order)
        ax.set_xlabel('Behavioral Group')
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)

    def _plot_survival_rates(self, ax):
        ax.set_title('Agent Survival Rate by AI Level')

        survival_data = self.results.get('performance_outcomes', {}).get('survival_rate_by_ai', {})

        if not survival_data:
            ax.text(0.5, 0.5, "Survival data not available.", ha='center', va='center', transform=ax.transAxes)
            return

        plot_df = pd.DataFrame(
            [(canonical_to_display(level), rate) for level, rate in survival_data.items()],
            columns=['ai_level_display', 'Survival Rate']
        )
        if plot_df.empty:
            ax.text(0.5, 0.5, "Survival data not available.", ha='center', va='center', transform=ax.transAxes)
            return

        order = [canonical_to_display(level) for level in ['none', 'basic', 'advanced', 'premium']
                 if canonical_to_display(level) in plot_df['ai_level_display'].unique()]
        if not order:
            order = sorted(plot_df['ai_level_display'].unique())
        palette = {label: self.color_palette.get(label, '#7f8c8d') for label in plot_df['ai_level_display'].unique()}

        sns.barplot(
            data=plot_df,
            x='ai_level_display',
            y='Survival Rate',
            ax=ax,
            palette=palette,
            order=order
        )

        ax.set_xlabel('Primary AI Level')
        ax.set_ylabel('Survival Rate')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=0)

    def _plot_success_factors(self, ax):
        ax.set_title('Correlation of Traits with Capital Growth')
        correlation_data = self.results.get('performance_outcomes', {}).get('performance_drivers_correlation', {})
        series = pd.Series(correlation_data)
        if 'capital_growth' in series.index:
            series = series.drop('capital_growth')
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if numeric_series.empty:
            ax.text(0.5, 0.5, 'Insufficient data for correlation plot', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('Spearman Correlation')
            return
        s = numeric_series.sort_values()
        colors = [self.color_palette['premium'] if val > 0 else self.color_palette['none'] for val in s]
        s.plot(kind='barh', ax=ax, color=colors)
        ax.set_xlabel('Spearman Correlation')

    def _plot_performance_vs_innovations(self, ax):
        ax.set_title('Capital Growth vs. Number of Innovations')

        if self.agent_df.empty or 'capital_growth' not in self.agent_df.columns:
            ax.text(0.5, 0.5, 'No capital growth data to plot.',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_xlabel('Total Innovations')
            ax.set_ylabel('Capital Growth')
            return

        hue_col = 'primary_ai_display' if 'primary_ai_display' in self.agent_df.columns else 'primary_ai_level'
        available_levels = [canonical_to_display(level) for level in ['none', 'basic', 'advanced', 'premium']]
        available_levels = [level for level in available_levels if level in self.agent_df[hue_col].unique()]
        if not available_levels:
            available_levels = sorted(self.agent_df[hue_col].dropna().unique())
        palette = {level: self.color_palette.get(level, '#7f8c8d') for level in available_levels}

        sns.scatterplot(
            data=self.agent_df,
            x='innovations',
            y='capital_growth',
            hue=hue_col,
            palette=palette,
            ax=ax,
            alpha=0.6,
            hue_order=available_levels
        )

        ax.set_xlabel('Total Innovations')
        ax.set_ylabel('Capital Growth')

    def _plot_ai_vs_uncertainty_subplot(self, ax, data, uncertainty_type, subdims):
        title = uncertainty_type.replace('_', ' ').title()
        ax.set_title(f'AI vs. {title}')

        plot_data = []
        if data:
            for level in ['none', 'basic', 'advanced', 'premium']:
                for subdim in subdims:
                    value = data.get(level, {}).get(uncertainty_type, {}).get(subdim)
                    if value is not None:
                        plot_data.append({
                            'AI Level': level,
                            'Subdimension': subdim.replace('_', ' ').title(),
                            'Value': value
                        })

        if not plot_data and not self.decision_df.empty:
            for subdim in subdims:
                col = f"perc_{uncertainty_type}_{subdim}" if subdim != 'value' else f"perc_{uncertainty_type}"
                if col in self.decision_df.columns:
                    fallback = self.decision_df.groupby('ai_level_used')[col].mean()
                    for level, value in fallback.items():
                        plot_data.append({
                            'AI Level': level,
                            'Subdimension': subdim.replace('_', ' ').title(),
                            'Value': value
                        })

        if not plot_data:
            ax.text(0.5, 0.5, f'No {title} data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return

        plot_df = pd.DataFrame(plot_data)
        sns.barplot(data=plot_df, x='Subdimension', y='Value', hue='AI Level',
                    palette=self.color_palette, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Average Perceived Level')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=[label.capitalize() for label in labels], title='AI Level')

    def _plot_uncertainty_evolution(self, ax):
        ax.set_title('Evolution of Knightian Uncertainty')
        if self.uncertainty_df.empty or 'round' not in self.uncertainty_df.columns:
            ax.text(0.5, 0.5, "Uncertainty data missing.", ha='center', va='center', transform=ax.transAxes)
            return
        df = (self.uncertainty_df.dropna(subset=['round'])
              .groupby('round')
              .mean(numeric_only=True)
              .sort_index())

        plot_cols = {
            'actor_ignorance_level': 'Actor Ignorance',
            'practical_indeterminism_level': 'Practical Indeterminism',
            'agentic_novelty_level': 'Agentic Novelty',
            'competitive_recursion_level': 'Competitive Recursion'
        }
        plotted = False
        for col, label in plot_cols.items():
            if col in df.columns:
                color_key = col.rsplit('_', 1)[0]
                ax.plot(df.index, df[col], label=label, color=self.color_palette.get(color_key, None))
                plotted = True

        if plotted:
            ax.legend()
            ax.set_xlabel("Simulation Round")
            ax.set_ylabel("Average Level")
        else:
            ax.text(0.5, 0.5, 'Uncertainty metrics unavailable', ha='center', va='center', transform=ax.transAxes)

    def _build_uncertainty_by_ai(self) -> pd.DataFrame:
        source_df: Optional[pd.DataFrame] = None
        metric_map: Dict[str, str] = {}

        # Prefer the detailed uncertainty records (per-agent perceptions).
        if not self.uncertainty_df.empty:
            required = {'round', 'ai_level_used'}
            if required.issubset(self.uncertainty_df.columns):
                source_df = self.uncertainty_df.copy()
                metric_map = {
                    'actor_ignorance_level': 'actor_ignorance_level',
                    'practical_indeterminism_level': 'practical_indeterminism_level',
                    'agentic_novelty_level': 'agentic_novelty_level',
                    'competitive_recursion_level': 'competitive_recursion_level',
                    'agentic_component_scarcity': 'agentic_component_scarcity',
                    'agentic_disruption_avg': 'agentic_disruption_avg',
                    'agentic_combo_rate': 'agentic_combo_rate',
                    'agentic_reuse_pressure': 'agentic_reuse_pressure',
                    'agentic_adoption_rate': 'agentic_adoption_rate',
                    'agentic_new_possibility_rate': 'agentic_new_possibility_rate',
                }
        # Fallback to decision exports if necessary
        if source_df is None:
            if self.decision_df.empty:
                return pd.DataFrame()
            metric_map = {
                'perc_actor_ignorance_level': 'actor_ignorance_level',
                'perc_practical_indeterminism_level': 'practical_indeterminism_level',
                'perc_agentic_novelty_potential': 'agentic_novelty_level',
                'perc_competitive_recursion_level': 'competitive_recursion_level',
                'perc_agentic_component_scarcity': 'agentic_component_scarcity',
                'perc_agentic_disruption_avg': 'agentic_disruption_avg',
                'perc_agentic_combo_rate': 'agentic_combo_rate',
                'perc_agentic_reuse_pressure': 'agentic_reuse_pressure',
                'perc_agentic_adoption_rate': 'agentic_adoption_rate',
                'perc_agentic_new_possibility_rate': 'agentic_new_possibility_rate',
            }
            available = {src: dst for src, dst in metric_map.items() if src in self.decision_df.columns}
            if not available:
                return pd.DataFrame()
            cols = ['run_id', 'round', 'ai_level_used'] + list(available.keys())
            source_df = self.decision_df[cols].dropna(subset=['round']).copy()
            metric_map = available

        if source_df is None or source_df.empty:
            return pd.DataFrame()

        source_df['round'] = pd.to_numeric(source_df['round'], errors='coerce')
        source_df = source_df.dropna(subset=['round'])
        if source_df.empty:
            return pd.DataFrame()
        source_df['round'] = source_df['round'].astype('Int64')

        agg_dict = {src: ['mean', 'std'] for src in metric_map.keys()}
        grouped = (
            source_df.groupby(['round', 'ai_level_used'])
            .agg(agg_dict)
            .reset_index()
        )
        if grouped.empty:
            return grouped

        def _flatten_column(col: Any) -> Any:
            if not isinstance(col, tuple):
                return col
            base, stat = col
            target = metric_map.get(base, base)
            if stat in (None, '', 'mean'):
                return target
            suffix = 'std' if stat == 'std' else str(stat)
            return f"{target}_{suffix}"

        grouped.columns = [_flatten_column(col) for col in grouped.columns]
        grouped = grouped.loc[:, ~grouped.columns.duplicated()]

        for dst in metric_map.values():
            if dst in grouped.columns:
                grouped[f'{dst}_delta'] = (
                    grouped.sort_values('round')
                    .groupby('ai_level_used')[dst]
                    .diff()
                    .fillna(0.0)
                )

        return grouped

    def _plot_uncertainty_by_ai_tier(self):
        data = self._build_uncertainty_by_ai()
        if data.empty:
            print("      • Skipping AI-tier uncertainty plot (insufficient data).")
            return

        plot_cols = {
            'actor_ignorance_level': 'Actor Ignorance',
            'practical_indeterminism_level': 'Practical Indeterminism',
            'agentic_novelty_level': 'Agentic Novelty',
            'competitive_recursion_level': 'Competitive Recursion',
        }
        fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True)
        axes = axes.flatten()
        palette = {lvl: self.color_palette.get(lvl, '#333333') for lvl in ['none', 'basic', 'advanced', 'premium']}

        for ax, (col, label) in zip(axes, plot_cols.items()):
            if col not in data.columns:
                ax.text(0.5, 0.5, f'{label} unavailable', ha='center', va='center', transform=ax.transAxes)
                continue
            sns.lineplot(
                data=data,
                x='round',
                y=col,
                hue='ai_level_used',
                hue_order=['none', 'basic', 'advanced', 'premium'],
                palette=palette,
                ax=ax,
            )
            ax.set_title(f'{label} by AI Tier')
            ax.set_ylabel('Average Level')
            ax.set_xlabel('Round')
            ax.legend(title='AI level')

        self._save_current_figure(fig, 'uncertainty_by_ai_tier')
        plt.show()

    def _plot_uncertainty_rate_of_change(self):
        if self.uncertainty_df.empty:
            print("      • Skipping uncertainty rate-of-change plot (no data).")
            return

        delta_cols = [
            ('actor_ignorance_delta', 'Actor Ignorance Δ'),
            ('practical_indeterminism_delta', 'Practical Indeterminism Δ'),
            ('agentic_novelty_delta', 'Agentic Novelty Δ'),
            ('competitive_recursion_delta', 'Competitive Recursion Δ'),
        ]
        available = [col for col, _ in delta_cols if col in self.uncertainty_df.columns]
        if not available:
            print("      • Skipping uncertainty rate-of-change plot (delta metrics missing).")
            return

        df = (self.uncertainty_df.dropna(subset=['round'])
              .groupby('round')
              .mean(numeric_only=True)
              .sort_index())
        if df.empty:
            print("      • Skipping uncertainty rate-of-change plot (no aggregated data).")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        for col, label in delta_cols:
            if col in df.columns:
                base_key = col.replace('_delta', '')
                ax.plot(df.index, df[col], label=label, color=self.color_palette.get(base_key, None))

        ax.axhline(0.0, color='black', linestyle='--', linewidth=1)
        ax.set_title('Rate of Change in Knightian Uncertainty')
        ax.set_xlabel('Simulation Round')
        ax.set_ylabel('Δ (Round-over-Round)')
        ax.legend()
        self._save_current_figure(fig, 'uncertainty_rate_of_change')
        plt.show()

    def _plot_market_regime_frequency(self, ax):
        ax.set_title('Market Regime Frequency')
        freqs = self.results.get('knightian_dynamics', {}).get('market_regime_frequency', {})
        if not freqs:
            if self.market_df.empty or 'regime' not in self.market_df.columns:
                ax.text(0.5, 0.5, 'No regime data available.', ha='center', va='center', transform=ax.transAxes)
                return
            freqs = self.market_df['regime'].value_counts(normalize=True).to_dict()
        labels = [label.capitalize() for label in freqs.keys()]
        sizes = list(freqs.values())
        if not sizes or sum(sizes) == 0:
            ax.text(0.5, 0.5, 'No regime data available.', ha='center', va='center', transform=ax.transAxes)
            return
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    def _plot_herding_evolution(self, ax):
        ax.set_title('Evolution of Herding Behavior')

        if self.uncertainty_df.empty or 'round' not in self.uncertainty_df.columns:
            ax.text(0.5, 0.5, "Uncertainty data missing.", ha='center', va='center', transform=ax.transAxes)
            return

        herding_col = None
        possible_herding_cols = ['competitive_recursion_herding_intensity', 'herding_intensity', 'competitive_recursion_level']

        for col in possible_herding_cols:
            if col in self.uncertainty_df.columns:
                herding_col = col
                break

        if herding_col is None:
            ax.text(0.5, 0.5, "Herding data not available.\nAvailable columns: " +
                    ", ".join(self.uncertainty_df.columns), ha='center', va='center', transform=ax.transAxes)
            return

        try:
            herding_df = (self.uncertainty_df.dropna(subset=['round'])
                          .groupby('round')[herding_col]
                          .mean())
            color = self.color_palette.get('competitive_recursion', '#d35400')
            if herding_df.empty:
                ax.text(0.5, 0.5, "Herding data not available.", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.plot(herding_df.index, herding_df.values, color=color)
                ax.fill_between(herding_df.index, herding_df.values, color=color, alpha=0.3)
                ax.set_xlabel('Simulation Round')
                ax.set_ylabel('Average Herding Intensity')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error creating herding plot: {str(e)}", ha='center', va='center', transform=ax.transAxes)

    def _plot_volatility_vs_innovations(self, ax):
        ax.set_title("Market Volatility vs. Innovation Rate")

        if self.market_df.empty or self.innovation_df.empty:
            ax.text(0.5, 0.5, "Market or Innovation data missing.", ha='center', va='center', transform=ax.transAxes)
            return

        innovation_rate = self.innovation_df.groupby('round').size() / self.framework.config.N_AGENTS

        plot_data = self.market_df.set_index('round').join(innovation_rate.rename('innovation_rate')).dropna()

        if not plot_data.empty:
            sns.scatterplot(data=plot_data, x='volatility', y='innovation_rate', ax=ax, alpha=0.6)
            ax.set_xlabel("Market Volatility")
            ax.set_ylabel("Innovation Rate (Innovations per Agent)")
        else:
            ax.text(0.5, 0.5, "No matching data to plot.", ha='center', va='center', transform=ax.transAxes)

class StatisticalAnalysisSuite:
    """Comprehensive statistical analysis suite for ABM results."""

    def __init__(self, analysis_framework: ComprehensiveAnalysisFramework):
        self.framework = analysis_framework
        self.agent_df = analysis_framework.agent_df
        self.decision_df = analysis_framework.decision_df
        self.statistical_tests = {}
        self.models = {}

    def run_comprehensive_analysis(self) -> Dict:
        """Run all statistical analyses and return a summary dictionary."""
        print("\n📊 RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
        print("=" * 70)

        self.statistical_tests['ai_effectiveness'] = self._test_ai_effectiveness()
        print("🔬 Testing AI Impact on Uncertainty Subdimensions...")
        self.statistical_tests['ai_vs_uncertainty_subdimensions'] = self._test_ai_vs_uncertainty_subdimensions()

        self.models = {
            'performance_prediction': self._build_performance_prediction_model(),
            'survival_analysis': self._perform_survival_analysis(),
            'strategy_clusters': self._perform_clustering_analysis()
        }

        return {'hypothesis_tests': self.statistical_tests, 'predictive_models': self.models}

    def _test_ai_effectiveness(self) -> Dict:
        """Test the hypothesis that AI augmentation improves performance."""
        print("🔬 Testing AI effectiveness on performance...")
        if self.agent_df.empty: return {}

        ai_levels = sorted(self.agent_df['primary_ai_level'].unique())
        performance_by_ai = [self.agent_df[self.agent_df['primary_ai_level'] == level]['capital_growth'] for level in ai_levels]

        valid_groups = [g for g in performance_by_ai if len(g) > 1]
        if len(valid_groups) < 2:
            return {'status': f"Skipped: Only {len(valid_groups)} AI group(s) with sufficient data."}

        h_stat, p_value = kruskal(*valid_groups)
        return {'test': 'Kruskal-Wallis (Performance vs. AI Level)', 'h_statistic': h_stat, 'p_value': p_value}

    def _test_ai_vs_uncertainty_subdimensions(self) -> Dict:
        """Runs Kruskal-Wallis H-test for perceived uncertainty across AI levels."""
        if self.decision_df.empty: return {}

        subdimensions_to_test = [col for col in self.decision_df.columns if col.startswith('perc_')]
        test_results = {}

        for dim in subdimensions_to_test:
            groups = [self.decision_df[self.decision_df['ai_level_used'] == level][dim].dropna() for level in ['none', 'basic', 'advanced', 'premium']]
            valid_groups = [g for g in groups if len(g) > 2]
            if len(valid_groups) < 2: continue

            try:
                h_stat, p_value = kruskal(*valid_groups)
                test_results[dim] = {'h_statistic': h_stat, 'p_value': p_value, 'is_significant': p_value < 0.05}
            except ValueError:
                test_results[dim] = {'error': 'Could not perform test.'}

        return test_results

    def _build_performance_prediction_model(self) -> Dict:
        """Build a model to identify key drivers of performance."""
        print("🔮 Building performance prediction model...")
        if self.agent_df.empty or len(self.agent_df) < 10: return {}

        features = ['uncertainty_tolerance', 'innovativeness', 'exploration_tendency', 'ai_trust', 'innovations', 'portfolio_diversity']
        X = self.agent_df[features].copy()
        y = self.agent_df['capital_growth']
        X = pd.concat([X, pd.get_dummies(self.agent_df['primary_ai_level'], prefix='ai')], axis=1)

        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if X.empty:
            return {'status': 'Insufficient clean data for performance model.'}

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=1,  # thread-safe for constrained environments
        )
        mean_r2 = None
        if len(X) >= 20:
            test_size = 0.25 if len(X) < 80 else 0.2
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            model.fit(X_train, y_train)
            try:
                mean_r2 = float(model.score(X_test, y_test))
            except ValueError:
                mean_r2 = None

        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        return {'model_type': 'RandomForestRegressor', 'mean_cv_r2': mean_r2, 'feature_importances': importances.to_dict()}

    def _perform_clustering_analysis(self) -> Dict:
        """Identify emergent agent strategies through clustering."""
        print("🎯 Identifying emergent agent strategies...")

        features = ['uncertainty_tolerance', 'innovativeness', 'exploration_tendency', 'portfolio_diversity', 'innovations']

        if len(self.agent_df) < 10: return {'status': 'Insufficient agent data for clustering.'}
        X = self.agent_df[features].fillna(0)
        X = X.loc[:, X.std() > 0]
        if X.shape[1] < 2: return {'status': 'Not enough feature variance for clustering.'}

        X_scaled = StandardScaler().fit_transform(X)

        best_k, best_score = -1, -1
        for k in range(3, 6):
            if len(X_scaled) < k: continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score = k, score

        if best_k == -1: return {'status': 'Could not determine optimal number of clusters.'}

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.agent_df['strategy_cluster'] = kmeans.fit_predict(X_scaled)

        cluster_profiles = self.agent_df.groupby('strategy_cluster')[features + ['capital_growth', 'survived']].mean()

        return {'num_clusters': best_k, 'silhouette_score': best_score, 'cluster_profiles': cluster_profiles.to_dict('index')}

    def _perform_survival_analysis(self) -> Dict:
        """Analyze agent survival using Cox Proportional Hazards model."""
        print("⏱️ Performing survival analysis...")
        if not HAS_LIFELINES or CoxPHFitter is None:
            return {"status": "lifelines package not available; skipping survival analysis."}
        if self.agent_df.empty or 'survived' not in self.agent_df.columns: return {}
        
        if len(self.agent_df) < 20:  # Cox model needs sufficient data
            return {"status": "Insufficient data for survival analysis (need at least 20 agents)"}
    
        try:
            df_survival = self.agent_df.copy()
            # Simplified duration for model stability
            df_survival['duration'] = self.framework.config.N_ROUNDS
            df_survival['event'] = 1 - df_survival['survived'] # Event is failure

            features = ['uncertainty_tolerance', 'innovations', 'portfolio_diversity']
            df_for_cox = pd.concat([df_survival[['duration', 'event'] + features], pd.get_dummies(df_survival['primary_ai_level'], prefix='ai', drop_first=True)], axis=1)

            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df_for_cox, duration_col='duration', event_col='event')

            return {'model': 'Cox Proportional Hazards', 'concordance_index': cph.concordance_index_, 'summary': cph.summary.to_dict('index')}
        except Exception as e:
            return {"status": f"Survival analysis failed: {e}"}

class InformationParadoxAnalyzer:
    """
    Analyzes the paradox of information in AI-augmented decision making.
    Examines how early information advantage can become late-stage disadvantage.
    """
    
    def __init__(self, results_directory: str, config: EmergentConfig = None):
        self.results_dir = results_directory
        self.config = config or EmergentConfig()
        
        # Load all data
        self.framework = ComprehensiveAnalysisFramework(results_directory, self.config)
        self.decisions = self.framework.decision_df.copy()
        if not self.decisions.empty and 'round' not in self.decisions.columns:
            for alt_round_col in ['round_num', 'round_number', 'round_x', 'Round']:
                if alt_round_col in self.decisions.columns:
                    self.decisions = self.decisions.rename(columns={alt_round_col: 'round'})
                    break
            else:
                print("Trap analysis warning: 'round' column missing from decisions dataframe.")
        self.agents = self.framework.agent_df
        self.market = self.framework.market_df
        self.uncertainty = self.framework.uncertainty_df
        self.summary = self.framework.summary_df
        
        # Define early/middle/late stage boundaries
        total_rounds = self.config.N_ROUNDS
        self.stage_boundaries = {
            'early': (0, total_rounds // 3),
            'middle': (total_rounds // 3, 2 * total_rounds // 3),
            'late': (2 * total_rounds // 3, total_rounds)
        }
        
        # Store results
        self.paradox_metrics = {}
    def analyze_none_category_advantages(self):
        """Analyze specific advantages of non-AI agents in avoiding the information paradox."""
        print("\n📊 Analyzing Non-AI Agent Advantages...")

        advantages = {}

        # 1. Diversity Preservation
        # Non-AI agents might maintain more diverse strategies
        if 'path_dependency' in self.paradox_metrics:
            df = self.paradox_metrics['path_dependency']
            none_diversity = df[df['primary_ai'] == 'none']['diversity'].mean()
            ai_diversity = df[df['primary_ai'] != 'none']['diversity'].mean()
            advantages['diversity_preservation'] = none_diversity - ai_diversity
            if np.isnan(advantages['diversity_preservation']):
                advantages['diversity_preservation'] = 0.0

        # 2. Anti-Herding Behavior
        # Non-AI agents might avoid information cascades
        if 'herding' in self.paradox_metrics:
            df = self.paradox_metrics['herding']
            none_herding = df[df['ai_level'] == 'none']['hhi'].mean()
            ai_herding = df[df['ai_level'] != 'none']['hhi'].mean()
            advantages['anti_herding'] = ai_herding - none_herding  # Higher is better for none
            if np.isnan(advantages['anti_herding']):
                advantages['anti_herding'] = 0.0

        # 3. Contrarian Success
        # Measure when going against the crowd pays off
        contrarian_success = self._calculate_contrarian_returns()
        advantages['contrarian_success'] = contrarian_success

        # 4. Information Cost Efficiency
        # ROI considering zero information costs
        advantages['cost_efficiency'] = self._calculate_information_roi()

        self.paradox_metrics['none_advantages'] = advantages
        return advantages

    def _calculate_contrarian_returns(self):
        """Helper method to measure success of non-herding behavior."""
        if self.decisions.empty or 'opportunity_id' not in self.decisions.columns:
            return 0.0

        investment_decisions = self.decisions[self.decisions['action'] == 'invest']
        if investment_decisions.empty:
            return 0.0

        # Identify opportunities with high herding
        opportunity_counts = investment_decisions['opportunity_id'].value_counts()
        herded_opportunities = opportunity_counts[opportunity_counts > 5].index

        # Filter for non-AI agents investing in non-herded opportunities
        contrarian_investments = investment_decisions[
            (investment_decisions['ai_level_used'] == 'none') &
            (~investment_decisions['opportunity_id'].isin(herded_opportunities))
        ]

        if contrarian_investments.empty:
            return 0.0

        # Compare their success rate to the overall market success rate
        contrarian_success_rate = contrarian_investments['success'].mean()
        overall_success_rate = investment_decisions['success'].mean()

        return contrarian_success_rate - overall_success_rate

    def _calculate_information_roi(self):
        """Helper method to calculate return on information cost for each AI level."""
        if self.agents.empty:
            return {}

        # Calculate ROI for each AI level
        roi = {}
        canonical_series = (self.agents['primary_ai_canonical']
                            if 'primary_ai_canonical' in self.agents.columns
                            else self.agents['primary_ai_level'].apply(normalize_ai_label))
        for ai_level in ['none', 'basic', 'advanced', 'premium']:
            level_mask = canonical_series == ai_level
            level_agents = self.agents[level_mask]
            if level_agents.empty:
                roi[ai_level] = 0.0
                continue

            total_capital_growth = level_agents['capital_growth'].sum()
            ai_cost = self.config.AI_LEVELS[ai_level]['cost'] * len(level_agents) * self.config.N_ROUNDS

            if ai_level == 'none':
                # For non-AI, information cost is effectively zero, so ROI is just a measure of performance
                roi[ai_level] = total_capital_growth / len(level_agents)
            elif ai_cost > 0:
                # For others, calculate a simple ROI
                net_capital_gain = (total_capital_growth - len(level_agents)) * self.config.INITIAL_CAPITAL
                roi[ai_level] = net_capital_gain / ai_cost
            else:
                roi[ai_level] = 0.0
        
        # We need a single metric for the advantage of 'none' agents
        # Let's use the ratio of non-AI ROI to the average AI ROI
        ai_rois = [roi[l] for l in roi if l != 'none']
        if ai_rois:
            avg_ai_roi = np.mean(ai_rois)
            if avg_ai_roi > 0:
                return roi['none'] / avg_ai_roi - 1.0 # Advantage > 0 if better than average AI
        
        return 0.0
    
    def analyze_strategic_heterogeneity(self):
        """Analyze how heterogeneity in non-AI agents creates resilience."""
        heterogeneity_metrics = {}
        for stage_name, (start, end) in self.stage_boundaries.items():
            stage_decisions = self.decisions[
                (self.decisions['round'] >= start) &
                (self.decisions['round'] < end)
            ]

            # Calculate strategy entropy for each AI level
            for ai_level in ['none', 'basic', 'advanced', 'premium']:
                level_decisions = stage_decisions[
                    stage_decisions['ai_level_used'] == ai_level
                ]

                if len(level_decisions) > 10:
                    # Calculate action diversity
                    action_entropy = entropy(
                        level_decisions['action'].value_counts(normalize=True)
                    )

                    # Calculate timing diversity (when agents act)
                    timing_std = level_decisions.groupby('agent_id')['round'].std().mean()
                    if np.isnan(timing_std):
                        timing_std = 0.0

                    heterogeneity_metrics[f"{stage_name}_{ai_level}"] = {
                        'action_diversity': action_entropy,
                        'timing_diversity': timing_std
                    }
        self.paradox_metrics['strategic_heterogeneity'] = heterogeneity_metrics
        return heterogeneity_metrics

    def test_none_resilience(self):
        """Test if non-AI agents show more resilience to market shocks."""
        # Identify crisis periods
        crisis_rounds = self.market[self.market['regime'] == 'crisis']['round'].unique()

        if len(crisis_rounds) > 0:
            # Compare performance during and after crises
            crisis_performance = {}

            for ai_level in ['none', 'basic', 'advanced', 'premium']:
                agent_decisions = self.decisions[
                    self.decisions['ai_level_used'] == ai_level
                ]

                # During crisis
                during_crisis = agent_decisions[
                    agent_decisions['round'].isin(crisis_rounds)
                ]['success'].mean()
                if np.isnan(during_crisis): during_crisis = 0.0

                # Recovery (5 rounds after crisis)
                recovery_rounds = [r + i for r in crisis_rounds for i in range(1, 6)]
                recovery = agent_decisions[
                    agent_decisions['round'].isin(recovery_rounds)
                ]['success'].mean()
                if np.isnan(recovery): recovery = 0.0

                crisis_performance[ai_level] = {
                    'during': during_crisis,
                    'recovery': recovery,
                    'resilience': recovery - during_crisis
                }
            self.paradox_metrics['crisis_resilience'] = crisis_performance
            return crisis_performance
        return None

    def _plot_none_as_solution(self, ax):
        """Plot scenarios where non-AI agents avoid the information paradox."""
        if 'none_advantages' not in self.paradox_metrics:
            ax.text(0.5, 0.5, 'No non-AI advantages data.', ha='center', va='center')
            return
            
        advantages = self.paradox_metrics['none_advantages']
        
        # Use a list of metrics and create a DataFrame
        metrics = list(advantages.keys())
        scores = [advantages[m] for m in metrics]
        
        # Create a single column DataFrame
        plot_df = pd.DataFrame(scores, index=[m.replace('_', ' ').title() for m in metrics], columns=['Advantage Score'])
        
        # Use a heatmap for visualization
        sns.heatmap(plot_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Advantage Score'})

        ax.set_title('Non-AI Strategies as Paradox Solution\n(Green = Advantage)')
        ax.set_ylabel('') # Remove y-label for clarity
        ax.tick_params(axis='x', rotation=0)
   
    def run_full_analysis(self):
        """Execute all paradox analyses and create visualizations."""
        print("\n🔬 INFORMATION PARADOX ANALYSIS")
        print("=" * 70)

        # 1. Temporal Performance Reversal Analysis
        self.analyze_temporal_performance_reversal()

        # 2. Information-Competition Dynamics
        self.analyze_information_competition_dynamics()

        # 3. Trust-Performance Decoupling
        self.analyze_trust_performance_decoupling()

        # 4. Path Dependency and Lock-in Effects
        self.analyze_path_dependency_effects()

        # 5. Agentic Novelty Saturation
        self.analyze_novelty_saturation()

        # 6. Herding and Diversity Loss
        self.analyze_herding_dynamics()
        
        # 7. Comparative Advantage Analysis
        self.analyze_none_category_advantages()
        
        # 8. Heterogeneity Analysis
        self.analyze_strategic_heterogeneity()
        
        # 9. Statistical Tests for Resilience
        crisis_results = self.test_none_resilience()
        if crisis_results:
            self.paradox_metrics['crisis_resilience'] = crisis_results
        
        # 10. Statistical Tests for Paradox
        self.run_statistical_tests()
        
        # 11. Create Comprehensive Dashboard
        self.create_paradox_dashboard()

        return self.paradox_metrics

    def analyze_temporal_performance_reversal(self):
        """Analyze if AI users outperform early but underperform late."""
        print("\n📊 Analyzing Temporal Performance Reversal...")
        
        if self.decisions.empty:
            print("   No decision data available")
            return
        
        # Calculate success rates by stage and AI level
        stages_data = []
        
        for stage_name, (start, end) in self.stage_boundaries.items():
            stage_decisions = self.decisions[
                (self.decisions['round'] >= start) & 
                (self.decisions['round'] < end)
            ]
            
            if 'success' in stage_decisions.columns:
                success_by_ai = stage_decisions.groupby('ai_level_used')['success'].agg([
                    'mean', 'count', 'std'
                ]).reset_index()
                success_by_ai['stage'] = stage_name
                stages_data.append(success_by_ai)
        
        if stages_data:
            temporal_df = pd.concat(stages_data)
            self.paradox_metrics['temporal_reversal'] = temporal_df
            
            # Calculate reversal score
            if len(temporal_df) > 0:
                early_ai_advantage = (
                    temporal_df[(temporal_df['stage'] == 'early') & 
                               (temporal_df['ai_level_used'] != 'none')]['mean'].mean() -
                    temporal_df[(temporal_df['stage'] == 'early') & 
                               (temporal_df['ai_level_used'] == 'none')]['mean'].mean()
                )
                
                late_ai_advantage = (
                    temporal_df[(temporal_df['stage'] == 'late') & 
                               (temporal_df['ai_level_used'] != 'none')]['mean'].mean() -
                    temporal_df[(temporal_df['stage'] == 'late') & 
                               (temporal_df['ai_level_used'] == 'none')]['mean'].mean()
                )
                
                reversal_score = early_ai_advantage - late_ai_advantage
                self.paradox_metrics['reversal_score'] = reversal_score
                print(f"   Reversal Score: {reversal_score:.3f}")
                print(f"   (Positive = AI advantage decreases over time)")
    
    def analyze_information_competition_dynamics(self):
        """Analyze how information symmetry creates competition."""
        print("\n📊 Analyzing Information-Competition Dynamics...")
        
        # Calculate opportunity concentration by round and AI level
        concentration_data = []
        
        for round_num in self.decisions['round'].unique():
            round_decisions = self.decisions[self.decisions['round'] == round_num]
            
            # Calculate Herfindahl index for each AI level
            for ai_level in ['none', 'basic', 'advanced', 'premium']:
                ai_decisions = round_decisions[round_decisions['ai_level_used'] == ai_level]
                
                if len(ai_decisions) > 0 and 'opportunity_id' in ai_decisions.columns:
                    # Count investments per opportunity
                    opp_counts = ai_decisions['opportunity_id'].value_counts()
                    total = len(ai_decisions)
                    
                    # Calculate Herfindahl-Hirschman Index
                    hhi = sum((count/total)**2 for count in opp_counts.values)
                    
                    concentration_data.append({
                        'round': round_num,
                        'ai_level': ai_level,
                        'hhi': hhi,
                        'n_unique_opps': len(opp_counts),
                        'n_decisions': total
                    })
        
        if concentration_data:
            concentration_df = pd.DataFrame(concentration_data)
            self.paradox_metrics['concentration'] = concentration_df
            
            # Calculate average concentration by stage
            for stage_name, (start, end) in self.stage_boundaries.items():
                stage_conc = concentration_df[
                    (concentration_df['round'] >= start) & 
                    (concentration_df['round'] < end)
                ]
                
                avg_hhi = stage_conc.groupby('ai_level')['hhi'].mean()
                print(f"\n   {stage_name.capitalize()} Stage HHI by AI Level:")
                for level, hhi in avg_hhi.items():
                    print(f"      {level}: {hhi:.3f}")
    
    def analyze_trust_performance_decoupling(self):
        """Analyze how trust evolves differently from actual performance."""
        print("\n📊 Analyzing Trust-Performance Decoupling...")
        
        # Track trust evolution vs actual accuracy
        trust_performance_data = []
        
        # Group decisions by agent and round
        if 'agent_id' in self.decisions.columns:
            for agent_id in self.decisions['agent_id'].unique():
                agent_decisions = self.decisions[self.decisions['agent_id'] == agent_id]
                
                # Get agent's trust level from final state
                if agent_id in self.agents.index or 'agent_id' in self.agents.columns:
                    if 'agent_id' in self.agents.columns:
                        agent_data = self.agents[self.agents['agent_id'] == agent_id]
                    else:
                        agent_data = self.agents.loc[[agent_id]]
                    
                    if not agent_data.empty and 'ai_trust' in agent_data.columns:
                        final_trust = agent_data['ai_trust'].iloc[0]
                        
                        # Calculate actual AI performance for this agent
                        ai_decisions = agent_decisions[agent_decisions['ai_level_used'] != 'none']
                        # Use realized performance rather than decision success
                        if 'capital_growth' in agent_data.columns:
                            actual_performance = agent_data['capital_growth'].iloc[0]
                        elif 'final_capital' in agent_data.columns:
                            actual_performance = agent_data['final_capital'].iloc[0]
                        else:
                            continue

                        trust_performance_data.append({
                            'agent_id': agent_id,
                            'final_trust': final_trust,
                            'actual_performance': actual_performance,
                            'trust_performance_gap': final_trust - actual_performance
                        })
        
        if trust_performance_data:
            trust_df = pd.DataFrame(trust_performance_data)
            self.paradox_metrics['trust_decoupling'] = trust_df
            
            correlation = trust_df['final_trust'].corr(trust_df['actual_performance'])
            print(f"   Trust-Performance Correlation: {correlation:.3f}")
            print(f"   Mean Trust-Performance Gap: {trust_df['trust_performance_gap'].mean():.3f}")
    
    def analyze_path_dependency_effects(self):
        """Analyze how early choices constrain later options."""
        print("\n📊 Analyzing Path Dependency Effects...")
        
        # Track portfolio concentration over time
        path_dependency_data = []
        
        if 'agent_id' in self.decisions.columns:
            for agent_id in self.decisions['agent_id'].unique():
                agent_decisions = self.decisions[
                    (self.decisions['agent_id'] == agent_id) & 
                    (self.decisions['action'] == 'invest')
                ]
                
                if len(agent_decisions) > 5:  # Need enough decisions
                    # Calculate portfolio concentration at different stages
                    for stage_name, (start, end) in self.stage_boundaries.items():
                        stage_decisions = agent_decisions[
                            (agent_decisions['round'] >= start) & 
                            (agent_decisions['round'] < end)
                        ]
                        
                        if len(stage_decisions) > 0 and 'opportunity_sector' in stage_decisions.columns:
                            sector_counts = stage_decisions['opportunity_sector'].value_counts()
                            
                            # Calculate entropy as measure of diversity
                            probs = sector_counts.values / sector_counts.sum()
                            diversity = entropy(probs) if len(probs) > 1 else 0
                            
                            # Get primary AI level
                            primary_ai = stage_decisions['ai_level_used'].mode()[0] if len(stage_decisions) > 0 else 'none'
                            
                            path_dependency_data.append({
                                'agent_id': agent_id,
                                'stage': stage_name,
                                'diversity': diversity,
                                'n_sectors': len(sector_counts),
                                'primary_ai': primary_ai
                            })
        
        if path_dependency_data:
            path_df = pd.DataFrame(path_dependency_data)
            self.paradox_metrics['path_dependency'] = path_df
            
            # Calculate diversity reduction over time
            diversity_by_stage = path_df.groupby(['stage', 'primary_ai'])['diversity'].mean()
            print("\n   Portfolio Diversity by Stage and AI Level:")
            for (stage, ai_level), diversity in diversity_by_stage.items():
                print(f"      {stage} - {ai_level}: {diversity:.3f}")
    
    def analyze_novelty_saturation(self):
        """Analyze how agentic novelty ebbs as combinations get reused."""
        print("\n📊 Analyzing Agentic Novelty Saturation...")
        
        if self.framework.innovation_df.empty:
            print("   No innovation data available")
            return
        
        innovations = self.framework.innovation_df
        
        # Track innovation success rate over time
        saturation_data = []
        
        for stage_name, (start, end) in self.stage_boundaries.items():
            stage_innovations = innovations[
                (innovations['round'] >= start) & 
                (innovations['round'] < end)
            ]
            
            if len(stage_innovations) > 0:
                # Group by innovation characteristics
                if 'type' in stage_innovations.columns:
                    type_counts = stage_innovations['type'].value_counts()
                    
                    # Calculate novelty metrics
                    avg_novelty = stage_innovations['novelty'].mean() if 'novelty' in stage_innovations.columns else 0
                    avg_quality = stage_innovations['quality'].mean() if 'quality' in stage_innovations.columns else 0
                    
                    saturation_data.append({
                        'stage': stage_name,
                        'n_innovations': len(stage_innovations),
                        'avg_novelty': avg_novelty,
                        'avg_quality': avg_quality,
                        'type_diversity': entropy(type_counts.values / type_counts.sum()) if len(type_counts) > 1 else 0
                    })
        
        if saturation_data:
            saturation_df = pd.DataFrame(saturation_data)
            self.paradox_metrics['innovation_saturation'] = saturation_df
            
            print("\n   Innovation Metrics by Stage:")
            for _, row in saturation_df.iterrows():
                print(f"      {row['stage']}: Novelty={row['avg_novelty']:.3f}, "
                      f"Quality={row['avg_quality']:.3f}, Count={row['n_innovations']}")
    
    def analyze_herding_dynamics(self):
        """Analyze herding behavior and its consequences."""
        print("\n📊 Analyzing Herding Dynamics...")
        
        herding_data = []
        
        # Calculate herding metrics by round
        investments = self.decisions[self.decisions['action'] == 'invest'].dropna(subset=['opportunity_id'])

        # Calculate herding metrics by run and round to avoid mixing opportunity universes
        for (run_id, round_num), round_decisions in investments.groupby(['run_id', 'round']):
            if len(round_decisions) <= 5:
                continue

            opp_counts = round_decisions['opportunity_id'].value_counts()
            total = len(round_decisions)
            overall_hhi = sum((count/total)**2 for count in opp_counts.values)

            for ai_level in ['none', 'basic', 'advanced', 'premium']:
                ai_decisions = round_decisions[round_decisions['ai_level_used'] == ai_level]
                if len(ai_decisions) <= 2:
                    continue

                ai_opp_counts = ai_decisions['opportunity_id'].value_counts()
                ai_total = len(ai_decisions)
                ai_hhi = sum((count/ai_total)**2 for count in ai_opp_counts.values)

                herding_data.append({
                    'run_id': run_id,
                    'round': round_num,
                    'ai_level': ai_level,
                    'hhi': ai_hhi,
                    'relative_herding': ai_hhi / overall_hhi if overall_hhi > 0 else 1,
                    'n_agents': ai_total
                })
        
        if herding_data:
            herding_df = pd.DataFrame(herding_data)
            self.paradox_metrics['herding'] = herding_df
            
            # Calculate herding progression
            for stage_name, (start, end) in self.stage_boundaries.items():
                stage_herding = herding_df[
                    (herding_df['round'] >= start) & 
                    (herding_df['round'] < end)
                ]
                
                avg_herding = stage_herding.groupby('ai_level')['relative_herding'].mean()
                print(f"\n   {stage_name.capitalize()} Stage Relative Herding:")
                for level, herding in avg_herding.items():
                    print(f"      {level}: {herding:.3f}")
    
    def run_statistical_tests(self):
        """Run statistical tests for the information paradox."""
        print("\n📊 Running Statistical Tests...")
        
        tests = {}
        
        # Test 1: Performance reversal significance
        if 'temporal_reversal' in self.paradox_metrics:
            df = self.paradox_metrics['temporal_reversal']
            
            # Compare AI vs non-AI performance in early vs late stages
            early_ai = df[(df['stage'] == 'early') & (df['ai_level_used'] != 'none')]['mean'].dropna()
            early_no_ai = df[(df['stage'] == 'early') & (df['ai_level_used'] == 'none')]['mean'].dropna()
            late_ai = df[(df['stage'] == 'late') & (df['ai_level_used'] != 'none')]['mean'].dropna()
            late_no_ai = df[(df['stage'] == 'late') & (df['ai_level_used'] == 'none')]['mean'].dropna()
            
            if len(early_ai) > 0 and len(early_no_ai) > 0:
                early_stat, early_p = mannwhitneyu(early_ai, early_no_ai, alternative='greater')
                tests['early_ai_advantage'] = {'statistic': early_stat, 'p_value': early_p}
                print(f"   Early AI Advantage: p={early_p:.4f}")
            
            if len(late_ai) > 0 and len(late_no_ai) > 0:
                late_stat, late_p = mannwhitneyu(late_ai, late_no_ai, alternative='greater')
                tests['late_ai_advantage'] = {'statistic': late_stat, 'p_value': late_p}
                print(f"   Late AI Advantage: p={late_p:.4f}")
        
        # Test 2: Herding differences
        if 'herding' in self.paradox_metrics:
            df = self.paradox_metrics['herding']
            
            # Test if AI users herd more than non-AI users
            ai_herding = df[df['ai_level'] != 'none']['hhi'].dropna()
            no_ai_herding = df[df['ai_level'] == 'none']['hhi'].dropna()
            
            if len(ai_herding) > 0 and len(no_ai_herding) > 0:
                herd_stat, herd_p = mannwhitneyu(ai_herding, no_ai_herding, alternative='greater')
                tests['ai_herding'] = {'statistic': herd_stat, 'p_value': herd_p}
                print(f"   AI Herding Effect: p={herd_p:.4f}")
        
        self.paradox_metrics['statistical_tests'] = tests
    
    def create_paradox_dashboard(self):
        """Create comprehensive visualization dashboard."""
        print("\n📊 Creating Information Paradox Dashboard...")

        fig = plt.figure(figsize=(20, 16))
        gs = plt.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Temporal Performance Reversal
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_temporal_performance(ax1)

        # 2. Herding Evolution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_herding_evolution(ax2)

        # 3. Competition Dynamics
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_competition_dynamics(ax3)

        # 4. Trust-Performance Gap
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_trust_performance_gap(ax4)

        # 5. Portfolio Diversity Loss
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_diversity_loss(ax5)

        # 6. Innovation Saturation
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_innovation_saturation(ax6)

        # 7. Non-AI as a Solution (NEW PLOT)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_none_as_solution(ax7)

        # 8. Paradox Summary Metrics
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_paradox_summary(ax8)

        fig.suptitle('Information Paradox in AI-Augmented Decision Making',
                     fontsize=20, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.show()

    
    def _plot_temporal_performance(self, ax):
        """Plot performance over time by AI level."""
        if 'temporal_reversal' not in self.paradox_metrics:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center')
            return
        
        df = self.paradox_metrics['temporal_reversal']
        
        # Pivot for plotting
        pivot_df = df.pivot(index='stage', columns='ai_level_used', values='mean')
        
        x = np.arange(len(pivot_df.index))
        width = 0.2
        
        colors = {'none': '#95a5a6', 'basic': '#3498db', 
                 'advanced': '#f39c12', 'premium': '#e74c3c'}
        
        for i, ai_level in enumerate(pivot_df.columns):
            if ai_level in colors:
                ax.bar(x + i*width, pivot_df[ai_level], width, 
                      label=ai_level, color=colors[ai_level], alpha=0.8)
        
        ax.set_xlabel('Stage')
        ax.set_ylabel('Success Rate')
        ax.set_title('Performance Reversal: AI Advantage Diminishes Over Time')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(['Early', 'Middle', 'Late'])
        ax.legend(title='AI Level')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_herding_evolution(self, ax):
        """Plot herding behavior evolution."""
        if 'herding' not in self.paradox_metrics:
            ax.text(0.5, 0.5, 'No herding data', ha='center', va='center')
            return
        
        df = self.paradox_metrics['herding']
        
        colors = {'none': '#95a5a6', 'basic': '#3498db', 
                 'advanced': '#f39c12', 'premium': '#e74c3c'}
        
        for ai_level in ['none', 'basic', 'advanced', 'premium']:
            level_data = df[df['ai_level'] == ai_level]
            if not level_data.empty:
                # Smooth using rolling average
                level_data = level_data.sort_values('round')
                smoothed = level_data.set_index('round')['hhi'].rolling(
                    window=10, min_periods=1).mean()
                ax.plot(smoothed.index, smoothed.values, 
                       label=ai_level, color=colors[ai_level], linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Herding Index (HHI)')
        ax.set_title('Herding Intensifies with AI Use')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_competition_dynamics(self, ax):
        """Plot competition effects from information symmetry."""
        if 'concentration' not in self.paradox_metrics:
            ax.text(0.5, 0.5, 'No concentration data', ha='center', va='center')
            return
        
        df = self.paradox_metrics['concentration']
        
        # Calculate average unique opportunities accessed
        avg_unique = df.groupby(['round', 'ai_level'])['n_unique_opps'].mean().reset_index()
        
        for ai_level in ['none', 'basic', 'advanced', 'premium']:
            level_data = avg_unique[avg_unique['ai_level'] == ai_level]
            if not level_data.empty:
                ax.scatter(level_data['round'], level_data['n_unique_opps'], 
                          label=ai_level, alpha=0.6, s=20)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Number of Unique Opportunities')
        ax.set_title('Information Symmetry Reduces Opportunity Diversity')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_trust_performance_gap(self, ax):
        """Plot the gap between trust and actual performance."""
        if 'trust_decoupling' not in self.paradox_metrics:
            ax.text(0.5, 0.5, 'No trust data', ha='center', va='center')
            return
        
        df = self.paradox_metrics['trust_decoupling']
        
        ax.scatter(df['actual_performance'], df['final_trust'], 
                  alpha=0.5, c=df['trust_performance_gap'], 
                  cmap='coolwarm', s=50)
        
        # Add diagonal line for perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')
        
        ax.set_xlabel('Actual AI Performance')
        ax.set_ylabel('Final AI Trust')
        ax.set_title('Trust-Performance Decoupling')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Trust - Performance Gap')
    
    def _plot_diversity_loss(self, ax):
        """Plot portfolio diversity reduction."""
        if 'path_dependency' not in self.paradox_metrics:
            ax.text(0.5, 0.5, 'No diversity data', ha='center', va='center')
            return
        
        df = self.paradox_metrics['path_dependency']
        
        # Group by stage and AI level
        diversity_by_stage = df.groupby(['stage', 'primary_ai'])['diversity'].mean().reset_index()
        
        # Pivot for plotting
        pivot_df = diversity_by_stage.pivot(index='stage', columns='primary_ai', values='diversity')
        
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Stage')
        ax.set_ylabel('Portfolio Diversity (Entropy)')
        ax.set_title('Path Dependency: Diversity Loss Over Time')
        ax.set_xticklabels(['Early', 'Middle', 'Late'], rotation=0)
        ax.legend(title='Primary AI')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_innovation_saturation(self, ax):
        """Plot innovation space saturation."""
        if 'innovation_saturation' not in self.paradox_metrics:
            ax.text(0.5, 0.5, 'No innovation data', ha='center', va='center')
            return
        
        df = self.paradox_metrics['innovation_saturation']
        
        stages = ['early', 'middle', 'late']
        metrics = ['avg_novelty', 'avg_quality', 'type_diversity']
        
        x = np.arange(len(stages))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [df[df['stage'] == s][metric].values[0] if len(df[df['stage'] == s]) > 0 else 0 
                     for s in stages]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Stage')
        ax.set_ylabel('Metric Value')
        ax.set_title('Innovation Space Saturation')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Early', 'Middle', 'Late'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_information_value_decay(self, ax):
        """Plot the declining marginal value of information."""
        # Calculate information value proxy: success rate / information cost
        value_data = []
        
        for stage_name, (start, end) in self.stage_boundaries.items():
            stage_decisions = self.decisions[
                (self.decisions['round'] >= start) & 
                (self.decisions['round'] < end)
            ]
            
            for ai_level in ['none', 'basic', 'advanced', 'premium']:
                level_decisions = stage_decisions[stage_decisions['ai_level_used'] == ai_level]
                
                if len(level_decisions) > 0 and 'success' in level_decisions.columns:
                    success_rate = level_decisions['success'].mean()
                    
                    # Normalize by cost
                    ai_cost = self.config.AI_LEVELS[ai_level]['cost']
                    if ai_cost > 0:
                        value = success_rate / np.log(ai_cost + 1)  # Log scale for cost
                    else:
                        value = success_rate
                    
                    value_data.append({
                        'stage': stage_name,
                        'ai_level': ai_level,
                        'value': value
                    })
        
        if value_data:
            df = pd.DataFrame(value_data)
            pivot_df = df.pivot(index='stage', columns='ai_level', values='value')
            
            pivot_df.plot(kind='line', ax=ax, marker='o', linewidth=2)
            ax.set_xlabel('Stage')
            ax.set_ylabel('Information Value (Success/Cost)')
            ax.set_title('Declining Marginal Value of AI Information')
            ax.legend(title='AI Level')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
    
    def _plot_paradox_summary(self, ax):
        """Plot summary metrics of the information paradox."""
        ax.axis('off')
        
        summary_text = "INFORMATION PARADOX SUMMARY\n" + "="*50 + "\n\n"
        
        if 'reversal_score' in self.paradox_metrics:
            score = self.paradox_metrics['reversal_score']
            summary_text += f"Performance Reversal Score: {score:.3f}\n"
            summary_text += f"   {'✓ Paradox Present' if score > 0 else '✗ No Clear Paradox'}\n\n"
        
        if 'statistical_tests' in self.paradox_metrics:
            tests = self.paradox_metrics['statistical_tests']
            
            if 'early_ai_advantage' in tests:
                p_early = tests['early_ai_advantage']['p_value']
                summary_text += f"Early AI Advantage: p={p_early:.4f} "
                summary_text += f"{'(Significant)' if p_early < 0.05 else '(Not Significant)'}\n"
            
            if 'late_ai_advantage' in tests:
                p_late = tests['late_ai_advantage']['p_value']
                summary_text += f"Late AI Advantage: p={p_late:.4f} "
                summary_text += f"{'(Significant)' if p_late < 0.05 else '(Not Significant)'}\n"
            
            if 'ai_herding' in tests:
                p_herd = tests['ai_herding']['p_value']
                summary_text += f"AI Herding Effect: p={p_herd:.4f} "
                summary_text += f"{'(Significant)' if p_herd < 0.05 else '(Not Significant)'}\n"
        
        # Add new content for non-AI agent advantages
        if 'none_advantages' in self.paradox_metrics:
            advantages = self.paradox_metrics['none_advantages']
            summary_text += "\n" + "="*50 + "\n"
            summary_text += "NON-AI AGENT ADVANTAGES:\n"
            for key, value in advantages.items():
                if value > 0:
                    summary_text += f"✓ {key.replace('_', ' ').title()}: +{value:.3f}\n"
        
        summary_text += "\n" + "="*50 + "\n"
        summary_text += "PARADOX SOLUTIONS VIA NON-AI STRATEGIES:\n"
        summary_text += "• Maintains strategic diversity\n"
        summary_text += "• Avoids information cascades\n"
        summary_text += "• Zero information costs\n"
        summary_text += "• Independent decision-making\n"
        summary_text += "• Exploits market inefficiencies created by AI herding\n"
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', 
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


__all__ = [
    "ComprehensiveAnalysisFramework",
    "ComprehensiveVisualizationSuite",
    "StatisticalAnalysisSuite",
    "InformationParadoxAnalyzer",
    "ANALYSIS_VERSION",
]
