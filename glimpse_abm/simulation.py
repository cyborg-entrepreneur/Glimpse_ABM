"""Simulation engine for Glimpse ABM."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from .config import EmergentConfig
from .agents import EmergentAgent, DEBUG_PORTFOLIO_LOG, DEBUG_DECISION_LOG
from .information import EnhancedInformationSystem
from .innovation import CombinationTracker, InnovationEngine
from .knowledge import KnowledgeBase
from .market import MarketEnvironment
from .uncertainty import KnightianUncertaintyEnvironment
from .utils import canonical_to_display, normalize_ai_label, safe_mean

class VectorizedAgentState:
    """Manages agent states in NumPy arrays for efficient computation"""
    
    def __init__(self, n_agents: int, trait_names: List[str], config: EmergentConfig):
        self.n_agents = n_agents
        self.trait_names = trait_names
        self.trait_indices = {name: i for i, name in enumerate(trait_names)}
        self.config = config  # Store config reference
        float_dtype = np.float32 if getattr(config, 'use_float32', False) else np.float64

        # Core state arrays
        self.alive = np.ones(n_agents, dtype=bool)
        low, high = getattr(config, 'INITIAL_CAPITAL_RANGE', (config.INITIAL_CAPITAL, config.INITIAL_CAPITAL))
        self.initial_capital = np.random.uniform(low, high, size=n_agents).astype(float_dtype, copy=False)
        self.capital = self.initial_capital.copy()
        self.traits = np.zeros((n_agents, len(trait_names)), dtype=float_dtype)
        self.ai_level = np.zeros(n_agents, dtype=np.int8)  # 0=none, 1=basic, 2=advanced, 3=premium

        # Performance tracking
        self.capital_growth = np.ones(n_agents, dtype=float_dtype)
        self.innovation_count = np.zeros(n_agents, dtype=np.int32)
        self.portfolio_size = np.zeros(n_agents, dtype=np.int32)
        self.roe = np.zeros(n_agents, dtype=float_dtype)
    
    def batch_update_capital(self, agent_ids: np.ndarray, amounts: np.ndarray):
        """Vectorized capital update"""
        self.capital[agent_ids] += amounts
        self.alive[self.capital < self.config.SURVIVAL_THRESHOLD] = False
    
    def get_alive_indices(self) -> np.ndarray:
        return np.where(self.alive)[0]
    
    def get_trait_vector(self, trait_name: str) -> np.ndarray:
        idx = self.trait_indices[trait_name]
        return self.traits[:, idx]

class EmergentSimulation:
    """
    Orchestrates the entire simulation, including setup, execution, and data collection.
    """

    def __init__(self, config: EmergentConfig, output_dir: str, run_id: Union[int, str]):
        # --- Core Simulation Setup ---
        self.config = config
        self.run_id = run_id
        self.debug_enabled = bool(getattr(self.config, 'ENABLE_DEBUG_LOGS', False))
        base_seed = getattr(self.config, 'RANDOM_SEED', None)
        if base_seed is not None:
            run_hash = int(hashlib.sha256(str(run_id).encode('utf-8')).hexdigest(), 16) % 1_000_000
            seed_value = (base_seed + run_hash) % (2**32 - 1)
        else:
            seed_value = random.SystemRandom().randint(0, 2**32 - 2)
        self._run_seed = seed_value
        np.random.seed(seed_value)
        random.seed(seed_value)

        # --- MOVED LOCK DEFINITIONS TO THE TOP ---
        import threading
        self._market_lock = threading.Lock()
        self._info_system_lock = threading.Lock()
        self._data_buffer_lock = threading.Lock()

        # 1. Instantiate components with NO OTHER CUSTOM CLASS dependencies.
        self.knowledge_base = KnowledgeBase(config=self.config)
        self.uncertainty_env = KnightianUncertaintyEnvironment(self.config, knowledge_base=self.knowledge_base)

        # 2. Instantiate the MarketEnvironment.
        self.market = MarketEnvironment(self.config, self.uncertainty_env, innovation_engine=None, knowledge_base=self.knowledge_base)

        # 3. Instantiate components that depend on the market.
        self.info_system = EnhancedInformationSystem(self.config, market_ref=self.market)
        # --- THIS CALL IS NOW SAFE ---
        self.info_system.set_lock(self._info_system_lock)
    
        self.combination_tracker = CombinationTracker(market_ref=self.market)

        # 4. Create the InnovationEngine.
        self.innovation_engine = InnovationEngine(self.config, self.knowledge_base, self.combination_tracker)
        self._last_uncertainty_state = None

        trait_names = list(config.TRAIT_DISTRIBUTIONS.keys())
        self.agent_state = VectorizedAgentState(config.N_AGENTS, trait_names, config)
    
        # 5. Inject the fully-built components back into the MarketEnvironment.
        self.market.innovation_engine = self.innovation_engine
        self.market.combination_tracker = self.combination_tracker

        # --- Agent and Network Initialization ---
        self.agents = self._initialize_enhanced_agents()
        self.agent_network = self._initialize_agent_network(seed=self._run_seed)

        for i, agent in enumerate(self.agents):
            for trait_name, value in agent.traits.items():
                if trait_name in self.agent_state.trait_indices:
                    idx = self.agent_state.trait_indices[trait_name]
                    self.agent_state.traits[i, idx] = value
            self.agent_state.capital[i] = agent.resources.capital
            if hasattr(agent.resources.performance, 'initial_equity'):
                self.agent_state.initial_capital[i] = agent.resources.performance.initial_equity
    
        # --- Data Collection and Final Linking ---
        self.output_dir = output_dir
        self.data_paths = self._setup_data_directories(output_dir, run_id)
        self._link_agent_learning_profiles()

        self.data_buffer = {
            'decisions': [],
            'market': [],
            'uncertainty': [],
            'innovations': [],
            'knowledge': [],
            'matured': [],
            'uncertainty_details': [],
            'summary': []
        }
        flush_val = getattr(config, 'buffer_flush_interval', 10)
        self.buffer_flush_interval = max(1, int(flush_val))
        self.write_intermediate_batches = bool(getattr(config, 'write_intermediate_batches', True))
        self.round_log_interval = max(1, int(getattr(config, 'round_log_interval', 1)))
        self.enable_round_logging = bool(getattr(config, 'enable_round_logging', True))
        self._previous_uncertainty_levels = None
        self._baseline_uncertainty_levels = None  # Tracks round 1 levels for cumulative delta
        self.round_log_path = os.path.join(self.data_paths['base'], 'run_log.jsonl')
        if self.enable_round_logging:
            with open(self.round_log_path, 'w', encoding='utf-8') as _log_file:
                _log_file.write('')
        if self.debug_enabled:
            self.debug_portfolio_path = os.path.join(self.data_paths['base'], 'debug_portfolio.jsonl')
            self.debug_decision_path = os.path.join(self.data_paths['base'], 'debug_decision.jsonl')
        else:
            self.debug_portfolio_path = None
            self.debug_decision_path = None
        self._reset_debug_logs()
        self._pending_shift_events: List[Dict[str, Any]] = []
    
    def _enforce_survival_threshold(self, agent: EmergentAgent, round_num: int) -> None:
        if not agent.alive:
            return
        reason = agent._evaluate_failure_conditions(float(agent.resources.capital))
        if reason:
            agent.alive = False
            agent.failure_round = round_num  # Track when failure occurred for Kaplan-Meier
            agent.failure_reason = reason

    def _apply_operating_costs(self, agents: List[EmergentAgent], market_conditions: Dict[str, Any], round_num: int) -> None:
        avg_comp = float(market_conditions.get("avg_competition", 0.0) or 0.0)
        volatility = float(market_conditions.get("volatility", self.config.MARKET_VOLATILITY) or self.config.MARKET_VOLATILITY)
        base_vol = float(getattr(self.config, "MARKET_VOLATILITY", 0.25))
        severity = 1.0 + avg_comp * 0.35 + max(0.0, volatility - base_vol) * 0.45
        severity = float(np.clip(severity, 0.7, 1.9))
        for agent in agents:
            if not agent.alive:
                continue
            try:
                estimated_cost = float(agent._estimate_operational_costs(self.market))
            except Exception:
                estimated_cost = float(self.config.BASE_OPERATIONAL_COST)
            operating_cost = float(max(0.0, estimated_cost * severity))
            agent.operating_cost_estimate = operating_cost
            if operating_cost <= 0.0:
                continue
            agent.resources.capital -= operating_cost
            agent.resources.performance.record_deployment('opex', operating_cost, 'none', round_num)
            self._enforce_survival_threshold(agent, round_num)
    
    def _record_round_to_buffer(self, round_num: int, decisions: list, market_state: dict,
                                uncertainty_state: dict, innovations: list,
                                matured_outcomes: Optional[List[Dict]] = None,
                                alive_agents: Optional[List['EmergentAgent']] = None):
        """Buffer data instead of immediate disk write"""

        flattened_decisions: List[Dict] = []
        invest_confidences: List[float] = []
        total_capital_deployed = 0.0
        total_capital_returned = 0.0
        total_capital_deployed_invest = 0.0
        total_capital_deployed_innovate = 0.0
        total_capital_deployed_explore = 0.0
        total_capital_returned_invest = 0.0
        total_capital_returned_innovate = 0.0
        total_capital_returned_explore = 0.0

        matured_outcomes = matured_outcomes or []
        matured_records: List[Dict[str, Any]] = []
        uncertainty_records: List[Dict[str, Any]] = []

        for action in decisions:
            success_value = action.get('success') if 'success' in action else None
            clean_action = {
                'run_id': self.run_id,
                'round': round_num,
                'action': action.get('action'),
                'agent_id': action.get('agent_id'),
                'success': success_value,
                'ai_level_used': action.get('ai_level_used', 'none'),
                'portfolio_size': action.get('portfolio_size', 0)
            }
            if 'executed' in action:
                clean_action['executed'] = bool(action.get('executed'))
            if 'outcome_pending' in action:
                clean_action['outcome_pending'] = bool(action.get('outcome_pending'))

            for field in ['investment_amount', 'rd_investment', 'cost', 'capital_deployed', 'capital_returned']:
                if field in action:
                    clean_action[field] = action[field]

            if 'chosen_opportunity_obj' in action and action['chosen_opportunity_obj']:
                opp = action['chosen_opportunity_obj']
                clean_action['opportunity_id'] = getattr(opp, 'id', None)
                clean_action['opportunity_sector'] = getattr(opp, 'sector', None)

            if 'perception_at_decision' in action and action['perception_at_decision']:
                perception = action['perception_at_decision']

                if 'actor_ignorance' in perception:
                    clean_action['perc_actor_ignorance_level'] = perception['actor_ignorance'].get('ignorance_level', 0)
                    clean_action['perc_actor_ignorance_info_sufficiency'] = perception['actor_ignorance'].get('info_sufficiency', 0)

                if 'practical_indeterminism' in perception:
                    clean_action['perc_practical_indeterminism_level'] = perception['practical_indeterminism'].get('indeterminism_level', 0)
                    clean_action['perc_practical_indeterminism_regime_stability'] = perception['practical_indeterminism'].get('regime_stability', 0)

                if 'agentic_novelty' in perception:
                    clean_action['perc_agentic_novelty_potential'] = perception['agentic_novelty'].get('novelty_potential', 0)
                    clean_action['perc_agentic_novelty_creative_confidence'] = perception['agentic_novelty'].get('creative_confidence', 0)
                    clean_action['perc_agentic_component_scarcity'] = perception['agentic_novelty'].get('component_scarcity', 0)
                    disruption_map = perception['agentic_novelty'].get('disruption_potential', {})
                    if isinstance(disruption_map, dict) and disruption_map:
                        clean_action['perc_agentic_disruption_avg'] = float(
                            np.clip(safe_mean(list(disruption_map.values())), 0.0, 1.0)
                        )
                    clean_action['perc_agentic_combo_rate'] = perception['agentic_novelty'].get('combo_rate')
                    clean_action['perc_agentic_reuse_pressure'] = perception['agentic_novelty'].get('reuse_pressure')
                    clean_action['perc_agentic_adoption_rate'] = perception['agentic_novelty'].get('adoption_rate')
                    clean_action['perc_agentic_new_possibility_rate'] = perception['agentic_novelty'].get('new_possibility_rate')

                if 'competitive_recursion' in perception:
                    clean_action['perc_competitive_recursion_level'] = perception['competitive_recursion'].get('recursion_level', 0)
                clean_action['perc_competitive_recursion_herding_awareness'] = perception['competitive_recursion'].get('herding_awareness', 0)

                clean_action['perc_decision_confidence'] = perception.get('decision_confidence', 0.5)
            if action.get('decision_confidence') is not None:
                clean_action['decision_confidence'] = action.get('decision_confidence')
            clean_action['paradox_signal'] = action.get('paradox_signal')

            for field in [
                'ai_estimated_return',
                'ai_estimated_uncertainty',
                'ai_confidence',
                'ai_contains_hallucination',
                'ai_analysis_domain',
                'ai_actual_accuracy',
                'ai_overconfidence_factor',
                'ai_cost_incurred',
                'subscription_cost',
                'cash_multiple',
                'capital_before_action',
                'capital_after_action',
                'counterfactual_roi',
                'paradox_gap',
            ]:
                if field in action:
                    clean_action[field] = action[field]
            if 'ai_per_use_cost' in action:
                clean_action['ai_per_use_cost'] = action.get('ai_per_use_cost', 0.0)
            if 'ai_call_count' in action:
                clean_action['ai_call_count'] = action.get('ai_call_count')
            if 'ai_cost_incurred' in action:
                clean_action['ai_cost_incurred'] = action.get('ai_cost_incurred', 0.0)

            capital_deployed = float(action.get('amount') or action.get('capital_deployed', 0.0) or 0.0)
            if math.isnan(capital_deployed) or math.isinf(capital_deployed):
                capital_deployed = 0.0
            capital_returned = float(action.get('capital_returned', 0.0) or 0.0)
            if math.isnan(capital_returned) or math.isinf(capital_returned):
                capital_returned = 0.0
            total_capital_deployed += capital_deployed
            total_capital_returned += capital_returned
            action_type = action.get('action')
            if action_type == 'invest':
                total_capital_deployed_invest += capital_deployed
                total_capital_returned_invest += capital_returned
                conf_value = clean_action.get('perc_decision_confidence')
                if conf_value is not None:
                    try:
                        invest_confidences.append(float(conf_value))
                    except Exception:
                        pass
            elif action_type == 'innovate':
                total_capital_deployed_innovate += capital_deployed
                total_capital_returned_innovate += capital_returned
            elif action_type == 'explore':
                total_capital_deployed_explore += capital_deployed
                total_capital_returned_explore += capital_returned

            flattened_decisions.append(clean_action)
            self.data_buffer['decisions'].append(clean_action)

            agent_idx = action.get('agent_id')
            agent_ref = self.agents[agent_idx] if agent_idx is not None and 0 <= agent_idx < len(self.agents) else None
            if agent_ref is not None:
                uncertainty_records.append({
                    'run_id': self.run_id,
                    'round': round_num,
                    'agent_id': agent_idx,
                    'action': action.get('action'),
                    'ai_level_used': clean_action.get('ai_level_used', 'none'),
                    'ai_switch_count': getattr(agent_ref, 'ai_switches', 0),
                    'ai_trust': agent_ref.traits.get('ai_trust', None),
                    'strategy_mode': getattr(agent_ref, 'strategy_mode', None),
                    'decision_confidence': clean_action.get('perc_decision_confidence'),
                    'actor_ignorance_level': clean_action.get('perc_actor_ignorance_level'),
                    'actor_ignorance_info_sufficiency': clean_action.get('perc_actor_ignorance_info_sufficiency'),
                    'practical_indeterminism_level': clean_action.get('perc_practical_indeterminism_level'),
                    'practical_indeterminism_regime_stability': clean_action.get('perc_practical_indeterminism_regime_stability'),
                    'agentic_novelty_potential': clean_action.get('perc_agentic_novelty_potential'),
                    'agentic_novelty_creative_confidence': clean_action.get('perc_agentic_novelty_creative_confidence'),
                    'agentic_combo_rate': clean_action.get('perc_agentic_combo_rate'),
                    'agentic_reuse_pressure': clean_action.get('perc_agentic_reuse_pressure'),
                    'agentic_adoption_rate': clean_action.get('perc_agentic_adoption_rate'),
                    'agentic_new_possibility_rate': clean_action.get('perc_agentic_new_possibility_rate'),
                    'competitive_recursion_level': clean_action.get('perc_competitive_recursion_level'),
                    'competitive_recursion_herding_awareness': clean_action.get('perc_competitive_recursion_herding_awareness'),
                    'ai_estimated_return': clean_action.get('ai_estimated_return'),
                    'ai_estimated_uncertainty': clean_action.get('ai_estimated_uncertainty'),
                    'ai_confidence': clean_action.get('ai_confidence'),
                    'ai_contains_hallucination': clean_action.get('ai_contains_hallucination'),
                    'ai_analysis_domain': clean_action.get('ai_analysis_domain'),
                    'ai_actual_accuracy': clean_action.get('ai_actual_accuracy'),
                    'ai_overconfidence_factor': clean_action.get('ai_overconfidence_factor'),
                    'paradox_signal': getattr(agent_ref, 'paradox_signal', None),
                    'paradox_gap': getattr(agent_ref, 'last_paradox_gap', None),
                    'paradox_cf_gap': getattr(agent_ref, 'last_paradox_cf_gap', None),
                })

        if matured_outcomes:
            for outcome in matured_outcomes:
                investment = outcome.get('investment') or {}
                opportunity_obj = investment.get('opportunity')
                ai_info = outcome.get('ai_info_at_investment')
                investment_amount = float(outcome.get('investment_amount', 0.0) or 0.0)
                capital_returned = float(outcome.get('capital_returned', 0.0) or 0.0)
                net_return = float(outcome.get('net_return', capital_returned - investment_amount) or 0.0)
                record = {
                    'run_id': self.run_id,
                    'round': round_num,
                    'agent_id': outcome.get('agent_id'),
                    'opportunity_id': outcome.get('opportunity_id') or getattr(opportunity_obj, 'id', None),
                    'sector': getattr(opportunity_obj, 'sector', investment.get('sector')),
                    'entry_round': investment.get('entry_round'),
                    'maturation_round': investment.get('maturation_round'),
                    'time_to_maturity': investment.get('time_to_maturity'),
                    'investment_amount': investment_amount,
                    'capital_returned': capital_returned,
                    'net_return': net_return,
                    'success': bool(outcome.get('success', False)),
                    'raw_success': bool(investment.get('raw_success', outcome.get('success', False))),
                    'ai_level_used': outcome.get('ai_level_used', 'none'),
                    'ai_estimated_return': getattr(ai_info, 'estimated_return', None),
                    'ai_estimated_uncertainty': getattr(ai_info, 'estimated_uncertainty', None),
                    'ai_confidence': getattr(ai_info, 'confidence', None),
                    'ai_actual_accuracy': getattr(ai_info, 'actual_accuracy', None),
                    'ai_contains_hallucination': getattr(ai_info, 'contains_hallucination', None),
                    'ai_analysis_domain': getattr(ai_info, 'domain', None),
                    'ai_overconfidence_factor': getattr(ai_info, 'overconfidence_factor', None),
                    'defaulted': bool(investment.get('defaulted', False)),
                    'realized_multiplier': investment.get('realized_multiplier'),
                    'realized_roi': investment.get('realized_roi'),
                    'decision_confidence': investment.get('decision_confidence'),
                    'counterfactual_roi': outcome.get('counterfactual_roi'),
                    'counterfactual_capital_returned': outcome.get('counterfactual_capital_returned'),
                    'paradox_signal_post': outcome.get('paradox_signal_post'),
                }
                matured_records.append(record)

        if matured_records:
            self.data_buffer['matured'].extend(matured_records)
        if uncertainty_records:
            self.data_buffer['uncertainty_details'].extend(uncertainty_records)

        knowledge_records = []
        agents_for_logging = alive_agents
        if agents_for_logging is None:
            agents_for_logging = [agent for agent in self.agents if agent.alive]
        if agents_for_logging:
            for agent in agents_for_logging:
                knowledge_ids = self.knowledge_base.agent_knowledge.get(agent.id, set())
                sector_knowledge = getattr(agent.resources, 'knowledge', {}) if hasattr(agent, 'resources') else {}
                knowledge_record = {
                    'run_id': self.run_id,
                    'round': round_num,
                    'agent_id': agent.id,
                    'knowledge_count': len(knowledge_ids),
                    'avg_sector_knowledge': safe_mean(list(sector_knowledge.values())) if sector_knowledge else None
                }
                for sector, value in sector_knowledge.items():
                    knowledge_record[f'sector_{sector}'] = value
                knowledge_records.append(knowledge_record)
        if knowledge_records:
            self.data_buffer['knowledge'].extend(knowledge_records)
            avg_knowledge_count = safe_mean([rec['knowledge_count'] for rec in knowledge_records])
            avg_sector_knowledge_value = safe_mean([
                rec['avg_sector_knowledge'] for rec in knowledge_records
                if rec['avg_sector_knowledge'] is not None
            ]) if knowledge_records else 0.0
        else:
            avg_knowledge_count = 0.0
            avg_sector_knowledge_value = None

        matured_return_total = 0.0
        for outcome in matured_outcomes:
            value = float(outcome.get('capital_returned', 0.0) or 0.0)
            if math.isnan(value) or math.isinf(value):
                value = 0.0
            matured_return_total += value
        total_capital_returned += matured_return_total
        total_capital_returned_invest += matured_return_total

        if market_state:
            self.data_buffer['market'].append({
                'run_id': self.run_id,
                'round': round_num,
                **market_state
            })

        if uncertainty_state:
            # Extract current levels
            actor_level = float(uncertainty_state.get('actor_ignorance', {}).get('level', 0.0))
            practical_level = float(uncertainty_state.get('practical_indeterminism', {}).get('level', 0.0))
            agentic_level = float(uncertainty_state.get('agentic_novelty', {}).get('level', 0.0))
            competitive_level = float(uncertainty_state.get('competitive_recursion', {}).get('level', 0.0))

            # Store baseline on first round with uncertainty data
            if self._baseline_uncertainty_levels is None:
                self._baseline_uncertainty_levels = {
                    'actor_ignorance': actor_level,
                    'practical_indeterminism': practical_level,
                    'agentic_novelty': agentic_level,
                    'competitive_recursion': competitive_level,
                }

            # Get previous levels (default to current if first round)
            prev = self._previous_uncertainty_levels or {}
            prev_actor = prev.get('actor_ignorance_level', actor_level)
            prev_practical = prev.get('practical_indeterminism_level', practical_level)
            prev_agentic = prev.get('agentic_novelty_level', agentic_level)
            prev_competitive = prev.get('competitive_recursion_level', competitive_level)

            # Get baseline levels
            base = self._baseline_uncertainty_levels
            base_actor = base.get('actor_ignorance', actor_level)
            base_practical = base.get('practical_indeterminism', practical_level)
            base_agentic = base.get('agentic_novelty', agentic_level)
            base_competitive = base.get('competitive_recursion', competitive_level)

            # Compute delta (round-over-round change)
            delta_actor = actor_level - prev_actor
            delta_practical = practical_level - prev_practical
            delta_agentic = agentic_level - prev_agentic
            delta_competitive = competitive_level - prev_competitive

            # Compute cumulative delta (change from baseline)
            cumulative_delta_actor = actor_level - base_actor
            cumulative_delta_practical = practical_level - base_practical
            cumulative_delta_agentic = agentic_level - base_agentic
            cumulative_delta_competitive = competitive_level - base_competitive

            # Compute portfolio composition (shares)
            uncertainty_total = actor_level + practical_level + agentic_level + competitive_level
            total_safe = max(uncertainty_total, 0.001)  # Avoid division by zero
            share_actor = actor_level / total_safe
            share_practical = practical_level / total_safe
            share_agentic = agentic_level / total_safe
            share_competitive = competitive_level / total_safe

            # Compute HHI (Herfindahl-Hirschman Index) - concentration measure
            uncertainty_hhi = share_actor**2 + share_practical**2 + share_agentic**2 + share_competitive**2

            # Compute entropy - diversity measure (avoid log(0))
            eps = 1e-10
            uncertainty_entropy = -(
                share_actor * math.log(share_actor + eps) +
                share_practical * math.log(share_practical + eps) +
                share_agentic * math.log(share_agentic + eps) +
                share_competitive * math.log(share_competitive + eps)
            )

            # Per-agent perception std calculations
            ignorance_vals = [
                action.get('perc_actor_ignorance_level')
                for action in flattened_decisions
                if action.get('perc_actor_ignorance_level') is not None
            ]
            indeterminism_vals = [
                action.get('perc_practical_indeterminism_level')
                for action in flattened_decisions
                if action.get('perc_practical_indeterminism_level') is not None
            ]
            novelty_vals = [
                action.get('perc_agentic_novelty_potential')
                for action in flattened_decisions
                if action.get('perc_agentic_novelty_potential') is not None
            ]
            recursion_vals = [
                action.get('perc_competitive_recursion_level')
                for action in flattened_decisions
                if action.get('perc_competitive_recursion_level') is not None
            ]
            ignorance_std = float(np.std(ignorance_vals)) if len(ignorance_vals) > 1 else 0.0
            indeterminism_std = float(np.std(indeterminism_vals)) if len(indeterminism_vals) > 1 else 0.0
            novelty_std = float(np.std(novelty_vals)) if len(novelty_vals) > 1 else 0.0
            recursion_std = float(np.std(recursion_vals)) if len(recursion_vals) > 1 else 0.0

            flat_uncertainty = {
                'run_id': self.run_id,
                'round': round_num,
                # Current levels
                'actor_ignorance_level': actor_level,
                'practical_indeterminism_level': practical_level,
                'agentic_novelty_level': agentic_level,
                'competitive_recursion_level': competitive_level,
                # Per-agent perception standard deviations
                'actor_ignorance_std': ignorance_std,
                'practical_indeterminism_std': indeterminism_std,
                'agentic_novelty_std': novelty_std,
                'competitive_recursion_std': recursion_std,
                # Round-over-round deltas (transformation rate)
                'delta_actor_ignorance': delta_actor,
                'delta_practical_indeterminism': delta_practical,
                'delta_agentic_novelty': delta_agentic,
                'delta_competitive_recursion': delta_competitive,
                # Cumulative change from baseline (total transformation)
                'cumulative_delta_actor': cumulative_delta_actor,
                'cumulative_delta_practical': cumulative_delta_practical,
                'cumulative_delta_agentic': cumulative_delta_agentic,
                'cumulative_delta_competitive': cumulative_delta_competitive,
                # Portfolio composition
                'uncertainty_total': uncertainty_total,
                'share_actor_ignorance': share_actor,
                'share_practical_indeterminism': share_practical,
                'share_agentic_novelty': share_agentic,
                'share_competitive_recursion': share_competitive,
                # Concentration and diversity measures
                'uncertainty_hhi': uncertainty_hhi,
                'uncertainty_entropy': uncertainty_entropy,
                # AI deltas (counterfactual effect)
                'ai_ignorance_delta': uncertainty_state.get('actor_ignorance', {}).get('ai_delta', 0.0),
                'ai_indeterminism_delta': uncertainty_state.get('practical_indeterminism', {}).get('ai_delta', 0.0),
                'ai_agentic_novelty_delta': uncertainty_state.get('agentic_novelty', {}).get('ai_delta', 0.0),
                'ai_recursion_delta': uncertainty_state.get('competitive_recursion', {}).get('ai_delta', 0.0),
            }

            # Add additional uncertainty state details
            component_scarcity = uncertainty_state.get('agentic_novelty', {}).get('component_scarcity')
            if component_scarcity is not None:
                flat_uncertainty['agentic_component_scarcity'] = component_scarcity
            combo_rate = uncertainty_state.get('agentic_novelty', {}).get('combo_rate')
            if combo_rate is not None:
                flat_uncertainty['agentic_combo_rate'] = combo_rate
            reuse_pressure = uncertainty_state.get('agentic_novelty', {}).get('reuse_pressure')
            if reuse_pressure is not None:
                flat_uncertainty['agentic_reuse_pressure'] = reuse_pressure
            adoption_rate = uncertainty_state.get('agentic_novelty', {}).get('adoption_rate')
            if adoption_rate is not None:
                flat_uncertainty['agentic_adoption_rate'] = adoption_rate
            new_poss_rate = uncertainty_state.get('agentic_novelty', {}).get('new_possibility_rate')
            if new_poss_rate is not None:
                flat_uncertainty['agentic_new_possibility_rate'] = new_poss_rate
            disruption_map = uncertainty_state.get('agentic_novelty', {}).get('disruption_potential', {})
            if isinstance(disruption_map, dict) and disruption_map:
                disruption_vals = list(disruption_map.values())
                flat_uncertainty['agentic_disruption_avg'] = float(
                    np.clip(safe_mean(disruption_vals), 0.0, 1.0)
                )
            tier_recursions = uncertainty_state.get('competitive_recursion', {}).get('tier_levels', {})
            if isinstance(tier_recursions, dict):
                for tier_name, tier_value in tier_recursions.items():
                    flat_uncertainty[f'competitive_recursion_{tier_name}'] = tier_value
            tier_actor = uncertainty_state.get('actor_ignorance', {}).get('tier_levels', {})
            if isinstance(tier_actor, dict):
                for tier_name, tier_value in tier_actor.items():
                    flat_uncertainty[f'actor_ignorance_{tier_name}'] = tier_value
            tier_practical = uncertainty_state.get('practical_indeterminism', {}).get('tier_levels', {})
            if isinstance(tier_practical, dict):
                for tier_name, tier_value in tier_practical.items():
                    flat_uncertainty[f'practical_indeterminism_{tier_name}'] = tier_value
            tier_novelty = uncertainty_state.get('agentic_novelty', {}).get('tier_levels', {})
            if isinstance(tier_novelty, dict):
                for tier_name, tier_value in tier_novelty.items():
                    flat_uncertainty[f'agentic_novelty_{tier_name}'] = tier_value

            self.data_buffer['uncertainty'].append(flat_uncertainty)

            # Update previous levels for next round
            self._previous_uncertainty_levels = {
                'actor_ignorance_level': actor_level,
                'practical_indeterminism_level': practical_level,
                'agentic_novelty_level': agentic_level,
                'competitive_recursion_level': competitive_level,
            }

        if innovations:
            for inn in innovations:
                if not inn:
                    continue
                self.data_buffer['innovations'].append({
                    'run_id': self.run_id,
                    'round': round_num,
                    'type': getattr(inn, 'type', None),
                    'quality': getattr(inn, 'quality', 0),
                    'novelty': getattr(inn, 'novelty', 0),
                    'creator_id': getattr(inn, 'creator_id', None),
                    'sector': getattr(inn, 'sector', None),
                    'combination_signature': getattr(inn, 'combination_signature', None),
                    'cash_multiple': getattr(inn, 'cash_multiple', None),
                })

        if flattened_decisions:
            total_actions = len(flattened_decisions)
            action_counts = collections.Counter(a.get('action') for a in flattened_decisions)
        else:
            total_actions = 0
            action_counts = collections.Counter()

        capital_snapshot = self.agent_state.capital
        if capital_snapshot.size:
            mean_capital = float(capital_snapshot.mean())
            median_capital = float(np.median(capital_snapshot))
            capital_std = float(capital_snapshot.std()) if capital_snapshot.size > 1 else 0.0
        else:
            mean_capital = median_capital = capital_std = 0.0

        alive_indices = self.agent_state.get_alive_indices()
        alive_agents = [self.agents[i] for i in alive_indices]
        if alive_agents:
            ai_counts = collections.Counter(getattr(agent, 'current_ai_level', 'none') for agent in alive_agents)
            total_alive = len(alive_agents)
            ai_shares = {level: ai_counts.get(level, 0) / max(1, total_alive) for level in ['none', 'basic', 'advanced', 'premium']}
            ai_trust_values = np.fromiter((agent.traits.get('ai_trust', 0.0) for agent in alive_agents), dtype=float)
            mean_trust = float(ai_trust_values.mean()) if ai_trust_values.size else 0.0
            std_trust = float(ai_trust_values.std()) if ai_trust_values.size > 1 else 0.0
            diversities = np.fromiter((agent.portfolio.diversification_score for agent in alive_agents if np.isfinite(agent.portfolio.diversification_score)), dtype=float)
            mean_diversity = float(diversities.mean()) if diversities.size else 0.0
            diversity_std = float(diversities.std()) if diversities.size > 1 else 0.0
            mean_roe = float(np.mean([agent.resources.performance.compute_roe(agent.resources.capital) for agent in alive_agents]))
            mean_roic_invest = float(np.mean([agent.resources.performance.compute_roic('invest') for agent in alive_agents]))
            mean_roic_innovate = float(np.mean([agent.resources.performance.compute_roic('innovate') for agent in alive_agents]))
            mean_roic_explore = float(np.mean([agent.resources.performance.compute_roic('explore') for agent in alive_agents]))
        else:
            mean_trust = mean_diversity = mean_roe = mean_roic_invest = mean_roic_innovate = mean_roic_explore = 0.0
            std_trust = diversity_std = 0.0
            ai_shares = {level: 0.0 for level in ['none', 'basic', 'advanced', 'premium']}

        innovation_attempts = len([a for a in flattened_decisions if a.get('action') == 'innovate'])
        innovation_successes = len([a for a in flattened_decisions if a.get('action') == 'innovate' and a.get('success')])
        innovation_success_rate = (innovation_successes / innovation_attempts) if innovation_attempts else 0.0

        action_shares = {key: action_counts.get(key, 0) / max(1, total_actions) for key in ['invest', 'innovate', 'explore', 'maintain']}

        investments = [a for a in flattened_decisions if a.get('action') == 'invest']
        if investments:
            opp_counts = collections.Counter(inv.get('opportunity_id') for inv in investments if inv.get('opportunity_id'))
            total_invests = len(investments)
            overall_hhi = float(sum((count / total_invests) ** 2 for count in opp_counts.values())) if opp_counts else 0.0
        else:
            overall_hhi = 0.0

        top_sector_share = 0.0
        sector_distribution = market_state.get('sector_distribution') if market_state else None
        if sector_distribution:
            total_sector = sum(sector_distribution.values())
            if total_sector > 0:
                top_sector_share = max(sector_distribution.values()) / total_sector

        mean_confidence_invest = float(safe_mean(invest_confidences)) if invest_confidences else 0.0

        alive_count = len(alive_agents)
        summary_entry = {
            'round': round_num,
            'mean_capital': mean_capital,
            'median_capital': median_capital,
            'capital_std': capital_std,
            'mean_ai_trust': mean_trust,
            'ai_trust_std': std_trust,
            'mean_portfolio_diversity': mean_diversity,
            'portfolio_diversity_std': diversity_std,
            'mean_roe': mean_roe,
            'mean_roic_invest': mean_roic_invest,
            'mean_roic_innovate': mean_roic_innovate,
            'mean_roic_explore': mean_roic_explore,
            'total_capital_deployed': total_capital_deployed,
            'total_capital_returned': total_capital_returned,
            'total_capital_deployed_invest': total_capital_deployed_invest,
            'total_capital_deployed_innovate': total_capital_deployed_innovate,
            'total_capital_deployed_explore': total_capital_deployed_explore,
            'total_capital_returned_invest': total_capital_returned_invest,
            'total_capital_returned_innovate': total_capital_returned_innovate,
            'total_capital_returned_explore': total_capital_returned_explore,
            'net_capital_flow_innovate': total_capital_returned_innovate - total_capital_deployed_innovate,
            'net_capital_flow_invest': total_capital_returned_invest - total_capital_deployed_invest,
            'ai_share_none': ai_shares['none'],
            'ai_share_basic': ai_shares['basic'],
            'ai_share_advanced': ai_shares['advanced'],
            'ai_share_premium': ai_shares['premium'],
            'action_share_invest': action_shares['invest'],
            'action_share_innovate': action_shares['innovate'],
            'action_share_explore': action_shares['explore'],
            'action_share_maintain': action_shares['maintain'],
            'innovation_attempts': innovation_attempts,
            'innovation_successes': innovation_successes,
            'innovation_success_rate': innovation_success_rate,
            'overall_hhi': overall_hhi,
            'top_sector_share': top_sector_share,
            'mean_agent_knowledge_count': avg_knowledge_count,
            'mean_sector_knowledge': avg_sector_knowledge_value,
            'mean_confidence_invest': mean_confidence_invest,
            'alive_agents': alive_count,
            'dead_agents': self.config.N_AGENTS - alive_count,
        }
        self.data_buffer['summary'].append(summary_entry)
        self._log_round_summary(summary_entry)

        should_flush = False
        if round_num == self.config.N_ROUNDS - 1:
            should_flush = True
        elif self.write_intermediate_batches and round_num % self.buffer_flush_interval == 0:
            should_flush = True
        if should_flush:
            self._flush_buffers(round_num)
    def _flush_buffers(self, round_num: int):
        """Write buffered data to disk"""

        if self.data_buffer.get('decisions', []):
            decisions_df = pd.DataFrame(self.data_buffer['decisions'])
            decisions_df['run_id'] = self.run_id
            decisions_df.to_pickle(os.path.join(self.data_paths['decisions'], f'batch_{round_num}.pkl'))
            self.data_buffer['decisions'] = []

        if self.data_buffer.get('market', []):
            market_df = pd.DataFrame(self.data_buffer['market'])
            market_df['run_id'] = self.run_id
            market_df.to_pickle(os.path.join(self.data_paths['market'], f'batch_{round_num}.pkl'))
            self.data_buffer['market'] = []

        if self.data_buffer.get('uncertainty', []):
            uncertainty_df = pd.DataFrame(self.data_buffer['uncertainty'])
            uncertainty_df['run_id'] = self.run_id
            uncertainty_df.to_pickle(os.path.join(self.data_paths['uncertainty'], f'batch_{round_num}.pkl'))
            self.data_buffer['uncertainty'] = []

        if self.data_buffer.get('innovations', []):
            innovations_df = pd.DataFrame(self.data_buffer['innovations'])
            innovations_df['run_id'] = self.run_id
            innovations_df.to_pickle(os.path.join(self.data_paths['innovations'], f'batch_{round_num}.pkl'))
            self.data_buffer['innovations'] = []

        if self.data_buffer.get('knowledge', []):
            knowledge_df = pd.DataFrame(self.data_buffer['knowledge'])
            knowledge_df['run_id'] = self.run_id
            knowledge_df.to_pickle(os.path.join(self.data_paths['knowledge'], f'batch_{round_num}.pkl'))
            self.data_buffer['knowledge'] = []

        if self.data_buffer.get('matured', []):
            matured_df = pd.DataFrame(self.data_buffer['matured'])
            matured_df['run_id'] = self.run_id
            matured_df.to_pickle(os.path.join(self.data_paths['matured'], f'batch_{round_num}.pkl'))
            self.data_buffer['matured'] = []

        if self.data_buffer.get('uncertainty_details', []):
            uncertainty_detail_df = pd.DataFrame(self.data_buffer['uncertainty_details'])
            uncertainty_detail_df['run_id'] = self.run_id
            uncertainty_detail_df.to_pickle(os.path.join(self.data_paths['uncertainty_details'], f'batch_{round_num}.pkl'))
            self.data_buffer['uncertainty_details'] = []

        if self.data_buffer.get('summary', []):
            summary_df = pd.DataFrame(self.data_buffer['summary'])
            summary_df['run_id'] = self.run_id
            summary_df.to_pickle(os.path.join(self.data_paths['summary'], f'batch_{round_num}.pkl'))
            self.data_buffer['summary'] = []

    def _log_round_summary(self, record: Dict[str, Any]) -> None:
        """Append a plain-text JSON record for quick diagnostics."""
        if not self.enable_round_logging:
            return
        round_num = int(record.get('round', 0))
        if round_num != self.config.N_ROUNDS - 1 and round_num % self.round_log_interval != 0:
            return
        enriched = dict(record)
        enriched.setdefault('run_id', self.run_id)
        for key, value in list(enriched.items()):
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    enriched[key] = 0.0
        with self._data_buffer_lock:
            with open(self.round_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(json.dumps(enriched, default=float) + '\n')

    def _reset_debug_logs(self) -> None:
        DEBUG_PORTFOLIO_LOG.clear()
        DEBUG_DECISION_LOG.clear()

    def _write_debug_logs(self) -> None:
        if not self.debug_enabled:
            DEBUG_PORTFOLIO_LOG.clear()
            DEBUG_DECISION_LOG.clear()
            return
        if DEBUG_PORTFOLIO_LOG:
            with open(self.debug_portfolio_path, 'w', encoding='utf-8') as log_file:
                for entry in DEBUG_PORTFOLIO_LOG:
                    log_file.write(json.dumps(entry, default=float) + '\n')
            DEBUG_PORTFOLIO_LOG.clear()
        if DEBUG_DECISION_LOG:
            with open(self.debug_decision_path, 'w', encoding='utf-8') as log_file:
                for entry in DEBUG_DECISION_LOG:
                    log_file.write(json.dumps(entry, default=float) + '\n')
            DEBUG_DECISION_LOG.clear()

    def _initialize_agent_network(self, seed: Optional[int] = None) -> Optional[nx.Graph]:
        """Creates the agent social network if enabled in the config."""
        if not self.config.USE_NETWORK_EFFECTS:
            return None
        # A Watts-Strogatz graph is a good model for social networks, balancing structure and randomness.
        g = nx.watts_strogatz_graph(
            n=self.config.N_AGENTS,
            k=self.config.NETWORK_N_NEIGHBORS,
            p=self.config.NETWORK_REWIRING_PROB,
            seed=seed if seed is not None else self.config.RANDOM_SEED
        )
        return g

    def _setup_data_directories(self, base_dir: str, run_id: str) -> Dict[str, str]:
        """Creates subdirectories for this specific simulation run to store output data."""
        run_dir = os.path.join(base_dir, run_id)
        paths = {
            "base": run_dir,
            "decisions": os.path.join(run_dir, "decisions"),
            "market": os.path.join(run_dir, "market"),
            "uncertainty": os.path.join(run_dir, "uncertainty"),
            "innovations": os.path.join(run_dir, "innovations"),
            "knowledge": os.path.join(run_dir, "knowledge"),
            "matured": os.path.join(run_dir, "matured"),
            "uncertainty_details": os.path.join(run_dir, "uncertainty_details"),
            "summary": os.path.join(run_dir, "summary"),
        }
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        return paths

    def _link_agent_learning_profiles(self):
        """Ensures each agent instance has a direct link to its learning profile for easy access."""
        for agent in self.agents:
            if agent.id in self.info_system.agent_learning_profiles:
                agent.ai_learning = self.info_system.agent_learning_profiles[agent.id]
            else:
                # This is a fallback, though it shouldn't be strictly necessary with the current __init__ logic
                self.info_system.initialize_agent_learning(agent.id)
                agent.ai_learning = self.info_system.agent_learning_profiles[agent.id]

    def _initialize_enhanced_agents(self):
        """
        Initializes agents with traits generated from the specified distributions
        in the configuration.
        """
        agents = []
        # Agent type distribution (for setting initial AI trust bias)
        dist_map = {'human': 0.55, 'basic_ai': 0.25, 'advanced_ai': 0.15, 'premium_ai': 0.05}
        agent_types = list(dist_map.keys())
        probabilities = list(dist_map.values())

        initial_capital_low, initial_capital_high = getattr(self.config, 'INITIAL_CAPITAL_RANGE', (self.config.INITIAL_CAPITAL, self.config.INITIAL_CAPITAL))
        initial_capitals = np.random.uniform(initial_capital_low, initial_capital_high, size=self.config.N_AGENTS)

        for i in range(self.config.N_AGENTS):
            traits = {}

            for name, dist_info in self.config.TRAIT_DISTRIBUTIONS.items():
                dist_type = dist_info.get('dist')
                params = dist_info.get('params', {})

                if dist_type == 'uniform':
                    traits[name] = np.random.uniform(
                        low=params.get('low', 0.0),
                        high=params.get('high', 1.0)
                    )
                elif dist_type == 'beta':
                    traits[name] = np.clip(
                        np.random.beta(
                            params.get('a', 1.0),
                            params.get('b', 1.0)
                        ), 0.0, 1.0)
                elif dist_type == 'lognormal':
                    value = np.random.lognormal(
                        mean=params.get('mean', 0.0),
                        sigma=params.get('sigma', 1.0)
                    )
                    traits[name] = np.clip(value, 0.0, 1.0)
                elif dist_type == 'normal':
                    value = np.random.normal(
                        loc=params.get('mean', 0.5),
                        scale=params.get('std', 0.2)
                    )
                    traits[name] = np.clip(value, 0.0, 1.0)
                elif dist_type == 'normal_clipped':
                    value = np.random.normal(
                        loc=params.get('mean', 0.5),
                        scale=params.get('std', 0.15)
                    )
                    traits[name] = np.clip(value, 0.0, 1.0)
                else:
                    traits[name] = np.random.uniform(0.0, 1.0)

            agent_type = np.random.choice(agent_types, p=probabilities)
            agent = EmergentAgent(
                agent_id=i,
                initial_traits=traits,
                config=self.config,
                knowledge_base=self.knowledge_base,
                innovation_engine=self.innovation_engine,
                agent_type=agent_type,
                initial_capital=initial_capitals[i]
            )
            self.knowledge_base.ensure_starter_knowledge(agent.id)
            self.info_system.initialize_agent_learning(agent.id)
            agent.ai_learning = self.info_system.agent_learning_profiles.get(agent.id)
            agents.append(agent)

    
        return agents

    def run(self):
        """Main simulation loop with the corrected save function call."""
        print(f"[{self.run_id}] Starting simulation...")
        for step in range(self.config.N_ROUNDS):
            self._step(step)

        # Ensure any remaining buffered data is persisted
        self._flush_buffers(self.config.N_ROUNDS - 1)
        self._save_final_agent_state()
        self._write_debug_logs()
        print(f"[{self.run_id}] Simulation finished.")

    def _step(self, round_num: int):
        """Main simulation step with corrected structure and all redundancies removed."""
        if round_num > 0 and round_num % self.buffer_flush_interval == 0:
            self._flush_buffers(round_num - 1)
        market_conditions = dict(self.market.get_market_conditions())
        if self._last_uncertainty_state is not None:
            market_conditions['uncertainty_state'] = self._last_uncertainty_state
        market_state: Optional[Dict[str, Any]] = None
        alive_indices = self.agent_state.get_alive_indices()
        alive_agents = [self.agents[i] for i in alive_indices]

        if not alive_agents:
            self._flush_buffers(round_num)
            return

        for agent in alive_agents:
            setattr(agent, "current_round", round_num)

        self.info_system.maybe_clear_cache(round_num)
        self._apply_operating_costs(alive_agents, market_conditions, round_num)

        # BLOCK 1: PROCESS MATURED INVESTMENTS & APPLY ALL INVESTMENT-RELATED LEARNING
        all_matured_outcomes = []
        for agent in alive_agents:
            matured_investments = agent.portfolio.check_matured_investments(round_num, market_conditions)
            for maturation in matured_investments:
                maturation['agent_id'] = agent.id
                all_matured_outcomes.append(maturation)

        for outcome in all_matured_outcomes:
            agent = self.agents[outcome['agent_id']]
            if not agent.alive:
                continue


            opp_obj = outcome['investment']['opportunity']
            invested_amount = outcome['investment']['amount']
            if hasattr(opp_obj, 'capital_history'):
                opp_obj.capital_history.append(invested_amount)
            realized_ratio = outcome['capital_returned'] / max(1, invested_amount)
            if hasattr(opp_obj, 'realized_returns'):
                opp_obj.realized_returns.append(realized_ratio)
                obs_window = opp_obj.realized_returns[-10:]
                opp_obj.latent_return_potential = 0.7 * opp_obj.latent_return_potential + 0.3 * safe_mean(obs_window)
                opp_obj.latent_return_potential = max(self.config.OPPORTUNITY_RETURN_RANGE[0],
                                                    min(self.config.OPPORTUNITY_RETURN_RANGE[1], opp_obj.latent_return_potential))
            sector = getattr(opp_obj, 'sector', None)
            if sector is not None:
                self.market.sector_performance_history[sector].append(realized_ratio - 1.0)
                # Adjust capital requirements based on success/failure feedback
                base_capital = getattr(self.config, 'OPPORTUNITY_CAPITAL_REQUIREMENTS', 10000)
                if outcome['success']:
                    opp_obj.capital_requirements = max(base_capital * 0.2, opp_obj.capital_requirements * 0.9)
                else:
                    opp_obj.capital_requirements = min(base_capital * 5.0, opp_obj.capital_requirements * 1.1)
            if hasattr(opp_obj, 'combination_signature'):
                self.market.record_innovation_outcome(
                    opportunity=opp_obj,
                    success=outcome['success'],
                    return_achieved=realized_ratio
                )
            if getattr(self, "knowledge_base", None) is not None and sector is not None:
                self.knowledge_base.apply_investment_outcome(
                    agent=agent,
                    sector=str(sector),
                    roi=realized_ratio - 1.0,
                    success=outcome['success'],
                    round_num=round_num,
                    opportunity=opp_obj,
                    ai_level=outcome.get('ai_level_used', 'none'),
                )
            decision_conf = (outcome.get('investment') or {}).get('decision_confidence')
            agent.record_paradox_observation(
                decision_confidence=decision_conf,
                realized_roi=realized_ratio,
                ai_used=outcome.get('ai_level_used', 'none') not in (None, 'none'),
                counterfactual_roi=outcome.get('counterfactual_roi'),
            )
            outcome['paradox_signal_post'] = agent.paradox_signal
            outcome['paradox_gap'] = getattr(agent, 'last_paradox_gap', None)
            outcome['paradox_cf_gap'] = getattr(agent, 'last_paradox_cf_gap', None)
            # Preserve success flag for downstream learning updates
            outcome['success'] = outcome.get('success', False)
            inv = outcome.get('investment', {})
            outcome['ai_level_used'] = inv.get('ai_level_used', 'none')
            outcome['ai_used'] = outcome['ai_level_used'] != 'none'

            ai_info = outcome.get('ai_info_at_investment')
            ai_was_accurate = None
            if ai_info is not None:
                analysis = getattr(ai_info, "_source_analysis", None)
                if analysis is not None:
                    outcome['ai_analysis'] = analysis
                    ai_was_accurate = self.info_system.update_agent_learning(agent.id, outcome, analysis)
                    if (
                        ai_was_accurate is False
                        and getattr(analysis, "contains_hallucination", False)
                        and getattr(self.market, "knowledge_base", None) is not None
                    ):
                        severity = float(np.clip(1.0 - getattr(analysis, "actual_accuracy", 0.0), 0.0, 1.0))
                        confidence = float(getattr(analysis, "confidence", 0.5) or 0.5)
                        severity *= float(np.clip(0.5 + 0.5 * confidence, 0.25, 1.0))
                        self.market.knowledge_base.apply_hallucination_penalty(agent, analysis.domain, severity)

            if inv.get('ai_level_used'):
                outcome['ai_used'] = inv['ai_level_used'] != 'none'
            outcome['round'] = round_num
            agent.resources.performance.record_return(
                'invest',
                outcome.get('capital_returned', 0.0),
                outcome.get('ai_level_used', 'none'),
                round_num
            )
            agent.update_state_from_outcome(outcome, ai_was_accurate=ai_was_accurate)
            self._enforce_survival_threshold(agent, round_num)
        # BLOCK 2: AGENT DECISION MAKING (Parallel Processing)
        def process_agent_decision(agent_data_local):
            """Processes a single agent's decision. Renamed arg to avoid confusion."""
            agent_id, _, _ = agent_data_local
            agent = self.agents[agent_id]
            local_market_conditions = dict(market_conditions)

            opportunities = self.market.get_opportunities(agent)
        
            neighbor_agents = []
            if self.config.USE_NETWORK_EFFECTS and self.agent_network:
                neighbor_ids = list(self.agent_network.neighbors(agent_id))
                if len(neighbor_ids) > 5:
                    neighbor_ids = random.sample(neighbor_ids, 5)
                neighbor_agents = [self.agents[nid] for nid in neighbor_ids if self.agents[nid].alive]
        
            decision = agent.make_decision(
                opportunities, local_market_conditions, self.info_system, 
                self.market, round_num, alive_agents, self.uncertainty_env,
                neighbor_agents=neighbor_agents
            )
        
            if decision is None: decision = {'action': 'maintain', 'ai_level_used': 'none'}
            decision['agent_id'] = agent_id

            ai_level = decision.get('ai_level_used', 'none')
            subscription_cost = float(decision.get('subscription_cost', 0.0) or 0.0)
            decision['subscription_cost'] = subscription_cost
            return decision, 0.0

        agent_data = [(agent.id, agent.traits.copy(), agent.resources.capital) for agent in alive_agents]
    
        if self.config.use_parallel and self.config.max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(process_agent_decision, agent_data))
            agent_actions = [res[0] for res in results]
            costs_to_apply = [(res[0]['agent_id'], res[1]) for res in results]
        else:
            agent_actions = []
            costs_to_apply = []
            for data in agent_data:
                decision, ai_cost = process_agent_decision(data)
                agent_actions.append(decision)
                costs_to_apply.append((decision['agent_id'], ai_cost))

        self.uncertainty_env.record_ai_signals(round_num, agent_actions)
        round_usage: Dict[int, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))

        # BLOCK 3: APPLY COSTS & PROCESS ACTIONS WITH IMMEDIATE OUTCOMES
        decision_costs: Dict[int, float] = collections.defaultdict(float)
        for agent_id, ai_cost in costs_to_apply:
            decision_costs[agent_id] += ai_cost

        for action in agent_actions:
            agent_id = action['agent_id']
            total_cost = decision_costs.get(agent_id, 0.0)
            per_use_cost = float(action.get('ai_per_use_cost', 0.0) or 0.0)
            subscription_cost = float(action.get('subscription_cost', 0.0) or 0.0)
            total_cost += per_use_cost
            combined_cost = total_cost + subscription_cost
            if combined_cost > 0:
                action['ai_cost_incurred'] = combined_cost
            if total_cost > 0:
                decision_costs[agent_id] = total_cost

        for agent_id, total_cost in decision_costs.items():
            agent = self.agents[agent_id]
            if agent.alive:
                if total_cost > 0:
                    agent.resources.capital = max(0.0, agent.resources.capital - total_cost)
                self._enforce_survival_threshold(agent, round_num)

        for agent in alive_agents:
            self._enforce_survival_threshold(agent, round_num)

        innovations_this_round = []
        for action in agent_actions:
            agent = self.agents[action['agent_id']]
            if not agent.alive:
                continue

            usage_profile = round_usage[action['agent_id']]
            action_type = action.get('action')
            if action_type == 'invest':
                details = action.get('chosen_opportunity_details') or {}
                sector = details.get('sector') or getattr(action.get('chosen_opportunity_obj'), 'sector', None)
                if sector:
                    usage_profile[sector] += 1.0
            elif action_type == 'explore':
                if action.get('exploration_type') == 'sector_knowledge':
                    sector = action.get('domain')
                    if sector:
                        usage_profile[sector] += 0.6
                elif action.get('exploration_type') == 'niche_discovery':
                    base_sector = action.get('base_sector')
                    if base_sector:
                        usage_profile[base_sector] += 0.4
            elif action_type == 'innovate':
                details = action.get('innovation_details') or {}
                sector = details.get('sector') or getattr(action.get('innovation_obj'), 'sector', None)
                if sector:
                    usage_profile[sector] += 0.8

            if action_type == 'innovate':
                rd_spend = float(action.get('rd_investment', 0.0) or 0.0)
                capital_return = 0.0
                cash_multiple = 0.0
                innovation_obj = action.get('innovation_obj')
                agent_resources = getattr(agent, 'resources', None)
                if innovation_obj is not None:
                    success, impact, cash_multiple = self.innovation_engine.evaluate_innovation_success(
                        innovation_obj, market_conditions, innovations_this_round
                    )
                    innovation_obj.success = success
                    innovation_obj.market_impact = impact
                    innovation_obj.cash_multiple = cash_multiple
                    action['success'] = success
                    action['market_impact'] = impact
                    action['cash_multiple'] = cash_multiple
                    if success:
                        self.knowledge_base.learn_from_success(agent.id, innovation_obj, agent_resources=agent_resources)
                        innovations_this_round.append(innovation_obj)
                    else:
                        self.knowledge_base.learn_from_failure(agent.id, innovation_obj, agent_resources=agent_resources)
                else:
                    success = False
                    impact = 0.0
                    cash_multiple = float(getattr(self.config, 'INNOVATION_FAIL_RECOVERY_RATIO', 0.15))
                    action['success'] = False
                    action['market_impact'] = impact
                    action['cash_multiple'] = cash_multiple

                capital_return = rd_spend * cash_multiple if rd_spend > 0 else 0.0
                action['capital_returned'] = capital_return
                action['net_return'] = capital_return - rd_spend
                agent.resources.performance.record_return(
                    'innovate',
                    capital_return,
                    action.get('ai_level_used', 'none'),
                    round_num
                )
                tier_used = normalize_ai_label(action.get('ai_level_used', 'none'))
                agent._update_ai_tier_belief(tier_used if tier_used != 'none' else 'none', cash_multiple)
                if innovation_obj is not None:
                    self.uncertainty_env.register_innovation_event(
                        opportunity_id=getattr(innovation_obj, 'id', f"innovation_{agent.id}_{round_num}"),
                        success=action.get('success'),
                        impact=action.get('market_impact', 0.0),
                        combination_signature=getattr(innovation_obj, 'combination_signature', None),
                        new_possibility_rate=self.uncertainty_env.agentic_novelty_state.get('new_possibility_rate', 0.0),
                        scarcity=innovation_obj.scarcity,
                    )
                    if success:
                        scarcity = getattr(innovation_obj, 'scarcity', None)
                        if scarcity is None:
                            scarcity = self.knowledge_base.get_combination_scarcity(innovation_obj.knowledge_components)
                        innovation_obj.scarcity = scarcity
                        self.market.spawn_opportunity_from_innovation(innovation_obj, cash_multiple)
                cf_multiple = cash_multiple
                if str(action.get('ai_level_used', 'none')).lower() != 'none':
                    cf_multiple = float(np.clip(cash_multiple - 0.15, 0.05, 2.5))
                agent.record_paradox_observation(
                    decision_confidence=action.get('decision_confidence'),
                    realized_roi=cash_multiple,
                    ai_used=str(action.get('ai_level_used', 'none')).lower() != 'none',
                    counterfactual_roi=cf_multiple,
                )
                action['counterfactual_roi'] = cf_multiple
                action['paradox_gap'] = getattr(agent, 'last_paradox_gap', None)
                action['paradox_cf_gap'] = getattr(agent, 'last_paradox_cf_gap', None)

            if action_type != 'invest':
                outcome = self._calculate_action_outcome(action, market_conditions, round_num)
                agent.update_state_from_outcome(outcome)
                self._enforce_survival_threshold(agent, round_num)

        for action in agent_actions:
            if action.get('action') == 'explore' and action.get('discovered_niche'):
                agent_id = action['agent_id']
                agent = self.agents[agent_id]
                niche_id = action['discovered_niche']
                n_niche_opps = np.random.randint(1, 4)
                for _ in range(n_niche_opps):
                    niche_opp = self.market.create_niche_opportunity(niche_id, agent_id, round_num)
                    if not hasattr(agent, 'discovered_niches'): agent.discovered_niches = []
                    agent.discovered_niches.append(niche_opp.id)

        decay_rate = float(getattr(self.config, 'KNOWLEDGE_DECAY_RATE', 0.0) or 0.0)
        decay_shift_events = list(self._pending_shift_events) if hasattr(self, "_pending_shift_events") else []
        sector_pressure = {}
        if market_state:
            sector_pressure = market_state.get('sector_pressure', {}) or {}
        if decay_rate > 0:
            for agent in alive_agents:
                usage_weights = round_usage.get(agent.id, {})
                usage_dict = {sector: float(weight) for sector, weight in usage_weights.items()} if usage_weights else {}
                level = str(getattr(agent, "current_ai_level", "none") or "none").lower()
                decay_mod = {"none": 1.08, "basic": 0.95, "advanced": 0.7, "premium": 0.5}.get(level, 1.0)
                adjusted_decay = decay_rate * decay_mod
                agent.resources.decay_resources(
                    adjusted_decay,
                    usage_dict,
                    current_round=round_num,
                    shift_events=decay_shift_events,
                    sector_pressure=sector_pressure,
                )
                if hasattr(self, "knowledge_base"):
                    max_knowledge = getattr(self.config, 'MAX_AGENT_KNOWLEDGE', None)
                    prune_threshold = float(getattr(self.config, 'SECTOR_STRENGTH_PRUNE_THRESHOLD', 0.1))
                    self.knowledge_base.forget_stale_knowledge(agent, round_num, max_size=max_knowledge)
                    self.knowledge_base.prune_by_sector_strength(agent, agent.resources.knowledge, threshold=prune_threshold)
        if hasattr(self, "_pending_shift_events"):
            self._pending_shift_events = []

        # BLOCK 4: UPDATE MARKET AND SAVE DATA
        for i in alive_indices:
            agent = self.agents[i]
            self.agent_state.capital[i] = agent.resources.capital
            self.agent_state.alive[i] = agent.alive
            self.agent_state.portfolio_size[i] = agent.portfolio.portfolio_size
            self.agent_state.innovation_count[i] = len(agent.innovations)
            initial_equity = getattr(agent.resources.performance, 'initial_equity', self.config.INITIAL_CAPITAL)
            if initial_equity > 0:
                self.agent_state.capital_growth[i] = agent.resources.capital / initial_equity
            else:
                self.agent_state.capital_growth[i] = 0.0

        alive_ratio = len(alive_agents) / max(1, self.config.N_AGENTS)
        setattr(self.market, "_current_alive_ratio", alive_ratio)
        market_state = self.market.step(round_num, agent_actions, innovations_this_round, matured_outcomes=all_matured_outcomes)
        shift_event = market_state.get('market_shift') if market_state else None
        if shift_event:
            impacts = shift_event.get('impacts', []) if isinstance(shift_event, dict) else []
            for impact in impacts:
                shift_sector = impact.get('sector')
                severity = float(impact.get('severity', 0.0) or 0.0)
                if shift_sector and severity > 0.0:
                    for agent in alive_agents:
                        knowledge_val = agent.resources.knowledge.get(shift_sector)
                        if knowledge_val is not None:
                            agent.resources.knowledge[shift_sector] = max(0.01, knowledge_val * (1.0 - severity))
                            agent.resources.knowledge_last_used[shift_sector] = round_num
            if hasattr(self, "_pending_shift_events"):
                self._pending_shift_events = impacts
        else:
            if hasattr(self, "_pending_shift_events"):
                self._pending_shift_events = []
        if market_state is None:
            market_state = {}
        if self._last_uncertainty_state is not None:
            market_state.setdefault('uncertainty_state', self._last_uncertainty_state)
    
        uncertainty_state = self.uncertainty_env.measure_uncertainty_state(
            self.market, agent_actions, innovations_this_round, round_num
        )
        self._last_uncertainty_state = uncertainty_state
        market_state['uncertainty_state'] = uncertainty_state
        self._record_round_to_buffer(
            round_num,
            agent_actions,
            market_state,
            uncertainty_state,
            innovations_this_round,
            all_matured_outcomes,
            alive_agents=alive_agents
        )
        for action in agent_actions:
            action.pop('perception_at_decision', None)
            action.pop('chosen_opportunity_obj', None)
            action.pop('innovation_obj', None)
            action.pop('ai_analysis', None)
            action.pop('ai_info_at_investment', None)
        for agent in self.agents:
            agent.portfolio.archive_matured_history()
                
    def _save_final_agent_state(self): # This is the correct method
        """Saves final agent data to a pickle file for the analysis framework."""
        agent_records = [
            {
                'run_id': self.run_id,
                'agent_id': a.id,
                'survived': a.alive,
                'failure_round': getattr(a, 'failure_round', None),  # NEW: When did agent fail
                'failure_reason': getattr(a, 'failure_reason', None),  # Why did agent fail
                'final_capital': a.resources.capital,
                'capital_growth': a.resources.capital / max(
                    1e-6,
                    getattr(a.resources.performance, 'initial_equity', self.config.INITIAL_CAPITAL)
                ),
                'primary_ai_level': canonical_to_display(
                    normalize_ai_label(
                        getattr(a, 'current_ai_level', None) or getattr(a, 'agent_type', 'none')
                    )
                ),
                'primary_ai_canonical': normalize_ai_label(
                    getattr(a, 'current_ai_level', None) or getattr(a, 'agent_type', 'none')
                ),
                'innovations': len(a.innovations),
                'portfolio_diversity': a.portfolio.diversification_score,
                **a.traits
            } for a in self.agents
        ]
        output_base = self.data_paths.get('base', self.output_dir)
        os.makedirs(output_base, exist_ok=True)
        pd.DataFrame(agent_records).to_pickle(os.path.join(output_base, 'final_agents.pkl'))
        
    def _calculate_action_outcome(self, action: dict, market_state: dict, round_num: int) -> dict:
        """Calculate outcomes - investments don't return immediately."""
        outcome = {
            'action': action.get('action'), 
            'round': round_num, 
            'ai_used': action.get('ai_level_used', 'none') != 'none'
        }
    
        if action.get('action') == 'invest':
            outcome['executed'] = bool(action.get('executed', False))
            outcome['success'] = action.get('success') if action.get('success') is not None else None
            outcome['capital_returned'] = 0  # No immediate return
            outcome['investment_locked'] = action.get('investment_amount', 0)
            outcome['outcome_pending'] = action.get('outcome_pending', True)

        elif action.get('action') == 'innovate':
            success = bool(action.get('success', False))
            outcome['success'] = success
            rd_spend = float(action.get('rd_investment', 0.0) or 0.0)
            capital_returned = float(action.get('capital_returned', 0.0) or 0.0)
            outcome['capital_returned'] = capital_returned
            outcome['net_return'] = capital_returned - rd_spend

        elif action.get('action') == 'explore':
            # Exploration costs and serendipity still immediate
            outcome['success'] = True
            if 'serendipity_reward' in action:
                outcome['capital_returned'] = action['serendipity_reward']
            else:
                outcome['capital_returned'] = 0
    
        else:  # maintain
            outcome['success'] = True
            outcome['capital_returned'] = 0
    
        return outcome
