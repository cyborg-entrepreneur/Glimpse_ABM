# GlimpseABM: Technical Documentation & Object-Oriented Design

## Comprehensive Documentation for the Agent-Based Model of AI Augmentation and the Paradox of Future Knowledge

**Version:** 1.0
**Last Updated:** February 2026
**Associated Paper:** "Into the Flux: AI Augmentation & The Paradox of Future Knowledge" (Entrepreneurship Theory & Practice)

---

## Table of Contents

1. [Overview & Theoretical Foundation](#1-overview--theoretical-foundation)
2. [Architecture & Module Dependencies](#2-architecture--module-dependencies)
3. [Core Data Models](#3-core-data-models)
4. [Agent Design & Cognition](#4-agent-design--cognition)
5. [Knightian Uncertainty System](#5-knightian-uncertainty-system)
6. [AI Augmentation & Information System](#6-ai-augmentation--information-system)
7. [Innovation & Knowledge Recombination](#7-innovation--knowledge-recombination)
8. [Market Environment](#8-market-environment)
9. [Simulation Engine](#9-simulation-engine)
10. [Key Variable Measurements](#10-key-variable-measurements)
11. [Calibration & Empirical Grounding](#11-calibration--empirical-grounding)
12. [References](#12-references)

---

## 1. Overview & Theoretical Foundation

### 1.1 Purpose

GlimpseABM is a comprehensive agent-based model designed to investigate the **paradox of future knowledge** in AI-augmented entrepreneurship. The model operationalizes four dimensions of Knightian uncertainty and examines how AI augmentation impacts entrepreneurs' abilities to identify and pursue opportunities in competitive markets.

> "At the heart of these dynamics is a paradox of future knowledge in competitive markets where the pursuit of entrepreneurial foresight systematically generates the uncertainties it seeks to resolve." (Townsend et al., 2026)

### 1.2 Theoretical Framework

The model builds on the Knightian uncertainty framework from Townsend et al. (2024, 2025a, 2025b), which identifies four irreducible problems for entrepreneurial action:

| Dimension | Definition | Code Operationalization |
|-----------|------------|------------------------|
| **Actor Ignorance** | Limits on what any entrepreneur can know about relevant conditions and plausible futures | `actor_ignorance_state` in `uncertainty.py` |
| **Practical Indeterminism** | Irreducible unpredictability in complex systems even when information is extensive | `practical_indeterminism_state` in `uncertainty.py` |
| **Agentic Novelty** | How entrepreneurial action introduces new possibilities that did not previously exist | `agentic_novelty_state` in `uncertainty.py` |
| **Competitive Recursion** | Strategic interdependence as entrepreneurs respond to one another's moves | `competitive_recursion_state` in `uncertainty.py` |

### 1.3 Central Hypothesis

The model tests whether AI augmentation resolves or amplifies the paradox of future knowledge:

> "AI can reduce actor ignorance through superior information processing (a first-order benefit). But by revealing and legitimizing 'high-value' opportunity spaces to many other AI-augmented actors at once, AI can also increase market convergence, accelerate overcrowding, and amplify competitive recursion." (Townsend et al., 2026)

### 1.4 Key Finding: The Innovation Equilibrium Trap

The model reveals that AI augmentation paradoxically reduces entrepreneurial survival through an **innovation equilibrium trap**:

```
Superior AI Information → Strategic Convergence → Market Crowding → Reduced Returns → Lower Survival
```

---

## 2. Architecture & Module Dependencies

### 2.1 Module Dependency Graph

```
                              EmergentConfig (config.py)
                                       │
                                       ▼
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
            KnowledgeBase      Data Models (models.py)   Utils
           (knowledge.py)              │              (utils.py)
                    │                  │
                    ▼                  ▼
        ┌───────────┴───────────┬──────┴──────┐
        │                       │             │
        ▼                       ▼             ▼
KnightianUncertainty    MarketEnvironment   Information
  (uncertainty.py)        (market.py)      (information.py)
        │                       │             │
        └───────────┬───────────┴─────────────┘
                    │
                    ▼
            ┌───────┴───────┐
            │               │
            ▼               ▼
    InnovationEngine   EmergentAgent
    (innovation.py)    (agents.py)
            │               │
            └───────┬───────┘
                    │
                    ▼
            EmergentSimulation
            (simulation.py)
```

### 2.2 File Summary

| File | Lines | Primary Purpose | Key Classes |
|------|-------|-----------------|-------------|
| `config.py` | ~1,200 | Configuration & calibration | `EmergentConfig`, `AILevelConfig`, `SectorProfile` |
| `models.py` | ~600 | Core data structures | `Opportunity`, `Innovation`, `Knowledge`, `Information` |
| `agents.py` | ~3,100 | Agent implementation | `EmergentAgent`, `AgentResources`, `Portfolio` |
| `uncertainty.py` | ~500 | Knightian uncertainty tracking | `KnightianUncertaintyEnvironment` |
| `market.py` | ~400 | Market dynamics | `MarketEnvironment` |
| `innovation.py` | ~550 | Knowledge recombination | `InnovationEngine`, `CombinationTracker` |
| `information.py` | ~400 | AI analysis generation | `EnhancedInformationSystem` |
| `knowledge.py` | ~500 | Knowledge base management | `KnowledgeBase` |
| `simulation.py` | ~500 | Orchestration | `EmergentSimulation` |

---

## 3. Core Data Models

### 3.1 Opportunity

**File:** `models.py`
**Theoretical Grounding:** Operationalizes "generative entrepreneurial futures" where opportunities have latent characteristics that manifest only through action (Ramoglou & McMullen, 2024).

```python
@dataclass
class Opportunity:
    """
    Represents a market investment target with latent and realized characteristics.

    The separation between latent (unknown) and realized returns reflects Knightian
    uncertainty - entrepreneurs cannot know the true value ex ante.
    """
    id: str
    latent_return_potential: float      # Hidden true return (0-1)
    latent_failure_potential: float     # Hidden risk (0-1)
    complexity: float                   # Technical complexity (0-1)
    sector: str                         # tech/retail/service/manufacturing
    lifecycle_stage: str                # emerging/growing/mature/declining
    discovery_round: int
    discovered_by: Optional[int] = None
```

**Key Method: `realized_return()`**
```python
def realized_return(self, market_conditions: Dict, investor_tier: str) -> float:
    """
    Generates stochastic returns incorporating:
    - Regime effects (crisis → boom multipliers)
    - Lifecycle stage effects
    - Market clearing dynamics (crowding penalties)
    - Scarcity effects (early mover advantages)
    - Power law distribution (Pareto α=3.0 for unicorn potential)

    Calibrated to Kaplan & Schoar / Korteweg & Sorensen VC return distributions.
    """
```

### 3.2 Innovation

**File:** `models.py`
**Theoretical Grounding:** Implements knowledge recombination theory (Weitzman, 1998; Fleming, 2001; Hargadon & Sutton, 1997).

```python
@dataclass
class Innovation:
    """
    Novel combinations of knowledge that create new opportunities.

    Innovation emerges through recombination with "recombinant uncertainty"
    about which combinations will succeed (Fleming, 2001).
    """
    id: str
    type: str                           # incremental/architectural/radical/disruptive
    knowledge_components: List[str]     # IDs of combined knowledge pieces
    novelty: float                      # Uniqueness score (0-1)
    quality: float                      # Execution quality (0-1)
    round_created: int
    creator_id: int
    ai_assisted: bool
    ai_domains_used: List[str]
    sector: str
    combination_signature: str          # Unique hash of knowledge combination
    is_new_combination: bool            # True if never attempted before
    success: Optional[bool] = None
    market_impact: Optional[float] = None
    scarcity: Optional[float] = None
```

### 3.3 Knowledge

**File:** `models.py`
**Theoretical Grounding:** Represents discrete knowledge pieces that can be recombined (Cohen & Levinthal, 1990; Katila & Ahuja, 2002).

```python
@dataclass
class Knowledge:
    """
    Discrete knowledge piece with domain and sophistication level.

    Knowledge constrains which opportunities can be credibly imagined
    and pursued (Shane, 2003; Grégoire & Shepherd, 2012).
    """
    id: str
    domain: str           # technology/market/process/business_model
    level: float          # Sophistication (0-1)
    discovered_round: int
    source: str           # exploration/innovation/ai_discovery
```

### 3.4 Information

**File:** `models.py`
**Theoretical Grounding:** AI-generated analysis with systematic errors operationalizing the "paradox of future knowledge."

```python
@dataclass
class Information:
    """
    AI-generated analysis with accuracy, hallucinations, and biases.

    Captures how AI systems "perform tasks and solve complex problems"
    (Obschonka & Audretsch, 2020) while introducing new uncertainty sources.
    """
    estimated_return: float
    estimated_uncertainty: float
    confidence: float
    insights: Dict[str, Any]
    hidden_factors: List[str]
    domain: str
    actual_accuracy: float
    contains_hallucination: bool
    bias_applied: float
    overconfidence_factor: float
```

### 3.5 AILearningProfile

**File:** `models.py`
**Theoretical Grounding:** Implements Bayesian learning of AI reliability by domain.

```python
@dataclass
class AILearningProfile:
    """
    Agent's learned model of AI capabilities across domains.

    Enables adaptive trust calibration based on observed accuracy.
    """
    domain_trust: Dict[str, float]              # Trust per domain (0-1)
    accuracy_estimates: Dict[str, List[float]]  # Rolling accuracy scores
    hallucination_experiences: Dict[str, int]   # Hallucination counts
    usage_count: Dict[str, int]                 # Usage frequency

    def update_trust(self, domain: str, accuracy: float, had_hallucination: bool):
        """Bayesian-like trust update from observed outcomes."""

    def should_use_ai_for_domain(self, domain: str) -> bool:
        """Decision rule for AI reliance based on learned reliability."""
```

---

## 4. Agent Design & Cognition

### 4.1 EmergentAgent

**File:** `agents.py`
**Theoretical Grounding:** Heterogeneous cognitive profiles based on entrepreneurship cognition research (Mitchell et al., 2002, 2004; Grégoire et al., 2011, 2015).

```python
class EmergentAgent:
    """
    Entrepreneurial agent with heterogeneous traits, resources, and learning.

    Each agent is created with distinct cognitive profiles governing
    opportunity-identification decision-making (Table 1 in paper).
    """

    # Identity
    id: int
    agent_type: str
    traits: Dict[str, float]        # 10-dimensional trait profile
    primary_sector: str             # NVCA-weighted sector assignment

    # Resources
    resources: AgentResources       # Capital, knowledge, capabilities
    portfolio: Portfolio            # Active and pending investments

    # AI Adaptation
    current_ai_level: str           # none/basic/advanced/premium
    fixed_ai_level: Optional[str]   # For fixed-tier experiments
    ai_learning: AILearningProfile  # Learned AI trust
    ai_tier_beliefs: Dict[str, Tuple[float, float]]  # Beta distribution params

    # Learning
    uncertainty_response: UncertaintyResponseProfile
    outcomes_history: deque         # Decision outcome memory
```

### 4.2 Ten-Dimensional Trait Profile

**Theoretical Grounding:** Table 1 in the paper; drawn from entrepreneurship cognition literature.

| Trait | Range | Influence on Decisions | Literature |
|-------|-------|----------------------|------------|
| `uncertainty_tolerance` | 0.3-0.9 | Willingness to act under ambiguity | McKelvie et al. (2011) |
| `innovativeness` | 0.2-0.8 | Propensity for novel combinations | Shane & Venkataraman (2000) |
| `market_awareness` | 0.3-0.7 | Sensitivity to market signals | Grégoire & Shepherd (2012) |
| `exploration_tendency` | 0.25-0.75 | Breadth vs depth search | Katila & Ahuja (2002) |
| `risk_appetite` | 0.2-0.7 | Return vs security preference | Wiltbank et al. (2009) |
| `ai_trust` | 0.3-0.8 | Initial reliance on AI systems | Shepherd & Majchrzak (2022) |
| `competence` | 0.4-0.85 | Domain expertise level | Cohen & Levinthal (1990) |
| `adaptability` | 0.35-0.8 | Learning rate from outcomes | Sarasvathy (2001) |
| `social_orientation` | 0.25-0.7 | Network utilization | Aldrich & Ruef (2018) |
| `strategic_patience` | 0.3-0.75 | Time horizon for decisions | McMullen & Dimov (2013) |

### 4.3 AgentResources

**File:** `agents.py`
**Theoretical Grounding:** Four capital categories calibrated to NVCA benchmarks.

```python
@dataclass
class AgentResources:
    """
    Multidimensional resource model tracking capital, knowledge, and capabilities.
    """
    capital: float                          # Liquid capital ($2.5M-$10M initial)
    knowledge: Dict[str, float]             # Sector-specific knowledge levels
    capabilities: Dict[str, float]          # market_timing, opportunity_evaluation,
                                           # innovation, uncertainty_management
    experience_units: float                 # Accumulated learning

    def decay_resources(self, config: EmergentConfig):
        """
        Sector-specific Ebbinghaus forgetting curve calibration.
        Knowledge decays without active use or reinforcement.
        """
```

### 4.4 Decision-Making Process

**Theoretical Grounding:** Figure 1 in paper; operationalizes opportunity identification as per McMullen & Shepherd (2006).

```python
def act_in_round(self, round: int, market: MarketEnvironment,
                 uncertainty_env: KnightianUncertaintyEnvironment) -> Dict:
    """
    Main decision-making loop executed each simulation round.

    Process:
    1. Gather information through AI systems (if using)
    2. Perceive uncertainty across four dimensions
    3. Select action via softmax over utilities
    4. Execute action (invest/innovate/explore/maintain)
    5. Update learning profiles based on outcomes
    """

def _choose_action_type(self, market_conditions: Dict,
                        uncertainty_perception: Dict) -> str:
    """
    Softmax selection with temperature scaling by AI tier.

    Premium AI → deterministic (highest utility wins)
    No AI → stochastic (randomness in near-ties)

    "When humans perceive two or more options to be essentially
    identical, their final choice is subject to a higher degree
    of randomness." (Paper, p. 56)
    """
```

### 4.5 Portfolio Management

**File:** `agents.py`

```python
@dataclass
class Portfolio:
    """
    Tracks investments through lifecycle: active → pending → matured.
    """
    active_investments: List[Investment]
    pending_investments: List[Investment]
    matured_investments: List[Investment]

    def check_matured_investments(self, round: int, market_conditions: Dict) -> List:
        """
        Realizes returns based on market conditions and opportunity characteristics.
        Returns follow power law distribution (Pareto α=3.0).
        """
```

---

## 5. Knightian Uncertainty System

### 5.1 KnightianUncertaintyEnvironment

**File:** `uncertainty.py`
**Theoretical Grounding:** Operationalizes the four-dimensional framework from Townsend et al. (2024, 2025a).

```python
class KnightianUncertaintyEnvironment:
    """
    Tracks four-dimensional Knightian uncertainty at environment and agent levels.

    "The uncertainty module operationalizes the four-dimensional framework
    from Townsend et al. (2024, 2025), implementing each uncertainty dimension
    as a continuous variable." (Paper, p. 61)
    """

    # State dictionaries for each dimension
    actor_ignorance_state: Dict[str, float]
    practical_indeterminism_state: Dict[str, float]
    agentic_novelty_state: Dict[str, float]
    competitive_recursion_state: Dict[str, float]

    # AI-specific signals (paradox mechanisms)
    hallucination_events: int
    confidence_miscalibration: float
    ai_herding_patterns: float
```

### 5.2 Actor Ignorance Measurement

**Theoretical Definition:** "Limits on what any entrepreneur can know about relevant conditions and plausible futures" (Townsend et al., 2025a).

```python
actor_ignorance_state = {
    "unknown_opportunities": float,    # Fraction of opportunities not yet discovered
    "knowledge_gaps": float,           # Missing domain knowledge
    "emergence_potential": float,      # Latent opportunity space
}

def compute_actor_ignorance(self) -> float:
    """
    Aggregates ignorance metrics into scalar (0-1).

    AI augmentation reduces this dimension through superior
    information processing and pattern recognition.
    """
    return weighted_mean([
        self.actor_ignorance_state["unknown_opportunities"] * 0.4,
        self.actor_ignorance_state["knowledge_gaps"] * 0.35,
        self.actor_ignorance_state["emergence_potential"] * 0.25
    ])
```

### 5.3 Practical Indeterminism Measurement

**Theoretical Definition:** "Irreducible unpredictability in complex systems even when information is extensive" (Townsend et al., 2025a).

```python
practical_indeterminism_state = {
    "path_volatility": float,          # Execution outcome variance
    "timing_criticality": float,       # Sensitivity to action timing
    "regime_instability": float,       # Macroeconomic transition risk
}

def compute_practical_indeterminism(self) -> float:
    """
    Captures inherent unpredictability even with perfect information.

    "The uncertainty inherent in attempting novel combinations is
    irreducible--the agent does not know what other agents plan
    to do in the same period." (Paper, p. 107)
    """
```

### 5.4 Agentic Novelty Measurement

**Theoretical Definition:** "How entrepreneurial action introduces new possibilities that did not previously exist" (Townsend et al., 2025a).

```python
agentic_novelty_state = {
    "creative_momentum": float,        # Rate of new combination creation
    "disruption_potential": float,     # Market-reshaping innovation rate
    "novelty_level": float,            # Uniqueness of recent innovations
}

def compute_agentic_novelty(self) -> float:
    """
    Tracks how entrepreneurial action creates new futures.

    "Novel innovations alter the very space of possibilities,
    preventing entrepreneurs from fully predicting or knowing
    ex ante what will be created." (Rescher, 2016, cited in paper)
    """
```

### 5.5 Competitive Recursion Measurement

**Theoretical Definition:** "Strategic interdependence as entrepreneurs respond to one another's moves, generating cascading strategic dynamism" (Townsend et al., 2025a).

```python
competitive_recursion_state = {
    "strategic_opacity": float,        # Difficulty predicting competitor moves
    "herding_pressure": float,         # Degree of strategic convergence
    "game_complexity": float,          # Strategic interaction intensity
}

def compute_competitive_recursion(self) -> float:
    """
    Captures cascading strategic responses among agents.

    This dimension is AMPLIFIED by AI augmentation as agents
    converge on similar opportunity spaces.
    """
```

### 5.6 AI Herding Detection

```python
def record_ai_signals(self, tier_actions: Dict, hallucination_occurred: bool,
                      confidence_gap: float):
    """
    Tracks AI-specific uncertainty amplification signals.

    - Hallucination events: AI errors that induce confident wrong action
    - Confidence miscalibration: Gap between stated and actual accuracy
    - Herding patterns: Convergence of AI-tier actions (HHI)

    These signals operationalize the "second-order costs" of AI augmentation.
    """
```

---

## 6. AI Augmentation & Information System

### 6.1 AI Tier Configuration

**File:** `config.py`
**Theoretical Grounding:** LLM scaling laws (Kaplan et al., 2020) applied to entrepreneurial decision support.

```python
@dataclass
class AILevelConfig:
    """
    AI capability tier specification based on scaling law extrapolations.

    "We adopted the LLM scaling laws to model realistic assumptions
    about the capabilities and cost structure of these systems." (Paper, p. 47)
    """
    cost: float              # Monthly subscription cost
    tier_name: str           # none/basic/advanced/premium
    info_quality: float      # Signal accuracy (0-1)
    info_breadth: float      # Opportunity landscape coverage (0-1)
    hallucination_rate: float  # Error frequency
```

**Tier Specifications (Table 2a in paper):**

| Tier | Info Quality Formula | Calibration Target |
|------|---------------------|-------------------|
| None | 0.25 | Human baseline |
| Basic | 0.25 + 0.09 × log₁₀(10³) = 0.43 | GPT-5 (2025) |
| Advanced | 0.25 + 0.09 × log₁₀(10⁶) = 0.70 | Frontier 2026 |
| Premium | 0.25 + 0.09 × log₁₀(10⁹) = 0.97 | AGI 2027 |

**Derived Parameters:**
```python
info_breadth = info_quality - 0.05
hallucination_rate = 0.30 * (1 - info_quality)
bias = ±0.08 * (1 - info_quality)
```

### 6.2 EnhancedInformationSystem

**File:** `information.py`
**Theoretical Grounding:** Operationalizes AI as "epistemic technologies" (Shepherd & Majchrzak, 2022) with systematic errors.

```python
class EnhancedInformationSystem:
    """
    Generates AI-augmented opportunity analyses with realistic error patterns.

    "The information system applies tier-specific noise, potential
    hallucinations, and systematic biases to the true underlying
    opportunity characteristics." (Paper, p. 55)
    """

    def get_ai_analysis(self, opportunity: Opportunity, ai_level: str,
                        domain: str) -> Information:
        """
        Generates AI analysis with:
        1. Base accuracy from tier capability
        2. Lognormal noise (quality-dependent variance)
        3. Hallucinations (probability = hallucination_rate)
        4. Systematic bias (positive/negative by tier)
        5. Overconfidence multiplier

        Returns Information object with stated confidence that may
        be miscalibrated relative to actual accuracy.
        """
```

### 6.3 Hallucination Mechanism

```python
def _generate_hallucination(self, true_value: float, ai_level: str) -> float:
    """
    Generates hallucinated (confidently wrong) analysis.

    "Hallucinations are not merely random errors but systematically
    overconfident ones. When an AI system hallucinates, it does so
    with high stated confidence, leading agents to act decisively
    on false information." (Paper, p. 48)
    """
    # Hallucinated values are far from truth with high confidence
    direction = np.random.choice([-1, 1])
    magnitude = np.random.uniform(0.3, 0.6)
    hallucinated = np.clip(true_value + direction * magnitude, 0, 1)
    return hallucinated
```

### 6.4 Domain-Specific Capabilities

```python
AI_DOMAIN_CAPABILITIES = {
    "market_analysis": {
        "base_accuracy_modifier": +0.03,
        "hallucination_modifier": -0.02,
    },
    "technical_assessment": {
        "base_accuracy_modifier": +0.02,
        "hallucination_modifier": +0.01,
    },
    "uncertainty_evaluation": {
        "base_accuracy_modifier": -0.03,  # AI struggles with deep uncertainty
        "hallucination_modifier": +0.03,
    },
    "innovation_potential": {
        "base_accuracy_modifier": +0.01,
        "hallucination_modifier": +0.02,
    },
}
```

---

## 7. Innovation & Knowledge Recombination

### 7.1 InnovationEngine

**File:** `innovation.py`
**Theoretical Grounding:** Knowledge recombination theory (Weitzman, 1998; Fleming, 2001; Hargadon & Sutton, 1997).

```python
class InnovationEngine:
    """
    Manages innovation through knowledge combination.

    "Innovation often emerges through knowledge recombination and
    brokerage across domains, and search behaviors whose uncertainty
    is inherently recombinant." (Paper, citing Fleming, 2001)
    """

    config: EmergentConfig
    knowledge_base: KnowledgeBase
    combination_tracker: CombinationTracker
    innovations: Dict[str, Innovation]
```

### 7.2 Innovation Attempt Process

```python
def attempt_innovation(self, agent: EmergentAgent, market_conditions: Dict,
                       round: int, ai_level: str) -> Optional[Innovation]:
    """
    Innovation attempt flow:

    1. Get accessible knowledge (AI tier affects breadth)
    2. Compute innovation probability:
       - Sector-specific base rate (NSF BRDIS calibrated)
       - Agent innovativeness and competence
       - Dynamic AI bonus from learned trust (NOT hardcoded by tier)
       - Human ingenuity bonus (exploration_tendency + market_awareness)

    3. Select knowledge components for combination
    4. Create innovation with novelty and quality scores
    5. Record combination signature for crowding tracking

    "AI augmentation increases the volume of innovation activity
    without improving its quality, success probability, or underlying
    sophistication." (Paper, Table 3K findings)
    """
```

### 7.3 Knowledge Combination Selection

```python
def _select_knowledge_combination(self, accessible_knowledge: List[Knowledge],
                                   n_components: int, ai_level: str) -> List[Knowledge]:
    """
    Selects knowledge pieces for combination using adjacency scoring.

    Domain Adjacency Matrix:
                    tech  market  process  business_model
    technology      1.0    0.4     0.4        0.1
    market          0.4    1.0     0.1        0.4
    process         0.4    0.1     1.0        0.4
    business_model  0.1    0.4     0.4        1.0

    AI-level effects on selection:
    - Premium/Advanced: Prefer RARE knowledge (scarcity advantage)
    - Basic: May use COMMON knowledge (familiarity bias)
    - None: Random selection weighted by compatibility
    """
```

### 7.4 Innovation Type Determination

```python
def _determine_innovation_type(self, knowledge_pieces: List, agent_traits: Dict,
                                market_conditions: Dict, ai_assisted: bool) -> str:
    """
    Classifies innovation by type based on context.

    Types (Henderson & Clark, 1990 taxonomy):
    - incremental: 40% base + experience bonus
    - architectural: 30% base + momentum bonus
    - radical: 20% base + innovativeness bonus
    - disruptive: 10% base + exploration bonus + crisis bonus

    AI assistance boosts architectural (+8%) and radical (+5%) types.
    """
```

### 7.5 CombinationTracker

```python
class CombinationTracker:
    """
    Tracks knowledge combinations across the market for crowding detection.

    Central to identifying the "innovation equilibrium trap" where
    AI-augmented agents converge on similar combinations.
    """

    combination_history: Dict[str, List[str]]   # signature → innovation_ids
    combination_success: Dict[str, List[float]] # signature → success outcomes
    total_combinations: int

    def is_new_signature(self, signature: str) -> bool:
        """Check if combination has ever been attempted."""

    def get_reuse_ratio(self, signature: str) -> float:
        """Fraction of total innovations using this combination."""
```

### 7.6 Innovation Success Evaluation

```python
def evaluate_innovation_success(self, innovation: Innovation,
                                 market_conditions: Dict,
                                 market_innovations: List[Innovation]) -> Tuple[bool, float, float]:
    """
    Evaluates innovation success with:

    1. Base potential from novelty × quality
    2. Competition factor (sector HHI-calibrated intensity)
    3. Market readiness factor (novelty timing effects)
    4. Scarcity boost (rare combinations rewarded)
    5. AI execution multiplier (for refutation tests)

    Returns: (success: bool, impact: float, cash_multiple: float)

    "Innovation success rates remain relatively constant across AI tiers
    (approximately 10-12%)... AI augmentation increases the volume of
    innovation attempts without meaningfully improving the probability
    that any individual attempt will succeed." (Paper, Table 3I)
    """
```

---

## 8. Market Environment

### 8.1 MarketEnvironment

**File:** `market.py`
**Theoretical Grounding:** Regime-switching dynamics calibrated to NBER business cycle data.

```python
class MarketEnvironment:
    """
    Dynamic market with regime transitions, sector heterogeneity, and crowding.

    "Five macroeconomic regimes characterize aggregate market conditions...
    Regime transitions follow a Markov switching process with empirically
    calibrated transition probabilities derived from NBER business cycle
    dating." (Paper, p. 65)
    """

    # State
    opportunities: List[Opportunity]
    market_regime: str              # crisis/recession/normal/growth/boom
    volatility: float

    # Sector tracking
    opportunities_by_sector: Dict[str, List[Opportunity]]
    sector_capital_history: Dict[str, List[float]]

    # Crowding metrics
    _tier_invest_share: Dict[str, float]
    _sector_clearing_index: Dict[str, float]
```

### 8.2 Regime Transition Matrix

```python
REGIME_TRANSITIONS = {
    #              crisis  recession  normal  growth  boom
    "crisis":     [0.60,   0.30,      0.08,   0.02,   0.00],
    "recession":  [0.15,   0.50,      0.30,   0.05,   0.00],
    "normal":     [0.02,   0.10,      0.70,   0.15,   0.03],
    "growth":     [0.01,   0.05,      0.15,   0.65,   0.14],
    "boom":       [0.05,   0.05,      0.10,   0.30,   0.50],
}

# Regime effects on returns and failure
REGIME_MULTIPLIERS = {
    "crisis":    {"return": 0.4, "failure": 2.5},
    "recession": {"return": 0.7, "failure": 1.5},
    "normal":    {"return": 1.0, "failure": 1.0},
    "growth":    {"return": 1.3, "failure": 0.8},
    "boom":      {"return": 1.6, "failure": 0.6},
}
```

### 8.3 Sector Profiles

**Theoretical Grounding:** NVCA 2024 + BLS calibration.

```python
SECTOR_PROFILES = {
    "tech": {
        "innovation_probability": 0.25,    # NSF BRDIS R&D intensity
        "competition_intensity": 1.2,      # Census HHI-based
        "innovation_return_multiplier": (2.0, 4.0),
        "survival_rate_5yr": 0.52,         # BLS BED
        "capital_range": (2_000_000, 10_000_000),
    },
    "retail": {
        "innovation_probability": 0.08,
        "competition_intensity": 1.4,
        "innovation_return_multiplier": (1.3, 2.2),
        "survival_rate_5yr": 0.48,
        "capital_range": (1_400_000, 3_500_000),
    },
    # ... service, manufacturing
}
```

### 8.4 Crowding Penalty (Capacity-Convexity Model)

```python
def compute_crowding_penalty(self, sector: str, n_competitors: int) -> float:
    """
    Exponential decay penalty for market congestion.

    penalty = 1 - λ × (1 - exp(-γ × (n/K)²))

    Where:
    - K = capacity threshold (default 1.5)
    - γ = convexity parameter (default 2.0)
    - λ = maximum penalty strength (default 0.50)

    "When multiple agents pursue identical opportunities, crowding
    penalties reduce returns according to a congestion function."
    (Paper, p. 66)
    """
```

---

## 9. Simulation Engine

### 9.1 EmergentSimulation

**File:** `simulation.py`
**Theoretical Grounding:** ABM methodology per Gilbert (2008), Epstein & Axtell (1996), Crawford (2009).

```python
class EmergentSimulation:
    """
    Orchestrates the simulation across agents, market, and uncertainty environment.

    "ABMs enable researchers to create, analyze, and experiment with
    models composed of agents that interact within an environment."
    (Gilbert, 2008, cited in paper)
    """

    # Components
    config: EmergentConfig
    agents: List[EmergentAgent]
    market: MarketEnvironment
    uncertainty_env: KnightianUncertaintyEnvironment
    knowledge_base: KnowledgeBase
    innovation_engine: InnovationEngine
    information_system: EnhancedInformationSystem

    # Network
    agent_network: nx.Graph  # Small-world topology

    # Data collection
    data_buffer: Dict[str, List]
```

### 9.2 Simulation Protocol

```python
def run_simulation(self) -> Dict:
    """
    Executes n_rounds periods (default 120 = 10 years monthly).

    Per-round sequence:
    1. Market update: regime transition, opportunity discovery
    2. Uncertainty calculation: compute 4D state
    3. Agent loop:
       a. Decay resources (knowledge, operating costs)
       b. Evaluate AI tier (if emergent mode)
       c. Choose action (invest/innovate/explore/maintain)
       d. Execute action
       e. Process matured investments
       f. Update learning profiles
       g. Check failure conditions
    4. Data collection
    5. Buffer management

    "The simulation runs for 60 periods, representing monthly performance
    intervals over 5 years of operating history." (Paper, p. 69)
    """
```

### 9.3 Causal Identification

```python
# Fixed-tier assignment (main experiments)
def initialize_agents_fixed_tier(self, tier: str):
    """
    Random assignment to AI tier for causal identification.

    "By randomly allocating agents to AI tiers independent of their
    trait profiles, initial capital endowments, or sector assignments,
    the design eliminates selection effects." (Paper, p. 72)
    """

# Dynamic adoption (learning experiments)
def initialize_agents_dynamic(self):
    """
    Agents learn optimal AI tier via Thompson Sampling.

    "Agents can freely select and switch their AI tier based on
    observed performance outcomes, using Thompson Sampling with
    Beta distributions to learn which tiers perform best."
    (Paper, Table 7)
    """
```

### 9.4 Survival Evaluation

```python
def _evaluate_failure_conditions(self, agent: EmergentAgent) -> Tuple[bool, str]:
    """
    Multi-criterion failure assessment.

    Failure conditions:
    1. Capital exhaustion: capital < SURVIVAL_THRESHOLD ($230K)
    2. Equity erosion: capital < SURVIVAL_CAPITAL_RATIO × initial (38%)
    3. Persistent burn: negative cash flow for 9+ consecutive rounds

    "Agent survival is evaluated at each period against the survival
    threshold of $230,000, calibrated using Bureau of Labor Statistics
    data on business failure patterns." (Paper, p. 70)
    """
```

---

## 10. Key Variable Measurements

### 10.1 Primary Outcome Variables

| Variable | Measurement | Code Location | Paper Reference |
|----------|-------------|---------------|-----------------|
| **Survival Rate** | Fraction of agents with capital > threshold at round end | `simulation.py:compute_survival_rate()` | Tables 3A-C |
| **Capital Multiplier** | Final capital / Initial capital for survivors | `agents.py:get_capital_multiplier()` | Table 3F |
| **Innovation Volume** | Count of innovation attempts per agent | `innovation.py:get_innovation_metrics()` | Tables 3G-H |
| **Innovation Success Rate** | Successful / Total innovations | `innovation.py:innovation_success_by_ai` | Table 3I |
| **Niche Creation** | Total unique market niches created | `combination_tracker.total_combinations` | Tables 3J, 5E |

### 10.2 Behavioral Variables

| Variable | Measurement | Code Location | Paper Reference |
|----------|-------------|---------------|-----------------|
| **Innovation Share** | Innovation actions / Total actions | `analysis.py:compute_action_shares()` | Tables 3D, 5A |
| **Exploration Share** | Explore actions / Total actions | `analysis.py:compute_action_shares()` | Tables 3E, 5B |
| **AI Reliance** | AI-assisted decisions / Total decisions | `agents.py:ai_usage_history` | Dynamic adoption |

### 10.3 Uncertainty Variables

| Variable | Measurement | Code Location | Paper Reference |
|----------|-------------|---------------|-----------------|
| **Actor Ignorance** | Composite of unknown_opportunities, knowledge_gaps | `uncertainty.py:compute_actor_ignorance()` | Table 2b |
| **Practical Indeterminism** | Composite of path_volatility, timing_criticality | `uncertainty.py:compute_practical_indeterminism()` | Table 2b |
| **Agentic Novelty** | Composite of creative_momentum, disruption_potential | `uncertainty.py:compute_agentic_novelty()` | Table 2b |
| **Competitive Recursion** | Composite of herding_pressure, strategic_opacity | `uncertainty.py:compute_competitive_recursion()` | Table 2b |

### 10.4 Market Variables

| Variable | Measurement | Code Location |
|----------|-------------|---------------|
| **Competition Intensity** | Sector-specific HHI-calibrated intensity | `market.py:sector_competition_intensity` |
| **Crowding Penalty** | Capacity-convexity congestion function | `market.py:compute_crowding_penalty()` |
| **Regime State** | Current macroeconomic regime | `market.py:market_regime` |

---

## 11. Calibration & Empirical Grounding

### 11.1 Data Sources

| Parameter Domain | Source | Application |
|-----------------|--------|-------------|
| Survival rates | BLS Business Employment Dynamics | 5-year survival targets by sector |
| Capital requirements | NVCA 2024 | Initial capital distributions |
| Operating costs | BLS QCEW | Monthly burn rates by sector |
| Innovation rates | NSF BRDIS, USPTO | Sector R&D intensity |
| Return distributions | Kaplan & Schoar, Korteweg & Sorensen | Power law VC returns |
| Regime transitions | NBER Business Cycle Dating | Markov transition matrix |
| Competition intensity | Census Bureau HHI | Sector concentration |
| AI capabilities | Kaplan et al. (2020) scaling laws | Tier info_quality |

### 11.2 Calibration Targets

```python
CALIBRATION_TARGETS = {
    "survival_rate_5yr": 0.55,      # ±8% (BLS BED baseline)
    "mean_investment_return": 1.12,  # ±0.20 (PitchBook Series A-D)
    "innovation_activity_share": 0.40,  # ±10% (NVCA R&D intensity)
    "unicorn_rate": 0.01,           # Top 1% outcomes (power law)
}
```

### 11.3 Validation Approach

The model employs multiple validation strategies:

1. **Internal Validity:** Fixed-tier randomization eliminates selection effects
2. **External Validity:** Calibration to empirical benchmarks
3. **Robustness:** 15/15 robustness tests confirm findings (Table 4)
4. **Placebo Tests:** Permutation tests confirm causal attribution
5. **Refutation Tests:** 31 extreme conditions tested (Table 6)

---

## 12. References

### Core Theoretical Foundations

- **Townsend, D. M., Hunt, R. A., Rady, J., Manocha, P., & Jin, J. h. (2025a).** Are the futures computable? Knightian uncertainty and artificial intelligence. *Academy of Management Review*.

- **Townsend, D. M., Hunt, R. A., & Rady, J. (2024).** Chance, probability, & uncertainty at the edge of human reasoning: What is Knightian uncertainty? *Strategic Entrepreneurship Journal*.

- **Townsend, D. M., Hunt, R. A., McMullen, J. S., & Sarasvathy, S. D. (2018).** Uncertainty, knowledge problems, and entrepreneurial action. *Academy of Management Annals*, 12(2), 659-687.

### AI and Entrepreneurship

- **Shepherd, D. A., & Majchrzak, A. (2022).** Machines augmenting entrepreneurs: Opportunities (and threats) at the nexus of artificial intelligence and entrepreneurship. *Journal of Business Venturing*, 37(4), 106227.

- **Chalmers, D., MacKenzie, N. G., & Carter, S. (2021).** Artificial intelligence and entrepreneurship: Implications for venture creation in the fourth industrial revolution. *Entrepreneurship Theory and Practice*, 45(5), 1028-1053.

- **Kaplan, J., et al. (2020).** Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.

### Knowledge Recombination

- **Weitzman, M. L. (1998).** Recombinant growth. *The Quarterly Journal of Economics*, 113(2), 331-360.

- **Fleming, L. (2001).** Recombinant uncertainty in technological search. *Management Science*, 47(1), 117-132.

- **Hargadon, A., & Sutton, R. I. (1997).** Technology brokering and innovation in a product development firm. *Administrative Science Quarterly*, 716-749.

- **Katila, R., & Ahuja, G. (2002).** Something old, something new: A longitudinal study of search behavior and new product introduction. *Academy of Management Journal*, 45(6), 1183-1194.

### Opportunity Identification

- **Shane, S., & Venkataraman, S. (2000).** The promise of entrepreneurship as a field of research. *Academy of Management Review*, 25(1), 217-226.

- **Grégoire, D. A., & Shepherd, D. A. (2012).** Technology-market combinations and the identification of entrepreneurial opportunities. *Academy of Management Journal*, 55(4), 753-785.

- **Ramoglou, S., & McMullen, J. S. (2024).** What is an opportunity?: From theoretical mystification to everyday understanding. *Academy of Management Review*.

### ABM Methodology

- **Gilbert, N. (2008).** *Agent-Based Models*. Thousand Oaks, CA: SAGE Publications.

- **Epstein, J. M., & Axtell, R. (1996).** *Growing artificial societies: Social science from the bottom up*. Brookings Institution Press.

- **Crawford, G. C. (2009).** A review and recommendation of simulation methodologies for entrepreneurship research. *Available at SSRN 1472113*.

- **Bort, J., Wiklund, J., Crawford, G. C., Lerner, D. A., & Hunt, R. A. (2024).** The strategic advantage of impulsivity in entrepreneurial action: An agent-based modeling approach. *Entrepreneurship Theory and Practice*, 48(2), 547-580.

### Uncertainty and Decision-Making

- **Knight, F. H. (1921).** *Risk, Uncertainty and Profit*. Boston: Houghton Mifflin Company.

- **Sarasvathy, S. D. (2001).** Causation and effectuation: Toward a theoretical shift from economic inevitability to entrepreneurial contingency. *Academy of Management Review*, 26(2), 243-263.

- **McKelvie, A., Haynie, J. M., & Gustavsson, V. (2011).** Unpacking the uncertainty construct: Implications for entrepreneurial action. *Journal of Business Venturing*, 26(3), 273-292.

- **Wiltbank, R., Read, S., Dew, N., & Sarasvathy, S. D. (2009).** Prediction and control under uncertainty: Outcomes in angel investing. *Journal of Business Venturing*, 24(2), 116-133.

### Cognition and Learning

- **Mitchell, R. K., et al. (2002).** Toward a theory of entrepreneurial cognition: Rethinking the people side of entrepreneurship research. *Entrepreneurship Theory and Practice*, 27(2), 93-104.

- **Grégoire, D. A., Corbett, A. C., & McMullen, J. S. (2011).** The cognitive perspective in entrepreneurship: An agenda for future research. *Journal of Management Studies*, 48(6), 1443-1477.

- **Cohen, W. M., & Levinthal, D. A. (1990).** Absorptive capacity: A new perspective on learning and innovation. *Administrative Science Quarterly*, 128-152.

---

## Appendix: Quick Reference

### Running a Simulation

```python
from glimpse_abm.config import EmergentConfig
from glimpse_abm.simulation import EmergentSimulation

# Create configuration
config = EmergentConfig(
    N_AGENTS=1000,
    N_ROUNDS=120,
    N_RUNS=50,
)

# Run fixed-tier experiment
for tier in ["none", "basic", "advanced", "premium"]:
    sim = EmergentSimulation(config, fixed_ai_tier=tier)
    results = sim.run_simulation()
```

### Key Configuration Parameters

```python
# Agent parameters
N_AGENTS = 1000
INITIAL_CAPITAL = 5_000_000
SURVIVAL_THRESHOLD = 230_000

# AI tier costs (monthly)
AI_COSTS = {"none": 0, "basic": 30, "advanced": 400, "premium": 3500}

# Uncertainty weights
CROWDING_STRENGTH_LAMBDA = 0.50
CROWDING_CONVEXITY_GAMMA = 2.0

# Innovation parameters
INNOVATION_REUSE_PROBABILITY = 0.15
INNOVATION_REUSE_LOOKBACK = 300
```

### Output Data Structure

```python
results = {
    "decisions": [...],      # Per-agent, per-round decisions
    "market": [...],         # Market state time series
    "uncertainty": [...],    # 4D uncertainty time series
    "innovations": [...],    # Innovation attempts and outcomes
    "matured": [...],        # Investment realizations
    "summary": {...},        # Aggregate statistics
}
```

---

*This documentation accompanies the GlimpseABM codebase at https://github.com/cyborg-entrepreneur/Glimpse_ABM*
