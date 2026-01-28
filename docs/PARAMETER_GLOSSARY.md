# Parameter Glossary

This document provides a comprehensive reference for all configurable parameters in the Glimpse ABM simulation, including their theoretical justification, default values, and empirical calibration targets.

**Important**: All parameters are calibrated for **monthly cadence** (120 rounds = 10 years).

## Theoretical Foundation

This simulation operationalizes the theoretical framework from:

- **Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).** Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.

- **Townsend, D. M., Hunt, R. A., & Rady, J. (2024).** Chance, probability, & uncertainty at the edge of human reasoning: What is Knightian uncertainty? *Strategic Entrepreneurship Journal*, 18(3), 451-474.

The simulation models entrepreneurial agents operating under the four dimensions of Knightian uncertainty: **actor ignorance**, **practical indeterminism**, **agentic novelty**, and **competitive recursion**.

---

## 1. Simulation Configuration

### Core Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_AGENTS` | 1000 | Number of entrepreneurial agents |
| `N_ROUNDS` | 120 | Simulation months (10 years) |
| `N_RUNS` | 50 | Monte Carlo replications per condition |
| `RANDOM_SEED` | 42 | Base random seed for reproducibility |

### Agent Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_CAPITAL` | 5,000,000 | Starting capital per agent ($) |
| `INITIAL_CAPITAL_RANGE` | (2.5M, 10M) | Heterogeneous capital distribution |
| `SURVIVAL_THRESHOLD` | 230,000 | Minimum capital before insolvency ($) |
| `SURVIVAL_CAPITAL_RATIO` | 0.38 | Fraction triggering survival pressure |
| `INSOLVENCY_GRACE_ROUNDS` | 7 | Months below threshold before exit |
| `BASE_OPERATIONAL_COST` | 16,667 | Monthly operating cost ($) |
| `COMPETITION_COST_MULTIPLIER` | 50.0 | Monthly competition cost scaling |

### Investment Parameters (Monthly)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_INVESTMENT_FRACTION` | 0.037 | Maximum monthly investment (% of capital) |
| `TARGET_INVESTMENT_FRACTION` | 0.033 | Target monthly investment (% of capital) |
| `LIQUIDITY_RESERVE_FRACTION` | 0.29 | Fraction kept as cash reserve |
| `DISCOVERY_PROBABILITY` | 0.10 | Monthly opportunity discovery rate |
| `INNOVATION_PROBABILITY` | 0.14 | Monthly innovation attempt rate |

---

## 2. AI Tool Configuration

### AI Tier Specifications

The four AI tiers represent increasing levels of AI capability and cost.

| Tier | Info Quality | Info Breadth | Monthly Cost | Per-Use Cost |
|------|-------------|--------------|--------------|--------------|
| `none` | 0.25 | 0.20 | $0 | $0 |
| `basic` | 0.43 | 0.38 | $30 | $3 |
| `advanced` | 0.70 | 0.65 | $400 | $35 |
| `premium` | 0.97 | 0.92 | $3,500 | $150 |

**Info Quality**: Accuracy of opportunity assessment (0 = random, 1 = perfect)
**Info Breadth**: Fraction of opportunities visible to agent

### AI Paradox Mechanism

The paradox emerges because:
- Higher `info_quality` → more accurate identification of "best" opportunities
- All Premium agents identify the same opportunities → convergent behavior
- Convergence → competitive crowding → diluted returns
- Result: Premium agents underperform None agents by 8-14 percentage points

### AI Domain Capabilities

| Domain | None | Basic | Advanced | Premium |
|--------|------|-------|----------|---------|
| `market_analysis` | 0.38 | 0.52 | 0.78 | 0.96 |
| `technical_assessment` | 0.40 | 0.54 | 0.80 | 0.97 |
| `uncertainty_evaluation` | 0.35 | 0.50 | 0.76 | 0.95 |
| `innovation_potential` | 0.33 | 0.48 | 0.74 | 0.94 |

### Hallucination Rates

| Tier | Rate | Description |
|------|------|-------------|
| None | 0.26-0.30 | Human baseline errors |
| Basic | 0.19-0.22 | Moderate AI errors |
| Advanced | 0.09-0.12 | Low error rate |
| Premium | 0.01-0.02 | Near-perfect accuracy |

### AI Subscription Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `AI_CREDIT_LINE_ROUNDS` | 90 | Months of AI credit before payment required |
| `AI_SUBSCRIPTION_AMORTIZATION_ROUNDS` | 180 | Months to amortize subscription costs |
| `AI_TIER_RECENT_WINDOW` | 45 | Months for recent performance evaluation |
| `AI_TRUST_RESERVE_DISCOUNT` | 0.25 | Discount for high-trust AI users |

---

## 3. Knightian Uncertainty Parameters

### Actor Ignorance

Controls how much agents don't know about opportunities.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DISCOVERY_PROBABILITY` | 0.10 | Monthly chance of finding new opportunities |
| `KNOWLEDGE_DECAY_RATE` | 0.075 | Monthly rate of knowledge obsolescence |
| `MAX_AGENT_KNOWLEDGE` | 90 | Maximum knowledge pieces per agent |

**Sector-Specific Knowledge Decay (Monthly)**:

| Sector | Decay Rate | Half-Life | Rationale |
|--------|------------|-----------|-----------|
| `tech` | 0.04 | 2-3 years | Fast obsolescence |
| `retail` | 0.023 | 4-5 years | Moderate turnover |
| `service` | 0.017 | 5-7 years | Stable expertise |
| `manufacturing` | 0.01 | 7-10 years | Durable process knowledge |

### Practical Indeterminism

Controls environmental uncertainty and shocks.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARKET_VOLATILITY` | 0.25 | Base return volatility |
| `MARKET_SHIFT_PROBABILITY` | 0.03 | Monthly regime transition probability |
| `BLACK_SWAN_PROBABILITY` | 0.017 | Monthly extreme event probability |

### Agentic Novelty

Controls innovation and creative destruction.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INNOVATION_PROBABILITY` | 0.14 | Monthly innovation attempt rate |
| `INNOVATION_REUSE_PROBABILITY` | 0.07 | Monthly probability of reusing combinations |
| `AI_NOVELTY_UPLIFT` | 0.08 | AI contribution to novelty (can be negative) |
| `NOVELTY_DISRUPTION_ENABLED` | true | Enable innovation disruption effects |
| `NOVELTY_DISRUPTION_THRESHOLD` | 0.6 | Novelty level triggering disruption |
| `NOVELTY_DISRUPTION_MAGNITUDE` | 0.25 | Disruption impact on existing investments |

**Sector-Specific Innovation (Monthly)**:

| Sector | Probability | Return Multiplier | Rationale |
|--------|------------|-------------------|-----------|
| `tech` | 0.16 | (2.0, 4.0) | High upside |
| `manufacturing` | 0.17 | (1.5, 2.8) | Incremental improvements |
| `service` | 0.13 | (1.6, 2.5) | Moderate returns |
| `retail` | 0.11 | (1.6, 2.5) | Moderate returns |

### Competitive Recursion

Controls strategic interdependence—the key driver of the AI paradox.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RECURSION_WEIGHTS.crowd_weight` | 0.35 | Weight of crowding in recursion |
| `RECURSION_WEIGHTS.volatility_weight` | 0.30 | Weight of market volatility |
| `RECURSION_WEIGHTS.ai_herd_weight` | 0.40 | Weight of AI-induced herding |
| `RECURSION_WEIGHTS.premium_reuse_weight` | 0.20 | Weight of premium convergence |

---

## 4. Competition and Crowding Parameters

These parameters are critical for reproducing the AI paradox.

### Capacity Constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPPORTUNITY_CAPACITY_ENABLED` | true | Enable capacity-based crowding |
| `OPPORTUNITY_BASE_CAPACITY` | 500,000 | Base investment capacity per opportunity |
| `OPPORTUNITY_CAPACITY_VARIANCE` | 0.3 | Variance in opportunity capacities |
| `CAPACITY_PENALTY_START` | 0.7 | Utilization threshold for penalties |
| `CAPACITY_PENALTY_MAX` | 0.4 | Maximum return penalty at full capacity |

### Competition Dynamics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DISABLE_COMPETITION_DYNAMICS` | false | Master switch for competition |
| `OPPORTUNITY_COMPETITION_PENALTY` | 0.5 | Return penalty from crowding |
| `OPPORTUNITY_COMPETITION_THRESHOLD` | 0.2 | Crowding level triggering penalties |
| `OPPORTUNITY_COMPETITION_FLOOR` | 0.1 | Minimum return multiplier |
| `COMPETITION_INTENSITY` | 1.0 | Global competition scaling (0-2) |

### Sequential Decision Making

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQUENTIAL_DECISIONS_ENABLED` | true | Enable early/late mover dynamics |
| `EARLY_DECISION_FRACTION` | 0.3 | Fraction making early decisions |
| `SIGNAL_VISIBILITY_WEIGHT` | 0.15 | Weight of early signals for late movers |

---

## 5. Market Regime Configuration

### Macro Regime States

The simulation implements a Markov regime-switching model calibrated from NBER Business Cycle data (1945-2024).

| Regime | Return Modifier | Failure Modifier | Volatility |
|--------|-----------------|------------------|------------|
| `crisis` | 0.85 | 1.18 | 0.55 |
| `recession` | 0.98 | 1.05 | 0.35 |
| `normal` | 1.10 | 1.00 | 0.20 |
| `growth` | 1.28 | 0.90 | 0.25 |
| `boom` | 1.45 | 0.78 | 0.30 |

### Monthly Transition Matrix

Converted from quarterly NBER data using: `p_monthly = 1 - (1-p_quarterly)/3`

| From / To | crisis | recession | normal | growth | boom |
|-----------|--------|-----------|--------|--------|------|
| **crisis** | 0.75 | 0.12 | 0.12 | 0.01 | 0.00 |
| **recession** | 0.01 | 0.77 | 0.17 | 0.04 | 0.01 |
| **normal** | 0.003 | 0.027 | 0.85 | 0.09 | 0.03 |
| **growth** | 0.003 | 0.01 | 0.07 | 0.84 | 0.077 |
| **boom** | 0.003 | 0.013 | 0.067 | 0.13 | 0.787 |

---

## 6. Sector Profiles (Monthly Cadence)

### Technology Sector

| Parameter | Value | Source |
|-----------|-------|--------|
| `initial_capital_range` | ($3M, $6M) | NVCA 2024 |
| `operational_cost_range` | ($20k, $30k)/month | BLS QCEW |
| `survival_threshold` | $150,000 | BLS BED |
| `innovation_probability` | 0.16/month | NSF BRDIS |
| `knowledge_decay_rate` | 0.04/month | Skill research |
| `competition_intensity` | 1.2 | Census HHI |

### Retail Sector

| Parameter | Value | Source |
|-----------|-------|--------|
| `initial_capital_range` | ($2.2M, $4M) | NVCA 2024 |
| `operational_cost_range` | ($13k, $23k)/month | BLS QCEW |
| `survival_threshold` | $130,000 | BLS BED |
| `innovation_probability` | 0.11/month | NSF BRDIS |
| `knowledge_decay_rate` | 0.023/month | Skill research |
| `competition_intensity` | 0.7 | Census HHI |

### Service Sector

| Parameter | Value | Source |
|-----------|-------|--------|
| `initial_capital_range` | ($1.4M, $2.5M) | NVCA 2024 |
| `operational_cost_range` | ($8.3k, $15k)/month | BLS QCEW |
| `survival_threshold` | $70,000 | BLS BED |
| `innovation_probability` | 0.13/month | NSF BRDIS |
| `knowledge_decay_rate` | 0.017/month | Skill research |
| `competition_intensity` | 0.9 | Census HHI |

### Manufacturing Sector

| Parameter | Value | Source |
|-----------|-------|--------|
| `initial_capital_range` | ($4M, $7.5M) | NVCA 2024 |
| `operational_cost_range` | ($26.7k, $40k)/month | BLS QCEW |
| `survival_threshold` | $200,000 | BLS BED |
| `innovation_probability` | 0.17/month | NSF BRDIS |
| `knowledge_decay_rate` | 0.01/month | Skill research |
| `competition_intensity` | 1.4 | Census HHI |

---

## 7. Learning and Adaptation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE_BASE` | 0.033 | Monthly learning rate |
| `EXPLORATION_DECAY` | 0.9983 | Monthly exploration decay |
| `COST_OF_CAPITAL` | 0.005 | Monthly cost of capital (~6% annual) |
| `SOCIAL_LEARNING_WEIGHT` | 0.2 | Weight of peer learning |

---

## 8. Uncertainty Volatility Controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UNCERTAINTY_SHORT_WINDOW` | 18 | Months for short-term averaging |
| `UNCERTAINTY_VOLATILITY_WINDOW` | 42 | Months for volatility measurement |
| `UNCERTAINTY_VOLATILITY_DECAY` | 0.87 | Monthly decay for smoothing |
| `UNCERTAINTY_CROWDING_WEIGHT` | 0.18 | Weight of capital crowding |
| `UNCERTAINTY_COMPETITIVE_WEIGHT` | 0.12 | Weight of competitive dynamics |

---

## 9. Robustness Test Parameters

These parameters allow systematic isolation of paradox mechanisms:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HALLUCINATION_INTENSITY` | 1.0 | Scale AI hallucination rates (0-1) |
| `OVERCONFIDENCE_INTENSITY` | 1.0 | Scale AI overconfidence (0-1) |
| `AI_NOVELTY_CONSTRAINT_INTENSITY` | 1.0 | Scale AI novelty constraints (0-1) |
| `AI_COST_INTENSITY` | 1.0 | Scale AI costs (0-1) |
| `COMPETITION_INTENSITY` | 1.0 | Scale competition effects (0-1) |

---

## Parameter Sensitivity Ranges

For Latin Hypercube Sampling (LHS) and sensitivity analysis:

| Parameter | Min | Max | Sensitivity |
|-----------|-----|-----|-------------|
| `BASE_OPERATIONAL_COST` | 12,000 | 22,000 | High |
| `SURVIVAL_CAPITAL_RATIO` | 0.28 | 0.48 | High |
| `INNOVATION_PROBABILITY` | 0.10 | 0.18 | Medium |
| `DISCOVERY_PROBABILITY` | 0.07 | 0.13 | Medium |
| `CAPACITY_PENALTY_MAX` | 0.2 | 0.6 | High |
| `AI_NOVELTY_UPLIFT` | 0.04 | 0.14 | Low-Medium |

---

## References

Knight, F. H. (1921). *Risk, uncertainty, and profit*. Houghton Mifflin.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025). Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.

Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, & uncertainty at the edge of human reasoning: What is Knightian uncertainty? *Strategic Entrepreneurship Journal*, 18(3), 451-474.
