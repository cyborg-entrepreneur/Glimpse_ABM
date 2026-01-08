# Parameter Glossary

This document provides a comprehensive reference for all configurable parameters in the Glimpse ABM simulation, including their theoretical justification, default values, and empirical calibration targets.

## Theoretical Foundation

This simulation operationalizes the theoretical framework from:

- **Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).** Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.

- **Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025).** Do androids dream of entrepreneurial possibilities? A reply to Ramoglou et al.'s "Artificial intelligence forces us to re-think Knightian uncertainty." *Academy of Management Review*, 50(2), 474-476.

- **Townsend, D. M., Hunt, R. A., & Rady, J. (2024).** Chance, probability, & uncertainty at the edge of human reasoning: What is Knightian uncertainty? *Strategic Entrepreneurship Journal*, 18(3), 451-474.

The simulation models entrepreneurial agents operating under the four dimensions of Knightian uncertainty identified in Townsend et al. (2025): actor ignorance, practical indeterminism, agentic novelty, and competitive recursion.

---

## 1. Agent Configuration

### Core Agent Parameters

| Parameter | Default | Range | Description | Theoretical Justification |
|-----------|---------|-------|-------------|---------------------------|
| `N_AGENTS` | 1000 | 50-5000 | Number of entrepreneurial agents | Provides sufficient statistical power for detecting AI tier effects while remaining computationally tractable |
| `INITIAL_CAPITAL` | 5,000,000 | 1M-20M | Starting capital per agent ($) | Reflects typical Series A venture capital, enabling meaningful investment decisions |
| `INITIAL_CAPITAL_RANGE` | (2.5M, 10M) | - | Heterogeneous capital distribution | Captures variance in entrepreneur resource endowments |
| `SURVIVAL_THRESHOLD` | 230,000 | 50K-500K | Minimum capital before insolvency ($) | Based on 6-12 months operating runway for typical ventures |
| `SURVIVAL_CAPITAL_RATIO` | 0.38 | 0.25-0.65 | Fraction of initial capital triggering survival pressure | Calibrated to BLS 5-year survival rates (~55% for venture cohorts) |
| `INSOLVENCY_GRACE_ROUNDS` | 7 | 3-12 | Rounds below threshold before exit | Reflects real-world runway management and bridge financing opportunities |

### Agent Trait Distributions

The simulation draws agent traits from specified probability distributions to capture heterogeneity in entrepreneurial cognition and capabilities.

| Trait | Distribution | Parameters | Theoretical Basis |
|-------|--------------|------------|-------------------|
| `uncertainty_tolerance` | Beta | a=1.05, b=0.65 | Tech entrepreneurs skew toward higher ambiguity tolerance; mean ~0.62 with optimistic tail |
| `innovativeness` | Lognormal | mean=0.5, σ=0.5 | Captures right-skewed distribution of innovative capacity in entrepreneur populations |
| `competence` | Uniform | [0.1, 0.8] | Broad distribution of general business competence |
| `ai_trust` | Normal (clipped) | μ=0.5, σ=0.38 | Heterogeneous initial AI trust with neutral mean |
| `exploration_tendency` | Beta | a=0.85, b=0.85 | U-shaped distribution: some agents explore aggressively, others exploit consistently |
| `entrepreneurial_drive` | Beta | a=2.2, b=1.8 | Right-skewed drive distribution reflecting entrepreneurial self-selection |

---

## 2. AI Tool Configuration

### AI Tier Specifications

The four AI tiers represent increasing levels of AI capability and cost, reflecting the current stratification of AI services available to entrepreneurs.

| Tier | Base Cost | Cost Type | Info Quality | Info Breadth | Per-Use Cost |
|------|-----------|-----------|--------------|--------------|--------------|
| `none` | $0 | - | 0.20 | 0.18 | $0 |
| `basic` | $45 | per_use | 0.48 | 0.38 | $6 |
| `advanced` | $1,500 | subscription | 0.78 | 0.68 | $60 |
| `premium` | $14,000 | subscription | 0.93 | 0.88 | $240 |

**Theoretical Justification**: The tier structure operationalizes the "paradox of future knowledge" from Townsend et al. (2025). Higher tiers provide better information quality (reducing actor ignorance) but at costs that affect capital allocation and with potential side effects on other uncertainty dimensions.

### AI Domain Capabilities

Each AI tier has different capabilities across analysis domains:

| Domain | Basic Accuracy | Advanced Accuracy | Premium Accuracy |
|--------|---------------|-------------------|------------------|
| `market_analysis` | 0.65 | 0.89 | 0.985 |
| `technical_assessment` | 0.66 | 0.91 | 0.992 |
| `uncertainty_evaluation` | 0.62 | 0.90 | 0.990 |
| `innovation_potential` | 0.60 | 0.89 | 0.988 |

**Hallucination Rates** (probability of false information):
- Basic: 0.17-0.20
- Advanced: 0.035-0.05
- Premium: 0.005-0.008

---

## 3. Knightian Uncertainty Parameters

### Actor Ignorance Parameters

| Parameter | Default | Description | Theoretical Link |
|-----------|---------|-------------|------------------|
| `DISCOVERY_PROBABILITY` | 0.30 | Base probability of discovering new opportunities | Controls rate of ignorance reduction through exploration |
| `KNOWLEDGE_DECAY_RATE` | 0.075 | Rate at which sector knowledge depreciates | Models obsolescence of entrepreneur's mental models |
| `MAX_AGENT_KNOWLEDGE` | 90 | Maximum knowledge pieces per agent | Bounded rationality constraint |

### Practical Indeterminism Parameters

| Parameter | Default | Description | Theoretical Link |
|-----------|---------|-------------|------------------|
| `MARKET_VOLATILITY` | 0.25 | Base market return volatility | Path dependency in execution outcomes |
| `MARKET_SHIFT_PROBABILITY` | 0.09 | Probability of regime transitions | Environmental contingency affecting timing |
| `BLACK_SWAN_PROBABILITY` | 0.05 | Probability of extreme events | Irreducible tail risks in entrepreneurship |

### Agentic Novelty Parameters

| Parameter | Default | Description | Theoretical Link |
|-----------|---------|-------------|------------------|
| `INNOVATION_PROBABILITY` | 0.42 | Base probability of innovation success | Rate of genuine novelty creation |
| `INNOVATION_REUSE_PROBABILITY` | 0.22 | Probability of reusing existing combinations | Tension between novelty and exploitation |
| `AI_NOVELTY_UPLIFT` | 0.08 | AI's contribution to novelty potential | Can be positive (facilitating combinations) or negative (anchoring on history) |

### Competitive Recursion Parameters

| Parameter | Default | Description | Theoretical Link |
|-----------|---------|-------------|------------------|
| `RECURSION_WEIGHTS.crowd_weight` | 0.35 | Weight of crowding in recursion calculation | Strategic interdependence from capital concentration |
| `RECURSION_WEIGHTS.volatility_weight` | 0.30 | Weight of volatility in recursion | Strategic uncertainty from market dynamics |
| `RECURSION_WEIGHTS.ai_herd_weight` | 0.40 | Weight of AI-induced herding | Correlated recommendations amplifying recursion |
| `RECURSION_WEIGHTS.premium_reuse_weight` | 0.20 | Weight of premium AI reuse patterns | High-tier AI convergence on similar strategies |

---

## 4. Market Dynamics Parameters

### Macro Regime Configuration

The simulation implements a Markov regime-switching model with five states reflecting macroeconomic conditions.

| Regime | Return Modifier | Failure Modifier | Volatility | Description |
|--------|-----------------|------------------|------------|-------------|
| `crisis` | 0.82 | 1.25 | 0.55 | Severe economic contraction |
| `recession` | 0.97 | 1.08 | 0.35 | Mild economic contraction |
| `normal` | 1.08 | 1.00 | 0.20 | Baseline economic conditions |
| `growth` | 1.25 | 0.88 | 0.25 | Economic expansion |
| `boom` | 1.45 | 0.72 | 0.30 | Strong economic expansion |

### Return and Risk Parameters

| Parameter | Default | Description | Calibration Target |
|-----------|---------|-------------|-------------------|
| `OPPORTUNITY_RETURN_RANGE` | (1.1, 25.0) | Range of possible investment multiples | Based on venture capital return distributions |
| `RETURN_OVERSUPPLY_PENALTY` | 0.52 | Penalty for crowded opportunities | Reflects diminishing returns to capital concentration |
| `RETURN_UNDERSUPPLY_BONUS` | 0.37 | Bonus for underexplored opportunities | Rewards contrarian strategies |
| `RETURN_DEMAND_CROWDING_THRESHOLD` | 0.42 | Threshold triggering crowding penalties | Calibrated to observed herding costs |
| `RETURN_LOWER_BOUND` | -1.0 | Minimum realized return (total loss) | Allows for complete investment wipeouts |

---

## 5. Sector Profiles

The simulation models four sectors with distinct risk-return characteristics reflecting real venture categories.

### Technology Sector

```yaml
return_range: [1.35, 3.10]
return_log_mu: ln(1.95)  # Log-mean of return distribution
return_log_sigma: 0.45   # Log-standard deviation
failure_range: [0.30, 0.50]
capital_range: [300,000, 1,200,000]
maturity_range: [15, 40] rounds
```

**Justification**: Later-stage venture-backed software/hardware with differentiated upside and higher failure rates.

### Retail Sector

```yaml
return_range: [1.15, 2.10]
return_log_mu: ln(1.45)
return_log_sigma: 0.32
failure_range: [0.20, 0.38]
capital_range: [50,000, 400,000]
maturity_range: [9, 30] rounds
```

**Justification**: Multi-unit retail concepts with moderate upside and higher operational variability.

### Service Sector

```yaml
return_range: [1.25, 2.20]
return_log_mu: ln(1.53)
return_log_sigma: 0.36
failure_range: [0.10, 0.28]
capital_range: [15,000, 200,000]
maturity_range: [6, 20] rounds
```

**Justification**: B2B/B2C recurring service ventures with low capex and resilient margins.

### Manufacturing Sector

```yaml
return_range: [1.30, 2.65]
return_log_mu: ln(1.78)
return_log_sigma: 0.40
failure_range: [0.25, 0.42]
capital_range: [250,000, 1,500,000]
maturity_range: [24, 72] rounds
```

**Justification**: Advanced manufacturing/industrial ventures with heavier capital loads and longer development cycles.

---

## 6. Calibration Profiles

### `venture_baseline_2024`

Anchors the simulation to US venture benchmarks.

**Target Metrics**:
- 5-year survival rate: 55% ± 8% (Source: BLS Business Employment Dynamics, 2019 cohort)
- Mean investment ROI: 1.12× ± 0.20 (Source: PitchBook cash-on-cash, Series A-D, 2020-2022)
- Innovation share: 40% ± 10% (Source: NVCA 2024 innovation activity estimates)

### `deeptech_capital_constrained`

Represents capital-intensive deep-tech ecosystems.

**Target Metrics**:
- 5-year survival rate: 35% ± 6% (Source: NVCA deep-tech survival analyses, 2017-2023)
- Mean investment ROI: 1.48× ± 0.25 (Source: CB Insights frontier tech benchmarks, 2023)
- Innovation share: 32% ± 8% (Source: OECD deep-tech investment mix, 2022)

---

## 7. Uncertainty Volatility Controls

These parameters control how uncertainty measurements evolve over time.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UNCERTAINTY_SHORT_WINDOW` | 6 | Rounds for short-term uncertainty averaging |
| `UNCERTAINTY_SHORT_DECAY` | 0.0 | Decay factor for short-term signals (0 = no decay) |
| `UNCERTAINTY_VOLATILITY_WINDOW` | 14 | Rounds for volatility measurement |
| `UNCERTAINTY_VOLATILITY_DECAY` | 0.6 | Exponential decay for volatility smoothing |
| `UNCERTAINTY_VOLATILITY_SCALING` | 0.45 | Scaling factor for volatility contribution |
| `UNCERTAINTY_CROWDING_WEIGHT` | 0.18 | Weight of capital crowding in uncertainty |
| `UNCERTAINTY_COMPETITIVE_WEIGHT` | 0.12 | Weight of competitive dynamics |
| `UNCERTAINTY_AI_HERDING_WEIGHT` | 0.10 | Weight of AI-induced herding patterns |

---

## 8. Performance and Execution Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ROUNDS` | 250 | Simulation rounds (≈ 5 years at quarterly frequency) |
| `N_RUNS` | 50 | Monte Carlo replications for statistical robustness |
| `RANDOM_SEED` | 42 | Base random seed for reproducibility |
| `use_parallel` | True | Enable parallel processing |
| `max_workers` | min(8, CPU-1) | Maximum parallel workers |

---

## Parameter Sensitivity Ranges

For Latin Hypercube Sampling (LHS) and sensitivity analysis, the following ranges are used:

| Parameter | Min | Max | Sensitivity |
|-----------|-----|-----|-------------|
| `BASE_OPERATIONAL_COST` | 52,000 | 65,000 | High |
| `SURVIVAL_CAPITAL_RATIO` | 0.28 | 0.36 | High |
| `INNOVATION_PROBABILITY` | 0.35 | 0.55 | Medium |
| `DISCOVERY_PROBABILITY` | 0.22 | 0.35 | Medium |
| `RECURSION_WEIGHTS.crowd_weight` | 0.20 | 0.50 | Medium |
| `AI_NOVELTY_UPLIFT` | 0.04 | 0.14 | Low-Medium |
| `RETURN_LOWER_BOUND` | -1.5 | -0.8 | Medium |

---

## References

Knight, F. H. (1921). *Risk, uncertainty, and profit*. Houghton Mifflin.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025). Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025). Do androids dream of entrepreneurial possibilities? A reply to Ramoglou et al.'s "Artificial intelligence forces us to re-think Knightian uncertainty." *Academy of Management Review*, 50(2), 474-476.

Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, & uncertainty at the edge of human reasoning: What is Knightian uncertainty? *Strategic Entrepreneurship Journal*, 18(3), 451-474.
