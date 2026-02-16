# GlimpseABM.jl

A high-performance Julia agent-based model for studying the **AI Information Paradox**—the phenomenon whereby widespread AI adoption can paradoxically amplify Knightian uncertainty through competitive convergence.

## Overview

GlimpseABM.jl simulates entrepreneurial decision-making under uncertainty, examining how AI augmentation transforms rather than eliminates uncertainty. The model demonstrates that superior AI capabilities can lead to worse collective outcomes due to convergent behavior and competitive crowding.

### The AI Information Paradox

The central finding of this simulation is counterintuitive: **agents with the best AI tools (Premium tier) consistently underperform agents with no AI assistance**. This paradox emerges through a specific causal mechanism:

1. **Superior Information Quality**: Premium AI provides near-perfect information about opportunity quality (info_quality = 0.97)
2. **Convergent Decision-Making**: All Premium agents identify the same "best" opportunities
3. **Competitive Crowding**: These opportunities become overcrowded as many agents invest simultaneously
4. **Diluted Returns**: Crowding penalties reduce returns, transforming "good" investments into mediocre ones
5. **Survival Penalty**: Lower returns lead to capital depletion and higher failure rates

Meanwhile, agents without AI ("None" tier) make more diverse, seemingly "worse" decisions that are distributed across the opportunity landscape. Some find uncrowded niches with superior returns, leading to better collective outcomes.

### Key Findings

The simulation consistently reproduces:
- **Premium AI agents exhibit ~8-14 percentage point lower survival** than human-only agents
- **Mechanism**: Better information → convergent decisions → competitive crowding → diluted returns
- **Unicorn generation**: Premium tier produces ~50% fewer exceptional performers (top 5% by capital)
- **Competition levels**: Premium agents experience 2-3x higher mean competition ratios
- Effects are robust across bootstrap tests, placebo checks, and balanced designs

## Theoretical Foundation

The model operationalizes four dimensions of Knightian uncertainty from Townsend et al. (2025, Academy of Management Review):

### 1. Actor Ignorance
The entrepreneur's incomplete knowledge of the opportunity landscape, market conditions, and causal relationships.

- **Operationalization**: `info_quality` and `info_breadth` parameters determine how accurately agents perceive opportunity characteristics
- **AI Effect**: Higher AI tiers dramatically reduce actor ignorance (None: 0.25 → Premium: 0.97)
- **Measurement**: Tracked via `actor_ignorance` in agent uncertainty metrics

### 2. Practical Indeterminism
The inherent unpredictability of execution paths, timing dependencies, and path-dependent outcomes.

- **Operationalization**: Stochastic shocks to investment outcomes, market regime transitions, and black swan events
- **AI Effect**: AI slightly reduces execution uncertainty through better planning
- **Measurement**: Tracked via `practical_indeterminism` derived from outcome variance

### 3. Agentic Novelty
The creative potential for genuinely new combinations, innovations, and possibilities that did not previously exist.

- **Operationalization**: Innovation mechanics allow agents to create new opportunities through knowledge recombination
- **AI Effect**: AI may constrain novelty by anchoring on historical patterns (configurable via `AI_NOVELTY_CONSTRAINT_INTENSITY`)
- **Measurement**: Tracked via `agentic_novelty` based on innovation success rates

### 4. Competitive Recursion
The strategic interdependence among actors where each agent's actions depend on anticipations of others' actions.

- **Operationalization**: Competition ratio measured at investment maturity, crowding penalties on returns
- **AI Effect**: This is where the paradox emerges—better AI increases competitive recursion through convergent behavior
- **Measurement**: Tracked via `competitive_recursion` and `competition_levels` arrays

### The Paradox Mechanism

The key insight is that AI creates a **trade-off between uncertainty dimensions**:

```
Actor Ignorance ↓↓↓ (AI dramatically reduces)
Competitive Recursion ↑↑↑ (AI dramatically increases via convergence)
```

When all agents have excellent information, they make similar decisions. This similarity creates a new form of uncertainty—not about what opportunities are good, but about how many competitors will arrive at the same conclusion.

## Model Architecture

### Simulation Cadence

The simulation operates on **monthly cadence**:
- `N_ROUNDS = 120` represents 10 years of operation
- All rate parameters (costs, probabilities, decay rates) are calibrated for monthly resolution
- Investment maturity periods range from 3-28 months depending on sector

### Agents

Each `EmergentAgent` represents an entrepreneurial firm characterized by:

```julia
struct EmergentAgent
    id::Int
    alive::Bool
    capital::Float64                    # Current capital
    fixed_ai_level::Union{String,Nothing}  # Locked AI tier for experiments
    current_ai_level::String            # Current AI tier
    portfolio::InvestmentPortfolio      # Active and pending investments
    uncertainty_metrics::AgentUncertaintyMetrics  # Four-dimensional tracking
    innovation_count::Int               # Successful innovations
    success_count::Int                  # Successful investments
    failure_count::Int                  # Failed investments
    total_invested::Float64             # Cumulative investment
    total_returned::Float64             # Cumulative returns
    # ... additional state
end
```

**Agent Decision Cycle** (each round):
1. **Opportunity Discovery**: Agents discover available investment opportunities
2. **Information Gathering**: AI tier determines quality of opportunity assessment
3. **Investment Decision**: Agents allocate capital based on perceived returns
4. **Innovation Attempt**: Agents may attempt to create new opportunities
5. **Outcome Resolution**: Matured investments resolve with competition-adjusted returns
6. **Survival Check**: Agents below survival threshold may exit

### Market Environment

The `MarketEnvironment` manages:

```julia
struct MarketEnvironment
    opportunities::Vector{Opportunity}  # Available investments
    market_regime::String              # "crisis"|"recession"|"normal"|"growth"|"boom"
    market_momentum::Float64           # Trend indicator
    crowding_metrics::Dict             # Competition tracking
    # ... additional state
end
```

**Opportunity Characteristics**:
- `latent_return_potential`: True underlying return (hidden from agents)
- `latent_failure_potential`: True failure probability (hidden from agents)
- `capacity`: Maximum investment before crowding penalties
- `total_invested`: Current investment level
- `competition_ratio`: Number of investors / capacity

**Crowding Mechanism**:
```julia
# When capacity utilization exceeds threshold, returns are penalized
if utilization > CAPACITY_PENALTY_START
    excess = (utilization - CAPACITY_PENALTY_START) / (1.0 - CAPACITY_PENALTY_START)
    penalty = excess * CAPACITY_PENALTY_MAX
    adjusted_return = base_return * (1.0 - penalty)
end
```

### AI Tiers

| Tier | Info Quality | Info Breadth | Monthly Cost | Description |
|------|-------------|--------------|--------------|-------------|
| None | 0.25 | 0.20 | $0 | Human baseline - high variance, diverse decisions |
| Basic | 0.50 | 0.40 | $500 | Consumer AI - moderate improvement |
| Advanced | 0.75 | 0.65 | $1,500 | Professional AI - significant improvement |
| Premium | 0.97 | 0.92 | $3,500 | Enterprise AI - near-perfect information |

**Info Quality**: Determines accuracy of opportunity assessment (0 = random, 1 = perfect)
**Info Breadth**: Determines fraction of opportunities visible to agent

### Competition Ratio Calculation

The competition ratio is the key metric driving the paradox:

```julia
function calculate_competition_ratio(opportunity, market)
    n_investors = count_investors(opportunity)
    capacity = opportunity.capacity

    # Competition ratio: how crowded is this opportunity?
    competition_ratio = n_investors / max(capacity, 1)

    # Record for agent uncertainty metrics
    push!(agent.uncertainty_metrics.competition_levels, competition_ratio)

    return competition_ratio
end
```

Premium agents consistently experience `competition_ratio > 10` while None agents average `competition_ratio ~ 3-5`.

### Investment Outcome Resolution

When investments mature:

```julia
function resolve_investment(investment, agent, market)
    # Base return from opportunity characteristics
    base_return = investment.opportunity.latent_return_potential

    # Competition penalty
    competition_ratio = calculate_competition_ratio(investment.opportunity, market)
    if competition_ratio > COMPETITION_THRESHOLD
        crowding_penalty = min((competition_ratio - COMPETITION_THRESHOLD) * PENALTY_RATE, MAX_PENALTY)
        base_return *= (1.0 - crowding_penalty)
    end

    # Stochastic outcome
    if rand() < failure_probability
        return investment.amount * FAILURE_RECOVERY_RATE
    else
        return investment.amount * base_return
    end
end
```

## Running Simulations

### Quick Start

```julia
using GlimpseABM

# Create configuration (monthly cadence)
config = EmergentConfig(
    N_AGENTS = 1000,
    N_ROUNDS = 60,          # 5 years
    INITIAL_CAPITAL = 100_000_000.0,
    SURVIVAL_THRESHOLD = 10_000.0
)

# Run simulation with fixed AI tier (for causal analysis)
tier_dist = Dict("none" => 0.0, "basic" => 0.0, "advanced" => 0.0, "premium" => 1.0)
sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)

for round in 1:60
    GlimpseABM.step!(sim, round)
end

# Analyze results
alive = count(a -> a.alive, sim.agents)
println("Survival rate: $(alive)/$(config.N_AGENTS)")

# Check competition levels
mean_cr = mean([mean(a.uncertainty_metrics.competition_levels)
                for a in sim.agents
                if !isempty(a.uncertainty_metrics.competition_levels)])
println("Mean competition ratio: $mean_cr")
```

### Complete Paradox Analysis

The main analysis script runs 200 simulations (50 per AI tier) with comprehensive mechanism tests:

```bash
julia --threads=auto --project=. scripts/run_complete_paradox_suite.jl
```

This produces:

**Primary Outcomes**:
- Survival rates by AI tier with bootstrap confidence intervals
- Average Treatment Effect (ATE) of Premium vs None
- Cohen's d effect sizes

**Mechanism Analysis**:
- Competition ratio distributions by tier
- Correlation between competition and survival
- Within-tier competition quartile analysis
- Investment convergence metrics

**Robustness Checks**:
- Placebo tests (shuffled tier assignments)
- Balanced design (equal tier sizes)
- Benjamini-Hochberg multiple comparison correction
- Bootstrap standard errors

**Financial Outcomes**:
- ROI distributions by tier
- Unicorn rates (top 5% performers)
- Capital accumulation trajectories

## Package Structure

```
GlimpseABM.jl/
├── src/
│   ├── GlimpseABM.jl      # Main module and exports
│   ├── config.jl          # EmergentConfig with all parameters
│   ├── models.jl          # Opportunity, Investment, Portfolio structs
│   ├── agents.jl          # EmergentAgent and decision logic
│   ├── market.jl          # MarketEnvironment and crowding
│   ├── simulation.jl      # EmergentSimulation orchestration
│   ├── uncertainty.jl     # Four-dimensional uncertainty tracking
│   ├── innovation.jl      # Innovation and novelty mechanics
│   ├── information.jl     # AI information processing
│   ├── knowledge.jl       # Knowledge accumulation and decay
│   ├── io.jl              # Save/load utilities
│   ├── numpy_rng.jl       # Cross-language RNG compatibility
│   └── utils.jl           # Helper functions
├── scripts/
│   └── run_complete_paradox_suite.jl  # Main analysis script
├── test/
│   └── runtests.jl        # Test suite (44 tests)
├── Project.toml           # Dependencies
├── Manifest.toml          # Locked versions
└── README.md
```

## Key Parameters

### Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_AGENTS` | 1000 | Agents per simulation |
| `N_ROUNDS` | 120 | Months (10 years) |
| `INITIAL_CAPITAL` | 100M | Starting capital |
| `SURVIVAL_THRESHOLD` | 10K | Minimum viable capital |
| `BASE_OPERATIONAL_COST` | 16,667 | Monthly operating cost |

### Competition Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `OPPORTUNITY_CAPACITY_ENABLED` | true | Enable capacity constraints |
| `OPPORTUNITY_BASE_CAPACITY` | 500,000 | Base capacity per opportunity |
| `CAPACITY_PENALTY_START` | 0.7 | Utilization threshold for penalties |
| `CAPACITY_PENALTY_MAX` | 0.4 | Maximum return penalty |
| `OPPORTUNITY_COMPETITION_PENALTY` | 0.5 | Additional competition penalty |

### Innovation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `INNOVATION_PROBABILITY` | 0.14 | Monthly innovation attempt rate |
| `NOVELTY_DISRUPTION_ENABLED` | true | Enable innovation disruption |
| `NOVELTY_DISRUPTION_THRESHOLD` | 0.6 | Novelty level for disruption |

## Reproducing Key Results

### The Paradox Effect

```julia
using GlimpseABM
using Statistics

# Run comparative analysis
results = Dict()
for tier in ["none", "premium"]
    survivals = Float64[]
    for run in 1:50
        config = EmergentConfig(N_AGENTS=1000, N_ROUNDS=60, RANDOM_SEED=42+run)
        tier_dist = Dict(t => (t == tier ? 1.0 : 0.0) for t in ["none","basic","advanced","premium"])
        sim = EmergentSimulation(config=config, initial_tier_distribution=tier_dist)
        for r in 1:60; GlimpseABM.step!(sim, r); end
        push!(survivals, count(a -> a.alive, sim.agents) / 1000)
    end
    results[tier] = (mean=mean(survivals), std=std(survivals))
end

println("None survival: $(results["none"].mean) ± $(results["none"].std)")
println("Premium survival: $(results["premium"].mean) ± $(results["premium"].std)")
println("Paradox effect: $(results["none"].mean - results["premium"].mean) pp")
```

Expected output:
```
None survival: 0.72 ± 0.03
Premium survival: 0.58 ± 0.04
Paradox effect: 0.14 pp
```

## Requirements

- Julia 1.9+
- Multi-threading recommended: `julia --threads=auto`
- Dependencies: DataFrames, Distributions, StatsBase, CairoMakie, CSV

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

All 44 tests should pass, covering configuration, models, agents, market, uncertainty, and simulation components.

## License

MIT License - see LICENSE file.

## Citation

```bibtex
@article{townsend2026flux,
  title={Into the Flux: {AI} Augmentation \& The Paradox of Future Knowledge},
  author={Townsend, David M. and Hunt, Richard A. and Rady, Judy},
  journal={Entrepreneurship Theory and Practice},
  year={2026},
  note={Forthcoming}
}

@software{glimpse_abm,
  title = {GlimpseABM: Agent-Based Model of the AI Information Paradox},
  author = {Townsend, David M. and Hunt, Richard A. and Rady, Judy},
  year = {2026},
  url = {https://github.com/cyborg-entrepreneur/Glimpse_ABM}
}
```

## References

Townsend, D. M., Hunt, R. A., Rady, R., Manocha, P., & Jin, J-H. (2025). Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.

Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, & uncertainty at the edge of human reasoning: What is Knightian uncertainty? *Strategic Entrepreneurship Journal*, 18(3), 451-474.
