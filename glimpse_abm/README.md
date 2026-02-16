# Glimpse ABM: Agent-Based Simulation of Knightian Uncertainty Under AI Augmentation

A comprehensive agent-based model for studying the **AI Information Paradox**—the phenomenon whereby widespread AI adoption can paradoxically amplify Knightian uncertainty through competitive convergence.

## Overview

GLIMPSE ABM simulates entrepreneurial decision-making under uncertainty, examining how AI augmentation transforms rather than eliminates uncertainty. The model demonstrates that superior AI capabilities can lead to worse collective outcomes due to convergent behavior and competitive crowding.

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

---

## Implementations

This repository contains **two implementations** of the GLIMPSE ABM:

| Implementation | Language | Performance | Use Case |
|----------------|----------|-------------|----------|
| **Python** (root) | Python 3.8+ | ~70k agent-rounds/s | Research, prototyping, analysis |
| **Julia** (`julia/`) | Julia 1.9+ | ~5M agent-rounds/s | Large-scale sweeps, production |

Both implementations produce statistically equivalent results. The Julia version is **~65x faster** and recommended for parameter sweeps and robustness analysis.

---

## Theoretical Foundation

### The Four Problems of Knightian Uncertainty

Knight (1921) distinguished between calculable *risk* and incalculable *uncertainty*. Townsend et al. (2025) extend this framework to identify four irreducible sources of Knightian uncertainty affecting entrepreneurial action:

#### 1. Actor Ignorance
The entrepreneur's incomplete knowledge of the opportunity landscape, market conditions, and causal relationships.

- **Operationalization**: `info_quality` and `info_breadth` parameters determine how accurately agents perceive opportunity characteristics
- **AI Effect**: Higher AI tiers dramatically reduce actor ignorance (None: 0.25 → Premium: 0.97)
- **Measurement**: Tracked via `actor_ignorance` in agent uncertainty metrics

#### 2. Practical Indeterminism
The inherent unpredictability of execution paths, timing dependencies, and path-dependent outcomes.

- **Operationalization**: Stochastic shocks to investment outcomes, market regime transitions, and black swan events
- **AI Effect**: AI slightly reduces execution uncertainty through better planning
- **Measurement**: Tracked via `practical_indeterminism` derived from outcome variance

#### 3. Agentic Novelty
The creative potential for genuinely new combinations, innovations, and possibilities that did not previously exist.

- **Operationalization**: Innovation mechanics allow agents to create new opportunities through knowledge recombination
- **AI Effect**: AI may constrain novelty by anchoring on historical patterns (configurable via `AI_NOVELTY_CONSTRAINT_INTENSITY`)
- **Measurement**: Tracked via `agentic_novelty` based on innovation success rates

#### 4. Competitive Recursion
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

---

## Model Architecture

### Simulation Cadence

The simulation operates on **monthly cadence**:
- `N_ROUNDS = 120` represents 10 years of operation
- All rate parameters (costs, probabilities, decay rates) are calibrated for monthly resolution
- Investment maturity periods range from 3-28 months depending on sector

### Agents

Each `EmergentAgent` represents an entrepreneurial firm characterized by:

```python
class EmergentAgent:
    id: int
    alive: bool
    capital: float                      # Current capital
    fixed_ai_level: Optional[str]       # Locked AI tier for experiments
    current_ai_level: str               # Current AI tier
    portfolio: InvestmentPortfolio      # Active and pending investments
    uncertainty_metrics: dict           # Four-dimensional tracking
    innovation_count: int               # Successful innovations
    success_count: int                  # Successful investments
    failure_count: int                  # Failed investments
    total_invested: float               # Cumulative investment
    total_returned: float               # Cumulative returns
    traits: dict                        # Cognitive traits (10 dimensions)
```

**Agent Decision Cycle** (each round):
1. **Opportunity Discovery**: Agents discover available investment opportunities
2. **Information Gathering**: AI tier determines quality of opportunity assessment
3. **Investment Decision**: Agents allocate capital based on perceived returns
4. **Innovation Attempt**: Agents may attempt to create new opportunities
5. **Outcome Resolution**: Matured investments resolve with competition-adjusted returns
6. **Survival Check**: Agents below survival threshold may exit

### Cognitive Traits

Each agent is initialized with 10 cognitive traits sampled from empirically-calibrated distributions:

| Trait | Distribution | Mean | Behavioral Effect |
|-------|-------------|------|-------------------|
| Uncertainty Tolerance | Beta(1.05, 0.65) | 0.62 | Investment thresholds, AI adoption |
| Innovativeness | Lognormal(0.5, 0.5) | 0.50 | Innovation attempts, search breadth |
| Competence | Uniform(0.1, 0.8) | 0.45 | Operating costs, learning rate |
| AI Trust | Normal(0.5, 0.38) | 0.50 | AI recommendation weight |
| Exploration Tendency | Beta(0.85, 0.85) | 0.50 | Portfolio diversification |
| Entrepreneurial Drive | Beta(2.2, 1.8) | 0.55 | Survival persistence |
| Trait Momentum | Uniform(0.6, 0.9) | 0.75 | Learning adaptation rate |
| Cognitive Style | Uniform(0.8, 1.2) | 1.00 | Information processing |
| Analytical Ability | Uniform(0.1, 0.9) | 0.50 | Herding resistance |
| Market Awareness | Uniform(0.1, 0.9) | 0.50 | Opportunity discovery |

See `docs/TABLE1_COGNITIVE_TRAITS.md` for detailed documentation with academic citations.

### Market Environment

The `MarketEnvironment` manages:

```python
class MarketEnvironment:
    opportunities: List[Opportunity]    # Available investments
    market_regime: str                  # "crisis"|"recession"|"normal"|"growth"|"boom"
    market_momentum: float              # Trend indicator
    crowding_metrics: dict              # Competition tracking
```

**Opportunity Characteristics**:
- `latent_return_potential`: True underlying return (hidden from agents)
- `latent_failure_potential`: True failure probability (hidden from agents)
- `capacity`: Maximum investment before crowding penalties
- `total_invested`: Current investment level
- `competition_ratio`: Number of investors / capacity

**Crowding Mechanism**:
```python
# When capacity utilization exceeds threshold, returns are penalized
if utilization > CAPACITY_PENALTY_START:
    excess = (utilization - CAPACITY_PENALTY_START) / (1.0 - CAPACITY_PENALTY_START)
    penalty = excess * CAPACITY_PENALTY_MAX
    adjusted_return = base_return * (1.0 - penalty)
```

### AI Tiers

| Tier | Info Quality | Info Breadth | Monthly Cost | Description |
|------|-------------|--------------|--------------|-------------|
| None | 0.25 | 0.20 | $0 | Human baseline - high variance, diverse decisions |
| Basic | 0.43 | 0.38 | $30 | Consumer AI - moderate improvement |
| Advanced | 0.70 | 0.65 | $400 | Professional AI - significant improvement |
| Premium | 0.97 | 0.92 | $3,500 | Enterprise AI - near-perfect information |

**Info Quality**: Determines accuracy of opportunity assessment (0 = random, 1 = perfect)
**Info Breadth**: Determines fraction of opportunities visible to agent

### Competition Ratio Calculation

The competition ratio is the key metric driving the paradox:

```python
def calculate_competition_ratio(opportunity, market):
    n_investors = count_investors(opportunity)
    capacity = opportunity.capacity

    # Competition ratio: how crowded is this opportunity?
    competition_ratio = n_investors / max(capacity, 1)

    # Record for agent uncertainty metrics
    agent.uncertainty_metrics['competition_levels'].append(competition_ratio)

    return competition_ratio
```

Premium agents consistently experience `competition_ratio > 10` while None agents average `competition_ratio ~ 3-5`.

---

## Repository Structure

```
glimpse_abm/
├── __init__.py              # Python package initialization
├── cli.py                   # Command-line interface
├── config.py                # Configuration and calibration (monthly cadence)
├── simulation.py            # Core simulation orchestration
├── agents.py                # Agent architecture and behavior
├── market.py                # Market dynamics and crowding
├── uncertainty.py           # Four-dimensional Knightian uncertainty
├── innovation.py            # Innovation and opportunity discovery
├── knowledge.py             # Knowledge management and AI augmentation
├── information.py           # Information systems and signals
├── models.py                # Data models and structures
├── analysis.py              # Results analysis and statistics
├── statistical_tests.py     # Causal inference utilities
├── causal/                  # Advanced causal analysis (DiD, PSM, RD)
├── docs/                    # Documentation and parameter guides
│   ├── PARAMETER_GLOSSARY.md      # Complete parameter reference
│   ├── CALIBRATION_GUIDE.md       # Calibration instructions
│   ├── TABLE1_COGNITIVE_TRAITS.md # Cognitive traits for paper
│   └── CALIBRATION_BASELINE.md    # Calibration workbook
├── parameters/              # LHS parameter range configurations
├── tests/                   # Test suite
└── julia/                   # Julia implementation (GlimpseABM.jl)
    ├── src/                 # Core Julia modules
    ├── scripts/             # Analysis scripts
    ├── test/                # Julia test suite
    └── README.md            # Julia-specific documentation
```

---

## Installation

### Python (Default)

```bash
# Clone the repository
git clone https://github.com/cyborg-entrepreneur/Glimpse_ABM.git
cd Glimpse_ABM

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn tqdm pyyaml

# Verify installation
python3 -m glimpse_abm.cli --help
```

### Julia (High-Performance)

```bash
# From Julia REPL
using Pkg
Pkg.add(url="https://github.com/cyborg-entrepreneur/Glimpse_ABM", subdir="glimpse_abm/julia")

# Or install locally
cd glimpse_abm/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Quick Start

### Python

```bash
# Basic simulation with venture calibration
python3 -m glimpse_abm.cli --task master --runs 5 --results-dir ./results \
  --calibration-profile venture_baseline_2024

# Smoke test (quick validation)
python3 -m glimpse_abm.cli --task master --smoke --results-dir ./smoke_test

# Fixed-tier causal analysis (recommended for publication)
python3 -m glimpse_abm.cli --task fixed --calibration-profile minimal_causal \
  --results-dir ./fixed_tier_results
```

### Julia

```julia
using GlimpseABM

# Create configuration (monthly cadence)
config = EmergentConfig(
    N_AGENTS = 1000,
    N_ROUNDS = 60,          # 5 years
    INITIAL_CAPITAL = 5_000_000.0,
    SURVIVAL_THRESHOLD = 230_000.0
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

See `julia/README.md` for more Julia examples including complete paradox analysis.

---

## Reproducing Key Results

### The Paradox Effect

```python
from glimpse_abm.config import EmergentConfig
from glimpse_abm.simulation import EmergentSimulation
import numpy as np

# Run comparative analysis
results = {}
for tier in ["none", "premium"]:
    survivals = []
    for run in range(50):
        config = EmergentConfig()
        config.N_AGENTS = 1000
        config.N_ROUNDS = 60
        config.RANDOM_SEED = 42 + run

        # Force all agents to single tier
        sim = EmergentSimulation(config, output_dir=f"./results/{tier}_{run}", run_id=run)
        sim.initial_tier_distribution = {t: (1.0 if t == tier else 0.0)
                                          for t in ["none","basic","advanced","premium"]}
        sim.run()

        alive = sum(1 for a in sim.agents if a.alive)
        survivals.append(alive / 1000)

    results[tier] = {"mean": np.mean(survivals), "std": np.std(survivals)}

print(f"None survival: {results['none']['mean']:.2%} ± {results['none']['std']:.2%}")
print(f"Premium survival: {results['premium']['mean']:.2%} ± {results['premium']['std']:.2%}")
print(f"Paradox effect: {(results['none']['mean'] - results['premium']['mean']):.1%} pp")
```

Expected output:
```
None survival: 72% ± 3%
Premium survival: 58% ± 4%
Paradox effect: 14% pp
```

---

## CLI Tasks

| Task | Purpose |
|------|---------|
| `master` | Full emergent simulation with AI tier learning |
| `fixed` | Locked AI-tier experiments for causal attribution |
| `scenarios` | Five Knightian scenario variants |
| `sensitivity` | Grid sweep over selected parameters |
| `lhs` | Latin Hypercube sweep for robustness testing |

### Key Options

```bash
--calibration-profile venture_baseline_2024  # Apply calibration bundle
--calibration-profile minimal_causal         # Minimal model for causal identification
--calibration-file path/to/custom.json       # Custom parameter overrides
--dump-config ./cfg.json                     # Export resolved configuration
--smoke                                       # Quick test (50 agents x 20 rounds)
--skip-visualizations                        # Skip figure generation
```

---

## Calibration Profiles

### `minimal_causal` (Recommended for Academic Research)

Minimal configuration optimized for causal identification:
- Single sector (tech)
- 1000 agents, 120 rounds (10 years monthly)
- 50 runs per AI tier
- Fixed exogenous AI assignment

### `venture_baseline_2024`

Full-scale configuration anchored to empirical benchmarks:
- **10-year survival rate**: 50% ± 8% (Source: BLS Business Employment Dynamics)
- **Mean investment ROI**: 1.12× ± 0.10 (Source: PitchBook Series A-D)
- **Innovation share**: 40% ± 10% (Source: NVCA 2024 estimates)

---

## Key Parameters (Monthly Cadence)

### Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_AGENTS` | 1000 | Agents per simulation |
| `N_ROUNDS` | 120 | Months (10 years) |
| `INITIAL_CAPITAL` | 5M | Starting capital |
| `SURVIVAL_THRESHOLD` | 230K | Minimum viable capital |
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

See `docs/PARAMETER_GLOSSARY.md` for complete parameter documentation.

---

## Sector-Specific Parameters (Monthly Cadence)

| Sector | Initial Capital | Monthly Op Cost | Innovation Prob | Knowledge Decay |
|--------|----------------|-----------------|-----------------|-----------------|
| **Tech** | $3M-$6M | $20k-$30k | 16%/month | 4%/month |
| **Retail** | $2.2M-$4M | $13k-$23k | 11%/month | 2.3%/month |
| **Service** | $1.4M-$2.5M | $8k-$15k | 13%/month | 1.7%/month |
| **Manufacturing** | $4M-$7.5M | $27k-$40k | 17%/month | 1%/month |

---

## Empirical Calibration Sources

| Parameter Category | Empirical Source | Key Metrics |
|-------------------|------------------|-------------|
| **Initial Capital** | NVCA 2024 Yearbook, PitchBook | Sector-specific seed/Series A medians |
| **Survival Thresholds** | BLS Business Employment Dynamics, Fed SBCS 2024 | 3-6 months operating expenses by sector |
| **Innovation Rates** | NSF BRDIS 2023, USPTO Patent Statistics | R&D intensity and patent grant rates |
| **Knowledge Decay** | Ebbinghaus forgetting curve, skill depreciation studies | Sector-specific half-lives (2-10 years) |
| **Market Regimes** | NBER Business Cycle Dating (1945-2024) | Expansion/recession durations and frequencies |
| **Competition** | Census Bureau Economic Census, DOJ HHI Guidelines | Industry concentration indices |

---

## Output Structure

```
results_dir/
├── config_snapshot.json       # Complete configuration for reproducibility
├── analysis_metadata.json     # Analysis version and table mappings
├── Fixed_AI_Level_none_run_0/
│   ├── run_log.jsonl         # Round-by-round metrics
│   ├── final_agents.pkl      # Serialized agent states
│   └── [component snapshots]
├── tables/                    # Publication-ready CSVs
│   ├── ai_stage_performance.csv
│   ├── ai_uncertainty_paradox_by_ai.csv
│   └── ai_paradox_signal.csv
└── figures/                   # Generated visualizations
```

---

## Testing

```bash
python3 -m pytest glimpse_abm/tests/test_smoke.py -v
```

---

## Reproducibility

Every simulation run captures:

1. **Configuration snapshot** (`config_snapshot.json`): All 150+ resolved parameters
2. **Random seed management**: Deterministic seeding with per-run variation
3. **Analysis version** (`analysis_metadata.json`): Pinned analysis code version

---

## Requirements

- Python 3.8+
- Dependencies: numpy, pandas, scipy, matplotlib, seaborn, tqdm, pyyaml
- Optional: Julia 1.9+ for high-performance runs

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Use**: If you use this software in research, please cite the associated publication (see Citation section below).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{townsend2026flux,
  title={Into the Flux: {AI} Augmentation \& The Paradox of Future Knowledge},
  author={Townsend, David M. and Hunt, Richard A. and Rady, Joseph and
          Manocha, Puneet and Jin, Jae-Hwan},
  journal={Entrepreneurship Theory and Practice},
  year={2026},
  note={Forthcoming}
}

@article{townsend2025futures,
  title={Are the Futures Computable? {K}nightian Uncertainty \& Artificial Intelligence},
  author={Townsend, David M. and Hunt, Richard A. and Rady, Joseph and
          Manocha, Puneet and Jin, Jae-Hwan},
  journal={Academy of Management Review},
  volume={50},
  number={2},
  pages={415--440},
  year={2025}
}
```

---

## References

Knight, F. H. (1921). *Risk, uncertainty, and profit*. Houghton Mifflin.

Sarasvathy, S. D. (2001). Causation and effectuation: Toward a theoretical shift from economic inevitability to entrepreneurial contingency. *Academy of Management Review*, 26(2), 243-263.

Townsend, D.M., Hunt, R.A., Rady, R., Manocha, P., & Jin, J-H. (2025). Are the Futures Computable? Knightian Uncertainty & Artificial Intelligence. *The Academy of Management Review*, 50(2): 415-440.

Townsend, D.M., Hunt, R.A., Rady, R., Manocha, P., & Jin, J-H. (2025). Do Androids Dream of Entrepreneurial Possibilities? A Reply to Ramoglou et al.'s "Artificial Intelligence Forces Us to Re-think Knightian Uncertainty." *The Academy of Management Review*, 50(2): 474-476.

Townsend, D.M., Hunt, R.A., & Rady, J. (2024). Chance, Probability, & Uncertainty at the Edge of Human Reasoning: What is Knightian Uncertainty? *Strategic Entrepreneurship Journal*, 18(3): 451-474.
