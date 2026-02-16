# GlimpseABM: The Paradox of Future Knowledge in AI-Augmented Entrepreneurship

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive agent-based model (ABM) for studying how AI augmentation impacts entrepreneurial decision-making under Knightian uncertainty. This model accompanies the paper **"Into the Flux: AI Augmentation & The Paradox of Future Knowledge"** (Entrepreneurship Theory & Practice).

## The Paradox of Future Knowledge

> "At the heart of these dynamics is a paradox of future knowledge in competitive markets where the pursuit of entrepreneurial foresight systematically generates the uncertainties it seeks to resolve."

**Central Finding:** AI augmentation paradoxically *reduces* entrepreneurial survival rates through an **innovation equilibrium trap**:

```
Superior AI Information → Strategic Convergence → Market Crowding → Reduced Returns → Lower Survival
```

Premium AI agents (with near-perfect information) consistently underperform human-only agents by 8-14 percentage points in survival rates. This counterintuitive result emerges because:

1. **First-order benefit:** AI reduces actor ignorance through superior information processing
2. **Second-order cost:** AI amplifies competitive recursion as agents converge on similar opportunities
3. **Net effect:** The competitive costs outweigh the informational benefits

## Theoretical Foundation

The model operationalizes four dimensions of **Knightian uncertainty** from Townsend et al. (2024, 2025):

| Dimension | Definition | AI Effect |
|-----------|------------|-----------|
| **Actor Ignorance** | Incomplete knowledge of opportunities | AI **reduces** (first-order benefit) |
| **Practical Indeterminism** | Irreducible execution unpredictability | AI slightly reduces |
| **Agentic Novelty** | Creative potential for new combinations | AI may **constrain** |
| **Competitive Recursion** | Strategic interdependence among actors | AI **amplifies** (second-order cost) |

## Repository Structure

```
GlimpseABM/
├── README.md                           # This file
├── GLIMPSE_ABM_TECHNICAL_DOCUMENTATION.md  # Comprehensive OOD+D documentation
├── glimpse_abm/                        # Main Python package
│   ├── config.py                       # Configuration & calibration
│   ├── agents.py                       # Agent architecture (~3,100 lines)
│   ├── simulation.py                   # Simulation orchestration
│   ├── uncertainty.py                  # 4D Knightian uncertainty tracking
│   ├── innovation.py                   # Knowledge recombination
│   ├── market.py                       # Market dynamics & crowding
│   ├── information.py                  # AI analysis generation
│   ├── knowledge.py                    # Knowledge base management
│   ├── models.py                       # Core data structures
│   ├── docs/                           # Additional documentation
│   └── julia/                          # High-performance Julia implementation
│       ├── src/                        # Julia source modules
│       └── README.md                   # Julia-specific documentation
└── .gitignore
```

## Implementations

| Language | Performance | Use Case |
|----------|-------------|----------|
| **Python** | ~70k agent-rounds/s | Research, prototyping, analysis |
| **Julia** | ~5M agent-rounds/s | Large-scale sweeps, production |

Both implementations produce statistically equivalent results. The Julia version is **~65x faster**.

## Quick Start

### Python Installation

```bash
git clone https://github.com/cyborg-entrepreneur/Glimpse_ABM.git
cd Glimpse_ABM

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn networkx tqdm pyyaml

# Run a basic simulation
python -m glimpse_abm.cli --task fixed --runs 5 --results-dir ./results
```

### Julia Installation

```bash
cd glimpse_abm/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Basic Usage (Python)

```python
from glimpse_abm.config import EmergentConfig
from glimpse_abm.simulation import EmergentSimulation

# Create configuration
config = EmergentConfig(
    N_AGENTS=1000,
    N_ROUNDS=120,  # 10 years (monthly cadence)
    N_RUNS=50,
)

# Run fixed-tier experiment
for tier in ["none", "basic", "advanced", "premium"]:
    sim = EmergentSimulation(config, fixed_ai_tier=tier)
    results = sim.run_simulation()

    # Analyze survival rates
    alive = sum(1 for a in sim.agents if a.alive)
    print(f"{tier}: {alive/1000:.1%} survival")
```

## Key Model Components

### Agent Design (Table 1 in paper)

Each agent has a 10-dimensional cognitive trait profile:
- Uncertainty tolerance, Innovativeness, Competence
- AI trust, Exploration tendency, Market awareness
- Risk appetite, Adaptability, Social orientation, Strategic patience

### AI Tiers (Table 2a in paper)

| Tier | Info Quality | Monthly Cost | Calibration |
|------|-------------|--------------|-------------|
| None | 0.25 | $0 | Human baseline |
| Basic | 0.43 | $30 | GPT-5 level (2025) |
| Advanced | 0.70 | $400 | Frontier AI (2026) |
| Premium | 0.97 | $3,500 | AGI projection (2027) |

Info quality follows LLM scaling laws: `info_quality = 0.25 + 0.09 × log₁₀(compute)`

### Market Environment

- **Regime switching:** Crisis → Recession → Normal → Growth → Boom (NBER-calibrated)
- **Sector profiles:** Tech, Retail, Service, Manufacturing (NVCA/BLS calibrated)
- **Crowding penalties:** Capacity-convexity model for competitive congestion

### Innovation System

- Knowledge recombination across 4 domains (technology, market, process, business model)
- Innovation types: Incremental, Architectural, Radical, Disruptive
- Combination tracking for detecting convergent behavior

## Empirical Calibration

| Parameter | Source | Target |
|-----------|--------|--------|
| Survival rates | BLS Business Employment Dynamics | 55% ± 8% (5-year) |
| Capital requirements | NVCA 2024 | $2.5M-$10M |
| Innovation rates | NSF BRDIS, USPTO | 40% ± 10% activity share |
| Return distributions | Kaplan & Schoar, Korteweg & Sorensen | Power law (α=3.0) |
| Regime transitions | NBER Business Cycle Dating | Markov probabilities |

## Key Results (from paper)

### Fixed-Tier Experiments (Tables 3A-K)

- **Survival:** None (60%) > Basic (52%) > Advanced (45%) > Premium (35%)
- **Treatment effect:** Premium AI reduces survival by ~23 percentage points vs None
- **Innovation volume:** Premium creates 11x more niches but with constant success rate (~11%)
- **Competition:** Premium agents experience 2-3x higher crowding ratios

### Robustness (Tables 4A-I)

- 15/15 robustness tests confirm the paradox
- Effect persists across capital levels, population sizes, time horizons
- Placebo and permutation tests confirm causal attribution

### Mechanism Analysis (Tables 5A-I)

- Behavioral shift: AI increases innovation share (27% → 32%)
- Crowding mediates the negative effect on survival
- Within-tier analysis confirms competition as mechanism

### Refutation Tests (Tables 6A-C)

- 31 extreme conditions tested (execution boosts up to 10x, quality boosts up to +50%)
- **Paradox persists in all conditions**
- Even "all favorable" conditions (no crowding, free AI, max boosts) show negative effect

## Documentation

- **[GLIMPSE_ABM_TECHNICAL_DOCUMENTATION.md](GLIMPSE_ABM_TECHNICAL_DOCUMENTATION.md)** - Comprehensive OOD+D with theoretical grounding
- **[glimpse_abm/README.md](glimpse_abm/README.md)** - Detailed Python package documentation
- **[glimpse_abm/julia/README.md](glimpse_abm/julia/README.md)** - Julia implementation guide
- **[glimpse_abm/docs/](glimpse_abm/docs/)** - Parameter glossary, calibration guides, trait documentation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{townsend2026flux,
  title={Into the Flux: {AI} Augmentation \& The Paradox of Future Knowledge},
  author={Townsend, David M. and Hunt, Richard A. and Rady, Judy},
  journal={Entrepreneurship Theory and Practice},
  year={2026},
  note={Forthcoming}
}

@article{townsend2025futures,
  title={Are the Futures Computable? {K}nightian Uncertainty \& Artificial Intelligence},
  author={Townsend, David M. and Hunt, Richard A. and Rady, Judy and
          Manocha, Parul and Jin, Ju hyeong},
  journal={Academy of Management Review},
  volume={50},
  number={2},
  pages={415--440},
  year={2025}
}
```

## References

- Knight, F. H. (1921). *Risk, Uncertainty and Profit*. Houghton Mifflin.
- Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, & uncertainty at the edge of human reasoning. *Strategic Entrepreneurship Journal*, 18(3), 451-474.
- Townsend, D. M., et al. (2025a). Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.
- Townsend, D. M., et al. (2025b). Do androids dream of entrepreneurial possibilities? *Academy of Management Review*, 50(2), 474-476.

## License

MIT License - see [LICENSE](glimpse_abm/LICENSE) for details.

## Contact

For questions about the model or paper, please open an issue or contact the authors.
