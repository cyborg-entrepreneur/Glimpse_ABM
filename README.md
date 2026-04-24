# GlimpseABM: The Paradox of Future Knowledge in AI-Augmented Entrepreneurship

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive agent-based model (ABM) for studying how AI augmentation impacts entrepreneurial decision-making under Knightian uncertainty. This model accompanies the paper **"Into the Flux: AI Augmentation & The Paradox of Future Knowledge"** (Entrepreneurship Theory & Practice).

> **Status (v3.3.4, 2026-04-24):** The v1 "paradox" framing below was an
> artifact of correctness bugs fixed across v2.1–v2.12. The current canonical
> finding is **advanced > basic > premium ≈ none**. AI adoption is broadly
> beneficial (basic and advanced both substantially beat no-AI), but top-tier
> AI is *trapped by its own effectiveness* via convergence-crowding through
> three mechanisms: capital saturation (v3.1), competitive recursion (v3.3),
> and component-scarcity erosion (v3.3.3). v3.3.4 produces this ordering with
> mean survival 0.527 in the BLS 50–55% band across three seeds.
>
> An ETP R&R was received 2026-04-24 (decision: major revision). Revision
> plan at `REVISION_PLAN_v1.md`; recommendation is to pivot the headline
> finding from fixed-tier "paradox" to dynamic-adoption equilibrium, frame
> the contribution against congestion-game literature, and lean the
> presented model to 4–5 load-bearing traits via ablation.
>
> See `~/.claude/.../memory/project_glimpse_arc_job.md` for the current
> tier numbers and tag list, and
> `glimpse_abm/julia/results/v32_max_fraction_sweep/ANALYSIS.md` for the
> sensitivity analysis. The paragraphs below describe the retired v1 framing
> and are preserved for paper context.

## The Paradox of Future Knowledge (v1 framing — retired)

> "At the heart of these dynamics is a paradox of future knowledge in competitive markets where the pursuit of entrepreneurial foresight systematically generates the uncertainties it seeks to resolve."

**Original claim (since retired):** AI augmentation paradoxically *reduces* entrepreneurial survival rates through an **innovation equilibrium trap**:

```
Superior AI Information → Strategic Convergence → Market Crowding → Reduced Returns → Lower Survival
```

Premium AI agents (with near-perfect information) consistently underperform human-only agents by 8-14 percentage points in survival rates. *(This result was driven by bugs: 3× subscription overcharge, InformationSystem bypass, missing crowding decay, etc. — fixed in v2.x.)*

1. **First-order benefit:** AI reduces actor ignorance through superior information processing
2. **Second-order cost:** AI amplifies competitive recursion as agents converge on similar opportunities
3. **Net effect (v1):** The competitive costs outweigh the informational benefits (**the mechanism is real at v3.3 but no longer inverts the ordering — advanced now dominates, premium underperforms advanced but still beats none**)

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

### Fixed-Tier Experiments (original Tables 3A-K — v1 numbers, retired)

- **v1 survival ordering:** None (60%) > Basic (52%) > Advanced (45%) > Premium (35%)
- **v1 treatment effect claim:** Premium AI reduces survival by ~23 percentage points vs None
- **Innovation volume:** Premium creates 11x more niches but with constant success rate (~11%)
- **Competition:** Premium agents experience 2-3x higher crowding ratios

### v3.3.4 canonical results (N=1000 × 60 × 4-tier fixed, K_sat=1.5, 3 seeds)

| seed | none | basic | advanced | premium | mean |
|---|---|---|---|---|---|
| 42 | 0.387 | 0.563 | 0.640 | 0.368 | 0.490 |
| 43 | 0.465 | 0.588 | 0.646 | 0.538 | 0.559 |
| 44 | 0.472 | 0.575 | 0.622 | 0.459 | 0.532 |

- **Tier ordering:** Advanced > Basic > Premium ≈ None — stable across seeds
- **Cross-seed mean:** 0.527 — squarely in BLS 5-year benchmark band (50–55%)
- **Premium/Advanced ratio:** ~0.65 — premium pays a meaningful trap cost but isn't strictly worse than no-AI
- **Sensitivity sweep:** `glimpse_abm/julia/results/v32_max_fraction_sweep/` shows the mechanism is monotone in `MAX_INVESTMENT_FRACTION`. Only 0.037 (the default) hits the BLS calibration band.

The v3.3.x audit rounds (4 of them, addressing 17 distinct findings)
landed end-to-end Knightian mechanism wiring: capital-saturation
convexity (v3.1), confidence × signal sizing (v3.2), Knightian
perception → utility pathway (v3.3), uniform-flag propagation +
demand clamping + signature honesty (v3.3.1), snapshot isolation +
innovate fallback + clearing-sign reconciliation (v3.3.2),
uncertainty-state snapshot + agentic scarcity wiring + novelty/scarcity
key emission (v3.3.3), knowledge learning + counter separation + sector
heterogeneity normalization (v3.3.4).

### Robustness (Tables 4A-I — v1 numbers, retired)

- *Original claim:* 15/15 robustness tests confirmed the paradox.
- *Current status:* the v1 "paradox" was driven by fixed bugs. The v3.3 trap mechanism (convergence → saturation → convexity penalty) is real but produces a softer ordering: advanced beats premium, premium beats none.

### Mechanism Analysis (Tables 5A-I)

- **v3.3 mechanism chain (code now instantiates this):**
  1. Tier-differentiated AI produces tier-differentiated `actor_ignorance` perception (premium 0.05 vs none 0.24 — 4.5× spread)
  2. Linear `ignorance_adjustment` translates that into 17% investment-utility differential
  3. High-tier agents converge on similar top opps → `competitive_recursion` perception rises (premium 0.28 vs none 0.13)
  4. Raised recursion coefficient (-0.25) hits pre-decision utility
  5. v3.1 capital-saturation convexity hits post-maturation returns
  6. Net: premium's ignorance benefit is partially offset by convergence-crowding cost → advanced (medium info quality, less convergence) is the sweet spot

### Refutation Tests (Tables 6A-C — v1 numbers, retired)

- *Original claim:* 31 extreme conditions tested; paradox persists in all.
- *Current status:* superseded by v2.6 calibration retirement of the paradox; the robustness question now is whether the convergence-crowding ordering survives sensitivity analysis — see `v32_max_fraction_sweep` for the empirical answer.

## Documentation

- **[GLIMPSE_ABM_TECHNICAL_DOCUMENTATION.md](GLIMPSE_ABM_TECHNICAL_DOCUMENTATION.md)** - Comprehensive OOD+D with theoretical grounding
- **[glimpse_abm/README.md](glimpse_abm/README.md)** - Detailed Python package documentation
- **[glimpse_abm/julia/README.md](glimpse_abm/julia/README.md)** - Julia implementation guide
- **[glimpse_abm/docs/](glimpse_abm/docs/)** - Parameter glossary, calibration guides, trait documentation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rady2026flux,
  title={Into the Flux: {AI} Augmentation \& The Paradox of Future Knowledge},
  author={Rady, Judy and Townsend, David M. and Hunt, Richard A.},
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

@software{glimpse_abm,
  title={GlimpseABM: Agent-Based Model of the AI Information Paradox},
  author={Townsend, David M. and Rady, Judy},
  year={2026},
  url={https://github.com/cyborg-entrepreneur/Glimpse_ABM}
}
```

## References

- Knight, F. H. (1921). *Risk, Uncertainty and Profit*. Houghton Mifflin.
- Townsend, D. M., Hunt, R. A., & Rady, J. (2024). Chance, probability, & uncertainty at the edge of human reasoning. *Strategic Entrepreneurship Journal*, 18(3), 451-474.
- Townsend, D. M., Hunt, R. A., Rady, J., Manocha, P., & Jin, J. h. (2025). Are the futures computable? Knightian uncertainty & artificial intelligence. *Academy of Management Review*, 50(2), 415-440.

## License

MIT License - see [LICENSE](glimpse_abm/LICENSE) for details.

## Contact

For questions about the model or paper, please open an issue or contact the authors.
