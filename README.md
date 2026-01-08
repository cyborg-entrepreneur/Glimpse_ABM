# Glimpse ABM: Agent-Based Simulation of Knightian Uncertainty Under AI Augmentation

This repository contains the agent-based model (ABM) for investigating how artificial intelligence affects entrepreneurial decision-making under Knightian uncertainty. The simulation operationalizes the theoretical framework from Townsend, Hunt, Rady, Manocha, & Jin (2025, The Academy of Management Review) and Townsend, Hunt, & Rady (2024, Strategic Entrepreneurship Journal).

## Repository Structure

```
glimpse_abm/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── config.py                # Configuration and calibration profiles
├── simulation.py            # Core simulation orchestration
├── agents.py                # Agent architecture and behavior
├── market.py                # Market dynamics and clearing
├── innovation.py            # Innovation and opportunity discovery
├── knowledge.py             # Knowledge management and AI augmentation
├── uncertainty.py           # Knightian uncertainty framework
├── information.py           # Information systems and signals
├── models.py                # Data models and structures
├── analysis.py              # Results analysis and statistics
├── statistical_tests.py     # Hypothesis testing framework
├── utils.py                 # Utility functions
├── profile_simulation.py    # Performance profiling
├── docs/                    # Documentation
│   ├── PARAMETER_GLOSSARY.md
│   ├── CALIBRATION_GUIDE.md
│   └── CALIBRATION_BASELINE.md
├── parameters/              # LHS parameter range configurations
│   ├── lhs_ranges.yaml
│   └── lhs_niche_ranges.yaml
├── scripts/                 # Automation scripts
│   ├── run_robustness_sweep.sh
│   └── analyze_robustness.py
├── tests/                   # Test suite
│   └── test_smoke.py
└── tools/                   # Progress tracking utilities
    ├── watch_progress.py
    └── progress_tracker.py
```

## Theoretical Foundation

### The Four Problems of Knightian Uncertainty

Knight (1921) distinguished between calculable *risk* and incalculable *uncertainty*. Townsend et al. (2025) extend this framework to identify four irreducible sources of Knightian uncertainty affecting entrepreneurial action:

1. **Actor Ignorance**: The entrepreneur's incomplete knowledge of the opportunity landscape, market conditions, and causal relationships.

2. **Practical Indeterminism**: The inherent unpredictability of execution paths, timing dependencies, and path-dependent outcomes.

3. **Agentic Novelty**: The creative potential for genuinely new combinations, innovations, and possibilities that did not previously exist.

4. **Competitive Recursion**: The strategic interdependence among actors where each agent's actions depend on anticipations of others' actions.

### The Paradox of Future Knowledge

A central insight from Townsend et al. (2025) is that AI creates a "paradox of future knowledge": while AI tools reduce actor ignorance, they may simultaneously:

- Increase practical indeterminism through faster competitive dynamics
- Reduce agentic novelty by anchoring on historical patterns
- Intensify competitive recursion through correlated recommendations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/glimpse_abm.git
cd glimpse_abm/..  # Move to parent directory

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn tqdm pyyaml

# Verify installation
python3 -m glimpse_abm.cli --help
```

## Quick Start

```bash
# Basic simulation with venture calibration
python3 -m glimpse_abm.cli --task master --runs 5 --results-dir ./results \
  --calibration-profile venture_baseline_2024

# Smoke test (quick validation)
python3 -m glimpse_abm.cli --task master --smoke --results-dir ./smoke_test

# Fixed-tier causal analysis (recommended for AMJ submission)
python3 -m glimpse_abm.cli --task fixed --calibration-profile minimal_causal \
  --results-dir ./fixed_tier_results
```

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

## Causal Identification

The `minimal_causal` calibration profile implements the fixed-tier experimental design for clean causal identification:

- **Single sector**: Eliminates sector-specific confounds
- **Exogenous AI assignment**: Agents randomly assigned to AI tiers (none/basic/advanced/premium)
- **30 runs per tier**: Sufficient statistical power (>99%)
- **Controlled parameters**: Isolates AI effects from market dynamics

## Robustness Analysis

Run the full robustness sweep across 8 parameter configurations:

```bash
# From parent directory of glimpse_abm
./glimpse_abm/scripts/run_robustness_sweep.sh

# Analyze results
python3 -m glimpse_abm.scripts.analyze_robustness glimpse_robustness_sweep
```

Configurations tested:
1. **baseline** - Default minimal_causal parameters
2. **high_cost** - BASE_OPERATIONAL_COST=90000
3. **low_cost** - BASE_OPERATIONAL_COST=40000
4. **high_threshold** - SURVIVAL_CAPITAL_RATIO=0.55
5. **low_threshold** - SURVIVAL_CAPITAL_RATIO=0.25
6. **high_noise** - RETURN_NOISE_SCALE=0.35
7. **low_noise** - RETURN_NOISE_SCALE=0.10
8. **multi_sector** - 4 sectors (ecological validity)

## Calibration Profiles

### `minimal_causal` (Recommended for Academic Research)

Minimal configuration optimized for causal identification:
- Single sector (tech)
- 1000 agents, 120 rounds
- 30 runs per AI tier
- Fixed exogenous AI assignment

### `venture_baseline_2024`

Full-scale configuration anchored to empirical benchmarks:
- **5-year survival rate**: 55% ± 8% (Source: BLS Business Employment Dynamics)
- **Mean investment ROI**: 1.12× ± 0.20 (Source: PitchBook Series A-D)
- **Innovation share**: 40% ± 10% (Source: NVCA 2024 estimates)

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

## Testing

```bash
python3 -m pytest glimpse_abm/tests/test_smoke.py -v
```

## Reproducibility

Every simulation run captures:

1. **Configuration snapshot** (`config_snapshot.json`): All 150+ resolved parameters
2. **Random seed management**: Deterministic seeding with per-run variation
3. **Analysis version** (`analysis_metadata.json`): Pinned analysis code version

## References

Knight, F. H. (1921). *Risk, uncertainty, and profit*. Houghton Mifflin.

Sarasvathy, S. D. (2001). Causation and effectuation: Toward a theoretical shift from economic inevitability to entrepreneurial contingency. *Academy of Management Review*, 26(2), 243-263.

Townsend, D.M., Hunt, R.A., Rady, R., Manocha, P., & Jin, J-H. (2025). Are the Futures Computable? Knightian Uncertainty & Artificial Intelligence. *The Academy of Management Review*, 50(2): 415-440.

Townsend, D.M., Hunt, R.A., Rady, R., Manocha, P., & Jin, J-H. (2025). Do Androids Dream of Entrepreneurial Possibilities? A Reply to Ramoglou et al.'s "Artificial Intelligence Forces Us to Re-think Knightian Uncertainty." *The Academy of Management Review*, 50(2): 474-476.

Townsend, D.M., Hunt, R.A., & Rady, J. (2024). Chance, Probability, & Uncertainty at the Edge of Human Reasoning: What is Knightian Uncertainty? *Strategic Entrepreneurship Journal*, 18(3): 451-474.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Use**: If you use this software in research, please cite the associated publication (see Citation section below).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{townsend2025futures,
  title={Are the Futures Computable? {K}nightian Uncertainty \& Artificial Intelligence},
  author={Townsend, D.M. and Hunt, R.A. and Rady, R. and Manocha, P. and Jin, J-H.},
  journal={The Academy of Management Review},
  volume={50},
  number={2},
  pages={415--440},
  year={2025}
}
```
