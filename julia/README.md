# GlimpseABM.jl

A high-performance Julia implementation of the GLIMPSE Agent-Based Model for studying AI adoption dynamics under Knightian uncertainty.

## Overview

GlimpseABM.jl is a complete port of the Python [glimpse_abm](https://github.com/yourusername/glimpse_abm) package, offering **~65x faster** simulation performance while maintaining statistical equivalence with the original implementation.

### Key Features

- **Agent-Based Modeling**: Simulate firms making investment decisions under uncertainty
- **AI Tier Analysis**: Study effects of different AI capability levels (none, basic, advanced, premium)
- **Knightian Uncertainty**: Four-dimensional uncertainty framework (actor ignorance, practical indeterminism, agentic novelty, competitive recursion)
- **Causal Analysis**: Fixed-tier sweeps for measuring true AI adoption effects
- **High Performance**: ~5M agent-rounds/second (vs ~70k in Python)
- **Empirical Calibration**: Sector-specific parameters calibrated to NVCA, BLS, NSF, and NBER data

## Empirical Calibration

All sector-specific parameters are calibrated to real-world empirical data:

| Parameter | Empirical Source |
|-----------|------------------|
| Initial Capital | NVCA 2024 Yearbook, PitchBook |
| Survival Threshold | BLS Business Employment Dynamics, Fed SBCS 2024 |
| Innovation Probability | NSF BRDIS 2023, USPTO Patent Statistics |
| Knowledge Decay | Ebbinghaus forgetting curve, skill depreciation studies |
| Market Regimes | NBER Business Cycle Dating (1945-2024) |
| Competition Intensity | Census Bureau Economic Census, DOJ HHI Guidelines |

See `docs/PARAMETER_GLOSSARY.md` for detailed parameter documentation and calibration sources.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/glimpse_abm", subdir="julia")
# Or if standalone repo:
# Pkg.add(url="https://github.com/yourusername/GlimpseABM.jl")
```

## Quick Start

```julia
using GlimpseABM

# Create configuration
config = EmergentConfig(
    N_AGENTS = 100,
    N_ROUNDS = 50,
    RANDOM_SEED = 42
)

# Run simulation
sim = EmergentSimulation(config)
run!(sim)

# Check results
alive = count(a -> a.alive, sim.agents)
println("Survival rate: $(alive)/$(config.N_AGENTS)")
```

## Running a Fixed-Tier Sweep

For causal analysis of AI effects:

```julia
using GlimpseABM

# Run sweep across all AI tiers
results = Dict{String, Vector{Float64}}()

for tier in ["none", "basic", "advanced", "premium"]
    survival_rates = Float64[]

    for seed in 1:20
        config = EmergentConfig(
            N_AGENTS = 100,
            N_ROUNDS = 100,
            RANDOM_SEED = seed,
            FIXED_AI_LEVEL = tier
        )
        sim = EmergentSimulation(config)
        run!(sim)

        rate = count(a -> a.alive, sim.agents) / config.N_AGENTS
        push!(survival_rates, rate)
    end

    results[tier] = survival_rates
    println("$tier: $(mean(survival_rates)*100)% mean survival")
end
```

## Package Structure

```
GlimpseABM.jl/
├── src/
│   ├── GlimpseABM.jl      # Main module
│   ├── config.jl          # EmergentConfig struct
│   ├── agents.jl          # EmergentAgent implementation
│   ├── market.jl          # MarketEnvironment
│   ├── uncertainty.jl     # Knightian uncertainty
│   ├── simulation.jl      # EmergentSimulation
│   └── io.jl              # File I/O utilities
├── causal/                # Causal analysis functions
├── analysis/              # Analysis & visualization
├── cli/                   # Command-line interface
├── test/                  # Test suite
└── validation/            # Python comparison scripts
```

## Performance Comparison

| Metric | Python | Julia | Speedup |
|--------|--------|-------|---------|
| Throughput | ~70k agent-rounds/s | ~5M agent-rounds/s | **65x** |
| 100-agent, 50-round sim | ~7s | ~0.1s | **70x** |
| Full parameter sweep | ~2.5 hours | ~7 minutes | **20x** |

## Validation

The Julia implementation has been validated against Python:

| Aspect | Status |
|--------|--------|
| Core simulation logic | Equivalent |
| Investment behavior | Equivalent |
| Innovation behavior | Equivalent |
| Agent survival | Equivalent (within RNG variance) |

See `validation/COMPARISON_REPORT.md` for detailed comparison.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

All 64 tests should pass.

## Requirements

- Julia 1.9+
- Dependencies (auto-installed): DataFrames, Distributions, StatsBase, JSON3, Arrow, JLD2

## License

MIT License - see LICENSE file.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{glimpse_abm,
  title = {GLIMPSE: Agent-Based Model for AI Adoption Dynamics},
  author = {Townsend, David},
  year = {2024},
  url = {https://github.com/yourusername/glimpse_abm}
}
```

## Related

- [glimpse_abm](https://github.com/yourusername/glimpse_abm) - Original Python implementation
