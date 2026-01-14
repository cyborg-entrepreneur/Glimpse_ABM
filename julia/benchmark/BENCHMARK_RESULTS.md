# GLIMPSE ABM: Julia vs Python Benchmark Results

## Summary

**Julia is ~65x faster than Python** for core ABM simulation patterns.

## Throughput Comparison (agent-rounds/second)

| Config | Agents | Rounds | Python | Julia | Speedup |
|--------|--------|--------|--------|-------|---------|
| Small | 100 | 50 | 58,184 | 3,702,378 | **64x** |
| Medium | 500 | 100 | 66,988 | 4,546,719 | **68x** |
| Standard | 1,000 | 200 | 80,936 | 5,466,068 | **68x** |
| Large | 2,000 | 200 | 78,975 | 5,161,422 | **65x** |

## Execution Time Comparison (seconds)

| Config | Agents | Rounds | Python | Julia | Speedup |
|--------|--------|--------|--------|-------|---------|
| Small | 100 | 50 | 0.086 | 0.001 | **61x** |
| Medium | 500 | 100 | 0.746 | 0.011 | **68x** |
| Standard | 1,000 | 200 | 2.471 | 0.037 | **67x** |
| Large | 2,000 | 200 | 5.065 | 0.078 | **65x** |

## Practical Implications

### Single Simulation (1000 agents, 200 rounds)
- **Python**: 2.47 seconds
- **Julia**: 0.04 seconds

### 200-Run Sensitivity Sweep
- **Python**: ~8 minutes
- **Julia**: ~7 seconds

### Full Research Workflow
- **Python 8-hour compute** = **Julia 7 minutes**

## Benchmark Methodology

Both benchmarks test identical computational patterns:
- Agent initialization with heterogeneous attributes
- Opportunity evaluation with AI-tier bonuses
- Investment decisions and outcomes
- Capital updates and survival checks
- Belief updates (Bayesian-style)

### Conditions
- Single-threaded execution (no parallelism)
- JIT warmup runs excluded from Julia timing
- NumPy used for Python array operations
- 5 trials per configuration, mean reported

## Running the Benchmarks

```bash
# Julia benchmark
cd GlimpseABM.jl
julia benchmark/run_benchmark.jl

# Python benchmark
python3 benchmark/run_benchmark.py
```

## System Information

- Machine: Apple Silicon Mac
- Julia: 1.12.x
- Python: 3.x with NumPy
- Date: January 2026
