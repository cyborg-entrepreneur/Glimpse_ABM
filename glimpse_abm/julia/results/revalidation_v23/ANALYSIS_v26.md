# Phase 5b Re-validation — v2.5 + v2.6 (calibrated + convexity active)

**Job:** ARC 5130292
**Ran:** 2026-04-23 23:10 → 23:29 (19 min wall, 1131s sim time)
**Design:** 4 tiers × 10 seeds = 40 runs. N=1000, R=60 months, AGENT_AI_MODE="fixed"

## Three-way comparison: v1 baseline → v2.3 corrected → v2.6 calibrated

| tier | v1 (broken) | v2.3 (corrected, uncalibrated) | v2.6 (calibrated + convexity) |
|---|---|---|---|
| none | 0.840 | 0.803 ± 0.031 | **0.388 ± 0.043** |
| basic | 0.847 | 0.907 ± 0.031 | 0.560 ± 0.067 |
| advanced | 0.820 | 0.943 ± 0.010 | **0.629 ± 0.021** |
| premium | 0.705 | 0.908 ± 0.016 | 0.537 ± 0.023 |
| **mean** | 0.803 | 0.890 | **0.528** |

**Mean survival ≈ 52.8% — right in the BLS 5-year target band (50-55%).**

## The story has three tiers

1. **v1 (broken model, published paradox):**
   Premium 0.705 < none 0.840. Paradox appeared real but was driven by
   bugs: 3× subscription overcharge, InformationSystem bypass, missing
   crowding decay, etc.

2. **v2.3 (correctness fixes, no calibration):**
   Premium 0.908 > none 0.803. Paradox flipped — premium *better* than
   none. But mean 89% survival was far above BLS benchmark. The
   calibration was masking the bugs; fixing the bugs exposed the
   calibration over-leniency.

3. **v2.6 (calibration + active convexity penalty):**
   Premium 0.537 > none 0.388 (NOT paradox), but premium < advanced
   (0.629) and premium ≈ basic (0.560). The cost-benefit story emerges.

## v2.6 tier ordering

```
advanced (0.629) > basic (0.560) > premium (0.537) > none (0.388)
```

**Economic interpretation:**
- **Advanced is the sweet spot.** $400/month subscription buys info_quality 0.70 — enough to help decisions without paying the premium's cost premium for marginal quality gains.
- **Premium (0.537) pays 8.75× more** than advanced for info_quality 0.97 vs 0.70. The info advantage is real but premium agents converge on the same top-ranked opportunities (the convexity mechanism), and the crowding penalty on those opps eats into returns.
- **Basic (0.560) at $30/use** actually beats premium marginally. Basic's tier-noise is higher (info_quality 0.43), which spreads its picks more, reducing convergence crowding. The cost is low.
- **None (0.388) lacks AI entirely.** Worst survival because decisions are noisy + no quality advantage + no discovery breadth.

## The paradox status

**Not a paradox anymore, but a nuanced finding:**
- Premium does NOT dominate despite highest info quality
- Advanced (sweet spot) beats premium by 9 points
- Premium beats none but by a smaller margin (15 points) than advanced beats none (24 points)

This is a **publishable story** that replaces the v1 paradox:
> AI-assisted entrepreneurship improves survival over non-AI baselines,
> but the marginal value of top-tier AI is reduced by the convergence-driven
> crowding mechanism: agents with high-quality AI see similar opportunities
> as "best" and pile into them, triggering capacity penalties that partially
> offset the quality advantage.

## What the v2.5 + v2.6 changes did

**v2.5 (tightened failure mechanism):**
- SURVIVAL_CAPITAL_RATIO: 0.38 → 0.40
- INSOLVENCY_GRACE_ROUNDS: 7 → 6
- BASE_OPERATIONAL_COST: 15K → 22.5K (1.5× for 2026-era compute)
- Sector survival_threshold: 50% → 65% of min capital
- Sector ops cost: 1.5× scale
- **Added equity_failure mechanism** (was missing from Julia, present in Python)

Effect: mean survival dropped from 89% (v2.3) to ~60%.

**v2.6 (lowered K to activate convexity):**
- CROWDING_CAPACITY_K: 8.0 → 1.5

Effect: convexity penalty actually fires (was dead code with K=8). Premium
drops from ~60% to ~54% because crowding mechanism engages selectively on
premium's convergent picks.

## Per-failure-reason breakdown

From the per-seed logs, failures now split between both mechanisms:
- liquidity_failure (capital below sector threshold for 6+ months)
- equity_failure (capital below 40% of initial for 6+ months)

Both contribute; exact breakdown varies by tier.

## Files

- Per-run JSON: `revalidation_v23_2026-04-23_2329.json` (Phase 5b)
- Stdout: `glimpse_revalv23_5130292.out`
- v2.3 predecessor (for comparison): `revalidation_v23_2026-04-23_2228.json`
- v1 baseline (original bugs): `../robustness_5129552/`

## Reproduction

Current `main` at commit 8adb53a (Python parity) / v2.6 is at 957d391.

```bash
cd ~/glimpse_abm_robustness
sbatch arc/job_revalidation_v23.sh
```
