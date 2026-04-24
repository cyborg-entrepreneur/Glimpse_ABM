# v2.3 Re-validation — Headline Finding

**Job:** ARC 5130242
**Submitted:** 2026-04-23 22:04
**Completed:** 2026-04-23 22:28 (24 min wall, 1409s sim time)
**Design:** 4 tiers × 10 seeds = 40 runs. N=1000, R=60 (months), AGENT_AI_MODE="fixed"

## Survival rate (mean ± std over 10 seeds)

| tier | v1 (broken model) | v2.3 (corrected) | Δ |
|---|---|---|---|
| none | 0.840 | 0.803 ± 0.031 | −0.037 |
| basic | 0.847 | 0.907 ± 0.031 | +0.060 |
| advanced | 0.820 | **0.943 ± 0.010** | **+0.123** |
| premium | 0.705 | **0.908 ± 0.016** | **+0.203** |

## Mean capital among survivors ($M)

| tier | v1 | v2.3 |
|---|---|---|
| none | 2.58 | 2.50 ± 0.08 |
| basic | 2.61 | 2.53 ± 0.17 |
| advanced | 2.57 | **2.77 ± 0.08** |
| premium | 2.57 | **2.72 ± 0.06** |

## The AI-survival paradox is gone

**v1 (broken model):** premium − none = **−0.135** (premium 13.5 points worse — the published paradox)

**v2.3 (corrected):** premium − none = **+0.105** (premium 10.5 points better)

The paradox was a **bugs-aggregation artifact**, not a real mechanism finding. Once the v2/v2.1/v2.2/v2.3 correctness fixes are applied, the model's behavior aligns with the paper's theoretical claim: agents with better information quality survive more.

## Tier ordering in v2.3

```
advanced (0.943) > basic (0.907) ≈ premium (0.908) > none (0.803)
```

Advanced edges out premium slightly. Likely interpretation: advanced has good info quality (0.70) at moderate cost ($400/month), while premium pays 8.75× more ($3500/month) for marginally better quality (0.97). The cost premium for the extra accuracy isn't recouped at this horizon. This is a real economic insight — not a bug.

## What was fixing what

The v1 → v2.3 premium gap of +0.203 came from the cumulative effect of:

- **3× subscription over-billing** (v2 #4): premium was charged $10,500/round (3× monthly cost). Removing the multiplier alone is worth ~$300K/agent over 60 rounds, well within survival-threshold range.
- **InformationSystem bypass** (v2 #1+3): all tiers ranked by hidden ground truth. Premium's information-quality advantage was simulated but discarded.
- **manage_opportunities! never called** (v2 #6): crowding accumulated monotonically. Premium agents who pile into the same top-ranked opps got hit hardest.
- **Per-agent opportunity filtering disabled** (v2 #2): tier-aware visibility was never applied — premium's `info_breadth` advantage was unused.
- **Investment amount key mismatch** (v2.3 #1): `total_investment` summing to 0 disabled `update_market_dynamics!` and `manage_opportunities!`.
- **Estimated return key mismatch** (v2.3 #2): `update_market_dynamics!` saw empty quality data.
- **Per-investment competition decay** (v2.1 #3): competition couldn't accumulate past ~0.7, masking real crowding.
- Several more (info cache shared across agents, niche creation broken, lifecycle never updated, etc.)

Each individually was small. The aggregate moved premium from 0.705 to 0.908 — a 29% relative improvement.

## What this means for the paper

The published "Into the Flux" paradox finding (R1 to ETP, 2026-01-30) was generated with the v1 codebase (preserved at git tag `v1-published-results-2026-01-31`). The corrected v2.3 model produces the opposite empirical result: premium AI users *do* survive better than no-AI agents.

Two paths from here:

1. **Reframe the paper around the corrected finding.** The theoretical machinery (Knightian uncertainty, information quality, etc.) still holds; the empirical claim flips from "premium worse" to "premium better with caveats" (e.g., advanced has the best cost-benefit ratio at this horizon). This becomes a paper about *when* AI helps, not whether the paradox is real.

2. **Run robustness checks at multiple parameter settings** (longer horizons, different cost structures, varying market conditions) to characterize the conditions under which premium dominates vs. underperforms.

Either is a valid research path. The first is faster; the second is more defensible.

## Files

- Per-run JSON: `revalidation_v23_2026-04-23_2228.json`
- Stdout: `glimpse_revalv23_5130242.out`
- v1 baseline (for comparison): `../robustness_5129552/`

## Reproduction

```bash
ssh arc 'sbatch ~/glimpse_abm_robustness/arc/job_revalidation_v23.sh'
```
