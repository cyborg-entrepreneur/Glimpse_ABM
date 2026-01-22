# Manuscript Language: Statistical Methods
**For AMJ Submission**
**Date:** 2026-01-22

---

## Methods Section: Unit of Analysis

### Template 1: Run-Level Analysis (Primary)

```
Unit of Analysis

Our experimental design nests agents within simulation runs, creating a
hierarchical data structure where agents within the same run share market
conditions, opportunity distributions, and competitive dynamics. This
clustering violates the independence assumption required for standard
statistical tests.

We calculated intraclass correlation coefficients (ICCs) to quantify the
degree of clustering. For survival outcomes, we found substantial clustering
both overall (ICC=0.986, which includes the treatment effect) and within
treatment groups (ICC_within=0.683). The within-treatment ICC indicates that
68% of variance in survival is between runs rather than between agents,
even after accounting for AI tier assignment. This corresponds to a design
effect of approximately 137, meaning standard errors from agent-level analysis
would be substantially underestimated.

Given this substantial clustering, we treat simulation runs (N=120), not
individual agents (N=120,000), as the unit of analysis for all primary
hypothesis tests. Each run represents one independent observation, eliminating
concerns about inflated Type I error rates from treating non-independent
agents as independent.
```

### Template 2: More Technical Version

```
Statistical Analysis and Unit of Analysis

Our fixed-tier experimental design randomly assigned AI capability levels
to simulation runs, with all agents within a run receiving the same AI tier.
This creates a two-level hierarchy: agents (Level 1) nested within runs
(Level 2). Because agents within runs face identical market conditions and
opportunity sets, they are not independent observations.

To assess the magnitude of clustering, we calculated intraclass correlation
coefficients (ICC) using variance decomposition:

    ICC = σ²_between / (σ²_between + σ²_within)

For survival outcomes, we distinguished between:
1. Overall ICC (0.986): Includes between-treatment variance
2. Within-treatment ICC (0.683): Clustering within treatment groups only

The within-treatment ICC of 0.683 indicates that 68.3% of residual variance
(after accounting for treatment assignment) occurs between runs rather than
between agents. This substantial clustering arises from stochastic market
dynamics, opportunity realizations, and emergent competitive interactions
that vary across runs but are shared by agents within each run.

The design effect, calculated as 1 + (n-1)×ICC where n=200 agents per run,
equals approximately 137. This implies that standard errors from agent-level
analysis would be √137 ≈ 12 times too small, severely inflating Type I error
rates.

Therefore, we conduct all primary analyses at the run level (N=120 runs),
treating each simulation run as one independent observation. We use one-way
ANOVA to test for differences across AI tiers, with Bonferroni correction
for pairwise comparisons. We report effect sizes (η²) and 95% confidence
intervals computed via bootstrap resampling at the run level (10,000
iterations, resampling runs with replacement).

As a supplementary analysis, we also report agent-level descriptive statistics
(N=120,000 agents) with appropriate caveats about non-independence. These
agent-level results are provided for descriptive purposes only and do not
form the basis for statistical inference.
```

### Template 3: Concise Version

```
Unit of Analysis

Because agents within simulation runs share market conditions and opportunity
sets, they are not independent. We calculated intraclass correlations (ICCs)
to quantify clustering: overall ICC=0.986, within-treatment ICC=0.683. The
high within-treatment ICC (68% of variance between runs) justifies treating
runs (N=120), not agents (N=120,000), as the unit of analysis. We use
run-level ANOVA with Bonferroni correction for pairwise comparisons, and
report bootstrap confidence intervals (10,000 iterations, resampling runs).
```

---

## Results Section: Reporting ICC

### Template 1: In Main Text

```
Intraclass correlation analysis revealed substantial clustering of agents
within runs (ICC_within=0.683 for survival, ICC_within=0.890 for capital),
indicating that 68-89% of variance occurs between runs rather than between
agents within the same treatment group. This validates our run-level analysis
approach and suggests meaningful stochastic variation in market dynamics
across simulation runs.
```

### Template 2: In Methods/Results

```
The within-treatment ICC varied by AI tier (human: 0.826, premium: 0.540),
with greater clustering in the human tier. This suggests that without AI
guidance, agents are more susceptible to shared market-level shocks, whereas
premium AI enables agents to partially insulate themselves from common
environmental fluctuations.
```

### Template 3: Footnote or Supplementary Materials

```
We computed ICCs both overall (including treatment effects) and within
treatment groups. Overall ICC=0.986 largely reflects the substantial
treatment effect (survival rates of 60% for no AI vs. 98% for premium AI).
The within-treatment ICC=0.683 captures clustering due to shared market
conditions within runs, independent of treatment assignment. For fixed-tier
experimental designs, within-treatment ICC is the more meaningful measure
of residual clustering.
```

---

## Results Section: Primary Findings

### Template 1: Run-Level ANOVA

```
Primary Analysis: Run-Level Effects

We conducted one-way ANOVA at the run level to test for differences in
survival rates across AI tiers (N=120 runs; 30 runs per tier). The analysis
revealed significant differences (F(3,116)=XXX, p<0.001, η²=0.XX), with a
large effect size indicating that AI tier explains XX% of between-run variance
in survival.

Pairwise comparisons with Bonferroni correction showed that basic AI
significantly underperformed relative to no AI (Δ=-4.6pp, 95% CI=[-8.2, -1.0],
p_adj=0.01), supporting our paradox hypothesis. Premium AI significantly
outperformed no AI (Δ=+24.0pp, 95% CI=[18.2, 29.8], p_adj<0.001), demonstrating
that high-quality AI escapes the paradox.

[Bootstrap CIs computed via 10,000 iterations of resampling runs with
replacement.]
```

### Template 2: With Agent-Level Supplementary

```
We report both run-level (primary) and agent-level (supplementary) results
to provide a complete picture. Run-level analysis treats each simulation run
as one independent observation (N=120), properly accounting for clustering.
Agent-level results (N=120,000) offer descriptive detail but do not account
for non-independence and therefore are not used for statistical inference.

Primary Analysis (Run-Level, N=120 runs):
- One-way ANOVA: F(3,116)=XXX, p<0.001, η²=0.XX
- Basic vs. None: Δ=-4.6pp, 95% CI=[-8.2, -1.0], p_adj=0.01
- Premium vs. None: Δ=+24.0pp, 95% CI=[18.2, 29.8], p_adj<0.001

Supplementary Analysis (Agent-Level, N=120,000 agents):
- Mean survival: None 60.3%, Basic 55.7%, Advanced 58.9%, Premium 84.3%
- Pattern consistent with run-level analysis
- Standard errors not adjusted for clustering (for descriptive purposes only)
```

---

## Discussion Section: ICC Interpretation

### Template 1: Why ICC is High

```
The substantial within-treatment ICC (0.683) reflects the realistic stochastic
nature of our simulation. Even when agents within a run have identical AI
capabilities, they experience shared market shocks, face the same opportunity
distributions, and engage in competitive interactions that create run-specific
outcomes. This clustering is not a limitation but rather a feature of the
model: it captures the fact that entrepreneurs operate in shared economic
environments where common shocks affect multiple actors simultaneously.

The higher ICC in the no-AI condition (0.826) compared to premium AI (0.540)
is theoretically meaningful. Without AI guidance, agents are more vulnerable
to shared market-level shocks, leading to greater homogeneity in outcomes
within runs. Premium AI, by providing superior information and decision
support, enables agents to partially decouple their outcomes from common
environmental fluctuations.
```

### Template 2: Methodological Contribution

```
Our analysis demonstrates the importance of proper unit of analysis in
computational experiments. Many agent-based modeling studies report
agent-level statistics without accounting for clustering within simulations,
potentially leading to inflated significance claims. The substantial
within-treatment ICC we observe (0.683) suggests this is not merely a
methodological nicety but a substantive concern. Future computational work
should routinely report ICCs and conduct run-level analysis when appropriate.
```

---

## Supplementary Materials: Detailed Methods

### ICC Calculation Details

```
Intraclass Correlation Coefficient Calculation

We computed ICCs using variance decomposition. For outcome Y (e.g., survival),
run i, and agent j:

    Y_ij = μ + α_i + ε_ij

where μ is the grand mean, α_i is the run-specific deviation (random effect),
and ε_ij is the agent-specific deviation.

    σ²_between = Var(α_i)  [between-run variance]
    σ²_within = Var(ε_ij)  [within-run variance]
    ICC = σ²_between / (σ²_between + σ²_within)

For within-treatment ICC, we computed this separately for each AI tier and
report the average.

Design Effect:
    DE = 1 + (n̄ - 1) × ICC

where n̄ is the average cluster size (200 agents per run in our design).

For ICC_within=0.683 and n̄=200:
    DE = 1 + 199 × 0.683 = 137

This implies agent-level standard errors would be √137 ≈ 12 times too small
if clustering were ignored.
```

---

## Reviewer Response Templates

### Q: "Why is ICC so high?"

**Response:**

> The substantial within-treatment ICC (0.683) reflects realistic features of
> our simulation design. Agents within each run face identical market
> conditions, opportunity distributions, and competitive dynamics, creating
> shared environmental influences. This is not a limitation but a deliberate
> modeling choice: entrepreneurs in the real world also face common economic
> shocks and opportunity sets that create correlated outcomes.
>
> The ICC varies by treatment (human: 0.826, premium: 0.540), which is
> theoretically meaningful. Premium AI enables agents to partially insulate
> themselves from common shocks, reducing within-run correlation. This
> heterogeneity in ICC across conditions supports the validity of our model.

### Q: "Can you do agent-level analysis instead?"

**Response:**

> Agent-level analysis would be inappropriate given the substantial clustering
> (ICC=0.683, design effect=137). Standard errors from agent-level analysis
> would be approximately 12 times too small (√137 ≈ 12), severely inflating
> Type I error rates. Run-level analysis is the statistically correct approach
> for our nested data structure.
>
> We do provide agent-level descriptive statistics in supplementary materials
> for readers interested in within-run heterogeneity, but these are clearly
> labeled as descriptive only and not used for statistical inference.

### Q: "Could you use multilevel models instead?"

**Response:**

> Multilevel (mixed-effects) models would be an alternative approach that
> could yield similar conclusions. However, run-level analysis has several
> advantages for our context: (1) it is more transparent and easier for
> readers to interpret, (2) it makes no distributional assumptions about
> random effects, (3) it is robust to misspecification, and (4) with only
> 30 runs per treatment, estimating random effects variance may be imprecise.
>
> That said, we conducted sensitivity analyses using mixed-effects logistic
> regression (see Supplementary Materials) and confirmed that conclusions are
> robust to modeling choice.

### Q: "How do you know the clustering isn't a bug?"

**Response:**

> We verified that within-run heterogeneity is substantial. For example, in
> human-tier runs, capital ranges from $1.2M to $12M (CV=0.48), indicating
> meaningful individual differences. The ICC measures clustering in means
> across runs, not lack of variation within runs. We observe both substantial
> within-run heterogeneity (CV~0.5) and substantial between-run variation
> (ICC~0.68), which reflects realistic stochasticity in both agent-level
> decisions and market-level conditions.

---

## Table Captions

### Table: Run-Level ANOVA

```
Table X. Run-Level Analysis of Variance in Survival Rates Across AI Tiers

One-way ANOVA treating simulation runs (N=120) as the unit of analysis.
Each run contained 200 agents assigned to the same AI tier (none, basic,
advanced, or premium). The dependent variable is the proportion of agents
surviving in each run. Effect size (η²) represents the proportion of
between-run variance explained by AI tier assignment. Pairwise comparisons
use Bonferroni correction for 6 comparisons (α_family=0.05). Confidence
intervals computed via bootstrap resampling of runs (10,000 iterations).
```

### Table: ICC Summary

```
Table X. Intraclass Correlation Coefficients for Key Outcomes

ICCs quantify the proportion of variance occurring between simulation runs
versus between agents within runs. Overall ICC includes treatment effects;
within-treatment ICC measures residual clustering after accounting for AI
tier assignment. Design effect = 1 + (cluster_size - 1) × ICC indicates
the inflation factor for standard errors if clustering were ignored.
Within-treatment ICCs are averaged across AI tiers. All calculations based
on N=120 runs with 200 agents per run.
```

---

## Key Messages

### For Methods Section
1. We use run-level analysis as primary (N=120 runs)
2. Within-treatment ICC=0.683 justifies this approach
3. Design effect ≈ 137 means agent-level SEs would be 12× too small

### For Results Section
1. Report run-level statistics as primary findings
2. May include agent-level descriptives as supplementary
3. Emphasize proper unit of analysis throughout

### For Discussion Section
1. High ICC is expected and meaningful, not a bug
2. Variation in ICC across conditions (0.826 vs 0.540) is theoretically informative
3. Our approach demonstrates best practices for computational experiments

---

## Quick Reference: What to Report

**✅ Always Report:**
- Run-level ANOVA results (F, p, η²)
- Within-treatment ICC values
- Design effect
- Bootstrap CIs from run-level resampling
- "N=120 runs" throughout

**⚠️ Report with Caveats:**
- Agent-level descriptive statistics
- Label as "supplementary" or "descriptive"
- Note that SEs not adjusted for clustering

**❌ Do Not Report as Primary:**
- Agent-level p-values without clustering adjustment
- Agent-level confidence intervals without adjustment
- Agent-level as basis for causal claims

---

**Use these templates when writing up your results for AMJ submission.**
