# Flux ETP R&R — Revision Plan v1

**Decision:** Major (high-risk) revision. Editor: Boris Nikolaev (Colorado State).
**Due:** 3 months from 2026-04-24 → ~2026-07-24.
**Split reviews:** R1 reject, R2 major, R3 minor. Editor overrode with major.
**Editor signal:** "All editors enthusiastic about a potential revision" + "confident you will successfully tackle the revisions."

This plan distills the underlying logic of each concern, identifies what we defend
vs. concede, and maps concrete work items. v3.x code work done 2026-04-24
(`main @ 5f7a741`) already addresses parts of several concerns — those are
called out as assets below.

---

## Executive orientation

Three observations shape everything below.

**First, the editor's critique is substantively correct.** His reading —
"correlated signals + deterministic action + congestion penalty = the
familiar congestion-game inefficiency, labeled as a novel paradox" — is the
sharpest characterization of the paper's actual mechanism anyone has offered.
We cannot wave it away. We must either (a) defend the framing by showing our
mechanism is not reducible to congestion games, or (b) concede the reduction
and reframe the contribution.

**Second, the dynamic-adoption result is the paper's real contribution.**
Editor and R2 converge on this independently. The population converging to
mostly-Basic with minority-Premium tracks actual AI diffusion patterns
(ChatGPT Free/Plus/Pro, LLM API usage). This is empirically familiar to
reviewers in a way the fixed-tier "paradox" is not.

**Third, we have leverage.** The v3.x code work addresses several reviewer
concerns head-on: capital-saturation convexity (v3.1) replaces the count-based
penalty the reviewer suspected was gamed; confidence × signal sizing (v3.2)
lets heterogeneity translate into action; the Knightian perception → decision
utility pathway (v3.3) gives premium agents real crowding awareness pre-
decision, partially addressing the "level-0 thinking" critique. Source code
is public on GitHub. The max_fraction sweep documents mechanism sensitivity.
None of this was available when the paper was reviewed.

**Update (post-plan, 2026-04-24 evening):** Four additional audit rounds
landed (v3.3.1 through v3.3.4) before any revision writing started, each
addressing concrete defects flagged by an external code reviewer:

- v3.3.1: 19 analysis scripts had silent uniform-flag failure (capital
  100M expected, 1.68-6.67M delivered); demand adjustment formula could
  produce negative failure pressure (oversubscribed sectors functionally
  immortal); Union signatures lied about Dict support.
- v3.3.2: MarketConditions wasn't actually a snapshot (live dict
  references); _execute_innovate! Dict fallback crashed
  attempt_innovation!; clearing-ratio sign conflict between market.jl
  and models.jl (hot markets net-penalized returns).
- v3.3.3: uncertainty_state still not snapshotted; agentic scarcity
  was a dead mechanism (knowledge_base never attached, function never
  defined); novelty_potential / component_scarcity producer/consumer
  key mismatch; capacity scale; spawned-opp double-apply; subscription
  planner overstated cost.
- v3.3.4: knowledge learning was dead (learn_from_success!/failure!
  defined but never called — innovations succeeded but the knowledge
  base never grew); niche opp double-apply (sibling of v3.3.3 spawn fix);
  innovation vs investment success counter conflation; tier_invest_share
  staleness on no-invest rounds; sector clearing saturation flattened
  heterogeneity at N=1000; orphan export.

Net: the paper's central Knightian mechanism is now actually wired
end-to-end. v3.3.4 calibration produces tier ordering advanced > basic
> premium ≈ none with cross-seed mean 0.527 (in BLS band). This is
substantially what the paper claims to test — for the first time. Prior
to this round of fixes, several mechanism descriptions in the paper
described code paths that didn't actually fire.

---

## Cluster 1 — The congestion game critique (CRITICAL)

*Editor comments 2, 3; Reviewer 2 comment 3.*

### Underlying logic

The reviewer is asking: *Is there anything about AI + Knightian uncertainty in
your model that isn't already well-understood in congestion games under common
information?* If not, the contribution collapses from "novel paradox of AI
augmentation" to "a specific parametrization of a standard game-theoretic
inefficiency." Standard findings from that literature would be cited, not
ours.

The deeper worry: *our premium agents are level-0 thinkers.* The editor's
concrete counter-example is decisive:

> Opportunity A pays 10 − 0.4×nA; opportunity B pays 8 − 0.1×nB. Under level-0
> reasoning, all 10 agents rush A, each earning 6. Under equilibrium-aware
> reasoning, agents split 6/4 and earn 7.6. Mixed-strategy randomization
> (p ≈ 0.6) approximates the equilibrium.

This matters because the editor notices something we hadn't framed: *the
"noise" in human agents' decisions may approximate mixed-strategy equilibrium*,
which is why they outperform precision-optimizing premium. Our result may not
be "AI is bad" — it may be "deterministic best-response is worse than
randomized play in congestion games," which is textbook.

R2 frames the same concern differently: "an AGI-level system would not persist
in those errors without updating." A genuinely strategic system would notice
it's in a congestion game and diversify. Ours doesn't.

### What reviewers are really asking

Three things:

1. **Structurally:** show what premium agents can and cannot reason about.
   Can they anticipate that their signal is shared? Can they model
   best-response of similarly-equipped others?
2. **Empirically:** if we give premium agents level-1 strategic reasoning, does
   our result survive?
3. **Framing:** if the result doesn't survive strategic extension, what is the
   honest contribution?

### Our position

**Defend:** The bounded rationality assumption is empirically correct for
current LLM-scaffolded decision systems. Current LLMs do not reliably do
N-th-order strategic reasoning about aggregate behavior of other LLM users.
This is documented in the literature the editor already cites (Chevallier
et al. 2026).

**Concede:** We have not tested whether a strategically-extended premium
agent preserves the ordering. We must.

**Refine:** Reframe the contribution from "AI creates paradox" to "AI-induced
signal correlation creates a congestion externality that is not resolved by
individual agent sophistication under current deployment patterns." This is
novel because the combination of (a) strong signal correlation across agents,
(b) level-0 or level-1 strategic reasoning, and (c) Knightian-scale emergence
of new niches is not cleanly addressed in any single prior literature.

### Game plan

**[CODE] v3.4a — Strategic premium variant** (~2 weeks)
Implement a "level-1 strategic" premium agent that forms a belief about how
many similarly-equipped agents would see the same top-K opportunities, and
diversifies or down-weights accordingly. Keep v3.3 baseline as comparator.
Concrete mechanism:
- Premium agent reads `market_conditions.tier_invest_share["premium"]`
- Estimates expected number of premium competitors on top opp
- Applies softmax with temperature scaled by expected competition (like
  Luce choice with congestion anticipation)
- Alternatively: stochastic top-K picking where k scales with anticipated
  crowding

Report: does `advanced > basic > premium > none` hold under level-1 strategic
premium? If yes → finding is robust. If no → finding depends on bounded
rationality, and we must defend that assumption theoretically.

**[CODE] v3.4b — Mixed-strategy human baseline verification** (~3 days)
Compute explicitly whether no-AI agents' decision noise resembles the
mixed-strategy equilibrium for our opportunity set. This is a direct answer
to the editor's intuition. If true, state it clearly in the paper — it's a
new theoretical insight, not a rebuke.

**[WRITING] Theoretical reframe** (~1 week)
Rewrite contribution claim: *we identify a novel mechanism by which AI-induced
signal correlation converts entrepreneurial decision-making from an
uncoordinated search problem into a coordination game — one that current AI
systems cannot solve internally because their strategic reasoning is
bounded.* Position against congestion-game literature (cite the obvious:
Rosenthal 1973, Milgrom & Weber 1982 on information concentration, Heidhues
& Kőszegi 2005 on common information hurting equilibrium quality).

**[WRITING] Address E2 sensitivity-test confusion** (~2 days)
The editor noted results survive removing crowding penalties — "if there's
no crowding cost, why does convergence hurt?" Our v3.2 sweep
(`results/v32_max_fraction_sweep/ANALYSIS.md`) already demonstrates crowding
IS the mechanism: premium/advanced ratio falls monotonically with sizing
aggressiveness, exactly as a crowding mechanism predicts. Cite this directly.
Clarify that our prior "no crowding" test likely removed only one component
(linear capacity penalty — now deleted in v3.1) while convexity and tier-
share-invest mechanisms remained.

---

## Cluster 2 — Dynamic adoption as main finding (CRITICAL)

*Editor comment 6; Reviewer 2 comment 4.*

### Underlying logic

Reviewers are comparing the paper's headline ("AI is broadly detrimental")
against what they see in real AI adoption: most paid users on ChatGPT Plus
($20), smaller power-user fraction on Pro ($200), small minority on API,
meaningful non-adoption. Our dynamic-adoption result reproduces this pattern
almost exactly. Our fixed-tier paradox doesn't match anything observable.

So the pragmatic reader concludes: the paper's real finding is the dynamic
adoption equilibrium — which is *interesting and novel in the entrepreneurship
literature* even though it's consistent with technology diffusion patterns
more broadly — while the fixed-tier paradox is a methodological artifact of
holding tier choice fixed in a way no real agent would.

### What reviewers are really asking

Make the paper honest about which finding matters. The dynamic adoption story
doesn't rely on bounded strategic reasoning, doesn't require defending the
AGI framing, and aligns with empirical AI adoption. The fixed-tier story
depends on a lot of specific assumptions and reads as "gotcha" rather than
"insight."

### Our position

**Accept the pivot in full.** This improves the paper. The fixed-tier
analysis becomes the *counterfactual baseline* — "what if all agents were
constrained to tier X?" — that sets up the dynamic result.

### Game plan

**[WRITING] Abstract and introduction rewrite** (~1 week)
New thesis: *Under realistic AI adoption dynamics, entrepreneurs sort into
tiered AI usage through competitive learning. The population equilibrium
concentrates at mid-tier AI, with a minority persisting at high-tier and
meaningful non-adoption. This pattern emerges from a congestion externality
on shared AI signals that cannot be resolved by individual tier upgrades.*

Four-paragraph intro (Zahra formula per R3): phenomenon (AI adoption is
heterogeneous), puzzle (why doesn't everyone adopt the best AI?), solution
(signal-correlation congestion + bounded strategic reasoning), contributions
(three deltas).

**[WRITING] Results reordering** (~3 days)
New order:
1. Dynamic adoption equilibrium (figure: tier-share convergence over 60
   rounds; table: final distribution across seeds)
2. Fixed-tier decomposition (what does each tier contribute — showing why
   premium isn't dominant)
3. Mechanism tests (sensitivity, ablations)
4. Robustness (strategic premium, compute scaling, YC calibration)

**[CODE] Rerun dynamic adoption at v3.3 calibration** (~2 days, ARC)
Current v3.3 calibration ran fixed-tier (for cleanest tier comparison).
Submit ARC job for N=1000 × 10 seeds × dynamic AGENT_AI_MODE="emergent".
Report final tier distribution and compare to ChatGPT Plus/Pro/API usage
data.

**[WRITING] Technology diffusion lit engagement** (~3 days)
Rogers 1962, Bass 1969, Geroski 2000 on diffusion; Goolsbee & Klenow 2002 on
digital adoption. Position our finding as: *what's novel is not the diffusion
pattern but the mechanism — signal-correlation-induced congestion, not just
learning or network effects.*

---

## Cluster 3 — Model parsimony + source transparency (CRITICAL)

*Editor comment 5; Reviewer 1 comments 1, 5.*

### Underlying logic

R1 invokes Arend 2024 (AMR) — ABM critique landscape — and Crawford 2009 on
ABM parsimony. The fear: with 10 traits × 4 uncertainty dimensions × regime
switches × network effects, you can get any result you want. The reader
cannot tell which mechanisms are load-bearing vs decorative.

R1 also notes: *we don't have the source code.* For an ABM whose entire
contribution is computational, this is disqualifying in their eyes. Our
133-test Julia suite and public GitHub repo are directly responsive — they
just weren't cited in the submission.

### What reviewers are really asking

Demonstrate which mechanisms are load-bearing for the core result, and remove
the rest. Give them the code so they can verify.

### Our position

**Lean the model in the paper**, not necessarily in the code. The code is
a research vehicle — it can remain rich. But the paper should present a
*focused* model with explicit ablation showing which components matter.
This matches how top ABM papers handle complexity (Axtell & Epstein; Gavetti
& Levinthal 2000).

### Game plan

**[CODE + ANALYSIS] Trait ablation study** (~2 weeks)
Run 9 ablation experiments: disable each trait one at a time (freeze at
mean). Report: does advanced > basic > premium > none ordering hold? Does
premium/advanced ratio change by >5%? Traits that pass both tests are
load-bearing; traits that fail are decorative and can be dropped from the
paper's description (keep in code for future work).

Expected load-bearing (hypothesis):
- analytical_ability (gates hallucination detection)
- ai_trust (gates AI usage / decision_confidence)
- uncertainty_tolerance (gates innovation utility)
- competence (base decision quality)
- risk_tolerance (investment sizing + failure handling)

Expected decorative:
- exploration_tendency (redundant with innovativeness?)
- cognitive_style (overlaps with analytical_ability per R1)
- social_influence (probably minor absent strong network effects)
- trait_momentum (second-order)
- learning_rate (flat emergent differences)

**[CODE] Mechanism ablation** (~1 week)
Similar test on mechanism components: crowding convexity, Knightian
perception pathway, confidence sizing. Cite `results/v32_max_fraction_sweep/`
as template. Produces Table S1 for supplementary.

**[WRITING] Parsimonious model description** (~1 week)
Rewrite Section 3 (Model) around the 4–5 load-bearing traits + 3 mechanism
components. Push remaining traits/mechanisms to Appendix B ("Extended
model specification"). Use Levinthal 1997 didactic structure.

**[WRITING] Source code disclosure** (~1 day)
Add footnote at first mention of the ABM: *"Source code, calibration data,
and reproduction scripts are available at github.com/cyborg-entrepreneur/
Glimpse_ABM under the MIT license. The model is implemented in Julia 1.11
with a Python parity port for validation."* Cite specific release tag
(`v3.3-knightian-mechanism`).

**[WRITING] Mechanism decomposition figure** (~3 days)
Figure 2 (new): stacked-bar showing each mechanism's contribution to the
premium/advanced gap. Reader sees at a glance: X% comes from capital-
saturation crowding, Y% from ignorance-perception differential, Z% from
subscription cost.

---

## Cluster 4 — AI architecture, pricing, 2027/AGI framing (HIGH)

*Editor comment 1; Reviewer 1 comment 2; Reviewer 2 comments 1, 2.*

### Underlying logic

Reviewers are two years ahead on AI. They know:
- "AGI" is a loaded term, means different things, and using it makes us
  look dated before publication
- Pinning to December 2027 is an invitation to obsolescence
- Real AI costs scale with compute, not fixed subscriptions
- Current LLM product segmentation (ChatGPT tiers, API pricing) already
  models our framework

The sharper version of this: *as compute pricing evolves toward per-token
marginal cost, high-capital agents will be able to afford more inference
per problem — running more rollouts, testing more scenarios, simulating
more counterfactuals.* This produces a "rich get richer" dynamic that is
fundamentally different from the fixed-tier paradox. Our paper doesn't
address it.

### What reviewers are really asking

Three things:

1. Drop calendar-date anchoring — frame tiers as abstract capability levels
2. Justify fixed-tier modeling as reasonable approximation of current
   product segmentation (which it IS — ChatGPT Free/Plus/Pro/API are
   exactly four tiers)
3. Add compute-scaling robustness showing what happens when capital flows
   into marginal AI compute

### Our position

**Defend fixed-tier as empirically defensible** for current product
segmentation (ChatGPT Free/Plus/Pro, Claude Pro/Max, Gemini Advanced).
**Drop "AGI" and "December 2027"** as framings — use "near-frontier
high-capability AI" for the premium tier, defined operationally via
info_quality/breadth parameters.

**Concede the compute-scaling concern** by adding it as robustness: a
version where agents with more capital can deploy more AI compute on
hard problems. This addresses the rich-get-richer dynamic the editor
flagged.

### Game plan

**[WRITING] Remove AGI and 2027 language** (~1 day)
Global search-replace: "AGI" → "near-frontier AI" or "Tier 4 AI". Drop
December 2027 anchor. Reframe scaling assumptions as abstract capability
levels: info_quality ∈ {0.25, 0.43, 0.70, 0.97} defined by architectural
properties, not calendar dates.

**[CODE + CONFIG] Compute-scaling robustness** (~1 week)
Add `COMPUTE_SCALING` config mode where agents with more capital can buy
more inference per decision. Concrete implementation: premium agents'
info_quality rises sub-linearly in their capital (log scaling), capped at
0.97. Run at N=1000 and compare to fixed-tier. Report as Section 5.3
("Compute-scaling robustness").

**[WRITING] Defend fixed-tier as current-reality proxy** (~3 days)
Section 3.2 paragraph: *"We model AI as four discrete tiers matching
current commercial product segmentation: ChatGPT Free / Plus / Pro / API
(or equivalents). While AI pricing is expected to shift toward more
granular per-compute models (Anderssen 2026 public remarks on AGI pricing;
Brown 2025 on intelligence-per-token), the tiered subscription model
dominates current deployment (Maslej et al. 2026 AI Index) and is the
empirical baseline against which compute-scaling deviations should be
measured."*

---

## Cluster 5 — Knightian asymmetry (HIGH)

*Editor comment 4.*

### Underlying logic

The editor's sharpest theoretical point. If our premium AI can identify the
best opportunities cleanly and consistently, then *by our own framework*,
actor_ignorance has been compressed. Calling the whole situation "Knightian
uncertainty" while the central feature is AI-solved discovery is semantic
inflation.

The editor suggests a natural resolution: *Knightian uncertainty shifts from
discovery to execution/appropriation.* A cancer-solution AI doesn't tell you
who can win IP, regulatory approval, manufacturing at scale. These downstream
problems are genuinely Knightian. This is actually a stronger theoretical
move for us.

### What reviewers are really asking

Be honest about what AI compresses and what it doesn't. Don't claim AI
helps with Knightian uncertainty if your model has AI solving most of the
Knightian part.

### Our position

**Refine the theoretical framework.** Distinguish:
- *Discovery Knightian* — which opportunities exist, what's their expected
  value. AI can compress this substantially.
- *Execution Knightian* — can this opportunity be profitably realized under
  concrete conditions (IP, timing, team, capital, regulation)? AI can
  compress this partially.
- *Competitive Knightian* — what will happen when many AI-users act on
  similar assessments simultaneously? This is *genuinely unknowable* because
  the strategic space explodes combinatorially under N high-info agents. AI
  *cannot* compress this — and in fact, *increases it* because coordination
  on similar assessments amplifies emergent dynamics.

Our paper's real contribution is then: *AI compresses two Knightian dimensions
but amplifies the third* — competitive recursion. The "paradox" is that the
first-order benefit (compressed discovery) generates a second-order cost
(amplified competitive Knightian) that dominates under strong enough signal
correlation.

### Game plan

**[WRITING] Theory section restructure** (~1 week)
Section 2 rewrite with three-dimensional Knightian framework (discovery /
execution / competitive). AI compresses discovery (documented in our
info_quality parameter), compresses execution partially (through better
feasibility assessment), amplifies competitive (through signal correlation).
This is the theoretical contribution.

Cite: Knight 1921 canonical; Alvarez & Barney 2007 on discovery vs creation;
Townsend et al. 2018 + 2024 + 2025 for actor_ignorance; existing common-
information literature (Morris & Shin 2002, Angeletos & Pavan 2007) for the
competitive dimension.

**[WRITING] Reposition AI's epistemic role** (~2 days)
Paper currently implies AI "reduces Knightian uncertainty" broadly. Revise
to: "AI compresses discovery-stage and partial execution-stage Knightian
uncertainty while amplifying competitive-recursion Knightian uncertainty."
This refined claim is both more defensible and more interesting.

---

## Cluster 6 — Empirical grounding (HIGH)

*Reviewer 1 comments 3, 4; Reviewer 2 comment 2 (partial).*

### Underlying logic

R1 is a methodology hawk. They've noticed:
- Trait distributions cite theoretical papers (Lumpkin & Dess 1996 is a
  conceptual paper on entrepreneurial orientation; you can't derive lognormal
  parameters from it)
- Clark et al. 2024 is missing from the reference list
- BLS 5-year survival is a population-mismatch: retail and food-service
  businesses dominate, and those agents don't resemble our $2.5-10M
  technology ventures

This reads as fishing for empirical anchors. The fix is partly substantive
(better sources) and partly about acknowledging where we're using
theoretical rather than empirical calibration.

### What reviewers are really asking

1. Trait distributions either better-grounded or dropped
2. Survival benchmark matched to focal population
3. Reference list audit

### Our position

**BLS benchmark — defensible with care.** Our model is about "founder-run
ventures with meaningful startup capital," which BLS does include. But R1 is
right that the population isn't restricted to tech. **Concession:** add Y
Combinator 5-year survival as secondary benchmark.

**Trait distributions — concede some, defend some.** analytical_ability,
competence, risk_tolerance can be grounded in Big Five / cognitive
psychometric data. innovativeness / ai_trust / exploration_tendency are
harder. **Solution:** restrict paper's primary trait profile to
empirically-grounded traits (ties to Cluster 3 parsimony); document others
as robustness parameterizations.

### Game plan

**[CODE] Y Combinator calibration robustness** (~1 week)
Add a second calibration target: YC Demo Day 5-year survival rate (~30-50%
depending on cohort, from Y Combinator annual reports and Wiltbank 2014 on
angel-backed ventures). Run at N=1000 × 10 seeds. Report alongside BLS.

**[WRITING] Reference audit** (~2 days)
Fix Clark et al. 2024 citation. Verify every trait distribution has a
proper source. For traits without psychometric grounding, acknowledge
uniform/beta priors as robustness-only parameterizations (R1's actual ask).

**[WRITING] Trait-grounding table** (new Table 1)
Table showing for each trait: (a) distribution, (b) source, (c) whether
grounded in empirical psychometric data or theoretical literature.
Transparent about what's calibrated vs. assumed.

**[WRITING] Engage Qu et al. 2026 SMJ** (R2 minor 1, ~2 days)
Qu finds AI *increases* novel recombination. We find AI increases
innovation volume without improving quality. Both can be true: more novel
combinations can still fail more frequently in the market. Frame as
complementary evidence.

---

## Cluster 7 — Writing and structure (MEDIUM)

*Reviewer 3 all comments.*

### Underlying logic

R3 is a stylistic hawk. The critique is concrete and actionable:
- 42 pages violates ETP's 40-page guideline
- Over-citation: 2-3 line citation brackets
- Introduction starts theorizing instead of pitching
- Theory section mixes assumptions with innovations
- Methods section not didactic
- Contributions mostly repeat findings ("delta" missing)
- "Statistical evidence" not "significant relationships"

### What reviewers are really asking

Tighter, more readable manuscript that respects ETP's format and doesn't
exhaust reviewers. This is genuinely important given R3's observation that
other journals would desk-reject a 42-page submission.

### Our position

**Accept all of it.** R3's feedback is calibrated and actionable. Our v3.x
code work gives us cleaner mechanism descriptions to work with.

### Game plan

**[WRITING] Four-paragraph introduction** (~2 days)
Paragraphs: (1) phenomenon — AI adoption heterogeneity; (2) puzzle — why
don't all agents upgrade?; (3) solution preview — signal correlation +
congestion + Knightian residue; (4) contributions — three deltas (new
mechanism; dynamic-adoption equilibrium finding; methodological/
transparency).

**[WRITING] Theory section restructure** (~3 days)
Separate assumptions (AI tiers, shared-signal architecture, decision rules)
from theoretical innovations (three-dimensional Knightian, congestion
under correlated AI signals, dynamic adoption equilibrium). Assumptions go
first, innovations go second. End with explicit 1-3 predictions.

**[WRITING] Didactic methods section** (~4 days)
Open with 1-page ABM primer for non-simulation readers (cite Levinthal 1997
and Gavetti & Levinthal 2000 as models). Use "here's what the simulation
tries to capture, here's how it does it" structure. Push implementation
details to appendix.

**[WRITING] Citation audit** (~2 days)
Reduce citation brackets to ≤2 per claim where possible. Cut references
that don't directly support the immediate clause. Target: 40% citation
reduction, mostly in background paragraphs.

**[WRITING] Contribution reframing** (~2 days)
Each contribution statement answers: *"before this paper, scholars thought
X; we show Y; the delta is Z."* Aim for three clean deltas. Match intro
contributions claim exactly.

**[WRITING] Page budget** (ongoing)
Target 35 pages body text. Move to appendices:
- Full trait profile specification (Appendix B)
- Detailed algorithm pseudocode (Appendix C)
- Complete robustness tables (Appendix D)
- Reference mapping: each paper section → source code file (Appendix E)

**[WRITING] Trivial fixes** (~1 hour)
"significant relationships" → "significant statistical evidence";
verify no tracked-changes artifacts; submit anonymous version.

---

## Timeline — 12 weeks

### Weeks 1–3 — Foundation
- Week 1: reframing plan solidified with co-authors; draft abstract + new
  intro (Cluster 2); identify load-bearing traits (Cluster 3 ablation
  design)
- Week 2: submit ARC jobs — dynamic adoption at v3.3 (Cluster 2);
  trait ablation sweep (Cluster 3); strategic-premium v3.4a coding
  (Cluster 1)
- Week 3: v3.4a strategic premium runs; ablation results analyzed;
  Y Combinator calibration runs (Cluster 6)

### Weeks 4–7 — Writing and theory
- Week 4: theory section restructure (Cluster 5 three-dimensional
  Knightian); methods section didactic rewrite (Cluster 7)
- Week 5: results section reorganization with dynamic adoption first
  (Cluster 2); contributions reframe
- Week 6: reference audit + Qu et al. engagement; trait-grounding table;
  Levinthal-inspired methods section polish
- Week 7: full draft internal review; resolve outstanding modeling
  questions

### Weeks 8–10 — Robustness and polish
- Week 8: compute-scaling robustness (Cluster 4); mechanism decomposition
  figure
- Week 9: citation audit; page budget enforcement; figure/table
  rationalization
- Week 10: response letter drafting; point-by-point mapping

### Weeks 11–12 — Submission prep
- Week 11: response letter finalization; anonymous version; supplementary
  materials (code repo citation + reproduction guide)
- Week 12: final internal review; resubmit

---

## Defense positions — what we don't cave on

1. **Capital-saturation crowding is a real economic mechanism** (v3.1).
   Cite v3.2 sensitivity sweep showing monotone dependence on sizing
   aggressiveness.
2. **Bounded strategic reasoning is empirically correct for current LLM
   systems** — Chevallier et al. 2026 on LLM strategic errors. Our v3.4a
   strategic variant quantifies the sensitivity; doesn't abandon the
   assumption.
3. **The dynamic-adoption equilibrium matches empirical AI diffusion** —
   this is a finding, not an artifact.
4. **Some heterogeneity is defensible** — risk_tolerance, analytical_ability,
   competence are psychometrically grounded.
5. **BLS is defensible as primary benchmark with YC as robustness** —
   founder-run ventures across sectors, not just tech.
6. **The three-dimensional Knightian framework (discovery / execution /
   competitive) is a genuine theoretical contribution** — not just semantic
   rescue.

---

## What we fully concede

1. Drop AGI language and December 2027 anchoring
2. Accept dynamic adoption as the main finding
3. Simplify the presented model (4-5 load-bearing traits)
4. Submit source code as supplementary material
5. Cut manuscript to 35 pages body
6. Add Y Combinator calibration
7. Engage congestion game literature explicitly

---

## Risks to monitor

- **v3.4a strategic premium may preserve ordering but compress the gap.**
  That's actually good for us — it shows the finding survives strategic
  reasoning with reduced effect size. If the gap inverts under strategic
  reasoning, we need to be honest that bounded rationality is doing the
  work.

- **Ablation may reveal more decorative mechanisms than we expect.** Fine —
  that's a stronger paper. Resist the urge to keep mechanisms that don't
  carry weight.

- **Y Combinator calibration may require different survival thresholds,**
  shifting tier numbers. Acceptable — we report both BLS and YC with the
  ordering preserved.

- **Compute-scaling robustness may show rich-get-richer dominates at high
  capital variance.** If so, this is itself a theoretical insight to
  discuss in the revised paper, not a failure.

---

## Success criteria for revision

- All three reviewers' Tier 1 concerns addressed with concrete code or
  writing evidence
- Response letter cites specific file paths / line numbers / test cases
  for each major technical claim
- Paper body ≤ 35 pages, references ≤ 120
- New headline finding: dynamic-adoption equilibrium mechanism
- Preserved headline: AI-induced signal correlation is a novel mechanism
  for congestion externality in entrepreneurial contexts
- Source code publicly available with MIT license and reproduction
  scripts for every paper figure
- Ablation analysis showing core result survives with 4–5 traits and
  strategic-premium variant

---

*Plan drafted 2026-04-24 following R&R receipt. v1 — subject to revision
after co-author discussion.*
