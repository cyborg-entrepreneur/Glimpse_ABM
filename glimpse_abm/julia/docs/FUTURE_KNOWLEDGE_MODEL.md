# Future Enhancement: Differentiated Knowledge Model

## Overview

This document describes a proposed enhancement to the GlimpseABM knowledge acquisition system. The current model treats knowledge as a single scalar value per sector. This enhancement would differentiate knowledge into four distinct types, each with different AI-tier interactions.

**Target Implementation:** Forked version for future research
**Priority:** Medium (core paradox findings do not depend on this)
**Complexity:** Moderate

---

## Current Limitation

The existing model uses a simple scalar knowledge representation:

```julia
# Current: Single scalar per sector (0.0 to 1.0)
agent.resources.knowledge[sector] = min(1.0, current_knowledge + knowledge_gain)

# AI tier affects learning speed uniformly
breadth_multiplier = 0.7 + 0.6 * info_breadth  # 0.82 (none) to 1.25 (premium)
knowledge_gain = rand(Uniform(0.05, 0.15)) * breadth_multiplier
```

This simplification ignores that real expertise comprises multiple dimensions with heterogeneous AI effects.

---

## Proposed Knowledge Architecture

### Four Knowledge Types

| Type | Description | AI Effect | Rationale |
|------|-------------|-----------|-----------|
| **Codified** | Facts, data, documented procedures | 5-10x faster | AI excels at retrieval, synthesis, pattern matching in structured data |
| **Tacit** | Intuition, judgment, "feel" for market | Neutral (1x) | Requires lived experience; AI cannot shortcut embodied learning |
| **Relational** | Network contacts, trust relationships | 0.7-0.9x (penalty) | AI interaction may reduce human networking; over-reliance weakens relationships |
| **Procedural** | Execution skills, operational know-how | 1.0-1.2x (slight boost) | AI can assist with checklists/processes but execution remains human |

### Knowledge Struct

```julia
"""
Differentiated knowledge representation for a single sector.
"""
@kwdef mutable struct SectorKnowledge
    codified::Float64 = 0.1      # Facts, data, documented info
    tacit::Float64 = 0.1         # Intuition, judgment, pattern recognition
    relational::Float64 = 0.1    # Network strength, trust relationships
    procedural::Float64 = 0.1    # Execution skills, operational ability

    # Metadata
    last_updated::Int = 0
    primary_source::String = "experience"  # "experience", "ai_assisted", "network"
end

"""
Compute effective knowledge score for decision-making.
Weights reflect relative importance for entrepreneurial success.
"""
function effective_knowledge(sk::SectorKnowledge)::Float64
    # Weights based on entrepreneurship literature
    # Tacit and relational knowledge are often most predictive of success
    w_codified = 0.20      # Necessary but not sufficient
    w_tacit = 0.35         # Critical for judgment under uncertainty
    w_relational = 0.30    # Networks provide resources, information, legitimacy
    w_procedural = 0.15    # Execution matters but is more learnable

    return w_codified * sk.codified +
           w_tacit * sk.tacit +
           w_relational * sk.relational +
           w_procedural * sk.procedural
end
```

---

## AI Tier Effects on Learning

### Learning Rate Multipliers by Knowledge Type

```julia
"""
AI tier multipliers for each knowledge type.
Based on realistic assessment of AI capabilities.
"""
const AI_KNOWLEDGE_MULTIPLIERS = Dict(
    "none" => Dict(
        "codified" => 1.0,      # Human baseline
        "tacit" => 1.0,         # Human baseline
        "relational" => 1.0,    # Human baseline
        "procedural" => 1.0     # Human baseline
    ),
    "basic" => Dict(
        "codified" => 2.5,      # Basic search/synthesis assistance
        "tacit" => 1.0,         # No effect on intuition
        "relational" => 0.95,   # Slight reduction (less human interaction)
        "procedural" => 1.05    # Minor assistance with processes
    ),
    "advanced" => Dict(
        "codified" => 5.0,      # Strong research assistance
        "tacit" => 1.0,         # Still no effect
        "relational" => 0.85,   # More AI reliance = less networking
        "procedural" => 1.10    # Better process optimization
    ),
    "premium" => Dict(
        "codified" => 10.0,     # Near-instant access to codified knowledge
        "tacit" => 1.0,         # Cannot shortcut experience
        "relational" => 0.70,   # Significant network atrophy risk
        "procedural" => 1.15    # AI-assisted execution planning
    )
)
```

### Rationale for Multipliers

**Codified Knowledge (1x to 10x)**
- AI dramatically accelerates access to documented information
- Premium AI can synthesize research papers, market reports, competitor analysis
- Limitation: Access != understanding (addressed by quality discount below)

**Tacit Knowledge (1x across all tiers)**
- "Knowing that" vs "knowing how" distinction (Polanyi, 1966)
- Tacit knowledge requires repeated experience and reflection
- AI cannot substitute for learning when to trust your gut
- Research: Expert intuition develops through 10,000+ hours of deliberate practice (Ericsson)

**Relational Knowledge (1x to 0.7x penalty for premium)**
- Higher AI tiers may reduce human interaction
- "Bowling Alone" effect: technology substitution for social capital
- Networks require maintenance; AI usage may crowd out networking time
- Empirical: Studies show heavy tech users have weaker professional networks

**Procedural Knowledge (1x to 1.15x slight boost)**
- AI can help with checklists, SOPs, project management
- But execution still requires human action
- Boost is modest because AI assists planning, not doing

---

## Knowledge Quality and Hallucination Effects

### Quality-Adjusted Learning

AI-assisted learning should include a quality discount based on hallucination risk:

```julia
"""
Compute quality-adjusted knowledge gain for AI-assisted learning.
Higher AI tiers learn faster but with potential accuracy issues.
"""
function ai_adjusted_knowledge_gain(
    base_gain::Float64,
    ai_tier::String,
    knowledge_type::String,
    config::EmergentConfig
)::Tuple{Float64, Float64}  # (actual_gain, perceived_gain)

    # Get AI tier config
    ai_config = config.AI_LEVELS[ai_tier]
    domain_cap = config.AI_DOMAIN_CAPABILITIES[ai_tier]

    # Speed multiplier
    speed_mult = AI_KNOWLEDGE_MULTIPLIERS[ai_tier][knowledge_type]
    raw_gain = base_gain * speed_mult

    # Quality discount for codified knowledge (hallucination risk)
    if knowledge_type == "codified" && ai_tier != "none"
        hallucination_rate = domain_cap.hallucination_rate
        accuracy = 1.0 - hallucination_rate

        # Actual knowledge gained is discounted by accuracy
        actual_gain = raw_gain * accuracy

        # But agent PERCEIVES full gain (overconfidence)
        perceived_gain = raw_gain

        return (actual_gain, perceived_gain)
    end

    # Other knowledge types: no quality discount
    return (raw_gain, raw_gain)
end
```

### Implications

This creates a subtle but important dynamic:
- Premium AI users accumulate codified knowledge 10x faster
- But ~3-8% of that "knowledge" is actually misinformation (hallucinations)
- Agents don't know which knowledge is corrupted
- Over time, this creates divergence between perceived and actual expertise

---

## Knowledge Decay Mechanisms

### Differential Decay Rates

Different knowledge types decay at different rates:

```julia
"""
Apply knowledge decay based on type and usage.
"""
function decay_knowledge!(sk::SectorKnowledge, current_round::Int, config::EmergentConfig)
    rounds_since_use = current_round - sk.last_updated

    # Decay rates per round of non-use
    decay_rates = Dict(
        "codified" => 0.02,     # Facts fade without reinforcement
        "tacit" => 0.005,       # Intuition is sticky (embodied)
        "relational" => 0.03,   # Networks require maintenance
        "procedural" => 0.01   # Skills decay slowly
    )

    # AI-acquired knowledge decays faster (shallow learning)
    ai_penalty = sk.primary_source == "ai_assisted" ? 1.5 : 1.0

    if rounds_since_use > 6  # 6 months of non-use
        sk.codified *= (1.0 - decay_rates["codified"] * ai_penalty)
        sk.tacit *= (1.0 - decay_rates["tacit"])  # No AI penalty for tacit
        sk.relational *= (1.0 - decay_rates["relational"] * ai_penalty)
        sk.procedural *= (1.0 - decay_rates["procedural"])
    end

    # Floor values
    sk.codified = max(0.05, sk.codified)
    sk.tacit = max(0.05, sk.tacit)
    sk.relational = max(0.05, sk.relational)
    sk.procedural = max(0.05, sk.procedural)
end
```

### Rationale for Decay Rates

- **Codified (2%/month)**: Facts are easily forgotten without use
- **Tacit (0.5%/month)**: "Riding a bike" - embodied knowledge persists
- **Relational (3%/month)**: Networks weaken quickly without maintenance
- **Procedural (1%/month)**: Skills rust but don't disappear

The 1.5x AI penalty for decay reflects "shallow learning" - knowledge acquired quickly through AI assistance may not be as deeply encoded as knowledge gained through struggle and experience.

---

## Integration with Decision-Making

### Updated Opportunity Scoring

```julia
"""
Score opportunity using differentiated knowledge.
"""
function score_with_differentiated_knowledge(
    agent::EmergentAgent,
    opportunity::Opportunity,
    base_score::Float64
)::Float64

    sector_knowledge = get(agent.resources.knowledge, opportunity.sector, default_sector_knowledge())

    # Different knowledge types matter differently for different decisions

    # Codified: Helps identify opportunity characteristics
    codified_bonus = sector_knowledge.codified * 0.3

    # Tacit: Critical for judging true potential vs. hype
    tacit_bonus = sector_knowledge.tacit * 0.5

    # Relational: Access to deal flow, partnerships, talent
    relational_bonus = sector_knowledge.relational * 0.4

    # Procedural: Execution confidence
    procedural_bonus = sector_knowledge.procedural * 0.2

    # Combined effect (multiplicative, centered at 1.0)
    knowledge_multiplier = 1.0 + (codified_bonus + tacit_bonus + relational_bonus + procedural_bonus) / 4

    return base_score * knowledge_multiplier
end
```

### Investment Success Probability

```julia
"""
Knowledge affects success probability differently by type.
"""
function knowledge_success_modifier(
    sector_knowledge::SectorKnowledge,
    opportunity::Opportunity
)::Float64

    # Tacit knowledge is most predictive of success
    # (Knowing WHEN to act, not just WHAT to do)
    tacit_effect = 0.3 * sector_knowledge.tacit

    # Relational knowledge provides resources and support
    relational_effect = 0.25 * sector_knowledge.relational

    # Procedural knowledge affects execution
    procedural_effect = 0.2 * sector_knowledge.procedural

    # Codified knowledge has diminishing returns
    # (Everyone can access the same information)
    codified_effect = 0.1 * log1p(sector_knowledge.codified * 5)

    return 1.0 + tacit_effect + relational_effect + procedural_effect + codified_effect
end
```

---

## Implementation Checklist

### Phase 1: Data Structures
- [ ] Create `SectorKnowledge` struct
- [ ] Update `AgentResources` to use `Dict{String, SectorKnowledge}`
- [ ] Add `AI_KNOWLEDGE_MULTIPLIERS` to config
- [ ] Create knowledge initialization functions

### Phase 2: Learning Mechanics
- [ ] Update `_execute_explore!` to differentiate knowledge gains
- [ ] Implement `ai_adjusted_knowledge_gain` with quality discount
- [ ] Add knowledge decay in round processing
- [ ] Track `primary_source` for AI vs. experience learning

### Phase 3: Decision Integration
- [ ] Update opportunity scoring to use `effective_knowledge`
- [ ] Modify success probability calculations
- [ ] Add perceived vs. actual knowledge divergence tracking

### Phase 4: Validation
- [ ] Unit tests for each knowledge type
- [ ] Integration tests for learning/decay dynamics
- [ ] Sensitivity analysis on multiplier values
- [ ] Compare results to current scalar model

### Phase 5: Analysis Extensions
- [ ] Add knowledge composition to output data
- [ ] Create visualizations for knowledge trajectories by type
- [ ] Analyze how knowledge composition predicts survival
- [ ] Test whether AI paradox persists/strengthens with differentiated model

---

## Expected Effects on AI Paradox

This enhancement would likely **strengthen** the paradox finding:

1. **Codified knowledge homogenization**: Premium AI users all acquire similar codified knowledge very quickly, intensifying crowding

2. **Relational knowledge atrophy**: Premium AI users lose networking advantage, reducing access to differentiated deal flow

3. **Tacit knowledge parity**: No AI advantage in judgment means Premium users aren't actually better decision-makers despite feeling more informed

4. **Overconfidence amplification**: Gap between perceived knowledge (high) and actual effective knowledge (moderate) is larger for Premium users

The differentiated model would provide a more mechanistically satisfying explanation for why better AI doesn't translate to better outcomes.

---

## Literature Support

- **Polanyi, M. (1966)**. *The Tacit Dimension*. The classic distinction between explicit and tacit knowledge.

- **Nonaka, I. & Takeuchi, H. (1995)**. *The Knowledge-Creating Company*. SECI model of knowledge conversion.

- **Ericsson, K.A. (2006)**. *The Cambridge Handbook of Expertise and Expert Performance*. Deliberate practice and intuition development.

- **Granovetter, M. (1973)**. "The Strength of Weak Ties". Network effects on information access.

- **Shane, S. (2000)**. "Prior Knowledge and the Discovery of Entrepreneurial Opportunities". Knowledge heterogeneity in opportunity recognition.

- **Sarasvathy, S. (2001)**. "Causation and Effectuation". Expert entrepreneurs rely on tacit/relational knowledge.

---

## Notes

- The multiplier values (10x for codified, 0.7x for relational) are illustrative. Calibration should be informed by empirical studies of AI tool usage patterns.

- Consider adding interaction effects: e.g., high codified + low tacit might actually *reduce* effective knowledge (overconfident novice effect).

- The model could be extended to include knowledge transfer between agents (social learning), with AI affecting the efficiency of such transfers differently by knowledge type.
