"""
Knowledge base management for GlimpseABM.jl

Manages collective knowledge in the system, including:
- Knowledge piece storage and retrieval
- Agent knowledge portfolios
- Knowledge compatibility and combination
- Learning from successes and failures

Port of: glimpse_abm/knowledge.py
"""

using Random
using Statistics

# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

"""
Domain to sector mapping.
"""
const DEFAULT_DOMAIN_TO_SECTOR = Dict(
    "technology" => "tech",
    "market" => "retail",
    "process" => "service",
    "business_model" => "manufacturing"
)

"""
Knowledge adjacency matrix for compatibility scoring.
"""
const DOMAIN_ADJACENCY = Dict(
    ("technology", "market") => true,
    ("technology", "process") => true,
    ("technology", "business_model") => false,
    ("market", "technology") => true,
    ("market", "process") => false,
    ("market", "business_model") => true,
    ("process", "technology") => true,
    ("process", "market") => false,
    ("process", "business_model") => true,
    ("business_model", "technology") => false,
    ("business_model", "market") => true,
    ("business_model", "process") => true
)

"""
Manages collective knowledge in the system.
"""
mutable struct KnowledgeBase
    config::Union{EmergentConfig,Nothing}
    knowledge_pieces::Dict{String,Knowledge}
    domain_knowledge::Dict{String,Vector{Knowledge}}
    agent_knowledge::Dict{Int,Set{String}}
    knowledge_graph::Dict{String,Set{String}}
    ai_discovered_knowledge::Set{String}
    compatibility_cache::Dict{Tuple{String,String},Float64}
    knowledge_registry::Vector{Knowledge}
    knowledge_usage::Dict{String,Int}
    domain_to_sector::Dict{String,String}
    sector_to_domain::Dict{String,String}
end

"""
Create a new KnowledgeBase.
"""
function KnowledgeBase(config::Union{EmergentConfig,Nothing}=nothing)
    # Build domain-to-sector map based on available sectors
    available_sectors = !isnothing(config) && !isempty(config.SECTORS) ?
        collect(config.SECTORS) : collect(values(DEFAULT_DOMAIN_TO_SECTOR))

    domain_to_sector = Dict{String,String}()
    if length(available_sectors) == 1
        # Single sector mode - map all domains to it
        for domain in keys(DEFAULT_DOMAIN_TO_SECTOR)
            domain_to_sector[domain] = available_sectors[1]
        end
    else
        for (domain, default_sector) in DEFAULT_DOMAIN_TO_SECTOR
            if default_sector in available_sectors
                domain_to_sector[domain] = default_sector
            else
                domain_to_sector[domain] = available_sectors[1]
            end
        end
    end

    # Build reverse mapping
    sector_to_domain = Dict{String,String}()
    for (domain, sector) in domain_to_sector
        if !haskey(sector_to_domain, sector)
            sector_to_domain[sector] = domain
        end
    end

    kb = KnowledgeBase(
        config,
        Dict{String,Knowledge}(),
        Dict{String,Vector{Knowledge}}(),
        Dict{Int,Set{String}}(),
        Dict{String,Set{String}}(),
        Set{String}(),
        Dict{Tuple{String,String},Float64}(),
        Knowledge[],
        Dict{String,Int}(),
        domain_to_sector,
        sector_to_domain
    )

    # Initialize base knowledge
    initialize_base_knowledge!(kb)

    return kb
end

"""
Initialize base knowledge pieces.
"""
function initialize_base_knowledge!(kb::KnowledgeBase)
    base_domains = ["technology", "market", "process", "business_model"]

    for domain in base_domains
        for level in [0.1, 0.2, 0.3]
            knowledge = Knowledge(
                id="base_$(domain)_$(level)",
                domain=domain,
                level=level,
                discovered_round=0,
                discovered_by=nothing,
                parent_knowledge=String[]
            )
            add_knowledge!(kb, knowledge)
            kb.knowledge_usage[knowledge.id] = 0
        end
    end
end

"""
Add knowledge to the knowledge base.
"""
function add_knowledge!(kb::KnowledgeBase, knowledge::Knowledge; ai_discovered::Bool=false)
    kb.knowledge_pieces[knowledge.id] = knowledge

    # Add to domain index
    if !haskey(kb.domain_knowledge, knowledge.domain)
        kb.domain_knowledge[knowledge.domain] = Knowledge[]
    end
    push!(kb.domain_knowledge[knowledge.domain], knowledge)

    if ai_discovered
        push!(kb.ai_discovered_knowledge, knowledge.id)
    end

    # Update knowledge graph
    for parent_id in knowledge.parent_knowledge
        if !haskey(kb.knowledge_graph, parent_id)
            kb.knowledge_graph[parent_id] = Set{String}()
        end
        push!(kb.knowledge_graph[parent_id], knowledge.id)
    end

    # Update compatibility cache
    for other in kb.knowledge_registry
        score = compute_compatibility(knowledge, other)
        kb.compatibility_cache[(knowledge.id, other.id)] = score
        kb.compatibility_cache[(other.id, knowledge.id)] = score
    end
    kb.compatibility_cache[(knowledge.id, knowledge.id)] = 1.0

    push!(kb.knowledge_registry, knowledge)
    kb.knowledge_usage[knowledge.id] = get(kb.knowledge_usage, knowledge.id, 0)
end

"""
Compute compatibility between two knowledge pieces.
"""
function compute_compatibility(k1::Knowledge, k2::Knowledge)::Float64
    if k1.id == k2.id
        return 1.0
    end

    # Same domain: high compatibility
    if k1.domain == k2.domain
        level_diff = abs(k1.level - k2.level)
        return 0.8 + 0.2 * (1.0 - level_diff)
    end

    # Adjacent domains: medium compatibility
    is_adjacent = get(DOMAIN_ADJACENCY, (k1.domain, k2.domain), false)
    if is_adjacent
        level_diff = abs(k1.level - k2.level)
        return 0.4 + 0.2 * (1.0 - level_diff)
    end

    # Non-adjacent: low compatibility
    level_diff = abs(k1.level - k2.level)
    return 0.1 + 0.1 * (1.0 - level_diff)
end

"""
Get compatibility between two knowledge pieces (cached).
"""
function get_compatibility(kb::KnowledgeBase, k1::Knowledge, k2::Knowledge)::Float64
    if k1.id == k2.id
        return 1.0
    end

    cache_key = (k1.id, k2.id)
    if haskey(kb.compatibility_cache, cache_key)
        return kb.compatibility_cache[cache_key]
    end

    score = compute_compatibility(k1, k2)
    kb.compatibility_cache[cache_key] = score
    kb.compatibility_cache[(k2.id, k1.id)] = score
    return score
end

"""
Ensure agent has starter knowledge.
"""
function ensure_starter_knowledge!(kb::KnowledgeBase, agent_id::Int)
    if !haskey(kb.agent_knowledge, agent_id)
        kb.agent_knowledge[agent_id] = Set{String}()
    end

    starter_set = kb.agent_knowledge[agent_id]
    if length(starter_set) >= 2
        return
    end

    base_domains = ["technology", "market", "process", "business_model"]
    primary_domain = base_domains[(agent_id % length(base_domains)) + 1]

    # Add from primary domain
    domain_pieces = get(kb.domain_knowledge, primary_domain, Knowledge[])
    base_pieces = filter(k -> startswith(k.id, "base_"), domain_pieces)
    if !isempty(base_pieces)
        push!(starter_set, base_pieces[1].id)
    end

    if length(starter_set) >= 2
        return
    end

    # Add cross-domain knowledge
    base_candidates = [k.id for k in values(kb.knowledge_pieces) if startswith(k.id, "base_")]
    cross_domain = filter(kid -> !startswith(kid, "base_$(primary_domain)_") && !(kid in starter_set), base_candidates)
    if isempty(cross_domain)
        cross_domain = filter(kid -> !(kid in starter_set), base_candidates)
    end
    if !isempty(cross_domain)
        push!(starter_set, rand(cross_domain))
    end
end

"""
Get AI info quality signals based on level.
"""
function get_ai_info_signals(kb::KnowledgeBase, ai_level::String)::Tuple{Float64,Float64}
    if isnothing(kb.config)
        default_map = Dict(
            "none" => (0.0, 0.0),
            "basic" => (0.35, 0.30),
            "advanced" => (0.65, 0.55),
            "premium" => (0.9, 0.8)
        )
        return get(default_map, lowercase(ai_level), (0.0, 0.0))
    end

    profile = get(kb.config.AI_LEVELS, ai_level, kb.config.AI_LEVELS["none"])
    # AILevelConfig is a struct, access fields directly
    if profile isa AILevelConfig
        return (Float64(profile.info_quality), Float64(profile.info_breadth))
    else
        # Fallback for Dict-style access (legacy)
        return (Float64(get(profile, "info_quality", 0.0)), Float64(get(profile, "info_breadth", 0.0)))
    end
end

"""
Get accessible knowledge for an agent.
"""
function get_accessible_knowledge(
    kb::KnowledgeBase,
    agent_id::Int,
    ai_level::String="none";
    agent_traits::Union{Dict{String,Float64},Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng()
)::Vector{Knowledge}
    ensure_starter_knowledge!(kb, agent_id)
    agent_knowledge_ids = kb.agent_knowledge[agent_id]

    # Get AI bonuses
    info_quality, info_breadth = get_ai_info_signals(kb, ai_level)
    ai_knowledge_bonus = Dict("none" => 0.02, "basic" => 0.12, "advanced" => 0.28, "premium" => 0.42)
    bonus = get(ai_knowledge_bonus, ai_level, 0.0) + info_quality * 0.55 + info_breadth * 0.45

    # Add trait bonuses
    exploration_trait = 0.0
    if !isnothing(agent_traits)
        exploration_trait = get(agent_traits, "exploration_tendency", 0.0)
        bonus += exploration_trait * 0.08
    end

    # Get accessible pieces
    accessible = [kb.knowledge_pieces[kid] for kid in agent_knowledge_ids if haskey(kb.knowledge_pieces, kid)]

    # Potentially discover new knowledge
    other_knowledge = [k for k in kb.knowledge_registry if !(k.id in agent_knowledge_ids)]

    if !isempty(other_knowledge) && bonus > 0
        attempt_multiplier = 1.0 + info_breadth * 5.0 + exploration_trait * 0.4
        attempts = max(1, round(Int, attempt_multiplier))

        for _ in 1:attempts
            knowledge_piece = other_knowledge[rand(rng, 1:length(other_knowledge))]
            novelty_bias = 1.0 + info_quality * (knowledge_piece.level - 0.5)
            novelty_bias = clamp(novelty_bias, 0.35, 1.5)
            threshold = clamp(bonus * (1.0 - knowledge_piece.level) * novelty_bias, 0.0, 0.99)

            if threshold > 0 && !(knowledge_piece.id in agent_knowledge_ids)
                if rand(rng) < threshold
                    push!(agent_knowledge_ids, knowledge_piece.id)
                    push!(accessible, knowledge_piece)
                end
            end
        end
    end

    return accessible
end

"""
Record knowledge usage.
"""
function record_usage!(kb::KnowledgeBase, knowledge_ids::Vector{String})
    for kid in knowledge_ids
        kb.knowledge_usage[kid] = get(kb.knowledge_usage, kid, 0) + 1
    end
end

"""
Get combination scarcity score.
"""
function get_combination_scarcity(kb::KnowledgeBase, knowledge_ids::Vector{String})::Float64
    if isempty(knowledge_ids)
        return 0.5
    end

    scarcity_values = Float64[]
    for kid in knowledge_ids
        usage = get(kb.knowledge_usage, kid, 0)
        push!(scarcity_values, 1.0 / (1.0 + usage))
    end

    return clamp(mean(scarcity_values), 0.0, 1.0)
end

"""
Get average knowledge per agent.
"""
function get_average_agent_knowledge(kb::KnowledgeBase)::Float64
    if isempty(kb.agent_knowledge)
        return 0.0
    end

    counts = [length(kset) for kset in values(kb.agent_knowledge)]
    return isempty(counts) ? 0.0 : mean(counts)
end

"""
Learn from successful innovation.
"""
function learn_from_success!(
    kb::KnowledgeBase,
    agent_id::Int,
    innovation::Innovation
)
    # Add knowledge components to agent
    for kid in innovation.knowledge_components
        if !haskey(kb.agent_knowledge, agent_id)
            kb.agent_knowledge[agent_id] = Set{String}()
        end
        push!(kb.agent_knowledge[agent_id], kid)
    end

    # Create derived knowledge for high-quality innovations
    if innovation.success && innovation.quality > 0.7
        new_knowledge = create_derived_knowledge(kb, innovation)
        if !isnothing(new_knowledge)
            add_knowledge!(kb, new_knowledge; ai_discovered=innovation.ai_assisted)
            push!(kb.agent_knowledge[agent_id], new_knowledge.id)
        end
    end
end

"""
Learn from failed innovation.
"""
function learn_from_failure!(
    kb::KnowledgeBase,
    agent_id::Int,
    innovation::Innovation;
    rng::AbstractRNG=Random.default_rng()
)
    if !haskey(kb.agent_knowledge, agent_id)
        kb.agent_knowledge[agent_id] = Set{String}()
    end

    # Partial learning from failure
    for kid in innovation.knowledge_components
        if rand(rng) < 0.3
            push!(kb.agent_knowledge[agent_id], kid)
        end
    end

    # Create failure-derived knowledge
    if innovation.quality > 0.5 && rand(rng) < 0.2
        failure_knowledge = Knowledge(
            id="failure_$(innovation.id)",
            domain="process",
            level=0.3 + 0.2 * innovation.quality,
            discovered_round=innovation.round_created,
            discovered_by=agent_id,
            parent_knowledge=innovation.knowledge_components[1:min(2, length(innovation.knowledge_components))]
        )
        add_knowledge!(kb, failure_knowledge)
        push!(kb.agent_knowledge[agent_id], failure_knowledge.id)
    end
end

"""
Create derived knowledge from a successful innovation.
"""
function create_derived_knowledge(kb::KnowledgeBase, innovation::Innovation)::Union{Knowledge,Nothing}
    if length(innovation.knowledge_components) < 2
        return nothing
    end

    component_domains = String[]
    max_level = 0.0

    for kid in innovation.knowledge_components[1:min(3, length(innovation.knowledge_components))]
        if haskey(kb.knowledge_pieces, kid)
            k = kb.knowledge_pieces[kid]
            push!(component_domains, k.domain)
            max_level = max(max_level, k.level)
        end
    end

    if isempty(component_domains)
        return nothing
    end

    # Find dominant domain
    domain_counts = Dict{String,Int}()
    for d in component_domains
        domain_counts[d] = get(domain_counts, d, 0) + 1
    end
    new_domain = first(sort(collect(domain_counts), by=x->x[2], rev=true))[1]

    # Level advance based on innovation type
    level_advance = Dict(
        "incremental" => 0.1,
        "architectural" => 0.15,
        "radical" => 0.25,
        "disruptive" => 0.3
    )
    new_level = min(1.0, max_level + get(level_advance, innovation.type, 0.1))

    return Knowledge(
        id="derived_$(innovation.id)",
        domain=new_domain,
        level=new_level,
        discovered_round=innovation.round_created,
        discovered_by=innovation.creator_id,
        parent_knowledge=innovation.knowledge_components[1:min(3, length(innovation.knowledge_components))]
    )
end

"""
Get knowledge IDs for agent in a specific domain.
"""
function get_agent_domain_knowledge(kb::KnowledgeBase, agent_id::Int, domain::String)::Vector{String}
    knowledge_ids = get(kb.agent_knowledge, agent_id, Set{String}())
    matches = String[]

    for kid in knowledge_ids
        piece = get(kb.knowledge_pieces, kid, nothing)
        if !isnothing(piece) && piece.domain == domain
            push!(matches, kid)
        end
    end

    return matches
end

"""
Map knowledge piece to sector.
"""
function knowledge_to_sector(kb::KnowledgeBase, knowledge::Knowledge)::String
    return get(kb.domain_to_sector, knowledge.domain, "tech")
end

"""
Map sector to domain.
"""
function sector_to_domain(kb::KnowledgeBase, sector::String)::String
    return get(kb.sector_to_domain, sector, "process")
end

# ============================================================================
# KNOWLEDGE DECAY AND FORGETTING
# ============================================================================

"""
Apply hallucination penalty when AI misleads an agent.
Reduces effective knowledge trust in the affected domain.
"""
function apply_hallucination_penalty!(
    kb::KnowledgeBase,
    agent,  # EmergentAgent - untyped to avoid circular deps
    domain::String,
    severity::Float64=0.2
)
    resources = agent.resources
    if isnothing(resources)
        return
    end

    knowledge_map = resources.knowledge
    if isempty(knowledge_map)
        return
    end

    # Get available sectors from agent's knowledge
    available_sectors = collect(keys(knowledge_map))
    default_sector = !isempty(available_sectors) ? first(available_sectors) : "tech"

    # Domain to sector mapping
    base_domain_to_sector = Dict(
        "market_analysis" => "retail",
        "technical_assessment" => "tech",
        "uncertainty_evaluation" => "service",
        "innovation_potential" => "manufacturing"
    )

    domain_to_sector = Dict{String,String}()
    for (d, s) in base_domain_to_sector
        domain_to_sector[d] = s in available_sectors ? s : default_sector
    end

    target_sector = get(domain_to_sector, domain, nothing)
    if isnothing(target_sector) || !haskey(knowledge_map, target_sector)
        return
    end

    # Apply penalty to target sector
    penalty = clamp(0.1 + severity * 0.45, 0.1, 0.65)
    knowledge_map[target_sector] = max(0.01, knowledge_map[target_sector] * (1.0 - penalty))

    # Propagate lighter penalty to adjacent sectors
    neighbor_penalty = penalty * 0.35
    for sector in available_sectors
        if sector != target_sector
            knowledge_map[sector] = max(0.01, knowledge_map[sector] * (1.0 - neighbor_penalty))
        end
    end

    # Reduce cognitive capabilities
    if !isnothing(resources.capabilities)
        for key in ["uncertainty_management", "opportunity_evaluation"]
            if haskey(resources.capabilities, key)
                resources.capabilities[key] = max(0.01, resources.capabilities[key] * (1.0 - penalty * 0.25))
            end
        end
    end

    # Suppress domain trust in AI learning profile
    ai_learning = agent.ai_learning
    if !isnothing(ai_learning) && haskey(ai_learning.domain_trust, domain)
        ai_learning.domain_trust[domain] = max(0.0, ai_learning.domain_trust[domain] - penalty * 0.4)
    end

    # Trigger targeted forgetting
    current_level = agent.current_ai_level
    forget_sector_knowledge!(kb, agent, target_sector, severity; ai_level=current_level)
end

"""
Forget knowledge in a specific sector based on severity.
"""
function forget_sector_knowledge!(
    kb::KnowledgeBase,
    agent,
    sector::String,
    severity::Float64;
    ai_level::String="none"
)
    if isempty(sector)
        return
    end

    agent_knowledge = get(kb.agent_knowledge, agent.id, Set{String}())
    if isempty(agent_knowledge)
        return
    end

    # Get AI info quality to modulate severity
    ai_config = get(kb.config.AI_LEVELS, ai_level, kb.config.AI_LEVELS["none"])
    info_quality = Float64(get(ai_config, "info_quality", 0.0))
    severity = clamp(severity * (1.0 - 0.25 * info_quality), 0.05, 1.0)

    # Find pieces in this sector
    piece_ids = String[]
    for kid in collect(agent_knowledge)
        piece = get(kb.knowledge_pieces, kid, nothing)
        if !isnothing(piece) && knowledge_to_sector(kb, piece) == sector
            push!(piece_ids, kid)
        end
    end

    if isempty(piece_ids)
        return
    end

    # Sort by usage (drop least used)
    sort!(piece_ids, by=k -> get(kb.knowledge_usage, k, 0))

    drop_count = max(1, Int(round(length(piece_ids) * clamp(severity * 0.4, 0.08, 0.65))))
    for kid in piece_ids[1:min(drop_count, length(piece_ids))]
        remove_agent_knowledge!(kb, agent, kid)
    end
end

"""
Remove knowledge from an agent.
"""
function remove_agent_knowledge!(kb::KnowledgeBase, agent, knowledge_id::String)
    if haskey(kb.agent_knowledge, agent.id)
        delete!(kb.agent_knowledge[agent.id], knowledge_id)
    end
end

"""
Cull least-used knowledge when portfolios become too large.
"""
function forget_stale_knowledge!(
    kb::KnowledgeBase,
    agent,
    current_round::Int;
    max_size::Union{Int,Nothing}=nothing,
    drop_fraction::Float64=0.1
)
    knowledge_ids = collect(get(kb.agent_knowledge, agent.id, Set{String}()))
    if isempty(knowledge_ids)
        return
    end

    config_max = get(kb.config.parameters, "MAX_AGENT_KNOWLEDGE", 120)
    max_keep = isnothing(max_size) ? config_max : max_size
    if max_keep <= 0
        max_keep = 60
    end

    if length(knowledge_ids) <= max_keep
        return
    end

    drop_count = length(knowledge_ids) - max_keep
    drop_extra = max(1, Int(round(max_keep * drop_fraction)))
    total_drop = max(1, drop_count + drop_extra)

    # Score function for keeping knowledge
    function keep_score(kid::String)::Float64
        piece = get(kb.knowledge_pieces, kid, nothing)
        usage = get(kb.knowledge_usage, kid, 0)
        discovered_round = isnothing(piece) ? current_round : piece.discovered_round
        age = max(0, current_round - discovered_round)
        novelty_bonus = isnothing(piece) ? 0.0 : piece.level
        return usage * 1.6 + novelty_bonus - age * 0.04
    end

    sorted_ids = sort(knowledge_ids, by=keep_score)

    # Structured drop (lowest scores)
    structured_drop = max(1, Int(round(total_drop * 0.6)))
    to_remove = sorted_ids[1:min(structured_drop, length(sorted_ids))]

    # Random additional drops
    remaining = total_drop - length(to_remove)
    if remaining > 0
        random_pool = [kid for kid in knowledge_ids if !(kid in to_remove)]
        if !isempty(random_pool)
            n_random = min(length(random_pool), remaining)
            random_remove = Random.shuffle(random_pool)[1:n_random]
            append!(to_remove, random_remove)
        end
    end

    # Remove knowledge
    for kid in to_remove[1:min(total_drop, length(to_remove))]
        remove_agent_knowledge!(kb, agent, kid)
    end
end

"""
Prune knowledge based on sector strength thresholds.
"""
function prune_by_sector_strength!(
    kb::KnowledgeBase,
    agent,
    sector_levels::Dict{String,Float64};
    threshold::Float64=0.05
)
    if isnothing(agent) || isempty(sector_levels)
        return
    end

    knowledge_ids = collect(get(kb.agent_knowledge, agent.id, Set{String}()))
    if isempty(knowledge_ids)
        return
    end

    for kid in knowledge_ids
        piece = get(kb.knowledge_pieces, kid, nothing)
        if isnothing(piece)
            continue
        end
        sector = knowledge_to_sector(kb, piece)
        if get(sector_levels, sector, 1.0) < threshold
            remove_agent_knowledge!(kb, agent, kid)
        end
    end
end

"""
Apply sector-specific knowledge decay to agent resources.
Uses calibrated decay rates from SectorProfile (based on Ebbinghaus forgetting curve
and industry skill depreciation research).

Decay rates by sector (per round):
- Tech: 0.12 (2-3 year half-life, fast obsolescence)
- Retail: 0.07 (4-5 year half-life)
- Service: 0.05 (5-7 year half-life, stable expertise)
- Manufacturing: 0.03 (7-10 year half-life, most durable)
"""
function apply_sector_decay!(
    kb::KnowledgeBase,
    agent,  # EmergentAgent - untyped to avoid circular deps
    current_round::Int;
    usage_weights::Dict{String,Float64}=Dict{String,Float64}(),
    rng::AbstractRNG=Random.default_rng()
)
    if isnothing(agent.resources)
        return
    end

    resources = agent.resources
    knowledge_map = resources.knowledge
    if isempty(knowledge_map)
        return
    end

    for sector in collect(keys(knowledge_map))
        # Get sector-specific decay rate from SectorProfile
        sector_profile = get(kb.config.SECTOR_PROFILES, sector, nothing)
        sector_decay_rate = if !isnothing(sector_profile) && hasproperty(sector_profile, :knowledge_decay_rate)
            sector_profile.knowledge_decay_rate
        else
            0.075  # Default fallback
        end

        # Activity modifier - recent usage reduces decay
        usage_factor = clamp(get(usage_weights, sector, 0.0), 0.0, 1.0)
        activity_mod = 1.0 - 0.5 * usage_factor

        # Add some noise
        noise = clamp(randn(rng) * 0.1 + 1.0, 0.7, 1.3)

        # Calculate effective decay rate
        effective_decay = sector_decay_rate * activity_mod * noise
        effective_decay = clamp(effective_decay, 0.0, 0.9)

        # Apply decay
        knowledge_map[sector] = max(0.01, knowledge_map[sector] * (1.0 - effective_decay))
    end
end

"""
Apply tier-specific knowledge decay.
"""
function apply_tier_decay!(
    kb::KnowledgeBase,
    agent_id::Int,
    ai_level::String;
    rng::AbstractRNG=Random.default_rng()
)
    tier = normalize_ai_label(ai_level)

    # Get AI info signals
    ai_config = get(kb.config.AI_LEVELS, ai_level, kb.config.AI_LEVELS["none"])
    info_quality = Float64(get(ai_config, "info_quality", 0.0))

    # Decay probabilities by tier
    decay_map = Dict("none" => 0.08, "basic" => 0.05, "advanced" => 0.025, "premium" => 0.0)
    drop_prob = get(decay_map, tier, 0.04)

    # Higher AI quality reduces decay
    retention_modifier = clamp(1.0 - 0.45 * info_quality, 0.2, 1.0)
    drop_prob *= retention_modifier

    knowledge_ids = get(kb.agent_knowledge, agent_id, Set{String}())
    if isempty(knowledge_ids) || drop_prob <= 0.0
        return
    end

    if rand(rng) >= drop_prob
        return
    end

    # Only drop non-base knowledge
    drop_candidates = [kid for kid in knowledge_ids if !startswith(kid, "base_")]
    if isempty(drop_candidates)
        return
    end

    drop_count = max(1, Int(round(length(drop_candidates) * max(0.05, drop_prob))))
    n_remove = min(length(drop_candidates), drop_count)
    removed = Random.shuffle(collect(drop_candidates))[1:n_remove]

    for kid in removed
        delete!(knowledge_ids, kid)
    end
end

"""
Update domain belief using Bayesian update.
"""
function update_domain_belief!(
    kb::KnowledgeBase,
    agent_id::Int,
    domain::String,
    evidence::Float64
)::Float64
    evidence = clamp(evidence, 0.0, 1.0)

    if !haskey(kb.agent_domain_beliefs, agent_id)
        kb.agent_domain_beliefs[agent_id] = Dict{String,Dict{String,Float64}}()
    end

    if !haskey(kb.agent_domain_beliefs[agent_id], domain)
        kb.agent_domain_beliefs[agent_id][domain] = Dict("alpha" => 2.0, "beta" => 2.0)
    end

    belief = kb.agent_domain_beliefs[agent_id][domain]
    belief["alpha"] += evidence
    belief["beta"] += max(0.0, 1.0 - evidence)

    total = belief["alpha"] + belief["beta"]
    return total <= 0 ? 0.5 : belief["alpha"] / total
end

"""
Get current domain belief for an agent.
"""
function get_domain_belief(kb::KnowledgeBase, agent_id::Int, domain::String)::Float64
    if !haskey(kb.agent_domain_beliefs, agent_id)
        return 0.5
    end

    belief = get(kb.agent_domain_beliefs[agent_id], domain, nothing)
    if isnothing(belief)
        return 0.5
    end

    total = belief["alpha"] + belief["beta"]
    return total <= 0 ? 0.5 : belief["alpha"] / total
end
