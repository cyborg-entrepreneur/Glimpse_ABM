"""
Main simulation orchestrator for GlimpseABM.jl

This module implements the EmergentSimulation that coordinates agents,
market, and uncertainty environments.

Port of: glimpse_abm/simulation.py
"""

using Random
using Statistics
using DataFrames
using Dates
using SHA

# ============================================================================
# EMERGENT SIMULATION
# ============================================================================

"""
Main simulation class that orchestrates the agent-based model.
"""
mutable struct EmergentSimulation
    config::EmergentConfig
    agents::Vector{EmergentAgent}
    market::MarketEnvironment
    uncertainty_env::KnightianUncertaintyEnvironment
    knowledge_base::KnowledgeBase
    innovation_engine::InnovationEngine
    info_system::InformationSystem  # AI information generation system
    current_round::Int
    history::Vector{Dict{String,Any}}
    run_id::String
    output_dir::String
    rng::Random.AbstractRNG
    start_time::DateTime
    # Uncertainty transformation tracking
    previous_uncertainty_levels::Dict{String,Float64}
    baseline_uncertainty_levels::Dict{String,Float64}
end

"""
Initialize a new simulation.
"""
function EmergentSimulation(;
    config::EmergentConfig = EmergentConfig(),
    output_dir::String = "results",
    run_id::String = "run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    seed::Union{Int,Nothing} = nothing,
    initial_tier_distribution::Union{Dict{String,Float64},Nothing} = nothing
)
    # Initialize configuration
    initialize!(config)

    # Set up RNG - matches Python's seed derivation with run_id hashing
    # Python: seed = (base_seed + SHA256(run_id) % 1_000_000) % (2^32 - 1)
    actual_seed = if !isnothing(seed)
        seed
    elseif !isnothing(config.RANDOM_SEED) && config.RANDOM_SEED > 0
        # Deterministic hash (avoid Julia's randomized hash)
        run_hash = reinterpret(UInt64, sha256(run_id)[1:8])[1] % 1_000_000
        mod(config.RANDOM_SEED + run_hash, 2^32 - 1)
    else
        # Fallback to random seed
        rand(1:2^31-1)
    end

    # Create RNG - use NumpyRNG for cross-language reproducibility if configured
    rng = if config.USE_NUMPY_RNG
        NumpyRNG(actual_seed)
    else
        MersenneTwister(actual_seed)
    end

    # Create uncertainty environment
    uncertainty_env = KnightianUncertaintyEnvironment(config; rng=rng)

    # Create market environment
    market = MarketEnvironment(config; rng=rng)

    # Create knowledge base and innovation engine (matches Python)
    knowledge_base = KnowledgeBase(config)
    combination_tracker = CombinationTracker()
    innovation_engine = InnovationEngine(config, knowledge_base, combination_tracker)

    # Create information system for AI-assisted analysis
    info_system = InformationSystem(config)

    # Determine initial AI tier distribution
    # Default: 100% none. Can specify e.g. Dict("none"=>0.25, "basic"=>0.25, "advanced"=>0.25, "premium"=>0.25)
    tier_order = ["none", "basic", "advanced", "premium"]
    tier_probs = if isnothing(initial_tier_distribution)
        [1.0, 0.0, 0.0, 0.0]  # Default: all start at none
    else
        [get(initial_tier_distribution, t, 0.0) for t in tier_order]
    end
    # Normalize probabilities
    total_prob = sum(tier_probs)
    if total_prob > 0
        tier_probs = tier_probs ./ total_prob
    else
        tier_probs = [1.0, 0.0, 0.0, 0.0]
    end

    # Create agents with distributed initial tiers. Two modes:
    # - AGENT_AI_MODE="fixed" (main paper analyses): lock each agent at their
    #   sampled tier by setting fixed_ai_level=<tier>. get_ai_level() returns
    #   this, choose_ai_level is never called, tier is permanent.
    # - AGENT_AI_MODE="emergent" (robustness checks): leave fixed_ai_level=nothing
    #   so make_decision!'s else-branch (agents.jl:2210) calls choose_ai_level
    #   each round. current_ai_level starts at the sampled initial_tier but
    #   evolves dynamically.
    # Previously the constructor unconditionally set fixed_ai_level=initial_tier,
    # which locked emergent-mode agents at their starting tier (even when
    # initial_tier="none") — emergent mode was effectively disabled.
    agent_ai_mode = getfield_default(config, :AGENT_AI_MODE, "fixed")
    agents = EmergentAgent[]
    for i in 1:config.N_AGENTS
        # Sample initial tier based on distribution
        r = rand(rng)
        cumsum = 0.0
        initial_tier = "none"
        for (j, tier) in enumerate(tier_order)
            cumsum += tier_probs[j]
            if r <= cumsum
                initial_tier = tier
                break
            end
        end
        fixed_kw = agent_ai_mode == "emergent" ? nothing : initial_tier
        agent = EmergentAgent(i, config; rng=rng, fixed_ai_level=fixed_kw)
        # Emergent agents start at the sampled tier but can switch; fixed
        # agents stay at initial_tier permanently.
        if agent_ai_mode == "emergent" && initial_tier != "none"
            agent.current_ai_level = initial_tier
        end
        # Initialize subscription schedule for the starting tier (fixed-tier
        # agents will use this tier all run; emergent agents may later cancel
        # and start different ones via ensure_subscription_schedule!).
        if initial_tier != "none"
            ensure_subscription_schedule!(agent, initial_tier)
        end
        push!(agents, agent)
    end

    return EmergentSimulation(
        config,
        agents,
        market,
        uncertainty_env,
        knowledge_base,
        innovation_engine,
        info_system,
        0,
        Dict{String,Any}[],
        run_id,
        output_dir,
        rng,
        now(),
        Dict{String,Float64}(),  # previous_uncertainty_levels
        Dict{String,Float64}()   # baseline_uncertainty_levels
    )
end

"""
Initialize agents with a fixed AI level (for causal analysis).
"""
function initialize_agents!(sim::EmergentSimulation; fixed_ai_level::Union{String,Nothing} = nothing)
    for agent in sim.agents
        if !isnothing(fixed_ai_level)
            agent.fixed_ai_level = fixed_ai_level
            agent.current_ai_level = fixed_ai_level
            # Start subscription schedule if this is a subscription tier
            ensure_subscription_schedule!(agent, fixed_ai_level)
        end
    end
end

"""
Run the full simulation.
"""
function run!(sim::EmergentSimulation)
    println("[$(sim.run_id)] Starting simulation...")

    for round in 1:sim.config.N_ROUNDS
        step!(sim, round)

        # Log progress periodically
        if sim.config.enable_round_logging && round % sim.config.round_log_interval == 0
            alive_count = count(a -> a.alive, sim.agents)
            capitals = [get_capital(a) for a in sim.agents if a.alive]
            mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
            # println("[$(sim.run_id)] Round $round: $(alive_count)/$(sim.config.N_AGENTS) alive, mean capital: \$$(round(Int, mean_capital))")
        end
    end

    println("[$(sim.run_id)] Simulation finished.")
    return sim
end

"""
Execute a single simulation round.
Matches Python _step order: operating costs -> matured investments -> decisions -> survival checks
"""
function step!(sim::EmergentSimulation, round::Int)
    sim.current_round = round
    # Stamp the market with the current round BEFORE discovery runs so that
    # opportunities discovered this round get the correct discovery_round.
    # Previously discovery-phase writes used the stale market.current_round
    # from the prior step (or 0 at start).
    sim.market.current_round = round

    # Invalidate the InformationSystem cache at the start of each round.
    # The cache key is (opp.id, ai_level, agent_id) but opportunity state
    # (competition, lifecycle_stage, disrupted_count) evolves across rounds,
    # so stale estimates from round N-1 leak into round N decisions. Clearing
    # here forces a fresh Information draw per round per agent.
    empty!(sim.info_system.information_cache)

    # Get current uncertainty state and market conditions.
    # v3.0: MarketConditions is now an immutable typed struct — uncertainty_state
    # is injected at construction (was mutated onto a Dict after creation).
    uncertainty_state = get_uncertainty_state(sim.uncertainty_env)
    market_conditions = get_market_conditions(sim.market; uncertainty_state=uncertainty_state)

    # Get alive agents
    alive_agents = filter(a -> a.alive, sim.agents)

    # Phase 1: Apply operational costs FIRST with severity multiplier (matching Python order)
    # Python applies this before matured investments and decisions
    avg_comp = market_conditions.avg_competition
    volatility = market_conditions.volatility
    base_vol = Float64(sim.config.MARKET_VOLATILITY)
    severity = 1.0 + avg_comp * 0.35 + max(0.0, volatility - base_vol) * 0.45
    severity = clamp(severity, 0.7, 1.9)

    for agent in sim.agents
        if !agent.alive
            continue
        end
        # Calculate base cost
        estimated_cost = estimate_operational_costs(agent, sim.market)
        # Apply severity multiplier (matching Python _apply_operating_costs)
        operating_cost = max(0.0, estimated_cost * severity)
        agent.operating_cost_estimate = operating_cost
        if operating_cost > 0.0
            set_capital!(agent, get_capital(agent) - operating_cost)
            # Check survival after operating costs (matches Python line 189)
            check_survival!(agent, round)
        end
    end

    # Phase 1.5: Charge AI subscription installments (matching Python _apply_subscription_carry)
    for agent in sim.agents
        if !agent.alive
            continue
        end
        subscription_cost = apply_subscription_carry!(agent, round)
        # Check survival after subscription charges
        if subscription_cost > 0.0
            check_survival!(agent, round)
        end
    end

    # Phase 2: Process matured investments (matching Python BLOCK 1)
    # Pass market_conditions with uncertainty_state (matches Python line 913)
    all_matured = Dict{String,Any}[]
    for agent in sim.agents
        if !agent.alive
            continue
        end
        matured = process_matured_investments!(agent, sim.market, round; market_conditions=market_conditions)
        for m in matured
            # Update tier beliefs based on investment outcomes
            ai_tier = get(m, "ai_level", get_ai_level(agent))
            success = get(m, "success", false)
            update_tier_belief!(agent.ai_learning, ai_tier, success)
            # v2.7: wire update_state_from_outcome! so AI-trust and
            # experience-driven trait evolution actually run. The function
            # was defined (agents.jl:update_state_from_outcome!) but had no
            # callers — agent trait/trust learning was dead. An outcome is
            # "AI-accurate" if the estimated_return at investment was within
            # 25% of the realized return_multiple.
            if ai_tier != "none"
                est = Float64(get(m, "estimated_return", 1.0))
                actual = Float64(get(m, "return_multiple", 1.0))
                ai_accurate = est > 0 && abs(actual - est) / max(abs(est), 0.1) < 0.25
                update_state_from_outcome!(agent, m; ai_was_accurate=ai_accurate)
            else
                update_state_from_outcome!(agent, m; ai_was_accurate=nothing)
            end
        end
        append!(all_matured, matured)
        # Check survival after matured investments (matches Python line 1003)
        check_survival!(agent, round)
    end

    # Get available opportunities (after applying costs, matching Python timing)
    available_opportunities = get_available_opportunities(sim.market)

    # Refresh alive agents after survival checks
    alive_agents = filter(a -> a.alive, sim.agents)

    # Phase 3: AI level selection and action decisions (matching Python BLOCK 2)
    # Check if sequential decisions are enabled
    sequential_enabled = hasfield(typeof(sim.config), :SEQUENTIAL_DECISIONS_ENABLED) ?
        sim.config.SEQUENTIAL_DECISIONS_ENABLED : false

    agent_actions = Dict{String,Any}[]

    if sequential_enabled && length(alive_agents) > 1
        # Sequential decision making: early agents decide first, their choices become visible signals
        early_fraction = hasfield(typeof(sim.config), :EARLY_DECISION_FRACTION) ?
            sim.config.EARLY_DECISION_FRACTION : 0.3
        signal_weight = hasfield(typeof(sim.config), :SIGNAL_VISIBILITY_WEIGHT) ?
            sim.config.SIGNAL_VISIBILITY_WEIGHT : 0.15

        n_early = max(1, Int(floor(length(alive_agents) * early_fraction)))

        # Shuffle and split into early and late deciders
        shuffled_agents = shuffle(sim.rng, collect(alive_agents))
        early_agents = shuffled_agents[1:n_early]
        late_agents = shuffled_agents[n_early+1:end]

        # Phase 3a: Early agents decide (no visibility signals)
        early_signals = Dict{String,Int}()  # Track which opportunities early agents invested in

        for agent in early_agents
            neighbor_agents = EmergentAgent[]
            other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
            if !isempty(other_agents)
                n_neighbors = min(5, length(other_agents))
                neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
            end

            # Per-agent opportunity filter (added 2026-04-23): use the
            # AI-tier-aware visibility set instead of the global pool. Mirrors
            # enhanced_step!'s pattern (lines ~1634, 1674, 1710). Falls back
            # to the global pool if filter returns empty.
            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            outcome = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            push!(agent_actions, outcome)

            # Record visible signal for invest actions
            if get(outcome, "action", "") == "invest"
                opp_id = string(get(outcome, "opportunity_id", ""))
                if !isempty(opp_id)
                    early_signals[opp_id] = get(early_signals, opp_id, 0) + 1
                end
            end
        end

        # Phase 3b: Late agents decide with visible signals
        for agent in late_agents
            neighbor_agents = EmergentAgent[]
            other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
            if !isempty(other_agents)
                n_neighbors = min(5, length(other_agents))
                neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
            end

            # Per-agent opportunity filter (added 2026-04-23): tier-aware visibility.
            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            outcome = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system,
                early_signals=early_signals,
                signal_weight=signal_weight
            )

            push!(agent_actions, outcome)
        end
    else
        # Original simultaneous decision logic
        for agent in sim.agents
            if !agent.alive
                continue
            end

            # Collect neighbor agents for social influence
            neighbor_agents = EmergentAgent[]
            if length(alive_agents) > 1
                other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
                if !isempty(other_agents)
                    n_neighbors = min(5, length(other_agents))
                    neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
                end
            end

            # Per-agent opportunity filter (added 2026-04-23): tier-aware visibility.
            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            # Use make_decision! which properly integrates AI level effects
            outcome = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            push!(agent_actions, outcome)
        end
    end

    # Update tier beliefs from immediate action outcomes (innovate, explore)
    for action in agent_actions
        agent_id = get(action, "agent_id", 0)
        if agent_id < 1 || agent_id > length(sim.agents)
            continue
        end
        agent = sim.agents[agent_id]
        if !agent.alive
            continue
        end
        action_type = get(action, "action", "maintain")
        if action_type in ["innovate", "explore"]
            success = get(action, "success", false)
            ai_tier = get(action, "ai_level_used", get_ai_level(agent))
            update_tier_belief!(agent.ai_learning, ai_tier, success)
        end
    end

    # Phase 4: Final survival check for all agents (matches Python lines 1077-1078)
    for agent in sim.agents
        check_survival!(agent, round)
    end

    # Phase 5: Update market
    # Build Innovation objects from the real innovation_outcome fields stored
    # by attempt_innovation! (agents.jl:843-851). Earlier this block fabricated
    # Innovation(novelty=rand(rng), quality=rand(rng), …) — the AI-assisted
    # innovation outcome was discarded at the simulation boundary, so all
    # downstream effects (opportunity spawning, novelty disruption) saw random
    # values regardless of what tier created them.
    innovations = Innovation[]
    for action in agent_actions
        if get(action, "action", "") == "innovate" && get(action, "success", false)
            innov_id = string(get(action, "innovation_id",
                generate_innovation_id(round, get(action, "agent_id", 0), 0)))
            innov_type = String(get(action, "innovation_type", "incremental"))
            knowledge_components = let kc = get(action, "knowledge_components", String[])
                kc isa Vector{String} ? kc : String[string(x) for x in kc]
            end
            novelty = Float64(get(action, "innovation_novelty", 0.5))
            quality = Float64(get(action, "innovation_quality", 0.5))
            ai_assisted = Bool(get(action, "ai_assisted", false))
            ai_domains_used = let dd = get(action, "ai_domains_used", String[])
                dd isa Vector{String} ? dd : String[string(x) for x in dd]
            end
            scarcity_v = get(action, "innovation_scarcity", nothing)
            impact_v = get(action, "market_impact", nothing)
            sector_v = get(action, "innovation_sector", nothing)
            combo_sig_v = get(action, "combination_signature", nothing)
            success_v = get(action, "success", nothing)

            innov = Innovation(
                id=innov_id,
                type=innov_type,
                knowledge_components=knowledge_components,
                novelty=novelty,
                quality=quality,
                round_created=round,
                creator_id=get(action, "agent_id", 0),
                ai_level_used=String(get(action, "ai_level_used", "none")),
                ai_assisted=ai_assisted,
                ai_domains_used=ai_domains_used,
                sector=isnothing(sector_v) ? nothing : String(sector_v),
                combination_signature=isnothing(combo_sig_v) ? nothing : String(combo_sig_v),
                cash_multiple=Float64(get(action, "cash_multiple", 1.5)),
                market_impact=isnothing(impact_v) ? nothing : Float64(impact_v),
                success=isnothing(success_v) ? nothing : Bool(success_v),
                scarcity=isnothing(scarcity_v) ? nothing : Float64(scarcity_v),
                is_new_combination=Bool(get(action, "is_new_combination", false)),
            )
            push!(innovations, innov)
            cash_multiple = Float64(get(action, "cash_multiple", 1.5))
            spawn_opportunity_from_innovation!(sim.market, innov, cash_multiple)
        end
    end

    # Phase 5b: Apply novelty disruption for high-novelty innovations
    # This implements the "DeepSeek Effect" - novel innovations disrupt crowded opportunities
    novelty_disruption_enabled = hasfield(typeof(sim.config), :NOVELTY_DISRUPTION_ENABLED) ?
        sim.config.NOVELTY_DISRUPTION_ENABLED : true
    if novelty_disruption_enabled
        for innov in innovations
            disrupted_count = apply_novelty_disruption!(sim.market, innov)
            if disrupted_count > 0
                # Track disruption in action for the innovator (optional: for analysis)
                for action in agent_actions
                    if get(action, "agent_id", 0) == innov.creator_id &&
                       get(action, "action", "") == "innovate"
                        action["disrupted_opportunities"] = disrupted_count
                        break
                    end
                end
            end
        end
    end

    # Create niche opportunities from exploration discoveries (Python match)
    for action in agent_actions
        if get(action, "action", "") == "explore" && get(action, "exploration_type", "") == "niche_discovery"
            agent_id = get(action, "agent_id", 0)
            niche_id = get(action, "discovered_sector", nothing)
            if !isnothing(niche_id)
                # Create 1-3 new opportunities in the discovered niche (matches Python)
                n_niche_opps = rand(sim.rng, 1:3)
                created_ids = String[]
                for _ in 1:n_niche_opps
                    new_opp = create_niche_opportunity(sim.market, string(niche_id), agent_id, round)
                    push!(created_ids, new_opp.id)
                end
                # Telemetry: stash the created opp id(s) on the action so
                # uncertainty.jl niche-creation accounting (line 733) sees them.
                # Single id stored under "new_opportunity_id" for the common
                # 1-opp case; full list under "new_opportunity_ids".
                if !isempty(created_ids)
                    action["new_opportunity_id"] = first(created_ids)
                    action["new_opportunity_ids"] = created_ids
                end
            end
        end
    end

    market_state = step!(sim.market, round, agent_actions, innovations; matured_outcomes=all_matured)

    # Phase 5.5: Market clearing + dynamics (sector demand/supply imbalance)
    # update_clearing_metrics! populates market.sector_clearing_index, which
    # realized_return (models.jl:204) consults for the demand-shortfall
    # multiplier. update_market_dynamics! moves market_momentum and the
    # boom-streak counter. Both functions existed pre-v2 but were never called
    # from EmergentSimulation, so the realism layer was effectively dead code.
    update_clearing_metrics!(sim.market, agent_actions)
    invest_actions = filter(a -> get(a, "action", "") == "invest", agent_actions)
    # Actions emit the invested amount under "amount" (agents.jl:715, 729).
    # Fall back to "investment_amount" for compatibility with matured-outcome
    # records (agents.jl:1110) that use a different key.
    total_investment_for_dynamics = sum(
        Float64(get(a, "amount", get(a, "investment_amount", 0.0))) for a in invest_actions;
        init=0.0
    )
    n_ai_invest = count(a -> normalize_ai_label(get(a, "ai_level_used", "none")) != "none",
                        invest_actions)
    ai_invest_share = isempty(invest_actions) ? 0.0 : n_ai_invest / length(invest_actions)
    update_market_dynamics!(sim.market, agent_actions, total_investment_for_dynamics,
                            ai_invest_share)

    # Phase 5.6: Opportunity lifecycle management (added 2026-04-23)
    # Build per-round opportunity demand from agent invest actions and call
    # manage_opportunities! to age opportunities, decay competition (* 0.9),
    # and remove dead opportunities. Previously this function existed in
    # market.jl but was never called from the EmergentSimulation step path,
    # so crowding pressure accumulated monotonically across rounds.
    opportunity_demand = Dict{String,Int}()
    total_investment = 0.0
    for action in agent_actions
        if get(action, "action", "") == "invest"
            opp_id = string(get(action, "opportunity_id", ""))
            if !isempty(opp_id)
                opportunity_demand[opp_id] = get(opportunity_demand, opp_id, 0) + 1
                # Actions emit invested capital under "amount" (agents.jl:715).
                total_investment += Float64(get(action, "amount", get(action, "investment_amount", 0.0)))
            end
        end
    end
    manage_opportunities!(sim.market, round, opportunity_demand, total_investment)

    # Phase 6: Update uncertainty measurements
    record_ai_signals!(sim.uncertainty_env, round, agent_actions)
    uncertainty_state = measure_uncertainty_state!(
        sim.uncertainty_env,
        sim.market,
        agent_actions,
        innovations,
        round
    )

    # Phase 7: Record history
    round_stats = compile_round_stats(sim, round, agent_actions, all_matured, uncertainty_state)
    push!(sim.history, round_stats)

    return round_stats
end

"""
Compile statistics for a round.
"""
function compile_round_stats(
    sim::EmergentSimulation,
    round::Int,
    agent_actions::Vector{Dict{String,Any}},
    matured_outcomes::Vector{Dict{String,Any}},
    uncertainty_state::Dict{String,Dict{String,Any}}
)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]
    n_alive = length(alive_agents)
    n_total = length(sim.agents)

    # Capital statistics
    capitals = [get_capital(a) for a in alive_agents]
    mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
    std_capital = isempty(capitals) || length(capitals) < 2 ? 0.0 : std(capitals)
    median_capital = isempty(capitals) ? 0.0 : median(capitals)

    # Action counts and capital tracking by action type
    action_counts = Dict{String,Int}()
    ai_usage = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)

    # Track capital deployed/returned by action type for ROIC calculation
    capital_deployed = Dict{String,Float64}("invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)
    capital_returned = Dict{String,Float64}("invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)

    # Track opportunity IDs for HHI calculation
    opportunity_ids = String[]
    invest_confidences = Float64[]

    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counts[act_type] = get(action_counts, act_type, 0) + 1

        ai_level = lowercase(string(get(action, "ai_level_used", "none")))
        if haskey(ai_usage, ai_level)
            ai_usage[ai_level] += 1
        end

        # Track capital deployed by action type
        if act_type == "invest"
            amount = Float64(get(action, "amount", 0.0))
            capital_deployed["invest"] += amount
            opp_id = get(action, "opportunity_id", nothing)
            if !isnothing(opp_id)
                push!(opportunity_ids, string(opp_id))
            end
            # Track decision confidence for invest actions
            perception = get(action, "perception", Dict{String,Any}())
            conf = Float64(get(perception, "decision_confidence", 0.5))
            push!(invest_confidences, conf)
        elseif act_type == "innovate"
            rd_spend = Float64(get(action, "rd_spend", 0.0))
            capital_deployed["innovate"] += rd_spend
            if get(action, "success", false)
                ret = Float64(get(action, "innovation_return", 0.0))
                capital_returned["innovate"] += ret
            else
                rec = Float64(get(action, "recovery", 0.0))
                capital_returned["innovate"] += rec
            end
        elseif act_type == "explore"
            # Actions emit the spent amount under "explore_cost" (agents.jl:1025).
            cost = Float64(get(action, "explore_cost", get(action, "cost", 0.0)))
            capital_deployed["explore"] += cost
            # Exploration doesn't have direct capital return
        end
    end

    # Add matured investment returns to capital_returned["invest"]
    for outcome in matured_outcomes
        ret = Float64(get(outcome, "capital_returned", 0.0))
        capital_returned["invest"] += ret
    end

    # Calculate ROIC by action type (matches Python)
    mean_roic_invest = capital_deployed["invest"] > 0 ?
        (capital_returned["invest"] - capital_deployed["invest"]) / capital_deployed["invest"] : 0.0
    mean_roic_innovate = capital_deployed["innovate"] > 0 ?
        (capital_returned["innovate"] - capital_deployed["innovate"]) / capital_deployed["innovate"] : 0.0
    mean_roic_explore = 0.0  # Explore doesn't have direct return

    # Net capital flow by action type
    net_capital_flow_invest = capital_returned["invest"] - capital_deployed["invest"]
    net_capital_flow_innovate = capital_returned["innovate"] - capital_deployed["innovate"]

    # Calculate HHI (Herfindahl-Hirschman Index) for investment concentration
    overall_hhi = 0.0
    if !isempty(opportunity_ids)
        opp_counts = Dict{String,Int}()
        for opp_id in opportunity_ids
            opp_counts[opp_id] = get(opp_counts, opp_id, 0) + 1
        end
        total_invests = length(opportunity_ids)
        overall_hhi = sum((count / total_invests)^2 for count in values(opp_counts))
    end

    # Action shares
    total_actions = sum(values(action_counts))
    action_shares = Dict(
        "invest" => total_actions > 0 ? get(action_counts, "invest", 0) / total_actions : 0.0,
        "innovate" => total_actions > 0 ? get(action_counts, "innovate", 0) / total_actions : 0.0,
        "explore" => total_actions > 0 ? get(action_counts, "explore", 0) / total_actions : 0.0,
        "maintain" => total_actions > 0 ? get(action_counts, "maintain", 0) / total_actions : 0.0
    )

    # AI tier shares
    total_ai = sum(values(ai_usage))
    ai_shares = Dict(
        "none" => total_ai > 0 ? ai_usage["none"] / total_ai : 0.0,
        "basic" => total_ai > 0 ? ai_usage["basic"] / total_ai : 0.0,
        "advanced" => total_ai > 0 ? ai_usage["advanced"] / total_ai : 0.0,
        "premium" => total_ai > 0 ? ai_usage["premium"] / total_ai : 0.0
    )

    # Innovation stats
    innovation_attempts = get(action_counts, "innovate", 0)
    innovation_successes = count(a -> get(a, "action", "") == "innovate" && get(a, "success", false), agent_actions)
    innovation_success_rate = innovation_attempts > 0 ? innovation_successes / innovation_attempts : 0.0

    # Mean confidence for invest actions
    mean_confidence_invest = isempty(invest_confidences) ? 0.0 : mean(invest_confidences)

    # Matured investment stats
    n_matured = length(matured_outcomes)
    n_success = count(o -> get(o, "success", false), matured_outcomes)
    n_failure = n_matured - n_success

    # Uncertainty levels (formula-based, environment level - kept for backwards compatibility)
    actor_ignorance = Float64(get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0))
    practical_indet = Float64(get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0))
    agentic_novelty = Float64(get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0))
    competitive_rec = Float64(get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0))

    # EMERGENT uncertainty (agent-level, computed from actual outcomes)
    # These metrics emerge from what actually happens to agents, not from formulas
    emergent_by_tier = aggregate_emergent_uncertainty_by_tier(sim.agents)

    # Get simulation's AI tier (all agents have same fixed tier in this simulation design)
    # Use fixed_ai_level if set, otherwise current_ai_level
    sim_tier = if !isempty(sim.agents)
        first_agent = sim.agents[1]
        if !isnothing(first_agent.fixed_ai_level)
            first_agent.fixed_ai_level
        else
            first_agent.current_ai_level
        end
    else
        "none"
    end

    # Get emergent metrics for this tier
    tier_emergent = get(emergent_by_tier, sim_tier, Dict{String,Float64}(
        "actor_ignorance" => 0.5,
        "practical_indeterminism" => 0.5,
        "agentic_novelty" => 0.5,
        "competitive_recursion" => 0.0
    ))

    emergent_actor_ignorance = Float64(get(tier_emergent, "actor_ignorance", 0.5))
    emergent_practical_indet = Float64(get(tier_emergent, "practical_indeterminism", 0.5))
    emergent_agentic_novelty = Float64(get(tier_emergent, "agentic_novelty", 0.5))
    emergent_competitive_rec = Float64(get(tier_emergent, "competitive_recursion", 0.0))

    # --- Uncertainty Transformation Metrics ---
    # Store baseline on first round (or first round with uncertainty data)
    if isempty(sim.baseline_uncertainty_levels)
        sim.baseline_uncertainty_levels["actor_ignorance"] = actor_ignorance
        sim.baseline_uncertainty_levels["practical_indeterminism"] = practical_indet
        sim.baseline_uncertainty_levels["agentic_novelty"] = agentic_novelty
        sim.baseline_uncertainty_levels["competitive_recursion"] = competitive_rec
    end

    # Get previous levels (default to current if first round)
    prev_actor = get(sim.previous_uncertainty_levels, "actor_ignorance", actor_ignorance)
    prev_practical = get(sim.previous_uncertainty_levels, "practical_indeterminism", practical_indet)
    prev_agentic = get(sim.previous_uncertainty_levels, "agentic_novelty", agentic_novelty)
    prev_competitive = get(sim.previous_uncertainty_levels, "competitive_recursion", competitive_rec)

    # Get baseline levels
    base_actor = get(sim.baseline_uncertainty_levels, "actor_ignorance", actor_ignorance)
    base_practical = get(sim.baseline_uncertainty_levels, "practical_indeterminism", practical_indet)
    base_agentic = get(sim.baseline_uncertainty_levels, "agentic_novelty", agentic_novelty)
    base_competitive = get(sim.baseline_uncertainty_levels, "competitive_recursion", competitive_rec)

    # Compute delta (round-over-round change)
    delta_actor = actor_ignorance - prev_actor
    delta_practical = practical_indet - prev_practical
    delta_agentic = agentic_novelty - prev_agentic
    delta_competitive = competitive_rec - prev_competitive

    # Compute cumulative delta (change from baseline)
    cumulative_delta_actor = actor_ignorance - base_actor
    cumulative_delta_practical = practical_indet - base_practical
    cumulative_delta_agentic = agentic_novelty - base_agentic
    cumulative_delta_competitive = competitive_rec - base_competitive

    # Compute portfolio composition (shares)
    uncertainty_total = actor_ignorance + practical_indet + agentic_novelty + competitive_rec
    total_safe = max(uncertainty_total, 0.001)  # Avoid division by zero
    share_actor = actor_ignorance / total_safe
    share_practical = practical_indet / total_safe
    share_agentic = agentic_novelty / total_safe
    share_competitive = competitive_rec / total_safe

    # Compute HHI (Herfindahl-Hirschman Index) - concentration measure
    uncertainty_hhi = share_actor^2 + share_practical^2 + share_agentic^2 + share_competitive^2

    # Compute entropy - diversity measure (avoid log(0))
    eps = 1e-10
    uncertainty_entropy = -(
        share_actor * log(share_actor + eps) +
        share_practical * log(share_practical + eps) +
        share_agentic * log(share_agentic + eps) +
        share_competitive * log(share_competitive + eps)
    )

    # Update previous levels for next round
    sim.previous_uncertainty_levels["actor_ignorance"] = actor_ignorance
    sim.previous_uncertainty_levels["practical_indeterminism"] = practical_indet
    sim.previous_uncertainty_levels["agentic_novelty"] = agentic_novelty
    sim.previous_uncertainty_levels["competitive_recursion"] = competitive_rec

    # Mean AI trust
    trust_values = [Float64(get(a.traits, "ai_trust", 0.5)) for a in alive_agents]
    mean_trust = isempty(trust_values) ? 0.5 : mean(trust_values)
    std_trust = length(trust_values) < 2 ? 0.0 : std(trust_values)

    return Dict{String,Any}(
        "round" => round,
        "n_alive" => n_alive,
        "n_total" => n_total,
        "survival_rate" => n_total > 0 ? n_alive / n_total : 0.0,
        "mean_capital" => mean_capital,
        "median_capital" => median_capital,
        "std_capital" => std_capital,
        "total_capital" => sum(capitals),
        # Action counts
        "invest_count" => get(action_counts, "invest", 0),
        "innovate_count" => get(action_counts, "innovate", 0),
        "explore_count" => get(action_counts, "explore", 0),
        "maintain_count" => get(action_counts, "maintain", 0),
        # Action shares
        "action_share_invest" => action_shares["invest"],
        "action_share_innovate" => action_shares["innovate"],
        "action_share_explore" => action_shares["explore"],
        "action_share_maintain" => action_shares["maintain"],
        # AI tier counts and shares
        "ai_none_count" => ai_usage["none"],
        "ai_basic_count" => ai_usage["basic"],
        "ai_advanced_count" => ai_usage["advanced"],
        "ai_premium_count" => ai_usage["premium"],
        "ai_share_none" => ai_shares["none"],
        "ai_share_basic" => ai_shares["basic"],
        "ai_share_advanced" => ai_shares["advanced"],
        "ai_share_premium" => ai_shares["premium"],
        # Capital deployed/returned by action type (matches Python)
        "total_capital_deployed" => sum(values(capital_deployed)),
        "total_capital_returned" => sum(values(capital_returned)),
        "total_capital_deployed_invest" => capital_deployed["invest"],
        "total_capital_deployed_innovate" => capital_deployed["innovate"],
        "total_capital_deployed_explore" => capital_deployed["explore"],
        "total_capital_returned_invest" => capital_returned["invest"],
        "total_capital_returned_innovate" => capital_returned["innovate"],
        "total_capital_returned_explore" => capital_returned["explore"],
        "net_capital_flow_invest" => net_capital_flow_invest,
        "net_capital_flow_innovate" => net_capital_flow_innovate,
        # ROIC by action type (matches Python)
        "mean_roic_invest" => mean_roic_invest,
        "mean_roic_innovate" => mean_roic_innovate,
        "mean_roic_explore" => mean_roic_explore,
        # HHI and sector metrics (matches Python)
        "overall_hhi" => overall_hhi,
        # Innovation metrics
        "innovation_attempts" => innovation_attempts,
        "innovation_successes" => innovation_successes,
        "innovation_success_rate" => innovation_success_rate,
        # Confidence metrics
        "mean_confidence_invest" => mean_confidence_invest,
        "mean_ai_trust" => mean_trust,
        "ai_trust_std" => std_trust,
        # Matured investment stats
        "n_matured" => n_matured,
        "n_success" => n_success,
        "n_failure" => n_failure,
        "success_rate" => n_matured > 0 ? n_success / n_matured : 0.0,
        # Uncertainty levels (formula-based, kept for backwards compatibility)
        "actor_ignorance" => actor_ignorance,
        "practical_indeterminism" => practical_indet,
        "agentic_novelty" => agentic_novelty,
        "competitive_recursion" => competitive_rec,
        # EMERGENT uncertainty (agent-level, from actual outcomes)
        "emergent_actor_ignorance" => emergent_actor_ignorance,
        "emergent_practical_indeterminism" => emergent_practical_indet,
        "emergent_agentic_novelty" => emergent_agentic_novelty,
        "emergent_competitive_recursion" => emergent_competitive_rec,
        # Uncertainty transformation metrics
        "delta_actor_ignorance" => delta_actor,
        "delta_practical_indeterminism" => delta_practical,
        "delta_agentic_novelty" => delta_agentic,
        "delta_competitive_recursion" => delta_competitive,
        "cumulative_delta_actor" => cumulative_delta_actor,
        "cumulative_delta_practical" => cumulative_delta_practical,
        "cumulative_delta_agentic" => cumulative_delta_agentic,
        "cumulative_delta_competitive" => cumulative_delta_competitive,
        "uncertainty_total" => uncertainty_total,
        "share_actor_ignorance" => share_actor,
        "share_practical_indeterminism" => share_practical,
        "share_agentic_novelty" => share_agentic,
        "share_competitive_recursion" => share_competitive,
        "uncertainty_hhi" => uncertainty_hhi,
        "uncertainty_entropy" => uncertainty_entropy,
        # Agent counts
        "alive_agents" => n_alive,
        "dead_agents" => n_total - n_alive
    )
end

"""
Convert simulation history to DataFrame.
"""
function history_to_dataframe(sim::EmergentSimulation)::DataFrame
    if isempty(sim.history)
        return DataFrame()
    end

    # Get all column names from first entry
    cols = collect(keys(sim.history[1]))

    # Create DataFrame
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(h, col, missing) for h in sim.history]
    end

    return df
end

"""
Get final agent data as DataFrame.
"""
function agents_to_dataframe(sim::EmergentSimulation)::DataFrame
    data = [snapshot(agent, sim.current_round) for agent in sim.agents]

    if isempty(data)
        return DataFrame()
    end

    cols = collect(keys(data[1]))
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(d, col, missing) for d in data]
    end

    return df
end

"""
Get summary statistics for the simulation.
"""
function summary_stats(sim::EmergentSimulation)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]

    # Final survival rate
    survival_rate = length(alive_agents) / length(sim.agents)

    # Capital statistics
    final_capitals = [get_capital(a) for a in alive_agents]
    mean_final_capital = isempty(final_capitals) ? 0.0 : mean(final_capitals)

    # AI tier distribution at end
    ai_distribution = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
    for agent in sim.agents
        tier = get_ai_level(agent)
        if haskey(ai_distribution, tier)
            ai_distribution[tier] += 1
        end
    end

    # Success/failure totals
    total_successes = sum(a.success_count for a in sim.agents)
    total_failures = sum(a.failure_count for a in sim.agents)
    total_innovations = sum(a.innovation_count for a in sim.agents)

    # Uncertainty averages from history
    if !isempty(sim.history)
        mean_actor_ignorance = mean(get(h, "actor_ignorance", 0.0) for h in sim.history)
        mean_practical_indet = mean(get(h, "practical_indeterminism", 0.0) for h in sim.history)
        mean_agentic_novelty = mean(get(h, "agentic_novelty", 0.0) for h in sim.history)
        mean_competitive_rec = mean(get(h, "competitive_recursion", 0.0) for h in sim.history)
    else
        mean_actor_ignorance = 0.0
        mean_practical_indet = 0.0
        mean_agentic_novelty = 0.0
        mean_competitive_rec = 0.0
    end

    return Dict{String,Any}(
        "run_id" => sim.run_id,
        "n_agents" => length(sim.agents),
        "n_rounds" => sim.config.N_ROUNDS,
        "final_survival_rate" => survival_rate,
        "n_survivors" => length(alive_agents),
        "mean_final_capital" => mean_final_capital,
        "total_successes" => total_successes,
        "total_failures" => total_failures,
        "total_innovations" => total_innovations,
        "ai_none_count" => ai_distribution["none"],
        "ai_basic_count" => ai_distribution["basic"],
        "ai_advanced_count" => ai_distribution["advanced"],
        "ai_premium_count" => ai_distribution["premium"],
        "mean_actor_ignorance" => mean_actor_ignorance,
        "mean_practical_indeterminism" => mean_practical_indet,
        "mean_agentic_novelty" => mean_agentic_novelty,
        "mean_competitive_recursion" => mean_competitive_rec,
        "elapsed_seconds" => (now() - sim.start_time).value / 1000.0
    )
end

"""
Save simulation results to disk.
"""
function save_results!(sim::EmergentSimulation)
    mkpath(sim.output_dir)

    # Save history
    history_df = history_to_dataframe(sim)
    if nrow(history_df) > 0
        save_dataframe_csv(history_df, joinpath(sim.output_dir, "history.csv"))
        save_dataframe_arrow(history_df, joinpath(sim.output_dir, "history.arrow"))
    end

    # Save agent data
    agents_df = agents_to_dataframe(sim)
    if nrow(agents_df) > 0
        save_dataframe_csv(agents_df, joinpath(sim.output_dir, "final_agents.csv"))
    end

    # Save config
    save_config_snapshot(sim.config, joinpath(sim.output_dir, "config_snapshot.json"))

    # Save summary
    stats = summary_stats(sim)
    open(joinpath(sim.output_dir, "summary.json"), "w") do io
        JSON3.write(io, stats)
    end

    println("[$(sim.run_id)] Results saved to $(sim.output_dir)")
end

# ============================================================================
# BATCH SIMULATION UTILITIES
# ============================================================================

"""
Run multiple simulations with different configurations.
"""
function run_batch(;
    base_config::EmergentConfig = EmergentConfig(),
    n_runs::Int = 10,
    output_base::String = "results",
    fixed_ai_levels::Vector{String} = String[],
    parallel::Bool = false
)::Vector{EmergentSimulation}
    results = EmergentSimulation[]

    if isempty(fixed_ai_levels)
        # Run with adaptive AI
        for run_idx in 1:n_runs
            config = deepcopy(base_config)
            config.RANDOM_SEED = base_config.RANDOM_SEED + run_idx

            run_id = "run_$(run_idx)"
            output_dir = joinpath(output_base, run_id)

            sim = EmergentSimulation(
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                seed=config.RANDOM_SEED
            )

            run!(sim)
            save_results!(sim)
            push!(results, sim)
        end
    else
        # Run fixed AI tier sweep
        for (tier_idx, ai_level) in enumerate(fixed_ai_levels)
            for run_idx in 1:n_runs
                config = deepcopy(base_config)
                config.RANDOM_SEED = base_config.RANDOM_SEED + (tier_idx - 1) * n_runs + run_idx

                run_id = "Fixed_AI_Level_$(ai_level)_run_$(run_idx)"
                output_dir = joinpath(output_base, run_id)

                sim = EmergentSimulation(
                    config=config,
                    output_dir=output_dir,
                    run_id=run_id,
                    seed=config.RANDOM_SEED
                )

                # Set fixed AI level for all agents
                initialize_agents!(sim; fixed_ai_level=ai_level)

                run!(sim)
                save_results!(sim)
                push!(results, sim)
            end
        end
    end

    return results
end

"""
Aggregate results from multiple simulations.
"""
function aggregate_results(simulations::Vector{EmergentSimulation})::DataFrame
    all_stats = Dict{String,Any}[]

    for sim in simulations
        stats = summary_stats(sim)
        push!(all_stats, stats)
    end

    if isempty(all_stats)
        return DataFrame()
    end

    cols = collect(keys(all_stats[1]))
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(s, col, missing) for s in all_stats]
    end

    return df
end
