# Diagnostic invariant tests — designed to catch the data-flow / semantic
# bugs that pure code review couldn't see. Each test sets up a small,
# targeted scenario and asserts a numeric invariant that would have failed
# under one of the v2/v2.1/v2.2/v2.3-era bugs.
#
# These are NOT smoke tests ("does it crash?"). Each assertion encodes
# "the mechanism produced the value it should produce."

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

const N_OPPS_DIAG = 20

"""Build a 1-agent fixed-tier setup with a known opportunity pool."""
function _diag_setup(tier::String; seed::Int = 42, n_opps::Int = N_OPPS_DIAG)
    cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=seed,
                         AGENT_AI_MODE="fixed")
    market = MarketEnvironment(cfg; rng=MersenneTwister(seed))
    info_system = InformationSystem(cfg)
    agent = EmergentAgent(1, cfg;
        primary_sector="tech",
        fixed_ai_level=tier,
        rng=MersenneTwister(seed))

    base_rng = MersenneTwister(seed + 1)
    opps = Opportunity[]
    for i in 1:n_opps
        sector = rand(base_rng, market.sectors)
        push!(opps, GlimpseABM._create_realistic_opportunity(market, "diag_$i", sector))
    end
    return cfg, market, info_system, agent, opps
end

@testset "Diagnostics" begin

    # ------------------------------------------------------------------
    # 1. Per-use cost accumulates for Basic tier
    # ------------------------------------------------------------------
    @testset "basic per-use cost is actually deducted" begin
        cfg, market, info_system, agent, opps = _diag_setup("basic")
        capital_before = get_capital(agent)
        per_use = Float64(cfg.AI_LEVELS["basic"].cost) * cfg.AI_COST_INTENSITY
        @test per_use > 0  # Sanity

        market_conditions = Dict{String,Any}("regime"=>"normal",
                                             "uncertainty_state"=>Dict{String,Any}())
        perception = Dict{String,Any}()
        evals = GlimpseABM.evaluate_portfolio_opportunities(
            agent, opps, market_conditions, perception;
            ai_level="basic", info_system=info_system,
        )
        capital_after = get_capital(agent)
        deducted = capital_before - capital_after
        # Should be ~length(opps) × per_use (one charge per fresh get_information)
        expected = length(opps) * per_use
        @test isapprox(deducted, expected, rtol=0.01)
    end

    # ------------------------------------------------------------------
    # 2. Per-use cost dedupe across utility + ranking calls
    # ------------------------------------------------------------------
    @testset "per-use cost dedupe on cache hit" begin
        cfg, market, info_system, agent, opps = _diag_setup("basic")
        per_use = Float64(cfg.AI_LEVELS["basic"].cost) * cfg.AI_COST_INTENSITY

        market_conditions = Dict{String,Any}("regime"=>"normal",
                                             "uncertainty_state"=>Dict{String,Any}())
        perception = Dict{String,Any}()

        # First call: ALL fresh — full charge
        c0 = get_capital(agent)
        GlimpseABM.evaluate_portfolio_opportunities(
            agent, opps, market_conditions, perception;
            ai_level="basic", info_system=info_system,
        )
        c1 = get_capital(agent)
        first_deduct = c0 - c1

        # Second call SAME ROUND, same opps: all cache hits → zero charge
        GlimpseABM.evaluate_portfolio_opportunities(
            agent, opps, market_conditions, perception;
            ai_level="basic", info_system=info_system,
        )
        c2 = get_capital(agent)
        second_deduct = c1 - c2

        @test first_deduct > 0
        @test isapprox(second_deduct, 0.0, atol=1.0)  # Cache hits → no charge
    end

    # ------------------------------------------------------------------
    # 3. Per-agent cache key produces within-tier heterogeneity
    # ------------------------------------------------------------------
    @testset "two premium agents get different estimates per opp" begin
        cfg, market, info_system, agent_a, opps = _diag_setup("premium"; seed=1001)
        # Same setup but different agent id and rng state
        agent_b = EmergentAgent(2, cfg;
            primary_sector="tech",
            fixed_ai_level="premium",
            rng=MersenneTwister(2001))

        differing = 0
        for opp in opps
            ia = GlimpseABM.get_information(info_system, opp, "premium";
                agent_id=agent_a.id, rng=agent_a.rng)
            ib = GlimpseABM.get_information(info_system, opp, "premium";
                agent_id=agent_b.id, rng=agent_b.rng)
            if !isapprox(ia.estimated_return, ib.estimated_return)
                differing += 1
            end
        end
        # If cache were keyed only by (opp, tier), every estimate would match
        # and `differing` would be 0. With per-agent keying, almost all should differ.
        @test differing >= length(opps) - 2
    end

    # ------------------------------------------------------------------
    # 4. Premium subscription billing matches monthly cost per round
    # ------------------------------------------------------------------
    @testset "premium subscription bills full monthly cost per round" begin
        cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=42,
                             AGENT_AI_MODE="fixed")
        agent = EmergentAgent(1, cfg;
            primary_sector="tech",
            fixed_ai_level="premium",
            rng=MersenneTwister(42))
        GlimpseABM.ensure_subscription_schedule!(agent, "premium")
        c0 = get_capital(agent)
        charged = GlimpseABM.charge_subscription_installment!(agent, "premium")
        c1 = get_capital(agent)
        # Documented: $3500/month × AI_COST_INTENSITY
        expected = 3500.0 * cfg.AI_COST_INTENSITY
        @test isapprox(charged, expected, rtol=0.01)
        @test isapprox(c0 - c1, expected, rtol=0.01)
    end

    # ------------------------------------------------------------------
    # 5. Investment-amount aggregation matches sum of per-action amounts
    # ------------------------------------------------------------------
    @testset "opp.total_invested matches summed action amounts" begin
        cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=2, RANDOM_SEED=42,
                             AGENT_AI_MODE="fixed")
        sim = EmergentSimulation(config=cfg, seed=42,
            initial_tier_distribution=Dict("none"=>1.0))
        GlimpseABM.run!(sim)

        # Each opp.total_invested is decremented as investments mature, so
        # this only matches in the absence of maturity events. We assert a
        # weaker invariant: total_invested across opps is non-negative and
        # bounded by total agent capital deployed.
        agg_opp = sum(o.total_invested for o in sim.market.opportunities; init=0.0)
        agg_agent = sum(a.total_invested for a in sim.agents; init=0.0)
        @test agg_opp >= 0
        @test agg_agent >= 0
        @test agg_opp <= agg_agent + 1.0  # Opp total ≤ what agents reported deploying
    end

    # ------------------------------------------------------------------
    # 6. Competition accumulates when many agents invest in the same opp
    # ------------------------------------------------------------------
    @testset "competition rises with concentrated investment" begin
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=42)
        market = MarketEnvironment(cfg; rng=MersenneTwister(42))
        opp = first(market.opportunities)
        # Hammer one opp with 100 invest updates
        for _ in 1:100
            GlimpseABM.update_opportunity_competition!(market, opp, 0.05)
        end
        # With per-investment decay removed, competition should be far above
        # the ~0.7 cap that v2.2 hit. Pre-fix value was capped at 0.7;
        # post-fix should be ≥ 1.0.
        @test opp.competition >= 1.0
    end

    # ------------------------------------------------------------------
    # 7. Population-scaling idempotency — no double-multiplication
    # ------------------------------------------------------------------
    @testset "initialize! is idempotent for population scaling" begin
        cfg = EmergentConfig(N_AGENTS=4000, RANDOM_SEED=42)
        GlimpseABM.initialize!(cfg)
        k_after_first = cfg.CROWDING_CAPACITY_K
        cap_after_first = cfg.OPPORTUNITY_BASE_CAPACITY
        # Second call should be a no-op
        GlimpseABM.initialize!(cfg)
        @test cfg.CROWDING_CAPACITY_K == k_after_first
        @test cfg.OPPORTUNITY_BASE_CAPACITY == cap_after_first
        # And third
        GlimpseABM.initialize!(cfg)
        @test cfg.CROWDING_CAPACITY_K == k_after_first
    end

    # ------------------------------------------------------------------
    # 8. RNG reproducibility — two markets with same seed produce same opp.capacity
    # ------------------------------------------------------------------
    @testset "opportunity capacity is reproducible across seeded markets" begin
        cfg = EmergentConfig(N_AGENTS=10, RANDOM_SEED=42)
        m1 = MarketEnvironment(cfg; rng=MersenneTwister(42))
        m2 = MarketEnvironment(cfg; rng=MersenneTwister(42))
        caps1 = [o.capacity for o in m1.opportunities]
        caps2 = [o.capacity for o in m2.opportunities]
        @test length(caps1) == length(caps2)
        @test caps1 == caps2
    end

    # ------------------------------------------------------------------
    # 9. Fixed mode → get_ai_level always returns the fixed tier
    # ------------------------------------------------------------------
    @testset "fixed mode locks tier" begin
        cfg = EmergentConfig(N_AGENTS=5, N_ROUNDS=2, RANDOM_SEED=42,
                             AGENT_AI_MODE="fixed")
        sim = EmergentSimulation(config=cfg, seed=42,
            initial_tier_distribution=Dict("premium"=>1.0))
        for agent in sim.agents
            @test get_ai_level(agent) == "premium"
        end
        GlimpseABM.run!(sim)
        for agent in sim.agents
            @test get_ai_level(agent) == "premium"
        end
    end

    # ------------------------------------------------------------------
    # 10. Emergent mode → fixed_ai_level is nothing, current can change
    # ------------------------------------------------------------------
    @testset "emergent mode allows tier switching" begin
        cfg = EmergentConfig(N_AGENTS=5, N_ROUNDS=1, RANDOM_SEED=42,
                             AGENT_AI_MODE="emergent")
        sim = EmergentSimulation(config=cfg, seed=42,
            initial_tier_distribution=Dict("premium"=>1.0))
        # Emergent agents must have nothing in fixed_ai_level so choose_ai_level fires
        for agent in sim.agents
            @test isnothing(agent.fixed_ai_level) || agent.fixed_ai_level == "none"
        end
    end

    # ------------------------------------------------------------------
    # 11. Lifecycle transitions emerging → growing at adoption_rate > 0.2
    # ------------------------------------------------------------------
    @testset "update_lifecycle! transitions emerging→growing" begin
        cfg = EmergentConfig(N_AGENTS=10, RANDOM_SEED=42)
        market = MarketEnvironment(cfg; rng=MersenneTwister(42))
        opp = first(market.opportunities)
        @test opp.lifecycle_stage == "emerging"
        GlimpseABM.update_lifecycle!(opp, 0.5)  # Above 0.2 threshold
        @test opp.lifecycle_stage == "growing"
        GlimpseABM.update_lifecycle!(opp, 0.7)  # Above 0.6 threshold
        @test opp.lifecycle_stage == "mature"
    end

    # ------------------------------------------------------------------
    # 12. Niche / spawn opportunity is visible to its creator (via override)
    # ------------------------------------------------------------------
    @testset "niche opportunity is visible to creator via created_by override" begin
        cfg = EmergentConfig(N_AGENTS=2, RANDOM_SEED=42)
        market = MarketEnvironment(cfg; rng=MersenneTwister(42))
        niche_opp = GlimpseABM.create_niche_opportunity(market, "tech", 1, 5)
        # v2.7: discovered starts false; creator sees via created_by override.
        @test niche_opp.discovered == false
        @test niche_opp.created_by == 1
        # Creator-visibility check: agent with id=1 should see this opp in
        # get_opportunities_for_agent even though discovered=false.
        traits = Dict("exploration_tendency"=>0.5, "market_awareness"=>0.5,
                      "ai_trust"=>0.5, "uncertainty_tolerance"=>0.5)
        creator = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="none",
                                rng=MersenneTwister(42))
        visible = GlimpseABM.get_opportunities_for_agent(market, creator)
        @test niche_opp.id in [o.id for o in visible]
    end

    # ------------------------------------------------------------------
    # 13. Fixed-mode runs are tier-homogeneous (ATE design invariant)
    # ------------------------------------------------------------------
    @testset "fixed-mode tier population is homogeneous" begin
        # Critical for ATE estimation: when a fixed-mode experiment runs the
        # "premium" cell, every agent must be premium. There must be no
        # mixed-tier competition contaminating the treatment assignment.
        for tier in ["none", "basic", "advanced", "premium"]
            cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=1, RANDOM_SEED=42,
                                 AGENT_AI_MODE="fixed")
            sim = EmergentSimulation(config=cfg, seed=42,
                initial_tier_distribution=Dict(tier => 1.0))
            tiers = unique(get_ai_level(a) for a in sim.agents)
            @test tiers == [tier]
            # And no agent should drift to another tier across rounds
            GlimpseABM.run!(sim)
            tiers_after = unique(get_ai_level(a) for a in sim.agents)
            @test tiers_after == [tier]
        end
    end

end

println("All diagnostic tests passed!")
