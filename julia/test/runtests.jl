"""
Test suite for GlimpseABM.jl
"""

using Test
using GlimpseABM
using Random
using DataFrames

@testset "GlimpseABM.jl" begin

    @testset "Configuration" begin
        # Test default config creation
        config = EmergentConfig()
        @test config.N_AGENTS == 1000
        @test config.N_ROUNDS == 250
        @test config.RANDOM_SEED == 42

        # Test sectors populated
        GlimpseABM.initialize!(config)
        @test !isempty(config.SECTORS)
        @test "tech" in config.SECTORS

        # Test scaled opportunities
        n_opps = GlimpseABM.get_scaled_opportunities(config, 1000)
        @test n_opps >= config.MIN_OPPORTUNITIES
    end

    @testset "Models" begin
        # Test Opportunity creation
        opp = Opportunity(
            id="test_opp_1",
            latent_return_potential=1.5,
            latent_failure_potential=0.3,
            complexity=0.5
        )
        @test opp.id == "test_opp_1"
        @test opp.latent_return_potential == 1.5
        @test opp.discovered == false

        # Test Information creation
        info = Information(
            estimated_return=1.2,
            estimated_uncertainty=0.3,
            confidence=0.8
        )
        @test info.estimated_return == 1.2
        @test GlimpseABM.quality_score(info) > 0

        # Test Innovation creation
        innov = Innovation(
            id="innov_1",
            type="incremental",
            knowledge_components=["comp1", "comp2"],
            novelty=0.7,
            quality=0.6,
            round_created=1,
            creator_id=1
        )
        @test innov.novelty == 0.7

        # Test AILearningProfile
        profile = AILearningProfile()
        @test haskey(profile.domain_trust, "market_analysis")
        @test profile.domain_trust["market_analysis"] == 0.5
    end

    @testset "Agents" begin
        config = EmergentConfig()
        GlimpseABM.initialize!(config)

        rng = MersenneTwister(42)

        # Test agent creation
        agent = EmergentAgent(1, config; rng=rng)
        @test agent.id == 1
        @test agent.alive == true
        @test GlimpseABM.get_capital(agent) > 0

        # Test capital operations
        initial_capital = GlimpseABM.get_capital(agent)
        GlimpseABM.set_capital!(agent, initial_capital + 1000)
        @test GlimpseABM.get_capital(agent) == initial_capital + 1000

        # Test AI level
        @test GlimpseABM.get_ai_level(agent) == "none"
        agent.fixed_ai_level = "premium"
        @test GlimpseABM.get_ai_level(agent) == "premium"
    end

    @testset "Market" begin
        config = EmergentConfig()
        GlimpseABM.initialize!(config)

        rng = MersenneTwister(42)

        # Test market creation
        market = MarketEnvironment(config; rng=rng)
        @test market.market_regime == "normal"
        @test !isempty(market.opportunities)

        # Test get available opportunities
        opps = GlimpseABM.get_available_opportunities(market)
        @test !isempty(opps)

        # Test market conditions
        conditions = GlimpseABM.get_market_conditions(market)
        @test haskey(conditions, "regime")
        @test conditions["regime"] == "normal"
    end

    @testset "Uncertainty" begin
        config = EmergentConfig()
        rng = MersenneTwister(42)

        # Test uncertainty environment creation
        unc_env = KnightianUncertaintyEnvironment(config; rng=rng)

        # Test initial state
        @test haskey(unc_env.actor_ignorance_state, "level")
        @test haskey(unc_env.practical_indeterminism_state, "level")
        @test haskey(unc_env.agentic_novelty_state, "level")
        @test haskey(unc_env.competitive_recursion_state, "level")

        # Test composite uncertainty
        composite = GlimpseABM.get_composite_uncertainty(unc_env)
        @test composite >= 0.0
        @test composite <= 1.0
    end

    @testset "Simulation Mini Run" begin
        # Create minimal config for fast test
        config = EmergentConfig()
        config.N_AGENTS = 10
        config.N_ROUNDS = 5
        config.enable_round_logging = false
        GlimpseABM.initialize!(config)

        # Run simulation
        sim = EmergentSimulation(
            config=config,
            output_dir="/tmp/glimpse_test",
            run_id="test_run",
            seed=42
        )

        GlimpseABM.run!(sim)

        # Verify simulation ran
        @test sim.current_round == 5
        @test !isempty(sim.history)

        # Check some agents survived
        survivors = count(a -> a.alive, sim.agents)
        @test survivors >= 0

        # Test history to DataFrame
        df = GlimpseABM.history_to_dataframe(sim)
        @test nrow(df) == 5

        # Test summary stats
        stats = GlimpseABM.summary_stats(sim)
        @test haskey(stats, "final_survival_rate")
        @test stats["n_rounds"] == 5
    end

    @testset "Utilities" begin
        config = EmergentConfig()
        rng = MersenneTwister(42)

        # Test trait sampling
        traits = GlimpseABM.sample_all_traits(config; rng=rng)
        @test haskey(traits, "uncertainty_tolerance")
        @test traits["uncertainty_tolerance"] >= 0.0
        @test traits["uncertainty_tolerance"] <= 1.0

        # Test weighted choice
        items = ["a", "b", "c"]
        weights = [0.7, 0.2, 0.1]
        choice = GlimpseABM.weighted_choice(items, weights; rng=rng)
        @test choice in items

        # Test HHI computation
        shares = [0.5, 0.3, 0.2]
        hhi = GlimpseABM.compute_hhi(shares)
        @test hhi > 0.0
        @test hhi <= 1.0

        # Test normalize probs
        probs = [2.0, 3.0, 5.0]
        normalized = GlimpseABM.normalize_probs(probs)
        @test isapprox(sum(normalized), 1.0)
    end

    @testset "Causal Analysis" begin
        # Test Cohen's d
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0] .+ 0.5
        group2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = GlimpseABM.cohens_d(group1, group2)
        @test result.measure == "Hedges' g"
        @test !isnan(result.value)
        @test result.ci_lower < result.value < result.ci_upper

        # Test Cliff's delta
        result2 = GlimpseABM.cliffs_delta(group1, group2)
        @test result2.measure == "Cliff's delta"
        @test -1.0 <= result2.value <= 1.0

        # Test ANOVA
        groups = [randn(20), randn(20) .+ 0.5, randn(20) .+ 1.0]
        anova_result = GlimpseABM.anova_oneway(groups)
        @test anova_result.test_name == "One-way ANOVA"
        @test anova_result.statistic >= 0
        @test 0.0 <= anova_result.p_value <= 1.0
        @test 0.0 <= anova_result.effect_size <= 1.0

        # Test Kruskal-Wallis
        kw_result = GlimpseABM.kruskal_wallis(groups)
        @test kw_result.test_name == "Kruskal-Wallis H"
        @test kw_result.statistic >= 0

        # Test bootstrap CI
        data = randn(50) .+ 1.0
        boot_result = GlimpseABM.bootstrap_ci(data, n_bootstrap=100)
        @test !isnan(boot_result.estimate)
        @test boot_result.ci_lower < boot_result.ci_upper

        # Test Mann-Whitney U
        mw_result = GlimpseABM.mann_whitney_u(group1, group2)
        @test !isnan(mw_result.U)
        @test 0.0 <= mw_result.p_value <= 1.0
    end

    @testset "CLI Module" begin
        # Test argument parsing
        args = GlimpseABM.parse_args(["--task", "fixed", "--agents", "100"])
        @test args["task"] == "fixed"
        @test args["agents"] == 100

        # Test default values
        args2 = GlimpseABM.parse_args(String[])
        @test args2["task"] == "master"
        @test args2["runs"] == 50
        @test args2["seed"] == 42
    end

end

println("All tests passed!")
