"""
Utility for profiling Glimpse ABM runs.

Port of: glimpse_abm/profile_simulation.py

Usage:
    julia profile_simulation.jl [--agents=N] [--rounds=N] [--output=DIR]
"""

module ProfileSimulation

using Profile
using Printf

# Import the main GlimpseABM module
# Note: This assumes the module is available in the load path
# In practice, you'd run this from the package directory

export profile_simulation, parse_args, build_simulation, main

"""
Parse command-line arguments.
"""
function parse_args(args::Vector{String})::Dict{Symbol,Any}
    result = Dict{Symbol,Any}(
        :agents => 500,
        :rounds => 100,
        :output => "./profiled_run",
        :stats => "./profile_stats.txt",
        :sort => :cumulative,
        :top => 30
    )

    for arg in args
        if startswith(arg, "--agents=")
            result[:agents] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--rounds=")
            result[:rounds] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--output=")
            result[:output] = split(arg, "=")[2]
        elseif startswith(arg, "--stats=")
            result[:stats] = split(arg, "=")[2]
        elseif startswith(arg, "--sort=")
            result[:sort] = Symbol(split(arg, "=")[2])
        elseif startswith(arg, "--top=")
            result[:top] = parse(Int, split(arg, "=")[2])
        elseif arg == "--help" || arg == "-h"
            println("""
            Usage: julia profile_simulation.jl [OPTIONS]

            Options:
                --agents=N      Number of agents (default: 500)
                --rounds=N      Number of rounds (default: 100)
                --output=DIR    Output directory for the profiled run (default: ./profiled_run)
                --stats=FILE    Path to write profile stats (default: ./profile_stats.txt)
                --sort=KEY      Sort order for profile output (default: cumulative)
                --top=N         Number of functions to display (default: 30)
                --help, -h      Show this help message
            """)
            exit(0)
        end
    end

    return result
end

"""
Build a simulation configuration for profiling.
"""
function build_simulation_config(n_agents::Int, n_rounds::Int)::Dict{Symbol,Any}
    Dict{Symbol,Any}(
        :N_AGENTS => n_agents,
        :N_ROUNDS => n_rounds,
        :N_RUNS => 1,
        :use_parallel => false,  # Deterministic profiling
        :RANDOM_SEED => 42
    )
end

"""
Profile a simulation run and report results.

This function provides Julia-native profiling using @profile macro.
"""
function profile_simulation(;
    n_agents::Int=500,
    n_rounds::Int=100,
    output_dir::String="./profiled_run",
    stats_file::String="./profile_stats.txt",
    top_n::Int=30
)
    println("=" ^ 60)
    println("SIMULATION PROFILING")
    println("=" ^ 60)
    println("\nConfiguration:")
    println("  Agents: $n_agents")
    println("  Rounds: $n_rounds")
    println("  Output: $output_dir")
    println()

    # Create output directory
    mkpath(output_dir)

    # Try to load GlimpseABM
    try
        # This assumes GlimpseABM is in scope
        @eval using GlimpseABM

        config = build_simulation_config(n_agents, n_rounds)

        println("Starting profiled run...")
        println()

        # Clear any existing profile data
        Profile.clear()

        # Time the run
        start_time = time()

        # Profile the simulation
        @profile begin
            # Create and run simulation
            sim = GlimpseABM.EmergentSimulation(
                n_agents=config[:N_AGENTS],
                n_rounds=config[:N_ROUNDS],
                seed=config[:RANDOM_SEED]
            )
            GlimpseABM.run!(sim)
        end

        elapsed = time() - start_time

        println("Simulation completed in $(round(elapsed, digits=2)) seconds")
        println()

        # Print profile results
        println("=" ^ 60)
        println("PROFILE RESULTS (Top $top_n functions)")
        println("=" ^ 60)
        println()

        # Get profile data
        Profile.print(maxdepth=20, noisefloor=2.0)

        # Save profile to file
        open(stats_file, "w") do f
            redirect_stdout(f) do
                Profile.print(maxdepth=20)
            end
        end
        println("\nProfile stats saved to: $stats_file")

        # Additional timing breakdown
        println()
        println("=" ^ 60)
        println("TIMING SUMMARY")
        println("=" ^ 60)
        println(@sprintf("  Total time:        %.2f seconds", elapsed))
        println(@sprintf("  Time per round:    %.4f seconds", elapsed / n_rounds))
        println(@sprintf("  Time per agent:    %.6f seconds", elapsed / (n_agents * n_rounds)))
        println(@sprintf("  Throughput:        %.0f agent-rounds/second", (n_agents * n_rounds) / elapsed))

    catch e
        # Fallback: profile a simple function to demonstrate the tool
        println("Note: GlimpseABM module not loaded. Running demonstration profile...")
        println()

        # Profile a simple simulation-like loop
        function demo_simulation(n_agents::Int, n_rounds::Int)
            # Simulate agent state
            capitals = ones(n_agents) * 500_000.0
            alive = trues(n_agents)

            for round in 1:n_rounds
                # Update capitals (simplified market dynamics)
                for i in 1:n_agents
                    if alive[i]
                        returns = randn() * 0.02  # 2% volatility
                        capitals[i] *= (1 + returns)
                        if capitals[i] < 100_000
                            alive[i] = false
                        end
                    end
                end
            end

            return (capitals, alive)
        end

        Profile.clear()

        start_time = time()
        @profile demo_simulation(n_agents, n_rounds)
        elapsed = time() - start_time

        println("Demo simulation completed in $(round(elapsed, digits=4)) seconds")
        println()
        Profile.print(maxdepth=10, noisefloor=2.0)

        # Save stats
        open(stats_file, "w") do f
            redirect_stdout(f) do
                println("DEMO PROFILING RESULTS")
                println("=" ^ 40)
                Profile.print(maxdepth=10)
            end
        end
        println("\nProfile stats saved to: $stats_file")
    end
end

"""
Alternative profiling using @timed for simpler timing analysis.
"""
function simple_timing(n_agents::Int, n_rounds::Int)
    println("=" ^ 60)
    println("SIMPLE TIMING ANALYSIS")
    println("=" ^ 60)

    # Timing individual operations
    function time_operation(name::String, f::Function)
        result = @timed f()
        println(@sprintf("  %-30s: %.4f s, %.2f MB allocated", name, result.time, result.bytes / 1e6))
        return result
    end

    println("\nOperation Breakdown:")

    # Initialize agents
    time_operation("Agent initialization") do
        [Dict(:capital => 500_000.0, :alive => true) for _ in 1:n_agents]
    end

    # Market generation
    time_operation("Market generation") do
        [Dict(:return => randn() * 0.05) for _ in 1:100]
    end

    # Decision making (per round)
    agents = [Dict(:capital => 500_000.0, :alive => true) for _ in 1:n_agents]
    time_operation("Single round decisions") do
        for agent in agents
            if agent[:alive]
                agent[:capital] *= (1 + randn() * 0.02)
            end
        end
    end

    # Full simulation
    println("\nFull Simulation:")
    result = time_operation("Complete simulation ($n_rounds rounds)") do
        capitals = ones(n_agents) * 500_000.0
        alive = trues(n_agents)
        for round in 1:n_rounds
            for i in 1:n_agents
                if alive[i]
                    capitals[i] *= (1 + randn() * 0.02)
                    alive[i] = capitals[i] >= 100_000
                end
            end
        end
        (capitals, alive)
    end

    println()
    println("Throughput Estimates:")
    println(@sprintf("  Agent-rounds/second: %.0f", (n_agents * n_rounds) / result.time))
    println(@sprintf("  Projected 1000-agent, 200-round runtime: %.2f seconds",
                    result.time * (1000/n_agents) * (200/n_rounds)))
end

"""
Main entry point.
"""
function main(args=ARGS)
    parsed = parse_args(args)

    profile_simulation(
        n_agents=parsed[:agents],
        n_rounds=parsed[:rounds],
        output_dir=parsed[:output],
        stats_file=parsed[:stats],
        top_n=parsed[:top]
    )

    # Also run simple timing
    println()
    simple_timing(parsed[:agents], parsed[:rounds])
end

# Run main if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module ProfileSimulation
