"""
Command-line interface for GlimpseABM.jl

Provides command-line experiment runners for:
- Master launcher (emergent + fixed-level experiments)
- Fixed AI level sweep (causal inference design)
- Sensitivity analysis with Latin Hypercube Sampling
- Progress monitoring and resume functionality

Port of: glimpse_abm/cli.py

CLI Version: 2025.01
"""

module CLI

using ArgParse
using Dates
using JSON3
using Random
using Statistics
using Base.Threads

# Import from parent module
using ..GlimpseABM: EmergentConfig, EmergentSimulation, initialize!, run!, save_results!,
                   summary_stats, history_to_dataframe, agents_to_dataframe,
                   save_dataframe_csv, save_dataframe_arrow, save_config_snapshot

export run_cli, parse_args, run_master_launcher, run_fixed_level_sweep
export run_sensitivity_analysis, latin_hypercube_samples
export apply_experiment_profile, estimate_runtime
export ProgressMonitor, start_monitor!, stop_monitor!
export run_uncertainty_scenario_sweep, run_sensitivity_sweep, run_lhs_sweep
export compute_parallel_plan, execute_parallel_tasks
export is_run_complete, build_rescue_config, reset_run_directory
export load_calibration_profile, apply_calibration_profile, list_calibration_profiles

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

"""
Parse command-line arguments.
"""
function parse_args(args::Vector{String}=ARGS)
    s = ArgParseSettings(
        description="GlimpseABM.jl experiment and analysis launcher",
        version="0.1.0",
        add_version=true
    )

    @add_arg_table! s begin
        "--task"
            help = "Task type: master, fixed, sweep, scenarios, sensitivity, lhs"
            arg_type = String
            default = "master"
        "--results-dir"
            help = "Output directory for results"
            arg_type = String
            default = "results"
        "--runs"
            help = "Number of simulation runs"
            arg_type = Int
            default = 50
        "--agents"
            help = "Number of agents per simulation"
            arg_type = Int
            default = 1000
        "--rounds"
            help = "Number of rounds per simulation"
            arg_type = Int
            default = 250
        "--ai-levels"
            help = "AI levels to test (comma-separated: none,basic,advanced,premium)"
            arg_type = String
            default = "none,basic,advanced,premium"
        "--include-fixed"
            help = "Include fixed AI level sweep in master task"
            action = :store_true
        "--skip-emergent"
            help = "Skip emergent AI simulation in master task"
            action = :store_true
        "--smoke"
            help = "Smoke test mode (small agent/round counts)"
            action = :store_true
        "--skip-visualizations"
            help = "Skip visualization generation"
            action = :store_true
        "--fast-stats"
            help = "Use reduced bootstrap iterations for faster stats"
            action = :store_true
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 42
        "--parallel"
            help = "Enable parallel execution"
            action = :store_true
        "--workers"
            help = "Number of worker threads"
            arg_type = Int
            default = 4
        "--calibration-profile"
            help = "Calibration profile name (default, high_uncertainty, stable_market, high_competition)"
            arg_type = String
        "--monitor-progress"
            help = "Print progress updates"
            action = :store_true
        "--monitor-interval"
            help = "Progress update interval (seconds)"
            arg_type = Float64
            default = 30.0
        "--human-only-baseline"
            help = "Run human-only baseline (no AI)"
            action = :store_true
        "--fixed-tier-sweep"
            help = "Run fixed AI tier sweep"
            action = :store_true
        "--dump-config"
            help = "Path to dump config JSON"
            arg_type = String
        "--verbose"
            help = "Verbose output"
            action = :store_true
        "--lhs-samples"
            help = "Number of Latin Hypercube samples for sensitivity analysis"
            arg_type = Int
            default = 20
        "--scenarios"
            help = "Comma-separated list of uncertainty scenarios to run"
            arg_type = String
            default = "Baseline,High_Ignorance,High_Indeterminism,High_AgenticNovelty,High_CompetitiveRecursion"
        "--runs-per-combo"
            help = "Number of runs per parameter combination (for sensitivity/lhs tasks)"
            arg_type = Int
            default = 1
        "--resume"
            help = "Resume incomplete runs from existing results directory"
            action = :store_true
        "--list-profiles"
            help = "List available calibration profiles and exit"
            action = :store_true
        "--parallel-mode"
            help = "Parallel execution mode: max (default) or safe"
            arg_type = String
            default = "max"
    end

    return ArgParse.parse_args(args, s)
end

# ============================================================================
# SMOKE TEST OVERRIDES
# ============================================================================

"""
Apply smoke test overrides for quick testing.
"""
function apply_smoke_overrides!(config::EmergentConfig)
    config.N_AGENTS = min(config.N_AGENTS, 50)
    config.N_ROUNDS = min(config.N_ROUNDS, 30)
    return config
end

# ============================================================================
# PARALLEL EXECUTION INFRASTRUCTURE
# ============================================================================

# Global tracking for cleanup
const _active_tasks = Ref{Vector{Task}}(Task[])
const _cleanup_registered = Ref{Bool}(false)

"""
Compute parallel execution plan based on workload.

Returns (max_workers, timeout_seconds, workload_estimate).
"""
function compute_parallel_plan(config::EmergentConfig)::Tuple{Int,Float64,Float64}
    cpu_count = Sys.CPU_THREADS
    max_workers = max(1, cpu_count - 1)

    # Respect explicit cap if set
    if hasfield(typeof(config), :MAX_PARALLEL_RUNS) && config.MAX_PARALLEL_RUNS > 0
        max_workers = min(max_workers, config.MAX_PARALLEL_RUNS)
    end

    # Compute workload estimate
    workload = (Float64(config.N_AGENTS) / 500.0) * (Float64(config.N_ROUNDS) / 100.0)

    # Apply workload-based caps in safe mode
    parallel_mode = get(Dict{Symbol,Any}(), :PARALLEL_MODE, "max")
    if parallel_mode == "safe"
        if workload >= 5.0
            max_workers = min(max_workers, 3)
        elseif workload >= 3.0
            max_workers = min(max_workers, 4)
        elseif workload >= 2.0
            max_workers = min(max_workers, 5)
        end
    end

    timeout = max(900.0, workload * 480.0)
    return (max_workers, timeout, workload)
end

"""
Execute tasks in parallel using Julia's thread pool.

Parameters
----------
tasks : Vector - Task arguments
worker : Function - Worker function to execute
n_workers : Int - Number of concurrent workers
desc : String - Description for logging
timeout : Float64 - Optional timeout in seconds

Returns
-------
Vector of results
"""
function execute_parallel_tasks(
    tasks::Vector,
    worker::Function,
    n_workers::Int;
    desc::String="tasks",
    timeout::Union{Float64,Nothing}=nothing
)::Vector{Any}
    if isempty(tasks)
        return Any[]
    end

    n_workers = max(1, n_workers)

    # Sequential execution for single worker or single task
    if n_workers == 1 || length(tasks) == 1
        return [worker(task) for task in tasks]
    end

    println("[Parallel] Executing $desc across $n_workers threads...")

    results = Vector{Any}(undef, length(tasks))
    results_lock = ReentrantLock()
    task_queue = Channel{Tuple{Int,Any}}(length(tasks))

    # Fill the queue
    for (i, task) in enumerate(tasks)
        put!(task_queue, (i, task))
    end
    close(task_queue)

    # Start worker threads
    worker_tasks = Task[]
    for _ in 1:min(n_workers, length(tasks))
        t = @async begin
            for (idx, task_args) in task_queue
                try
                    result = worker(task_args)
                    lock(results_lock) do
                        results[idx] = result
                    end
                catch e
                    lock(results_lock) do
                        results[idx] = Dict{String,Any}(
                            "status" => "failed",
                            "error" => string(e)
                        )
                    end
                end
            end
        end
        push!(worker_tasks, t)
    end

    # Wait for completion with optional timeout
    start_time = time()
    for t in worker_tasks
        remaining = isnothing(timeout) ? nothing : max(0.0, timeout - (time() - start_time))
        try
            if isnothing(remaining)
                wait(t)
            else
                timedwait(() -> istaskdone(t), remaining)
            end
        catch e
            @warn "Worker task error" exception=e
        end
    end

    # Check for incomplete results
    incomplete_count = count(i -> !isassigned(results, i), 1:length(results))
    if incomplete_count > 0
        println("[Parallel] $incomplete_count tasks did not complete; running sequentially...")
        for i in 1:length(results)
            if !isassigned(results, i)
                try
                    results[i] = worker(tasks[i])
                catch e
                    results[i] = Dict{String,Any}("status" => "failed", "error" => string(e))
                end
            end
        end
    end

    return results
end

# ============================================================================
# RESUME/RECOVERY FUNCTIONALITY
# ============================================================================

"""
Build a rescue configuration for retrying failed runs.
Disables parallelism for more reliable execution.
"""
function build_rescue_config(base_config::EmergentConfig)::EmergentConfig
    rescue = deepcopy(base_config)
    # Disable any parallel features for rescue attempts
    return rescue
end

"""
Reset a run directory for retry by removing existing files.
"""
function reset_run_directory(results_dir::String, run_id::String)
    run_dir = joinpath(results_dir, run_id)
    if isdir(run_dir)
        rm(run_dir; recursive=true, force=true)
    end
end

"""
Retry incomplete runs sequentially with rescue configuration.
"""
function retry_incomplete_runs!(
    results::Vector{Dict{String,Any}},
    tasks::Vector,
    worker::Function,
    results_dir::String,
    expected_rounds::Int,
    base_config::EmergentConfig
)
    incomplete_indices = Int[]

    for (i, result) in enumerate(results)
        run_id = get(result, "run_id", "")
        status = get(result, "status", "")

        if status != "completed" || !is_run_complete(results_dir, run_id, expected_rounds)
            push!(incomplete_indices, i)
        end
    end

    if isempty(incomplete_indices)
        return
    end

    println("[Recovery] Retrying $(length(incomplete_indices)) incomplete runs...")
    rescue_config = build_rescue_config(base_config)

    for idx in incomplete_indices
        run_id = get(results[idx], "run_id", "run_$idx")
        println("  Retrying $run_id...")

        reset_run_directory(results_dir, run_id)

        try
            result = worker(tasks[idx])
            results[idx] = result

            if result["status"] == "completed" && is_run_complete(results_dir, run_id, expected_rounds)
                println("  ✓ $run_id completed on retry")
            else
                println("  ✗ $run_id still incomplete after retry")
                results[idx]["status"] = "failed"
            end
        catch e
            println("  ✗ $run_id failed: $e")
            results[idx] = Dict{String,Any}(
                "run_id" => run_id,
                "status" => "failed",
                "error" => string(e)
            )
        end
    end
end

# ============================================================================
# CALIBRATION PROFILES
# ============================================================================

"""
Built-in calibration profiles.
"""
const CALIBRATION_PROFILES = Dict{String,Dict{String,Any}}(
    "default" => Dict{String,Any}(
        "name" => "default",
        "description" => "Default calibration settings"
    ),
    "high_uncertainty" => Dict{String,Any}(
        "name" => "high_uncertainty",
        "description" => "High Knightian uncertainty environment",
        "BASE_ACTOR_IGNORANCE" => 0.6,
        "BASE_PRACTICAL_INDETERMINISM" => 0.5,
        "BASE_AGENTIC_NOVELTY" => 0.55,
        "BASE_COMPETITIVE_RECURSION" => 0.6
    ),
    "stable_market" => Dict{String,Any}(
        "name" => "stable_market",
        "description" => "Stable market conditions with low volatility",
        "MARKET_SHIFT_PROBABILITY" => 0.02,
        "BASE_PRACTICAL_INDETERMINISM" => 0.25,
        "BASE_COMPETITIVE_RECURSION" => 0.3
    ),
    "high_competition" => Dict{String,Any}(
        "name" => "high_competition",
        "description" => "High competitive pressure environment",
        "BASE_COMPETITIVE_RECURSION" => 0.7,
        "COMPETITION_COST_MULTIPLIER" => 1.5,
        "COMPETITION_EFFECT" => 0.8
    )
)

"""
List available calibration profiles.
"""
function list_calibration_profiles()::Vector{String}
    return collect(keys(CALIBRATION_PROFILES))
end

"""
Load a calibration profile by name.
"""
function load_calibration_profile(name::String)::Union{Dict{String,Any},Nothing}
    return get(CALIBRATION_PROFILES, lowercase(name), nothing)
end

"""
Apply a calibration profile to a configuration.
"""
function apply_calibration_profile!(config::EmergentConfig, profile_name::String)::EmergentConfig
    profile = load_calibration_profile(profile_name)

    if isnothing(profile)
        @warn "Unknown calibration profile: $profile_name"
        return config
    end

    println("[CLI] Applying calibration profile: $(profile["name"])")

    for (key, value) in profile
        if key in ["name", "description"]
            continue
        end

        sym = Symbol(key)
        if hasfield(typeof(config), sym)
            setfield!(config, sym, value)
        end
    end

    return config
end

# ============================================================================
# DEFAULT SENSITIVITY GRID AND LHS RANGES
# ============================================================================

"""
Default sensitivity grid for parameter sweeps.
"""
const DEFAULT_SENSITIVITY_GRID = Dict{String,Vector{Any}}(
    "BASE_OPERATIONAL_COST" => [60_000.0, 70_000.0, 80_000.0],
    "SURVIVAL_CAPITAL_RATIO" => [0.45, 0.52, 0.58],
    "RETURN_OVERSUPPLY_PENALTY" => [0.4, 0.55, 0.7]
)

"""
Default ranges for Latin Hypercube Sampling.
"""
const DEFAULT_LHS_RANGES = Dict{String,Tuple{Float64,Float64}}(
    # Economics/viability
    "BASE_OPERATIONAL_COST" => (60_000.0, 80_000.0),
    "SURVIVAL_CAPITAL_RATIO" => (0.45, 0.60),
    # Demand/returns shape
    "RETURN_OVERSUPPLY_PENALTY" => (0.45, 0.70),
    "RETURN_UNDERSUPPLY_BONUS" => (0.20, 0.45),
    "RETURN_DEMAND_CROWDING_THRESHOLD" => (0.35, 0.50),
    "RETURN_DEMAND_CROWDING_PENALTY" => (0.35, 0.60),
    "FAILURE_DEMAND_PRESSURE" => (0.15, 0.30),
    "MARKET_SHIFT_PROBABILITY" => (0.03, 0.10),
    # AI-related
    "AI_NOVELTY_UPLIFT" => (0.04, 0.06),
    "DOWNSIDE_OVERSUPPLY_WEIGHT" => (0.55, 0.90),
    "RETURN_LOWER_BOUND" => (-1.2, -1.0)
)

# ============================================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================================

"""
Generate Latin Hypercube Samples for sensitivity analysis.

Returns a vector of dictionaries, each containing parameter values sampled
using the Latin Hypercube method for efficient coverage of the parameter space.
"""
function latin_hypercube_samples(
    bounds::Dict{String,Tuple{Float64,Float64}},
    n_samples::Int;
    seed::Union{Int,Nothing}=nothing
)::Vector{Dict{String,Float64}}
    if n_samples <= 0 || isempty(bounds)
        return Dict{String,Float64}[]
    end

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    param_names = collect(keys(bounds))
    n_params = length(param_names)

    # Create Latin Hypercube structure
    intervals = range(0.0, 1.0, length=n_samples + 1)
    points = rand(rng, n_params, n_samples)
    spans = diff(intervals)

    lhs = zeros(n_params, n_samples)
    for i in 1:n_samples
        for dim in 1:n_params
            lhs[dim, i] = intervals[i] + points[dim, i] * spans[i]
        end
    end

    # Shuffle each dimension independently
    for dim in 1:n_params
        lhs[dim, :] = lhs[dim, randperm(rng, n_samples)]
    end

    # Convert to parameter dictionaries
    samples = Dict{String,Float64}[]
    for i in 1:n_samples
        sample = Dict{String,Float64}()
        for (dim, name) in enumerate(param_names)
            low, high = bounds[name]
            sample[name] = low + lhs[dim, i] * (high - low)
        end
        push!(samples, sample)
    end

    return samples
end

# ============================================================================
# EXPERIMENT PROFILES
# ============================================================================

"""
Apply predefined experimental profiles (e.g., AGI-tier pricing).
"""
function apply_experiment_profile(config::EmergentConfig, profile::String)::EmergentConfig
    cfg = deepcopy(config)

    if profile == "agi2027"
        # Near-AGI tier pricing structure
        cfg.AI_LEVELS = Dict{String,Dict{String,Any}}(
            "none" => Dict{String,Any}(
                "cost" => 0.0,
                "cost_type" => "none",
                "info_quality" => 0.0,
                "info_breadth" => 0.0,
                "per_use_cost" => 0.0
            ),
            "basic" => Dict{String,Any}(
                "cost" => 50.0,
                "cost_type" => "subscription",
                "info_quality" => 0.55,
                "info_breadth" => 0.45,
                "per_use_cost" => 4.0
            ),
            "advanced" => Dict{String,Any}(
                "cost" => 1250.0,
                "cost_type" => "subscription",
                "info_quality" => 0.82,
                "info_breadth" => 0.70,
                "per_use_cost" => 5.0
            ),
            "premium" => Dict{String,Any}(
                "cost" => 18000.0,
                "cost_type" => "subscription",
                "info_quality" => 0.96,
                "info_breadth" => 0.92,
                "per_use_cost" => 25.0
            )
        )
        println("[CLI] Applied AGI-2027 pricing profile")
    elseif profile == "high_uncertainty"
        # High Knightian uncertainty environment
        cfg.BASE_ACTOR_IGNORANCE = 0.6
        cfg.BASE_PRACTICAL_INDETERMINISM = 0.5
        cfg.BASE_AGENTIC_NOVELTY = 0.55
        cfg.BASE_COMPETITIVE_RECURSION = 0.6
        println("[CLI] Applied high-uncertainty environment profile")
    elseif profile == "stable_market"
        # Stable market conditions
        cfg.MARKET_SHIFT_PROBABILITY = 0.02
        cfg.BASE_PRACTICAL_INDETERMINISM = 0.25
        cfg.BASE_COMPETITIVE_RECURSION = 0.3
        println("[CLI] Applied stable-market profile")
    else
        @warn "Unknown experiment profile: $profile - using default config"
    end

    return cfg
end

# ============================================================================
# PROGRESS MONITORING
# ============================================================================

"""
Progress monitoring for long-running experiments.
"""
mutable struct ProgressMonitor
    results_dir::String
    interval::Float64
    stop_flag::Ref{Bool}
    task::Union{Task,Nothing}
    limit::Union{Int,Nothing}
end

"""
Create a progress monitor.
"""
function ProgressMonitor(results_dir::String; interval::Float64=30.0, limit::Union{Int,Nothing}=nothing)
    return ProgressMonitor(results_dir, max(1.0, interval), Ref(false), nothing, limit)
end

"""
Start the progress monitor background task.
"""
function start_monitor!(monitor::ProgressMonitor)
    monitor.stop_flag[] = false
    monitor.task = @async begin
        last_report = ""
        while !monitor.stop_flag[]
            if !isdir(monitor.results_dir)
                report = "[monitor] Waiting for results directory: $(monitor.results_dir)"
            else
                try
                    report = _collect_progress_report(monitor.results_dir, monitor.limit)
                catch e
                    report = "[monitor] Unable to read progress: $e"
                end
            end

            timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            full_report = "[monitor] [$timestamp]\n$report"

            if full_report != last_report
                println(full_report)
                flush(stdout)
                last_report = full_report
            end

            sleep(monitor.interval)
        end
    end
    println("[monitor] Started progress monitoring (interval: $(monitor.interval)s)")
end

"""
Stop the progress monitor.
"""
function stop_monitor!(monitor::ProgressMonitor)
    monitor.stop_flag[] = true
    if !isnothing(monitor.task)
        try
            wait(monitor.task)
        catch
            # Task may have already finished
        end
    end
    println("[monitor] Stopped progress monitoring")
end

"""
Collect progress information from results directory.
"""
function _collect_progress_report(results_dir::String, limit::Union{Int,Nothing})::String
    run_dirs = filter(isdir, [joinpath(results_dir, d) for d in readdir(results_dir)])

    if isempty(run_dirs)
        return "No runs found yet"
    end

    # Limit to most recent runs if specified
    if !isnothing(limit) && length(run_dirs) > limit
        run_dirs = run_dirs[(end-limit+1):end]
    end

    completed = 0
    failed = 0
    in_progress = 0

    for dir in run_dirs
        log_path = joinpath(dir, "run_log.jsonl")
        if isfile(log_path)
            last_round = _read_last_round(log_path)
            if !isnothing(last_round)
                if last_round >= 199  # Assuming 200 rounds
                    completed += 1
                else
                    in_progress += 1
                end
            else
                in_progress += 1
            end
        else
            in_progress += 1
        end
    end

    return "Runs: $(length(run_dirs)) total | $completed completed | $in_progress in progress | $failed failed"
end

"""
Read the last round from a JSONL log file.
"""
function _read_last_round(log_path::String)::Union{Int,Nothing}
    if !isfile(log_path)
        return nothing
    end

    last_line = ""
    for line in eachline(log_path)
        stripped = strip(line)
        if !isempty(stripped)
            last_line = stripped
        end
    end

    if isempty(last_line)
        return nothing
    end

    try
        record = JSON3.read(last_line)
        round_val = get(record, :round, nothing)
        if isnothing(round_val)
            return nothing
        end
        return Int(round_val)
    catch
        return nothing
    end
end

# ============================================================================
# RUNTIME ESTIMATION
# ============================================================================

"""
Estimate runtime for a simulation configuration.
"""
function estimate_runtime(config::EmergentConfig; n_runs::Int=50)::Dict{String,Any}
    # Empirical constants from benchmarking
    base_time_per_round = 0.015  # seconds per round per 100 agents
    agent_factor = config.N_AGENTS / 100.0
    round_factor = config.N_ROUNDS

    estimated_per_run = base_time_per_round * agent_factor * round_factor
    estimated_total = estimated_per_run * n_runs

    # Account for I/O overhead
    io_overhead = 1.15  # 15% overhead for disk writes
    estimated_total *= io_overhead

    return Dict{String,Any}(
        "per_run_seconds" => estimated_per_run,
        "total_seconds" => estimated_total,
        "total_minutes" => estimated_total / 60,
        "n_runs" => n_runs,
        "n_agents" => config.N_AGENTS,
        "n_rounds" => config.N_ROUNDS
    )
end

# ============================================================================
# RUN COMPLETION CHECK
# ============================================================================

"""
Check if a run is complete by examining its output.
"""
function is_run_complete(results_dir::String, run_id::String, expected_rounds::Int)::Bool
    run_dir = joinpath(results_dir, run_id)
    if !isdir(run_dir)
        return false
    end

    # Check run log
    log_path = joinpath(run_dir, "run_log.jsonl")
    last_round = _read_last_round(log_path)

    if !isnothing(last_round) && last_round >= expected_rounds - 1
        return true
    end

    # Check for final agents file
    for ext in [".csv", ".arrow", ".jld2"]
        if isfile(joinpath(run_dir, "final_agents$ext"))
            return true
        end
    end

    return false
end

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

"""
Run sensitivity analysis with Latin Hypercube Sampling.
"""
function run_sensitivity_analysis(;
    base_config::EmergentConfig,
    output_dir::String,
    n_samples::Int=20,
    ranges::Dict{String,Tuple{Float64,Float64}}=DEFAULT_LHS_RANGES,
    base_seed::Int=42,
    verbose::Bool=false
)::Dict{String,Any}
    println("\n" * "="^70)
    println("SENSITIVITY ANALYSIS (Latin Hypercube Sampling)")
    println("="^70)
    println("Samples: $n_samples")
    println("Parameters: $(join(keys(ranges), ", "))")
    println("="^70 * "\n")

    mkpath(output_dir)

    # Generate LHS samples
    samples = latin_hypercube_samples(ranges, n_samples; seed=base_seed)

    results = Dict{String,Any}[]

    for (sample_idx, params) in enumerate(samples)
        run_id = "sensitivity_sample_$(sample_idx)"
        run_output = joinpath(output_dir, run_id)

        config = deepcopy(base_config)
        initialize!(config)

        # Apply parameter values
        for (param, value) in params
            if hasproperty(config, Symbol(param))
                setfield!(config, Symbol(param), value)
            end
        end

        if verbose
            println("[$run_id] Running with parameters:")
            for (p, v) in params
                println("  $p = $(round(v, digits=4))")
            end
        end

        result = run_single_simulation(
            config=config,
            output_dir=run_output,
            run_id=run_id,
            seed=base_seed + sample_idx,
            verbose=verbose
        )

        result["parameters"] = params
        result["sample_index"] = sample_idx
        push!(results, result)

        if sample_idx % 5 == 0
            completed = count(r -> r["status"] == "completed" for r in results)
            println("[Progress] $sample_idx/$n_samples samples completed ($completed successful)")
        end
    end

    # Save sensitivity results
    meta_path = joinpath(output_dir, "sensitivity_results.json")
    open(meta_path, "w") do io
        JSON3.write(io, results)
    end

    println("\n" * "="^70)
    println("SENSITIVITY ANALYSIS COMPLETE")
    println("="^70)
    completed = count(r -> r["status"] == "completed" for r in results)
    println("Successful samples: $completed / $n_samples")
    println("Results saved to: $output_dir")
    println("="^70)

    return Dict{String,Any}(
        "n_samples" => n_samples,
        "completed" => completed,
        "results" => results,
        "output_dir" => output_dir
    )
end

# ============================================================================
# SIMULATION RUNNERS
# ============================================================================

"""
Run a single simulation with optional fixed AI level.
"""
function run_single_simulation(;
    config::EmergentConfig,
    output_dir::String,
    run_id::String,
    seed::Int,
    fixed_ai_level::Union{String,Nothing}=nothing,
    verbose::Bool=false
)::Dict{String,Any}
    try
        sim = EmergentSimulation(
            config=config,
            output_dir=output_dir,
            run_id=run_id,
            seed=seed
        )

        # Set fixed AI level if specified
        if !isnothing(fixed_ai_level)
            for agent in sim.agents
                agent.fixed_ai_level = fixed_ai_level
                agent.current_ai_level = fixed_ai_level
            end
        end

        run!(sim)
        save_results!(sim)

        stats = summary_stats(sim)

        return Dict{String,Any}(
            "run_id" => run_id,
            "status" => "completed",
            "survival_rate" => stats["final_survival_rate"],
            "mean_capital" => stats["mean_final_capital"],
            "elapsed_seconds" => stats["elapsed_seconds"]
        )
    catch e
        @warn "Simulation $run_id failed" exception=(e, catch_backtrace())
        return Dict{String,Any}(
            "run_id" => run_id,
            "status" => "failed",
            "error" => string(e)
        )
    end
end

"""
Run fixed AI level sweep (causal inference design).
"""
function run_fixed_level_sweep(;
    base_config::EmergentConfig,
    output_dir::String,
    n_runs::Int=50,
    ai_levels::Vector{String}=["none", "basic", "advanced", "premium"],
    base_seed::Int=42,
    verbose::Bool=false
)::Vector{Dict{String,Any}}
    results = Dict{String,Any}[]

    println("\n" * "="^70)
    println("FIXED AI LEVEL SWEEP")
    println("="^70)
    println("AI Levels: $(join(ai_levels, ", "))")
    println("Runs per level: $n_runs")
    println("Total simulations: $(length(ai_levels) * n_runs)")
    println("="^70 * "\n")

    for (tier_idx, ai_level) in enumerate(ai_levels)
        println("\n[$(uppercase(ai_level))] Running $n_runs simulations...")

        for run_idx in 1:n_runs
            run_id = "Fixed_AI_Level_$(ai_level)_run_$(run_idx)"
            run_output = joinpath(output_dir, run_id)

            # Compute seed
            seed = base_seed + (tier_idx - 1) * n_runs + run_idx

            config = deepcopy(base_config)
            initialize!(config)

            result = run_single_simulation(
                config=config,
                output_dir=run_output,
                run_id=run_id,
                seed=seed,
                fixed_ai_level=ai_level,
                verbose=verbose
            )

            result["ai_level"] = ai_level
            result["run_index"] = run_idx
            push!(results, result)

            if verbose || run_idx % 10 == 0
                status = result["status"]
                if status == "completed"
                    sr = round(result["survival_rate"] * 100, digits=1)
                    println("  [$run_id] Completed - Survival: $sr%")
                else
                    println("  [$run_id] $status")
                end
            end
        end
    end

    # Print summary
    println("\n" * "="^70)
    println("SWEEP SUMMARY")
    println("="^70)

    for ai_level in ai_levels
        level_results = filter(r -> get(r, "ai_level", "") == ai_level && r["status"] == "completed", results)
        if !isempty(level_results)
            survival_rates = [r["survival_rate"] for r in level_results]
            mean_sr = mean(survival_rates) * 100
            std_sr = std(survival_rates) * 100
            println("$(uppercase(ai_level)):  $(round(mean_sr, digits=1))% ± $(round(std_sr, digits=1))% survival")
        end
    end

    return results
end

"""
Run emergent simulation batch.
"""
function run_emergent_batch(;
    base_config::EmergentConfig,
    output_dir::String,
    n_runs::Int=50,
    base_seed::Int=42,
    verbose::Bool=false
)::Vector{Dict{String,Any}}
    results = Dict{String,Any}[]

    println("\n" * "="^70)
    println("EMERGENT AI SELECTION BATCH")
    println("="^70)
    println("Runs: $n_runs")
    println("="^70 * "\n")

    for run_idx in 1:n_runs
        run_id = "emergent_run_$(run_idx)"
        run_output = joinpath(output_dir, run_id)

        config = deepcopy(base_config)
        initialize!(config)

        result = run_single_simulation(
            config=config,
            output_dir=run_output,
            run_id=run_id,
            seed=base_seed + run_idx,
            verbose=verbose
        )

        result["run_index"] = run_idx
        push!(results, result)

        if verbose || run_idx % 10 == 0
            status = result["status"]
            if status == "completed"
                sr = round(result["survival_rate"] * 100, digits=1)
                println("  [$run_id] Completed - Survival: $sr%")
            else
                println("  [$run_id] $status")
            end
        end
    end

    return results
end

"""
Run master launcher (emergent + optional fixed sweep).
"""
function run_master_launcher(;
    base_config::EmergentConfig,
    output_dir::String="results",
    n_runs::Int=50,
    run_emergent::Bool=true,
    run_fixed::Bool=false,
    ai_levels::Vector{String}=["none", "basic", "advanced", "premium"],
    base_seed::Int=42,
    skip_visualizations::Bool=false,
    verbose::Bool=false
)::Dict{String,Any}
    mkpath(output_dir)

    start_time = now()
    all_results = Dict{String,Any}(
        "output_dir" => output_dir,
        "start_time" => string(start_time),
        "emergent_results" => Dict{String,Any}[],
        "fixed_results" => Dict{String,Any}[]
    )

    # Save config snapshot
    config_path = joinpath(output_dir, "config_snapshot.json")
    save_config_snapshot(base_config, config_path)

    # Run emergent batch
    if run_emergent
        emergent_results = run_emergent_batch(
            base_config=base_config,
            output_dir=output_dir,
            n_runs=n_runs,
            base_seed=base_seed,
            verbose=verbose
        )
        all_results["emergent_results"] = emergent_results
    end

    # Run fixed level sweep
    if run_fixed
        fixed_results = run_fixed_level_sweep(
            base_config=base_config,
            output_dir=output_dir,
            n_runs=n_runs,
            ai_levels=ai_levels,
            base_seed=base_seed + 10000,  # Different seed range
            verbose=verbose
        )
        all_results["fixed_results"] = fixed_results
    end

    elapsed = (now() - start_time).value / 1000.0
    all_results["elapsed_seconds"] = elapsed
    all_results["end_time"] = string(now())

    # Save results metadata
    meta_path = joinpath(output_dir, "run_metadata.json")
    open(meta_path, "w") do io
        JSON3.write(io, all_results)
    end

    println("\n" * "="^70)
    println("MASTER LAUNCHER COMPLETE")
    println("="^70)
    println("Total elapsed: $(round(elapsed / 60, digits=1)) minutes")
    println("Results saved to: $output_dir")
    println("="^70)

    return all_results
end

# ============================================================================
# UNCERTAINTY SCENARIO SWEEP
# ============================================================================

"""
Build uncertainty scenarios for sweep experiments.
"""
function build_scenarios(base_config::EmergentConfig)::Dict{String,EmergentConfig}
    scenarios = Dict{String,EmergentConfig}()

    # Baseline scenario
    baseline = deepcopy(base_config)
    scenarios["Baseline"] = baseline

    # High Ignorance scenario
    ignorance = deepcopy(base_config)
    if hasfield(typeof(ignorance), :BASE_ACTOR_IGNORANCE)
        ignorance.BASE_ACTOR_IGNORANCE = 0.7
    end
    scenarios["High_Ignorance"] = ignorance

    # High Indeterminism scenario
    indeterminism = deepcopy(base_config)
    if hasfield(typeof(indeterminism), :BASE_PRACTICAL_INDETERMINISM)
        indeterminism.BASE_PRACTICAL_INDETERMINISM = 0.6
    end
    if hasfield(typeof(indeterminism), :MARKET_VOLATILITY)
        indeterminism.MARKET_VOLATILITY = 0.45
    end
    scenarios["High_Indeterminism"] = indeterminism

    # High Agentic Novelty scenario
    novelty = deepcopy(base_config)
    if hasfield(typeof(novelty), :BASE_AGENTIC_NOVELTY)
        novelty.BASE_AGENTIC_NOVELTY = 0.65
    end
    scenarios["High_AgenticNovelty"] = novelty

    # High Competitive Recursion scenario
    recursion = deepcopy(base_config)
    if hasfield(typeof(recursion), :BASE_COMPETITIVE_RECURSION)
        recursion.BASE_COMPETITIVE_RECURSION = 0.7
    end
    scenarios["High_CompetitiveRecursion"] = recursion

    return scenarios
end

"""
Run uncertainty scenario sweep experiment.

Executes simulations across multiple uncertainty scenarios and collects results.
"""
function run_uncertainty_scenario_sweep(;
    base_config::EmergentConfig,
    output_dir::String,
    n_runs::Int=10,
    scenario_order::Vector{String}=["Baseline", "High_Ignorance", "High_Indeterminism",
                                    "High_AgenticNovelty", "High_CompetitiveRecursion"],
    base_seed::Int=42,
    verbose::Bool=false
)::Dict{String,Any}
    println("\n" * "="^70)
    println("UNCERTAINTY SCENARIO SWEEP")
    println("="^70)
    println("Scenarios: $(join(scenario_order, ", "))")
    println("Runs per scenario: $n_runs")
    println("Total simulations: $(length(scenario_order) * n_runs)")
    println("="^70 * "\n")

    mkpath(output_dir)

    # Build scenarios
    scenarios = build_scenarios(base_config)

    # Compute parallel plan
    max_workers, timeout, workload = compute_parallel_plan(base_config)

    results = Dict{String,Any}(
        "scenarios" => Dict{String,Vector{Dict{String,Any}}}(),
        "output_dir" => output_dir,
        "start_time" => string(now())
    )

    for scenario_name in scenario_order
        if !haskey(scenarios, scenario_name)
            println("⚠️ Unknown scenario: $scenario_name - skipping")
            continue
        end

        scenario_config = scenarios[scenario_name]
        scenario_dir = joinpath(output_dir, scenario_name)
        mkpath(scenario_dir)

        println("\n[$scenario_name] Running $n_runs simulations...")

        scenario_results = Dict{String,Any}[]

        for run_idx in 1:n_runs
            run_id = "$(scenario_name)_run_$(run_idx)"
            run_output = joinpath(scenario_dir, run_id)

            config = deepcopy(scenario_config)
            initialize!(config)

            result = run_single_simulation(
                config=config,
                output_dir=run_output,
                run_id=run_id,
                seed=base_seed + run_idx,
                verbose=verbose
            )

            result["scenario"] = scenario_name
            result["run_index"] = run_idx
            push!(scenario_results, result)

            if verbose || run_idx % 5 == 0
                status = result["status"]
                if status == "completed"
                    sr = round(result["survival_rate"] * 100, digits=1)
                    println("  [$run_id] Completed - Survival: $sr%")
                else
                    println("  [$run_id] $status")
                end
            end
        end

        results["scenarios"][scenario_name] = scenario_results
    end

    # Print summary
    println("\n" * "="^70)
    println("SCENARIO SWEEP SUMMARY")
    println("="^70)

    for scenario_name in scenario_order
        if haskey(results["scenarios"], scenario_name)
            scenario_results = results["scenarios"][scenario_name]
            completed = filter(r -> r["status"] == "completed", scenario_results)
            if !isempty(completed)
                survival_rates = [r["survival_rate"] for r in completed]
                mean_sr = mean(survival_rates) * 100
                std_sr = length(survival_rates) > 1 ? std(survival_rates) * 100 : 0.0
                println("$scenario_name:  $(round(mean_sr, digits=1))% ± $(round(std_sr, digits=1))% survival")
            end
        end
    end

    # Save results
    meta_path = joinpath(output_dir, "scenario_results.json")
    open(meta_path, "w") do io
        JSON3.write(io, results)
    end

    return results
end

# ============================================================================
# SENSITIVITY SWEEP (GRID-BASED)
# ============================================================================

"""
Run sensitivity sweep using grid-based parameter combinations.
"""
function run_sensitivity_sweep(;
    base_config::EmergentConfig,
    output_dir::String,
    param_grid::Dict{String,Vector{Any}}=DEFAULT_SENSITIVITY_GRID,
    runs_per_combo::Int=1,
    base_seed::Int=42,
    verbose::Bool=false
)::Dict{String,Any}
    println("\n" * "="^70)
    println("SENSITIVITY SWEEP (GRID-BASED)")
    println("="^70)

    param_names = collect(keys(param_grid))
    println("Parameters: $(join(param_names, ", "))")

    # Generate all combinations
    param_values = [param_grid[name] for name in param_names]
    combos = collect(Iterators.product(param_values...))

    println("Total combinations: $(length(combos))")
    println("Runs per combination: $runs_per_combo")
    println("Total simulations: $(length(combos) * runs_per_combo)")
    println("="^70 * "\n")

    mkpath(output_dir)

    results = Dict{String,Any}[]

    for (combo_idx, combo_values) in enumerate(combos)
        overrides = Dict{String,Any}(
            param_names[i] => combo_values[i] for i in 1:length(param_names)
        )

        combo_dir = joinpath(output_dir, "combo_$combo_idx")
        mkpath(combo_dir)

        if verbose
            println("[Combo $combo_idx] Running with: $overrides")
        end

        for run_idx in 1:runs_per_combo
            config = deepcopy(base_config)
            initialize!(config)

            # Apply parameter overrides
            for (param, value) in overrides
                sym = Symbol(param)
                if hasfield(typeof(config), sym)
                    setfield!(config, sym, value)
                end
            end

            run_id = "sensitivity_combo_$(combo_idx)_run_$(run_idx)"
            run_output = joinpath(combo_dir, run_id)

            result = run_single_simulation(
                config=config,
                output_dir=run_output,
                run_id=run_id,
                seed=base_seed + combo_idx * 100 + run_idx,
                verbose=verbose
            )

            result["combo_index"] = combo_idx
            result["parameters"] = overrides
            result["run_index"] = run_idx
            push!(results, result)
        end

        if combo_idx % 5 == 0
            completed = count(r -> r["status"] == "completed", results)
            println("[Progress] Combo $combo_idx/$(length(combos)) ($(completed) successful)")
        end
    end

    # Compute sensitivity effects
    effects = compute_sensitivity_effects(results, param_names)

    # Save results
    summary = Dict{String,Any}(
        "n_combos" => length(combos),
        "runs_per_combo" => runs_per_combo,
        "parameters" => param_names,
        "results" => results,
        "effects" => effects,
        "output_dir" => output_dir
    )

    meta_path = joinpath(output_dir, "sensitivity_results.json")
    open(meta_path, "w") do io
        JSON3.write(io, summary)
    end

    println("\n" * "="^70)
    println("SENSITIVITY SWEEP COMPLETE")
    println("="^70)
    completed = count(r -> r["status"] == "completed", results)
    println("Successful runs: $completed / $(length(results))")
    println("Results saved to: $output_dir")
    println("="^70)

    return summary
end

"""
Compute sensitivity effects from sweep results.
"""
function compute_sensitivity_effects(
    results::Vector{Dict{String,Any}},
    param_names::Vector{String}
)::Dict{String,Any}
    effects = Dict{String,Any}()

    # Get completed results with survival rates
    completed = filter(r -> r["status"] == "completed" && haskey(r, "survival_rate"), results)

    if isempty(completed)
        return effects
    end

    for param in param_names
        param_values = unique([r["parameters"][param] for r in completed])

        if length(param_values) < 2
            continue
        end

        # Group by parameter value
        grouped = Dict{Any,Vector{Float64}}()
        for r in completed
            val = r["parameters"][param]
            if !haskey(grouped, val)
                grouped[val] = Float64[]
            end
            push!(grouped[val], r["survival_rate"])
        end

        # Compute mean survival rate for each value
        means = Dict(val => mean(rates) for (val, rates) in grouped)

        # Compute range/effect
        min_val = minimum(values(means))
        max_val = maximum(values(means))
        effect_range = max_val - min_val

        effects[param] = Dict{String,Any}(
            "values" => param_values,
            "mean_survival_by_value" => means,
            "effect_range" => effect_range
        )
    end

    return effects
end

# ============================================================================
# LHS SWEEP
# ============================================================================

"""
Run Latin Hypercube Sampling sweep for comprehensive sensitivity analysis.
"""
function run_lhs_sweep(;
    base_config::EmergentConfig,
    output_dir::String,
    n_samples::Int=20,
    ranges::Dict{String,Tuple{Float64,Float64}}=DEFAULT_LHS_RANGES,
    runs_per_sample::Int=1,
    base_seed::Int=42,
    verbose::Bool=false
)::Dict{String,Any}
    println("\n" * "="^70)
    println("LATIN HYPERCUBE SAMPLING SWEEP")
    println("="^70)
    println("Samples: $n_samples")
    println("Parameters: $(join(keys(ranges), ", "))")
    println("Runs per sample: $runs_per_sample")
    println("Total simulations: $(n_samples * runs_per_sample)")
    println("="^70 * "\n")

    mkpath(output_dir)

    # Generate LHS samples
    samples = latin_hypercube_samples(ranges, n_samples; seed=base_seed)

    results = Dict{String,Any}[]

    for (sample_idx, params) in enumerate(samples)
        sample_dir = joinpath(output_dir, "lhs_$sample_idx")
        mkpath(sample_dir)

        if verbose
            println("[Sample $sample_idx] Parameters:")
            for (p, v) in params
                println("  $p = $(round(v, digits=4))")
            end
        end

        for run_idx in 1:runs_per_sample
            config = deepcopy(base_config)
            initialize!(config)

            # Apply parameter values
            for (param, value) in params
                sym = Symbol(param)
                if hasfield(typeof(config), sym)
                    setfield!(config, sym, value)
                end
            end

            run_id = "lhs_$(sample_idx)_run_$(run_idx)"
            run_output = joinpath(sample_dir, run_id)

            result = run_single_simulation(
                config=config,
                output_dir=run_output,
                run_id=run_id,
                seed=base_seed + sample_idx * 100 + run_idx,
                verbose=verbose
            )

            result["sample_index"] = sample_idx
            result["parameters"] = params
            result["run_index"] = run_idx
            push!(results, result)
        end

        if sample_idx % 5 == 0
            completed = count(r -> r["status"] == "completed", results)
            println("[Progress] $sample_idx/$n_samples samples ($(completed) successful)")
        end
    end

    # Save results
    summary = Dict{String,Any}(
        "n_samples" => n_samples,
        "runs_per_sample" => runs_per_sample,
        "parameter_ranges" => ranges,
        "results" => results,
        "output_dir" => output_dir
    )

    meta_path = joinpath(output_dir, "lhs_results.json")
    open(meta_path, "w") do io
        JSON3.write(io, summary)
    end

    println("\n" * "="^70)
    println("LHS SWEEP COMPLETE")
    println("="^70)
    completed = count(r -> r["status"] == "completed", results)
    println("Successful samples: $completed / $(length(results))")
    println("Results saved to: $output_dir")
    println("="^70)

    return summary
end

# ============================================================================
# MAIN CLI ENTRY POINT
# ============================================================================

"""
Main CLI entry point.
"""
function run_cli(args::Vector{String}=ARGS)
    parsed = parse_args(args)

    # Handle list-profiles early exit
    if parsed["list-profiles"]
        println("\nAvailable calibration profiles:")
        println("="^40)
        for name in list_calibration_profiles()
            profile = load_calibration_profile(name)
            desc = get(profile, "description", "No description")
            println("  $name: $desc")
        end
        println("="^40)
        return nothing
    end

    # Create base config
    config = EmergentConfig()
    config.N_AGENTS = parsed["agents"]
    config.N_ROUNDS = parsed["rounds"]
    config.RANDOM_SEED = parsed["seed"]

    # Apply calibration profile if specified
    if !isnothing(parsed["calibration-profile"])
        apply_calibration_profile!(config, parsed["calibration-profile"])
    end

    # Apply smoke overrides if requested
    if parsed["smoke"]
        apply_smoke_overrides!(config)
        println("[CLI] Smoke mode enabled - using reduced parameters")
    end

    # Dump config if requested
    if !isnothing(parsed["dump-config"])
        save_config_snapshot(config, parsed["dump-config"])
        println("[CLI] Config dumped to $(parsed["dump-config"])")
    end

    # Parse AI levels
    ai_levels = split(parsed["ai-levels"], ",") |> collect .|> strip .|> String

    # Parse scenarios
    scenarios = split(parsed["scenarios"], ",") |> collect .|> strip .|> String

    # Initialize config
    initialize!(config)

    # Dispatch based on task
    task = parsed["task"]
    results_dir = parsed["results-dir"]
    n_runs = parsed["runs"]
    verbose = parsed["verbose"]
    base_seed = config.RANDOM_SEED

    println("\n" * "="^70)
    println("GlimpseABM.jl CLI")
    println("="^70)
    println("Task: $task")
    println("Agents: $(config.N_AGENTS)")
    println("Rounds: $(config.N_ROUNDS)")
    println("Runs: $n_runs")
    println("Output: $results_dir")
    if !isnothing(parsed["calibration-profile"])
        println("Profile: $(parsed["calibration-profile"])")
    end
    println("="^70 * "\n")

    # Start progress monitor if requested
    monitor = nothing
    if parsed["monitor-progress"]
        monitor = ProgressMonitor(results_dir; interval=parsed["monitor-interval"])
        start_monitor!(monitor)
    end

    try
        if task == "master"
            return run_master_launcher(
                base_config=config,
                output_dir=results_dir,
                n_runs=n_runs,
                run_emergent=!parsed["skip-emergent"],
                run_fixed=parsed["include-fixed"] || parsed["fixed-tier-sweep"],
                ai_levels=ai_levels,
                base_seed=base_seed,
                skip_visualizations=parsed["skip-visualizations"],
                verbose=verbose
            )
        elseif task == "fixed"
            return run_fixed_level_sweep(
                base_config=config,
                output_dir=results_dir,
                n_runs=n_runs,
                ai_levels=ai_levels,
                base_seed=base_seed,
                verbose=verbose
            )
        elseif task == "sweep"
            return run_fixed_level_sweep(
                base_config=config,
                output_dir=results_dir,
                n_runs=n_runs,
                ai_levels=ai_levels,
                base_seed=base_seed,
                verbose=verbose
            )
        elseif task == "scenarios"
            return run_uncertainty_scenario_sweep(
                base_config=config,
                output_dir=results_dir,
                n_runs=n_runs,
                scenario_order=scenarios,
                base_seed=base_seed,
                verbose=verbose
            )
        elseif task == "sensitivity"
            return run_sensitivity_sweep(
                base_config=config,
                output_dir=results_dir,
                runs_per_combo=parsed["runs-per-combo"],
                base_seed=base_seed,
                verbose=verbose
            )
        elseif task == "lhs"
            return run_lhs_sweep(
                base_config=config,
                output_dir=results_dir,
                n_samples=parsed["lhs-samples"],
                runs_per_sample=parsed["runs-per-combo"],
                base_seed=base_seed,
                verbose=verbose
            )
        else
            println("[CLI] Unknown task: $task")
            println("Available tasks: master, fixed, sweep, scenarios, sensitivity, lhs")
            return nothing
        end
    finally
        if !isnothing(monitor)
            stop_monitor!(monitor)
        end
    end
end

# Entry point when run as script
if abspath(PROGRAM_FILE) == @__FILE__
    run_cli(ARGS)
end

end # module CLI
