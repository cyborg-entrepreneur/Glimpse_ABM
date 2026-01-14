"""
Lightweight, read-only progress monitor for active Glimpse ABM runs.

Port of: glimpse_abm/tools/progress_tracker.py
"""

module ProgressTracker

using Dates
using DataFrames
using Arrow

export RunProgress, collect_progress, render_report, watch_progress

const RUN_DIR_PATTERN = r"emergent_run_(\d+)"

"""
Progress information for a single run.
"""
struct RunProgress
    name::String
    rounds_completed::Int
end

"""
Get formatted round label.
"""
function round_label(rp::RunProgress)::String
    return lpad(string(rp.rounds_completed), 5)
end

"""
Sort run directories by index.
"""
function sorted_run_dirs(results_dir::String)::Vector{String}
    dirs = String[]

    if !isdir(results_dir)
        return dirs
    end

    for entry in readdir(results_dir)
        path = joinpath(results_dir, entry)
        if isdir(path) && occursin(RUN_DIR_PATTERN, entry)
            push!(dirs, path)
        end
    end

    # Sort by run index
    sort!(dirs, by=d -> begin
        m = match(RUN_DIR_PATTERN, basename(d))
        isnothing(m) ? 0 : parse(Int, m.captures[1])
    end)

    return dirs
end

"""
Load latest round from Arrow files in a subdirectory.
"""
function load_latest_round_from_subdir(subdir::String)::Union{Int,Nothing}
    if !isdir(subdir)
        return nothing
    end

    # Check for Arrow files
    arrow_files = filter(f -> endswith(f, ".arrow"), readdir(subdir))
    if isempty(arrow_files)
        return nothing
    end

    # Sort by name (assuming batch_N.arrow format)
    sort!(arrow_files, by=f -> begin
        m = match(r"batch_(\d+)", f)
        isnothing(m) ? 0 : parse(Int, m.captures[1])
    end, rev=true)

    for file in arrow_files
        try
            path = joinpath(subdir, file)
            df = DataFrame(Arrow.Table(path))
            if "round" in names(df) && nrow(df) > 0
                rounds = df.round
                valid_rounds = filter(x -> !ismissing(x), rounds)
                if !isempty(valid_rounds)
                    return Int(maximum(valid_rounds))
                end
            end
        catch
            continue
        end
    end

    return nothing
end

"""
Load latest round from run log file.
"""
function load_latest_round_from_log(log_path::String)::Union{Int,Nothing}
    if !isfile(log_path)
        return nothing
    end

    try
        count = 0
        open(log_path, "r") do f
            for line in eachline(f)
                if !isempty(strip(line))
                    count += 1
                end
            end
        end
        return count > 0 ? count - 1 : nothing
    catch
        return nothing
    end
end

"""
Detect progress for a single run directory.
"""
function detect_run_progress(run_dir::String)::RunProgress
    candidates = [
        load_latest_round_from_log(joinpath(run_dir, "run_log.jsonl")),
        load_latest_round_from_subdir(joinpath(run_dir, "summary")),
        load_latest_round_from_subdir(joinpath(run_dir, "decisions")),
        load_latest_round_from_subdir(joinpath(run_dir, "market"))
    ]

    # Filter out nothing values and get max
    valid = filter(!isnothing, candidates)
    max_round = isempty(valid) ? -1 : maximum(valid)
    rounds_completed = max_round >= 0 ? max_round + 1 : 0

    return RunProgress(basename(run_dir), rounds_completed)
end

"""
Collect progress from all run directories.
"""
function collect_progress(results_dir::String; limit::Union{Int,Nothing}=nothing)::Dict{String,RunProgress}
    progress = Dict{String,RunProgress}()

    run_dirs = sorted_run_dirs(results_dir)

    for (idx, run_dir) in enumerate(run_dirs)
        if !isnothing(limit) && idx > limit
            break
        end
        name = basename(run_dir)
        progress[name] = detect_run_progress(run_dir)
    end

    return progress
end

"""
Render a progress report as a string.
"""
function render_report(progress::Dict{String,RunProgress})::String
    if isempty(progress)
        return "No run directories discovered yet."
    end

    # Sort by run index
    ordered = sort(collect(values(progress)), by=rp -> begin
        m = match(RUN_DIR_PATTERN, rp.name)
        isnothing(m) ? 0 : parse(Int, m.captures[1])
    end)

    total_rounds = sum(rp.rounds_completed for rp in ordered)
    max_rounds = maximum(rp.rounds_completed for rp in ordered; init=0)
    active_runs = count(rp -> rp.rounds_completed > 0, ordered)

    lines = [
        "Runs tracked: $(length(ordered))  |  Active: $active_runs  |  Total rounds: $total_rounds  |  Max rounds: $max_rounds",
        "Recent runs:"
    ]

    # Show last 5 runs
    preview = length(ordered) > 5 ? ordered[end-4:end] : ordered
    for rp in preview
        push!(lines, "  $(rpad(rp.name, 16)) -> rounds: $(round_label(rp))")
    end

    return join(lines, "\n")
end

"""
Watch progress continuously with polling interval.
"""
function watch_progress(
    results_dir::String;
    interval::Float64=15.0,
    limit::Union{Int,Nothing}=nothing,
    once::Bool=false
)
    if !isdir(results_dir)
        println("Results directory not found: $results_dir")
        return
    end

    while true
        progress = collect_progress(results_dir; limit=limit)
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        println("[$timestamp]")
        println(render_report(progress))

        if once
            break
        end

        sleep(interval)
    end
end

"""
Main entry point for command-line usage.
"""
function main(args=ARGS)
    if isempty(args)
        println("Usage: julia progress_tracker.jl <results_dir> [--once] [--interval=N] [--limit=N]")
        return
    end

    results_dir = args[1]
    once = "--once" in args

    interval = 15.0
    for arg in args
        if startswith(arg, "--interval=")
            interval = parse(Float64, split(arg, "=")[2])
        end
    end

    limit = nothing
    for arg in args
        if startswith(arg, "--limit=")
            limit = parse(Int, split(arg, "=")[2])
        end
    end

    watch_progress(results_dir; interval=interval, limit=limit, once=once)
end

end # module ProgressTracker
