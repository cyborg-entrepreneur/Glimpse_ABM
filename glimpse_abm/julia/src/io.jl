"""
I/O utilities for GlimpseABM.jl

Provides dual format support:
- Python-compatible: pickle, CSV
- Julia-native: JLD2, Arrow

Port of: glimpse_abm file I/O patterns
"""

using DataFrames
using Arrow
using JLD2
using JSON3
using Dates

# ============================================================================
# JULIA NATIVE FORMATS
# ============================================================================

"""
Save agents to JLD2 format (Julia native).
"""
function save_agents_jld2(agents::Vector, path::String)
    jldsave(path; agents=agents)
end

"""
Load agents from JLD2 format.
"""
function load_agents_jld2(path::String)
    return load(path, "agents")
end

"""
Save DataFrame to Arrow format (cross-platform, efficient).
"""
function save_dataframe_arrow(df::DataFrame, path::String)
    Arrow.write(path, df)
end

"""
Load DataFrame from Arrow format.
"""
function load_dataframe_arrow(path::String)::DataFrame
    return DataFrame(Arrow.Table(path))
end

"""
Save DataFrame to CSV format (Python-compatible).
"""
function save_dataframe_csv(df::DataFrame, path::String)
    open(path, "w") do io
        # Write header
        println(io, join(names(df), ","))
        # Write rows
        for row in eachrow(df)
            vals = [ismissing(v) ? "" : string(v) for v in row]
            println(io, join(vals, ","))
        end
    end
end

# ============================================================================
# CONFIGURATION PERSISTENCE
# ============================================================================

"""
Save configuration snapshot to JSON.
"""
function save_config_snapshot(config::EmergentConfig, path::String)
    data = snapshot(config)
    open(path, "w") do io
        JSON3.write(io, data)
    end
end

"""
Load configuration from JSON snapshot.
"""
function load_config_snapshot(path::String)::EmergentConfig
    data = JSON3.read(read(path, String))
    config = EmergentConfig()
    apply_overrides!(config, Dict{String,Any}(String(k) => v for (k,v) in data))
    initialize!(config)
    return config
end

# ============================================================================
# RESULTS PERSISTENCE
# ============================================================================

"""
Results container for simulation output.
"""
struct SimulationResults
    config::EmergentConfig
    history::DataFrame
    final_agents::Vector{Any}
    run_id::String
    timestamp::DateTime
    metadata::Dict{String,Any}
end

"""
Save complete simulation results.
"""
function save_results(results::SimulationResults, output_dir::String)
    mkpath(output_dir)

    # Save config
    save_config_snapshot(results.config, joinpath(output_dir, "config_snapshot.json"))

    # Save history as Arrow (efficient) and CSV (compatible)
    save_dataframe_arrow(results.history, joinpath(output_dir, "history.arrow"))
    save_dataframe_csv(results.history, joinpath(output_dir, "history.csv"))

    # Save agents as JLD2
    save_agents_jld2(results.final_agents, joinpath(output_dir, "final_agents.jld2"))

    # Save metadata
    meta = merge(results.metadata, Dict(
        "run_id" => results.run_id,
        "timestamp" => string(results.timestamp),
        "n_agents" => results.config.N_AGENTS,
        "n_rounds" => results.config.N_ROUNDS,
    ))
    open(joinpath(output_dir, "metadata.json"), "w") do io
        JSON3.write(io, meta)
    end

    return output_dir
end

"""
Load simulation results from directory.
"""
function load_results(output_dir::String)::SimulationResults
    config = load_config_snapshot(joinpath(output_dir, "config_snapshot.json"))

    # Prefer Arrow if available, fallback to CSV
    history_path = joinpath(output_dir, "history.arrow")
    history = if isfile(history_path)
        load_dataframe_arrow(history_path)
    else
        # Fallback to CSV
        csv_path = joinpath(output_dir, "history.csv")
        if isfile(csv_path)
            # Simple CSV parser
            lines = readlines(csv_path)
            if isempty(lines)
                DataFrame()
            else
                headers = split(lines[1], ",")
                data = [split(line, ",") for line in lines[2:end]]
                df = DataFrame()
                for (i, h) in enumerate(headers)
                    col = [length(row) >= i ? row[i] : "" for row in data]
                    df[!, Symbol(strip(String(h)))] = col
                end
                df
            end
        else
            DataFrame()
        end
    end

    # Load agents
    agents_path = joinpath(output_dir, "final_agents.jld2")
    final_agents = isfile(agents_path) ? load_agents_jld2(agents_path) : Any[]

    # Load metadata
    meta_path = joinpath(output_dir, "metadata.json")
    metadata = if isfile(meta_path)
        Dict{String,Any}(String(k) => v for (k,v) in JSON3.read(read(meta_path, String)))
    else
        Dict{String,Any}()
    end

    run_id = get(metadata, "run_id", basename(output_dir))
    timestamp = if haskey(metadata, "timestamp")
        try
            DateTime(metadata["timestamp"])
        catch
            now()
        end
    else
        now()
    end

    return SimulationResults(config, history, final_agents, run_id, timestamp, metadata)
end

# ============================================================================
# PYTHON COMPATIBILITY STUBS
# Note: PyCall integration deferred to avoid conda dependency issues.
# For Python interop, use Arrow format which is cross-compatible.
# ============================================================================

"""
Check if PyCall is available (stub - returns false).
For actual Python interop, use Arrow/CSV formats.
"""
function pycall_available()::Bool
    return false
end

"""
Load pickle file (stub - not implemented).
Use load_dataframe_arrow() instead for cross-language compatibility.
"""
function load_pickle(path::String)
    @warn "Pickle loading not available in Julia port. Use Arrow format for Python compatibility."
    return nothing
end

"""
Save to pickle format (stub - not implemented).
Use save_dataframe_arrow() instead for cross-language compatibility.
"""
function save_pickle(obj, path::String)
    @warn "Pickle saving not available in Julia port. Use Arrow format for Python compatibility."
    return nothing
end

# ============================================================================
# BATCH RESULTS AGGREGATION
# ============================================================================

"""
Aggregate results from multiple simulation runs.
"""
function aggregate_run_results(run_dirs::Vector{String})::DataFrame
    all_results = DataFrame[]

    for dir in run_dirs
        if !isdir(dir)
            continue
        end

        try
            results = load_results(dir)
            if nrow(results.history) > 0
                # Add run identifier
                results.history[!, :run_id] .= results.run_id
                push!(all_results, results.history)
            end
        catch e
            @warn "Failed to load results from $dir: $e"
        end
    end

    if isempty(all_results)
        return DataFrame()
    end

    return vcat(all_results...)
end
