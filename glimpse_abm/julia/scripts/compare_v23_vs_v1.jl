# DEPRECATED v3.5.11 — single-population / pre-mixed-tier script.
# Not invoked by the current paper pipeline. Mixed-tier ports live at
# run_mixed_tier_{analysis_full,refutation_v3,mechanism,placebo,lambda_sweep}.jl.
# Contains pre-v3.5 calibration assumptions (e.g., BASE_OPERATIONAL_COST
# overrides where production reads sector cost; legacy decay_map dicts;
# generic success_count counters that conflate investment vs innovation).
# Kept for historical reference only; do not run for paper figures.

error("DEPRECATED v3.5.12 — single-population / pre-mixed-tier script.
" *
      "Replaced by run_mixed_tier_{analysis_full,refutation_v3,mechanism,placebo,lambda_sweep}.jl.
" *
      "This script contains pre-v3.5 calibration assumptions and stale knobs (e.g.,
" *
      "BASE_OPERATIONAL_COST overrides where production reads sector cost). It will
" *
      "silently produce wrong numbers if executed. Comment out this error() and rerun
" *
      "explicitly only if reading historical context — never for paper figures.")


# Compare v2.3 re-validation results (job 5130242) against v1 robustness
# experiment (job 5129552). Documents the magnitude and direction of the
# correctness-fix impact on the AI-survival paradox.

using Statistics, Printf, JSON3
const JSON = JSON3

# v1 baseline numbers (recovered from job 5129552 stdout, see
# results/robustness_5129552/ANALYSIS.md):
const V1_BASELINE = Dict(
    "none"     => (0.840, 2.58),
    "basic"    => (0.847, 2.61),
    "advanced" => (0.820, 2.57),
    "premium"  => (0.705, 2.57),
)

function load_v23_results(path::String)
    raw = JSON.read(read(path, String))
    by_tier = Dict{String,Vector{Dict}}()
    for row in raw
        tier = String(row[:tier])
        # Convert JSON3 object to plain Dict{String,Any}
        d = Dict{String,Any}(string(k) => v for (k, v) in pairs(row))
        push!(get!(by_tier, tier, Dict{String,Any}[]), d)
    end
    return by_tier
end

function summarize_tier(rows)
    surv = [Float64(r["survival_rate"]) for r in rows]
    cap = [Float64(r["mean_capital_alive"]) / 1e6 for r in rows]
    (mean(surv), std(surv), mean(cap), std(cap))
end

function main(path::String)
    by_tier = load_v23_results(path)

    println("="^80)
    println("v2.3 RE-VALIDATION vs v1 BASELINE — paradox status after correctness fixes")
    println("="^80)
    @printf "%-10s  %-25s  %-25s  %-15s\n" "tier" "v1 baseline (broken)" "v2.3 corrected" "Δ"
    println("-"^80)

    for tier in ["none", "basic", "advanced", "premium"]
        rows = get(by_tier, tier, Dict{String,Any}[])
        if isempty(rows)
            @printf "%-10s  no data\n" tier
            continue
        end
        v1_surv, v1_cap = V1_BASELINE[tier]
        sm, ss, cm, cs = summarize_tier(rows)
        @printf "%-10s  surv=%.3f  cap=\$%.2fM  surv=%.3f±%.3f  cap=\$%.2fM±%.2f  Δsurv=%+0.3f\n" tier v1_surv v1_cap sm ss cm cs (sm - v1_surv)
    end

    println("\n" * "="^80)
    println("PARADOX STATUS")
    println("="^80)

    none_rows = get(by_tier, "none", Dict{String,Any}[])
    prem_rows = get(by_tier, "premium", Dict{String,Any}[])
    none_surv = isempty(none_rows) ? NaN : mean(Float64(r["survival_rate"]) for r in none_rows)
    prem_surv = isempty(prem_rows) ? NaN : mean(Float64(r["survival_rate"]) for r in prem_rows)
    gap = none_surv - prem_surv

    v1_gap = V1_BASELINE["none"][1] - V1_BASELINE["premium"][1]

    @printf "  v1 broken model: premium − none = %+0.3f (paradox: premium WORSE)\n" -v1_gap
    @printf "  v2.3 corrected:  premium − none = %+0.3f\n" -gap
    if gap > 0.05
        println("  → Paradox SURVIVES corrections: premium still worse than none")
    elseif abs(gap) <= 0.05
        println("  → Paradox NEUTRALIZED: tiers within seed noise of each other")
    else
        println("  → Paradox FLIPPED: premium now BETTER than none")
    end
end

if length(ARGS) >= 1
    main(ARGS[1])
else
    println("Usage: julia compare_v23_vs_v1.jl <path/to/revalidation_v23_*.json>")
end
