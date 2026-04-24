# Shared test helpers — do not include this file at top level of runtests.jl.
# Each test file that needs the helper should `include("test_helpers.jl")`.

using GlimpseABM: MarketConditions

"""
Build a `MarketConditions` for tests with sensible defaults. Overrides via
kwargs. Introduced in v3.0 to replace the ad-hoc `Dict{String,Any}(...)`
test constructions that were drifting away from the production schema.
"""
function test_market_conditions(; regime::String="normal",
                                  volatility::Float64=0.2,
                                  round::Int=0,
                                  uncertainty_state::Dict{String,Any}=Dict{String,Any}(),
                                 )
    return MarketConditions(
        regime,
        volatility,
        0.0,      # trend
        1.0,      # momentum
        1.0,      # regime_return_multiplier
        1.0,      # regime_failure_multiplier
        0,        # n_opportunities
        0.0,      # exploration_activity
        round,
        Dict{String,Float64}(),                 # tier_invest_share
        Dict{String,Float64}(),                 # sector_clearing_index
        1.0,                                    # aggregate_clearing_ratio
        Dict{String,Float64}(),                 # crowding_metrics
        Dict{String,Dict{String,Float64}}(),    # sector_demand_adjustments
        0.0,                                    # avg_competition
        uncertainty_state,
        Dict{String,Any}(),                     # extras
    )
end
