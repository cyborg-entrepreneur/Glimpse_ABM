"""
Typed market-state payload consumed by agent decisions, `realized_return`,
and the uncertainty layer. Introduced in v3.0 to close out the
silent-zero-dataflow bug class that bit six times between v2.0 and v2.12
at dict-based module boundaries.

Every field is guaranteed present and typed; misuse fails at field-access
time rather than silently returning a default. Producer is
`market.jl:get_market_conditions`; consumers migrated to field access in
v3.0.

The `extras::Dict{String,Any}` escape hatch exists for transition-era
fields (e.g., `combo_hhi`) whose producer/consumer relationship was
unclear at migration time. Prefer adding a typed field above to reaching
for `extras`.
"""
struct MarketConditions
    # Regime + macro
    regime::String
    volatility::Float64
    trend::Float64
    momentum::Float64
    regime_return_multiplier::Float64
    regime_failure_multiplier::Float64

    # Market scale
    n_opportunities::Int
    exploration_activity::Float64
    round::Int

    # Crowding / clearing
    tier_invest_share::Dict{String,Float64}
    sector_clearing_index::Dict{String,Float64}
    aggregate_clearing_ratio::Float64
    crowding_metrics::Dict{String,Float64}
    sector_demand_adjustments::Dict{String,Dict{String,Float64}}
    avg_competition::Float64

    # Uncertainty hook — genuinely variant shape, stays flexible
    uncertainty_state::Dict{String,Any}

    # Escape hatch for one-off experimental fields
    extras::Dict{String,Any}
end

# ────────────────────────────────────────────────────────────────────────
# Legacy Dict-like read access. Keeps any straggling `get(mc, "regime", …)`
# call sites working during the v3.0 migration. All core consumers have
# been migrated to field access; this shim is defensive for scripts we
# may have missed.
#
# Silent-zero prevention: a read against a truly-unknown key returns the
# provided `default` (matching Dict semantics). A read against a field
# that EXISTS returns the field value (no default applied), so the v2.12
# `avg_competition` pattern — read with default `0.0` while producer was
# missing — is impossible by construction.
# ────────────────────────────────────────────────────────────────────────

function Base.get(mc::MarketConditions, key::AbstractString, default)
    sym = Symbol(key)
    if hasfield(MarketConditions, sym)
        return getfield(mc, sym)
    end
    return get(mc.extras, String(key), default)
end

function Base.haskey(mc::MarketConditions, key::AbstractString)
    sym = Symbol(key)
    return hasfield(MarketConditions, sym) || haskey(mc.extras, String(key))
end

function Base.getindex(mc::MarketConditions, key::AbstractString)
    sym = Symbol(key)
    if hasfield(MarketConditions, sym)
        return getfield(mc, sym)
    end
    return mc.extras[String(key)]
end
