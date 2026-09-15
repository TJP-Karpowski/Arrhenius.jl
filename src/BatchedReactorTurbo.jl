"""
    TurboBatchedIdealGasReactor{K}

Cell-major (reactor-innermost) batched RHS workspace. Species/reaction arrays
are shaped `(n_reactors, n_species)` / `(n_reactors, n_reactions)` -- the
*opposite* orientation from [`BatchedIdealGasReactor`](@ref), which is
species-major `(n_species, n_reactors)`. The reactor dimension is the
fastest-varying (contiguous) one here specifically so the per-reactor Arrhenius
rate loop is a plain dense loop that the `ArrheniusLoopVectorizationExt`
extension can vectorize across reactors with `@turbo` (including the `exp`
calls, which plain `@simd`/`@inbounds` does not vectorize).

Requires `using LoopVectorization` to be loaded alongside `Arrhenius` --
without it, [`turbo_batched_isobaric_rhs!`](@ref) raises an error explaining
that. This mirrors how the CPU batched path stays usable without
`KernelAbstractions`/`Metal` loaded.

Scope: ordinary Arrhenius, third-body, Lindemann/Troe falloff, and reversible
reactions. Pressure-dependent (PLOG) reactions are not supported yet --
construction raises an error for mechanisms that have any.
"""
struct TurboBatchedIdealGasReactor{K}
    kinetics::K
    MW::Vector{Float64}
    nasa_low::Matrix{Float64}
    nasa_high::Matrix{Float64}
    n_species::Int
    n_reactions::Int
end

"""
    TurboBatchedIdealGasReactor(gas)

Build a cell-major batched reactor for `gas`. Errors if `gas`'s mechanism
contains pressure-dependent (PLOG) reactions -- see the type docstring.
"""
function TurboBatchedIdealGasReactor(gas)
    kinetics = packed_kinetics_for_batch(gas.kinetics)
    isempty(kinetics.index_pressure_dependent) ||
        throw(ArgumentError("TurboBatchedIdealGasReactor does not support pressure-dependent (PLOG) reactions yet"))
    return TurboBatchedIdealGasReactor(kinetics, gas.MW, gas.thermo.nasa_low, gas.thermo.nasa_high,
        gas.n_species, gas.n_reactions)
end

"""
    TurboBatchedBuffers(n_reactors, n_species, n_reactions)

Reusable scratch buffers for [`turbo_batched_isobaric_rhs!`](@ref), sized for
up to `n_reactors` cells. Allocate once and reuse across calls to keep the RHS
evaluation non-allocating.
"""
struct TurboBatchedBuffers
    X::Matrix{Float64}
    C::Matrix{Float64}
    h_RT::Matrix{Float64}
    s0_R::Matrix{Float64}
    g_RT::Matrix{Float64}
    cp_R::Matrix{Float64}
    cp_mass::Vector{Float64}
    mean_MW::Vector{Float64}
    rho_mass::Vector{Float64}
    kf::Matrix{Float64}
    kr::Matrix{Float64}
    wdot::Matrix{Float64}
    logT::Vector{Float64}
    rec_RcalT::Vector{Float64}
    Mbuf::Vector{Float64}
    delta_g_buf::Vector{Float64}
    logPbuf::Vector{Float64}
end

function TurboBatchedBuffers(n_reactors::Integer, n_species::Integer, n_reactions::Integer)
    z(m, n) = zeros(m, n)
    v(m) = zeros(m)
    return TurboBatchedBuffers(
        z(n_reactors, n_species), z(n_reactors, n_species), z(n_reactors, n_species),
        z(n_reactors, n_species), z(n_reactors, n_species), z(n_reactors, n_species),
        v(n_reactors), v(n_reactors), v(n_reactors),
        z(n_reactors, n_reactions), z(n_reactors, n_reactions), z(n_reactors, n_species),
        v(n_reactors), v(n_reactors), v(n_reactors), v(n_reactors), v(n_reactors),
    )
end

"""
    turbo_batched_isobaric_rhs!(dYdt, dTdt, reactor, Y, T, p, buffers)

Evaluate the constant-pressure ideal-gas reactor RHS for `size(Y, 1)`
independent reactors, cell-major layout: `Y`/`dYdt` are `(n_reactors,
n_species)`, `T`/`p`/`dTdt` are `(n_reactors,)`. Requires `using
LoopVectorization`; provided by `ArrheniusLoopVectorizationExt`.
"""
function turbo_batched_isobaric_rhs!(dYdt, dTdt, reactor::TurboBatchedIdealGasReactor, Y, T, p, buffers)
    error("LoopVectorization is required for turbo_batched_isobaric_rhs!. Load LoopVectorization alongside Arrhenius.")
end

export TurboBatchedIdealGasReactor, TurboBatchedBuffers, turbo_batched_isobaric_rhs!
