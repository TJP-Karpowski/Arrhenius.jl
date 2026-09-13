"""Base type for chemistry models with algebraically reconstructed species."""
abstract type AbstractQSSModel end

"""
    BoivinQSSModel()

Published three-step H2/air reduction of Boivin et al. (2011). The transported
species, in the required mechanism order, are `H2, H, O2, H2O, HO2, N2`.
`O`, `OH`, and `H2O2` are quasi-steady-state species. The released rate
routine explicitly reconstructs `O` and `OH`; `H2O2` is eliminated from the
global rates and has no separately reconstructed concentration.

The implementation is a direct translation of the authors' released
`h2_3s_rates_PCI_nocorr` routine. This is the flame version without the optional
autoignition correction, which the paper does not activate in deflagrations.
"""
struct BoivinQSSModel <: AbstractQSSModel end

dynamic_species(::BoivinQSSModel) = ("H2", "H", "O2", "H2O", "HO2", "N2")
qss_species(::BoivinQSSModel) = ("O", "OH", "H2O2")

"""
    BoivinFourStepModel()

Published wide-range four-step H2/air reduction of Boivin, Sanchez, and
Williams (2013). `O` and `OH` are quasi-steady; `H`, `HO2`, and `H2O2`
remain dynamic. This is intentionally a separate model from the 2011
three-step flame reduction, which also eliminates `H2O2`.
"""
struct BoivinFourStepModel <: AbstractQSSModel end

dynamic_species(::BoivinFourStepModel) =
    ("H2", "H", "O2", "H2O", "HO2", "H2O2", "N2")
qss_species(::BoivinFourStepModel) = ("O", "OH")

"""
    BoivinHOnlyModel()

Experimental two-step member of the Boivin hierarchy. Only `H2, H, O2, H2O,
N2` remain dynamic; `O`, `OH`, `HO2`, and `H2O2` are algebraic. The model is
obtained from the published 2011 three-step expressions by imposing `q3 = 0`
and solving that scalar constraint for `HO2`. It is therefore a transparent
H-only QSS experiment, not a separately published mechanism.
"""
struct BoivinHOnlyModel <: AbstractQSSModel end

dynamic_species(::BoivinHOnlyModel) = ("H2", "H", "O2", "H2O", "N2")
qss_species(::BoivinHOnlyModel) = ("O", "OH", "HO2", "H2O2")

@inline _boivin_arrhenius(A, b, E_kJ_mol, logT, T) =
    A * exp(b * logT - 1000 * E_kJ_mol / 8.314 / T)

@inline function _boivin_troe(k_inf, k_low, collider, fcent, small)
    Pr = k_low * collider / k_inf
    logPr = log10(max(Pr, small))
    logFc = log10(max(fcent, small))
    n = 0.75 - 1.27 * logFc
    c = logPr - (0.4 + 0.67 * logFc)
    F = 10^(logFc / (1 + (c / (n - 0.14 * c))^2))
    return k_inf * F * Pr / (1 + Pr)
end

"""
    boivin_four_step_rates!(forward, reverse, algebraic, C, T)

Evaluate the published four-step global rates from Combustion and Flame 160
(2013), 76--82, doi:10.1016/j.combustflame.2012.09.014. Concentrations and
rates use Arrhenius' native `kmol/m^3` and `kmol/m^3/s` units. `C` is ordered
as `H2, H, O2, H2O, HO2, H2O2, N2`; `algebraic` receives `O, OH`.
"""
function boivin_four_step_rates!(forward::AbstractVector,
        reverse::AbstractVector, algebraic::AbstractVector,
        C::AbstractVector, T::Real)
    length(forward) == length(reverse) == 4 ||
        throw(DimensionMismatch("Boivin four-step needs four global-rate entries"))
    length(algebraic) == 2 ||
        throw(DimensionMismatch("Boivin four-step needs O/OH storage"))
    length(C) == 7 || throw(DimensionMismatch(
        "Boivin four-step expects H2/H/O2/H2O/HO2/H2O2/N2 concentrations"))
    T > 0 || throw(DomainError(T, "temperature must be positive"))

    ctot = 1.0e-3 * sum(C)
    H2 = 1.0e-3 * max(C[1], zero(C[1]))
    H = 1.0e-3 * max(C[2], zero(C[2]))
    O2 = 1.0e-3 * max(C[3], zero(C[3]))
    H2O = 1.0e-3 * max(C[4], zero(C[4]))
    HO2 = 1.0e-3 * max(C[5], zero(C[5]))
    H2O2 = 1.0e-3 * max(C[6], zero(C[6]))
    logT = log(T)
    small = oftype(T, 1.0e-100)

    M4 = ctot + 1.5H2 + 15H2O
    M8 = ctot + 1.5H2 + 11H2O
    M9 = M8
    M12 = ctot + H2 + 5H2O
    delta_H2 = 1.0e-12 * ctot

    k1f = _boivin_arrhenius(3.52e16, -0.7, 71.42, logT, T)
    k1r = _boivin_arrhenius(7.03556e13, -0.264291, 0.599604, logT, T)
    k2f = _boivin_arrhenius(5.06e4, 2.67, 26.32, logT, T)
    k2r = _boivin_arrhenius(3.0335e4, 2.63385, 20.2306, logT, T)
    k3f = _boivin_arrhenius(1.17e9, 1.3, 15.21, logT, T)
    k3r = _boivin_arrhenius(1.28718e10, 1.19625, 78.2545, logT, T)

    k4inf = _boivin_arrhenius(4.65e12, 0.44, 0.0, logT, T)
    k4low = _boivin_arrhenius(5.75e19, -1.4, 0.0, logT, T)
    fc4 = 0.5 * exp(-T / 1.0e30) + 0.5 * exp(-T / 1.0e-30)
    k4f = _boivin_troe(k4inf, k4low, M4, fc4, small)

    k5f = _boivin_arrhenius(7.08e13, 0.0, 1.234, logT, T)
    k6f = _boivin_arrhenius(1.66e13, 0.0, 3.443, logT, T)
    k6r = _boivin_arrhenius(2.69347e12, 0.362073, 231.867, logT, T)
    k7f = _boivin_arrhenius(2.89e13, 0.0, -2.08, logT, T)
    k8f = _boivin_arrhenius(4.0e22, -2.0, 0.0, logT, T) * M8
    k8r = _boivin_arrhenius(1.02857e23, -1.75351, 496.136, logT, T) * M8
    k9f = _boivin_arrhenius(1.3e18, -1.0, 0.0, logT, T) * M9
    k9r = _boivin_arrhenius(3.03853e17, -0.649765, 433.091, logT, T) * M9
    k10f = _boivin_arrhenius(3.02e12, 0.0, 5.8, logT, T)
    k11f = _boivin_arrhenius(1.62439e11, 0.606961, 100.139, logT, T)

    k12inf = _boivin_arrhenius(2.62328e19, -1.38836, 214.738, logT, T)
    k12low = _boivin_arrhenius(8.15342e23, -1.91836, 207.618, logT, T)
    fc12 = 0.265 * exp(-T / 94) + 0.735 * exp(-T / 1756) + exp(-5182 / T)
    k12f = _boivin_troe(k12inf, k12low, M12, fc12, small)

    # Equation (4) of the paper, written with the sign convention used by the
    # released 2011 routine. Unlike the three-step model, peroxide destruction
    # is evaluated from transported H2O2 rather than eliminating H2O2 by QSS.
    peroxide_destruction = 2k12f * H2O2
    A0 = H * H2 * k2f * (2k1f * O2 + k3r * H2O + 2k5f * HO2) +
         H2 * k2f * (peroxide_destruction + k8r * H2O)
    A1 = -H2 * k2f * (k8f * H + k7f * HO2 + k3f * H2) +
         k1r * H * (k3r * H2O + 2k5f * HO2) +
         k1r * (k8r * H2O + peroxide_destruction)
    A2 = k1r * (2k2r * H + k3f * (H2 + delta_H2) + k7f * HO2 + k8f * H)
    OH = max(zero(A2),
        (A1 + sqrt(max(A1^2 + 4A0 * A2, zero(A2)))) / (2A2))
    O = max(zero(OH), (k1f * H * O2 + k2r * OH * H) /
        (k1r * OH + k2f * (H2 + delta_H2)))
    algebraic[1] = 1.0e3 * O
    algebraic[2] = 1.0e3 * OH

    # Eqs. (2)--(3): global rates are assembled from elementary fluxes.
    r1f = k1f * H * O2; r1r = k1r * OH * O
    r4f = k4f * H * O2
    r5f = k5f * HO2 * H
    r6f = k6f * HO2 * H; r6r = k6r * O2 * H2
    r7f = k7f * HO2 * OH
    r8f = k8f * H * OH; r8r = k8r * H2O
    r9f = k9f * H^2; r9r = k9r * H2
    r10f = k10f * HO2^2
    r11f = k11f * HO2 * H2
    r12f = k12f * H2O2

    forward[1] = 1.0e3 * (r1f + r5f + r12f)
    reverse[1] = 1.0e3 * r1r
    forward[2] = 1.0e3 * (r4f + r8f + r9f)
    reverse[2] = 1.0e3 * (r8r + r9r + r10f + r11f)
    forward[3] = 1.0e3 * (r4f + r6r)
    reverse[3] = 1.0e3 * (r5f + r6f + r7f + 2r10f + r11f)
    forward[4] = 1.0e3 * (r10f + r11f)
    reverse[4] = 1.0e3 * r12f
    return forward, reverse, algebraic
end

"""
    boivin_qss_rates!(forward, reverse, algebraic, C, T)

Evaluate the published three global forward/reverse rates in `kmol/m^3/s`.
`C` contains the six dynamic-species concentrations in `kmol/m^3` and
`algebraic` receives `[C_O, C_OH]`, also in `kmol/m^3`.
"""
function boivin_qss_rates!(forward::AbstractVector, reverse::AbstractVector,
        algebraic::AbstractVector, C::AbstractVector, T::Real)
    length(forward) == length(reverse) == 3 ||
        throw(DimensionMismatch("Boivin QSS needs three global-rate entries"))
    length(algebraic) == 2 ||
        throw(DimensionMismatch("Boivin QSS needs O/OH storage"))
    length(C) == 6 || throw(DimensionMismatch(
        "Boivin QSS expects H2/H/O2/H2O/HO2/N2 concentrations"))
    T > 0 || throw(DomainError(T, "temperature must be positive"))

    # The released routine works internally in mol/cm^3. Concentration values
    # in those units are 1e-3 times Arrhenius' native kmol/m^3 values.
    ctot = 1.0e-3 * sum(C)
    H2  = 1.0e-3 * max(C[1], zero(C[1]))
    H   = 1.0e-3 * max(C[2], zero(C[2]))
    O2  = 1.0e-3 * max(C[3], zero(C[3]))
    H2O = 1.0e-3 * max(C[4], zero(C[4]))
    HO2 = 1.0e-3 * max(C[5], zero(C[5]))
    logT = log(T)
    small = oftype(T, 1.0e-100)

    M4 = ctot + 1.5H2 + 15H2O
    M8 = ctot + 1.5H2 + 11H2O
    M9 = M8
    M12 = ctot + H2 + 5H2O
    delta_H2 = 1.0e-12 * ctot

    k1f = _boivin_arrhenius(3.52e16, -0.7, 71.42, logT, T)
    k1r = _boivin_arrhenius(7.03556e13, -0.264291, 0.599604, logT, T)
    k2f = _boivin_arrhenius(5.06e4, 2.67, 26.32, logT, T)
    k2r = _boivin_arrhenius(3.0335e4, 2.63385, 20.2306, logT, T)
    k3f = _boivin_arrhenius(1.17e9, 1.3, 15.21, logT, T)
    k3r = _boivin_arrhenius(1.28718e10, 1.19625, 78.2545, logT, T)

    k4inf = _boivin_arrhenius(4.65e12, 0.44, 0.0, logT, T)
    k4low = _boivin_arrhenius(5.75e19, -1.4, 0.0, logT, T)
    fc4 = 0.5 * exp(-T / 1.0e30) + 0.5 * exp(-T / 1.0e-30)
    k4f = _boivin_troe(k4inf, k4low, M4, fc4, small)

    k5f = _boivin_arrhenius(7.08e13, 0.0, 1.234, logT, T)
    k5r = _boivin_arrhenius(1.37653e10, 0.761631, 152.748, logT, T)
    k6f = _boivin_arrhenius(1.66e13, 0.0, 3.443, logT, T)
    k6r = _boivin_arrhenius(2.69347e12, 0.362073, 231.867, logT, T)
    k7f = _boivin_arrhenius(2.89e13, 0.0, -2.08, logT, T)
    k8f = _boivin_arrhenius(4.0e22, -2.0, 0.0, logT, T) * M8
    k8r = _boivin_arrhenius(1.02857e23, -1.75351, 496.136, logT, T) * M8
    k9f = _boivin_arrhenius(1.3e18, -1.0, 0.0, logT, T) * M9
    k9r = _boivin_arrhenius(3.03853e17, -0.649765, 433.091, logT, T) * M9
    k10f = _boivin_arrhenius(3.02e12, 0.0, 5.8, logT, T)
    k11f = _boivin_arrhenius(1.62439e11, 0.606961, 100.139, logT, T)

    k12inf = _boivin_arrhenius(2.62328e19, -1.38836, 214.738, logT, T)
    k12low = _boivin_arrhenius(8.15342e23, -1.91836, 207.618, logT, T)
    fc12 = 0.265 * exp(-T / 94) + 0.735 * exp(-T / 1756) + exp(-5182 / T)
    k12f = _boivin_troe(k12inf, k12low, M12, fc12, small)

    A0 = H * H2 * k2f * (2k1f * O2 + k3r * H2O + 2k5f * HO2) +
         H2 * k2f * (2k10f * HO2^2 + 2k11f * HO2 * H2 + k8r * H2O)
    A1 = -H2 * k2f * (k8f * H + k7f * HO2 + k3f * H2) +
         k1r * H * (k3r * H2O + 2k5f * HO2) +
         k1r * (k8r * H2O + 2k10f * HO2^2 + 2k11f * HO2 * H2)
    A2 = k1r * (2k2r * H + k3f * (H2 + delta_H2) + k7f * HO2 + k8f * H)
    OH = max(zero(A2), (A1 + sqrt(max(A1^2 + 4A0 * A2, zero(A2)))) / (2A2))
    O = max(zero(OH), (k1f * H * O2 + k2r * OH * H) /
        (k1r * OH + k2f * (H2 + delta_H2)))
    # Convert reconstructed concentrations back to kmol/m^3.
    algebraic[1] = 1.0e3 * O
    algebraic[2] = 1.0e3 * OH

    # The source expressions are mol/cm^3/s internally. The factor 1e3 gives
    # the numerically equivalent kmol/m^3/s values used by Arrhenius.
    forward[1] = 1.0e3 * (k1f * H * O2 + k5f * HO2 * H +
        k10f * HO2^2 + k11f * HO2 * H2)
    reverse[1] = 1.0e3 * k1r * OH * O
    forward[2] = 1.0e3 * (k8f * H * OH + k9f * H^2 + k4f * H * O2)
    reverse[2] = 1.0e3 * (k10f * HO2^2 + k11f * HO2 * H2 +
        k8r * H2O + k9r * H2)
    forward[3] = 1.0e3 * (k4f * H * O2 + k6r * O2 * H2)
    reverse[3] = 1.0e3 * (k6f * HO2 * H + k7f * HO2 * OH +
        k5f * HO2 * H + 2k10f * HO2^2 + k11f * HO2 * H2)
    return forward, reverse, algebraic
end

@inline _boivin_primal(x) = x
@inline _boivin_primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)

function _boivin_qss_peroxide(C, T, HO2)
    ctot = 1.0e-3 * (sum(C) + HO2)
    H2 = 1.0e-3 * max(C[1], zero(C[1]))
    H2O = 1.0e-3 * max(C[4], zero(C[4]))
    ho2 = 1.0e-3 * max(HO2, zero(HO2))
    logT = log(T)
    small = oftype(T, 1.0e-100)
    M12 = ctot + H2 + 5H2O
    k10f = _boivin_arrhenius(3.02e12, 0.0, 5.8, logT, T)
    k11f = _boivin_arrhenius(1.62439e11, 0.606961, 100.139, logT, T)
    k12inf = _boivin_arrhenius(2.62328e19, -1.38836, 214.738, logT, T)
    k12low = _boivin_arrhenius(8.15342e23, -1.91836, 207.618, logT, T)
    fc12 = 0.265 * exp(-T / 94) + 0.735 * exp(-T / 1756) + exp(-5182 / T)
    k12f = _boivin_troe(k12inf, k12low, M12, fc12, small)
    return 1.0e3 * (k10f * ho2^2 + k11f * ho2 * H2) / k12f
end

"""
    boivin_h_only_rates!(forward, reverse, algebraic, extended_C, C, T)

Evaluate the experimental H-only two-step reduction. `C` is ordered as
`H2,H,O2,H2O,N2`; `algebraic` receives `O,OH,HO2,H2O2`. The caller-provided
`extended_C` is scratch storage for the parent three-step concentration vector.
"""
function boivin_h_only_rates!(forward::AbstractVector, reverse::AbstractVector,
        algebraic::AbstractVector, extended_C::AbstractVector,
        C::AbstractVector, T::Real)
    length(forward) == length(reverse) == 3 || throw(DimensionMismatch(
        "Boivin H-only evaluator needs three parent-rate scratch entries"))
    length(algebraic) == 4 || throw(DimensionMismatch(
        "Boivin H-only evaluator needs O/OH/HO2/H2O2 storage"))
    length(extended_C) == 6 || throw(DimensionMismatch(
        "Boivin H-only evaluator needs six concentration scratch entries"))
    length(C) == 5 || throw(DimensionMismatch(
        "Boivin H-only expects H2/H/O2/H2O/N2 concentrations"))

    extended_C[1] = C[1]
    extended_C[2] = C[2]
    extended_C[3] = C[3]
    extended_C[4] = C[4]
    extended_C[6] = C[5]
    parent_algebraic = @view algebraic[1:2]

    function q3_at(ho2)
        extended_C[5] = ho2
        boivin_qss_rates!(forward, reverse, parent_algebraic, extended_C, T)
        return forward[3] - reverse[3]
    end

    lo = zero(eltype(extended_C))
    qlo = q3_at(lo)
    concentration_scale = max(_boivin_primal(sum(abs, C)), 1.0e-30)
    hi = oftype(lo, concentration_scale * 1.0e-12)
    qhi = q3_at(hi)
    for _ in 1:80
        _boivin_primal(qhi) <= 0 && break
        hi *= 2
        qhi = q3_at(hi)
    end
    _boivin_primal(qhi) <= 0 || throw(ErrorException(
        "failed to bracket the Boivin H-only HO2 QSS root"))

    if _boivin_primal(qlo) <= 0
        hi = lo
    else
        # A fixed iteration count makes the result reproducible and keeps this
        # algebraic solve compatible with ForwardDiff's dual-number evaluation.
        for _ in 1:56
            mid = (lo + hi) / 2
            qmid = q3_at(mid)
            if _boivin_primal(qmid) > 0
                lo = mid
            else
                hi = mid
            end
        end
    end
    HO2 = (lo + hi) / 2
    q3_at(HO2)
    algebraic[3] = HO2
    algebraic[4] = _boivin_qss_peroxide(C, T, HO2)
    return forward, reverse, algebraic
end

mutable struct BoivinQSSReactor{G, R, V} <: Function
    gas::G
    thermo::R
    forward::V
    reverse::V
    algebraic::V
    wdot::V
end

function BoivinQSSReactor(gas)
    Tuple(gas.species_names) == dynamic_species(BoivinQSSModel()) ||
        throw(ArgumentError("BoivinQSSReactor requires the `boivin-qss` phase " *
            "with species H2, H, O2, H2O, HO2, N2 in that order"))
    return BoivinQSSReactor(gas, isobaricReactor(gas), zeros(3), zeros(3),
        zeros(2), zeros(gas.n_species))
end

function BoivinQSSReactor(gas, ::Type{T}) where {T}
    Tuple(gas.species_names) == dynamic_species(BoivinQSSModel()) ||
        throw(ArgumentError("BoivinQSSReactor requires the `boivin-qss` phase " *
            "with species H2, H, O2, H2O, HO2, N2 in that order"))
    return BoivinQSSReactor(gas, isobaricReactor(gas, T), zeros(T, 3),
        zeros(T, 3), zeros(T, 2), zeros(T, gas.n_species))
end

function qss_concentrations(state::BoivinQSSReactor)
    return (; O = state.algebraic[1], OH = state.algebraic[2])
end

function (state::BoivinQSSReactor)(du, u, p, t)
    ns = state.gas.n_species
    Y = @view u[1:ns]
    updateReactor!(state.thermo, Y, u[end], p[1])
    boivin_qss_rates!(state.forward, state.reverse, state.algebraic,
        state.thermo.C, u[end])
    q1 = state.forward[1] - state.reverse[1]
    q2 = state.forward[2] - state.reverse[2]
    q3 = state.forward[3] - state.reverse[3]
    w = state.wdot
    w[1] = -3q1 + q2 - q3
    w[2] = 2q1 - 2q2 + q3
    w[3] = -q1 - q3
    w[4] = 2q1
    w[5] = q3
    w[6] = zero(q1)

    du[end] = zero(eltype(du))
    @inbounds for i in 1:ns
        du[end] -= state.thermo.h_RT[i] * w[i]
        du[i] = w[i] * state.gas.MW[i] / state.thermo.ρ_mass
    end
    du[end] *= R * u[end] / (state.thermo.cp_mass * state.thermo.ρ_mass)
    return nothing
end

mutable struct BoivinHOnlyReactor{G, R, V} <: Function
    gas::G
    thermo::R
    forward::V
    reverse::V
    algebraic::V
    extended_C::V
    wdot::V
end

function BoivinHOnlyReactor(gas)
    Tuple(gas.species_names) == dynamic_species(BoivinHOnlyModel()) ||
        throw(ArgumentError("BoivinHOnlyReactor requires the `boivin-h-only` " *
            "phase with species H2, H, O2, H2O, N2 in that order"))
    return BoivinHOnlyReactor(gas, isobaricReactor(gas), zeros(3), zeros(3),
        zeros(4), zeros(6), zeros(gas.n_species))
end

function BoivinHOnlyReactor(gas, ::Type{T}) where {T}
    Tuple(gas.species_names) == dynamic_species(BoivinHOnlyModel()) ||
        throw(ArgumentError("BoivinHOnlyReactor requires the `boivin-h-only` " *
            "phase with species H2, H, O2, H2O, N2 in that order"))
    return BoivinHOnlyReactor(gas, isobaricReactor(gas, T), zeros(T, 3),
        zeros(T, 3), zeros(T, 4), zeros(T, 6), zeros(T, gas.n_species))
end

qss_concentrations(state::BoivinHOnlyReactor) =
    (; O = state.algebraic[1], OH = state.algebraic[2],
       HO2 = state.algebraic[3], H2O2 = state.algebraic[4])

function (state::BoivinHOnlyReactor)(du, u, p, t)
    ns = state.gas.n_species
    Y = @view u[1:ns]
    updateReactor!(state.thermo, Y, u[end], p[1])
    boivin_h_only_rates!(state.forward, state.reverse, state.algebraic,
        state.extended_C, state.thermo.C, u[end])
    q1 = state.forward[1] - state.reverse[1]
    q2 = state.forward[2] - state.reverse[2]
    w = state.wdot
    w[1] = -3q1 + q2
    w[2] = 2q1 - 2q2
    w[3] = -q1
    w[4] = 2q1
    w[5] = zero(q1)

    du[end] = zero(eltype(du))
    @inbounds for i in 1:ns
        du[end] -= state.thermo.h_RT[i] * w[i]
        du[i] = w[i] * state.gas.MW[i] / state.thermo.ρ_mass
    end
    du[end] *= R * u[end] / (state.thermo.cp_mass * state.thermo.ρ_mass)
    return nothing
end

mutable struct BoivinFourStepReactor{G, R, V} <: Function
    gas::G
    thermo::R
    forward::V
    reverse::V
    algebraic::V
    wdot::V
end

function BoivinFourStepReactor(gas)
    Tuple(gas.species_names) == dynamic_species(BoivinFourStepModel()) ||
        throw(ArgumentError("BoivinFourStepReactor requires the `boivin-four-step` " *
            "phase with species H2, H, O2, H2O, HO2, H2O2, N2 in that order"))
    return BoivinFourStepReactor(gas, isobaricReactor(gas), zeros(4),
        zeros(4), zeros(2), zeros(gas.n_species))
end

function BoivinFourStepReactor(gas, ::Type{T}) where {T}
    Tuple(gas.species_names) == dynamic_species(BoivinFourStepModel()) ||
        throw(ArgumentError("BoivinFourStepReactor requires the `boivin-four-step` " *
            "phase with species H2, H, O2, H2O, HO2, H2O2, N2 in that order"))
    return BoivinFourStepReactor(gas, isobaricReactor(gas, T), zeros(T, 4),
        zeros(T, 4), zeros(T, 2), zeros(T, gas.n_species))
end

qss_concentrations(state::BoivinFourStepReactor) =
    (; O = state.algebraic[1], OH = state.algebraic[2])

function (state::BoivinFourStepReactor)(du, u, p, t)
    ns = state.gas.n_species
    Y = @view u[1:ns]
    updateReactor!(state.thermo, Y, u[end], p[1])
    boivin_four_step_rates!(state.forward, state.reverse, state.algebraic,
        state.thermo.C, u[end])
    q1 = state.forward[1] - state.reverse[1]
    q2 = state.forward[2] - state.reverse[2]
    q3 = state.forward[3] - state.reverse[3]
    q4 = state.forward[4] - state.reverse[4]
    w = state.wdot
    w[1] = -3q1 + q2 - q3 - q4
    w[2] = 2q1 - 2q2 + q3
    w[3] = -q1 - q3 - q4
    w[4] = 2q1
    w[5] = q3
    w[6] = q4
    w[7] = zero(q1)

    du[end] = zero(eltype(du))
    @inbounds for i in 1:ns
        du[end] -= state.thermo.h_RT[i] * w[i]
        du[i] = w[i] * state.gas.MW[i] / state.thermo.ρ_mass
    end
    du[end] *= R * u[end] / (state.thermo.cp_mass * state.thermo.ρ_mass)
    return nothing
end

export AbstractQSSModel, BoivinQSSModel, BoivinFourStepModel, BoivinHOnlyModel
export BoivinQSSReactor, BoivinFourStepReactor, BoivinHOnlyReactor
export dynamic_species, qss_species, qss_concentrations
export boivin_qss_rates!, boivin_four_step_rates!, boivin_h_only_rates!
