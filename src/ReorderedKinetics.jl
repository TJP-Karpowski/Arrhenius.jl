
# basic_kernels 

const MIN_RATE_ARGUMENT = 1.0e-300

minimum_rate_argument(x) = oftype(x, MIN_RATE_ARGUMENT)
minimum_rate_argument(x::Float32) = floatmin(Float32)
minimum_rate_argument(x::Float16) = floatmin(Float16)
positive_rate_argument(x) = ifelse(x > zero(x), x, minimum_rate_argument(x))
safe_log(x) = log(positive_rate_argument(x))
safe_log10(x) = log10(positive_rate_argument(x))

function arrheniusLaw!(kf::AbstractVector, arr_coeffs::AbstractArray,T::Real)
    logT = safe_log(T)
    rec_RcalT = (4184.0 / Arrhenius.R / T)
    forwardRate!(kf,arr_coeffs,logT,rec_RcalT)
    return nothing
end
function arrheniusLaw!(kf::AbstractVector, arr_coeffs::AbstractArray,logT::Real,rec_RcalT::Real)
    for i in eachindex(kf) 
        @inbounds  kf[i] = arr_coeffs[i, 1]  * exp(arr_coeffs[i, 2] * logT - arr_coeffs[i, 3] * rec_RcalT)
    end
    return nothing
end

function pressure_dependent_rate(
    pressures::AbstractVector,
    arrhenius_coeffs::AbstractMatrix,
    first::Int,
    last::Int,
    pressure_atm::Real,
    logT::Real,
    rec_RcalT::Real,
)
    first == last && return arrhenius_coeffs[first, 1] *
                            exp(arrhenius_coeffs[first, 2] * logT - arrhenius_coeffs[first, 3] * rec_RcalT)

    p = positive_rate_argument(pressure_atm)
    if p <= pressures[first]
        i = first
        return arrhenius_coeffs[i, 1] * exp(arrhenius_coeffs[i, 2] * logT - arrhenius_coeffs[i, 3] * rec_RcalT)
    elseif p >= pressures[last]
        i = last
        return arrhenius_coeffs[i, 1] * exp(arrhenius_coeffs[i, 2] * logT - arrhenius_coeffs[i, 3] * rec_RcalT)
    end

    logp = safe_log(p)
    for i in first:last-1
        p_low = pressures[i]
        p_high = pressures[i + 1]
        if p >= p_low && p <= p_high
            logp_low = safe_log(p_low)
            logp_high = safe_log(p_high)
            k_low = arrhenius_coeffs[i, 1] *
                    exp(arrhenius_coeffs[i, 2] * logT - arrhenius_coeffs[i, 3] * rec_RcalT)
            k_high = arrhenius_coeffs[i + 1, 1] *
                     exp(arrhenius_coeffs[i + 1, 2] * logT - arrhenius_coeffs[i + 1, 3] * rec_RcalT)
            weight = (logp - logp_low) / (logp_high - logp_low)
            return exp((1 - weight) * safe_log(k_low) + weight * safe_log(k_high))
        end
    end

    i = last
    return arrhenius_coeffs[i, 1] * exp(arrhenius_coeffs[i, 2] * logT - arrhenius_coeffs[i, 3] * rec_RcalT)
end

function arrhenius_rate_temperature_derivative(coeffs::AbstractMatrix, row::Int, T, k)
    return k * (coeffs[row, 2] / T + coeffs[row, 3] * 4184.0 / Arrhenius.R / T^2)
end

function linear_burke_efficiency(reactions, species::Int, pdep_index::Int, T, logT, rec_RcalT)
    coeffs = reactions.linear_burke_efficiencies
    A = coeffs[species, pdep_index, 1]
    b = coeffs[species, pdep_index, 2]
    Ea = coeffs[species, pdep_index, 3]
    return A * exp(b * logT - Ea * rec_RcalT)
end

function linear_burke_efficiency_temperature_derivative(eff, reactions, species::Int, pdep_index::Int, T)
    coeffs = reactions.linear_burke_efficiencies
    return eff * (coeffs[species, pdep_index, 2] / T + coeffs[species, pdep_index, 3] * 4184.0 / Arrhenius.R / T^2)
end

function effective_pressure_atm(reactions, C, T, logT, rec_RcalT, pdep_index::Int)
    pressure_scale = R * T / one_atm
    if !reactions.pressure_dependent_is_linear_burke[pdep_index]
        return pressure_scale * sum(C)
    end

    weighted_concentration = zero(eltype(C))
    for species in eachindex(C)
        weighted_concentration += C[species] * linear_burke_efficiency(reactions, species, pdep_index, T, logT, rec_RcalT)
    end
    return pressure_scale * weighted_concentration
end

function effective_pressure_derivative_C(reactions, species::Int, T, logT, rec_RcalT, pdep_index::Int)
    pressure_scale = R * T / one_atm
    reactions.pressure_dependent_is_linear_burke[pdep_index] || return pressure_scale
    return pressure_scale * linear_burke_efficiency(reactions, species, pdep_index, T, logT, rec_RcalT)
end

function effective_pressure_derivative_T(reactions, C, T, logT, rec_RcalT, pdep_index::Int)
    concentration_sum = zero(eltype(C))
    derivative_sum = zero(eltype(C))
    if reactions.pressure_dependent_is_linear_burke[pdep_index]
        for species in eachindex(C)
            eff = linear_burke_efficiency(reactions, species, pdep_index, T, logT, rec_RcalT)
            concentration_sum += C[species] * eff
            derivative_sum += C[species] * linear_burke_efficiency_temperature_derivative(eff, reactions, species, pdep_index, T)
        end
    else
        concentration_sum = sum(C)
    end
    return R / one_atm * concentration_sum + R * T / one_atm * derivative_sum
end

function pressure_dependent_rate_derivatives(
    pressures::AbstractVector,
    arrhenius_coeffs::AbstractMatrix,
    first::Int,
    last::Int,
    pressure_atm::Real,
    T::Real,
    logT::Real,
    rec_RcalT::Real,
)
    k = pressure_dependent_rate(pressures, arrhenius_coeffs, first, last, pressure_atm, logT, rec_RcalT)
    first == last && return k, zero(k), arrhenius_rate_temperature_derivative(arrhenius_coeffs, first, T, k)

    p = positive_rate_argument(pressure_atm)
    if p <= pressures[first]
        return k, zero(k), arrhenius_rate_temperature_derivative(arrhenius_coeffs, first, T, k)
    elseif p >= pressures[last]
        return k, zero(k), arrhenius_rate_temperature_derivative(arrhenius_coeffs, last, T, k)
    end

    for i in first:last-1
        p_low = pressures[i]
        p_high = pressures[i + 1]
        if p >= p_low && p <= p_high
            k_low = arrhenius_coeffs[i, 1] * exp(arrhenius_coeffs[i, 2] * logT - arrhenius_coeffs[i, 3] * rec_RcalT)
            k_high = arrhenius_coeffs[i + 1, 1] * exp(arrhenius_coeffs[i + 1, 2] * logT - arrhenius_coeffs[i + 1, 3] * rec_RcalT)
            logp_low = safe_log(p_low)
            logp_high = safe_log(p_high)
            weight = (safe_log(p) - logp_low) / (logp_high - logp_low)
            slope = (safe_log(k_high) - safe_log(k_low)) / (logp_high - logp_low)
            dk_dP = k * slope / p
            dlogk_dT =
                (1 - weight) * (arrhenius_coeffs[i, 2] / T + arrhenius_coeffs[i, 3] * 4184.0 / Arrhenius.R / T^2) +
                weight * (arrhenius_coeffs[i + 1, 2] / T + arrhenius_coeffs[i + 1, 3] * 4184.0 / Arrhenius.R / T^2)
            return k, dk_dP, k * dlogk_dT
        end
    end
    return k, zero(k), zero(k)
end

function pressure_dependent_rates!(kf, reactions::ReorderedKinetics, C, T, logT, rec_RcalT)
    isempty(reactions.index_pressure_dependent) && return nothing
    for (i, reaction_index) in enumerate(reactions.index_pressure_dependent)
        first = reactions.pressure_dependent_ptr[i]
        last = reactions.pressure_dependent_ptr[i + 1] - 1
        pressure_atm = effective_pressure_atm(reactions, C, T, logT, rec_RcalT, i)
        @inbounds kf[reaction_index] = pressure_dependent_rate(
            reactions.pressure_dependent_pressures,
            reactions.pressure_dependent_arrhenius,
            first,
            last,
            pressure_atm,
            logT,
            rec_RcalT,
        )
    end
    return nothing
end

"""
    k_Pi_C_rectantOrder!(kf, C, reactant_orders)

Multiply each reaction's forward-rate entry in `kf` by
`∏ C[species]^order` over its reactants. `order` is generally an integer
(matching stoichiometry) but need not be -- fractional reaction orders are
a standard way to fit reduced mechanisms (e.g. BFER-type 2-step schemes)
to a wider operating range. `C[i]` raised to a *fractional* power is only
real-valued for `C[i] >= 0`; ordinary ODE/PDE truncation error routinely
produces small negative excursions in a species concentration that's
supposed to be ~0, and `x^y` for negative `x`/non-integer `y` throws
`DomainError` in Julia (confirmed directly: this crashed a BFER 0D
homogeneous-reactor smoke test on step 1, `C[CH4] ≈ -1.24e-5` raised to
its `0.5` reactant order). `positive_rate_argument` (already used
elsewhere in this file for the same reason, e.g. `safe_log`) floors that
to a tiny positive value -- applied only when `order` is non-integer, so
integer-order reactions (the overwhelming majority, where `x^order` for
negative `x` is already well-defined and real) keep their exact prior
behavior unchanged.
"""
function k_Pi_C_rectantOrder!(kf,C,reactant_orders)
    @inbounds for i in eachindex(kf)
        for j in reactant_orders.colptr[i]:reactant_orders.colptr[i+1]-1
            order = reactant_orders.nzval[j]
            Ci = C[reactant_orders.rowval[j]]
            # Avoid generic floating-point pow for common integral orders.
            # Fractional orders retain the existing positive-argument floor.
            power = order == 1 ? Ci : order == 2 ? Ci * Ci :
                order == 0 ? one(Ci) : order == 3 ? Ci * Ci * Ci :
                (isinteger(order) ? Ci : positive_rate_argument(Ci))^order
            kf[i] *= power
        end
    end
end

function kr!(kr, kf,g_RT,vk,vk_sum,T)
    mul!(kr,vk,g_RT) #kr proxies as Δg_RT as intermediate storage'
    logP = safe_log(one_atm / R / T)
    @. kr = kf / exp(-kr + logP * vk_sum)
    return nothing
end

function vk_kf!(wdot::AbstractVector,vk::AbstractArray,kf::AbstractVector) # TODO use likwid to compare the two methods
    # transposed vk - workflow 
    for isp in eachindex(wdot) 
        for j in vk.colptr[isp]:vk.colptr[isp+1]-1
            wdot[isp] += vk.nzval[j] * kf[vk.rowval[j]]
        end
    end 
    # for ir in eachindex(kf)
    #     for j in vk.colptr[ir]:vk.colptr[ir+1]-1
    #         wdot[vk.rowval[j]] += vk.nzval[j] * kf[ir]
    #     end
    # end
    #mul!(wdot,vk',kf)
    return nothing
end


function efficency!( kf,effCoeff,C)
    for ir in eachindex(kf)
        dot_C_eff = 0.0
        @inbounds for j in effCoeff.colptr[ir]:effCoeff.colptr[ir+1]-1
            @inbounds dot_C_eff += effCoeff.nzval[j] * C[effCoeff.rowval[j]]
        end
        @inbounds kf[ir] *= dot_C_eff
    end
    return nothing
end

function troe!(kf, logPr,troeCoeffs,T)
    for i in eachindex(kf)
      @inbounds F_cent =
                  (1 - troeCoeffs[i, 1]) * exp(-T / troeCoeffs[i, 4]) +
                  troeCoeffs[i, 1] * exp(-T / troeCoeffs[i, 2]) +
                  exp(-troeCoeffs[i, 3] / T)
      lF_cent = safe_log10(F_cent)
      _C = -0.4 - 0.67 * lF_cent
      N = 0.75 - 1.27 * lF_cent
      f1 = (logPr[i] + _C) / (N - 0.14 * (logPr[i] + _C))
      @inbounds kf[i] *= exp(log(10.0) * lF_cent / (1 + f1^2))
    end
  end


  function wdot_func!(wdot::AbstractVector,kf::AbstractVector,kr::AbstractVector,C::AbstractVector,g_RT::AbstractVector,reactions::ReorderedKinetics ,T::Real, reaction_multipliers=nothing)
    wdot.=0.0
    logT = safe_log(T)
    rec_RcalT = (4184.0 / Arrhenius.R / T)
    
    #forwardRate from Arrhe
    arrheniusLaw!(kf, reactions.arrhenius_coeffs,logT, rec_RcalT)
    pressure_dependent_rates!(kf, reactions, C, T, logT, rec_RcalT)

    #threeBody
    tB_range = reactions.index_three_body_start:reactions.index_fallOff_start-1
    efficency!(@view(kf[tB_range]),reactions.efficiency_coeffs_tB,C)

    #fallOff adaptation
    fO_start=  reactions.index_fallOff_start#:end#reactions.index_fallOff_start-1
    arrheniusLaw!(@view(kr[fO_start:end]),reactions.Arrhenius_0,logT, rec_RcalT) #kr proxies as k0
    efficency!(@view(kr[fO_start:end]),reactions.efficiency_coeffs_fO,C) #kr proxies as k0 * dot(eff,C)
    @inbounds for i in fO_start:length(kf)
        kr[i] = positive_rate_argument(kr[i] / kf[i]) #kr proxies as Pr
        kf[i] *= kr[i] / (1 + kr[i])
    end

    #troe
    troe_start = reactions.index_troe_start
    @inbounds for i in troe_start:length(kr)
        kr[i] = safe_log10(kr[i]) #kr proxies as logPr
    end
    troe!(@view(kf[troe_start:end]),@view(kr[troe_start:end]), reactions.troe,T)


    # Representation-aware effective kinetics. Apply one positive multiplier
    # to each complete reaction pair after all native pressure/falloff/Troe
    # corrections, but before reverse rates are derived from equilibrium.
    # Consequently kf/kr and the native equilibrium constant are unchanged.
    if reaction_multipliers !== nothing
        length(reaction_multipliers) == length(kf) || throw(DimensionMismatch(
            "reaction multiplier length $(length(reaction_multipliers)) does not match $(length(kf)) reactions"))
        @inbounds for ir in eachindex(kf)
            multiplier = reaction_multipliers[ir]
            multiplier >= zero(multiplier) || throw(DomainError(multiplier,
                "reaction multipliers must be non-negative"))
            kf[ir] *= multiplier
        end
    end


    kr!(@view(kr[reactions.index_rev_start:end]),@view(kf[reactions.index_rev_start:end]),g_RT,reactions.vk_rev,reactions.vk_sum,T)
    k_Pi_C_rectantOrder!(kf,C,reactions.reactant_orders)
    k_Pi_C_rectantOrder!(@view(kr[reactions.index_rev_start:end]),C,reactions.product_stoich_coeffs)
    kf .-=kr
    vk_kf!(wdot,reactions.vk,kf)
    return nothing
end

"""
    promote_kinetics(reactions::ReorderedKinetics, ::Type{T}) where {T}

Return a new `ReorderedKinetics` with every floating-point-valued array
field (`arrhenius_coeffs`, stoichiometry/order matrices, efficiency
coefficients, etc.) promoted to element type `T`, values preserved.
Integer/`Bool` bookkeeping fields (reaction-count offsets, falloff/troe
flags) are left untouched -- they are not physical quantities and were
never meant to carry derivative information.

This is what lets a caller make a specific reaction's own rate
coefficients differentiable (e.g. w.r.t. a fitted closure's own
parameters, for online/gradient-based mechanism calibration): promote to
`T = eltype(θ)` (a `ForwardDiff.Dual` type when differentiating), then
overwrite the targeted `arrhenius_coeffs[reaction_index, :]` entries with
values computed from `θ` -- every other reaction's coefficients are
carried through unchanged, as `T` values with zero partials. Mirrors
`isobaricReactor(gas, ::Type{T})`'s existing convention for AD-compatible
reactor evaluation, applied here to the kinetics coefficients themselves
rather than just the reactor state. Field promotion is driven by
`fieldnames`/`getfield` reflection, not a hardcoded field list, so it
stays correct if `ReorderedKinetics`'s own fields change.
"""
function promote_kinetics(reactions::ReorderedKinetics, ::Type{T}) where {T}
    fields = fieldnames(typeof(reactions))
    vals = map(fields) do f
        v = getfield(reactions, f)
        v isa AbstractArray{<:AbstractFloat} ? T.(v) : v
    end
    return ReorderedKinetics(vals...)
end

export wdot_func!, promote_kinetics
