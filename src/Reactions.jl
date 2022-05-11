abstract type reversibleReaction <: AbstractReaction end

struct arrheniusReaction{AbArray<:AbstractArray,AbArray2<:AbstractArray} <: AbstractReaction 
    product_stoich_coeffs::AbArray2
    reactant_stoich_coeffs::AbArray2
    reactant_orders::AbArray2
    Arrhenius_coeffs::AbArray
    vk_sum

end

struct reversibleArrheniusReaction{AbArray<:AbstractArray,AbArray2<:AbstractArray} <: reversibleReaction 
    product_stoich_coeffs::AbArray2
    reactant_stoich_coeffs::AbArray2
    reactant_orders::AbArray2
    Arrhenius_coeffs::AbArray
    vk_sum
end

struct reversibleThreeBodyReaction{AbArray<:AbstractArray,AbArray2<:AbstractArray} <: reversibleReaction 
    product_stoich_coeffs::AbArray2
    reactant_stoich_coeffs::AbArray2
    reactant_orders::AbArray2
    Arrhenius_coeffs::AbArray
    vk_sum
    efficiency_coeffs::AbArray2
    
end

struct reversibleFalloffReaction{AbArray<:AbstractArray,AbArray2<:AbstractArray} <: reversibleReaction 
    product_stoich_coeffs::AbArray2
    reactant_stoich_coeffs::AbArray2
    reactant_orders::AbArray2
    Arrhenius_coeffs::AbArray
    vk_sum
    efficiency_coeffs::AbArray2
    Arrhenius_coeffs_falloff::AbArray
end

struct reversibleTroeReaction{AbArray<:AbstractArray,AbArray2<:AbstractArray} <: reversibleReaction 
    product_stoich_coeffs::AbArray2
    reactant_stoich_coeffs::AbArray2
    reactant_orders::AbArray2
    Arrhenius_coeffs::AbArray
    vk_sum
    efficiency_coeffs::AbArray2
    Arrhenius_coeffs_falloff::AbArray
    Troe::AbArray
end

function ArrheniusLaw(Arrhenius_coeffs::AbstractArray,T::Real)
    kf = Arrhenius_coeffs[1] * exp(
        Arrhenius_coeffs[2] * log(T) -
        Arrhenius_coeffs[3] * (4184.0 / R / T)
    )
    return kf
end
ArrheniusLaw(reaction::AbstractReaction,T::Real) = ArrheniusLaw(reaction.Arrhenius_coeffs,T)

forwardRate(reaction::AbstractReaction,T::Real,C::AbstractArray) = ArrheniusLaw(reaction,T)

function forwardRate(reaction::reversibleThreeBodyReaction,T::Real,C::AbstractArray)
    return  ArrheniusLaw(reaction,T) * dot(reaction.efficiency_coeffs, C)
end

function forwardRate(reaction::reversibleFalloffReaction, T::Real, C::AbstractArray)
        kf = ArrheniusLaw(reaction,T)
        Pr =  ArrheniusLaw(reaction.Arrhenius_coeffs_falloff,T) * dot(reaction.efficiency_coeffs, C) / kf
        kf *= (Pr / (1 + Pr))
        return kf
end
function forwardRate(reaction::reversibleTroeReaction, T::Real, C::AbstractArray)
    kf = ArrheniusLaw(reaction,T)
    Pr =  ArrheniusLaw(reaction.Arrhenius_coeffs_falloff,T) * dot(reaction.efficiency_coeffs, C) / kf
    kf *= (Pr / (1 + Pr))
    lPr = log10(Pr)
    F_cent =
        (1 - reaction.Troe[1]) * exp(-T / reaction.Troe[4]) +
        reaction.Troe[ 1] * exp(-T / reaction.Troe[2]) +
        exp(-reaction.Troe[3] / T)
    lF_cent = log10(F_cent)
    _C = -0.4 - 0.67 * lF_cent
    N = 0.75 - 1.27 * lF_cent
    f1 = (lPr + _C) / (N - 0.14 * (lPr + _C))
    kf *= exp(log(10.0) * lF_cent / (1 + f1^2))
    return kf
end

function backwardRate(reaction::reversibleReaction, T,kf, S0_R, h_RT) 
    ΔS_R = dot((reaction.product_stoich_coeffs - reaction.reactant_stoich_coeffs), S0_R)
    ΔH_RT = dot((reaction.product_stoich_coeffs - reaction.reactant_stoich_coeffs), h_RT)
    return kf / exp(ΔS_R - ΔH_RT + log(one_atm / R / T) * reaction.vk_sum) 
end

function backwardRate(reaction::arrheniusReaction, T,kf, S0_R, h_RT) 
    return return T * 0
end

# function rateOfProgress(reaction::arrheniusReaction, T, C, S0_R, h_RT) 
#     Q = forwardRate(reaction,T,C)
#     for (i, order)  in enumerate(reaction.reactant_orders)#zip(findnz(reaction.reactant_orders)...)
#         Q *= C[i]^order
#     end
#     return Q
# end
function Omega_C_powOrder(C::AbstractArray, order::AbstractArray)
    return prod(C.^order)
end

function Omega_C_powOrder(C::AbstractArray, order::AbstractSparseVector)
    indexes, values = findnz(order)
    return prod(C[indexes].^values)
end

function rateOfProgress(reaction::AbstractReaction,  T, C, S0_R, h_RT) 
    kf = forwardRate(reaction,T, C) 
    kr = backwardRate(reaction,T, kf, S0_R, h_RT) 
    return kf * Omega_C_powOrder(C,reaction.reactant_orders) - kr * Omega_C_powOrder(C,reaction.product_stoich_coeffs)
end

function calc_omegaY!(omegaY::AbstractArray,gas::Solution, T, C, S0_R, h_RT,rho) 
    omegaY .= 0
    for reaction in gas.reactions
        omegaY += (reaction.product_stoich_coeffs - reaction.reactant_stoich_coeffs) .* rateOfProgress(reaction, T, C, S0_R, h_RT) 
    end
    omegaY .= omegaY #.* gas.MW ./ rho
    
    return nothing
end

function calc_omegaY(gas::Solution, T, C, S0_R, h_RT,rho) 
    omegaY = similar(C) .* 0
    for reaction in gas.reactions
        omegaY .= omegaY + ((reaction.product_stoich_coeffs - reaction.reactant_stoich_coeffs) .* rateOfProgress(reaction, T, C, S0_R, h_RT)) 
    end
    #omegaY .= omegaY #.* gas.MW ./ rho
    
    return omegaY
end

export calc_omegaY!, calc_omegaY