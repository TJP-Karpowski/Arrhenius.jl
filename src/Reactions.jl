abstract type reversibleReaction{AT1,AT2} <: AbstractReaction{AT1,AT2} end

struct arrheniusReaction{AT1 <: AbstractVector, AT2 <: AbstractVector} <: AbstractReaction{AT1,AT2} 
    product_stoich_coeffs::AT2
    reactant_stoich_coeffs::AT2
    reactant_orders::AT2
    Arrhenius_coeffs::AT1
    vk_sum::Float64
    vk::AT2

end

struct reversibleArrheniusReaction{AT1 <: AbstractVector, AT2 <: AbstractVector} <: reversibleReaction{AT1,AT2} 
    product_stoich_coeffs::AT2
    reactant_stoich_coeffs::AT2
    reactant_orders::AT2
    Arrhenius_coeffs::AT1
    vk_sum::Float64
    vk::AT2
end

struct reversibleThreeBodyReaction{AT1 <: AbstractVector, AT2 <: AbstractVector} <: reversibleReaction{AT1,AT2} 
    product_stoich_coeffs::AT2
    reactant_stoich_coeffs::AT2
    reactant_orders::AT2
    Arrhenius_coeffs::AT1
    vk_sum::Float64
    vk::AT2
    efficiency_coeffs::AT2
    
end

struct reversibleFalloffReaction{AT1 <: AbstractVector, AT2 <: AbstractVector} <: reversibleReaction{AT1,AT2} 
    product_stoich_coeffs::AT2
    reactant_stoich_coeffs::AT2
    reactant_orders::AT2
    Arrhenius_coeffs::AT1
    vk_sum::Float64
    vk::AT2
    efficiency_coeffs::AT2
    Arrhenius_coeffs_falloff::AT1
end

struct reversibleTroeReaction{AT1 <: AbstractVector, AT2 <: AbstractVector} <: reversibleReaction{AT1,AT2} 
    product_stoich_coeffs::AT2
    reactant_stoich_coeffs::AT2
    reactant_orders::AT2
    Arrhenius_coeffs::AT1
    vk_sum::Float64
    vk::AT2
    efficiency_coeffs::AT2
    Arrhenius_coeffs_falloff::AT1
    Troe::AT1
end

@inline function ArrheniusLaw(Arrhenius_coeffs,T::Real) 
    return Arrhenius_coeffs[1] * exp(
        Arrhenius_coeffs[2] * log(T) -
        Arrhenius_coeffs[3] * (4184.0 / R / T)
    )
     
end
ArrheniusLaw(reaction::AbstractReaction,T::Real) = ArrheniusLaw(reaction.Arrhenius_coeffs,T) 

forwardRate(reaction::AbstractReaction,T::Real,C::AbstractVector) = ArrheniusLaw(reaction,T) 

function forwardRate(reaction::reversibleThreeBodyReaction,T::Real,C::AbstractVector)
    return  ArrheniusLaw(reaction,T) * dot(reaction.efficiency_coeffs, C)
end

function forwardRate(reaction::reversibleFalloffReaction, T::Real, C::AbstractVector)
        kf = ArrheniusLaw(reaction,T)
        Pr =  ArrheniusLaw(reaction.Arrhenius_coeffs_falloff,T) * dot(reaction.efficiency_coeffs, C) / kf
        kf *= (Pr / (1 + Pr))
        return kf
end
function forwardRate(reaction::reversibleTroeReaction{AT1,<:AbstractVector}, T::Real, C::AbstractVector) where AT1 <: AbstractVector
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

function backwardRate(reaction::reversibleReaction, T,kf, S0_R::AbstractVector, h_RT::AbstractVector) 
    ΔS_R = dot(reaction.vk, S0_R)
    ΔH_RT = dot(reaction.vk, h_RT)
    return kf / exp(ΔS_R - ΔH_RT + log(one_atm / R / T) * reaction.vk_sum) 
end

function backwardRate(reaction::arrheniusReaction{AT1,<:AbstractVector}, T,kf, S0_R::AT1, h_RT::AT1) where AT1 <: AbstractVector
    return return T * 0
end

function rateOfProgress(reaction::arrheniusReaction, T, C, S0_R, h_RT,tmr) 
    return @timeit tmr "RateOfProgress irreversibe" reaction.vk .* forwardRate(reaction,T,C) * Omega_C_powOrder(C,reaction.reactant_orders)
end
function Omega_C_powOrder(C::AbstractVector, order::AbstractVector)
    return @inbounds prod(C.^order)
end

function Omega_C_powOrder(C::AbstractVector, order::AbstractSparseVector)
    @inbounds Qj = one(C[1])
    for (i,index_dense) in enumerate(order.nzind)
        @inbounds Qj *= C[index_dense]^order.nzval[i]
    end
    return Qj
end

function rateOfProgress!(omegaY::AbstractVector,reaction::AbstractReaction,  T::Real, C::AbstractVector, S0_R::AbstractVector, h_RT::AbstractVector) 
    for (i, index_dense) in enumerate(reaction.vk.nzind)
    @inbounds omegaY[index_dense] += reaction.vk.nzval[i] * forwardRate(reaction,T, C)  * Omega_C_powOrder(C,reaction.reactant_orders) 
    end
    return nothing
end

function rateOfProgress!(omegaY::AbstractVector,reaction::reversibleReaction,  T::Real, C::AbstractVector, S0_R::AbstractVector, h_RT::AbstractVector) 
    kf = forwardRate(reaction,T, C) 
    kr = backwardRate(reaction,T, kf, S0_R, h_RT) 
    Qj = (kf * Omega_C_powOrder(C,reaction.reactant_orders) - kr * Omega_C_powOrder(C,reaction.product_stoich_coeffs))
    for (i, index_dense) in enumerate(reaction.vk.nzind)
        @inbounds omegaY[index_dense] += reaction.vk.nzval[i] * Qj
    end
    return nothing
end

function calc_omegaY!(omegaY::AbstractVector,gas::Solution, T, C, S0_R, h_RT,rho) 
    omegaY .= 0.0
    #RoP = sparse(similar(C))
    for i in 1:gas.n_reactions
        @inbounds rateOfProgress!(omegaY, gas.reactions[i], T, C, S0_R, h_RT)
          #omegaY[RoP.nzind] += RoP.nzval 
    end
    #omegaY .= omegaY 
    
    return nothing
end

# function calc_omegaY(gas::Solution, T, C, S0_R, h_RT,rho) 
#     omegaY = similar(C) .* 0
#     for i in gas.n_reactions
#        omegaY .= omegaY + (gas.reactions[i].vk .* rateOfProgress(gas.reactions[i], T, C, S0_R, h_RT)) 
#     end
#     omegaY .= omegaY #.* gas.MW ./ rho
    
#     return omegaY
# end

export calc_omegaY!, calc_omegaY