using Pkg
Pkg.activate("/Users/jkarpowski/Dendron/notes/assets/scripts/")
using SparseArrays
using LinearAlgebra
using BenchmarkTools

using Revise
using Arrhenius

## definition of new Types 




# generation of necessary dummy fields 

    gas = CreateSolution("../mechanism/gri30.yaml",sparseCoeffs=true);
    h_RT = rand(gas.n_species)
    s0_R = rand(gas.n_species)
    g_RT = h_RT - s0_R
    kr = rand(gas.n_reactions);
    kf = zeros(gas.n_reactions);
    C = rand(gas.n_species)
    C = C./sum(C)
    wdot = zeros(gas.n_species)
    T =900

# kernels: 


function forwardRate!(kf::AbstractVector, arr_coeffs::AbstractArray,T::Real)
    logT = log(T)
    Eexp = (4184.0 / Arrhenius.R / T)
    for i in eachindex(kf) 
        @inbounds  kf[i] = arr_coeffs[i, 1]  * exp(arr_coeffs[i, 2] * logT - arr_coeffs[i, 3] * Eexp)
    end
    #@. kf = @view(arr_coeffs[:, 1])  * exp(@view(arr_coeffs[:, 2]) * logT - @view(arr_coeffs[:, 3]) * Eexp)
end

function kf_Pi_c_rectantOrder!(kf,C,reactant_orders)
    for i in eachindex(kf)
        for j in reactant_orders.colptr[i]:reactant_orders.colptr[i+1]-1
            @inbounds kf[i] *= C[reactant_orders.rowval[j]]^reactant_orders.nzval[j]
        end
    end
end

function kr!(kr, kf,g_RT,vk,vk_sum,T)
    mul!(kr,vk,g_RT) #kr proxies as Δg_RT as intermediate storage'
    logP = log(one_atm / R / T)
    @. kr = kf / exp(-kr + logP * vk_sum)
    return nothing
end

function vk_kf!(wdot,kf,vk) # TODO use likwid to compare the two methods
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
      lF_cent = log10(F_cent)
      _C = -0.4 - 0.67 * lF_cent
      N = 0.75 - 1.27 * lF_cent
      f1 = (logPr[i] + _C) / (N - 0.14 * (logPr[i] + _C))
      @inbounds kf[i] *= exp(log(10.0) * lF_cent / (1 + f1^2))
    end
  end


#Forward
function wdot_func!(wdot::AbstractVector,kf::AbstractVector,C::AbstractVector,forwardReactions::ForwardReactions ,T::Real)
    forwardRate!(kf, forwardReactions.arrhenius_coeffs,T)
    kf_Pi_c_rectantOrder!(kf,C,forwardReactions.reactant_orders)
    vk_kf!(wdot,kf,forwardReactions.vk)
    return nothing
end

#reversible 
function wdot_func!(wdot::AbstractVector,kf::AbstractVector,kr::AbstractVector,C::AbstractVector,reversibleReactions::ReversibleReactions,g_RT::AbstractVector ,T::Real)
    forwardRate!(kf, reversibleReactions.arrhenius_coeffs,T)
    kr!(kr,kf,g_RT,reversibleReactions.vk,reversibleReactions.vk_sum,T)
    kf_Pi_c_rectantOrder!(kf,C,reversibleReactions.reactant_orders)
    kf_Pi_c_rectantOrder!(kr,C,reversibleReactions.product_stoich_coeffs)
    kf .-= kr
    vk_kf!(wdot,kf,reversibleReactions.vk)
    return nothing
end

#threeBodyReactions
function wdot_func!(wdot::AbstractVector,kf::AbstractVector,kr::AbstractVector,C::AbstractVector,threeBodyReactions::ThreeBodyReactions,g_RT::AbstractVector ,T::Real)
    forwardRate!(kf, threeBodyReactions.arrhenius_coeffs,T)
    efficency!(kf,threeBodyReactions.efficiency_coeffs,C)
    kr!(kr,kf,g_RT,threeBodyReactions.vk,threeBodyReactions.vk_sum,T)
    kf_Pi_c_rectantOrder!(kf,C,threeBodyReactions.reactant_orders)
    kf_Pi_c_rectantOrder!(kr,C,threeBodyReactions.product_stoich_coeffs)
    kf .-= kr
    vk_kf!(wdot,kf,threeBodyReactions.vk)
    return nothing
end

#fallOff
function wdot_func!(wdot::AbstractVector,kf::AbstractVector,kr::AbstractVector,C::AbstractVector,fallOffReactions::FallOffReactions,g_RT::AbstractVector ,T::Real)
    forwardRate!(kf, fallOffReactions.arrhenius_coeffs,T)    
    #fallOff adaptation
    forwardRate!(kr,fallOffReactions.Arrhenius_0,T) #kr proxies as k0
    efficency!(kr,fallOffReactions.efficiency_coeffs,C) #kr proxies as k0 * dot(eff,C)
    @inbounds kr ./= kf #kr proxies as Pr
    @inbounds @. kf *= (kr / (1 + kr))
    
    kr!(kr,kf,g_RT,fallOffReactions.vk,fallOffReactions.vk_sum,T)
    kf_Pi_c_rectantOrder!(kf,C,fallOffReactions.reactant_orders)
    kf_Pi_c_rectantOrder!(kr,C,fallOffReactions.product_stoich_coeffs)
    kf .-= kr
    vk_kf!(wdot,kf,fallOffReactions.vk)
    return nothing
end

function wdot_func!(wdot::AbstractVector,kf::AbstractVector,kr::AbstractVector,C::AbstractVector,troeReactions::TroeReactions,g_RT::AbstractVector ,T::Real)
    forwardRate!(kf, troeReactions.arrhenius_coeffs,T)
    
    #fallOff adaptation
    forwardRate!(kr,troeReactions.Arrhenius_0,T) #kr proxies as k0
    efficency!(kr,troeReactions.efficiency_coeffs,C) #kr proxies as k0 * dot(eff,C)
    @inbounds kr ./= kf #kr proxies as Pr
    @inbounds @. kf *= (kr / (1 + kr))
    @inbounds kr .= log10.(kr) #kr proxies as logPr
    troe!(kf,kr, troeReactions.troe,T)

    kr!(kr,kf,g_RT,troeReactions.vk,troeReactions.vk_sum,T)
    kf_Pi_c_rectantOrder!(kf,C,troeReactions.reactant_orders)
    kf_Pi_c_rectantOrder!(kr,C,troeReactions.product_stoich_coeffs)
    kf .-= kr
    vk_kf!(wdot,kf,troeReactions.vk)
    return nothing
end

# generate 