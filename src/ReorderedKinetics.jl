
# basic_kernels 


function arrheniusLaw!(kf::AbstractVector, arr_coeffs::AbstractArray,T::Real)
    logT = log(T)
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

function k_Pi_C_rectantOrder!(kf,C,reactant_orders)
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
      lF_cent = log10(F_cent)
      _C = -0.4 - 0.67 * lF_cent
      N = 0.75 - 1.27 * lF_cent
      f1 = (logPr[i] + _C) / (N - 0.14 * (logPr[i] + _C))
      @inbounds kf[i] *= exp(log(10.0) * lF_cent / (1 + f1^2))
    end
  end


  function wdot_func!(wdot::AbstractVector,kf::AbstractVector,kr::AbstractVector,C::AbstractVector,g_RT::AbstractVector,reactions::ReorderedKinetics ,T::Real)
    wdot.=0.0
    logT = log(T)
    rec_RcalT = (4184.0 / Arrhenius.R / T)
    
    #forwardRate from Arrhe
    arrheniusLaw!(kf, reactions.arrhenius_coeffs,logT, rec_RcalT)

    #threeBody
    tB_range = reactions.index_three_body_start:reactions.index_fallOff_start-1
    efficency!(@view(kf[tB_range]),reactions.efficiency_coeffs_tB,C)

    #fallOff adaptation
    fO_start=  reactions.index_fallOff_start#:end#reactions.index_fallOff_start-1
    arrheniusLaw!(@view(kr[fO_start:end]),reactions.Arrhenius_0,logT, rec_RcalT) #kr proxies as k0
    efficency!(@view(kr[fO_start:end]),reactions.efficiency_coeffs_fO,C) #kr proxies as k0 * dot(eff,C)
    @inbounds @view(kr[fO_start:end]) ./= @view(kf[fO_start:end]) #kr proxies as Pr
    @inbounds @. @view(kf[fO_start:end]) *= (@view(kr[fO_start:end]) / (1 + @view(kr[fO_start:end])))

    #troe
    troe_start = reactions.index_troe_start
    @inbounds kr[troe_start:end] .= log10.(@view(kr[troe_start:end])) #kr proxies as logPr
    troe!(@view(kf[troe_start:end]),@view(kr[troe_start:end]), reactions.troe,T)


    kr!(@view(kr[reactions.index_rev_start:end]),@view(kf[reactions.index_rev_start:end]),g_RT,reactions.vk_rev,reactions.vk_sum,T)
    k_Pi_C_rectantOrder!(kf,C,reactions.reactant_orders)
    k_Pi_C_rectantOrder!(@view(kr[reactions.index_rev_start:end]),C,reactions.product_stoich_coeffs)
    kf .-=kr
    vk_kf!(wdot,reactions.vk,kf)
    return nothing
end

export wdot_func!