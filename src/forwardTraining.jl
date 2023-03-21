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

function rateOfProgress!(reaction::AbstractReaction{AT1,AT1} ,  T::Real, C::AbstractVector, S0_R::AbstractVector, h_RT::AbstractVector) where AT1<:AbstractVector 
    kf = forwardRate(reaction,T, C) 
    kr = backwardRate(reaction,T, kf, S0_R, h_RT) 
    Qj = (kf * Omega_C_powOrder(C,reaction.reactant_orders) - kr * Omega_C_powOrder(C,reaction.product_stoich_coeffs))
    vk = reaction.product_stoich_coeffs - reaction.reactant_stoich_coeffs
    index_dense = vk .!= 0 
    omegaY[index_dense] += vk[index_dense] * Qj
    return nothing
end