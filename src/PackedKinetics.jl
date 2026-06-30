function PackedSparseColumns(A::SparseMatrixCSC{Float64,Int64})
    return PackedSparseColumns(
        copy(A.colptr),
        copy(A.rowval),
        copy(A.nzval),
        size(A, 1),
        size(A, 2),
    )
end

function PackedKinetics(reactions::ReorderedKinetics)
    return PackedKinetics(
        copy(reactions.arrhenius_coeffs),
        PackedSparseColumns(reactions.reactant_orders),
        PackedSparseColumns(reactions.product_stoich_coeffs),
        PackedSparseColumns(reactions.reactant_stoich_coeffs),
        PackedSparseColumns(reactions.vk),
        PackedSparseColumns(reactions.vk_rev),
        copy(reactions.vk_sum),
        PackedSparseColumns(reactions.efficiency_coeffs_tB),
        PackedSparseColumns(reactions.efficiency_coeffs_fO),
        copy(reactions.Arrhenius_0),
        copy(reactions.troe),
        reactions.index_rev_start,
        reactions.index_three_body_start,
        reactions.index_fallOff_start,
        reactions.index_troe_start,
    )
end

function k_Pi_C_rectantOrder!(kf, C, reactant_orders::PackedSparseColumns, first_col::Int, last_col::Int, offset::Int)
    for col in first_col:last_col
        i = col + offset
        for j in reactant_orders.colptr[col]:reactant_orders.colptr[col + 1] - 1
            @inbounds kf[i] *= C[reactant_orders.rowval[j]]^reactant_orders.nzval[j]
        end
    end
    return nothing
end

function efficiency!(kf, effCoeff::PackedSparseColumns, C, first_col::Int, last_col::Int, offset::Int)
    for col in first_col:last_col
        i = col + offset
        dot_C_eff = 0.0
        for j in effCoeff.colptr[col]:effCoeff.colptr[col + 1] - 1
            @inbounds dot_C_eff += effCoeff.nzval[j] * C[effCoeff.rowval[j]]
        end
        @inbounds kf[i] *= dot_C_eff
    end
    return nothing
end

function packed_mul!(y, A::PackedSparseColumns, x)
    y .= 0.0
    for col in 1:A.ncol
        x_col = x[col]
        for j in A.colptr[col]:A.colptr[col + 1] - 1
            @inbounds y[A.rowval[j]] += A.nzval[j] * x_col
        end
    end
    return nothing
end

function packed_mul_offset!(y, A::PackedSparseColumns, x, first::Int)
    last = first + A.nrow - 1
    for i in first:last
        y[i] = 0.0
    end
    for col in 1:A.ncol
        x_col = x[col]
        for j in A.colptr[col]:A.colptr[col + 1] - 1
            @inbounds y[first + A.rowval[j] - 1] += A.nzval[j] * x_col
        end
    end
    return nothing
end

function arrheniusLaw!(k, arr_coeffs, logT::Real, rec_RcalT::Real, first::Int)
    for i in axes(arr_coeffs, 1)
        @inbounds k[first + i - 1] =
            arr_coeffs[i, 1] * exp(arr_coeffs[i, 2] * logT - arr_coeffs[i, 3] * rec_RcalT)
    end
    return nothing
end

function kr!(kr, kf, g_RT, vk::PackedSparseColumns, vk_sum, T, first::Int, last::Int)
    packed_mul_offset!(kr, vk, g_RT, first)
    logP = log(one_atm / R / T)
    for i in first:last
        local_i = i - first + 1
        @inbounds kr[i] = kf[i] / exp(-kr[i] + logP * vk_sum[local_i])
    end
    return nothing
end

function vk_kf!(wdot, vk::PackedSparseColumns, kf)
    for isp in eachindex(wdot)
        for j in vk.colptr[isp]:vk.colptr[isp + 1] - 1
            @inbounds wdot[isp] += vk.nzval[j] * kf[vk.rowval[j]]
        end
    end
    return nothing
end

function wdot_func!(
    wdot::AbstractVector,
    kf::AbstractVector,
    kr::AbstractVector,
    C::AbstractVector,
    g_RT::AbstractVector,
    reactions::PackedKinetics,
    T::Real,
)
    wdot .= 0.0
    logT = log(T)
    rec_RcalT = 4184.0 / Arrhenius.R / T
    n_reactions = size(reactions.arrhenius_coeffs, 1)

    arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)

    tB_first = reactions.index_three_body_start
    tB_last = reactions.index_fallOff_start - 1
    if tB_first <= tB_last
        efficiency!(kf, reactions.efficiency_coeffs_tB, C, 1, tB_last - tB_first + 1, tB_first - 1)
    end

    fO_first = reactions.index_fallOff_start
    if fO_first <= n_reactions
        arrheniusLaw!(kr, reactions.Arrhenius_0, logT, rec_RcalT, fO_first)
        efficiency!(kr, reactions.efficiency_coeffs_fO, C, 1, n_reactions - fO_first + 1, fO_first - 1)
        for i in fO_first:n_reactions
            @inbounds kr[i] /= kf[i]
            @inbounds kf[i] *= kr[i] / (1 + kr[i])
        end
    end

    troe_first = reactions.index_troe_start
    if troe_first <= n_reactions
        for i in troe_first:n_reactions
            @inbounds kr[i] = log10(kr[i])
        end
        for i in troe_first:n_reactions
            troe_i = i - troe_first + 1
            @inbounds F_cent =
                (1 - reactions.troe[troe_i, 1]) * exp(-T / reactions.troe[troe_i, 4]) +
                reactions.troe[troe_i, 1] * exp(-T / reactions.troe[troe_i, 2]) +
                exp(-reactions.troe[troe_i, 3] / T)
            lF_cent = log10(F_cent)
            _C = -0.4 - 0.67 * lF_cent
            N = 0.75 - 1.27 * lF_cent
            @inbounds f1 = (kr[i] + _C) / (N - 0.14 * (kr[i] + _C))
            @inbounds kf[i] *= exp(log(10.0) * lF_cent / (1 + f1^2))
        end
    end

    rev_first = reactions.index_rev_start
    if rev_first <= n_reactions
        kr!(kr, kf, g_RT, reactions.vk_rev, reactions.vk_sum, T, rev_first, n_reactions)
    end

    k_Pi_C_rectantOrder!(kf, C, reactions.reactant_orders, 1, reactions.reactant_orders.ncol, 0)
    if rev_first <= n_reactions
        k_Pi_C_rectantOrder!(
            kr,
            C,
            reactions.product_stoich_coeffs,
            1,
            reactions.product_stoich_coeffs.ncol,
            rev_first - 1,
        )
    end

    for i in 1:n_reactions
        @inbounds kf[i] -= kr[i]
    end
    vk_kf!(wdot, reactions.vk, kf)
    return nothing
end

export PackedKinetics, PackedSparseColumns
