module ArrheniusKernelAbstractionsExt

using Arrhenius
using KernelAbstractions

function sync_backend(backend, event)
    event === nothing || wait(event)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline gpu_positive(x) = ifelse(x > zero(x), x, oftype(x, 1.0e-30))
@inline gpu_log(x) = log(gpu_positive(x))
@inline gpu_log10(x) = log10(gpu_positive(x))

@inline function arrhenius_rate(coeffs, row::Int, logT, rec_RcalT)
    return coeffs[row, 1] * exp(coeffs[row, 2] * logT - coeffs[row, 3] * rec_RcalT)
end

@inline function sparse_product(colptr, rowval, nzval, values, reactor::Int, col::Int)
    product = one(eltype(values))
    for j in colptr[col]:colptr[col + 1] - 1
        @inbounds product *= values[rowval[j], reactor]^nzval[j]
    end
    return product
end

@inline function sparse_dot(colptr, rowval, nzval, values, reactor::Int, col::Int)
    total = zero(eltype(values))
    for j in colptr[col]:colptr[col + 1] - 1
        @inbounds total += nzval[j] * values[rowval[j], reactor]
    end
    return total
end

@inline function reverse_delta_g(colptr, rowval, nzval, g_RT, reactor::Int, local_reaction::Int, n_species::Int)
    total = zero(eltype(g_RT))
    for species in 1:n_species
        for j in colptr[species]:colptr[species + 1] - 1
            if rowval[j] == local_reaction
                @inbounds total += nzval[j] * g_RT[species, reactor]
            end
        end
    end
    return total
end

@inline function pressure_dependent_rate(pressures, coeffs, first::Int, last::Int, pressure_atm, logT, rec_RcalT)
    first == last && return arrhenius_rate(coeffs, first, logT, rec_RcalT)
    p = gpu_positive(pressure_atm)
    if p <= pressures[first]
        return arrhenius_rate(coeffs, first, logT, rec_RcalT)
    elseif p >= pressures[last]
        return arrhenius_rate(coeffs, last, logT, rec_RcalT)
    end

    logp = gpu_log(p)
    for i in first:last-1
        p_low = pressures[i]
        p_high = pressures[i + 1]
        if p >= p_low && p <= p_high
            k_low = arrhenius_rate(coeffs, i, logT, rec_RcalT)
            k_high = arrhenius_rate(coeffs, i + 1, logT, rec_RcalT)
            weight = (logp - gpu_log(p_low)) / (gpu_log(p_high) - gpu_log(p_low))
            return exp((one(weight) - weight) * gpu_log(k_low) + weight * gpu_log(k_high))
        end
    end
    return arrhenius_rate(coeffs, last, logT, rec_RcalT)
end

@inline function concentration_at(Y, p, T, R, MW, mean_MW, species::Int, reactor::Int)
    return Y[species, reactor] * p * mean_MW / (R * T * MW[species])
end

@inline function thermo_h_s(nasa_low, nasa_high, T, species::Int)
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    logT = gpu_log(T)
    invT = inv(T)
    use_low = T <= oftype(T, 1000.0)
    a1 = use_low ? nasa_low[species, 1] : nasa_high[species, 1]
    a2 = use_low ? nasa_low[species, 2] : nasa_high[species, 2]
    a3 = use_low ? nasa_low[species, 3] : nasa_high[species, 3]
    a4 = use_low ? nasa_low[species, 4] : nasa_high[species, 4]
    a5 = use_low ? nasa_low[species, 5] : nasa_high[species, 5]
    a6 = use_low ? nasa_low[species, 6] : nasa_high[species, 6]
    a7 = use_low ? nasa_low[species, 7] : nasa_high[species, 7]
    cp_R = a1 + a2 * T + a3 * T2 + a4 * T3 + a5 * T4
    h_RT = a1 + a2 * T / 2 + a3 * T2 / 3 + a4 * T3 / 4 + a5 * T4 / 5 + a6 * invT
    s0_R = a1 * logT + a2 * T + a3 * T2 / 2 + a4 * T3 / 3 + a5 * T4 / 4 + a7
    return cp_R, h_RT, s0_R
end

@inline function sparse_product_concentration(colptr, rowval, nzval, Y, p, T, R, MW, mean_MW, reactor::Int, col::Int)
    product = one(T)
    for j in colptr[col]:colptr[col + 1] - 1
        @inbounds product *= concentration_at(Y, p, T, R, MW, mean_MW, rowval[j], reactor)^nzval[j]
    end
    return product
end

@inline function sparse_dot_concentration(colptr, rowval, nzval, Y, p, T, R, MW, mean_MW, reactor::Int, col::Int)
    total = zero(T)
    for j in colptr[col]:colptr[col + 1] - 1
        @inbounds total += nzval[j] * concentration_at(Y, p, T, R, MW, mean_MW, rowval[j], reactor)
    end
    return total
end

@inline function reverse_delta_g_thermo(vk_rev_colptr, vk_rev_rowval, vk_rev_nzval, nasa_low, nasa_high, T, local_reaction::Int, n_species::Int)
    total = zero(T)
    for species in 1:n_species
        for j in vk_rev_colptr[species]:vk_rev_colptr[species + 1] - 1
            if vk_rev_rowval[j] == local_reaction
                _, h_RT, s0_R = thermo_h_s(nasa_low, nasa_high, T, species)
                @inbounds total += vk_rev_nzval[j] * (h_RT - s0_R)
            end
        end
    end
    return total
end

@inline real_at(realbuf, offset::Int, i::Int) = realbuf[offset + i - 1]
@inline int_at(intbuf, offset::Int, i::Int) = intbuf[offset + i - 1]
@inline real_mat(realbuf, offset::Int, nrow::Int, row::Int, col::Int) = realbuf[offset + (col - 1) * nrow + row - 1]

@inline function packed_arrhenius_rate(realbuf, arrhenius_offset::Int, row::Int, nrow::Int, logT, rec_RcalT)
    A = real_mat(realbuf, arrhenius_offset, nrow, row, 1)
    b = real_mat(realbuf, arrhenius_offset, nrow, row, 2)
    Ea = real_mat(realbuf, arrhenius_offset, nrow, row, 3)
    return A * exp(b * logT - Ea * rec_RcalT)
end

@inline function packed_thermo_h_s(realbuf, nasa_low_offset::Int, nasa_high_offset::Int, n_species::Int, T, species::Int)
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    logT = gpu_log(T)
    invT = inv(T)
    use_low = T <= oftype(T, 1000.0)
    nasa_offset = use_low ? nasa_low_offset : nasa_high_offset
    a1 = real_mat(realbuf, nasa_offset, n_species, species, 1)
    a2 = real_mat(realbuf, nasa_offset, n_species, species, 2)
    a3 = real_mat(realbuf, nasa_offset, n_species, species, 3)
    a4 = real_mat(realbuf, nasa_offset, n_species, species, 4)
    a5 = real_mat(realbuf, nasa_offset, n_species, species, 5)
    a6 = real_mat(realbuf, nasa_offset, n_species, species, 6)
    a7 = real_mat(realbuf, nasa_offset, n_species, species, 7)
    cp_R = a1 + a2 * T + a3 * T2 + a4 * T3 + a5 * T4
    h_RT = a1 + a2 * T / 2 + a3 * T2 / 3 + a4 * T3 / 4 + a5 * T4 / 5 + a6 * invT
    s0_R = a1 * logT + a2 * T + a3 * T2 / 2 + a4 * T3 / 3 + a5 * T4 / 4 + a7
    return cp_R, h_RT, s0_R
end

@inline function packed_concentration_at(Y, p, T, R, realbuf, mw_offset::Int, mean_MW, species::Int, reactor::Int)
    return Y[species, reactor] * p * mean_MW / (R * T * real_at(realbuf, mw_offset, species))
end

@inline function packed_sparse_product_concentration(
    realbuf,
    intbuf,
    colptr_offset::Int,
    rowval_offset::Int,
    nzval_offset::Int,
    Y,
    p,
    T,
    R,
    mw_offset::Int,
    mean_MW,
    reactor::Int,
    col::Int,
)
    product = one(T)
    first = int_at(intbuf, colptr_offset, col)
    last = int_at(intbuf, colptr_offset, col + 1) - 1
    for j in first:last
        species = int_at(intbuf, rowval_offset, j)
        exponent = real_at(realbuf, nzval_offset, j)
        @inbounds product *= packed_concentration_at(Y, p, T, R, realbuf, mw_offset, mean_MW, species, reactor)^exponent
    end
    return product
end

@inline function packed_sparse_dot_concentration(
    realbuf,
    intbuf,
    colptr_offset::Int,
    rowval_offset::Int,
    nzval_offset::Int,
    Y,
    p,
    T,
    R,
    mw_offset::Int,
    mean_MW,
    reactor::Int,
    col::Int,
)
    total = zero(T)
    first = int_at(intbuf, colptr_offset, col)
    last = int_at(intbuf, colptr_offset, col + 1) - 1
    for j in first:last
        species = int_at(intbuf, rowval_offset, j)
        coefficient = real_at(realbuf, nzval_offset, j)
        @inbounds total += coefficient * packed_concentration_at(Y, p, T, R, realbuf, mw_offset, mean_MW, species, reactor)
    end
    return total
end

@inline function packed_reverse_delta_g_thermo(
    realbuf,
    intbuf,
    vkrev_colptr_offset::Int,
    vkrev_rowval_offset::Int,
    vkrev_nzval_offset::Int,
    nasa_low_offset::Int,
    nasa_high_offset::Int,
    T,
    local_reaction::Int,
    n_species::Int,
)
    total = zero(T)
    for species in 1:n_species
        first = int_at(intbuf, vkrev_colptr_offset, species)
        last = int_at(intbuf, vkrev_colptr_offset, species + 1) - 1
        for j in first:last
            if int_at(intbuf, vkrev_rowval_offset, j) == local_reaction
                _, h_RT, s0_R = packed_thermo_h_s(realbuf, nasa_low_offset, nasa_high_offset, n_species, T, species)
                @inbounds total += real_at(realbuf, vkrev_nzval_offset, j) * (h_RT - s0_R)
            end
        end
    end
    return total
end

@kernel function batched_isobaric_rhs_packed_no_pdep_kernel!(
    dYdt,
    dTdt,
    Y,
    Tvec,
    pvec,
    realbuf,
    intbuf,
    q,
)
    reactor = @index(Global)
    n_species = int_at(intbuf, 1, 1)
    n_reactions = int_at(intbuf, 1, 2)
    index_rev_start = int_at(intbuf, 1, 3)
    index_three_body_start = int_at(intbuf, 1, 4)
    index_falloff_start = int_at(intbuf, 1, 5)
    index_troe_start = int_at(intbuf, 1, 6)
    n_reactors = int_at(intbuf, 1, 7)
    mw_offset = int_at(intbuf, 1, 8)
    nasa_low_offset = int_at(intbuf, 1, 9)
    nasa_high_offset = int_at(intbuf, 1, 10)
    arrhenius_offset = int_at(intbuf, 1, 11)
    reactant_colptr_offset = int_at(intbuf, 1, 12)
    reactant_rowval_offset = int_at(intbuf, 1, 13)
    reactant_nzval_offset = int_at(intbuf, 1, 14)
    product_colptr_offset = int_at(intbuf, 1, 15)
    product_rowval_offset = int_at(intbuf, 1, 16)
    product_nzval_offset = int_at(intbuf, 1, 17)
    vk_colptr_offset = int_at(intbuf, 1, 18)
    vk_rowval_offset = int_at(intbuf, 1, 19)
    vk_nzval_offset = int_at(intbuf, 1, 20)
    vkrev_colptr_offset = int_at(intbuf, 1, 21)
    vkrev_rowval_offset = int_at(intbuf, 1, 22)
    vkrev_nzval_offset = int_at(intbuf, 1, 23)
    vksum_offset = int_at(intbuf, 1, 24)
    tb_colptr_offset = int_at(intbuf, 1, 25)
    tb_rowval_offset = int_at(intbuf, 1, 26)
    tb_nzval_offset = int_at(intbuf, 1, 27)
    fo_colptr_offset = int_at(intbuf, 1, 28)
    fo_rowval_offset = int_at(intbuf, 1, 29)
    fo_nzval_offset = int_at(intbuf, 1, 30)
    arrhenius0_offset = int_at(intbuf, 1, 31)
    troe_offset = int_at(intbuf, 1, 32)
    if reactor <= n_reactors
        T = Tvec[reactor]
        p = pvec[reactor]
        R = oftype(T, Arrhenius.R)
        one_atm = oftype(T, Arrhenius.one_atm)
        logT = gpu_log(T)
        rec_RcalT = oftype(T, 4184.0) / R / T

        mean_MW_inv = zero(T)
        cp_mass_sum = zero(T)
        for species in 1:n_species
            @inbounds mean_MW_inv += Y[species, reactor] / real_at(realbuf, mw_offset, species)
        end
        mean_MW = inv(mean_MW_inv)
        rho_mass = p / R / T * mean_MW
        for species in 1:n_species
            cp_R, _, _ = packed_thermo_h_s(realbuf, nasa_low_offset, nasa_high_offset, n_species, T, species)
            @inbounds cp_mass_sum += cp_R * Y[species, reactor] / real_at(realbuf, mw_offset, species) * mean_MW
        end
        cp_mass = cp_mass_sum * R / mean_MW

        for i in 1:n_reactions
            @inbounds q[i, reactor] = packed_arrhenius_rate(realbuf, arrhenius_offset, i, n_reactions, logT, rec_RcalT)
        end

        for i in index_three_body_start:index_falloff_start-1
            col = i - index_three_body_start + 1
            @inbounds q[i, reactor] *= packed_sparse_dot_concentration(
                realbuf,
                intbuf,
                tb_colptr_offset,
                tb_rowval_offset,
                tb_nzval_offset,
                Y,
                p,
                T,
                R,
                mw_offset,
                mean_MW,
                reactor,
                col,
            )
        end

        for i in index_falloff_start:n_reactions
            col = i - index_falloff_start + 1
            k0 = packed_arrhenius_rate(realbuf, arrhenius0_offset, col, n_reactions - index_falloff_start + 1, logT, rec_RcalT)
            k0 *= packed_sparse_dot_concentration(
                realbuf,
                intbuf,
                fo_colptr_offset,
                fo_rowval_offset,
                fo_nzval_offset,
                Y,
                p,
                T,
                R,
                mw_offset,
                mean_MW,
                reactor,
                col,
            )
            @inbounds begin
                Pr = gpu_positive(k0 / q[i, reactor])
                q[i, reactor] *= Pr / (one(T) + Pr)
            end
        end

        for i in index_troe_start:n_reactions
            col = i - index_falloff_start + 1
            k0 = packed_arrhenius_rate(realbuf, arrhenius0_offset, col, n_reactions - index_falloff_start + 1, logT, rec_RcalT)
            k0 *= packed_sparse_dot_concentration(
                realbuf,
                intbuf,
                fo_colptr_offset,
                fo_rowval_offset,
                fo_nzval_offset,
                Y,
                p,
                T,
                R,
                mw_offset,
                mean_MW,
                reactor,
                col,
            )
            @inbounds begin
                Pr = gpu_positive(k0 / packed_arrhenius_rate(realbuf, arrhenius_offset, i, n_reactions, logT, rec_RcalT))
                logPr = gpu_log10(Pr)
                troe_i = i - index_troe_start + 1
                F_cent =
                    (one(T) - real_mat(realbuf, troe_offset, n_reactions - index_troe_start + 1, troe_i, 1)) *
                    exp(-T / real_mat(realbuf, troe_offset, n_reactions - index_troe_start + 1, troe_i, 4)) +
                    real_mat(realbuf, troe_offset, n_reactions - index_troe_start + 1, troe_i, 1) *
                    exp(-T / real_mat(realbuf, troe_offset, n_reactions - index_troe_start + 1, troe_i, 2)) +
                    exp(-real_mat(realbuf, troe_offset, n_reactions - index_troe_start + 1, troe_i, 3) / T)
                lF_cent = gpu_log10(F_cent)
                c = oftype(T, -0.4) - oftype(T, 0.67) * lF_cent
                N = oftype(T, 0.75) - oftype(T, 1.27) * lF_cent
                f1 = (logPr + c) / (N - oftype(T, 0.14) * (logPr + c))
                q[i, reactor] *= exp(gpu_log(oftype(T, 10.0)) * lF_cent / (one(T) + f1^2))
            end
        end

        for i in 1:n_reactions
            k_eff = q[i, reactor]
            forward = k_eff * packed_sparse_product_concentration(
                realbuf,
                intbuf,
                reactant_colptr_offset,
                reactant_rowval_offset,
                reactant_nzval_offset,
                Y,
                p,
                T,
                R,
                mw_offset,
                mean_MW,
                reactor,
                i,
            )
            reverse = zero(T)
            if i >= index_rev_start
                local_i = i - index_rev_start + 1
                delta_g = packed_reverse_delta_g_thermo(
                    realbuf,
                    intbuf,
                    vkrev_colptr_offset,
                    vkrev_rowval_offset,
                    vkrev_nzval_offset,
                    nasa_low_offset,
                    nasa_high_offset,
                    T,
                    local_i,
                    n_species,
                )
                k_rev = k_eff / exp(-delta_g + gpu_log(one_atm / R / T) * real_at(realbuf, vksum_offset, local_i))
                reverse = k_rev * packed_sparse_product_concentration(
                    realbuf,
                    intbuf,
                    product_colptr_offset,
                    product_rowval_offset,
                    product_nzval_offset,
                    Y,
                    p,
                    T,
                    R,
                    mw_offset,
                    mean_MW,
                    reactor,
                    local_i,
                )
            end
            @inbounds q[i, reactor] = forward - reverse
        end

        dT = zero(T)
        for species in 1:n_species
            total = zero(T)
            first = int_at(intbuf, vk_colptr_offset, species)
            last = int_at(intbuf, vk_colptr_offset, species + 1) - 1
            for j in first:last
                @inbounds total += real_at(realbuf, vk_nzval_offset, j) * q[int_at(intbuf, vk_rowval_offset, j), reactor]
            end
            _, h_RT, _ = packed_thermo_h_s(realbuf, nasa_low_offset, nasa_high_offset, n_species, T, species)
            @inbounds begin
                dYdt[species, reactor] = total * real_at(realbuf, mw_offset, species) / rho_mass
                dT -= h_RT * total
            end
        end
        @inbounds dTdt[reactor] = dT * R * T / cp_mass / rho_mass
    end
end

@kernel function batched_isobaric_rhs_no_pdep_kernel!(
    dYdt,
    dTdt,
    Y,
    Tvec,
    pvec,
    MW,
    nasa_low,
    nasa_high,
    q,
    arrhenius_coeffs,
    reactant_colptr,
    reactant_rowval,
    reactant_nzval,
    product_colptr,
    product_rowval,
    product_nzval,
    vk_colptr,
    vk_rowval,
    vk_nzval,
    vk_rev_colptr,
    vk_rev_rowval,
    vk_rev_nzval,
    vk_sum,
    tb_colptr,
    tb_rowval,
    tb_nzval,
    fo_colptr,
    fo_rowval,
    fo_nzval,
    arrhenius_0,
    troe,
    n_species::Int,
    n_reactions::Int,
    index_rev_start::Int,
    index_three_body_start::Int,
    index_falloff_start::Int,
    index_troe_start::Int,
    n_reactors::Int,
)
    reactor = @index(Global)
    if reactor <= n_reactors
        T = Tvec[reactor]
        p = pvec[reactor]
        R = oftype(T, Arrhenius.R)
        one_atm = oftype(T, Arrhenius.one_atm)
        logT = gpu_log(T)
        rec_RcalT = oftype(T, 4184.0) / R / T

        mean_MW_inv = zero(T)
        cp_mass_sum = zero(T)
        for species in 1:n_species
            @inbounds begin
                mean_MW_inv += Y[species, reactor] / MW[species]
            end
        end
        mean_MW = inv(mean_MW_inv)
        rho_mass = p / R / T * mean_MW
        for species in 1:n_species
            cp_R, _, _ = thermo_h_s(nasa_low, nasa_high, T, species)
            @inbounds cp_mass_sum += cp_R * Y[species, reactor] / MW[species] * mean_MW
        end
        cp_mass = cp_mass_sum * R / mean_MW

        for i in 1:n_reactions
            @inbounds q[i, reactor] = arrhenius_rate(arrhenius_coeffs, i, logT, rec_RcalT)
        end

        for i in index_three_body_start:index_falloff_start-1
            col = i - index_three_body_start + 1
            @inbounds q[i, reactor] *= sparse_dot_concentration(tb_colptr, tb_rowval, tb_nzval, Y, p, T, R, MW, mean_MW, reactor, col)
        end

        for i in index_falloff_start:n_reactions
            col = i - index_falloff_start + 1
            k0 = arrhenius_rate(arrhenius_0, col, logT, rec_RcalT)
            k0 *= sparse_dot_concentration(fo_colptr, fo_rowval, fo_nzval, Y, p, T, R, MW, mean_MW, reactor, col)
            @inbounds begin
                Pr = gpu_positive(k0 / q[i, reactor])
                q[i, reactor] *= Pr / (one(T) + Pr)
                q[i, reactor] = q[i, reactor]
            end
        end

        for i in index_troe_start:n_reactions
            col = i - index_falloff_start + 1
            k0 = arrhenius_rate(arrhenius_0, col, logT, rec_RcalT)
            k0 *= sparse_dot_concentration(fo_colptr, fo_rowval, fo_nzval, Y, p, T, R, MW, mean_MW, reactor, col)
            @inbounds begin
                Pr = gpu_positive(k0 / arrhenius_rate(arrhenius_coeffs, i, logT, rec_RcalT))
                logPr = gpu_log10(Pr)
                troe_i = i - index_troe_start + 1
                F_cent =
                    (one(T) - troe[troe_i, 1]) * exp(-T / troe[troe_i, 4]) +
                    troe[troe_i, 1] * exp(-T / troe[troe_i, 2]) +
                    exp(-troe[troe_i, 3] / T)
                lF_cent = gpu_log10(F_cent)
                c = oftype(T, -0.4) - oftype(T, 0.67) * lF_cent
                N = oftype(T, 0.75) - oftype(T, 1.27) * lF_cent
                f1 = (logPr + c) / (N - oftype(T, 0.14) * (logPr + c))
                q[i, reactor] *= exp(gpu_log(oftype(T, 10.0)) * lF_cent / (one(T) + f1^2))
            end
        end

        for i in 1:n_reactions
            k_eff = q[i, reactor]
            forward = k_eff * sparse_product_concentration(reactant_colptr, reactant_rowval, reactant_nzval, Y, p, T, R, MW, mean_MW, reactor, i)
            reverse = zero(T)
            if i >= index_rev_start
                local_i = i - index_rev_start + 1
                delta_g = reverse_delta_g_thermo(vk_rev_colptr, vk_rev_rowval, vk_rev_nzval, nasa_low, nasa_high, T, local_i, n_species)
                k_rev = k_eff / exp(-delta_g + gpu_log(one_atm / R / T) * vk_sum[local_i])
                reverse = k_rev * sparse_product_concentration(product_colptr, product_rowval, product_nzval, Y, p, T, R, MW, mean_MW, reactor, local_i)
            end
            @inbounds q[i, reactor] = forward - reverse
        end

        dT = zero(T)
        for species in 1:n_species
            total = zero(T)
            for j in vk_colptr[species]:vk_colptr[species + 1] - 1
                @inbounds total += vk_nzval[j] * q[vk_rowval[j], reactor]
            end
            _, h_RT, _ = thermo_h_s(nasa_low, nasa_high, T, species)
            @inbounds begin
                dYdt[species, reactor] = total * MW[species] / rho_mass
                dT -= h_RT * total
            end
        end
        @inbounds dTdt[reactor] = dT * R * T / cp_mass / rho_mass
    end
end

@kernel function batched_isobaric_rhs_kernel!(
    dYdt,
    dTdt,
    Y,
    Tvec,
    pvec,
    MW,
    nasa_low,
    nasa_high,
    C,
    h_RT,
    g_RT,
    kf,
    kr,
    arrhenius_coeffs,
    reactant_colptr,
    reactant_rowval,
    reactant_nzval,
    product_colptr,
    product_rowval,
    product_nzval,
    vk_colptr,
    vk_rowval,
    vk_nzval,
    vk_rev_colptr,
    vk_rev_rowval,
    vk_rev_nzval,
    vk_sum,
    tb_colptr,
    tb_rowval,
    tb_nzval,
    fo_colptr,
    fo_rowval,
    fo_nzval,
    arrhenius_0,
    troe,
    pdep_pressures,
    pdep_arrhenius,
    pdep_ptr,
    pdep_index,
    pdep_is_linear_burke,
    linear_burke_efficiencies,
    n_species::Int,
    n_reactions::Int,
    n_pdep::Int,
    index_rev_start::Int,
    index_three_body_start::Int,
    index_falloff_start::Int,
    index_troe_start::Int,
    n_reactors::Int,
)
    reactor = @index(Global)
    if reactor <= n_reactors
        T = Tvec[reactor]
        p = pvec[reactor]
        R = oftype(T, Arrhenius.R)
        one_atm = oftype(T, Arrhenius.one_atm)
        mean_MW_inv = zero(T)
        for i in 1:n_species
            @inbounds mean_MW_inv += Y[i, reactor] / MW[i]
        end
        mean_MW = inv(mean_MW_inv)
        rho_mass = p / R / T * mean_MW

        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        invT = inv(T)
        logT = gpu_log(T)
        rec_RcalT = oftype(T, 4184.0) / R / T
        cp_mass_sum = zero(T)
        for i in 1:n_species
            @inbounds begin
                Xi = Y[i, reactor] / MW[i] * mean_MW
                C[i, reactor] = Xi * rho_mass / mean_MW
                use_low = T <= oftype(T, 1000.0)
                a1 = use_low ? nasa_low[i, 1] : nasa_high[i, 1]
                a2 = use_low ? nasa_low[i, 2] : nasa_high[i, 2]
                a3 = use_low ? nasa_low[i, 3] : nasa_high[i, 3]
                a4 = use_low ? nasa_low[i, 4] : nasa_high[i, 4]
                a5 = use_low ? nasa_low[i, 5] : nasa_high[i, 5]
                a6 = use_low ? nasa_low[i, 6] : nasa_high[i, 6]
                a7 = use_low ? nasa_low[i, 7] : nasa_high[i, 7]
                cpi = a1 + a2 * T + a3 * T2 + a4 * T3 + a5 * T4
                hi = a1 + a2 * T / 2 + a3 * T2 / 3 + a4 * T3 / 4 + a5 * T4 / 5 + a6 * invT
                si = a1 * logT + a2 * T + a3 * T2 / 2 + a4 * T3 / 3 + a5 * T4 / 4 + a7
                h_RT[i, reactor] = hi
                g_RT[i, reactor] = hi - si
                cp_mass_sum += cpi * Xi
            end
        end
        cp_mass = cp_mass_sum * R / mean_MW

        for i in 1:n_reactions
            @inbounds begin
                kf[i, reactor] = arrhenius_rate(arrhenius_coeffs, i, logT, rec_RcalT)
                kr[i, reactor] = zero(T)
            end
        end

        for pdep_i in 1:n_pdep
            reaction_index = pdep_index[pdep_i]
            first = pdep_ptr[pdep_i]
            last = pdep_ptr[pdep_i + 1] - 1
            pressure_scale = R * T / one_atm
            weighted_concentration = zero(T)
            if pdep_is_linear_burke[pdep_i]
                for species in 1:n_species
                    @inbounds begin
                        A = linear_burke_efficiencies[species, pdep_i, 1]
                        b = linear_burke_efficiencies[species, pdep_i, 2]
                        Ea = linear_burke_efficiencies[species, pdep_i, 3]
                        weighted_concentration += C[species, reactor] * A * exp(b * logT - Ea * rec_RcalT)
                    end
                end
            else
                for species in 1:n_species
                    @inbounds weighted_concentration += C[species, reactor]
                end
            end
            pressure_atm = pressure_scale * weighted_concentration
            @inbounds kf[reaction_index, reactor] =
                pressure_dependent_rate(pdep_pressures, pdep_arrhenius, first, last, pressure_atm, logT, rec_RcalT)
        end

        for i in index_three_body_start:index_falloff_start-1
            col = i - index_three_body_start + 1
            @inbounds kf[i, reactor] *= sparse_dot(tb_colptr, tb_rowval, tb_nzval, C, reactor, col)
        end

        for i in index_falloff_start:n_reactions
            col = i - index_falloff_start + 1
            k0 = arrhenius_rate(arrhenius_0, col, logT, rec_RcalT)
            k0 *= sparse_dot(fo_colptr, fo_rowval, fo_nzval, C, reactor, col)
            @inbounds begin
                kr[i, reactor] = gpu_positive(k0 / kf[i, reactor])
                kf[i, reactor] *= kr[i, reactor] / (one(T) + kr[i, reactor])
            end
        end

        for i in index_troe_start:n_reactions
            troe_i = i - index_troe_start + 1
            @inbounds begin
                logPr = gpu_log10(kr[i, reactor])
                F_cent =
                    (one(T) - troe[troe_i, 1]) * exp(-T / troe[troe_i, 4]) +
                    troe[troe_i, 1] * exp(-T / troe[troe_i, 2]) +
                    exp(-troe[troe_i, 3] / T)
                lF_cent = gpu_log10(F_cent)
                c = oftype(T, -0.4) - oftype(T, 0.67) * lF_cent
                N = oftype(T, 0.75) - oftype(T, 1.27) * lF_cent
                f1 = (logPr + c) / (N - oftype(T, 0.14) * (logPr + c))
                kf[i, reactor] *= exp(gpu_log(oftype(T, 10.0)) * lF_cent / (one(T) + f1^2))
            end
        end

        for i in index_rev_start:n_reactions
            local_i = i - index_rev_start + 1
            delta_g = reverse_delta_g(vk_rev_colptr, vk_rev_rowval, vk_rev_nzval, g_RT, reactor, local_i, n_species)
            @inbounds kr[i, reactor] = kf[i, reactor] / exp(-delta_g + gpu_log(one_atm / R / T) * vk_sum[local_i])
        end

        for i in 1:n_reactions
            @inbounds kf[i, reactor] *= sparse_product(reactant_colptr, reactant_rowval, reactant_nzval, C, reactor, i)
        end
        for i in index_rev_start:n_reactions
            col = i - index_rev_start + 1
            @inbounds kr[i, reactor] *= sparse_product(product_colptr, product_rowval, product_nzval, C, reactor, col)
        end
        for i in 1:n_reactions
            @inbounds kf[i, reactor] -= kr[i, reactor]
        end

        dT = zero(T)
        for species in 1:n_species
            total = zero(T)
            for j in vk_colptr[species]:vk_colptr[species + 1] - 1
                @inbounds total += vk_nzval[j] * kf[vk_rowval[j], reactor]
            end
            @inbounds begin
                dYdt[species, reactor] = total * MW[species] / rho_mass
                dT -= h_RT[species, reactor] * total
            end
        end
        @inbounds dTdt[reactor] = dT * R * T / cp_mass / rho_mass
    end
end

@kernel function batched_isobaric_rhs_simple_kernel!(
    dYdt,
    dTdt,
    Y,
    Tvec,
    pvec,
    MW,
    nasa_low,
    nasa_high,
    q,
    arrhenius_coeffs,
    reactant_colptr,
    reactant_rowval,
    reactant_nzval,
    product_colptr,
    product_rowval,
    product_nzval,
    vk_colptr,
    vk_rowval,
    vk_nzval,
    vk_rev_colptr,
    vk_rev_rowval,
    vk_rev_nzval,
    vk_sum,
    n_species::Int,
    n_reactions::Int,
    index_rev_start::Int,
    n_reactors::Int,
)
    reactor = @index(Global)
    if reactor <= n_reactors
        T = Tvec[reactor]
        p = pvec[reactor]
        R = oftype(T, Arrhenius.R)
        one_atm = oftype(T, Arrhenius.one_atm)
        logT = gpu_log(T)
        rec_RcalT = oftype(T, 4184.0) / R / T

        mean_MW_inv = zero(T)
        cp_mass_sum = zero(T)
        for species in 1:n_species
            @inbounds mean_MW_inv += Y[species, reactor] / MW[species]
        end
        mean_MW = inv(mean_MW_inv)
        rho_mass = p / R / T * mean_MW
        for species in 1:n_species
            cp_R, _, _ = thermo_h_s(nasa_low, nasa_high, T, species)
            @inbounds cp_mass_sum += cp_R * Y[species, reactor] / MW[species] * mean_MW
        end
        cp_mass = cp_mass_sum * R / mean_MW

        for i in 1:n_reactions
            k_eff = arrhenius_rate(arrhenius_coeffs, i, logT, rec_RcalT)
            forward = k_eff * sparse_product_concentration(reactant_colptr, reactant_rowval, reactant_nzval, Y, p, T, R, MW, mean_MW, reactor, i)
            reverse = zero(T)
            if i >= index_rev_start
                local_i = i - index_rev_start + 1
                delta_g = reverse_delta_g_thermo(vk_rev_colptr, vk_rev_rowval, vk_rev_nzval, nasa_low, nasa_high, T, local_i, n_species)
                k_rev = k_eff / exp(-delta_g + gpu_log(one_atm / R / T) * vk_sum[local_i])
                reverse = k_rev * sparse_product_concentration(product_colptr, product_rowval, product_nzval, Y, p, T, R, MW, mean_MW, reactor, local_i)
            end
            @inbounds q[i, reactor] = forward - reverse
        end

        dT = zero(T)
        for species in 1:n_species
            total = zero(T)
            for j in vk_colptr[species]:vk_colptr[species + 1] - 1
                @inbounds total += vk_nzval[j] * q[vk_rowval[j], reactor]
            end
            _, h_RT, _ = thermo_h_s(nasa_low, nasa_high, T, species)
            @inbounds begin
                dYdt[species, reactor] = total * MW[species] / rho_mass
                dT -= h_RT * total
            end
        end
        @inbounds dTdt[reactor] = dT * R * T / cp_mass / rho_mass
    end
end

function Arrhenius.batched_isobaric_rhs!(
    dYdt,
    dTdt,
    packed::Arrhenius.NoPdepPackedBatchedMechanism,
    Y,
    T,
    p,
    backend::KernelAbstractions.Backend;
    ndrange = size(Y, 2),
    workgroupsize = 64,
)
    kernel = batched_isobaric_rhs_packed_no_pdep_kernel!(backend, workgroupsize)
    event = kernel(
        dYdt,
        dTdt,
        Y,
        T,
        p,
        packed.realbuf,
        packed.intbuf,
        packed.q;
        ndrange = ndrange,
    )
    sync_backend(backend, event)
    return nothing
end

function Arrhenius.batched_isobaric_rhs!(
    dYdt,
    dTdt,
    batch::Arrhenius.BatchedIdealGasReactor,
    Y,
    T,
    p,
    backend::KernelAbstractions.Backend;
    ndrange = size(Y, 2),
    workgroupsize = 64,
)
    reactions = batch.kinetics
    if isempty(reactions.index_pressure_dependent) && reactions.index_three_body_start > size(batch.kf, 1)
        kernel = batched_isobaric_rhs_simple_kernel!(backend, workgroupsize)
        event = kernel(
            dYdt,
            dTdt,
            Y,
            T,
            p,
            batch.MW,
            batch.nasa_low,
            batch.nasa_high,
            batch.kf,
            reactions.arrhenius_coeffs,
            reactions.reactant_orders.colptr,
            reactions.reactant_orders.rowval,
            reactions.reactant_orders.nzval,
            reactions.product_stoich_coeffs.colptr,
            reactions.product_stoich_coeffs.rowval,
            reactions.product_stoich_coeffs.nzval,
            reactions.vk.colptr,
            reactions.vk.rowval,
            reactions.vk.nzval,
            reactions.vk_rev.colptr,
            reactions.vk_rev.rowval,
            reactions.vk_rev.nzval,
            reactions.vk_sum,
            size(Y, 1),
            size(batch.kf, 1),
            reactions.index_rev_start,
            ndrange;
            ndrange = ndrange,
        )
        sync_backend(backend, event)
        return nothing
    end

    if isempty(reactions.index_pressure_dependent)
        kernel = batched_isobaric_rhs_no_pdep_kernel!(backend, workgroupsize)
        event = kernel(
            dYdt,
            dTdt,
            Y,
            T,
            p,
            batch.MW,
            batch.nasa_low,
            batch.nasa_high,
            batch.kf,
            reactions.arrhenius_coeffs,
            reactions.reactant_orders.colptr,
            reactions.reactant_orders.rowval,
            reactions.reactant_orders.nzval,
            reactions.product_stoich_coeffs.colptr,
            reactions.product_stoich_coeffs.rowval,
            reactions.product_stoich_coeffs.nzval,
            reactions.vk.colptr,
            reactions.vk.rowval,
            reactions.vk.nzval,
            reactions.vk_rev.colptr,
            reactions.vk_rev.rowval,
            reactions.vk_rev.nzval,
            reactions.vk_sum,
            reactions.efficiency_coeffs_tB.colptr,
            reactions.efficiency_coeffs_tB.rowval,
            reactions.efficiency_coeffs_tB.nzval,
            reactions.efficiency_coeffs_fO.colptr,
            reactions.efficiency_coeffs_fO.rowval,
            reactions.efficiency_coeffs_fO.nzval,
            reactions.Arrhenius_0,
            reactions.troe,
            size(Y, 1),
            size(batch.kf, 1),
            reactions.index_rev_start,
            reactions.index_three_body_start,
            reactions.index_fallOff_start,
            reactions.index_troe_start,
            ndrange;
            ndrange = ndrange,
        )
        sync_backend(backend, event)
        return nothing
    end

    kernel = batched_isobaric_rhs_kernel!(backend, workgroupsize)
    event = kernel(
        dYdt,
        dTdt,
        Y,
        T,
        p,
        batch.MW,
        batch.nasa_low,
        batch.nasa_high,
        batch.C,
        batch.h_RT,
        batch.g_RT,
        batch.kf,
        batch.kr,
        reactions.arrhenius_coeffs,
        reactions.reactant_orders.colptr,
        reactions.reactant_orders.rowval,
        reactions.reactant_orders.nzval,
        reactions.product_stoich_coeffs.colptr,
        reactions.product_stoich_coeffs.rowval,
        reactions.product_stoich_coeffs.nzval,
        reactions.vk.colptr,
        reactions.vk.rowval,
        reactions.vk.nzval,
        reactions.vk_rev.colptr,
        reactions.vk_rev.rowval,
        reactions.vk_rev.nzval,
        reactions.vk_sum,
        reactions.efficiency_coeffs_tB.colptr,
        reactions.efficiency_coeffs_tB.rowval,
        reactions.efficiency_coeffs_tB.nzval,
        reactions.efficiency_coeffs_fO.colptr,
        reactions.efficiency_coeffs_fO.rowval,
        reactions.efficiency_coeffs_fO.nzval,
        reactions.Arrhenius_0,
        reactions.troe,
        reactions.pressure_dependent_pressures,
        reactions.pressure_dependent_arrhenius,
        reactions.pressure_dependent_ptr,
        reactions.index_pressure_dependent,
        reactions.pressure_dependent_is_linear_burke,
        reactions.linear_burke_efficiencies,
        size(Y, 1),
        size(batch.kf, 1),
        length(reactions.index_pressure_dependent),
        reactions.index_rev_start,
        reactions.index_three_body_start,
        reactions.index_fallOff_start,
        reactions.index_troe_start,
        ndrange;
        ndrange = ndrange,
    )
    sync_backend(backend, event)
    return nothing
end

end
