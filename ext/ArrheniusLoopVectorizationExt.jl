module ArrheniusLoopVectorizationExt

using Arrhenius
using Arrhenius: TurboBatchedIdealGasReactor, TurboBatchedBuffers
using LoopVectorization

const R_ = Arrhenius.R
const one_atm_ = Arrhenius.one_atm

function thermo_all!(X, C, h_RT, s0_R, g_RT, cp_R, cp_mass, mean_MW, rho_mass,
        mech::TurboBatchedIdealGasReactor, Y, Tvec, pvec)
    n_reactors, n_species = size(Y)
    MW = mech.MW
    nasa_low = mech.nasa_low
    nasa_high = mech.nasa_high

    @inbounds for r in 1:n_reactors
        mean_MW[r] = 0.0
    end
    @inbounds for i in 1:n_species
        invMWi = inv(MW[i])
        @simd for r in 1:n_reactors
            mean_MW[r] += Y[r, i] * invMWi
        end
    end
    @inbounds @simd for r in 1:n_reactors
        mean_MW[r] = inv(mean_MW[r])
        rho_mass[r] = pvec[r] / R_ / Tvec[r] * mean_MW[r]
    end

    @inbounds @simd for r in 1:n_reactors
        cp_mass[r] = 0.0
    end

    @inbounds for i in 1:n_species
        a1l, a2l, a3l, a4l, a5l, a6l, a7l = nasa_low[i, 1], nasa_low[i, 2], nasa_low[i, 3],
            nasa_low[i, 4], nasa_low[i, 5], nasa_low[i, 6], nasa_low[i, 7]
        a1h, a2h, a3h, a4h, a5h, a6h, a7h = nasa_high[i, 1], nasa_high[i, 2], nasa_high[i, 3],
            nasa_high[i, 4], nasa_high[i, 5], nasa_high[i, 6], nasa_high[i, 7]
        invMWi = inv(MW[i])
        @simd for r in 1:n_reactors
            T = Tvec[r]
            low = T <= 1000.0
            a1 = ifelse(low, a1l, a1h); a2 = ifelse(low, a2l, a2h)
            a3 = ifelse(low, a3l, a3h); a4 = ifelse(low, a4l, a4h)
            a5 = ifelse(low, a5l, a5h); a6 = ifelse(low, a6l, a6h)
            a7 = ifelse(low, a7l, a7h)
            T2 = T * T; T3 = T2 * T; T4 = T2 * T2
            invT = inv(T)
            logT = log(T)
            X_i = Y[r, i] * invMWi * mean_MW[r]
            X[r, i] = X_i
            C[r, i] = X_i * rho_mass[r] / mean_MW[r]
            cp_R_i = a1 + a2 * T + a3 * T2 + a4 * T3 + a5 * T4
            h_RT_i = a1 + a2 * T / 2 + a3 * T2 / 3 + a4 * T3 / 4 + a5 * T4 / 5 + a6 * invT
            s0_R_i = a1 * logT + a2 * T + a3 * T2 / 2 + a4 * T3 / 3 + a5 * T4 / 4 + a7
            cp_R[r, i] = cp_R_i
            h_RT[r, i] = h_RT_i
            s0_R[r, i] = s0_R_i
            g_RT[r, i] = h_RT_i - s0_R_i
            cp_mass[r] += cp_R_i * X_i
        end
    end
    @inbounds @simd for r in 1:n_reactors
        cp_mass[r] *= R_ / mean_MW[r]
    end
    return nothing
end

"turbo-accelerated dense Arrhenius rate loop across reactors -- the piece plain `@simd` can't vectorize (transcendentals)"
@inline function arrhenius_all!(kf_col, A, b, Ea, logT, rec_RcalT, n_reactors)
    @turbo for r in 1:n_reactors
        kf_col[r] = A * exp(b * logT[r] - Ea * rec_RcalT[r])
    end
end

function wdot_all!(wdot, kf, kr, C, g_RT, Tvec, logT, rec_RcalT, mech::TurboBatchedIdealGasReactor, M, delta_g, logP)
    reactions = mech.kinetics
    n_reactors = size(C, 1)
    n_reactions = mech.n_reactions
    arr = reactions.arrhenius_coeffs

    @inbounds for i in 1:n_reactions
        arrhenius_all!(@view(kf[:, i]), arr[i, 1], arr[i, 2], arr[i, 3], logT, rec_RcalT, n_reactors)
    end

    # third body (pure three-body reactions)
    @inbounds for i in reactions.index_three_body_start:reactions.index_fallOff_start-1
        col = i - reactions.index_three_body_start + 1
        @simd for r in 1:n_reactors
            M[r] = 0.0
        end
        for j in reactions.efficiency_coeffs_tB.colptr[col]:reactions.efficiency_coeffs_tB.colptr[col+1]-1
            sp = reactions.efficiency_coeffs_tB.rowval[j]
            coeff = reactions.efficiency_coeffs_tB.nzval[j]
            @simd for r in 1:n_reactors
                M[r] += coeff * C[r, sp]
            end
        end
        @simd for r in 1:n_reactors
            kf[r, i] *= M[r]
        end
    end

    # falloff (Lindemann + Troe)
    @inbounds for i in reactions.index_fallOff_start:n_reactions
        col = i - reactions.index_fallOff_start + 1
        @simd for r in 1:n_reactors
            M[r] = 0.0
        end
        for j in reactions.efficiency_coeffs_fO.colptr[col]:reactions.efficiency_coeffs_fO.colptr[col+1]-1
            sp = reactions.efficiency_coeffs_fO.rowval[j]
            coeff = reactions.efficiency_coeffs_fO.nzval[j]
            @simd for r in 1:n_reactors
                M[r] += coeff * C[r, sp]
            end
        end
        A0, b0, Ea0 = reactions.Arrhenius_0[col, 1], reactions.Arrhenius_0[col, 2], reactions.Arrhenius_0[col, 3]
        is_troe = i >= reactions.index_troe_start
        troe_row = is_troe ? i - reactions.index_troe_start + 1 : 0
        if is_troe
            trA, trT3, trT1, trT2 = reactions.troe[troe_row, 1], reactions.troe[troe_row, 2],
                reactions.troe[troe_row, 3], reactions.troe[troe_row, 4]
        end
        @simd for r in 1:n_reactors
            k0 = A0 * exp(b0 * logT[r] - Ea0 * rec_RcalT[r])
            k_inf = kf[r, i]
            Pr = Arrhenius.positive_rate_argument(k0 * M[r] / k_inf)
            lindemann = k_inf * Pr / (1 + Pr)
            factor = 1.0
            if is_troe
                T = Tvec[r]
                F_cent = (1 - trA) * exp(-T / trT2) + trA * exp(-T / trT3) + exp(-trT1 / T)
                F_cent = Arrhenius.positive_rate_argument(F_cent)
                lF_cent = log10(F_cent)
                c = -0.4 - 0.67 * lF_cent
                n = 0.75 - 1.27 * lF_cent
                lPr = log10(Arrhenius.positive_rate_argument(Pr))
                x = lPr + c
                f1 = x / (n - 0.14 * x)
                factor = exp(log(10.0) * lF_cent / (1 + f1^2))
            end
            kf[r, i] = lindemann * factor
        end
    end

    # reversible rate constants (vk_rev is reaction-major, see BatchedReactor.jl's
    # _batch_transposed_packed_sparse_columns)
    @inbounds @simd for r in 1:n_reactors
        logP[r] = log(one_atm_ / R_ / Tvec[r])
    end
    @inbounds for i in reactions.index_rev_start:n_reactions
        local_i = i - reactions.index_rev_start + 1
        @simd for r in 1:n_reactors
            delta_g[r] = 0.0
        end
        for j in reactions.vk_rev.colptr[local_i]:reactions.vk_rev.colptr[local_i+1]-1
            sp = reactions.vk_rev.rowval[j]
            coeff = reactions.vk_rev.nzval[j]
            @simd for r in 1:n_reactors
                delta_g[r] += coeff * g_RT[r, sp]
            end
        end
        vksum = reactions.vk_sum[local_i]
        @simd for r in 1:n_reactors
            kr[r, i] = kf[r, i] / exp(-delta_g[r] + logP[r] * vksum)
        end
    end
    @inbounds for i in 1:(reactions.index_rev_start-1)
        @simd for r in 1:n_reactors
            kr[r, i] = 0.0
        end
    end

    # reactant-order concentration products (forward)
    @inbounds for i in 1:n_reactions
        for j in reactions.reactant_orders.colptr[i]:reactions.reactant_orders.colptr[i+1]-1
            sp = reactions.reactant_orders.rowval[j]
            order = reactions.reactant_orders.nzval[j]
            @simd for r in 1:n_reactors
                kf[r, i] *= C[r, sp]^order
            end
        end
    end
    # product-order concentration products (reverse)
    @inbounds for i in reactions.index_rev_start:n_reactions
        local_i = i - reactions.index_rev_start + 1
        for j in reactions.product_stoich_coeffs.colptr[local_i]:reactions.product_stoich_coeffs.colptr[local_i+1]-1
            sp = reactions.product_stoich_coeffs.rowval[j]
            order = reactions.product_stoich_coeffs.nzval[j]
            @simd for r in 1:n_reactors
                kr[r, i] *= C[r, sp]^order
            end
        end
    end

    @inbounds for i in 1:n_reactions
        @simd for r in 1:n_reactors
            kf[r, i] -= kr[r, i]
        end
    end

    @inbounds for sp in 1:mech.n_species
        @simd for r in 1:n_reactors
            wdot[r, sp] = 0.0
        end
    end
    @inbounds for sp in 1:mech.n_species
        for j in reactions.vk.colptr[sp]:reactions.vk.colptr[sp+1]-1
            rxn = reactions.vk.rowval[j]
            coeff = reactions.vk.nzval[j]
            @simd for r in 1:n_reactors
                wdot[r, sp] += coeff * kf[r, rxn]
            end
        end
    end
    return nothing
end

function Arrhenius.turbo_batched_isobaric_rhs!(dYdt, dTdt, mech::TurboBatchedIdealGasReactor, Y, Tvec, pvec, buffers::TurboBatchedBuffers)
    n_reactors, n_species = size(Y)
    size(dYdt) == size(Y) || throw(DimensionMismatch("dYdt and Y must have the same shape"))
    length(dTdt) == n_reactors || throw(DimensionMismatch("dTdt length must match reactor count"))
    length(Tvec) == n_reactors || throw(DimensionMismatch("T length must match reactor count"))
    length(pvec) == n_reactors || throw(DimensionMismatch("p length must match reactor count"))

    X, C, h_RT, s0_R, g_RT, cp_R = buffers.X, buffers.C, buffers.h_RT, buffers.s0_R, buffers.g_RT, buffers.cp_R
    cp_mass, mean_MW, rho_mass = buffers.cp_mass, buffers.mean_MW, buffers.rho_mass
    kf, kr, wdot = buffers.kf, buffers.kr, buffers.wdot
    logT, rec_RcalT = buffers.logT, buffers.rec_RcalT
    Mbuf, delta_g_buf, logPbuf = buffers.Mbuf, buffers.delta_g_buf, buffers.logPbuf

    thermo_all!(X, C, h_RT, s0_R, g_RT, cp_R, cp_mass, mean_MW, rho_mass, mech, Y, Tvec, pvec)
    @inbounds @simd for r in 1:n_reactors
        logT[r] = log(Tvec[r])
        rec_RcalT[r] = 4184.0 / R_ / Tvec[r]
    end
    wdot_all!(wdot, kf, kr, C, g_RT, Tvec, logT, rec_RcalT, mech, Mbuf, delta_g_buf, logPbuf)

    @inbounds @simd for r in 1:n_reactors
        dTdt[r] = 0.0
    end
    @inbounds for sp in 1:n_species
        MWsp = mech.MW[sp]
        @simd for r in 1:n_reactors
            dYdt[r, sp] = wdot[r, sp] * MWsp / rho_mass[r]
            dTdt[r] -= h_RT[r, sp] * wdot[r, sp]
        end
    end
    @inbounds @simd for r in 1:n_reactors
        dTdt[r] = dTdt[r] * R_ * Tvec[r] / cp_mass[r] / rho_mass[r]
    end
    return nothing
end

end
