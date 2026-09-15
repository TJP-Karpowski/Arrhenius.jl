struct BatchedIdealGasReactor{K,MT,VT}
    kinetics::K
    MW::VT
    nasa_low::MT
    nasa_high::MT
    X::MT
    C::MT
    h_RT::MT
    s0_R::MT
    g_RT::MT
    cp_R::MT
    wdot::MT
    kf::MT
    kr::MT
end

"""
    NoPdepPackedBatchedMechanism(gas, n_reactors; array=Array, real_type=Float32, int_type=Int64)

Pack an ideal-gas mechanism into a small number of real and integer buffers for
batched backend RHS evaluation. This experimental wrapper is intended for
device backends with strict kernel argument limits, such as Metal.

The current packed path supports mechanisms without pressure-dependent rate
tables. It covers ordinary Arrhenius, reversible, third-body, falloff, and Troe
reactions. Pressure-dependent Arrhenius and `linear-Burke` tables are rejected
by this wrapper until their packed device path is implemented.
"""
struct NoPdepPackedBatchedMechanism{RB,IB,QB,O}
    realbuf::RB
    intbuf::IB
    q::QB
    offsets::O
    n_species::Int
    n_reactions::Int
    index_rev_start::Int
    index_three_body_start::Int
    index_fallOff_start::Int
    index_troe_start::Int
end

_batch_array(::Type{Array}, x) = x
_batch_array(array, x) = array(x)
_batch_real_array(array, x, ::Type{T}) where {T} = _batch_array(array, T.(x))

"""
    _batch_transposed_packed_sparse_columns(array, A, real_type)

`A` (e.g. `reactions.vk_rev`) is stored row-major-by-reaction /
column-major-by-species, because that orientation is what `packed_mul_offset!`
needs for the O(nnz) sparse mat-vec in `PackedKinetics.kr!`. The batched
per-cell path instead looks up "all species in reaction `r`" one reaction at a
time (`_sparse_column_dot`), which needs the transposed (species-row,
reaction-column) layout for O(nnz-per-reaction) access. Without transposing,
each lookup degenerates into an O(nnz(A)) scan of every column for a row
match -- O(n_reactions * nnz(A)) total per cell, confirmed ~13-16x slower than
the scalar `wdot_func!` path for GRI30/NH3-Stagni-sized mechanisms.
"""
function _batch_transposed_packed_sparse_columns(array, A::PackedSparseColumns, ::Type{T}) where {T}
    csc = SparseMatrixCSC{Float64,Int64}(A.nrow, A.ncol, copy(A.colptr), copy(A.rowval), copy(A.nzval))
    transposed = PackedSparseColumns(sparse(csc'))
    return PackedSparseColumns(
        _batch_array(array, copy(transposed.colptr)),
        _batch_array(array, copy(transposed.rowval)),
        _batch_real_array(array, transposed.nzval, T),
        transposed.nrow,
        transposed.ncol,
    )
end

function packed_kinetics_for_batch(reactions::PackedKinetics; array = Array, real_type = Float64)
    return PackedKinetics(
        _batch_real_array(array, reactions.arrhenius_coeffs, real_type),
        PackedSparseColumns(
            _batch_array(array, copy(reactions.reactant_orders.colptr)),
            _batch_array(array, copy(reactions.reactant_orders.rowval)),
            _batch_real_array(array, reactions.reactant_orders.nzval, real_type),
            reactions.reactant_orders.nrow,
            reactions.reactant_orders.ncol,
        ),
        PackedSparseColumns(
            _batch_array(array, copy(reactions.product_stoich_coeffs.colptr)),
            _batch_array(array, copy(reactions.product_stoich_coeffs.rowval)),
            _batch_real_array(array, reactions.product_stoich_coeffs.nzval, real_type),
            reactions.product_stoich_coeffs.nrow,
            reactions.product_stoich_coeffs.ncol,
        ),
        PackedSparseColumns(
            _batch_array(array, copy(reactions.reactant_stoich_coeffs.colptr)),
            _batch_array(array, copy(reactions.reactant_stoich_coeffs.rowval)),
            _batch_real_array(array, reactions.reactant_stoich_coeffs.nzval, real_type),
            reactions.reactant_stoich_coeffs.nrow,
            reactions.reactant_stoich_coeffs.ncol,
        ),
        PackedSparseColumns(
            _batch_array(array, copy(reactions.vk.colptr)),
            _batch_array(array, copy(reactions.vk.rowval)),
            _batch_real_array(array, reactions.vk.nzval, real_type),
            reactions.vk.nrow,
            reactions.vk.ncol,
        ),
        _batch_transposed_packed_sparse_columns(array, reactions.vk_rev, real_type),
        _batch_real_array(array, reactions.vk_sum, real_type),
        PackedSparseColumns(
            _batch_array(array, copy(reactions.efficiency_coeffs_tB.colptr)),
            _batch_array(array, copy(reactions.efficiency_coeffs_tB.rowval)),
            _batch_real_array(array, reactions.efficiency_coeffs_tB.nzval, real_type),
            reactions.efficiency_coeffs_tB.nrow,
            reactions.efficiency_coeffs_tB.ncol,
        ),
        PackedSparseColumns(
            _batch_array(array, copy(reactions.efficiency_coeffs_fO.colptr)),
            _batch_array(array, copy(reactions.efficiency_coeffs_fO.rowval)),
            _batch_real_array(array, reactions.efficiency_coeffs_fO.nzval, real_type),
            reactions.efficiency_coeffs_fO.nrow,
            reactions.efficiency_coeffs_fO.ncol,
        ),
        _batch_real_array(array, reactions.Arrhenius_0, real_type),
        _batch_real_array(array, reactions.troe, real_type),
        _batch_real_array(array, reactions.pressure_dependent_pressures, real_type),
        _batch_real_array(array, reactions.pressure_dependent_arrhenius, real_type),
        _batch_array(array, copy(reactions.pressure_dependent_ptr)),
        _batch_array(array, copy(reactions.index_pressure_dependent)),
        _batch_array(array, copy(reactions.pressure_dependent_is_linear_burke)),
        _batch_real_array(array, reactions.linear_burke_efficiencies, real_type),
        reactions.index_rev_start,
        reactions.index_three_body_start,
        reactions.index_fallOff_start,
        reactions.index_troe_start,
    )
end

packed_kinetics_for_batch(reactions::ReorderedKinetics; array = Array, real_type = Float64) =
    packed_kinetics_for_batch(PackedKinetics(reactions); array = array, real_type = real_type)

"""
    BatchedIdealGasReactor(gas, n_reactors; array=Array, real_type=Float64)

Create a structure-of-arrays workspace for evaluating many independent
isobaric ideal-gas reactor RHS states. Inputs and outputs use shape
`(n_species, n_reactors)` for species fields and `(n_reactors,)` for
temperature/pressure fields.

Pass `array=Metal.MtlArray` or another device-array constructor together with
the KernelAbstractions extension to place the packed mechanism and scratch
arrays on that device.

`real_type=Float32` is experimental and is not generally valid for detailed
mechanisms yet. Raw pre-exponential factors and low-temperature reversible
rate coefficients can exceed the Float32 range even when the final net source
is finite. A robust implementation requires log-domain progress-rate
evaluation; callers should retain Float64 kinetics until that path exists.
"""
function BatchedIdealGasReactor(gas, n_reactors::Integer; array = Array, real_type = Float64)
    n_species = gas.n_species
    n_reactions = gas.n_reactions
    matrix(rows, cols) = _batch_array(array, zeros(real_type, rows, cols))
    return BatchedIdealGasReactor(
        packed_kinetics_for_batch(gas.kinetics; array = array, real_type = real_type),
        _batch_real_array(array, gas.MW, real_type),
        _batch_real_array(array, gas.thermo.nasa_low, real_type),
        _batch_real_array(array, gas.thermo.nasa_high, real_type),
        matrix(n_species, n_reactors),
        matrix(n_species, n_reactors),
        matrix(n_species, n_reactors),
        matrix(n_species, n_reactors),
        matrix(n_species, n_reactors),
        matrix(n_species, n_reactors),
        matrix(n_species, n_reactors),
        matrix(n_reactions, n_reactors),
        matrix(n_reactions, n_reactors),
    )
end

function NoPdepPackedBatchedMechanism(gas, n_reactors::Integer; array = Array, real_type = Float32, int_type = Int64)
    reactions = gas.kinetics isa PackedKinetics ? gas.kinetics : PackedKinetics(gas.kinetics)
    isempty(reactions.index_pressure_dependent) ||
        throw(ArgumentError("NoPdepPackedBatchedMechanism does not support pressure-dependent rate tables yet"))

    realbuf = real_type[]
    metadata_slots = 32
    intbuf = fill(int_type(0), metadata_slots)
    real_offsets = Dict{Symbol,Int}()
    int_offsets = Dict{Symbol,Int}()

    function push_real!(name::Symbol, values)
        real_offsets[name] = length(realbuf) + 1
        append!(realbuf, vec(real_type.(values)))
        return nothing
    end

    function push_int!(name::Symbol, values)
        int_offsets[name] = length(intbuf) + 1
        append!(intbuf, vec(int_type.(values)))
        return nothing
    end

    function push_sparse!(prefix::Symbol, sparse::PackedSparseColumns)
        push_int!(Symbol(prefix, :_colptr), sparse.colptr)
        push_int!(Symbol(prefix, :_rowval), sparse.rowval)
        push_real!(Symbol(prefix, :_nzval), sparse.nzval)
        return nothing
    end

    push_real!(:mw, gas.MW)
    push_real!(:nasa_low, gas.thermo.nasa_low)
    push_real!(:nasa_high, gas.thermo.nasa_high)
    push_real!(:arrhenius, reactions.arrhenius_coeffs)
    push_sparse!(:reactant, reactions.reactant_orders)
    push_sparse!(:product, reactions.product_stoich_coeffs)
    push_sparse!(:vk, reactions.vk)
    push_sparse!(:vkrev, reactions.vk_rev)
    push_real!(:vksum, reactions.vk_sum)
    push_sparse!(:tb, reactions.efficiency_coeffs_tB)
    push_sparse!(:fo, reactions.efficiency_coeffs_fO)
    push_real!(:arrhenius0, reactions.Arrhenius_0)
    push_real!(:troe, reactions.troe)

    offsets = (;
        mw = real_offsets[:mw],
        nasa_low = real_offsets[:nasa_low],
        nasa_high = real_offsets[:nasa_high],
        arrhenius = real_offsets[:arrhenius],
        reactant_colptr = int_offsets[:reactant_colptr],
        reactant_rowval = int_offsets[:reactant_rowval],
        reactant_nzval = real_offsets[:reactant_nzval],
        product_colptr = int_offsets[:product_colptr],
        product_rowval = int_offsets[:product_rowval],
        product_nzval = real_offsets[:product_nzval],
        vk_colptr = int_offsets[:vk_colptr],
        vk_rowval = int_offsets[:vk_rowval],
        vk_nzval = real_offsets[:vk_nzval],
        vkrev_colptr = int_offsets[:vkrev_colptr],
        vkrev_rowval = int_offsets[:vkrev_rowval],
        vkrev_nzval = real_offsets[:vkrev_nzval],
        vksum = real_offsets[:vksum],
        tb_colptr = int_offsets[:tb_colptr],
        tb_rowval = int_offsets[:tb_rowval],
        tb_nzval = real_offsets[:tb_nzval],
        fo_colptr = int_offsets[:fo_colptr],
        fo_rowval = int_offsets[:fo_rowval],
        fo_nzval = real_offsets[:fo_nzval],
        arrhenius0 = real_offsets[:arrhenius0],
        troe = real_offsets[:troe],
    )

    intbuf[1:metadata_slots] .= int_type.([
        gas.n_species,
        gas.n_reactions,
        reactions.index_rev_start,
        reactions.index_three_body_start,
        reactions.index_fallOff_start,
        reactions.index_troe_start,
        n_reactors,
        offsets.mw,
        offsets.nasa_low,
        offsets.nasa_high,
        offsets.arrhenius,
        offsets.reactant_colptr,
        offsets.reactant_rowval,
        offsets.reactant_nzval,
        offsets.product_colptr,
        offsets.product_rowval,
        offsets.product_nzval,
        offsets.vk_colptr,
        offsets.vk_rowval,
        offsets.vk_nzval,
        offsets.vkrev_colptr,
        offsets.vkrev_rowval,
        offsets.vkrev_nzval,
        offsets.vksum,
        offsets.tb_colptr,
        offsets.tb_rowval,
        offsets.tb_nzval,
        offsets.fo_colptr,
        offsets.fo_rowval,
        offsets.fo_nzval,
        offsets.arrhenius0,
        offsets.troe,
    ])

    q = _batch_array(array, zeros(real_type, gas.n_reactions, n_reactors))
    return NoPdepPackedBatchedMechanism(
        _batch_array(array, realbuf),
        _batch_array(array, intbuf),
        q,
        offsets,
        gas.n_species,
        gas.n_reactions,
        reactions.index_rev_start,
        reactions.index_three_body_start,
        reactions.index_fallOff_start,
        reactions.index_troe_start,
    )
end

Base.size(batch::BatchedIdealGasReactor) = size(batch.X)

@inline function _arrhenius_rate(coeffs, row::Int, logT, rec_RcalT)
    return coeffs[row, 1] * exp(coeffs[row, 2] * logT - coeffs[row, 3] * rec_RcalT)
end

@inline function _concentration_sum(C, reactor::Int, n_species::Int)
    total = zero(eltype(C))
    for i in 1:n_species
        @inbounds total += C[i, reactor]
    end
    return total
end

@inline function _linear_burke_efficiency(reactions, species::Int, pdep_index::Int, logT, rec_RcalT)
    A = reactions.linear_burke_efficiencies[species, pdep_index, 1]
    b = reactions.linear_burke_efficiencies[species, pdep_index, 2]
    Ea = reactions.linear_burke_efficiencies[species, pdep_index, 3]
    return A * exp(b * logT - Ea * rec_RcalT)
end

@inline function _effective_pressure_atm_cell(reactions, C, reactor::Int, n_species::Int, T, logT, rec_RcalT, pdep_index::Int)
    pressure_scale = oftype(T, R) * T / oftype(T, one_atm)
    if !reactions.pressure_dependent_is_linear_burke[pdep_index]
        return pressure_scale * _concentration_sum(C, reactor, n_species)
    end
    weighted_concentration = zero(T)
    for species in 1:n_species
        @inbounds weighted_concentration +=
            C[species, reactor] * _linear_burke_efficiency(reactions, species, pdep_index, logT, rec_RcalT)
    end
    return pressure_scale * weighted_concentration
end

@inline function _pressure_dependent_rate_cell(reactions, first::Int, last::Int, pressure_atm, logT, rec_RcalT)
    first == last && return _arrhenius_rate(reactions.pressure_dependent_arrhenius, first, logT, rec_RcalT)

    p = positive_rate_argument(pressure_atm)
    if p <= reactions.pressure_dependent_pressures[first]
        return _arrhenius_rate(reactions.pressure_dependent_arrhenius, first, logT, rec_RcalT)
    elseif p >= reactions.pressure_dependent_pressures[last]
        return _arrhenius_rate(reactions.pressure_dependent_arrhenius, last, logT, rec_RcalT)
    end

    logp = safe_log(p)
    for i in first:last-1
        p_low = reactions.pressure_dependent_pressures[i]
        p_high = reactions.pressure_dependent_pressures[i + 1]
        if p >= p_low && p <= p_high
            k_low = _arrhenius_rate(reactions.pressure_dependent_arrhenius, i, logT, rec_RcalT)
            k_high = _arrhenius_rate(reactions.pressure_dependent_arrhenius, i + 1, logT, rec_RcalT)
            weight = (logp - safe_log(p_low)) / (safe_log(p_high) - safe_log(p_low))
            return exp((1 - weight) * safe_log(k_low) + weight * safe_log(k_high))
        end
    end
    return _arrhenius_rate(reactions.pressure_dependent_arrhenius, last, logT, rec_RcalT)
end

@inline function _sparse_column_product(A::PackedSparseColumns, values, reactor::Int, col::Int)
    product = one(eltype(values))
    for j in A.colptr[col]:A.colptr[col + 1] - 1
        @inbounds product *= values[A.rowval[j], reactor]^A.nzval[j]
    end
    return product
end

@inline function _sparse_column_dot(A::PackedSparseColumns, values, reactor::Int, col::Int)
    total = zero(eltype(values))
    for j in A.colptr[col]:A.colptr[col + 1] - 1
        @inbounds total += A.nzval[j] * values[A.rowval[j], reactor]
    end
    return total
end

@inline function _reverse_delta_g(A::PackedSparseColumns, g_RT, reactor::Int, local_reaction::Int)
    return _sparse_column_dot(A, g_RT, reactor, local_reaction)
end

function batched_wdot_cell!(batch::BatchedIdealGasReactor, reactor::Int, T,
        multipliers=nothing)
    reactions = batch.kinetics
    C = batch.C
    g_RT = batch.g_RT
    wdot = batch.wdot
    kf = batch.kf
    kr = batch.kr
    n_species = size(C, 1)
    n_reactions = size(kf, 1)
    logT = safe_log(T)
    rec_RcalT = oftype(T, 4184.0) / oftype(T, R) / T

    for i in 1:n_species
        @inbounds wdot[i, reactor] = zero(eltype(wdot))
    end
    for i in 1:n_reactions
        @inbounds begin
            kf[i, reactor] = _arrhenius_rate(reactions.arrhenius_coeffs, i, logT, rec_RcalT)
            kr[i, reactor] = zero(eltype(kr))
        end
    end

    for pdep_i in eachindex(reactions.index_pressure_dependent)
        reaction_index = reactions.index_pressure_dependent[pdep_i]
        first = reactions.pressure_dependent_ptr[pdep_i]
        last = reactions.pressure_dependent_ptr[pdep_i + 1] - 1
        pressure_atm = _effective_pressure_atm_cell(reactions, C, reactor, n_species, T, logT, rec_RcalT, pdep_i)
        @inbounds kf[reaction_index, reactor] =
            _pressure_dependent_rate_cell(reactions, first, last, pressure_atm, logT, rec_RcalT)
    end

    for i in reactions.index_three_body_start:reactions.index_fallOff_start-1
        col = i - reactions.index_three_body_start + 1
        @inbounds kf[i, reactor] *= _sparse_column_dot(reactions.efficiency_coeffs_tB, C, reactor, col)
    end

    for i in reactions.index_fallOff_start:n_reactions
        col = i - reactions.index_fallOff_start + 1
        k0 = _arrhenius_rate(reactions.Arrhenius_0, col, logT, rec_RcalT)
        k0 *= _sparse_column_dot(reactions.efficiency_coeffs_fO, C, reactor, col)
        @inbounds begin
            kr[i, reactor] = positive_rate_argument(k0 / kf[i, reactor])
            kf[i, reactor] *= kr[i, reactor] / (1 + kr[i, reactor])
        end
    end

    for i in reactions.index_troe_start:n_reactions
        troe_i = i - reactions.index_troe_start + 1
        @inbounds begin
            logPr = safe_log10(kr[i, reactor])
            F_cent =
                (1 - reactions.troe[troe_i, 1]) * exp(-T / reactions.troe[troe_i, 4]) +
                reactions.troe[troe_i, 1] * exp(-T / reactions.troe[troe_i, 2]) +
                exp(-reactions.troe[troe_i, 3] / T)
            lF_cent = safe_log10(F_cent)
            c = -0.4 - 0.67 * lF_cent
            N = 0.75 - 1.27 * lF_cent
            f1 = (logPr + c) / (N - 0.14 * (logPr + c))
            kf[i, reactor] *= exp(log(oftype(T, 10.0)) * lF_cent / (one(T) + f1^2))
        end
    end

    for i in reactions.index_rev_start:n_reactions
        local_i = i - reactions.index_rev_start + 1
        delta_g = _reverse_delta_g(reactions.vk_rev, g_RT, reactor, local_i)
        @inbounds kr[i, reactor] = kf[i, reactor] / exp(-delta_g +
            safe_log(oftype(T, one_atm) / oftype(T, R) / T) * reactions.vk_sum[local_i])
    end

    for i in 1:n_reactions
        @inbounds kf[i, reactor] *= _sparse_column_product(reactions.reactant_orders, C, reactor, i)
    end
    for i in reactions.index_rev_start:n_reactions
        col = i - reactions.index_rev_start + 1
        @inbounds kr[i, reactor] *= _sparse_column_product(reactions.product_stoich_coeffs, C, reactor, col)
    end
    for i in 1:n_reactions
        @inbounds kf[i, reactor] -= kr[i, reactor]
    end

    for species in 1:n_species
        total = zero(eltype(wdot))
        for j in reactions.vk.colptr[species]:reactions.vk.colptr[species + 1] - 1
            reaction = reactions.vk.rowval[j]
            multiplier = multipliers === nothing ? one(eltype(wdot)) : multipliers[reaction, reactor]
            @inbounds total += reactions.vk.nzval[j] * kf[reaction, reactor] * multiplier
        end
        @inbounds wdot[species, reactor] = total
    end
    return nothing
end

function batched_isobaric_rhs_cell!(dYdt, dTdt, batch::BatchedIdealGasReactor, Y, Tvec, pvec, reactor::Int)
    return batched_isobaric_rhs_cell!(dYdt, dTdt, batch, Y, Tvec, pvec,
        reactor, nothing)
end

function batched_isobaric_rhs_cell!(dYdt, dTdt, batch::BatchedIdealGasReactor,
        Y, Tvec, pvec, reactor::Int, multipliers)
    T = Tvec[reactor]
    p = pvec[reactor]
    n_species = size(Y, 1)
    mean_MW_inv = zero(T)
    for i in 1:n_species
        @inbounds mean_MW_inv += Y[i, reactor] / batch.MW[i]
    end
    mean_MW = inv(mean_MW_inv)
    rho_mass = p / oftype(T, R) / T * mean_MW

    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    invT = inv(T)
    logT = safe_log(T)
    cp_mass_sum = zero(T)
    for i in 1:n_species
        @inbounds begin
            X_i = Y[i, reactor] / batch.MW[i] * mean_MW
            batch.X[i, reactor] = X_i
            batch.C[i, reactor] = X_i * rho_mass / mean_MW
            a1 = T <= oftype(T, 1000.0) ? batch.nasa_low[i, 1] : batch.nasa_high[i, 1]
            a2 = T <= oftype(T, 1000.0) ? batch.nasa_low[i, 2] : batch.nasa_high[i, 2]
            a3 = T <= oftype(T, 1000.0) ? batch.nasa_low[i, 3] : batch.nasa_high[i, 3]
            a4 = T <= oftype(T, 1000.0) ? batch.nasa_low[i, 4] : batch.nasa_high[i, 4]
            a5 = T <= oftype(T, 1000.0) ? batch.nasa_low[i, 5] : batch.nasa_high[i, 5]
            a6 = T <= oftype(T, 1000.0) ? batch.nasa_low[i, 6] : batch.nasa_high[i, 6]
            a7 = T <= oftype(T, 1000.0) ? batch.nasa_low[i, 7] : batch.nasa_high[i, 7]
            cp_R_i = a1 + a2 * T + a3 * T2 + a4 * T3 + a5 * T4
            h_RT_i = a1 + a2 * T / oftype(T, 2) + a3 * T2 / oftype(T, 3) +
                a4 * T3 / oftype(T, 4) + a5 * T4 / oftype(T, 5) + a6 * invT
            s0_R_i = a1 * logT + a2 * T + a3 * T2 / oftype(T, 2) +
                a4 * T3 / oftype(T, 3) + a5 * T4 / oftype(T, 4) + a7
            batch.cp_R[i, reactor] = cp_R_i
            batch.h_RT[i, reactor] = h_RT_i
            batch.s0_R[i, reactor] = s0_R_i
            batch.g_RT[i, reactor] = h_RT_i - s0_R_i
            cp_mass_sum += cp_R_i * X_i
        end
    end
    cp_mass = cp_mass_sum * oftype(T, R) / mean_MW

    batched_wdot_cell!(batch, reactor, T, multipliers)

    dT = zero(T)
    for i in 1:n_species
        @inbounds begin
            dYdt[i, reactor] = batch.wdot[i, reactor] * batch.MW[i] / rho_mass
            dT -= batch.h_RT[i, reactor] * batch.wdot[i, reactor]
        end
    end
    @inbounds dTdt[reactor] = dT * oftype(T, R) * T / cp_mass / rho_mass
    return nothing
end

"""
    batched_isobaric_rhs!(dYdt, dTdt, batch, Y, T, p)
    batched_isobaric_rhs!(dYdt, dTdt, batch, Y, T, p, backend; ndrange=size(Y, 2), workgroupsize=64)

Evaluate constant-pressure ideal-gas reactor RHS values for many independent
reactors. Species arrays use shape `(n_species, n_reactors)` and temperature
and pressure arrays use length `n_reactors`.

The CPU method accepts `BatchedIdealGasReactor`. Backend execution is provided
by package extensions when `KernelAbstractions` and a backend package are
loaded. The experimental packed backend path accepts
`NoPdepPackedBatchedMechanism` for mechanisms without pressure-dependent rate
tables.
"""
function batched_isobaric_rhs!(dYdt, dTdt, batch::BatchedIdealGasReactor, Y, T, p)
    n_reactors = size(Y, 2)
    size(dYdt) == size(Y) || throw(DimensionMismatch("dYdt and Y must have the same shape"))
    length(dTdt) == n_reactors || throw(DimensionMismatch("dTdt length must match reactor count"))
    length(T) == n_reactors || throw(DimensionMismatch("T length must match reactor count"))
    length(p) == n_reactors || throw(DimensionMismatch("p length must match reactor count"))
    for reactor in 1:n_reactors
        batched_isobaric_rhs_cell!(dYdt, dTdt, batch, Y, T, p, reactor)
    end
    return nothing
end

function batched_isobaric_rhs!(dYdt, dTdt, batch::BatchedIdealGasReactor,
        Y, T, p, multipliers::AbstractMatrix)
    n_reactors = size(Y, 2)
    size(dYdt) == size(Y) || throw(DimensionMismatch("dYdt and Y must have the same shape"))
    size(multipliers) == size(batch.kf) ||
        throw(DimensionMismatch("multipliers must have shape (n_reactions, n_reactors)"))
    length(dTdt) == n_reactors || throw(DimensionMismatch("dTdt length must match reactor count"))
    length(T) == n_reactors || throw(DimensionMismatch("T length must match reactor count"))
    length(p) == n_reactors || throw(DimensionMismatch("p length must match reactor count"))
    for reactor in 1:n_reactors
        batched_isobaric_rhs_cell!(dYdt, dTdt, batch, Y, T, p, reactor, multipliers)
    end
    return nothing
end

function batched_isobaric_rhs!(dYdt, dTdt, batch::BatchedIdealGasReactor, Y, T, p, backend; kwargs...)
    error("KernelAbstractions is required for backend execution. Load KernelAbstractions and a backend package such as Metal.")
end

function batched_isobaric_rhs!(dYdt, dTdt, packed::NoPdepPackedBatchedMechanism, Y, T, p, backend; kwargs...)
    error("KernelAbstractions is required for packed backend execution. Load KernelAbstractions and a backend package such as Metal.")
end

export BatchedIdealGasReactor,
    NoPdepPackedBatchedMechanism,
    batched_isobaric_rhs!,
    batched_isobaric_rhs_cell!,
    packed_kinetics_for_batch
