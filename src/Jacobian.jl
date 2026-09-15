struct IsobaricJacobianWorkspace{Sol<:Arrhenius.Solution}
    gas::Sol
end

mutable struct AnalyticalIsobaricJacobianWorkspace{
    Reactor<:isobaricReactor,
    Mat<:AbstractMatrix,
    Vec<:AbstractVector,
    Schedule,
}
    state::Reactor
    schedule::Schedule
    dwdot_dC::Mat
    dwdot_dT::Vec
    dg_RT_dT::Vec
    dh_RT_dT::Vec
    wdot::Vec
end

struct GeneratedProductionJacobianKernel{ID} end

mutable struct GeneratedIsobaricReactor{
    Reactor<:isobaricReactor,
    Schedule,
    Kernel,
} <: Function
    state::Reactor
    schedule::Schedule
    kernel::Kernel
end

mutable struct GeneratedAnalyticalIsobaricJacobianWorkspace{
    Reactor<:isobaricReactor,
    Mat<:AbstractMatrix,
    Vec<:AbstractVector,
    Schedule,
    Kernel,
}
    state::Reactor
    schedule::Schedule
    kernel::Kernel
    dwdot_dC::Mat
    dwdot_dT::Vec
    dg_RT_dT::Vec
    dh_RT_dT::Vec
    wdot::Vec
end

struct ProductionJacobianSchedule{T<:AbstractFloat}
    reactant_ptr::Vector{Int}
    reactant_species::Vector{Int}
    reactant_orders::Vector{T}
    product_ptr::Vector{Int}
    product_species::Vector{Int}
    product_orders::Vector{T}
    stoich_ptr::Vector{Int}
    stoich_species::Vector{Int}
    stoich_coeffs::Vector{T}
    reverse_ptr::Vector{Int}
    reverse_species::Vector{Int}
    reverse_coeffs::Vector{T}
    efficiency_ptr::Vector{Int}
    efficiency_species::Vector{Int}
    efficiency_coeffs::Vector{T}
end

const GENERATED_WDOT_BODIES = Dict{UInt64,Expr}()
const GENERATED_PRODUCTION_JACOBIAN_BODIES = Dict{UInt64,Expr}()
const GENERATED_PRODUCTION_JACOBIAN_WDOT_BODIES = Dict{UInt64,Expr}()
const GENERATED_PRODUCTION_TEMPERATURE_BODIES = Dict{UInt64,Expr}()
const GENERATED_PRODUCTION_JACOBIAN_SCHEDULES = Dict{UInt64,ProductionJacobianSchedule}()
const GENERATED_PRESSURE_DEPENDENT_REACTION_LIMIT = 128

"""
    IsobaricJacobianWorkspace(gas)

Create a workspace for evaluating the dense Jacobian of the isobaric reactor
RHS with respect to the ODE state vector `[Y..., T]`.
"""
function IsobaricJacobianWorkspace(gas)
    return IsobaricJacobianWorkspace{typeof(gas)}(gas)
end

"""
    AnalyticalIsobaricJacobianWorkspace(gas)

Create a reusable SciML-compatible Jacobian callback workspace backed by the
analytical dense isobaric reactor Jacobian.
"""
function AnalyticalIsobaricJacobianWorkspace(gas)
    return AnalyticalIsobaricJacobianWorkspace(gas, Float64)
end

function AnalyticalIsobaricJacobianWorkspace(gas, ::Type{T}) where {T<:AbstractFloat}
    return AnalyticalIsobaricJacobianWorkspace(
        isobaricReactor(gas, T),
        ProductionJacobianSchedule(gas.kinetics, T),
        zeros(T, gas.n_species, gas.n_species),
        zeros(T, gas.n_species),
        zeros(T, gas.n_species),
        zeros(T, gas.n_species),
        zeros(T, gas.n_species),
    )
end

"""
    GeneratedAnalyticalIsobaricJacobianWorkspace(gas)

Create a reusable Jacobian workspace that uses a mechanism-specialized emitted
kernel for `d(wdot) / dC` and the generic analytical path for the remaining
reactor assembly and temperature derivatives.
"""
function GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
    return GeneratedAnalyticalIsobaricJacobianWorkspace(gas, Float64)
end

function GeneratedAnalyticalIsobaricJacobianWorkspace(gas, ::Type{T}) where {T<:AbstractFloat}
    schedule = ProductionJacobianSchedule(gas.kinetics, T)
    kernel = GeneratedProductionJacobianKernel(gas.kinetics, schedule)
    return GeneratedAnalyticalIsobaricJacobianWorkspace(
        isobaricReactor(gas, T),
        schedule,
        kernel,
        zeros(T, gas.n_species, gas.n_species),
        zeros(T, gas.n_species),
        zeros(T, gas.n_species),
        zeros(T, gas.n_species),
        zeros(T, gas.n_species),
    )
end

"""
    GeneratedIsobaricReactor(gas)

Create an isobaric reactor RHS that uses a mechanism-specialized generated
kernel for the chemistry source term while retaining the standard reactor
state update and temperature assembly.
"""
function GeneratedIsobaricReactor(gas)
    schedule = ProductionJacobianSchedule(gas.kinetics)
    kernel = GeneratedProductionJacobianKernel(gas.kinetics, schedule)
    return GeneratedIsobaricReactor(isobaricReactor(gas), schedule, kernel)
end

function isobaric_rhs(gas, u, p, t)
    state = isobaricReactor(gas, eltype(u))
    du = similar(u)
    state(du, u, p, t)
    return du
end

"""
    jacobian!(J, workspace::IsobaricJacobianWorkspace, u, p, t)

Evaluate the dense Jacobian of the isobaric reactor RHS in-place. This first
implementation is ForwardDiff-backed and intended as the package-level
correctness and SciML integration reference.
"""
function jacobian!(J, workspace::IsobaricJacobianWorkspace, u, p, t)
    J .= ForwardDiff.jacobian(x -> isobaric_rhs(workspace.gas, x, p, t), u)
    return nothing
end

function (workspace::IsobaricJacobianWorkspace)(J, u, p, t)
    return jacobian!(J, workspace, u, p, t)
end

function jacobian!(J, workspace::AnalyticalIsobaricJacobianWorkspace, u, p, t)
    return isobaric_reactor_jacobian!(
        J,
        workspace.state,
        workspace.schedule,
        workspace.dwdot_dC,
        workspace.dwdot_dT,
        workspace.dg_RT_dT,
        workspace.dh_RT_dT,
        workspace.wdot,
        u,
        p,
        t,
    )
end

function (workspace::AnalyticalIsobaricJacobianWorkspace)(J, u, p, t)
    return jacobian!(J, workspace, u, p, t)
end

function jacobian!(J, workspace::GeneratedAnalyticalIsobaricJacobianWorkspace, u, p, t)
    return isobaric_reactor_jacobian!(
        J,
        workspace.state,
        workspace.schedule,
        workspace.kernel,
        workspace.dwdot_dC,
        workspace.dwdot_dT,
        workspace.dg_RT_dT,
        workspace.dh_RT_dT,
        workspace.wdot,
        u,
        p,
        t,
    )
end

function (workspace::GeneratedAnalyticalIsobaricJacobianWorkspace)(J, u, p, t)
    return jacobian!(J, workspace, u, p, t)
end

"""
    isobaric_jacobian_prototype(gas)

Return a sparse `jac_prototype` for SciML ODE functions using the isobaric
reactor state `[Y..., T]`. The full isobaric reactor Jacobian is structurally
dense because mixture molecular weight, density, and the energy equation couple
all state variables, so this prototype stores every entry.
"""
function isobaric_jacobian_prototype(gas)
    n = gas.n_species + 1
    rows = Vector{Int}(undef, n * n)
    cols = Vector{Int}(undef, n * n)
    vals = zeros(n * n)
    idx = 1
    for col in 1:n
        for row in 1:n
            rows[idx] = row
            cols[idx] = col
            idx += 1
        end
    end
    return sparse(rows, cols, vals, n, n)
end

function sparse_row_entries(A, row::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    rows = Int[]
    vals = T[]
    for col in 1:size(A, 2)
        val = sparse_column_value(A, row, col)
        if val != 0.0
            push!(rows, col)
            push!(vals, val)
        end
    end
    return rows, vals
end

function sparse_column_entries(A, col::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    rows = Int[]
    vals = T[]
    for j in A.colptr[col]:A.colptr[col + 1] - 1
        push!(rows, A.rowval[j])
        push!(vals, A.nzval[j])
    end
    return rows, vals
end

ProductionJacobianSchedule(reactions::ReorderedKinetics) =
    ProductionJacobianSchedule(reactions, Float64)

function ProductionJacobianSchedule(reactions::ReorderedKinetics, ::Type{T}) where {T<:AbstractFloat}
    n_reactions = size(reactions.arrhenius_coeffs, 1)
    reactant_ptr = Vector{Int}(undef, n_reactions + 1)
    product_ptr = Vector{Int}(undef, n_reactions + 1)
    stoich_ptr = Vector{Int}(undef, n_reactions + 1)
    reverse_ptr = Vector{Int}(undef, n_reactions + 1)
    efficiency_ptr = Vector{Int}(undef, n_reactions + 1)
    reactant_species = Int[]
    reactant_orders = T[]
    product_species = Int[]
    product_orders = T[]
    stoich_species = Int[]
    stoich_coeffs = T[]
    reverse_species = Int[]
    reverse_coeffs = T[]
    efficiency_species = Int[]
    efficiency_coeffs = T[]

    for ir in 1:n_reactions
        reactant_ptr[ir] = length(reactant_species) + 1
        species, values = sparse_column_entries(reactions.reactant_orders, ir, T)
        append!(reactant_species, species)
        append!(reactant_orders, values)

        stoich_ptr[ir] = length(stoich_species) + 1
        species, values = sparse_row_entries(reactions.vk, ir, T)
        append!(stoich_species, species)
        append!(stoich_coeffs, values)

        if ir >= reactions.index_rev_start
            rev_col = ir - reactions.index_rev_start + 1
            product_ptr[ir] = length(product_species) + 1
            species, values = sparse_column_entries(reactions.product_stoich_coeffs, rev_col, T)
            append!(product_species, species)
            append!(product_orders, values)

            reverse_ptr[ir] = length(reverse_species) + 1
            species, values = sparse_row_entries(reactions.vk_rev, rev_col, T)
            append!(reverse_species, species)
            append!(reverse_coeffs, values)
        else
            product_ptr[ir] = length(product_species) + 1
            reverse_ptr[ir] = length(reverse_species) + 1
        end

        if ir >= reactions.index_three_body_start
            efficiency_ptr[ir] = length(efficiency_species) + 1
            if ir < reactions.index_fallOff_start
                eff_col = ir - reactions.index_three_body_start + 1
                species, values = sparse_column_entries(reactions.efficiency_coeffs_tB, eff_col, T)
            else
                eff_col = ir - reactions.index_fallOff_start + 1
                species, values = sparse_column_entries(reactions.efficiency_coeffs_fO, eff_col, T)
            end
            append!(efficiency_species, species)
            append!(efficiency_coeffs, values)
        else
            efficiency_ptr[ir] = length(efficiency_species) + 1
        end
    end
    reactant_ptr[end] = length(reactant_species) + 1
    product_ptr[end] = length(product_species) + 1
    stoich_ptr[end] = length(stoich_species) + 1
    reverse_ptr[end] = length(reverse_species) + 1
    efficiency_ptr[end] = length(efficiency_species) + 1

    return ProductionJacobianSchedule(
        reactant_ptr,
        reactant_species,
        reactant_orders,
        product_ptr,
        product_species,
        product_orders,
        stoich_ptr,
        stoich_species,
        stoich_coeffs,
        reverse_ptr,
        reverse_species,
        reverse_coeffs,
        efficiency_ptr,
        efficiency_species,
        efficiency_coeffs,
    )
end

function sparse_column_value(A, row::Int, col::Int)
    for j in A.colptr[col]:A.colptr[col + 1] - 1
        A.rowval[j] == row && return A.nzval[j]
    end
    return 0.0
end

function generated_kernel_id(reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule)
    return UInt64(hash((
        size(reactions.arrhenius_coeffs),
        reactions.index_rev_start,
        reactions.index_three_body_start,
        reactions.index_fallOff_start,
        reactions.index_troe_start,
        reactions.index_pressure_dependent,
        reactions.pressure_dependent_ptr,
        reactions.pressure_dependent_is_linear_burke,
        size(reactions.pressure_dependent_pressures),
        size(reactions.pressure_dependent_arrhenius),
        size(reactions.linear_burke_efficiencies),
        schedule.reactant_ptr,
        schedule.reactant_species,
        schedule.reactant_orders,
        schedule.product_ptr,
        schedule.product_species,
        schedule.product_orders,
        schedule.stoich_ptr,
        schedule.stoich_species,
        schedule.stoich_coeffs,
        schedule.reverse_ptr,
        schedule.reverse_species,
        schedule.reverse_coeffs,
        schedule.efficiency_ptr,
        schedule.efficiency_species,
        schedule.efficiency_coeffs,
    ), UInt(0)))
end

function scheduled_range(ptr, i::Int)
    return ptr[i]:(ptr[i + 1] - 1)
end

function product_expr(Csym, species, orders, indices)
    expr = :(one(eltype($Csym)))
    for idx in indices
        expr = :($expr * $Csym[$(species[idx])]^$(orders[idx]))
    end
    return expr
end

function product_derivative_expr(Csym, species, orders, indices, derivative_index::Int)
    expr = :($(orders[derivative_index]))
    for idx in indices
        exponent = idx == derivative_index ? orders[idx] - 1 : orders[idx]
        expr = :($expr * $Csym[$(species[idx])]^$exponent)
    end
    return expr
end

function reverse_denominator_expr(gsym, Tsym, reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule, reaction::Int)
    local_reaction = reaction - reactions.index_rev_start + 1
    delta_g = :(zero(eltype($gsym)))
    for idx in scheduled_range(schedule.reverse_ptr, reaction)
        delta_g = :($delta_g + $(schedule.reverse_coeffs[idx]) * $gsym[$(schedule.reverse_species[idx])])
    end
    return :(exp(-($delta_g) + safe_log(one_atm / R / $Tsym) * $(reactions.vk_sum[local_reaction])))
end

function generated_stoich_updates(Jsym, schedule::ProductionJacobianSchedule, reaction::Int, species_j::Int, value_expr)
    updates = Expr[]
    for idx in scheduled_range(schedule.stoich_ptr, reaction)
        push!(
            updates,
            :($Jsym[$(schedule.stoich_species[idx]), $species_j] += $(schedule.stoich_coeffs[idx]) * $value_expr),
        )
    end
    return updates
end

function generated_wdot_updates(wdotsym, schedule::ProductionJacobianSchedule, reaction::Int, value_expr)
    updates = Expr[]
    for idx in scheduled_range(schedule.stoich_ptr, reaction)
        push!(
            updates,
            :($wdotsym[$(schedule.stoich_species[idx])] += $(schedule.stoich_coeffs[idx]) * $value_expr),
        )
    end
    return updates
end

function reverse_delta_derivative_expr(dgsym, schedule::ProductionJacobianSchedule, reaction::Int)
    delta = :(0.0)
    for idx in scheduled_range(schedule.reverse_ptr, reaction)
        delta = :($delta + $(schedule.reverse_coeffs[idx]) * $dgsym[$(schedule.reverse_species[idx])])
    end
    return :($dgsym === nothing ? 0.0 : $delta)
end

function generated_wdot_body(reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule)
    blocks = Expr[]
    n_reactions = size(reactions.arrhenius_coeffs, 1)

    for ir in 1:n_reactions
        pdep_i = pressure_dependent_local_index(reactions, ir)
        reactant_indices = collect(scheduled_range(schedule.reactant_ptr, ir))
        product_indices = collect(scheduled_range(schedule.product_ptr, ir))
        reactant_product = product_expr(:C, schedule.reactant_species, schedule.reactant_orders, reactant_indices)

        statements = Expr[:(k_base = kf[$ir])]
        if pdep_i !== nothing
            push!(
                statements,
                :(pdep_result = pressure_dependent_base_rate_and_pressure_derivative(
                    reactions,
                    C,
                    T,
                    logT,
                    rec_RcalT,
                    $pdep_i,
                )),
                :(k_base = pdep_result[1]),
            )
        end
        append!(
            statements,
            Expr[
                :(rate_result = effective_rate_and_derivative(reactions, k_base, C, T, logT, rec_RcalT, $ir)),
                :(k_eff = rate_result[1]),
                :(kf[$ir] = k_eff * $reactant_product),
                :(net_rate = kf[$ir]),
            ],
        )

        if ir >= reactions.index_rev_start
            reverse_denominator = reverse_denominator_expr(:g_RT, :T, reactions, schedule, ir)
            product_product = product_expr(:C, schedule.product_species, schedule.product_orders, product_indices)
            push!(statements, :(kr[$ir] = k_eff / $reverse_denominator * $product_product))
            push!(statements, :(net_rate -= kr[$ir]))
        end

        append!(statements, generated_wdot_updates(:wdot, schedule, ir, :net_rate))

        push!(
            blocks,
            quote
                local k_base, pdep_result, rate_result, k_eff, net_rate
                $(statements...)
            end,
        )
    end

    return quote
        wdot .= 0.0
        n_reactions = size(reactions.arrhenius_coeffs, 1)
        n_reactions == 0 && return nothing

        logT = safe_log(T)
        rec_RcalT = 4184.0 / Arrhenius.R / T
        arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
        kr .= 0.0

        $(blocks...)
        return nothing
    end
end

function generated_production_jacobian_body(reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule)
    blocks = Expr[]
    n_reactions = size(reactions.arrhenius_coeffs, 1)

    for ir in 1:n_reactions
        pdep_i = pressure_dependent_local_index(reactions, ir)
        reactant_indices = collect(scheduled_range(schedule.reactant_ptr, ir))
        product_indices = collect(scheduled_range(schedule.product_ptr, ir))
        efficiency_indices = collect(scheduled_range(schedule.efficiency_ptr, ir))
        reactant_product = product_expr(:C, schedule.reactant_species, schedule.reactant_orders, reactant_indices)
        reactant_species = Set(schedule.reactant_species[reactant_indices])

        statements = Expr[:(k_base = kf[$ir]), :(dk_base_dP = zero(k_base))]
        if pdep_i !== nothing
            push!(
                statements,
                :(pdep_result = pressure_dependent_base_rate_and_pressure_derivative(
                    reactions,
                    C,
                    T,
                    logT,
                    rec_RcalT,
                    $pdep_i,
                )),
                :(k_base = pdep_result[1]),
                :(dk_base_dP = pdep_result[2]),
            )
        end
        append!(
            statements,
            Expr[
                :(rate_result = effective_rate_and_derivative(reactions, k_base, C, T, logT, rec_RcalT, $ir)),
                :(k_eff = rate_result[1]),
                :(dk_dM = rate_result[2]),
                :(kf[$ir] = k_eff),
                :(reactant_product = $reactant_product),
            ],
        )

        for idx in reactant_indices
            species_j = schedule.reactant_species[idx]
            derivative = product_derivative_expr(:C, schedule.reactant_species, schedule.reactant_orders, reactant_indices, idx)
            dq = pdep_i === nothing ? :(kf[$ir] * $derivative) :
                 :(kf[$ir] * $derivative +
                   dk_base_dP * effective_pressure_derivative_C(reactions, $species_j, T, logT, rec_RcalT, $pdep_i) *
                   reactant_product)
            append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
        end

        if pdep_i !== nothing
            for species_j in axes(reactions.reactant_orders, 1)
                species_j in reactant_species && continue
                dq = :(dk_base_dP *
                       effective_pressure_derivative_C(reactions, $species_j, T, logT, rec_RcalT, $pdep_i) *
                       reactant_product)
                append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
            end
        end

        if ir >= reactions.index_three_body_start
            for idx in efficiency_indices
                species_j = schedule.efficiency_species[idx]
                dq = :(dk_dM * $(schedule.efficiency_coeffs[idx]) * reactant_product)
                append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
            end
        end

        if ir >= reactions.index_rev_start
            reverse_denominator = reverse_denominator_expr(:g_RT, :T, reactions, schedule, ir)
            product_product = product_expr(:C, schedule.product_species, schedule.product_orders, product_indices)
            push!(statements, :(reverse_denominator = $reverse_denominator))
            push!(statements, :(kr[$ir] = kf[$ir] / reverse_denominator))
            push!(statements, :(product_product = $product_product))

            for idx in product_indices
                species_j = schedule.product_species[idx]
                derivative = product_derivative_expr(:C, schedule.product_species, schedule.product_orders, product_indices, idx)
                dq = :(-(kr[$ir] * $derivative))
                append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
            end

            if ir >= reactions.index_three_body_start
                for idx in efficiency_indices
                    species_j = schedule.efficiency_species[idx]
                    dq = :(-(dk_dM * $(schedule.efficiency_coeffs[idx]) / reverse_denominator * product_product))
                    append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
                end
            end

            if pdep_i !== nothing
                for species_j in axes(reactions.reactant_orders, 1)
                    dk_reverse_dC = :(dk_base_dP *
                                      effective_pressure_derivative_C(reactions, $species_j, T, logT, rec_RcalT, $pdep_i) /
                                      reverse_denominator)
                    dq = :(-($dk_reverse_dC * product_product))
                    append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
                end
            end
        end

        push!(
            blocks,
            quote
                local k_base, dk_base_dP, pdep_result, rate_result, k_eff, dk_dM, reactant_product, reverse_denominator,
                    product_product
                $(statements...)
            end,
        )
    end

    return quote
        J .= 0.0
        n_reactions = size(reactions.arrhenius_coeffs, 1)
        n_reactions == 0 && return nothing

        logT = safe_log(T)
        rec_RcalT = 4184.0 / Arrhenius.R / T
        arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
        kr .= 0.0

        $(blocks...)
        return nothing
    end
end

function generated_production_jacobian_wdot_body(reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule)
    blocks = Expr[]
    n_reactions = size(reactions.arrhenius_coeffs, 1)

    for ir in 1:n_reactions
        pdep_i = pressure_dependent_local_index(reactions, ir)
        reactant_indices = collect(scheduled_range(schedule.reactant_ptr, ir))
        product_indices = collect(scheduled_range(schedule.product_ptr, ir))
        efficiency_indices = collect(scheduled_range(schedule.efficiency_ptr, ir))
        reactant_product = product_expr(:C, schedule.reactant_species, schedule.reactant_orders, reactant_indices)
        reactant_species = Set(schedule.reactant_species[reactant_indices])

        statements = Expr[:(k_base = kf[$ir]), :(dk_base_dP = zero(k_base))]
        if pdep_i !== nothing
            push!(
                statements,
                :(pdep_result = pressure_dependent_base_rate_and_pressure_derivative(
                    reactions,
                    C,
                    T,
                    logT,
                    rec_RcalT,
                    $pdep_i,
                )),
                :(k_base = pdep_result[1]),
                :(dk_base_dP = pdep_result[2]),
            )
        end
        append!(
            statements,
            Expr[
                :(rate_result = effective_rate_and_derivative(reactions, k_base, C, T, logT, rec_RcalT, $ir)),
                :(k_eff = rate_result[1]),
                :(dk_dM = rate_result[2]),
                :(kf[$ir] = k_eff),
                :(reactant_product = $reactant_product),
                :(net_rate = kf[$ir] * reactant_product),
            ],
        )

        for idx in reactant_indices
            species_j = schedule.reactant_species[idx]
            derivative = product_derivative_expr(:C, schedule.reactant_species, schedule.reactant_orders, reactant_indices, idx)
            dq = pdep_i === nothing ? :(kf[$ir] * $derivative) :
                 :(kf[$ir] * $derivative +
                   dk_base_dP * effective_pressure_derivative_C(reactions, $species_j, T, logT, rec_RcalT, $pdep_i) *
                   reactant_product)
            append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
        end

        if pdep_i !== nothing
            for species_j in axes(reactions.reactant_orders, 1)
                species_j in reactant_species && continue
                dq = :(dk_base_dP *
                       effective_pressure_derivative_C(reactions, $species_j, T, logT, rec_RcalT, $pdep_i) *
                       reactant_product)
                append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
            end
        end

        if ir >= reactions.index_three_body_start
            for idx in efficiency_indices
                species_j = schedule.efficiency_species[idx]
                dq = :(dk_dM * $(schedule.efficiency_coeffs[idx]) * reactant_product)
                append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
            end
        end

        if ir >= reactions.index_rev_start
            reverse_denominator = reverse_denominator_expr(:g_RT, :T, reactions, schedule, ir)
            product_product = product_expr(:C, schedule.product_species, schedule.product_orders, product_indices)
            push!(statements, :(reverse_denominator = $reverse_denominator))
            push!(statements, :(kr[$ir] = kf[$ir] / reverse_denominator))
            push!(statements, :(product_product = $product_product))
            push!(statements, :(net_rate -= kr[$ir] * product_product))

            for idx in product_indices
                species_j = schedule.product_species[idx]
                derivative = product_derivative_expr(:C, schedule.product_species, schedule.product_orders, product_indices, idx)
                dq = :(-(kr[$ir] * $derivative))
                append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
            end

            if ir >= reactions.index_three_body_start
                for idx in efficiency_indices
                    species_j = schedule.efficiency_species[idx]
                    dq = :(-(dk_dM * $(schedule.efficiency_coeffs[idx]) / reverse_denominator * product_product))
                    append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
                end
            end

            if pdep_i !== nothing
                for species_j in axes(reactions.reactant_orders, 1)
                    dk_reverse_dC = :(dk_base_dP *
                                      effective_pressure_derivative_C(reactions, $species_j, T, logT, rec_RcalT, $pdep_i) /
                                      reverse_denominator)
                    dq = :(-($dk_reverse_dC * product_product))
                    append!(statements, generated_stoich_updates(:J, schedule, ir, species_j, dq))
                end
            end
        end

        append!(statements, generated_wdot_updates(:wdot, schedule, ir, :net_rate))

        push!(
            blocks,
            quote
                local k_base, dk_base_dP, pdep_result, rate_result, k_eff, dk_dM, reactant_product, reverse_denominator,
                    product_product, net_rate
                $(statements...)
            end,
        )
    end

    return quote
        J .= 0.0
        wdot .= 0.0
        n_reactions = size(reactions.arrhenius_coeffs, 1)
        n_reactions == 0 && return nothing

        logT = safe_log(T)
        rec_RcalT = 4184.0 / Arrhenius.R / T
        arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
        kr .= 0.0

        $(blocks...)
        return nothing
    end
end

function generated_production_temperature_body(reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule)
    blocks = Expr[]
    n_reactions = size(reactions.arrhenius_coeffs, 1)

    for ir in 1:n_reactions
        pdep_i = pressure_dependent_local_index(reactions, ir)
        reactant_indices = collect(scheduled_range(schedule.reactant_ptr, ir))
        product_indices = collect(scheduled_range(schedule.product_ptr, ir))
        reactant_product = product_expr(:C, schedule.reactant_species, schedule.reactant_orders, reactant_indices)

        statements = Expr[
            :(k_base = kf[$ir]),
            :(dk_base_dT = arrhenius_rate_temperature_derivative(reactions.arrhenius_coeffs, $ir, T, k_base)),
        ]
        if pdep_i !== nothing
            push!(
                statements,
                :(pdep_result = pressure_dependent_base_rate_temperature_derivative(
                    reactions,
                    C,
                    T,
                    logT,
                    rec_RcalT,
                    $pdep_i,
                )),
                :(k_base = pdep_result[1]),
                :(dk_base_dT = pdep_result[2]),
            )
        end
        append!(
            statements,
            Expr[
                :(rate_result = effective_rate_and_temperature_derivative(reactions, k_base, dk_base_dT, C, T, logT, rec_RcalT, $ir)),
                :(k_eff = rate_result[1]),
                :(dk_eff_dT = rate_result[2]),
                :(kf[$ir] = k_eff),
                :(dq_dT = dk_eff_dT * $reactant_product),
            ],
        )

        if ir >= reactions.index_rev_start
            local_reaction = ir - reactions.index_rev_start + 1
            reverse_denominator = reverse_denominator_expr(:g_RT, :T, reactions, schedule, ir)
            product_product = product_expr(:C, schedule.product_species, schedule.product_orders, product_indices)
            ddelta_g = reverse_delta_derivative_expr(:dg_RT_dT, schedule, ir)
            push!(statements, :(kr[$ir] = k_eff / $reverse_denominator))
            push!(
                statements,
                :(
                    dkr_dT =
                        kr[$ir] * (dk_eff_dT / k_eff + ($ddelta_g) + $(reactions.vk_sum[local_reaction]) / T)
                ),
            )
            push!(statements, :(dq_dT -= dkr_dT * $product_product))
        end

        append!(statements, generated_wdot_updates(:dwdot_dT, schedule, ir, :dq_dT))

        push!(
            blocks,
            quote
                local k_base, dk_base_dT, pdep_result, rate_result, k_eff, dk_eff_dT, dq_dT, dkr_dT
                $(statements...)
            end,
        )
    end

    return quote
        dwdot_dT .= 0.0
        n_reactions = size(reactions.arrhenius_coeffs, 1)
        n_reactions == 0 && return nothing

        logT = safe_log(T)
        rec_RcalT = 4184.0 / Arrhenius.R / T
        arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
        kr .= 0.0

        $(blocks...)
        return nothing
    end
end

function generated_pressure_dependent_wdot_fallback_body(id::UInt64)
    return quote
        return wdot_func!(wdot, kf, kr, C, g_RT, reactions, T)
    end
end

function generated_pressure_dependent_jacobian_fallback_body(id::UInt64)
    return quote
        return production_rate_jacobian_C!(
            J,
            kf,
            kr,
            C,
            g_RT,
            reactions,
            GENERATED_PRODUCTION_JACOBIAN_SCHEDULES[$id],
            T,
        )
    end
end

function generated_pressure_dependent_jacobian_wdot_fallback_body(id::UInt64)
    return quote
        wdot_func!(wdot, kf, kr, C, g_RT, reactions, T)
        return production_rate_jacobian_C!(
            J,
            kf,
            kr,
            C,
            g_RT,
            reactions,
            GENERATED_PRODUCTION_JACOBIAN_SCHEDULES[$id],
            T,
        )
    end
end

function generated_pressure_dependent_temperature_fallback_body(id::UInt64)
    return quote
        return production_rate_temperature_derivative!(
            dwdot_dT,
            kf,
            kr,
            C,
            g_RT,
            reactions,
            GENERATED_PRODUCTION_JACOBIAN_SCHEDULES[$id],
            T,
            dg_RT_dT,
        )
    end
end

function GeneratedProductionJacobianKernel(reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule)
    id = generated_kernel_id(reactions, schedule)
    GENERATED_PRODUCTION_JACOBIAN_SCHEDULES[id] = schedule
    use_pressure_dependent_fallback =
        !isempty(reactions.index_pressure_dependent) &&
        size(reactions.arrhenius_coeffs, 1) > GENERATED_PRESSURE_DEPENDENT_REACTION_LIMIT
    if !haskey(GENERATED_WDOT_BODIES, id)
        GENERATED_WDOT_BODIES[id] = use_pressure_dependent_fallback ?
                                    generated_pressure_dependent_wdot_fallback_body(id) :
                                    generated_wdot_body(reactions, schedule)
    end
    if !haskey(GENERATED_PRODUCTION_JACOBIAN_BODIES, id)
        GENERATED_PRODUCTION_JACOBIAN_BODIES[id] = use_pressure_dependent_fallback ?
                                                   generated_pressure_dependent_jacobian_fallback_body(id) :
                                                   generated_production_jacobian_body(reactions, schedule)
    end
    if !haskey(GENERATED_PRODUCTION_JACOBIAN_WDOT_BODIES, id)
        GENERATED_PRODUCTION_JACOBIAN_WDOT_BODIES[id] = use_pressure_dependent_fallback ?
                                                        generated_pressure_dependent_jacobian_wdot_fallback_body(id) :
                                                        generated_production_jacobian_wdot_body(reactions, schedule)
    end
    if !haskey(GENERATED_PRODUCTION_TEMPERATURE_BODIES, id)
        GENERATED_PRODUCTION_TEMPERATURE_BODIES[id] = use_pressure_dependent_fallback ?
                                                      generated_pressure_dependent_temperature_fallback_body(id) :
                                                      generated_production_temperature_body(reactions, schedule)
    end
    return GeneratedProductionJacobianKernel{id}()
end

@generated function generated_wdot_func!(
    wdot,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    ::GeneratedProductionJacobianKernel{ID},
    T,
) where {ID}
    return GENERATED_WDOT_BODIES[ID]
end

@generated function generated_production_rate_jacobian_C!(
    J,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    ::GeneratedProductionJacobianKernel{ID},
    T,
) where {ID}
    return GENERATED_PRODUCTION_JACOBIAN_BODIES[ID]
end

@generated function generated_production_rate_jacobian_C_and_wdot!(
    J,
    wdot,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    ::GeneratedProductionJacobianKernel{ID},
    T,
) where {ID}
    return GENERATED_PRODUCTION_JACOBIAN_WDOT_BODIES[ID]
end

@generated function generated_production_rate_temperature_derivative!(
    dwdot_dT,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    ::GeneratedProductionJacobianKernel{ID},
    T,
    dg_RT_dT,
) where {ID}
    return GENERATED_PRODUCTION_TEMPERATURE_BODIES[ID]
end

function wdot_func!(
    wdot,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    kernel::GeneratedProductionJacobianKernel,
    T,
)
    return generated_wdot_func!(wdot, kf, kr, C, g_RT, reactions, kernel, T)
end

function (reactor::GeneratedIsobaricReactor)(du, u, p, t)
    state = reactor.state
    Y = @view(u[1:state.gas.n_species])

    updateReactor!(state, Y, u[end], p[1])
    wdot = @view(du[1:state.gas.n_species])
    wdot_func!(wdot, state.kf, state.kr, state.C, state.g_RT, state.gas.kinetics, reactor.kernel, state.T)

    du[end] = 0.0
    @inbounds for i in 1:state.gas.n_species
        du[end] -= state.h_RT[i] * du[i]
    end
    du[end] *= Arrhenius.R * state.T
    du[end] /= state.cp_mass
    @inbounds for i in 1:state.gas.n_species
        du[i] *= state.gas.MW[i]
    end
    du ./= state.ρ_mass
    return nothing
end

function production_rate_jacobian_C!(
    J,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    kernel::GeneratedProductionJacobianKernel,
    T,
)
    return generated_production_rate_jacobian_C!(J, kf, kr, C, g_RT, reactions, kernel, T)
end

function production_rate_jacobian_C_and_wdot!(
    J,
    wdot,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    kernel::GeneratedProductionJacobianKernel,
    T,
)
    return generated_production_rate_jacobian_C_and_wdot!(J, wdot, kf, kr, C, g_RT, reactions, kernel, T)
end

function production_rate_temperature_derivative!(
    dwdot_dT,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    kernel::GeneratedProductionJacobianKernel,
    T,
    dg_RT_dT = nothing,
)
    return generated_production_rate_temperature_derivative!(dwdot_dT, kf, kr, C, g_RT, reactions, kernel, T, dg_RT_dT)
end

function concentration_product(C, orders, reaction::Int)
    q = one(eltype(C))
    for j in orders.colptr[reaction]:orders.colptr[reaction + 1] - 1
        @inbounds q *= C[orders.rowval[j]]^orders.nzval[j]
    end
    return q
end

function concentration_product_derivative(C, orders, reaction::Int, species::Int)
    derivative_order = sparse_column_value(orders, species, reaction)
    derivative_order == 0.0 && return zero(eltype(C))

    dq_dC = derivative_order
    for j in orders.colptr[reaction]:orders.colptr[reaction + 1] - 1
        row = orders.rowval[j]
        order = orders.nzval[j]
        exponent = row == species ? order - 1 : order
        @inbounds dq_dC *= C[row]^exponent
    end
    return dq_dC
end

function arrhenius_rate(arr_coeffs, row::Int, logT, rec_RcalT)
    return arr_coeffs[row, 1] * exp(arr_coeffs[row, 2] * logT - arr_coeffs[row, 3] * rec_RcalT)
end

function arrhenius_rate_temperature_derivative(arr_coeffs, row::Int, T, k)
    return k * (arr_coeffs[row, 2] / T + arr_coeffs[row, 3] * 4184.0 / Arrhenius.R / T^2)
end

function reverse_rate_constant_from_kf(reactions::ReorderedKinetics, k_forward, g_RT, T, reaction::Int)
    local_reaction = reaction - reactions.index_rev_start + 1
    delta_g_RT = zero(eltype(g_RT))
    for col in 1:size(reactions.vk_rev, 2)
        delta_g_RT += sparse_column_value(reactions.vk_rev, local_reaction, col) * g_RT[col]
    end
    logP = safe_log(one_atm / R / T)
    return k_forward / exp(-delta_g_RT + logP * reactions.vk_sum[local_reaction])
end

function reverse_rate_constant(reactions::ReorderedKinetics, kf, g_RT, T, reaction::Int)
    return reverse_rate_constant_from_kf(reactions, kf[reaction], g_RT, T, reaction)
end

function reverse_rate_temperature_derivative_from_kf(reactions::ReorderedKinetics, k_reverse, k_forward, dk_forward_dT, dg_RT_dT, T, reaction::Int)
    local_reaction = reaction - reactions.index_rev_start + 1
    dlog_k_forward_dT = dk_forward_dT / k_forward
    ddelta_g_RT_dT = zero(k_reverse)
    if dg_RT_dT !== nothing
        for col in 1:size(reactions.vk_rev, 2)
            ddelta_g_RT_dT += sparse_column_value(reactions.vk_rev, local_reaction, col) * dg_RT_dT[col]
        end
    end
    return k_reverse * (dlog_k_forward_dT + ddelta_g_RT_dT + reactions.vk_sum[local_reaction] / T)
end

function third_body_efficiency(C, efficiency_coeffs, reaction::Int)
    M_eff = zero(eltype(C))
    for j in efficiency_coeffs.colptr[reaction]:efficiency_coeffs.colptr[reaction + 1] - 1
        @inbounds M_eff += efficiency_coeffs.nzval[j] * C[efficiency_coeffs.rowval[j]]
    end
    return M_eff
end

function efficiency_value(efficiency_coeffs, species::Int, reaction::Int)
    return sparse_column_value(efficiency_coeffs, species, reaction)
end

function order_value(orders, species::Int, reaction::Int)
    return sparse_column_value(orders, species, reaction)
end

@inline function troe_factor_and_derivative(Pr, troe, row::Int, T)
    Pr = positive_rate_argument(Pr)
    F_cent =
        (1 - troe[row, 1]) * exp(-T / troe[row, 4]) +
        troe[row, 1] * exp(-T / troe[row, 2]) +
        exp(-troe[row, 3] / T)
    F_cent = positive_rate_argument(F_cent)
    logF_cent = safe_log10(F_cent)
    c = -0.4 - 0.67 * logF_cent
    n = 0.75 - 1.27 * logF_cent
    x = safe_log10(Pr) + c
    denominator = n - 0.14 * x
    f1 = x / denominator
    factor = exp(log(10.0) * logF_cent / (1 + f1^2))
    dlog_factor_dPr = -2 * logF_cent * f1 * n / (Pr * denominator^2 * (1 + f1^2)^2)
    return factor, factor * dlog_factor_dPr
end

@inline function troe_factor_and_temperature_derivative(Pr, dPr_dT, troe, row::Int, T)
    Pr = positive_rate_argument(Pr)
    exp1 = exp(-T / troe[row, 4])
    exp2 = exp(-T / troe[row, 2])
    exp3 = exp(-troe[row, 3] / T)
    F_cent = (1 - troe[row, 1]) * exp1 + troe[row, 1] * exp2 + exp3
    dF_cent_dT =
        -(1 - troe[row, 1]) * exp1 / troe[row, 4] -
        troe[row, 1] * exp2 / troe[row, 2] +
        exp3 * troe[row, 3] / T^2

    F_cent = positive_rate_argument(F_cent)
    logF_cent = safe_log10(F_cent)
    dlogF_cent_dT = dF_cent_dT / (F_cent * log(10.0))
    c = -0.4 - 0.67 * logF_cent
    dc_dT = -0.67 * dlogF_cent_dT
    n = 0.75 - 1.27 * logF_cent
    dn_dT = -1.27 * dlogF_cent_dT
    x = safe_log10(Pr) + c
    dx_dT = dPr_dT / (Pr * log(10.0)) + dc_dT
    denominator = n - 0.14 * x
    ddenominator_dT = dn_dT - 0.14 * dx_dT
    f1 = x / denominator
    df1_dT = (dx_dT * denominator - x * ddenominator_dT) / denominator^2
    exponent = log(10.0) * logF_cent / (1 + f1^2)
    dexponent_dT =
        log(10.0) * (
            dlogF_cent_dT / (1 + f1^2) -
            logF_cent * 2 * f1 * df1_dT / (1 + f1^2)^2
        )
    factor = exp(exponent)
    return factor, factor * dexponent_dT
end

@inline function effective_rate_and_derivative(reactions::ReorderedKinetics, k_inf, C, T, logT, rec_RcalT, reaction::Int)
    if reaction < reactions.index_three_body_start
        return k_inf, zero(k_inf), 0
    end

    if reaction < reactions.index_fallOff_start
        efficiency_col = reaction - reactions.index_three_body_start + 1
        M_eff = third_body_efficiency(C, reactions.efficiency_coeffs_tB, efficiency_col)
        return k_inf * M_eff, k_inf, efficiency_col
    end

    efficiency_col = reaction - reactions.index_fallOff_start + 1
    M_eff = third_body_efficiency(C, reactions.efficiency_coeffs_fO, efficiency_col)
    k0 = arrhenius_rate(reactions.Arrhenius_0, efficiency_col, logT, rec_RcalT)
    Pr = positive_rate_argument(k0 * M_eff / k_inf)
    lindemann = k_inf * Pr / (1 + Pr)
    dlindemann_dM = k0 / (1 + Pr)^2

    if reaction >= reactions.index_troe_start
        troe_row = reaction - reactions.index_troe_start + 1
        factor, dfactor_dPr = troe_factor_and_derivative(Pr, reactions.troe, troe_row, T)
        dk_dM = dlindemann_dM * factor + lindemann * dfactor_dPr * k0 / k_inf
        return lindemann * factor, dk_dM, efficiency_col
    end

    return lindemann, dlindemann_dM, efficiency_col
end

@inline function effective_rate_and_temperature_derivative(reactions::ReorderedKinetics, k_inf, dk_inf_dT, C, T, logT, rec_RcalT, reaction::Int)
    if reaction < reactions.index_three_body_start
        return k_inf, dk_inf_dT
    end

    if reaction < reactions.index_fallOff_start
        efficiency_col = reaction - reactions.index_three_body_start + 1
        M_eff = third_body_efficiency(C, reactions.efficiency_coeffs_tB, efficiency_col)
        return k_inf * M_eff, dk_inf_dT * M_eff
    end

    efficiency_col = reaction - reactions.index_fallOff_start + 1
    M_eff = third_body_efficiency(C, reactions.efficiency_coeffs_fO, efficiency_col)
    k0 = arrhenius_rate(reactions.Arrhenius_0, efficiency_col, logT, rec_RcalT)
    dk0_dT = arrhenius_rate_temperature_derivative(reactions.Arrhenius_0, efficiency_col, T, k0)
    Pr = positive_rate_argument(k0 * M_eff / k_inf)
    dPr_dT = Pr * (dk0_dT / k0 - dk_inf_dT / k_inf)
    lindemann = k_inf * Pr / (1 + Pr)
    dlindemann_dT = dk_inf_dT * Pr / (1 + Pr) + k_inf * dPr_dT / (1 + Pr)^2

    if reaction >= reactions.index_troe_start
        troe_row = reaction - reactions.index_troe_start + 1
        factor, dfactor_dT = troe_factor_and_temperature_derivative(Pr, dPr_dT, reactions.troe, troe_row, T)
        return lindemann * factor, dlindemann_dT * factor + lindemann * dfactor_dT
    end

    return lindemann, dlindemann_dT
end

function effective_rate_derivative_species(reactions::ReorderedKinetics, dk_dM, species::Int, reaction::Int, efficiency_col::Int)
    reaction < reactions.index_three_body_start && return zero(dk_dM)
    if reaction < reactions.index_fallOff_start
        return dk_dM * efficiency_value(reactions.efficiency_coeffs_tB, species, efficiency_col)
    end
    return dk_dM * efficiency_value(reactions.efficiency_coeffs_fO, species, efficiency_col)
end

function pressure_dependent_local_index(reactions::ReorderedKinetics, reaction::Int)
    return findfirst(==(reaction), reactions.index_pressure_dependent)
end

function pressure_dependent_base_rate_and_pressure_derivative(reactions::ReorderedKinetics, C, T, logT, rec_RcalT, pdep_i::Int)
    first = reactions.pressure_dependent_ptr[pdep_i]
    last = reactions.pressure_dependent_ptr[pdep_i + 1] - 1
    pressure_atm = effective_pressure_atm(reactions, C, T, logT, rec_RcalT, pdep_i)
    k_base, dk_dP, _ = pressure_dependent_rate_derivatives(
        reactions.pressure_dependent_pressures,
        reactions.pressure_dependent_arrhenius,
        first,
        last,
        pressure_atm,
        T,
        logT,
        rec_RcalT,
    )
    return k_base, dk_dP
end

function pressure_dependent_base_rate_temperature_derivative(reactions::ReorderedKinetics, C, T, logT, rec_RcalT, pdep_i::Int)
    first = reactions.pressure_dependent_ptr[pdep_i]
    last = reactions.pressure_dependent_ptr[pdep_i + 1] - 1
    pressure_atm = effective_pressure_atm(reactions, C, T, logT, rec_RcalT, pdep_i)
    k_base, dk_dP, dk_dT_fixed_pressure = pressure_dependent_rate_derivatives(
        reactions.pressure_dependent_pressures,
        reactions.pressure_dependent_arrhenius,
        first,
        last,
        pressure_atm,
        T,
        logT,
        rec_RcalT,
    )
    dP_dT = effective_pressure_derivative_T(reactions, C, T, logT, rec_RcalT, pdep_i)
    return k_base, dk_dT_fixed_pressure + dk_dP * dP_dT
end

"""
    scheduled_product(C, species, orders, first, last)

`∏ C[species[i]]^orders[i]` over `i in first:last`. `orders` need not be
integer (fractional reaction orders are standard for reduced/fitted
mechanisms, e.g. BFER-type 2-step schemes) -- a fractional power of a
negative `C` is complex, and ordinary ODE/PDE truncation error routinely
produces small negative excursions in a species concentration that's
supposed to be ~0 (confirmed directly: `DomainError` here on a BFER 0D
homogeneous-reactor Jacobian evaluation). `positive_rate_argument`-floor
`C[i]` only when its own order is non-integer, so integer-order terms
(the overwhelming majority) keep their exact prior behavior.
"""
function scheduled_product(C, species, orders, first::Int, last::Int)
    q = one(eltype(C))
    for i in first:last
        @inbounds begin
            order = orders[i]
            Ci = C[species[i]]
            base = isinteger(order) ? Ci : positive_rate_argument(Ci)
            q *= base^order
        end
    end
    return q
end

"""
    scheduled_product_derivative(C, species, orders, first, last, derivative_index)

`∂/∂C[species[derivative_index]]` of `scheduled_product` -- same
non-integer-order positivity guard as `scheduled_product` itself, applied
per-term (the differentiated term's own exponent is `orders[i]-1`, which
is non-integer exactly when `orders[i]` is).
"""
function scheduled_product_derivative(C, species, orders, first::Int, last::Int, derivative_index::Int)
    dq_dC = orders[derivative_index]
    for i in first:last
        exponent = i == derivative_index ? orders[i] - 1 : orders[i]
        @inbounds begin
            Ci = C[species[i]]
            base = isinteger(exponent) ? Ci : positive_rate_argument(Ci)
            dq_dC *= base^exponent
        end
    end
    return dq_dC
end

function scheduled_efficiency(C, species, coeffs, first::Int, last::Int)
    M_eff = zero(eltype(C))
    for i in first:last
        @inbounds M_eff += coeffs[i] * C[species[i]]
    end
    return M_eff
end

function scheduled_reverse_rate_constant_from_kf(reactions::ReorderedKinetics, schedule::ProductionJacobianSchedule, k_forward, g_RT, T, reaction::Int)
    local_reaction = reaction - reactions.index_rev_start + 1
    delta_g_RT = zero(eltype(g_RT))
    for i in schedule.reverse_ptr[reaction]:schedule.reverse_ptr[reaction + 1] - 1
        @inbounds delta_g_RT += schedule.reverse_coeffs[i] * g_RT[schedule.reverse_species[i]]
    end
    logP = safe_log(one_atm / R / T)
    return k_forward / exp(-delta_g_RT + logP * reactions.vk_sum[local_reaction])
end

function scheduled_add_stoich!(J, schedule::ProductionJacobianSchedule, reaction::Int, species_j::Int, value)
    value == 0.0 && return nothing
    for i in schedule.stoich_ptr[reaction]:schedule.stoich_ptr[reaction + 1] - 1
        @inbounds J[schedule.stoich_species[i], species_j] += schedule.stoich_coeffs[i] * value
    end
    return nothing
end

function scheduled_add_temperature_stoich!(dwdot_dT, schedule::ProductionJacobianSchedule, reaction::Int, value)
    value == 0.0 && return nothing
    for i in schedule.stoich_ptr[reaction]:schedule.stoich_ptr[reaction + 1] - 1
        @inbounds dwdot_dT[schedule.stoich_species[i]] += schedule.stoich_coeffs[i] * value
    end
    return nothing
end

"""
    production_rate_jacobian_C!(J, kf, kr, C, g_RT, reactions, T)

Evaluate the analytical concentration Jacobian of molar species production
rates, `J[i, j] = d(wdot[i]) / d(C[j])`, for Arrhenius mass-action reactions
with third-body, falloff, and Troe pressure corrections at fixed temperature
and fixed thermodynamic state.
"""
function production_rate_jacobian_C!(
    J,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    T,
)
    J .= 0.0
    n_supported = size(reactions.arrhenius_coeffs, 1)
    n_supported == 0 && return nothing

    logT = safe_log(T)
    rec_RcalT = 4184.0 / Arrhenius.R / T
    arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
    kr .= 0.0

    for ir in 1:n_supported
        pdep_i = pressure_dependent_local_index(reactions, ir)
        k_base = kf[ir]
        dk_base_dP = zero(k_base)
        if pdep_i !== nothing
            k_base, dk_base_dP = pressure_dependent_base_rate_and_pressure_derivative(reactions, C, T, logT, rec_RcalT, pdep_i)
            kf[ir] = k_base
        end
        k_eff, dk_dM, efficiency_col = effective_rate_and_derivative(
            reactions,
            k_base,
            C,
            T,
            logT,
            rec_RcalT,
            ir,
        )
        kf[ir] = k_eff

        reactant_product = concentration_product(C, reactions.reactant_orders, ir)
        q_forward = kf[ir] * reactant_product
        q_reverse = zero(q_forward)
        if ir >= reactions.index_rev_start
            kr[ir] = reverse_rate_constant(reactions, kf, g_RT, T, ir)
            q_reverse = kr[ir] * concentration_product(C, reactions.product_stoich_coeffs, ir - reactions.index_rev_start + 1)
        end

        for species_j in axes(J, 2)
            dk_eff_dC = effective_rate_derivative_species(reactions, dk_dM, species_j, ir, efficiency_col)
            if pdep_i !== nothing
                dk_eff_dC += dk_base_dP * effective_pressure_derivative_C(reactions, species_j, T, logT, rec_RcalT, pdep_i)
            end
            dq_dC = kf[ir] * concentration_product_derivative(C, reactions.reactant_orders, ir, species_j)
            dq_dC += dk_eff_dC * reactant_product
            dq_dC == 0.0 && continue
            for species_i in 1:size(J, 1)
                nu = sparse_column_value(reactions.vk, ir, species_i)
                nu == 0.0 && continue
                J[species_i, species_j] += nu * dq_dC
            end
        end

        if ir >= reactions.index_rev_start
            rev_col = ir - reactions.index_rev_start + 1
            product_product = concentration_product(C, reactions.product_stoich_coeffs, rev_col)
            for species_j in axes(J, 2)
                dq_dC = kr[ir] * concentration_product_derivative(C, reactions.product_stoich_coeffs, rev_col, species_j)
                dk_eff_dC = effective_rate_derivative_species(reactions, dk_dM, species_j, ir, efficiency_col)
                if pdep_i !== nothing
                    dk_eff_dC += dk_base_dP * effective_pressure_derivative_C(reactions, species_j, T, logT, rec_RcalT, pdep_i)
                end
                dk_reverse_dC = reverse_rate_constant_from_kf(reactions, dk_eff_dC, g_RT, T, ir)
                dq_dC += dk_reverse_dC * product_product
                dq_dC == 0.0 && continue
                for species_i in 1:size(J, 1)
                    nu = sparse_column_value(reactions.vk, ir, species_i)
                    nu == 0.0 && continue
                    J[species_i, species_j] -= nu * dq_dC
                end
            end
        end
    end

    return nothing
end

function production_rate_jacobian_C!(
    J,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    schedule::ProductionJacobianSchedule,
    T,
)
    J .= 0.0
    n_reactions = size(reactions.arrhenius_coeffs, 1)
    n_reactions == 0 && return nothing

    logT = safe_log(T)
    rec_RcalT = 4184.0 / Arrhenius.R / T
    arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
    kr .= 0.0

    for ir in 1:n_reactions
        pdep_i = pressure_dependent_local_index(reactions, ir)
        k_base = kf[ir]
        dk_base_dP = zero(k_base)
        if pdep_i !== nothing
            k_base, dk_base_dP = pressure_dependent_base_rate_and_pressure_derivative(reactions, C, T, logT, rec_RcalT, pdep_i)
            kf[ir] = k_base
        end
        k_eff, dk_dM, _ = effective_rate_and_derivative(reactions, k_base, C, T, logT, rec_RcalT, ir)
        kf[ir] = k_eff
        reactant_first = schedule.reactant_ptr[ir]
        reactant_last = schedule.reactant_ptr[ir + 1] - 1
        reactant_product = scheduled_product(C, schedule.reactant_species, schedule.reactant_orders, reactant_first, reactant_last)

        for idx in reactant_first:reactant_last
            species_j = schedule.reactant_species[idx]
            dk_eff_dC = pdep_i === nothing ? zero(k_eff) :
                dk_base_dP * effective_pressure_derivative_C(reactions, species_j, T, logT, rec_RcalT, pdep_i)
            dq_dC = kf[ir] * scheduled_product_derivative(C, schedule.reactant_species, schedule.reactant_orders, reactant_first, reactant_last, idx)
            dq_dC += dk_eff_dC * reactant_product
            scheduled_add_stoich!(J, schedule, ir, species_j, dq_dC)
        end
        if pdep_i !== nothing
            for species_j in axes(J, 2)
                species_j in schedule.reactant_species[reactant_first:reactant_last] && continue
                dq_dC = dk_base_dP * effective_pressure_derivative_C(reactions, species_j, T, logT, rec_RcalT, pdep_i) * reactant_product
                dq_dC == 0.0 && continue
                scheduled_add_stoich!(J, schedule, ir, species_j, dq_dC)
            end
        end
        if ir >= reactions.index_three_body_start
            for idx in schedule.efficiency_ptr[ir]:schedule.efficiency_ptr[ir + 1] - 1
                species_j = schedule.efficiency_species[idx]
                dq_dC = dk_dM * schedule.efficiency_coeffs[idx] * reactant_product
                scheduled_add_stoich!(J, schedule, ir, species_j, dq_dC)
            end
        end

        if ir >= reactions.index_rev_start
            kr[ir] = scheduled_reverse_rate_constant_from_kf(reactions, schedule, kf[ir], g_RT, T, ir)
            product_first = schedule.product_ptr[ir]
            product_last = schedule.product_ptr[ir + 1] - 1
            product_product = scheduled_product(C, schedule.product_species, schedule.product_orders, product_first, product_last)
            for idx in product_first:product_last
                species_j = schedule.product_species[idx]
                dq_dC = kr[ir] * scheduled_product_derivative(C, schedule.product_species, schedule.product_orders, product_first, product_last, idx)
                scheduled_add_stoich!(J, schedule, ir, species_j, -dq_dC)
            end
            if ir >= reactions.index_three_body_start
                for idx in schedule.efficiency_ptr[ir]:schedule.efficiency_ptr[ir + 1] - 1
                    species_j = schedule.efficiency_species[idx]
                    dk_reverse_dC = scheduled_reverse_rate_constant_from_kf(
                        reactions,
                        schedule,
                        dk_dM * schedule.efficiency_coeffs[idx],
                        g_RT,
                        T,
                        ir,
                    )
                    scheduled_add_stoich!(J, schedule, ir, species_j, -dk_reverse_dC * product_product)
                end
            end
            if pdep_i !== nothing
                for species_j in axes(J, 2)
                    dk_eff_dC = dk_base_dP * effective_pressure_derivative_C(reactions, species_j, T, logT, rec_RcalT, pdep_i)
                    dk_reverse_dC = scheduled_reverse_rate_constant_from_kf(reactions, schedule, dk_eff_dC, g_RT, T, ir)
                    scheduled_add_stoich!(J, schedule, ir, species_j, -dk_reverse_dC * product_product)
                end
            end
        end
    end

    return nothing
end

"""
    production_rate_temperature_derivative!(dwdot_dT, kf, kr, C, g_RT, reactions, T)

Evaluate the analytical temperature derivative of molar species production
rates at fixed concentrations and fixed thermodynamic state.
"""
function production_rate_temperature_derivative!(
    dwdot_dT,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    T,
    dg_RT_dT = nothing,
)
    dwdot_dT .= 0.0
    n_reactions = size(reactions.arrhenius_coeffs, 1)
    n_reactions == 0 && return nothing

    logT = safe_log(T)
    rec_RcalT = 4184.0 / Arrhenius.R / T
    arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
    kr .= 0.0

    for ir in 1:n_reactions
        pdep_i = pressure_dependent_local_index(reactions, ir)
        k_base = kf[ir]
        dk_base_dT = arrhenius_rate_temperature_derivative(reactions.arrhenius_coeffs, ir, T, k_base)
        if pdep_i !== nothing
            k_base, dk_base_dT = pressure_dependent_base_rate_temperature_derivative(reactions, C, T, logT, rec_RcalT, pdep_i)
        end
        k_eff, dk_eff_dT = effective_rate_and_temperature_derivative(
            reactions,
            k_base,
            dk_base_dT,
            C,
            T,
            logT,
            rec_RcalT,
            ir,
        )
        kf[ir] = k_eff

        reactant_product = concentration_product(C, reactions.reactant_orders, ir)
        dq_dT = dk_eff_dT * reactant_product

        if ir >= reactions.index_rev_start
            kr[ir] = reverse_rate_constant(reactions, kf, g_RT, T, ir)
            dkr_dT = reverse_rate_temperature_derivative_from_kf(reactions, kr[ir], k_eff, dk_eff_dT, dg_RT_dT, T, ir)
            product_product = concentration_product(C, reactions.product_stoich_coeffs, ir - reactions.index_rev_start + 1)
            dq_dT -= dkr_dT * product_product
        end

        dq_dT == 0.0 && continue
        for species_i in eachindex(dwdot_dT)
            nu = sparse_column_value(reactions.vk, ir, species_i)
            nu == 0.0 && continue
            dwdot_dT[species_i] += nu * dq_dT
        end
    end

    return nothing
end

function production_rate_temperature_derivative!(
    dwdot_dT,
    kf,
    kr,
    C,
    g_RT,
    reactions::ReorderedKinetics,
    schedule::ProductionJacobianSchedule,
    T,
    dg_RT_dT = nothing,
)
    dwdot_dT .= 0.0
    n_reactions = size(reactions.arrhenius_coeffs, 1)
    n_reactions == 0 && return nothing

    logT = safe_log(T)
    rec_RcalT = 4184.0 / Arrhenius.R / T
    arrheniusLaw!(kf, reactions.arrhenius_coeffs, logT, rec_RcalT)
    kr .= 0.0

    for ir in 1:n_reactions
        pdep_i = pressure_dependent_local_index(reactions, ir)
        k_base = kf[ir]
        dk_base_dT = arrhenius_rate_temperature_derivative(reactions.arrhenius_coeffs, ir, T, k_base)
        if pdep_i !== nothing
            k_base, dk_base_dT = pressure_dependent_base_rate_temperature_derivative(reactions, C, T, logT, rec_RcalT, pdep_i)
        end
        k_eff, dk_eff_dT = effective_rate_and_temperature_derivative(
            reactions,
            k_base,
            dk_base_dT,
            C,
            T,
            logT,
            rec_RcalT,
            ir,
        )
        kf[ir] = k_eff

        reactant_first = schedule.reactant_ptr[ir]
        reactant_last = schedule.reactant_ptr[ir + 1] - 1
        reactant_product = scheduled_product(C, schedule.reactant_species, schedule.reactant_orders, reactant_first, reactant_last)
        dq_dT = dk_eff_dT * reactant_product

        if ir >= reactions.index_rev_start
            kr[ir] = scheduled_reverse_rate_constant_from_kf(reactions, schedule, kf[ir], g_RT, T, ir)
            dkr_dT = reverse_rate_temperature_derivative_from_kf(reactions, kr[ir], k_eff, dk_eff_dT, dg_RT_dT, T, ir)
            product_first = schedule.product_ptr[ir]
            product_last = schedule.product_ptr[ir + 1] - 1
            product_product = scheduled_product(C, schedule.product_species, schedule.product_orders, product_first, product_last)
            dq_dT -= dkr_dT * product_product
        end

        scheduled_add_temperature_stoich!(dwdot_dT, schedule, ir, dq_dT)
    end

    return nothing
end

function g_RT_temperature_derivative!(dg_RT_dT, state::isobaricReactor)
    T = state.T
    for i in eachindex(dg_RT_dT)
        a1 = state.nasa_poly[i, 1]
        a2 = state.nasa_poly[i, 2]
        a3 = state.nasa_poly[i, 3]
        a4 = state.nasa_poly[i, 4]
        a5 = state.nasa_poly[i, 5]
        a6 = state.nasa_poly[i, 6]
        dh_RT_dT = a2 / 2 + 2 * a3 * T / 3 + 3 * a4 * T^2 / 4 + 4 * a5 * T^3 / 5 - a6 / T^2
        ds_R_dT = a1 / T + a2 + a3 * T + a4 * T^2 + a5 * T^3
        dg_RT_dT[i] = dh_RT_dT - ds_R_dT
    end
    return nothing
end

function h_RT_temperature_derivative!(dh_RT_dT, state::isobaricReactor)
    T = state.T
    for i in eachindex(dh_RT_dT)
        a2 = state.nasa_poly[i, 2]
        a3 = state.nasa_poly[i, 3]
        a4 = state.nasa_poly[i, 4]
        a5 = state.nasa_poly[i, 5]
        a6 = state.nasa_poly[i, 6]
        dh_RT_dT[i] = a2 / 2 + 2 * a3 * T / 3 + 3 * a4 * T^2 / 4 + 4 * a5 * T^3 / 5 - a6 / T^2
    end
    return nothing
end

function cp_R_temperature_derivative(state::isobaricReactor, species::Int)
    T = state.T
    return (
        state.nasa_poly[species, 2] +
        2 * state.nasa_poly[species, 3] * T +
        3 * state.nasa_poly[species, 4] * T^2 +
        4 * state.nasa_poly[species, 5] * T^3
    )
end

function concentration_row_contractions!(row_contraction, dwdot_dC, Y, MW)
    n_species = length(Y)
    for i in 1:n_species
        acc = zero(eltype(row_contraction))
        for k in 1:n_species
            acc += dwdot_dC[i, k] * Y[k] / MW[k]
        end
        row_contraction[i] = acc
    end
    return nothing
end

function assemble_isobaric_species_rows!(
    J,
    state::isobaricReactor,
    dwdot_dC,
    dwdot_dT,
    row_contraction,
    wdot,
    u,
)
    n_species = state.gas.n_species
    Y = @view(u[1:n_species])
    concentration_scale = state.p / (Arrhenius.R * state.T)
    inv_rho = inv(state.ρ_mass)

    concentration_row_contractions!(row_contraction, dwdot_dC, Y, state.gas.MW)

    for i in 1:n_species
        for j in 1:n_species
            dmean_MW_dYj = -state.mean_MW^2 / state.gas.MW[j]
            dwdot_dYj = concentration_scale * (
                state.mean_MW * dwdot_dC[i, j] / state.gas.MW[j] +
                dmean_MW_dYj * row_contraction[i]
            )
            dinv_rho_dYj = Arrhenius.R * state.T / (state.p * state.gas.MW[j])
            J[i, j] = state.gas.MW[i] * (dwdot_dYj * inv_rho + wdot[i] * dinv_rho_dYj)
        end
    end

    for i in 1:n_species
        dwdot_total_dT = dwdot_dT[i]
        for j in 1:n_species
            dwdot_total_dT -= dwdot_dC[i, j] * state.C[j] / state.T
        end
        J[i, n_species + 1] = state.gas.MW[i] * (dwdot_total_dT * inv_rho + wdot[i] * inv_rho / state.T)
    end

    return nothing
end

function assemble_isobaric_temperature_row!(
    J,
    state::isobaricReactor,
    dwdot_dC,
    dwdot_dT,
    row_contraction,
    dh_RT_dT,
    wdot,
    u,
)
    n_species = state.gas.n_species
    Y = @view(u[1:n_species])
    concentration_scale = state.p / (Arrhenius.R * state.T)
    prefactor = -Arrhenius.R * state.T / (state.ρ_mass * state.cp_mass)
    enthalpy_source = dot(state.h_RT, wdot)

    enthalpy_row_contraction = zero(eltype(J))
    for i in 1:n_species
        enthalpy_row_contraction += state.h_RT[i] * row_contraction[i]
    end

    for j in 1:n_species
        enthalpy_column_contraction = zero(eltype(J))
        for i in 1:n_species
            enthalpy_column_contraction += state.h_RT[i] * dwdot_dC[i, j]
        end
        dmean_MW_dYj = -state.mean_MW^2 / state.gas.MW[j]
        dwdot_dYj_enthalpy = concentration_scale * (
            state.mean_MW * enthalpy_column_contraction / state.gas.MW[j] +
            dmean_MW_dYj * enthalpy_row_contraction
        )
        dcp_mass_dYj = Arrhenius.R * state.cp_R[j] / state.gas.MW[j]
        dprefactor_dYj = prefactor * (state.mean_MW / state.gas.MW[j] - dcp_mass_dYj / state.cp_mass)
        J[n_species + 1, j] = dprefactor_dYj * enthalpy_source + prefactor * dwdot_dYj_enthalpy
    end

    dwdot_dT_enthalpy = zero(eltype(J))
    for i in 1:n_species
        dwdot_total_dT = dwdot_dT[i]
        for j in 1:n_species
            dwdot_total_dT -= dwdot_dC[i, j] * state.C[j] / state.T
        end
        dwdot_dT_enthalpy += dh_RT_dT[i] * wdot[i] + state.h_RT[i] * dwdot_total_dT
    end

    dcp_mass_dT = zero(eltype(J))
    for i in 1:n_species
        dcp_mass_dT += Y[i] * cp_R_temperature_derivative(state, i) / state.gas.MW[i]
    end
    dcp_mass_dT *= Arrhenius.R
    dprefactor_dT = prefactor * (2 / state.T - dcp_mass_dT / state.cp_mass)
    J[n_species + 1, n_species + 1] = dprefactor_dT * enthalpy_source + prefactor * dwdot_dT_enthalpy

    return nothing
end

"""
    isobaric_species_jacobian!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, wdot, u, p, t)

Assemble the analytical species rows of the isobaric reactor Jacobian. `J` has
shape `(n_species, n_species + 1)` and stores derivatives of `dY/dt` with
respect to `[Y..., T]`.
"""
function isobaric_species_jacobian!(
    J,
    state::isobaricReactor,
    dwdot_dC,
    dwdot_dT,
    dg_RT_dT,
    wdot,
    u,
    p,
    t,
)
    Y = @view(u[1:state.gas.n_species])
    updateReactor!(state, Y, u[end], p[1])
    wdot_func!(wdot, state.kf, state.kr, state.C, state.g_RT, state.gas.kinetics, state.T)
    production_rate_jacobian_C!(dwdot_dC, state.kf, state.kr, state.C, state.g_RT, state.gas.kinetics, state.T)
    g_RT_temperature_derivative!(dg_RT_dT, state)
    production_rate_temperature_derivative!(
        dwdot_dT,
        state.kf,
        state.kr,
        state.C,
        state.g_RT,
        state.gas.kinetics,
        state.T,
        dg_RT_dT,
    )

    return assemble_isobaric_species_rows!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, wdot, u)
end

function isobaric_species_jacobian!(
    J,
    state::isobaricReactor,
    schedule::ProductionJacobianSchedule,
    dwdot_dC,
    dwdot_dT,
    dg_RT_dT,
    wdot,
    u,
    p,
    t,
)
    Y = @view(u[1:state.gas.n_species])
    updateReactor!(state, Y, u[end], p[1])
    wdot_func!(wdot, state.kf, state.kr, state.C, state.g_RT, state.gas.kinetics, state.T)
    production_rate_jacobian_C!(dwdot_dC, state.kf, state.kr, state.C, state.g_RT, state.gas.kinetics, schedule, state.T)
    g_RT_temperature_derivative!(dg_RT_dT, state)
    production_rate_temperature_derivative!(
        dwdot_dT,
        state.kf,
        state.kr,
        state.C,
        state.g_RT,
        state.gas.kinetics,
        schedule,
        state.T,
        dg_RT_dT,
    )

    return assemble_isobaric_species_rows!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, wdot, u)
end

function isobaric_species_jacobian!(
    J,
    state::isobaricReactor,
    schedule::ProductionJacobianSchedule,
    kernel::GeneratedProductionJacobianKernel,
    dwdot_dC,
    dwdot_dT,
    dg_RT_dT,
    wdot,
    u,
    p,
    t,
)
    Y = @view(u[1:state.gas.n_species])
    updateReactor!(state, Y, u[end], p[1])
    production_rate_jacobian_C_and_wdot!(
        dwdot_dC,
        wdot,
        state.kf,
        state.kr,
        state.C,
        state.g_RT,
        state.gas.kinetics,
        kernel,
        state.T,
    )
    g_RT_temperature_derivative!(dg_RT_dT, state)
    production_rate_temperature_derivative!(
        dwdot_dT,
        state.kf,
        state.kr,
        state.C,
        state.g_RT,
        state.gas.kinetics,
        kernel,
        state.T,
        dg_RT_dT,
    )

    return assemble_isobaric_species_rows!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, wdot, u)
end

"""
    isobaric_reactor_jacobian!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, dh_RT_dT, wdot, u, p, t)

Assemble the analytical dense isobaric reactor Jacobian for `[Y..., T]`.
"""
function isobaric_reactor_jacobian!(
    J,
    state::isobaricReactor,
    dwdot_dC,
    dwdot_dT,
    dg_RT_dT,
    dh_RT_dT,
    wdot,
    u,
    p,
    t,
)
    n_species = state.gas.n_species
    species_rows = @view(J[1:n_species, :])
    isobaric_species_jacobian!(
        species_rows,
        state,
        dwdot_dC,
        dwdot_dT,
        dg_RT_dT,
        wdot,
        u,
        p,
        t,
    )
    h_RT_temperature_derivative!(dh_RT_dT, state)

    return assemble_isobaric_temperature_row!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, dh_RT_dT, wdot, u)
end

function isobaric_reactor_jacobian!(
    J,
    state::isobaricReactor,
    schedule::ProductionJacobianSchedule,
    dwdot_dC,
    dwdot_dT,
    dg_RT_dT,
    dh_RT_dT,
    wdot,
    u,
    p,
    t,
)
    n_species = state.gas.n_species
    species_rows = @view(J[1:n_species, :])
    isobaric_species_jacobian!(
        species_rows,
        state,
        schedule,
        dwdot_dC,
        dwdot_dT,
        dg_RT_dT,
        wdot,
        u,
        p,
        t,
    )
    h_RT_temperature_derivative!(dh_RT_dT, state)

    return assemble_isobaric_temperature_row!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, dh_RT_dT, wdot, u)
end

function isobaric_reactor_jacobian!(
    J,
    state::isobaricReactor,
    schedule::ProductionJacobianSchedule,
    kernel::GeneratedProductionJacobianKernel,
    dwdot_dC,
    dwdot_dT,
    dg_RT_dT,
    dh_RT_dT,
    wdot,
    u,
    p,
    t,
)
    n_species = state.gas.n_species
    species_rows = @view(J[1:n_species, :])
    isobaric_species_jacobian!(
        species_rows,
        state,
        schedule,
        kernel,
        dwdot_dC,
        dwdot_dT,
        dg_RT_dT,
        wdot,
        u,
        p,
        t,
    )
    h_RT_temperature_derivative!(dh_RT_dT, state)

    return assemble_isobaric_temperature_row!(J, state, dwdot_dC, dwdot_dT, dg_RT_dT, dh_RT_dT, wdot, u)
end

export IsobaricJacobianWorkspace,
    AnalyticalIsobaricJacobianWorkspace,
    GeneratedAnalyticalIsobaricJacobianWorkspace,
    GeneratedIsobaricReactor,
    isobaric_jacobian_prototype,
    jacobian!,
    production_rate_jacobian_C!,
    production_rate_temperature_derivative!,
    isobaric_species_jacobian!,
    isobaric_reactor_jacobian!
