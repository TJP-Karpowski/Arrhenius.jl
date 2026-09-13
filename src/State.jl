mutable struct GasState{Sol<:Solution}
    gas::Sol
    T::Float64
    p::Float64
    X::Vector{Float64}
    Y::Vector{Float64}
    C::Vector{Float64}
    mean_MW::Float64
    ρ_mass::Float64
    cp_R::Vector{Float64}
    h_RT::Vector{Float64}
    s_R::Vector{Float64}
    s0_R::Vector{Float64}
    g_RT::Vector{Float64}
    u_RT::Vector{Float64}
end

"""
    EquilibriumResult

Convergence summary returned by `equilibrate!`.

Fields:

- `mode`: equilibrium mode that was solved, currently `:TP` or `:HP`.
- `converged`: true when the nonlinear residual met the requested tolerance.
- `iterations`: total nonlinear iterations used by the solve.
- `residual_norm`: final scaled residual norm.
"""
struct EquilibriumResult
    mode::Symbol
    converged::Bool
    iterations::Int
    residual_norm::Float64
end

"""
    EquilibriumWorkspace(gas)

Reusable scratch storage for repeated ideal-gas equilibrium solves with one
`Solution`. Pass it to `equilibrate!(state, workspace, mode)` to reduce
per-solve allocations while keeping the same equilibrium algorithm.
"""
mutable struct EquilibriumWorkspace{Sol<:Solution}
    gas::Sol
    active_elements::Vector{Int}
    active_species::Vector{Int}
    b::Vector{Float64}
    A::Matrix{Float64}
    g0_RT::Vector{Float64}
    y::Vector{Float64}
    y0::Vector{Float64}
    λ::Vector{Float64}
    λ0::Vector{Float64}
    n::Vector{Float64}
    residual::Vector{Float64}
    trial_residual::Vector{Float64}
    J::Matrix{Float64}
    trial_y::Vector{Float64}
    trial_λ::Vector{Float64}
    Xeq::Vector{Float64}
    element_moles::Vector{Float64}
end

function EquilibriumWorkspace(gas::Solution)
    n_elements = length(gas.elements)
    max_unknowns = gas.n_species + n_elements
    return EquilibriumWorkspace(
        gas,
        Int[],
        Int[],
        zeros(n_elements),
        zeros(n_elements, gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(n_elements),
        zeros(n_elements),
        zeros(gas.n_species),
        zeros(max_unknowns),
        zeros(max_unknowns),
        zeros(max_unknowns, max_unknowns),
        zeros(gas.n_species),
        zeros(n_elements),
        zeros(gas.n_species),
        zeros(n_elements),
    )
end

"""
    GasState(gas; T=300.0, p=one_atm, X=nothing, Y=nothing)

Create a mutable thermodynamic state for a `Solution`. The state owns cached
composition, concentration, density, and ideal-gas thermodynamic property arrays.
"""
function GasState(gas::Solution; T = 300.0, p = one_atm, X = nothing, Y = nothing)
    initial_X = X === nothing && Y === nothing ? fill(1 / gas.n_species, gas.n_species) : zeros(gas.n_species)
    state = GasState(
        gas,
        Float64(T),
        Float64(p),
        copy(initial_X),
        zeros(gas.n_species),
        zeros(gas.n_species),
        0.0,
        0.0,
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
        zeros(gas.n_species),
    )
    if Y !== nothing
        set_TPY!(state, T, p, Y)
    else
        set_TPX!(state, T, p, X === nothing ? initial_X : X)
    end
    return state
end

function normalize_composition!(values)
    total = sum(values)
    total > 0 || error("Composition must have a positive sum")
    values ./= total
    return values
end

function composition_vector(gas::Solution, composition)
    values = zeros(gas.n_species)
    if composition isa AbstractVector
        length(composition) == gas.n_species || error("Composition vector length $(length(composition)) does not match $(gas.n_species) species")
        values .= composition
    elseif composition isa AbstractDict
        for (species, amount) in composition
            idx = species_index(gas, String(species))
            idx === nothing && error("Species '$species' is not present in this solution")
            values[idx] = Float64(amount)
        end
    elseif composition isa AbstractString
        for token in split(composition, ',')
            item = strip(token)
            isempty(item) && continue
            parts = split(item, ':')
            length(parts) == 2 || error("Composition entries must use 'species:amount', got '$item'")
            species = strip(parts[1])
            amount = parse(Float64, strip(parts[2]))
            idx = species_index(gas, species)
            idx === nothing && error("Species '$species' is not present in this solution")
            values[idx] = amount
        end
    else
        try
            for (species, amount) in composition
                idx = species_index(gas, String(species))
                idx === nothing && error("Species '$species' is not present in this solution")
                values[idx] = Float64(amount)
            end
        catch err
            err isa MethodError || rethrow()
            error("Unsupported composition type: $(typeof(composition))")
        end
    end
    any(<(0), values) && error("Composition cannot contain negative entries")
    return normalize_composition!(values)
end

function update_state_cache!(state::GasState)
    state.mean_MW = dot(state.X, state.gas.MW)
    state.Y .= state.X .* state.gas.MW ./ state.mean_MW
    state.ρ_mass = state.p / Arrhenius.R / state.T * state.mean_MW
    state.C .= state.X .* state.p ./ (Arrhenius.R * state.T)
    state.cp_R .= cal_cp_R(state.gas, state.T, state.p, state.X)
    state.h_RT .= cal_h_RT(state.gas, state.T, state.p, state.X)
    state.s0_R .= cal_s0_R(state.gas, state.T, state.p, state.X)
    state.s_R .= cal_s_R(state.gas, state.T, state.p, state.X)
    state.g_RT .= state.h_RT .- state.s_R
    state.u_RT .= state.h_RT .- 1
    return state
end

"""
    set_TPX!(state, T, p, X)

Set temperature, pressure, and mole composition. `X` may be a vector, dictionary,
pair collection, or a string such as `"CH4:1,O2:2,N2:7.52"`.
"""
function set_TPX!(state::GasState, T::Real, p::Real, X)
    state.T = Float64(T)
    state.p = Float64(p)
    state.T > 0 || error("Temperature must be positive")
    state.p > 0 || error("Pressure must be positive")
    state.X .= composition_vector(state.gas, X)
    return update_state_cache!(state)
end

"""
    set_TPY!(state, T, p, Y)

Set temperature, pressure, and mass composition. `Y` accepts the same composition
formats as `set_TPX!`, but amounts are interpreted on a mass basis.
"""
function set_TPY!(state::GasState, T::Real, p::Real, Y)
    Y_values = composition_vector(state.gas, Y)
    X_values = Y_values ./ state.gas.MW
    normalize_composition!(X_values)
    return set_TPX!(state, T, p, X_values)
end

"""
    set_TP!(state, T, p)

Set temperature and pressure while keeping the current composition.
"""
function set_TP!(state::GasState, T::Real, p::Real)
    state.T = Float64(T)
    state.p = Float64(p)
    state.T > 0 || error("Temperature must be positive")
    state.p > 0 || error("Pressure must be positive")
    return update_state_cache!(state)
end

"""Return the state temperature in K."""
temperature(state::GasState) = state.T

"""Return the state pressure in Pa."""
pressure(state::GasState) = state.p

"""Return a copy of the species mole fractions."""
mole_fractions(state::GasState) = copy(state.X)

"""Return a copy of the species mass fractions."""
mass_fractions(state::GasState) = copy(state.Y)

"""Return a copy of the species molar concentrations in mol/m^3."""
concentrations(state::GasState) = copy(state.C)

"""Return the mass density in kg/m^3."""
density(state::GasState) = state.ρ_mass

"""Return the mixture mean molecular weight in kg/kmol-compatible Arrhenius units."""
mean_molecular_weight(state::GasState) = state.mean_MW

"""Return species constant-pressure heat capacities on a molar basis."""
species_cp_mole(state::GasState) = state.cp_R .* Arrhenius.R

"""Return species constant-pressure heat capacities on a mass basis."""
species_cp_mass(state::GasState) = species_cp_mole(state) ./ state.gas.MW

"""Return mixture constant-pressure heat capacity on a molar basis."""
cp_mole(state::GasState) = dot(state.X, state.cp_R) * Arrhenius.R

"""Return mixture constant-pressure heat capacity on a mass basis."""
cp_mass(state::GasState) = cp_mole(state) / state.mean_MW

"""Return species enthalpies on a molar basis."""
species_enthalpies_mole(state::GasState) = state.h_RT .* (Arrhenius.R * state.T)

"""Return species enthalpies on a mass basis."""
species_enthalpies_mass(state::GasState) = species_enthalpies_mole(state) ./ state.gas.MW

"""Return mixture enthalpy on a molar basis."""
enthalpy_mole(state::GasState) = dot(state.X, state.h_RT) * Arrhenius.R * state.T

"""Return mixture enthalpy on a mass basis."""
enthalpy_mass(state::GasState) = enthalpy_mole(state) / state.mean_MW

"""Return species internal energies on a molar basis."""
species_internal_energies_mole(state::GasState) = state.u_RT .* (Arrhenius.R * state.T)

"""Return species internal energies on a mass basis."""
species_internal_energies_mass(state::GasState) = species_internal_energies_mole(state) ./ state.gas.MW

"""Return mixture internal energy on a molar basis."""
internal_energy_mole(state::GasState) = dot(state.X, state.u_RT) * Arrhenius.R * state.T

"""Return mixture internal energy on a mass basis."""
internal_energy_mass(state::GasState) = internal_energy_mole(state) / state.mean_MW

"""Return species entropies on a molar basis."""
species_entropies_mole(state::GasState) = state.s_R .* Arrhenius.R

"""Return species entropies on a mass basis."""
species_entropies_mass(state::GasState) = species_entropies_mole(state) ./ state.gas.MW

"""Return mixture entropy on a molar basis."""
entropy_mole(state::GasState) = dot(state.X, state.s_R) * Arrhenius.R

"""Return mixture entropy on a mass basis."""
entropy_mass(state::GasState) = entropy_mole(state) / state.mean_MW

"""Return species Gibbs free energies on a molar basis."""
species_gibbs_mole(state::GasState) = state.g_RT .* (Arrhenius.R * state.T)

"""Return species Gibbs free energies on a mass basis."""
species_gibbs_mass(state::GasState) = species_gibbs_mole(state) ./ state.gas.MW

"""Return mixture Gibbs free energy on a molar basis."""
gibbs_mole(state::GasState) = dot(state.X, state.g_RT) * Arrhenius.R * state.T

"""Return mixture Gibbs free energy on a mass basis."""
gibbs_mass(state::GasState) = gibbs_mole(state) / state.mean_MW

"""Write species net production rates for the current state into `wdot`."""
function net_production_rates!(wdot, state::GasState)
    length(wdot) == state.gas.n_species || error("wdot length $(length(wdot)) does not match $(state.gas.n_species) species")
    kf = zeros(eltype(wdot), state.gas.n_reactions)
    kr = zeros(eltype(wdot), state.gas.n_reactions)
    # Equilibrium constants are formed from standard-state Gibbs energies.
    # `state.g_RT` contains composition-dependent chemical potentials for the
    # public thermodynamic state API, so it must not be used here.
    g0_RT = state.h_RT .- state.s0_R
    wdot_func!(wdot, kf, kr, state.C, g0_RT, state.gas.kinetics, state.T)
    return wdot
end

"""Return species net production rates for the current state."""
function net_production_rates(state::GasState)
    wdot = zeros(state.gas.n_species)
    return net_production_rates!(wdot, state)
end

function active_equilibrium_sets(state::GasState; element_atol = 1.0e-14)
    element_moles = state.gas.ele_matrix * state.X
    return active_equilibrium_sets(state, element_moles; element_atol = element_atol)
end

function active_equilibrium_sets(state::GasState, element_moles; element_atol = 1.0e-14)
    active_elements = findall(>(element_atol), element_moles)
    isempty(active_elements) && error("Cannot equilibrate a state with no active elements")

    inactive_elements = setdiff(1:length(state.gas.elements), active_elements)
    active_species = Int[]
    for i in 1:state.gas.n_species
        if all(e -> state.gas.ele_matrix[e, i] == 0.0, inactive_elements) &&
           any(e -> state.gas.ele_matrix[e, i] > 0.0, active_elements)
            push!(active_species, i)
        end
    end
    isempty(active_species) && error("No species are compatible with the active element set")
    return active_elements, active_species, element_moles[active_elements]
end

function active_equilibrium_sets!(workspace::EquilibriumWorkspace, state::GasState, element_moles; element_atol = 1.0e-14)
    workspace.gas === state.gas || error("EquilibriumWorkspace was constructed for a different Solution")
    empty!(workspace.active_elements)
    empty!(workspace.active_species)
    for e in eachindex(element_moles)
        element_moles[e] > element_atol && push!(workspace.active_elements, e)
    end
    isempty(workspace.active_elements) && error("Cannot equilibrate a state with no active elements")

    for i in 1:state.gas.n_species
        compatible = true
        participates = false
        for e in 1:length(state.gas.elements)
            active = e in workspace.active_elements
            amount = state.gas.ele_matrix[e, i]
            compatible &= active || amount == 0.0
            participates |= active && amount > 0.0
        end
        compatible && participates && push!(workspace.active_species, i)
    end
    isempty(workspace.active_species) && error("No species are compatible with the active element set")

    for (i, e) in enumerate(workspace.active_elements)
        workspace.b[i] = element_moles[e]
    end
    return workspace.active_elements, workspace.active_species, @view(workspace.b[1:length(workspace.active_elements)])
end

function active_equilibrium_sets!(workspace::EquilibriumWorkspace, state::GasState; element_atol = 1.0e-14)
    workspace.element_moles .= state.gas.ele_matrix * state.X
    return active_equilibrium_sets!(workspace, state, workspace.element_moles; element_atol = element_atol)
end

function equilibrium_standard_gibbs_RT(state::GasState)
    return cal_h_RT(state.gas, state.T, state.p, state.X) .-
           cal_s0_R(state.gas, state.T, state.p, state.X)
end

function equilibrium_residual_jacobian(y, λ, A, b, g0_RT, logp)
    ns = length(y)
    ne = length(b)
    residual = zeros(ns + ne)
    J = zeros(ns + ne, ns + ne)
    n = zeros(ns)
    equilibrium_residual_jacobian!(residual, J, n, y, λ, A, b, g0_RT, logp)
    return residual, J, n
end

function equilibrium_residual_jacobian!(residual, J, n, y, λ, A, b, g0_RT, logp)
    ns = length(y)
    ne = length(b)
    for i in 1:ns
        n[i] = exp(y[i])
    end
    N = sum(n)

    for i in 1:ns
        Xi = n[i] / N
        residual[i] = g0_RT[i] + log(Xi) + logp - dot(@view(A[:, i]), λ)
    end
    for e in 1:ne
        element_residual = -b[e]
        for j in 1:ns
            element_residual += A[e, j] * n[j]
        end
        residual[ns + e] = element_residual
    end

    for i in 1:ns
        for j in 1:ns
            J[i, j] = (i == j ? 1.0 : 0.0) - n[j] / N
        end
        for e in 1:ne
            J[i, ns + e] = -A[e, i]
        end
    end
    for e in 1:ne
        for j in 1:ns
            J[ns + e, j] = A[e, j] * n[j]
        end
    end
    return residual, J, n
end

function initial_equilibrium_unknowns(state::GasState, A, active_species, g0_RT, logp)
    ns = length(active_species)
    n0 = max.(state.X[active_species], 1.0e-30)
    normalize_composition!(n0)
    y = log.(n0)
    chemical_potential = g0_RT .+ log.(n0 ./ sum(n0)) .+ logp
    λ = transpose(A) \ chemical_potential
    length(λ) == size(A, 1) || (λ = zeros(size(A, 1)))
    return y, λ
end

function equilibrium_element_potential_residual(λ, A, b, g0_RT, logp, dropped_element)
    z = vec(transpose(A) * λ) .- g0_RT .- logp
    zmax = maximum(z)
    scaled_X = exp.(z .- zmax)
    scaled_element_moles = A * scaled_X

    residual = zeros(length(b))
    residual[1] = zmax + log(sum(scaled_X))
    row = 2
    for e in eachindex(b)
        e == dropped_element && continue
        residual[row] = log(scaled_element_moles[e] / scaled_element_moles[dropped_element]) -
                        log(b[e] / b[dropped_element])
        row += 1
    end
    X = scaled_X .* exp(-residual[1])
    return residual, X
end

function equilibrium_element_potential_jacobian(λ, A, b, g0_RT, logp, dropped_element, residual)
    z = vec(transpose(A) * λ) .- g0_RT .- logp
    scaled_X = exp.(z .- maximum(z))
    scaled_element_moles = A * scaled_X

    J = zeros(length(residual), length(λ))
    J[1, :] .= vec(A * scaled_X) ./ sum(scaled_X)
    row = 2
    for e in eachindex(b)
        e == dropped_element && continue
        for j in eachindex(λ)
            numerator_e = sum(A[e, i] * scaled_X[i] * A[j, i] for i in axes(A, 2))
            numerator_dropped = sum(A[dropped_element, i] * scaled_X[i] * A[j, i] for i in axes(A, 2))
            J[row, j] = numerator_e / scaled_element_moles[e] -
                        numerator_dropped / scaled_element_moles[dropped_element]
        end
        row += 1
    end
    return J
end

function solve_equilibrium_element_potentials(λ0, A, b, g0_RT, logp; rtol = 1.0e-10, maxiters = 100)
    dropped_element = argmax(b)
    scale = max(norm(b ./ sum(b)), 1.0)
    best_λ = copy(λ0)
    best_X = zeros(length(g0_RT))
    best_norm = Inf
    best_iterations = 0

    candidates = (λ0, zeros(length(λ0)))
    for candidate in candidates
        λ = copy(candidate)
        for iter in 1:maxiters
            residual, X = equilibrium_element_potential_residual(λ, A, b, g0_RT, logp, dropped_element)
            residual_norm = norm(residual) / scale
            if residual_norm < best_norm
                best_λ .= λ
                best_X .= X
                best_norm = residual_norm
                best_iterations = iter
            end
            residual_norm <= rtol && return best_λ, best_X, iter, residual_norm

            J = equilibrium_element_potential_jacobian(λ, A, b, g0_RT, logp, dropped_element, residual)
            delta = try
                -(J \ residual)
            catch err
                try
                    -(qr(J) \ residual)
                catch
                    -(pinv(J) * residual)
                end
            end

            current_norm = norm(residual)
            accepted = false
            for exponent in 0:30
                damping = exp2(-exponent)
                trial_λ = λ .+ damping .* delta
                if all(isfinite, trial_λ)
                    trial_residual, _ = equilibrium_element_potential_residual(trial_λ, A, b, g0_RT, logp, dropped_element)
                    if all(isfinite, trial_residual) && norm(trial_residual) < current_norm
                        λ = trial_λ
                        accepted = true
                        break
                    end
                end
            end
            accepted || break
        end
    end

    best_norm <= rtol && return best_λ, best_X, best_iterations, best_norm
    error("TP equilibrium element-potential solve did not converge after $best_iterations iterations; residual_norm=$best_norm")
end

function tp_equilibrium!(state::GasState, active_elements, active_species, b; rtol = 1.0e-10, maxiters = 100)
    workspace = EquilibriumWorkspace(state.gas)
    return tp_equilibrium!(state, workspace, active_elements, active_species, b; rtol = rtol, maxiters = maxiters)
end

function tp_equilibrium!(state::GasState, workspace::EquilibriumWorkspace, active_elements, active_species, b; rtol = 1.0e-10, maxiters = 100)
    workspace.gas === state.gas || error("EquilibriumWorkspace was constructed for a different Solution")
    ne = length(active_elements)
    ns = length(active_species)
    A = @view(workspace.A[1:ne, 1:ns])
    g0_RT = @view(workspace.g0_RT[1:ns])
    for (j, species) in enumerate(active_species)
        for (eidx, element) in enumerate(active_elements)
            A[eidx, j] = state.gas.ele_matrix[element, species]
        end
    end
    standard_gibbs = equilibrium_standard_gibbs_RT(state)
    for (j, species) in enumerate(active_species)
        g0_RT[j] = standard_gibbs[species]
    end
    logp = log(state.p / Arrhenius.one_atm)
    y0_tmp, λ0_tmp = initial_equilibrium_unknowns(state, A, active_species, g0_RT, logp)
    y = @view(workspace.y[1:ns])
    y0 = @view(workspace.y0[1:ns])
    λ = @view(workspace.λ[1:ne])
    λ0 = @view(workspace.λ0[1:ne])
    y0 .= y0_tmp
    λ0 .= λ0_tmp
    y .= y0
    λ .= λ0
    scale = max(norm(b), 1.0)
    residual_norm = Inf
    converged = false
    iterations = 0
    n = @view(workspace.n[1:ns])
    trial_y = @view(workspace.trial_y[1:ns])
    trial_λ = @view(workspace.trial_λ[1:ne])
    residual = @view(workspace.residual[1:ns+ne])
    trial_residual = @view(workspace.trial_residual[1:ns+ne])
    J = @view(workspace.J[1:ns+ne, 1:ns+ne])

    try
        λ_reduced, X_active, reduced_iterations, element_residual_norm = solve_equilibrium_element_potentials(
            λ0,
            A,
            b,
            g0_RT,
            logp;
            rtol = max(rtol, 1.0e-10),
            maxiters = maxiters,
        )
        n_total = dot(b, A * X_active) / dot(A * X_active, A * X_active)
        if n_total > 0
            λ .= λ_reduced
            n .= n_total .* X_active
            y .= log.(max.(n, 1.0e-300))
            equilibrium_residual_jacobian!(residual, J, n, y, λ, A, b, g0_RT, logp)
            residual_norm = max(norm(residual) / scale, element_residual_norm)
            converged = residual_norm <= max(sqrt(rtol), 1.0e-8)
            iterations = reduced_iterations
        end
    catch
        y .= y0
        λ .= λ0
    end

    for iter in 1:maxiters
        converged && break
        iterations += 1
        equilibrium_residual_jacobian!(residual, J, n, y, λ, A, b, g0_RT, logp)
        residual_norm = norm(residual) / scale
        if residual_norm <= rtol
            converged = true
            break
        end

        delta = try
            -(J \ residual)
        catch
            -(qr(J) \ residual)
        end

        current_norm = norm(residual)
        accepted = false
        for exponent in 0:30
            damping = exp2(-exponent)
            for i in eachindex(y)
                trial_y[i] = clamp(y[i] + damping * delta[i], -745.0, 700.0)
            end
            for i in eachindex(λ)
                trial_λ[i] = λ[i] + damping * delta[length(y) + i]
            end
            if all(isfinite, trial_y) && all(isfinite, trial_λ)
                equilibrium_residual_jacobian!(trial_residual, J, n, trial_y, trial_λ, A, b, g0_RT, logp)
                if all(isfinite, trial_residual) && norm(trial_residual) < current_norm
                    y .= trial_y
                    λ .= trial_λ
                    accepted = true
                    break
                end
            end
        end
        accepted || break
    end

    equilibrium_residual_jacobian!(residual, J, n, y, λ, A, b, g0_RT, logp)
    residual_norm = norm(residual) / scale
    converged = converged || residual_norm <= rtol
    converged || error("TP equilibrium did not converge after $iterations iterations; residual_norm=$residual_norm")

    Xeq = workspace.Xeq
    fill!(Xeq, 0.0)
    Xeq[active_species] .= n ./ sum(n)
    set_TPX!(state, state.T, state.p, Xeq)
    return EquilibriumResult(:TP, true, iterations, residual_norm), active_species, n
end

function extensive_enthalpy_mole(state::GasState, active_species, n)
    h_mole = species_enthalpies_mole(state)
    return dot(n, @view(h_mole[active_species]))
end

function equilibrate_TP!(state::GasState; rtol = 1.0e-10, maxiters = 100)
    active_elements, active_species, b = active_equilibrium_sets(state)
    result, _, _ = tp_equilibrium!(state, active_elements, active_species, b; rtol = rtol, maxiters = maxiters)
    return result
end

function equilibrate_TP!(state::GasState, workspace::EquilibriumWorkspace; rtol = 1.0e-10, maxiters = 100)
    active_elements, active_species, b = active_equilibrium_sets!(workspace, state)
    result, _, _ = tp_equilibrium!(state, workspace, active_elements, active_species, b; rtol = rtol, maxiters = maxiters)
    return result
end

function hp_trial!(state::GasState, T, p, active_elements, active_species, b; rtol, maxiters)
    set_TP!(state, T, p)
    result, species, n = tp_equilibrium!(state, active_elements, active_species, b; rtol = rtol, maxiters = maxiters)
    return extensive_enthalpy_mole(state, species, n), result
end

function hp_trial!(state::GasState, workspace::EquilibriumWorkspace, T, p, active_elements, active_species, b; rtol, maxiters)
    set_TP!(state, T, p)
    result, species, n = tp_equilibrium!(state, workspace, active_elements, active_species, b; rtol = rtol, maxiters = maxiters)
    return extensive_enthalpy_mole(state, species, n), result
end

function equilibrate_HP!(state::GasState; rtol = 1.0e-10, maxiters = 100, temperature_atol = 1.0e-8)
    workspace = EquilibriumWorkspace(state.gas)
    return equilibrate_HP!(state, workspace; rtol = rtol, maxiters = maxiters, temperature_atol = temperature_atol)
end

function equilibrate_HP!(state::GasState, workspace::EquilibriumWorkspace; rtol = 1.0e-10, maxiters = 100, temperature_atol = 1.0e-8)
    p = state.p
    workspace.element_moles .= state.gas.ele_matrix * state.X
    active_elements, active_species, b = active_equilibrium_sets!(workspace, state, workspace.element_moles)
    target_h = enthalpy_mole(state)
    enthalpy_scale = max(abs(target_h), 1.0)

    function residual_at!(T)
        h, result = hp_trial!(state, workspace, T, p, active_elements, active_species, b; rtol = rtol, maxiters = maxiters)
        return h - target_h, result
    end

    T0 = state.T
    # Avoid starting the nested TP solve in the poorly conditioned,
    # near-boundary low-temperature regime. Adiabatic combustion roots are
    # normally above this floor; the expansion loop can lower it if bracketing
    # an endothermic or weakly reacting case requires that.
    low = max(T0, 1000.0)
    high = max(2.0 * T0, low + 100.0)
    f_low, low_result = residual_at!(low)
    f_high, high_result = residual_at!(high)
    bracket_iterations = 0
    while signbit(f_low) == signbit(f_high) && bracket_iterations < 20
        bracket_iterations += 1
        if abs(f_low) < abs(f_high)
            low = max(50.0, low * 0.5)
            f_low, low_result = residual_at!(low)
        else
            high *= 1.5
            f_high, high_result = residual_at!(high)
        end
    end
    signbit(f_low) == signbit(f_high) && error("HP equilibrium could not bracket temperature; residuals=($f_low, $f_high)")

    best_T = abs(f_low) < abs(f_high) ? low : high
    best_f = abs(f_low) < abs(f_high) ? f_low : f_high
    best_result = abs(f_low) < abs(f_high) ? low_result : high_result
    temperature_iterations = 0

    for iter in 1:maxiters
        temperature_iterations = iter
        mid = 0.5 * (low + high)
        f_mid, mid_result = residual_at!(mid)
        if abs(f_mid) < abs(best_f)
            best_T = mid
            best_f = f_mid
            best_result = mid_result
        end
        if abs(f_mid) / enthalpy_scale <= 0.01 * rtol || abs(high - low) <= temperature_atol
            best_T = mid
            best_f = f_mid
            best_result = mid_result
            break
        end
        if signbit(f_mid) == signbit(f_low)
            low = mid
            f_low = f_mid
        else
            high = mid
            f_high = f_mid
        end
    end

    final_f, final_result = residual_at!(best_T)
    residual_norm = max(abs(final_f) / enthalpy_scale, final_result.residual_norm)
    residual_norm <= max(sqrt(rtol), 1.0e-8) || error("HP equilibrium did not converge; residual_norm=$residual_norm")
    return EquilibriumResult(:HP, true, bracket_iterations + temperature_iterations, residual_norm)
end

"""
    equilibrate!(state, mode; kwargs...)
    equilibrate!(state, workspace, mode=:TP; kwargs...)

Equilibrate an ideal-gas `GasState` in place and return an `EquilibriumResult`.

Supported modes:

- `:TP`: fixed temperature, fixed pressure equilibrium by Gibbs minimization.
- `:HP`: fixed enthalpy, fixed pressure equilibrium using nested `:TP` solves.

Keyword arguments:

- `rtol=1e-10`: nonlinear residual tolerance.
- `maxiters=100`: maximum nonlinear iterations.
- `temperature_atol=1e-8`: `:HP` temperature bracket tolerance.

Only ideal-gas single-phase equilibrium is currently supported. Multiphase
equilibrium and `:UV`/`:SV` modes are not implemented.
"""
function equilibrate!(state::GasState, mode::Symbol = :TP; kwargs...)
    if mode == :TP
        return equilibrate_TP!(state; kwargs...)
    elseif mode == :HP
        return equilibrate_HP!(state; kwargs...)
    end
    error("Unsupported equilibrium mode: $mode")
end

function equilibrate!(state::GasState, workspace::EquilibriumWorkspace, mode::Symbol = :TP; kwargs...)
    if mode == :TP
        return equilibrate_TP!(state, workspace; kwargs...)
    elseif mode == :HP
        return equilibrate_HP!(state, workspace; kwargs...)
    end
    error("Unsupported equilibrium mode: $mode")
end

"""
    equilibrate(state, mode=:TP; kwargs...)

Return an equilibrated copy of `state`, leaving the input unchanged.

Accepts the same modes and keyword arguments as `equilibrate!`.
"""
function equilibrate(state::GasState, mode::Symbol = :TP; kwargs...)
    copy_state = GasState(state.gas; T = state.T, p = state.p, X = state.X)
    equilibrate!(copy_state, mode; kwargs...)
    return copy_state
end

"""
    equilibrium_composition(gas; T, p, X, mode=:TP)

Return equilibrium mole fractions for a `Solution` at the requested state.

This is a convenience wrapper around `GasState` and `equilibrate!`.
"""
function equilibrium_composition(gas::Solution; T, p, X, mode = :TP, kwargs...)
    state = GasState(gas; T = T, p = p, X = X)
    equilibrate!(state, mode; kwargs...)
    return mole_fractions(state)
end

export GasState,
    EquilibriumResult,
    EquilibriumWorkspace,
    set_TPX!,
    set_TPY!,
    set_TP!,
    temperature,
    pressure,
    mole_fractions,
    mass_fractions,
    concentrations,
    density,
    mean_molecular_weight,
    species_cp_mole,
    species_cp_mass,
    cp_mole,
    cp_mass,
    species_enthalpies_mole,
    species_enthalpies_mass,
    enthalpy_mole,
    enthalpy_mass,
    species_internal_energies_mole,
    species_internal_energies_mass,
    internal_energy_mole,
    internal_energy_mass,
    species_entropies_mole,
    species_entropies_mass,
    entropy_mole,
    entropy_mass,
    species_gibbs_mole,
    species_gibbs_mass,
    gibbs_mole,
    gibbs_mass,
    equilibrate!,
    equilibrate,
    equilibrium_composition,
    net_production_rates!,
    net_production_rates
