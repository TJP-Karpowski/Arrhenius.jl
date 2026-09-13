function parse_activation_energy_cal_per_mol(Ea_raw, default_unit = "cal/mol")::Float64
    # Bare numeric values use the mechanism-level activation-energy unit.
    if Ea_raw isa Number
        value = float(Ea_raw)
        unit = lowercase(String(default_unit))
        return activation_energy_to_cal_per_mol(value, unit)
    end

    # Otherwise, treat as string with possible units, e.g. "1.6599e+04 cal/mol"
    s = String(Ea_raw) |> strip
    isempty(s) && error("Empty Ea string")

    # Split on whitespace: ["1.6599e+04", "cal/mol"]
    parts = split(s)
    value = parse(Float64, parts[1])
    unit  = length(parts) >= 2 ? lowercase(parts[2]) : lowercase(String(default_unit))

    # Normalize some common variants
    unit = replace(unit, " " => "")
    unit = replace(unit, "per" => "/")  # e.g. "J per mol" -> "J/mol"

    return activation_energy_to_cal_per_mol(value, unit)
end

function activation_energy_to_cal_per_mol(value, unit)
    # Convert to cal/mol, which is what ReorderedKinetics currently expects.
    unit = replace(lowercase(String(unit)), " " => "")
    unit = replace(unit, "per" => "/")
    if unit in ("j/mol", "j")
        return value / 4.184
    elseif unit == "j/kmol"
        return value / 4184.0
    elseif unit in ("kj/mol", "kj")
        return value / 4.184e-3
    elseif unit == "kj/kmol"
        return value / 4.184
    elseif unit in ("cal/mol", "cal")
        return value
    elseif unit == "cal/kmol"
        return value / 1000.0
    elseif unit in ("kcal/mol", "kcal")
        return value * 1e3
    elseif unit == "kcal/kmol"
        return value
    elseif unit in ("k", "kelvin")
        # Sometimes Ea is given as Ea/R in K
        return value * R / 4.184
    else
        error("Unsupported activation-energy unit '$unit'")
    end
end

function parse_pressure_atm(P_raw)::Float64
    if P_raw isa Number
        return float(P_raw)
    end

    s = String(P_raw) |> strip
    parts = split(s)
    value = parse(Float64, parts[1])
    unit = length(parts) >= 2 ? lowercase(parts[2]) : "atm"
    unit = replace(unit, " " => "")

    if unit in ("atm", "atmosphere", "atmospheres")
        return value
    elseif unit in ("pa", "pascal", "pascals")
        return value / one_atm
    elseif unit in ("bar",)
        return value * 1.0e5 / one_atm
    elseif unit in ("torr",)
        return value / 760.0
    else
        error("Unsupported pressure unit '$unit' in '$s'")
    end
end

function pressure_dependent_rate_constants(reaction)
    if get(reaction, "type", nothing) == "pressure-dependent-Arrhenius"
        return reaction["rate-constants"]
    elseif get(reaction, "type", nothing) == "linear-Burke"
        for collider in get(reaction, "colliders", Any[])
            if get(collider, "name", nothing) == "M" &&
               get(collider, "type", nothing) == "pressure-dependent-Arrhenius"
                return collider["rate-constants"]
            end
        end
    end
    return nothing
end

function linear_burke_efficiency_coefficients!(coeffs, reaction, species_names,
        activation_energy_unit = "cal/mol")
    get(reaction, "type", nothing) == "linear-Burke" || return nothing
    coeffs[:, 1] .= 1.0
    coeffs[:, 2] .= 0.0
    coeffs[:, 3] .= 0.0
    idx_species = species_lookup(species_names)
    for collider in get(reaction, "colliders", Any[])
        haskey(collider, "efficiency") || continue
        collider_name = strip(String(collider["name"]))
        idx = if haskey(idx_species, collider_name)
            idx_species[collider_name]
        elseif haskey(idx_species, lowercase(collider_name))
            idx_species[lowercase(collider_name)]
        else
            continue
        end
        eff = collider["efficiency"]
        coeffs[idx, 1] = float(eff["A"])
        coeffs[idx, 2] = float(eff["b"])
        coeffs[idx, 3] = parse_activation_energy_cal_per_mol(
            eff["Ea"], activation_energy_unit)
    end
    return nothing
end

function unit_value(units, key, default)
    units === nothing && return default
    haskey(units, key) && return lowercase(String(units[key]))
    haskey(units, String(key)) && return lowercase(String(units[String(key)]))
    return default
end

function length_unit_meters(unit)
    unit = replace(lowercase(unit), " " => "")
    if unit in ("m", "meter", "meters")
        return 1.0
    elseif unit in ("cm", "centimeter", "centimeters")
        return 1.0e-2
    elseif unit in ("mm", "millimeter", "millimeters")
        return 1.0e-3
    else
        error("Unsupported mechanism length unit '$unit'")
    end
end

function quantity_unit_kmol(unit)
    unit = replace(lowercase(unit), " " => "")
    if unit in ("kmol", "kilomol", "kilomole", "kilomoles")
        return 1.0
    elseif unit in ("mol", "mole", "moles")
        return 1.0e-3
    else
        error("Unsupported mechanism quantity unit '$unit'")
    end
end

function time_unit_seconds(unit)
    unit = replace(lowercase(unit), " " => "")
    if unit in ("s", "sec", "second", "seconds")
        return 1.0
    elseif unit in ("ms", "millisecond", "milliseconds")
        return 1.0e-3
    elseif unit in ("us", "microsecond", "microseconds")
        return 1.0e-6
    else
        error("Unsupported mechanism time unit '$unit'")
    end
end

function mechanism_concentration_unit_kmol_per_m3(yaml)
    units = get(yaml, "units", nothing)
    units === nothing && return 1.0
    length_m = length_unit_meters(unit_value(units, "length", "m"))
    quantity_kmol = quantity_unit_kmol(unit_value(units, "quantity", "kmol"))
    return quantity_kmol / length_m^3
end

function mechanism_time_rate_scale(yaml)
    units = get(yaml, "units", nothing)
    units === nothing && return 1.0
    return inv(time_unit_seconds(unit_value(units, "time", "s")))
end

function convert_preexponential_to_si(A, total_concentration_order, concentration_unit_scale, time_rate_scale)
    return float(A) * time_rate_scale * concentration_unit_scale^(1 - total_concentration_order)
end

function reaction_concentration_order(reactant_orders, reaction::Int)
    order = 0.0
    for idx in reactant_orders.colptr[reaction]:reactant_orders.colptr[reaction + 1] - 1
        order += reactant_orders.nzval[idx]
    end
    return order
end

# Remove parenthesized collider markers such as "(+M)", "(+ M)",
# "(+N2)", or "(+H2O)" before splitting the stoichiometric equation.  The
# collider identity belongs to the falloff-rate metadata, not to net reaction
# stoichiometry.
function clean_third_body_marks(s)
    s = replace(s, r"\(\+\s*[^)]+\)" => "")
    return s
end

"""
    parenthesized_collider(equation)

Return the collider named in a parenthesized falloff marker such as `(+M)` or
`(+N2)`. A named collider denotes a collider-specific falloff channel and is
not equivalent to the generic bath-gas collider `M`.
"""
function parenthesized_collider(equation)
    matched = match(r"\(\+\s*([^)\s]+)\s*\)", equation)
    matched === nothing ? nothing : strip(String(matched.captures[1]))
end

function species_lookup(names)
    lookup = Dict{String,Int}()
    for (i, name) in enumerate(names)
        stripped = strip(String(name))
        lookup[stripped] = i
        lookup[lowercase(stripped)] = i
    end
    return lookup
end

function species_dict_by_name(species)
    by_name = Dict{String,Any}()
    for sp in species
        sname = strip(String(sp["name"]))
        by_name[sname] = sp
        by_name[lowercase(sname)] = sp
    end
    return by_name
end

function species_index_from_lookup(lookup, name)
    stripped = strip(String(name))
    haskey(lookup, stripped) && return lookup[stripped]
    return lookup[lowercase(stripped)]
end

function species_entry_from_lookup(lookup, name)
    stripped = strip(String(name))
    haskey(lookup, stripped) && return lookup[stripped]
    return lookup[lowercase(stripped)]
end

# helper: parse one side of the equation (LHS or RHS)
function parse_side!(
    ν::SparseMatrixCSC{Float64, Int},  # reactant or product stoich matrix
    j_rxn::Int,
    terms::Vector{SubString{String}},
    idx_species::Dict{String,Int},
)
    for term in terms
        term = strip(term)
        # ignore explicit third-body markers
        if term == "M" || term == "(+M)"  || term == "(+ M)"
            continue
        end
        # strip "(+M)" suffix: "CO (+M)" -> "CO"
        term = replace(term, "(+M)" => "")
        term = strip(term)
        isempty(term) && continue

        # pattern: optional coefficient + species
        # e.g. "2 O", "0.5 H2", "5.00E-01 O2", "CH4"
        m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+(.+)$", term)
        coeff::Float64 = 0.0
        spname::String = ""
        if m === nothing
            coeff = 1.0
            spname = term
        else
            coeff  = parse(Float64, m.captures[1])
            spname = m.captures[end]
        end

        i_sp = species_index_from_lookup(idx_species, spname)
        ν[i_sp, j_rxn] += coeff
    end
end

function build_kinetic_data_from_yaml(yaml, species_names::Vector{String})
    reactions = yaml["reactions"]
    n_reac    = length(reactions)
    n_spec    = length(species_names)
    concentration_unit_scale = mechanism_concentration_unit_kmol_per_m3(yaml)
    time_rate_scale = mechanism_time_rate_scale(yaml)
    activation_energy_unit = unit_value(get(yaml, "units", nothing),
        "activation-energy", "cal/mol")
    # count falloff reactions
    n_falloff = count(r -> get(r, "type", nothing) == "falloff", reactions)
    n_troe = count(r -> get(r, "type", nothing) == "falloff" && haskey(r, "Troe"),
               reactions)
    idx_species = species_lookup(species_names)
    index_pressure_dependent = Int64[]
    pressure_dependent_pressures = Float64[]
    pressure_dependent_arrhenius_rows = Vector{NTuple{3,Float64}}()
    pressure_dependent_ptr = Int64[1]
    pressure_dependent_is_linear_burke = Bool[]
    linear_burke_efficiencies = zeros(Float64, n_spec, 0, 3)
    for (j, reaction) in enumerate(reactions)
        rates = pressure_dependent_rate_constants(reaction)
        rates === nothing && continue
        push!(index_pressure_dependent, j)
        is_linear_burke = get(reaction, "type", nothing) == "linear-Burke"
        push!(pressure_dependent_is_linear_burke, is_linear_burke)
        coeffs = zeros(Float64, n_spec, 1, 3)
        if is_linear_burke
            coeff_view = @view coeffs[:, 1, :]
            linear_burke_efficiency_coefficients!(coeff_view, reaction,
                species_names, activation_energy_unit)
        end
        linear_burke_efficiencies = cat(linear_burke_efficiencies, coeffs; dims = 2)
        for rate in rates
            push!(pressure_dependent_pressures, parse_pressure_atm(rate["P"]))
            push!(pressure_dependent_arrhenius_rows, (
                float(rate["A"]),
                float(rate["b"]),
                parse_activation_energy_cal_per_mol(rate["Ea"], activation_energy_unit),
            ))
        end
        push!(pressure_dependent_ptr, length(pressure_dependent_pressures) + 1)
    end

    reactant_stoich_coeffs = spzeros(Float64, n_spec, n_reac)
    product_stoich_coeffs  = spzeros(Float64, n_spec, n_reac)
    reactant_orders        = spzeros(Float64, n_spec, n_reac)
    efficiencies_coeffs    = ones(Float64, n_spec, n_reac)
    is_reversible          = falses(n_reac)
    Arrhenius_coeffs       = zeros(Float64,  n_reac,3)  # [A; b; Ea] per reaction
    Arrhenius_0      = zeros(Float64, n_falloff, 3)   # low-P  [A0, b0, Ea0]
    Troe_ = zeros(Float64, n_troe, 4)               # columns: A, T1, T2, T3

    
    falloff_idx = 0  # counter within the loop
    troe_idx    = 0 # conter within the loop
    for (j, mech_reaction) in enumerate(reactions)
        eq = mech_reaction["equation"]::String

        # For a named marker such as `(+N2)`, the effective collider
        # concentration is C_N2. Generic `(+M)` channels retain the all-ones
        # default and any explicit YAML efficiencies are applied below.
        collider = parenthesized_collider(eq)
        if get(mech_reaction, "type", nothing) == "falloff" &&
                collider !== nothing && uppercase(collider) != "M"
            efficiencies_coeffs[:, j] .= 0.0
            efficiencies_coeffs[
                species_index_from_lookup(idx_species, collider), j] = 1.0
        end

        # --- reversibility ---
        if !haskey(mech_reaction, "type")
            if occursin("<=>", eq)
                is_reversible[j] = true
            end
        else
            is_reversible[j] = true
        end

        # --- split equation into LHS/RHS ---
        if occursin("<=>", eq)
            lhs_str, rhs_str = split(eq, "<=>"; limit = 2)
        elseif occursin("=>", eq)
            lhs_str, rhs_str = split(eq, "=>"; limit = 2)
        else
            error("Unknown reaction arrow in equation: $eq")
        end


        # strip and remove (+ M) / (+M) before splitting by '+'
        lhs_str = clean_third_body_marks(strip(lhs_str))
        rhs_str = clean_third_body_marks(strip(rhs_str))
        lhs_terms = split(strip(lhs_str), '+')
        rhs_terms = split(strip(rhs_str), '+')

        # fill stoichiometric matrices
        parse_side!(reactant_stoich_coeffs, j, lhs_terms, idx_species)
        parse_side!(product_stoich_coeffs,  j, rhs_terms, idx_species)

        # default reaction orders = reactant stoichiometry (elementary reaction)
        reactant_orders[:, j] .= reactant_stoich_coeffs[:, j]
        if haskey(mech_reaction, "orders")
            reactant_orders[:, j] .= 0.0
            for (sp, order) in mech_reaction["orders"]
                reactant_orders[species_index_from_lookup(idx_species, sp), j] = float(order)
            end
        end

        # --- Arrhenius coefficients ---
        pdep_rates = pressure_dependent_rate_constants(mech_reaction)
        reaction_order = reaction_concentration_order(reactant_orders, j)
        effective_reaction_order = get(mech_reaction, "type", nothing) == "three-body" ? reaction_order + 1 : reaction_order
        if haskey(mech_reaction, "rate-constant")
            k = mech_reaction["rate-constant"]
            Arrhenius_coeffs[j, 1] =
                convert_preexponential_to_si(k["A"], effective_reaction_order, concentration_unit_scale, time_rate_scale)
            Arrhenius_coeffs[j, 2] = k["b"]
            Arrhenius_coeffs[j, 3] = parse_activation_energy_cal_per_mol(
                k["Ea"], activation_energy_unit)
        elseif pdep_rates !== nothing
            k = pdep_rates[1]
            Arrhenius_coeffs[j, 1] =
                convert_preexponential_to_si(k["A"], reaction_order, concentration_unit_scale, time_rate_scale)
            Arrhenius_coeffs[j, 2] = k["b"]
            Arrhenius_coeffs[j, 3] = parse_activation_energy_cal_per_mol(
                k["Ea"], activation_energy_unit)
        elseif get(mech_reaction, "type", nothing) == "falloff"
            # choose a convention: use high-P as main Arrhenius for now
            high = mech_reaction["high-P-rate-constant"]
            Arrhenius_coeffs[j, 1] =
                convert_preexponential_to_si(high["A"], reaction_order, concentration_unit_scale, time_rate_scale)
            Arrhenius_coeffs[j, 2] = high["b"]
            Arrhenius_coeffs[j, 3] = parse_activation_energy_cal_per_mol(
                high["Ea"], activation_energy_unit)
            # low-pressure limit goes into Arrhenius_0 (n_falloff × 3)
            low = mech_reaction["low-P-rate-constant"]
            falloff_idx += 1
            Arrhenius_0[falloff_idx, 1] =
                convert_preexponential_to_si(low["A"], reaction_order + 1, concentration_unit_scale, time_rate_scale)
            Arrhenius_0[falloff_idx, 2] = low["b"]
            Arrhenius_0[falloff_idx, 3] = parse_activation_energy_cal_per_mol(
                low["Ea"], activation_energy_unit)

            # --- Troe data (optional for some falloff reactions) ---
            if haskey(mech_reaction, "Troe")
                troe = mech_reaction["Troe"]
                troe_idx += 1

                A_Troe = troe["A"]              # dimensionless
                T1     = haskey(troe, "T1") ? troe["T1"] : 0.0
                T2     = haskey(troe, "T2") ? troe["T2"] : T1   # your convention
                T3     = haskey(troe, "T3") ? troe["T3"] : 0.0

                Troe_[troe_idx, 1] = A_Troe
                Troe_[troe_idx, 2] = T1
                Troe_[troe_idx, 3] = T2
                Troe_[troe_idx, 4] = T3
            end
        else
            error("No usable rate-constant for reaction $j : $eq")
        end

        # --- third-body / falloff efficiencies ---
        if haskey(mech_reaction, "efficiencies")
            eff_map = mech_reaction["efficiencies"]  # Dict{String,Any}
            for (sp, α) in eff_map
                i = species_index_from_lookup(idx_species, sp)
                efficiencies_coeffs[i, j] = α
            end
        end
    end

    pressure_dependent_arrhenius = if isempty(pressure_dependent_arrhenius_rows)
        zeros(Float64, 0, 3)
    else
        reduce(vcat, ([row[1] row[2] row[3]] for row in pressure_dependent_arrhenius_rows))
    end
    for (pdep_idx, reaction_index) in enumerate(index_pressure_dependent)
        order = reaction_concentration_order(reactant_orders, reaction_index)
        for row in pressure_dependent_ptr[pdep_idx]:(pressure_dependent_ptr[pdep_idx + 1] - 1)
            pressure_dependent_arrhenius[row, 1] = convert_preexponential_to_si(
                pressure_dependent_arrhenius[row, 1],
                order,
                concentration_unit_scale,
                time_rate_scale,
            )
        end
    end

    return (
        reactant_stoich_coeffs      = reactant_stoich_coeffs,
        product_stoich_coeffs       = product_stoich_coeffs,
        reactant_orders             = reactant_orders,
        efficiencies_coeffs_full    = efficiencies_coeffs,
        is_reversible               = is_reversible,
        Arrhenius_coeffs            = Arrhenius_coeffs,
        Arrhenius_0                 = Arrhenius_0,
        Troe_                       = Troe_,
        index_pressure_dependent    = index_pressure_dependent,
        pressure_dependent_pressures = pressure_dependent_pressures,
        pressure_dependent_arrhenius = pressure_dependent_arrhenius,
        pressure_dependent_ptr      = pressure_dependent_ptr,
        pressure_dependent_is_linear_burke = pressure_dependent_is_linear_burke,
        linear_burke_efficiencies = linear_burke_efficiencies,
    )
end


"""
    build_MW_from_yaml(yaml, species_names) -> Vector{Float64}

Compute molecular weights for `species_names` (in that order) from a Cantera-style
YAML mechanism, using hard-coded element molar masses in `ELEMENT_MW`.

Assumes:
- `yaml["species"]` is a list of species dicts with `"name"` and `"composition"`.
- `composition` maps element symbols (e.g. "H", "O", "C", "AR") to stoichiometric counts.
"""
function build_MW_from_yaml(yaml, species_names::Vector{String})
    @assert haskey(yaml, "species") "YAML has no 'species' block"

    species_by_name = species_dict_by_name(yaml["species"])
    
    MW = zeros(Float64, length(species_names))

    for (i, sname) in enumerate(species_names)
        sp = species_entry_from_lookup(species_by_name, sname)
        comp = sp["composition"]  # Dict{String,Any}: element => count

        mw = 0.0
        for (el, n_el) in comp
            el_sym = String(el)
            @assert haskey(ELEMENT_MW, el_sym) "No hard-coded molar mass for element '$el_sym'"
            mw += ELEMENT_MW[el_sym] * float(n_el)
        end

        MW[i] = mw
    end

    return MW
end

function transport_geometry_dof(geometry)
    geometry = lowercase(String(geometry))
    geometry == "atom" && return 0.0
    geometry == "linear" && return 2.0
    geometry == "nonlinear" && return 3.0
    error("Unsupported transport geometry '$geometry'")
end

function species_transport_data(yaml, species_names::Vector{String})
    species_by_name = species_dict_by_name(yaml["species"])

    data = Vector{NamedTuple}(undef, length(species_names))
    for (i, sname) in enumerate(species_names)
        sp = species_entry_from_lookup(species_by_name, sname)
        @assert haskey(sp, "transport") "Species '$sname' has no transport block"
        trans = sp["transport"]
        @assert get(trans, "model", "gas") == "gas" "Only gas transport is supported for '$sname'"

        data[i] = (
            geometry = String(trans["geometry"]),
            dof = transport_geometry_dof(trans["geometry"]),
            well_depth = float(trans["well-depth"]),
            diameter = float(trans["diameter"]),
            rotational_relaxation = float(get(trans, "rotational-relaxation", 0.0)),
        )
    end
    return data
end

# Neufeld, Janzen, and Aziz (J. Chem. Phys., 1972) Lennard-Jones
# collision-integral fits for Ω_D and Ω_η.
function collision_integral_diffusion(Tstar)
    return 1.06036 / Tstar^0.15610 +
           0.19300 / exp(0.47635 * Tstar) +
           1.03587 / exp(1.52996 * Tstar) +
           1.76474 / exp(3.89411 * Tstar)
end

function collision_integral_viscosity(Tstar)
    return 1.16145 / Tstar^0.14874 +
           0.52487 / exp(0.77320 * Tstar) +
           2.16178 / exp(2.43787 * Tstar)
end

function species_viscosity(T, MW, trans)
    Tstar = T / trans.well_depth
    omega = collision_integral_viscosity(Tstar)
    return 2.6693e-6 * sqrt(MW * T) / (trans.diameter^2 * omega)
end

function binary_diffusion_1atm(T, MW_i, MW_j, trans_i, trans_j)
    sigma = (trans_i.diameter + trans_j.diameter) / 2
    epsilon = sqrt(trans_i.well_depth * trans_j.well_depth)
    Tstar = T / epsilon
    omega = collision_integral_diffusion(Tstar)
    D_cm2_per_s = 0.0018583 * sqrt(T^3 * (1 / MW_i + 1 / MW_j)) / (sigma^2 * omega)
    return 1.0e-4 * D_cm2_per_s
end

function species_cp_R(thermo, T)
    coeffs = T <= 1000.0 ? thermo.nasa_low : thermo.nasa_high
    cp_T = (1.0, T, T^2, T^3, T^4)
    cp_R = zeros(size(coeffs, 1))
    for i in eachindex(cp_R)
        for j in 1:5
            cp_R[i] += coeffs[i, j] * cp_T[j]
        end
    end
    return cp_R
end

function species_conductivity(T, MW, thermo, trans, viscosity)
    cp_molar = species_cp_R(thermo, T) * R
    lambda = similar(MW)
    for i in eachindex(MW)
        cv_molar = cp_molar[i] - R
        internal_cv = max(cv_molar - 1.5 * R, 0.0)
        translational = 2.5 * R
        rotational = trans[i].dof > 0 ? R : 0.0
        # Modified Eucken-style split. This keeps the parser self-contained
        # while using the YAML Lennard-Jones parameters directly.
        eucken = translational + 1.5 * rotational + internal_cv
        lambda[i] = viscosity[i] * eucken / MW[i]
    end
    return lambda
end

function fit_log_temperature_polynomial(values, temperatures; order = 6)
    logT = log.(temperatures)
    A = zeros(length(temperatures), order + 1)
    for i in eachindex(temperatures)
        for p in 0:order
            A[i, p + 1] = logT[i]^(order - p)
        end
    end
    return A \ values
end

function build_transport_data_from_yaml(yaml, species_names, MW, thermo; T_min = 300.0, T_max = 3000.0, n_points = 120)
    n_species = length(species_names)
    if !all(haskey(sp, "transport") for sp in yaml["species"] if String(sp["name"]) in species_names)
        poly_order = 6
        return (
            poly_order = poly_order,
            species_viscosities_poly = zeros(poly_order + 1, n_species),
            thermal_conductivity_poly = zeros(poly_order + 1, n_species),
            binary_diff_coeffs_poly = zeros(poly_order + 1, n_species * n_species),
        )
    end

    trans = species_transport_data(yaml, species_names)
    temperatures = collect(range(T_min, T_max; length = n_points))

    species_viscosities = zeros(length(temperatures), n_species)
    thermal_conductivity = zeros(length(temperatures), n_species)
    binary_diff_coeffs = zeros(length(temperatures), n_species * n_species)

    for (it, T) in enumerate(temperatures)
        for i in 1:n_species
            species_viscosities[it, i] = species_viscosity(T, MW[i], trans[i])
        end

        lambda = species_conductivity(T, MW, thermo, trans, @view(species_viscosities[it, :]))
        thermal_conductivity[it, :] .= lambda

        for j in 1:n_species
            for i in 1:n_species
                idx = i + (j - 1) * n_species
                if i == j
                    binary_diff_coeffs[it, idx] = Inf
                else
                    binary_diff_coeffs[it, idx] =
                        binary_diffusion_1atm(T, MW[i], MW[j], trans[i], trans[j])
                end
            end
        end
    end

    poly_order = 6
    species_viscosities_poly = zeros(poly_order + 1, n_species)
    thermal_conductivity_poly = zeros(poly_order + 1, n_species)
    binary_diff_coeffs_poly = zeros(poly_order + 1, n_species * n_species)

    for i in 1:n_species
        species_viscosities_poly[:, i] .=
            fit_log_temperature_polynomial(species_viscosities[:, i], temperatures; order = poly_order)
        thermal_conductivity_poly[:, i] .=
            fit_log_temperature_polynomial(thermal_conductivity[:, i], temperatures; order = poly_order)
    end

    for j in 1:n_species
        for i in 1:n_species
            idx = i + (j - 1) * n_species
            if i == j
                binary_diff_coeffs_poly[:, idx] .= 0.0
            else
                binary_diff_coeffs_poly[:, idx] .=
                    fit_log_temperature_polynomial(binary_diff_coeffs[:, idx], temperatures; order = poly_order)
            end
        end
    end

    return (
        poly_order = poly_order,
        species_viscosities_poly = species_viscosities_poly,
        thermal_conductivity_poly = thermal_conductivity_poly,
        binary_diff_coeffs_poly = binary_diff_coeffs_poly,
    )
end
