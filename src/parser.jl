function parse_activation_energy_cal_per_mol(Ea_raw)::Float64
    # Already numeric: assume the mechanism-level units are cal/mol.
    if Ea_raw isa Number
        return float(Ea_raw)
    end

    # Otherwise, treat as string with possible units, e.g. "1.6599e+04 cal/mol"
    s = String(Ea_raw) |> strip
    isempty(s) && error("Empty Ea string")

    # Split on whitespace: ["1.6599e+04", "cal/mol"]
    parts = split(s)
    value = parse(Float64, parts[1])
    unit  = length(parts) >= 2 ? lowercase(parts[2]) : "j/mol"  # default if no unit

    # Normalize some common variants
    unit = replace(unit, " " => "")
    unit = replace(unit, "per" => "/")  # e.g. "J per mol" -> "J/mol"

    # Convert to cal/mol, which is what ReorderedKinetics currently expects.
    if unit in ("j/mol", "j")
        return value / 4.184
    elseif unit in ("kj/mol", "kj")
        return value / 4.184e-3
    elseif unit in ("cal/mol", "cal")
        return value 
    elseif unit in ("kcal/mol", "kcal")
        return value * 1e3
    elseif unit in ("k", "kelvin")
        # Sometimes Ea is given as Ea/R in K
        return value * R / 4.184
    else
        error("Unsupported Ea unit '$unit' in '$s'")
    end
end

# Remove third-body markers like "(+ M)" or "(+M)" before splitting
function clean_third_body_marks(s)
    # remove "(+ M)", "(+M)", with any amount of whitespace before M
    s = replace(s, r"\(\+\s*M\)" => "")
    return s
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
    # count falloff reactions
    n_falloff = count(r -> get(r, "type", nothing) == "falloff", reactions)
    n_troe = count(r -> get(r, "type", nothing) == "falloff" && haskey(r, "Troe"),
               reactions)


    idx_species = species_lookup(species_names)

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
        if haskey(mech_reaction, "rate-constant")
            k = mech_reaction["rate-constant"]
            Arrhenius_coeffs[j, 1] = k["A"]
            Arrhenius_coeffs[j, 2] = k["b"]
            Arrhenius_coeffs[j, 3] = parse_activation_energy_cal_per_mol(k["Ea"])
        elseif get(mech_reaction, "type", nothing) == "falloff"
            # choose a convention: use high-P as main Arrhenius for now
            high = mech_reaction["high-P-rate-constant"]
            Arrhenius_coeffs[j, 1] = high["A"]
            Arrhenius_coeffs[j, 2] = high["b"]
            Arrhenius_coeffs[j, 3] = parse_activation_energy_cal_per_mol(high["Ea"])
            # low-pressure limit goes into Arrhenius_0 (n_falloff × 3)
            low = mech_reaction["low-P-rate-constant"]
            falloff_idx += 1
            Arrhenius_0[falloff_idx, 1] = low["A"]
            Arrhenius_0[falloff_idx, 2] = low["b"]
            Arrhenius_0[falloff_idx, 3] = parse_activation_energy_cal_per_mol(low["Ea"])

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

    return (
        reactant_stoich_coeffs      = reactant_stoich_coeffs,
        product_stoich_coeffs       = product_stoich_coeffs,
        reactant_orders             = reactant_orders,
        efficiencies_coeffs_full    = efficiencies_coeffs,
        is_reversible               = is_reversible,
        Arrhenius_coeffs            = Arrhenius_coeffs,
        Arrhenius_0                 = Arrhenius_0,
        Troe_                       = Troe_
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
