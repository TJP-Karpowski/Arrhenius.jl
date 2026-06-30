function parse_Ea_SI(Ea_raw)::Float64
    # Already numeric: assume it's in J/mol
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

    # Convert to J/mol
    if unit in ("j/mol", "j")
        return value ./  4.184 
    elseif unit in ("kj/mol", "kj")
        return value  ./  4.184e3
    elseif unit in ("cal/mol", "cal")
        return value 
    elseif unit in ("kcal/mol", "kcal")
        return value .* 1e3
    elseif unit in ("k", "kelvin")
        # Sometimes Ea is given as Ea/R in K
        return value * R_gas ./  4.184 
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
        # e.g. "2 O", "0.5 H2", "CH4"
        m = match(r"^(\d+(\.\d+)?)\s+(\S+)$", term)
        coeff::Float64 = 0.0
        spname::String = ""
        if m === nothing
            coeff = 1.0
            spname = term
        else
            coeff  = parse(Float64, m.captures[1])
            spname = m.captures[end]
        end

        i_sp = idx_species[spname]
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


    idx_species = Dict{String,Int}(s => i for (i, s) in enumerate(species_names))

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

        # --- reversibility (same logic as your npz version) ---
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
        # you can override here if you later support explicit "orders" in yaml
        reactant_orders[:, j] .= reactant_stoich_coeffs[:, j]

        # --- Arrhenius coefficients ---
        if haskey(mech_reaction, "rate-constant")
            k = mech_reaction["rate-constant"]
            Arrhenius_coeffs[j, 1] = k["A"]
            Arrhenius_coeffs[j, 2] = k["b"]
            Arrhenius_coeffs[j, 3] = parse_Ea_SI(k["Ea"])
        elseif get(mech_reaction, "type", nothing) == "falloff"
            # choose a convention: use high-P as main Arrhenius for now
            high = mech_reaction["high-P-rate-constant"]
            Arrhenius_coeffs[j, 1] = high["A"]
            Arrhenius_coeffs[j, 2] = high["b"]
            Arrhenius_coeffs[j, 3] = parse_Ea_SI(high["Ea"])
            # low-pressure limit goes into Arrhenius_0 (n_falloff × 3)
            low = mech_reaction["low-P-rate-constant"]
            falloff_idx += 1
            Arrhenius_0[falloff_idx, 1] = low["A"]
            Arrhenius_0[falloff_idx, 2] = low["b"]
            Arrhenius_0[falloff_idx, 3] = parse_Ea_SI(low["Ea"])

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
                i = idx_species[sp]
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

    # index species by name
    species_by_name = Dict{String, Any}()
    for sp in yaml["species"]
        sname = String(sp["name"])
        species_by_name[sname] = sp
    end
    
    MW = zeros(Float64, length(species_names))

    for (i, sname) in enumerate(species_names)
        @assert haskey(species_by_name, sname) "Species '$sname' not found in YAML"

        sp = species_by_name[sname]
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
