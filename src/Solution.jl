function phase_index(yaml, phase)
    phase isa Integer && return phase
    for (i, item) in enumerate(yaml["phases"])
        if get(item, "name", nothing) == phase
            return i
        end
    end
    error("Phase '$phase' not found")
end

function phase_data(yaml; phase = 1)
    return yaml["phases"][phase_index(yaml, phase)]
end

function read_species_basics(yaml, phase = phase_data(yaml))
    n_species = length(phase["species"])
    n_reactions = length(yaml["reactions"])
    species_names = String.(strip.(phase["species"]))
    elements = phase["elements"]
    n_elements = length(elements)

    ele_matrix = zeros(n_elements, n_species)

    species_by_name = species_dict_by_name(yaml["species"])

    for (i, species) in enumerate(species_names)
        spec = species_entry_from_lookup(species_by_name, species)
        for j = 1:n_elements
            if haskey(spec["composition"], elements[j])
                ele_matrix[j, i] = spec["composition"][elements[j]]
            end
        end
    end

    return n_species, n_reactions, species_names, elements, n_elements, ele_matrix
end

function build_thermo(yaml, phase)
    if phase["thermo"] == "ideal-gas"
        return IdealGasThermo(yaml, phase)
    end
    constructorThermo = Symbol(phase["thermo"], :Thermo)
    return getfield(@__MODULE__, constructorThermo)(yaml)
end

function build_kinetics(yaml, species_names; reorderReactions = true)
    n_reactions = length(yaml["reactions"])
    kin = build_kinetic_data_from_yaml(yaml, species_names)
    reactant_stoich_coeffs = kin.reactant_stoich_coeffs
    product_stoich_coeffs = kin.product_stoich_coeffs
    reactant_orders = kin.reactant_orders
    efficiencies_coeffs_full = kin.efficiencies_coeffs_full
    Arrhenius_coeffs = kin.Arrhenius_coeffs
    Arrhenius_0 = kin.Arrhenius_0
    is_reversible = kin.is_reversible
    Troe_ = kin.Troe_

    index_three_body = Int64[]
    index_falloff = Int64[]
    index_falloff_Troe = Int64[]
    j = 1
    for i = 1:n_reactions
        reaction = yaml["reactions"][i]
        if haskey(reaction, "type")
            if reaction["type"] == "three-body"
                push!(index_three_body, i)
            end
            if reaction["type"] == "falloff"
                push!(index_falloff, i)
                if haskey(reaction, "Troe")
                    push!(index_falloff_Troe, j)
                    j = j + 1
                else
                    push!(index_falloff_Troe, -1)
                end
            end
        end
    end

    i_reactant = Vector{Int64}[]
    i_product = Vector{Int64}[]
    for i = 1:n_reactions
        push!(i_reactant, findall(reactant_orders[:, i] .> 0.01))
        push!(i_product, findall(product_stoich_coeffs[:, i] .> 0.01))
    end

    vk = sparse(product_stoich_coeffs - reactant_stoich_coeffs)
    vk_sum = sum(vk, dims = 1)[1, :]

    for i in 1:n_reactions
        if !((i in index_three_body) | (i in index_falloff))
            efficiencies_coeffs_full[:, i] .= 0.0
        end
    end
    efficiencies_coeffs = sparse(efficiencies_coeffs_full)

    kinetics = StandardKinetics(
        product_stoich_coeffs,
        reactant_stoich_coeffs,
        reactant_orders,
        is_reversible,
        Arrhenius_coeffs,
        Arrhenius_0,
        Troe_,
        index_three_body,
        index_falloff,
        index_falloff_Troe,
        efficiencies_coeffs,
        i_reactant,
        i_product,
        n_reactions,
        vk,
        vk_sum,
    )

    return reorderReactions ? ReorderedKinetics(kinetics) : kinetics
end

function build_transport(yaml, species_names, MW, thermo)
    transport_data = build_transport_data_from_yaml(yaml, species_names, MW, thermo)
    return Transport(
        transport_data.poly_order,
        transport_data.species_viscosities_poly,
        transport_data.thermal_conductivity_poly,
        transport_data.binary_diff_coeffs_poly,
    )
end

"""
    CreateSolution(mech)
    
Create a gas solution from a Cantera-style YAML mechanism.
"""
function CreateSolution(mech; reorderReactions = true, phase = 1, phase_index = nothing, phase_name = nothing)
    yaml = YAML.load_file(mech)
    selected_phase = phase_name === nothing ? (phase_index === nothing ? phase : phase_index) : phase_name
    phase_yaml = phase_data(yaml; phase = selected_phase)

    #### Basics
    n_species, n_reactions, species_names,
    elements, n_elements, ele_matrix = read_species_basics(yaml, phase_yaml)

    #### Thermo
    thermo = build_thermo(yaml, phase_yaml)
    MW = build_MW_from_yaml(yaml, species_names)
    
    #### Kinetic data
    kinetics = build_kinetics(yaml, species_names; reorderReactions = reorderReactions)
    #### Transport data

    trans = build_transport(yaml, species_names, MW, thermo)

    gas = Solution(
        n_species,
        n_reactions,
        MW,
        species_names,
        elements,
        ele_matrix,
        thermo,
        trans,
        kinetics,
    )
    return gas
end

export CreateSolution

function ReorderedKinetics(kinetics::StandardKinetics)
    i_forward = .!(kinetics.is_reversible)
    i_rev = Int64[]
    for (i) in 1:kinetics.n_reactions
        if !(i in  kinetics.index_three_body) & !(i in  kinetics.index_falloff) & kinetics.is_reversible[i]
            push!(i_rev,i)
        end

    end
    i_tB = kinetics.index_three_body
    i_fO = Int64[]
    i_troe = Int64[]
    i_A0_fO = Int64[]
    i_A0_troe = Int64[]
    for (i, index) in enumerate(kinetics.index_falloff )
        if kinetics.index_falloff_Troe[i] < 0
            push!(i_fO,index)
            push!(i_A0_fO,i)
        else
            push!(i_troe,index)
            push!(i_A0_troe,i)
        end
    end

    arr_coeffs = vcat(
        kinetics.Arrhenius_coeffs[i_forward,:],
        kinetics.Arrhenius_coeffs[i_rev,:],
        kinetics.Arrhenius_coeffs[i_tB,:],
        kinetics.Arrhenius_coeffs[i_fO,:],
        kinetics.Arrhenius_coeffs[i_troe,:],
        )

    reactant_orders = hcat(
        kinetics.reactant_orders[:,i_forward],
        kinetics.reactant_orders[:,i_rev],
        kinetics.reactant_orders[:,i_tB],
        kinetics.reactant_orders[:,i_fO],
        kinetics.reactant_orders[:,i_troe])

    product_stoich_coeffs = hcat(
        kinetics.product_stoich_coeffs[:,i_rev],
        kinetics.product_stoich_coeffs[:,i_tB],
        kinetics.product_stoich_coeffs[:,i_fO],
        kinetics.product_stoich_coeffs[:,i_troe])

    reactant_stoich_coeffs = hcat(
        kinetics.reactant_stoich_coeffs[:,i_rev],
        kinetics.reactant_stoich_coeffs[:,i_tB],
        kinetics.reactant_stoich_coeffs[:,i_fO],
        kinetics.reactant_stoich_coeffs[:,i_troe])

    vk = sparse(vcat(
        kinetics.vk[:,i_forward]',
        kinetics.vk[:,i_rev]',
        kinetics.vk[:,i_tB]',
        kinetics.vk[:,i_fO]',
        kinetics.vk[:,i_troe]'
        ))

    vk_rev = sparse(vcat(
        kinetics.vk[:,i_rev]',
        kinetics.vk[:,i_tB]',
        kinetics.vk[:,i_fO]',
        kinetics.vk[:,i_troe]'))

    vk_sum = vcat(
        kinetics.vk_sum[i_rev],
        kinetics.vk_sum[i_tB],
        kinetics.vk_sum[i_fO],
        kinetics.vk_sum[i_troe]
    )

    efficiency_coeffs_tB=kinetics.efficiencies_coeffs[:,i_tB]

    efficiency_coeffs_fO = hcat(
        kinetics.efficiencies_coeffs[:,i_fO],
        kinetics.efficiencies_coeffs[:,i_troe])

    Arrhenius_0 =vcat(
        kinetics.Arrhenius_0[i_A0_fO,:],
        kinetics.Arrhenius_0[i_A0_troe,:])

    troe =kinetics.Troe_

    index_rev_start = sum(i_forward)+1
    index_three_body_start = index_rev_start + length(i_rev) 
    index_fallOff_start = index_three_body_start + length(i_tB) 
    index_troe_start = index_fallOff_start + length(i_fO) 
    index_troe_start+length(i_troe)

    return ReorderedKinetics(
        arr_coeffs,
        reactant_orders,
        product_stoich_coeffs,
        reactant_stoich_coeffs,
        vk,
        vk_rev,
        vk_sum,
        efficiency_coeffs_tB,
        efficiency_coeffs_fO,
        Arrhenius_0,
        troe,
        index_rev_start,
        index_three_body_start,
        index_fallOff_start,
        index_troe_start,
    )

end
