function read_species_basics(yaml)
    n_species = length(yaml["phases"][1]["species"])
    n_reactions = length(yaml["reactions"])
    species_names = yaml["phases"][1]["species"]
    elements = yaml["phases"][1]["elements"]
    n_elements = length(elements)

    ele_matrix = zeros(n_elements, n_species)

    _species_names =
        [yaml["species"][i]["name"] for i = 1:length(yaml["species"])]

    for (i, species) in enumerate(species_names)
        spec = yaml["species"][findfirst(x -> x == species, _species_names)]
        for j = 1:n_elements
            if haskey(spec["composition"], elements[j])
                ele_matrix[j, i] = spec["composition"][elements[j]]
            end
        end
    end

    return n_species, n_reactions, species_names, elements, n_elements, ele_matrix
end
"""
    CreateSolution(mech)
    
Reaction mechanism is interepreted here. Part of the infomation are read in
from the yaml file, pary of them are from the pre-processed .npz file from
ReacTorch and Cantera
test for math enviroment
"""
function CreateSolution(mech; reorderReactions = true)
    yaml = YAML.load_file(mech)

    #### Basics
    n_species, n_reactions, species_names,
    elements, n_elements, ele_matrix = read_species_basics(yaml)

    #### Thermo
    if yaml["phases"][1]["thermo"] == "ideal-gas" # switch to work with Cantera standard
        thermo = IdealGasThermo(yaml)
    else
        constructorThermo = Symbol(yaml["phases"][1]["thermo"], :Thermo)
        thermo = @eval($constructorThermo)(yaml)
    end
    MW = build_MW_from_yaml(yaml, species_names)
    
    #### Kinetic data
    #npz = npzread("$mech.npz")
    #npz["molecular_weights"]
    #println(MW-npz["molecular_weights"])
    # efficiencies_coeffs_full = npz["efficiencies_coeffs"]
    # product_stoich_coeffs = sparse(npz["product_stoich_coeffs"])
    # reactant_stoich_coeffs = sparse(npz["reactant_stoich_coeffs"])
    # reactant_orders = sparse(npz["reactant_orders"])
    # is_reversible = npz["is_reversible"]
    
    kin = build_kinetic_data_from_yaml(yaml, species_names)
    reactant_stoich_coeffs   = kin.reactant_stoich_coeffs
    product_stoich_coeffs  =  kin.product_stoich_coeffs
    reactant_orders         =  kin.reactant_orders
    efficiencies_coeffs_full = kin.efficiencies_coeffs_full
    Arrhenius_coeffs = kin.Arrhenius_coeffs
    Arrhenius_0      = kin.Arrhenius_0
    is_reversible    = kin.is_reversible
    Troe_ =  kin.Troe_

    index_three_body = []
    index_falloff = []
    index_falloff_Troe = []
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

    i_reactant = []
    i_product = []
    for i = 1:n_reactions
        push!(i_reactant, findall(reactant_orders[:, i] .> 0.01))
        push!(i_product, findall(product_stoich_coeffs[:, i] .> 0.01))
    end

    vk = sparse(product_stoich_coeffs - reactant_stoich_coeffs)
    vk_sum = sum(vk, dims=1)[1, :]

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

    if reorderReactions 
        kinetics = ReorderedKinetics(kinetics)
    end
    #### Transport data

    if false#haskey(npz, "species_viscosities_poly")
        species_viscosities_poly = npz["species_viscosities_poly"]
        thermal_conductivity_poly = npz["thermal_conductivity_poly"]
        binary_diff_coeffs_poly = npz["binary_diff_coeffs_poly"]
        poly_order = size(species_viscosities_poly)[1]
    else
        species_viscosities_poly = zeros(2, 2)
        thermal_conductivity_poly = zeros(2, 2)
        binary_diff_coeffs_poly = zeros(2, 2)
        poly_order = 6
    end

    trans = Transport(poly_order,
                      species_viscosities_poly, 
                      thermal_conductivity_poly, 
                      binary_diff_coeffs_poly)

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