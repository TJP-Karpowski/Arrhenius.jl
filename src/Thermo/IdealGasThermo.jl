"""Struct for the ideal gas thermo.

nasa_low: Array with low temperature nasa coeff. for each species

nasa_high: Array with high temperature nasa coeff. for each species

Trange: Array with temperature ranges for each species

isTcommon: bool which indicates if both polynoms share same T at intersection

"""

struct IdealGasThermo <: Thermo 
    nasa_low::Array{Float64,2}
    nasa_high::Array{Float64,2}
    Trange::Array{Float64,2}
    isTcommon::Bool
end

"""
Constructor for the idealGasThermo:

yaml:: Dict of the input yaml file
"""
function IdealGasThermo(yaml::AbstractDict, phase = phase_data(yaml))
    n_species, n_reactions, species_names,
    elements, n_elements, ele_matrix = read_species_basics(yaml, phase)

    nasa_low = zeros(n_species, 7)
    nasa_high = zeros(n_species, 7)
    Trange = zeros(n_species, 3)

    species_by_name = species_dict_by_name(yaml["species"])

    for (i, species) in enumerate(species_names)
        spec = species_entry_from_lookup(species_by_name, species)
        nasa_low[i, :] = spec["thermo"]["data"][1]
        nasa_high[i, :] = spec["thermo"]["data"][2]
        Trange[i, :] .= spec["thermo"]["temperature-ranges"]

        for j = 1:n_elements
            if haskey(spec["composition"], elements[j])
                ele_matrix[j, i] = spec["composition"][elements[j]]
            end
        end
    end
    isTcommon = (maximum(Trange[:, 2]) - minimum(Trange[:, 2])) < 0.01
    return  IdealGasThermo(nasa_low, nasa_high, Trange, isTcommon)
end

# Evaluate one selected NASA7 interval without blending or matrix temporaries.
# Branch comparisons work for ForwardDiff Dual temperatures; the selected
# polynomial remains differentiable away from interval boundaries.
function nasa_property(thermo::IdealGasThermo, T::Real, columns, terms)
    result = zeros(promote_type(eltype(thermo.nasa_low), typeof(T)), size(thermo.nasa_low, 1))
    @inbounds for species in eachindex(result)
        coefficients = T <= thermo.Trange[species, 2] ? thermo.nasa_low : thermo.nasa_high
        value = zero(eltype(result))
        for term in eachindex(terms)
            value += coefficients[species, columns[term]] * terms[term]
        end
        result[species] = value
    end
    return result
end

function selectPoly(thermo::IdealGasThermo, T::Real, range)
    if thermo.isTcommon
        coefficients = T <= thermo.Trange[1, 2] ? thermo.nasa_low : thermo.nasa_high
        return @view coefficients[:, range]
    end
    coefficients = similar(thermo.nasa_low)
    @inbounds for species in axes(coefficients, 1)
        selected = T <= thermo.Trange[species, 2] ? thermo.nasa_low : thermo.nasa_high
        for column in axes(coefficients, 2)
            coefficients[species, column] = selected[species, column]
        end
    end
    return @view coefficients[:, range]
end
#@register_symbolic selectPoly(thermo::IdealGasThermo, T::Real,range)

"""
    cal_h_RT(gas, T, p, X)

calculates the dimensionless mole based enthalpy (h) for each species
"""
function cal_h_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return nasa_property(thermo, T, (1, 2, 3, 4, 5, 6), (one(T), T/2, T^2/3, T^3/4, T^4/5, one(T)/T))
end

"""
    cal_s0_R(gas, T, p, X)

calculates the dimensionless mole based reference state entropy (s0) for each species
"""
function cal_s0_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return nasa_property(thermo, T, (1, 2, 3, 4, 5, 7), (log(T), T, T^2/2, T^3/3, T^4/4, one(T)))
end
export cal_s0_R

"""
    cal_s_R(gas, T, p, X)

calculates the dimensionless mole based entropy (s) for each species
"""
function cal_s_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
   return cal_s0_R(gas,thermo, T,p,X) - log.(max.(X,1e-30)) .- log(p/one_atm)
end

"""
    cal_g_RT(gas, T, p, X)

calculates the dimensionless mole based free gibbs energy (g) for each species
"""
function cal_g_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return cal_h_RT(gas,thermo, T, p, X) - cal_s_R(gas,thermo, T, p, X)
end

"""
    cal_u_RT(gas, T, p, X)

calculates the dimensionless mole based internal energy (u) for each species
"""
function cal_u_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return cal_h_RT(gas,thermo, T, p, X) .- 1
end

"""
    cal_a_RT(gas, T, p, X)

calculates the dimensionless mole based helmholz free energy (a) for each species
"""
function cal_a_RT(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return cal_u_RT(gas,thermo, T, p, X) - cal_s_R(gas,thermo, T, p, X)
end

"""
    cal_cp_R(gas, T, p, X)

calculates the dimensionless mole based heat capacity 
at constant pressure (cp) for each species
"""
function cal_cp_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
    return nasa_property(thermo, T, (1, 2, 3, 4, 5), (one(T), T, T^2, T^3, T^4))
end

"""
    cal_cv_R(gas, T, p, X)

calculates the dimensionless mole based heat capacity 
at constant volume (cv) for each species
"""
function cal_cv_R(gas::Solution,thermo::IdealGasThermo, T::Real, p::Real, X::AbstractArray)
   return cal_cp_R(gas,thermo, T, p, X) .- 1
end
