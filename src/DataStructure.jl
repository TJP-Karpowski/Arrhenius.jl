abstract type Kinetics end


struct StandardKinetics<: Kinetics
    product_stoich_coeffs::SparseMatrixCSC{Float64,Int64}
    reactant_stoich_coeffs::SparseMatrixCSC{Float64,Int64}
    reactant_orders::SparseMatrixCSC{Float64,Int64}
    is_reversible::Array{Bool,1}
    Arrhenius_coeffs::Array{Float64,2}
    Arrhenius_0::Array{Float64,2}
    Troe_::Array{Float64,2}
    index_three_body::Array{Int64,1}
    index_falloff::Array{Int64,1}
    index_falloff_Troe::Array{Int64,1}
    efficiencies_coeffs::SparseMatrixCSC{Float64,Int64}
    i_reactant::Array{Array{Int64,1},1}
    i_product::Array{Array{Int64,1},1}
    n_reactions::Int64
    vk::SparseMatrixCSC{Float64,Int64}
    vk_sum::Array{Float64,1}
end

abstract type Thermo end



struct ReorderedKinetics{Ty<:AbstractArray,sTy<:AbstractArray,vTy<:AbstractVector} <: Kinetics
    arrhenius_coeffs::Ty
    reactant_orders::sTy
    product_stoich_coeffs::sTy
    reactant_stoich_coeffs::sTy
    vk::sTy
    vk_rev::sTy
    vk_sum::vTy
    efficiency_coeffs_tB::sTy
    efficiency_coeffs_fO::sTy
    Arrhenius_0::Ty
    troe::Ty
    index_rev_start::Int64
    index_three_body_start::Int64
    index_fallOff_start::Int64
    index_troe_start::Int64
end

struct Transport
    poly_order::Int64
    species_viscosities_poly::Array{Float64,2}
    thermal_conductivity_poly::Array{Float64,2}
    binary_diff_coeffs_poly::Array{Float64,2}
end

struct Solution{Ty<:Thermo,kinTy<:Kinetics}
    n_species::Int64
    n_reactions::Int64
    MW::Vector{Float64}
    species_names::Vector{String}
    elements::Vector{String}
    ele_matrix::Array{Float64,2}
    thermo::Ty
    trans::Transport
    kinetics::kinTy
end

### Isobaric Reactor struct to retain variables for ODE function without allocation overhead
mutable struct isobaricReactor{vecT<:AbstractVector,arrayT<:AbstractArray,typ<:Real,Sol<:Arrhenius.Solution} <: Function
    gas::Sol
    nasa_poly::arrayT
    T::typ 
    p::typ
    X::vecT
    C::vecT
    h_RT::vecT
    s0_R::vecT
    g_RT::vecT
    cp_R::vecT
    kr::vecT
    kf::vecT
    enthalpy_T_poly::vecT
    entropy_T_poly::vecT
    cp_T_poly::vecT
    ρ_mass::typ
    mean_MW::typ
    cp_mass::typ
end