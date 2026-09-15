function isobaric!(du, u, p, t, gas)
    T = u[end]
    pressure =p[1]
    Y = @view(u[1:gas.n_species])
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    ρ_mass = pressure / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mass =  Arrhenius.cal_cpmass_mean(gas,T,pressure,X)
    h_RT = Arrhenius.cal_h_RT(gas, T, pressure, X)
    S0_R = Arrhenius.cal_s0_R(gas, T, pressure, X)
    #wY = du[1:end-1]
    #@timeit tmr "new" 
    calc_omegaY!( @view(du[1:gas.n_species]),gas, T, C, S0_R, h_RT,ρ_mass) 
    du[1:gas.n_species] =du[1:gas.n_species] .* gas.MW ./ ρ_mass
    #@timeit tmr "old" wdot_new = wdot_func(gas.kinetics, T, C, S0_R * R, h_RT * R * T)
    #du[1:end-1] = wdot_new / ρ_mass .* gas.MW
    du[end] = -dot(h_RT * R * T ./ gas.MW, du[1:gas.n_species]) / cp_mass
    return nothing
end

function isochoric!(du, u, p, t, gas)
    T = u[end]
    ρ_mass = p[1]
    Y = @view(u[1:gas.n_species])
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    pressure = ρ_mass * R * T / mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    #C = Y2C(gas, Y, ρ_mass)
    h_RT = Arrhenius.cal_h_RT(gas, T, pressure, X)
    S0_R = Arrhenius.cal_s0_R(gas, T, pressure, X)
    calc_omegaY!( @view(du[1:gas.n_species]),gas, T, C, S0_R, h_RT,ρ_mass) 
    du[1:gas.n_species] .*=  gas.MW ./ ρ_mass
    du[end] = -dot(Arrhenius.cal_umass(gas, T, pressure, X), @view(du[1:gas.n_species])) / Arrhenius.cal_cvmass_mean(gas,T,pressure,X) # dU = cv dT = Q_dot => Q_dot/cv = dT
    return nothing
end

export isobaric!, isochoric!




### The following functions are optimized versions of the above isobaric! function
### that retain memory allocations within a mutable struct
### to avoid overhead of allocations within the ODE function call   

"""
    isobaricReactor(gas)
Create an isobaric reactor struct to be used as an ODE function with the given gas object
"""
function isobaricReactor(gas) 
return isobaricReactor(
    gas,
    zeros(gas.n_species,7),
    300.0,
    101325.0,
    zeros(gas.n_species),
    zeros(gas.n_species),
    zeros(gas.n_species),
    zeros(gas.n_species),
    zeros(gas.n_species),
    zeros(gas.n_species),
    zeros(gas.n_reactions),
    zeros(gas.n_reactions),
    zeros(6),
    zeros(6),
    zeros(5),
    0.0,
    0.0,0.0)
end

"""
    isobaricReactor(gas, ::Type{T}) where {T}

Create an isobaric reactor workspace whose floating-point work arrays have
element type `T`. This is useful for automatic differentiation and other
number types that need to flow through the reactor RHS.
"""
function isobaricReactor(gas, ::Type{T}) where {T}
return isobaricReactor(
    gas,
    zeros(T, gas.n_species, 7),
    T(300),
    T(101325),
    zeros(T, gas.n_species),
    zeros(T, gas.n_species),
    zeros(T, gas.n_species),
    zeros(T, gas.n_species),
    zeros(T, gas.n_species),
    zeros(T, gas.n_species),
    zeros(T, gas.n_reactions),
    zeros(T, gas.n_reactions),
    zeros(T, 6),
    zeros(T, 6),
    zeros(T, 5),
    zero(T),
    zero(T),
    zero(T))
end

"""
    cal_h_RT!(state::isobaricReactor,T,p)

    Calculates the dimensionless enthalpy h/RT and stores it in state.h_RT
    using the NASA polynomial coefficients stored in state.nasa_poly
"""
 function cal_h_RT!(state::isobaricReactor,T,p)
    state.enthalpy_T_poly[1] = 1.0
    state.enthalpy_T_poly[2] = T/2
    state.enthalpy_T_poly[3] = T^2/3.
    state.enthalpy_T_poly[4] = T^3/4.
    state.enthalpy_T_poly[5] = T^4/5.
    state.enthalpy_T_poly[6] = 1/T
    
        #poly = selectPoly(thermo,T,1:6)
    #h_mole = poly * enthalpy_T_poly

  #state.h_RT .= @view(state.nasa_poly[:,1:6]) * state.enthalpy_T_poly
    h_RT_kernel!(state.h_RT,state.nasa_poly,state.enthalpy_T_poly)
    

    return nothing
end
"""
    h_RT_kernel!(h_RT,nasa_poly,enthalpy_T_poly)

    Helper Kernel to calculate h/RT from NASA polynomial coefficients and enthalpy temperature polynomial terms.
    Kernel to avoid allocations.

"""
function h_RT_kernel!(h_RT,nasa_poly,enthalpy_T_poly)
    h_RT .=0.
    for j in 1:6
        for i in 1:length(h_RT)
            h_RT[i] += nasa_poly[i,j] * enthalpy_T_poly[j]
        end
    end
end


"""
    cal_s0_R!(state::isobaricReactor,T,p)

calculates the dimensionless entropy s/R and stores it in state.s0_R
using the NASA polynomial coefficients stored in state.nasa_poly.
"""
function cal_s0_R!(state::isobaricReactor,T,p)
    state.entropy_T_poly[1] = safe_log(T)
    state.entropy_T_poly[2] = T
    state.entropy_T_poly[3] = T^2/2.
    state.entropy_T_poly[4] = T^3/3.
    state.entropy_T_poly[5] = T^4/4.
    state.entropy_T_poly[6] = 1

    s0_R_kernel!(state.s0_R,state.nasa_poly,state.entropy_T_poly)
    

    return nothing
end

"""
    s0_R_kernel!(s0_R,nasa_poly,entropy_T_poly)

    Helper Kernel to calculate s/R from NASA polynomial coefficients and entropy temperature polynomial terms.
    Kernel to avoid allocations.

"""
function s0_R_kernel!(s0_R,nasa_poly,entropy_T_poly)
    s0_R .=0.
    for j in 1:5
        for i in 1:length(s0_R)
            s0_R[i] += nasa_poly[i,j] * entropy_T_poly[j]
        end
    end
    for i in 1:length(s0_R)
        s0_R[i] += nasa_poly[i,7] * entropy_T_poly[6]
    end
end

"""
    cal_cpmass_mean!(state::isobaricReactor,T,p)

    Calculates the mean mass-based specific heat capacity cp_mass and stores it in state.cp_mass
    using the NASA polynomial coefficients stored in state.nasa_poly
"""
function cal_cpmass_mean!(state::isobaricReactor,T,p)
    state.cp_T_poly[1] = 1.0
    state.cp_T_poly[2] = T
    state.cp_T_poly[3] = T^2
    state.cp_T_poly[4] = T^3
    state.cp_T_poly[5] = T^4
    cp_R_kernel!(state.cp_R,state.nasa_poly,state.cp_T_poly)
    state.cp_mass *= 0.0
    for i in 1:state.gas.n_species
        state.cp_mass += state.cp_R[i] * state.X[i]
    end
    state.cp_mass *= Arrhenius.R / state.mean_MW

    #state.cp_R .= @view(state.nasa_poly[:,1:5]) * state.cp_T_poly
    #state.cp_mass = dot(state.X,state.cp_R ) * Arrhenius.R / state.mean_MW

    return nothing
end

"""
    cp_R_kernel!(cp_R,nasa_poly,cp_T_poly)

    Helper Kernel to calculate cp/R from NASA polynomial coefficients and cp temperature polynomial terms.
    Kernel to avoid allocations.

"""
function cp_R_kernel!(cp_R,nasa_poly,cp_T_poly)
    cp_R .=0.
    for j in 1:5
        for i in 1:length(cp_R)
            cp_R[i] += nasa_poly[i,j] * cp_T_poly[j]
        end
    end
end

"""
    cal_nasa_poly!(state::isobaricReactor,T,p)

    Helper function to make call easier
"""
function cal_nasa_poly!(state::isobaricReactor,T,p)
    thermo = state.gas.thermo
    if thermo.isTcommon
        state.nasa_poly .= T <= thermo.Trange[1, 2] ? thermo.nasa_low : thermo.nasa_high
    else
        @inbounds for species in axes(state.nasa_poly, 1)
            coefficients = T <= thermo.Trange[species, 2] ? thermo.nasa_low : thermo.nasa_high
            for column in axes(state.nasa_poly, 2)
                state.nasa_poly[species, column] = coefficients[species, column]
            end
        end
    end
    return nothing
end

"""
    cal_nasa_poly!(nasa_poly,nasa_low,nasa_high,T,p)

    Calculates the NASA polynomial coefficients for all species at temperature T and pressure p
    and stores them in nasa_poly. 
"""
function cal_nasa_poly!(nasa_poly,nasa_low,nasa_high,T,p)
    #ramp_low = (1-(0.5 * tanh(T-1000) + 0.5))
    ramp_high =  (0.5 * tanh(T-1000) + 0.5)
    nasa_poly .= T<=1000 ? nasa_low : nasa_high
    return nothing
end

"""
    calc_meanMW(Y,MW)

    Calculates the mean molecular weight from mass fractions Y and molecular weights MW
"""
function calc_meanMW(Y::AbstractVector,MW::AbstractVector)
    out = 0.0
    for i in 1:length(Y)
        out += Y[i] /MW[i]
    end
    return 1/out
end


"""
    updateReactor!(sate::Reactor,Y::AbstractVector,T::Real,p::Real)

    updates the state Object based on the provided Y (mass-fraction), T (temperature) and p, (pressure)
"""
function updateReactor!(state::isobaricReactor,Y::AbstractVector,T::Real,p::Real)
    state.p = p
    state.T = T
    state.mean_MW = calc_meanMW(Y,state.gas.MW) 
    state.ρ_mass = state.p / R / state.T * state.mean_MW
    @. state.X = Y / state.gas.MW * state.mean_MW 
    @. state.C = state.X * state.ρ_mass / state.mean_MW
    cal_nasa_poly!(state,state.T,state.p)
    cal_cpmass_mean!(state,state.T,state.p)
    cal_h_RT!(state,state.T,state.p)
    cal_s0_R!(state,state.T,state.p)
    @. state.g_RT = state.h_RT - state.s0_R
    return nothing
end



"""
    (state::isobaricReactor)(du, u, p, t)

    ODE function call for isobaric reactor using the state stored in the isobaricReactor struct
"""
function (state::isobaricReactor)(du, u, p, t)

    Y = @view(u[1:state.gas.n_species])

    updateReactor!(state,Y,u[end],p[1])
    wdot_func!(@view(du[1:end-1]),state.kf,state.kr,state.C,state.g_RT,state.gas.kinetics,state.T)    
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


export isobaricReactor, updateReactor!
