using Pkg
Pkg.activate("/home/jeremy/flame-solver")
using LinearAlgebra
using DifferentialEquations
using ForwardDiff
#using DiffEqSensitivity
using Sundials
using Plots
using DelimitedFiles
using Profile
using BenchmarkTools
using ModelingToolkit
#using PProf
using Revise
using Arrhenius




using TimerOutputs
const tmr = TimerOutput();

#@register Arrhenius.selectPoly(thermo, T,range)

#@register_symbolic wdot_here(reaction::Any, T::Any, C::Any, S0::Any, h_mole::Any)

cd("/home/jeremy/flame-solver/Arrhenius-jl/development")


const gas_H2 = CreateSolution("../mechanism/h2o2.yaml",sparseCoeffs=true)
ns = gas_H2.n_species

T0 = 980.0 
P = 15 * 101325

X0 = zeros(ns);
X0[species_index(gas_H2, "H2")] = 1.0 / 2.0
X0[species_index(gas_H2, "O2")] = 1.0
X0[species_index(gas_H2, "AR")] = 0
X0 = X0 ./ sum(X0);
Y0 = X2Y(gas_H2, X0, dot(X0, gas_H2.MW));

# Y0 = ones(ns) ./ns
u0 = vcat(Y0, T0)
du0 = zeros(length(u0))

mutable struct isobaricReactor{Type<:Number,Sol<:Arrhenius.Solution} <: Function
    gas::Sol
    nasa_poly::Array{Type}
    T::Type 
    p::Type
    X::Vector{Type}
    C::Vector{Type}
    h_RT::Vector{Type}
    s0_R::Vector{Type}
    cp_R::Vector{Type}
    enthalpy_T_poly::Vector{Type}
    entropy_T_poly::Vector{Type}
    cp_T_poly::Vector{Type}
    ρ_mass::Type
    mean_MW::Type
    cp_mass::Type
   
end

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
    zeros(6),
    zeros(6),
    zeros(5),
    0.0,
    0.0,0.0)
end


@inbounds function cal_h_RT!(state::isobaricReactor,T,p)
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

@inbounds function h_RT_kernel!(h_RT,nasa_poly,enthalpy_T_poly)
    h_RT .=0.
    for j in 1:6
        for i in 1:length(h_RT)
            h_RT[i] += nasa_poly[i,j] * enthalpy_T_poly[j]
        end
    end
end


function cal_s0_R!(state::isobaricReactor,T,p)
    state.entropy_T_poly[1] = log(T)
    state.entropy_T_poly[2] = T
    state.entropy_T_poly[3] = T^2/2.
    state.entropy_T_poly[4] = T^3/3.
    state.entropy_T_poly[5] = T^4/4.
    state.entropy_T_poly[6] = 1

    s0_R_kernel!(state.s0_R,state.nasa_poly,state.entropy_T_poly)
    

    return nothing
end

@inbounds function s0_R_kernel!(s0_R,nasa_poly,entropy_T_poly)
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

@inbounds function cal_cpmass_mean!(state::isobaricReactor,T,p)
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

@inbounds function cp_R_kernel!(cp_R,nasa_poly,cp_T_poly)
    cp_R .=0.
    for j in 1:5
        for i in 1:length(cp_R)
            cp_R[i] += nasa_poly[i,j] * cp_T_poly[j]
        end
    end
end


function cal_nasa_poly!(state::isobaricReactor,T,p)
    cal_nasa_poly!(state.nasa_poly,state.gas.thermo.nasa_low,state.gas.thermo.nasa_high,T,p)
    return nothing
end

@inbounds function cal_nasa_poly!(nasa_poly,nasa_low,nasa_high,T,p)
    ramp_low = (1-(0.5 * tanh(T-1000) + 0.5))
    ramp_high =  (0.5 * tanh(T-1000) + 0.5)
      @. nasa_poly = ramp_low *  nasa_low + ramp_high * nasa_high
    return nothing
end

function calc_meanMW(Y::AbstractVector,MW::AbstractVector)
    out = 0.0
    for i in 1:length(Y)
        out += Y[i] /MW[i]
    end
    return 1/out
end


@inbounds function (state::isobaricReactor)(du, u, p, t)
    state.p = p[1]
    state.T = u[end]
    Y = @view(u[1:state.gas.n_species])

    state.mean_MW = calc_meanMW(Y,state.gas.MW) 
    state.ρ_mass = state.p / R / state.T * state.mean_MW
    @. state.X = Y / state.gas.MW * state.mean_MW 
    @. state.C = state.X * state.ρ_mass / state.mean_MW
    cal_nasa_poly!(state,state.T,state.p)
    cal_cpmass_mean!(state,state.T,state.p)
    #(state.cp_mass-cal_cpmass_mean(state.gas,state.T,state.p,state.X))
    #state.cp_mass = cal_cpmass_mean(state.gas,state.T,state.p,state.X)
    #state.h_RT = cal_h_RT(state.gas,state.T,state.p,state.X)
    #state.s0_R = cal_s0_R(state.gas,state.T,state.p,state.X)
    cal_h_RT!(state,state.T,state.p)
    cal_s0_R!(state,state.T,state.p)
    #wY = du[1:end-1]
    @timeit tmr "calcOmaga" calc_omegaY!( @view(du[1:end-1]),state.gas, state.T, state.C, state.s0_R, state.h_RT,state.ρ_mass) 
    #du[1:end-1] .= wdot_func(state.gas.kinetics, state.T, state.C, state.s0_R * Arrhenius.R, state.h_RT * Arrhenius.R * state.T )
   
    du[end] *= 0.0
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



functordu! = isobaricReactor(gas_H2)
# function f_jac(J,u,p,t)
#     J.= forwardDiff.jacobian(x->dudt!())
# end


tspan = [0.0, 0.025];
prob = ODEProblem(functordu!, u0, tspan, [P]);
#sys = modelingtoolkitize(prob)
#prob_jac = ODEProblem(sys,[],tspan ,sparse=true,jac=true) #jac=true
u0 = vcat(Y0, T0)
du0 = zeros(gas_H2.n_species+1)
#dudt!(du0, u0, [P],0)

#dudt!(du0, u0, [P],0)
println("CVODE_BDF:")
#@btime sol = solve(prob, CVODE_BDF(), dtmax=1e-4, reltol = 1e-6, abstol = 1e-9,alg_hints = [:stiff],saveat=0.1) seconds = 5
#println("Time jacobian:")
#@btime solve(prob_jac, CVODE_BDF(), reltol = 1e-6, abstol = 1e-9,alg_hints = [:stiff]) seconds = 5

#println("radau5:")
#@btime sol = solve(prob, radau5(autodiff=false),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
#println("Rosenbrock23 (no Autodiff):")
#@btime sol = solve(prob, Rosenbrock23(autodiff=false),  reltol = 1e-6, abstol = 1e-9,saveat=0.1)  seconds = 5
#println("Rosenbrock23 (with Autodiff):")
#@btime sol = solve(prob, Rosenbrock23(autodiff=true),  reltol = 1e-6, abstol = 1e-9,saveat=0.1)  seconds = 5
# println("Rodas4:")
# @btime sol = solve(prob, Rodas4(),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
#println("Rodas3:")
#@btime sol = solve(prob, Rodas3(),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
println("Just the call:")
@btime functordu!($du0,$u0,$([P]),$(0))
println("------------------------------------------------")
display(tmr)
 reset_timer!(tmr);
#sol = solve(prob, CVODE_BDF(),dtmax=1e-4,dt=1e-8,reltol = 1e-6, abstol = 1e-9,alg_hints = [:stiff])
plt = plot(
    sol.t ,
    sol[species_index(gas_H2, "H2"), :],
    lw = 2,
    label = "Arrhenius.jl",
);
ylabel!(plt, "Mass Fraction of H")
xlabel!(plt, "Time [s]")
pltT = plot(sol.t , sol[end, :], lw = 2, label = "Arrhenius.jl");
ylabel!(pltT, "Temperature [K]")
xlabel!(pltT, "Time [s]")
title!(plt, "GRI30 CH4 Ignition @980K/15atm")
pltsum = plot(plt, pltT, xscale = :identity, legend = :left, framestyle = :box)
display(pltsum)

#@pprof dudt!(du0, u0, [],0)
#"profile.pb.gz"
