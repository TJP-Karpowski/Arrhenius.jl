
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
using IfElse
using PProf
using Revise
using Arrhenius

using TimerOutputs
const tmr = TimerOutput();
#@register_symbolic Arrhenius.cal_cpmass_mean(gas::Arrhenius.Solution, T::Num, p::Float64, X::Vector{Num})
#@register_symbolic cal_h(gas::Arrhenius.Solution, T::Any, p::Any, X::Any)
#@register_symbolic Arrhenius.cal_s0(gas::Arrhenius.Solution, T::Real, p::Any, X::Any) 
#@register_symbolic wdot_here(reaction::Any, T::Any, C::Any, S0::Any, h_mole::Any)

#cd("./example/ignition")

cantera_data = readdlm("ignition.dat")
ct_ts = cantera_data[:, 1]
ct_T = cantera_data[:, 2]
ct_Y = cantera_data[:, 3:end]

const gas = CreateSolution("../../mechanism/gri30.yaml",sparseCoeffs=true)
ns = gas.n_species

T0 = 980.0
P = one_atm * 15.0

X0 = zeros(ns);
X0[species_index(gas, "CH4")] = 1.0 / 2.0
X0[species_index(gas, "O2")] = 1.0
X0[species_index(gas, "N2")] = 3.76
X0 = X0 ./ sum(X0);
Y0 = X2Y(gas, X0, dot(X0, gas.MW));

# Y0 = ones(ns) ./ns
u0 = vcat(Y0, T0)
@inbounds function dudt!(du, u, p, t, gas)
    T = u[end]
    Y = @view(u[1:ns])
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    ρ_mass = P / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    @timeit tmr "cp" cp_mass =  cal_cpmass_mean(gas,T,P,X)#IfElse.ifelse(T <= 1000.0,@view(gas.thermo.nasa_low[:, 1:5])*[1.0, T, T^2, T^3, T^4] ,@view(gas.thermo.nasa_high[:, 1:5]) *[1.0, T, T^2, T^3, T^4])#cal_cpmass_mean(gas,T,P,X)#get_cp(gas, T, X, mean_MW)
    @timeit tmr "h" h_RT = cal_h_RT(gas, T, P, X)
    @timeit tmr "S0" S0_R = cal_s0_R(gas, T, P, X)
    #wY = du[1:end-1]
    @timeit tmr "wdot" calc_omegaY!( @view(du[1:end-1]), gas, T, C, S0_R, h_RT,ρ_mass,tmr) 
    du[1:end-1] =du[1:end-1] .* gas.MW ./ ρ_mass
    #println("difference: ",maximum(wdot_new)-maximum(wdot),"   ",minimum(wdot_new)-minimum(wdot))
    #wdot_new = wdot_here(gas.kinetics, T, C, S0_R * R, h_RT * R * T)
    #du[1:end-1] = wdot_new / ρ_mass .* gas.MW
    du[end] = -dot(h_RT * R * T ./ gas.MW, du[1:end-1]) / cp_mass
    return nothing
end
dudt!(du, u, p, t)=dudt!(du, u, p, t, gas)
# function f_jac(J,u,p,t)
#     J.= forwardDiff.jacobian(x->dudt!())
# end


tspan = [0.0, 0.1];
prob = ODEProblem(dudt!, u0, tspan);
#sys = modelingtoolkitize(prob)

println("CVODE_BDF:")
u0 = vcat(Y0, T0)
du0 = zeros(54)
dudt!(du0, u0, [],0)
reset_timer!(tmr);
dudt!(du0, u0, [],0)
#@btime solve(prob, CVODE_BDF(), reltol = 1e-6, abstol = 1e-9,alg_hints = [:stiff]) seconds = 5

#println("radau5:")
#@btime sol = solve(prob, radau5(autodiff=false),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
# println("Rosenbrock23 (no Autodiff):")
# @btime sol = solve(prob, Rosenbrock23(autodiff=false),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
#println("Rosenbrock23 (with Autodiff):")
#@btime sol = solve(prob, Rosenbrock23(autodiff=true),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
# println("Rodas4:")
# @btime sol = solve(prob, Rodas4(),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
#println("Rodas3:")
#@btime sol = solve(prob, Rodas3(),  reltol = 1e-6, abstol = 1e-9)  seconds = 5


sol = solve(prob, CVODE_BDF(),  reltol = 1e-6, abstol = 1e-9,alg_hints = [:stiff])
tmin = 1.e-2
plt = plot(
    sol.t .+ tmin,
    sol[species_index(gas, "H"), :],
    lw = 2,
    label = "Arrhenius.jl",
);
plot!(plt, ct_ts .+ tmin, ct_Y[:, species_index(gas, "H")], label = "Cantera")
ylabel!(plt, "Mass Fraction of H")
xlabel!(plt, "Time [s]")
pltT = plot(sol.t .+ tmin, sol[end, :], lw = 2, label = "Arrhenius.jl");
plot!(pltT, ct_ts .+ tmin, ct_T, label = "Cantera")
ylabel!(pltT, "Temperature [K]")
xlabel!(pltT, "Time [s]")
title!(plt, "GRI30 CH4 Ignition @980K/15atm")
pltsum = plot(plt, pltT, xscale = :identity, legend = :left, framestyle = :box)
display(pltsum)
#png(pltsum, "CH4_ignition.png")

u0 = vcat(Y0, T0)
du0 = zeros(54);
show(tmr)
#@pprof dudt!(du0, u0, [],0)
#"profile.pb.gz"