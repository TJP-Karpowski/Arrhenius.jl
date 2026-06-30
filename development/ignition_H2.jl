#cd("/Users/jkarpowski/source-work/Arrhenius/development/")
using Pkg
#Pkg.activate(".")
using LinearAlgebra
using DifferentialEquations
#using ForwardDiff
#using DiffEqSensitivity
using CairoMakie
using Sundials
using BenchmarkTools
#using PProf
#using ModelingToolkit
using Revise
using Arrhenius

#@register Arrhenius.selectPoly(thermo, T,range)

#@register_symbolic wdot_here(reaction::Any, T::Any, C::Any, S0::Any, h_mole::Any)

#cd("/home/jeremy/flame-solver/Arrhenius-jl/development")


const gas_H2 = CreateSolution("../mechanism/h2_li_2004.yaml")
const gas_H2_old = CreateSolution("../mechanism/h2_li_2004.yaml";reorderReactions=false)
ns = gas_H2.n_species

T0 = 980.0 
P = 15 * 101325

X0 = zeros(ns);
X0[species_index(gas_H2, "H2")] = 1.0 / 2.0
X0[species_index(gas_H2, "O2")] = 1.0
X0[species_index(gas_H2, "N2")] = 3.76
X0 = X0 ./ sum(X0);
Y0 = X2Y(gas_H2, X0, dot(X0, gas_H2.MW));

# Y0 = ones(ns) ./ns
u0 = vcat(Y0, T0)
du0 = zeros(length(u0))
functordu! = isobaricReactor(gas_H2)



tspan = [0.0, 0.025];
prob = ODEProblem(functordu!, u0, tspan, [P]);

#dudt!(du, u, p, t)=dudt!(du, u, p, t, gas_H2_old)
#probOld = ODEProblem(dudt!, u0, tspan, [P]);

#sys = modelingtoolkitize(prob)
#prob_jac = ODEProblem(sys,[],tspan ,sparse=true,jac=true) #jac=true
#dudt!(du0, u0, [P],0)

println("Just the call:")
@btime functordu!($du0,$u0,$([P]),$(0))
#@btime dudt!($du0,$u0,$([P]),$(0))
println("------------------------------------------------")


#dudt!(du0, u0, [P],0)



println("CVODE_BDF:")
bm = @benchmark sol = solve(prob, CVODE_BDF(), dtmax=1e-4,dt=1e-8, reltol = 1e-6, abstol = 1e-9,alg_hints = [:stiff],save_everystep = false) seconds = 5
display(bm)

println("Rosenbrock23 (no Autodiff):")
@btime sol = solve(prob, Rosenbrock23(autodiff=false), dtmax=1e-4,dt=1e-12,  reltol = 1e-6, abstol = 1e-9,save_everystep = false)  seconds = 5
#println("Rosenbrock23 (with Autodiff):")
#@btime sol = solve(probOld, Rosenbrock23(autodiff=true), dtmax=1e-4,dt=1e-12, reltol = 1e-6, abstol = 1e-9,save_everystep = false)  seconds = 5
# println("Rodas4:")
# @btime sol = solve(prob, Rodas4(),  reltol = 1e-6, abstol = 1e-9)  seconds = 5
#println("Rodas3:")
#@btime sol = solve(prob, Rodas3(),  reltol = 1e-6, abstol = 1e-9)  seconds = 5

sol = solve(prob, CVODE_BDF(),dtmax=1e-4,dt=1e-8,reltol = 1e-6, abstol = 1e-9,alg_hints = [:stiff])


#@pprof dudt!(du0, u0, [],0)
#"profile.pb.gz"
fig = Figure(size = (900, 400))

# Left plot: species mass fraction
ax1 = Axis(fig[1, 1];
    xlabel = "Time [s]",
    ylabel = "Mass Fraction of H",
    title  = "GRI30 CH4 Ignition @980K/15atm",
)

lines!(ax1, sol.t, sol[species_index(gas_H2, "H2"), :];
    linewidth = 2, label = "Arrhenius.jl"
)
axislegend(ax1, position = :lt)  # legend = :left/top

# Right plot: temperature
ax2 = Axis(fig[1, 2];
    xlabel = "Time [s]",
    ylabel = "Temperature [K]",
)

lines!(ax2, sol.t, sol[end, :];
    linewidth = 2, label = "Arrhenius.jl"
)
axislegend(ax2, position = :lt)

# Boxed frame look (similar to framestyle = :box)
ax1.spinewidth = 1
ax2.spinewidth = 1
hidedecorations!(ax1, grid = false)  # keep ticks/labels; grid off by default
hidedecorations!(ax2, grid = false)

display(fig)