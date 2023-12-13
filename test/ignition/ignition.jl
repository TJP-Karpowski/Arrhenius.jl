using Arrhenius
using DelimitedFiles
using LinearAlgebra
#cd("./example/ignition")

function ignition_test(;plotting=false)
    cantera_data = readdlm("ignition/ignition.dat")
    ct_ts = cantera_data[:, 1]
    ct_T = cantera_data[:, 2]
    ct_Y = cantera_data[:, 3:end]

    gas = CreateSolution("../mechanism/gri30.yaml")
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

    dudt!(du, u, p, t)=isobaric!(du, u, p, t, gas)

    tspan = [0.0, 0.1];
    prob = ODEProblem(dudt!, u0, tspan,[P]);
    u0 = vcat(Y0, T0)

    sol=solve(prob, CVODE_BDF(), reltol = 1e-9, abstol = 1e-12,alg_hints = [:stiff],saveat=ct_ts)
    tmin = 1.e-2
    if plotting
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
        savefig(pltsum, "outputs/ignition.pdf")
    end
    return isapprox(ct_T , sol[end, :] ,rtol=1e-2)
end

