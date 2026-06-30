function read_transport_fixture(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    columns = Dict(name => i for (i, name) in enumerate(header))
    rows = [parse.(Float64, split(line, ',')) for line in lines[2:end]]
    return columns, rows
end

function transport_test(; rtol = 6.0e-2)
    gas = CreateSolution(joinpath(@__DIR__, "..", "..", "mechanism", "h2_li_2004.yaml"))
    columns, rows = read_transport_fixture(joinpath(@__DIR__, "1D_flame_scalars.csv"))

    max_diffusion_error = 0.0
    max_conductivity_error = 0.0
    max_viscosity_error = 0.0

    for row in rows
        T = row[columns["T"]]
        p = row[columns["p"]]
        Y = [row[columns[species]] for species in gas.species_names]
        X = Y2X(gas, Y)
        mean_MW = dot(X, gas.MW)

        mu, lambda, D = Arrhenius.mix_trans(gas, p, T, X, mean_MW)

        max_viscosity_error = max(
            max_viscosity_error,
            abs(mu - row[columns["trans:mu"]]) / abs(row[columns["trans:mu"]]),
        )
        max_conductivity_error = max(
            max_conductivity_error,
            abs(lambda - row[columns["lambda"]]) / abs(row[columns["lambda"]]),
        )

        for (i, species) in enumerate(gas.species_names)
            ref = row[columns["Dmix_" * species]]
            max_diffusion_error = max(max_diffusion_error, abs(D[i] - ref) / abs(ref))
        end
    end

    return max_diffusion_error < rtol &&
           max_conductivity_error < rtol &&
           max_viscosity_error < rtol
end
