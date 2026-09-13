using Arrhenius
using LinearAlgebra
using OrdinaryDiffEq
using Sundials
using Test

function read_stagni_ignition_profiles(path)
    lines = readlines(path)
    metadata = Dict{String,String}()
    data_lines = String[]
    for line in lines
        if startswith(line, "# ")
            key, value = split(line[3:end], "="; limit = 2)
            metadata[key] = value
        elseif !isempty(strip(line))
            push!(data_lines, line)
        end
    end

    header = split(first(data_lines), ',')
    species = header[4:end]
    profiles = Dict{String,Vector{Vector{Float64}}}()
    for line in @view data_lines[2:end]
        fields = split(line, ',')
        row = parse.(Float64, fields[2:end])
        push!(get!(profiles, fields[1], Vector{Vector{Float64}}()), row)
    end
    return (; metadata, species, profiles)
end

function stagni_mass_fractions(gas, composition)
    X = zeros(gas.n_species)
    for (species, amount) in composition
        X[species_index(gas, species)] = amount
    end
    X ./= sum(X)
    return X2Y(gas, X, dot(X, gas.MW))
end

function stagni_threshold_crossing_time(times, values, threshold)
    index = findfirst(>=(threshold), values)
    index === nothing && return Inf
    index == 1 && return times[1]
    fraction = (threshold - values[index - 1]) /
        (values[index] - values[index - 1])
    return times[index - 1] + fraction * (times[index] - times[index - 1])
end

@testset "Stagni 2025 NH3 ignition profiles against Cantera" begin
    reference = read_stagni_ignition_profiles(joinpath(
        @__DIR__, "cantera_stagni_2025_profiles.csv"))
    @test reference.metadata["cantera_version"] == "3.2.0"

    gas = CreateSolution(joinpath(
        @__DIR__, "..", "..", "mechanism", "nh3_stagni_2025.yaml"))
    @test gas.n_species == 34
    @test gas.n_reactions == 273
    @test gas.species_names == reference.species

    cases = Dict(
        "nh3_air_1000K_20bar" => (
            composition = ("NH3" => 1.0, "O2" => 0.75, "N2" => 2.82),
            temperature = 1000.0,
            pressure = 2.0e6,
        ),
        "nh3_h2_air_1000K_10bar" => (
            composition = ("NH3" => 0.5, "H2" => 0.5,
                           "O2" => 0.625, "N2" => 2.35),
            temperature = 1000.0,
            pressure = 1.0e6,
        ),
    )
    profile_species = ("NH3", "H2", "O2", "H2O", "N2", "NO", "N2O", "NH2")

    for (name, case) in cases
        @testset "$name" begin
            rows = reference.profiles[name]
            times = [row[1] for row in rows]
            reference_temperature = [row[2] for row in rows]
            reference_Y = reduce(hcat, (row[3:end] for row in rows))

            Y0 = stagni_mass_fractions(gas, case.composition)
            u0 = vcat(Y0, case.temperature)
            problem = ODEProblem(
                isobaricReactor(gas),
                u0,
                (first(times), last(times)),
                (case.pressure,),
            )
            solution = solve(problem, CVODE_BDF();
                reltol = 1.0e-10,
                abstol = 1.0e-16,
                saveat = times,
            )

            @test string(solution.retcode) == "Success"
            computed_temperature = [u[end] for u in solution.u]
            computed_Y = reduce(hcat,
                (@view(u[1:gas.n_species]) for u in solution.u))
            @test computed_temperature ≈ reference_temperature rtol = 5.0e-4 atol = 1.0
            @test computed_Y ≈ reference_Y rtol = 3.0e-3 atol = 2.0e-8

            for species in profile_species
                @testset "$species mass-fraction profile" begin
                    index = species_index(gas, species)
                    reference_profile = reference_Y[index, :]
                    profile_scale = max(maximum(abs, reference_profile), 1.0e-6)
                    profile_error = maximum(abs,
                        computed_Y[index, :] .- reference_profile)
                    @test profile_error <= 1.5e-2 * profile_scale + 2.0e-8
                end
            end

            reference_delay = stagni_threshold_crossing_time(
                times, reference_temperature, case.temperature + 400.0)
            computed_delay = stagni_threshold_crossing_time(
                times, computed_temperature, case.temperature + 400.0)
            @test isfinite(reference_delay)
            @test computed_delay ≈ reference_delay rtol = 2.0e-3
        end
    end
end
