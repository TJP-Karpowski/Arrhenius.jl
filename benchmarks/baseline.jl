#!/usr/bin/env julia

using Arrhenius
using Dates
using LinearAlgebra
using Printf
using Statistics

const MECHANISMS = [
    "h2o2" => joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"),
    "h2_li_2004" => joinpath(@__DIR__, "..", "mechanism", "h2_li_2004.yaml"),
    "gri30" => joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"),
]

const DEFAULT_PRESSURE = Arrhenius.one_atm
const DEFAULT_TEMPERATURE = 1200.0

function balanced_mass_fractions(gas)
    Y = fill(1 / gas.n_species, gas.n_species)
    return Y ./ sum(Y)
end

function methane_air_mass_fractions(gas)
    X = zeros(gas.n_species)
    for (species, amount) in (("CH4", 0.5), ("O2", 1.0), ("N2", 3.76))
        idx = species_index(gas, species)
        idx === nothing && return balanced_mass_fractions(gas)
        X[idx] = amount
    end
    X ./= sum(X)
    return X2Y(gas, X, dot(X, gas.MW))
end

function hydrogen_air_mass_fractions(gas)
    X = zeros(gas.n_species)
    for (species, amount) in (("H2", 2.0), ("O2", 1.0), ("N2", 3.76))
        idx = species_index(gas, species)
        idx === nothing && return balanced_mass_fractions(gas)
        X[idx] = amount
    end
    X ./= sum(X)
    return X2Y(gas, X, dot(X, gas.MW))
end

function initial_mass_fractions(name, gas)
    occursin("gri30", name) && return methane_air_mass_fractions(gas)
    return hydrogen_air_mass_fractions(gas)
end

function sample!(f; samples::Int, evals::Int)
    f()
    times_ns = Vector{Int64}(undef, samples)
    allocations = Vector{Int64}(undef, samples)
    for i in 1:samples
        GC.gc(false)
        allocations[i] = @allocated begin
            for _ in 1:evals
                f()
            end
        end
        t0 = time_ns()
        for _ in 1:evals
            f()
        end
        times_ns[i] = time_ns() - t0
    end
    times = times_ns ./ evals
    allocs = allocations ./ evals
    return (
        min_ns = minimum(times),
        median_ns = median(times),
        mean_ns = mean(times),
        max_ns = maximum(times),
        alloc_bytes = median(allocs),
    )
end

function benchmark_case(name, mechanism; samples::Int, evals::Int)
    rows = NamedTuple[]

    load_stats = sample!(samples=samples, evals=1) do
        CreateSolution(mechanism)
    end
    gas = CreateSolution(mechanism)

    Y = initial_mass_fractions(name, gas)
    u = vcat(Y, DEFAULT_TEMPERATURE)
    du = similar(u)
    state = isobaricReactor(gas)
    generated_state = GeneratedIsobaricReactor(gas)
    pressure = DEFAULT_PRESSURE
    updateReactor!(state, Y, DEFAULT_TEMPERATURE, pressure)
    wdot = zeros(gas.n_species)
    jac_forwarddiff = IsobaricJacobianWorkspace(gas)
    jac_analytical = AnalyticalIsobaricJacobianWorkspace(gas)
    jac_generated = GeneratedAnalyticalIsobaricJacobianWorkspace(gas)
    J_dense = zeros(gas.n_species + 1, gas.n_species + 1)
    J_sparse = isobaric_jacobian_prototype(gas)
    packed = PackedKinetics(gas.kinetics)
    wdot_packed = zeros(gas.n_species)
    kf_packed = zeros(gas.n_reactions)
    kr_packed = zeros(gas.n_reactions)

    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "CreateSolution",
        load_stats...,
    ))

    update_stats = sample!(samples=samples, evals=evals) do
        updateReactor!(state, Y, DEFAULT_TEMPERATURE, pressure)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "updateReactor!",
        update_stats...,
    ))

    wdot_stats = sample!(samples=samples, evals=evals) do
        wdot_func!(wdot, state.kf, state.kr, state.C, state.g_RT, gas.kinetics, state.T)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "wdot_func!",
        wdot_stats...,
    ))

    packed_wdot_stats = sample!(samples=samples, evals=evals) do
        wdot_func!(wdot_packed, kf_packed, kr_packed, state.C, state.g_RT, packed, state.T)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "wdot_func!_packed",
        packed_wdot_stats...,
    ))

    generated_wdot_stats = sample!(samples=samples, evals=evals) do
        wdot_func!(
            wdot,
            state.kf,
            state.kr,
            state.C,
            state.g_RT,
            gas.kinetics,
            jac_generated.kernel,
            state.T,
        )
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "wdot_func!_generated",
        generated_wdot_stats...,
    ))

    rhs_stats = sample!(samples=samples, evals=evals) do
        state(du, u, (pressure,), 0.0)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "isobaricReactor_rhs",
        rhs_stats...,
    ))

    generated_rhs_stats = sample!(samples=samples, evals=evals) do
        generated_state(du, u, (pressure,), 0.0)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "GeneratedIsobaricReactor_rhs",
        generated_rhs_stats...,
    ))

    jac_forwarddiff_stats = sample!(samples=samples, evals=evals) do
        jacobian!(J_dense, jac_forwarddiff, u, (pressure,), 0.0)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "jacobian_forwarddiff_dense",
        jac_forwarddiff_stats...,
    ))

    jac_analytical_stats = sample!(samples=samples, evals=evals) do
        jacobian!(J_dense, jac_analytical, u, (pressure,), 0.0)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "jacobian_analytical_dense",
        jac_analytical_stats...,
    ))

    jac_generated_stats = sample!(samples=samples, evals=evals) do
        jacobian!(J_dense, jac_generated, u, (pressure,), 0.0)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "jacobian_generated_dense",
        jac_generated_stats...,
    ))

    jac_analytical_sparse_stats = sample!(samples=samples, evals=evals) do
        jacobian!(J_sparse, jac_analytical, u, (pressure,), 0.0)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "jacobian_analytical_sparse",
        jac_analytical_sparse_stats...,
    ))

    jac_generated_sparse_stats = sample!(samples=samples, evals=evals) do
        jacobian!(J_sparse, jac_generated, u, (pressure,), 0.0)
    end
    push!(rows, (
        mechanism = name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        benchmark = "jacobian_generated_sparse",
        jac_generated_sparse_stats...,
    ))

    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "timestamp,julia_version,mechanism,species,reactions,benchmark,min_ns,median_ns,mean_ns,max_ns,alloc_bytes")
        timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        for row in rows
            println(io, join((
                timestamp,
                VERSION,
                row.mechanism,
                row.species,
                row.reactions,
                row.benchmark,
                round(Int, row.min_ns),
                round(Int, row.median_ns),
                round(Int, row.mean_ns),
                round(Int, row.max_ns),
                round(Int, row.alloc_bytes),
            ), ","))
        end
    end
end

function write_markdown(path, rows; samples, evals)
    open(path, "w") do io
        println(io, "# Arrhenius Baseline")
        println(io)
        println(io, "- Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        println(io, "- Julia: $(VERSION)")
        println(io, "- Samples: $samples")
        println(io, "- Evals per sample: $evals")
        println(io)
        println(io, "| Mechanism | Species | Reactions | Benchmark | Median time | Median alloc |")
        println(io, "|---|---:|---:|---|---:|---:|")
        for row in rows
            time_us = row.median_ns / 1_000
            alloc = row.alloc_bytes
            @printf(io, "| %s | %d | %d | `%s` | %.3f us | %d B |\n",
                row.mechanism,
                row.species,
                row.reactions,
                row.benchmark,
                time_us,
                round(Int, alloc),
            )
        end
    end
end

function main()
    samples = parse(Int, get(ENV, "ARRHENIUS_BASELINE_SAMPLES", "15"))
    evals = parse(Int, get(ENV, "ARRHENIUS_BASELINE_EVALS", "100"))

    rows = NamedTuple[]
    for (name, mechanism) in MECHANISMS
        println("Benchmarking $name")
        append!(rows, benchmark_case(name, mechanism; samples=samples, evals=evals))
    end

    mkpath(joinpath(@__DIR__, "results"))
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    csv_path = joinpath(@__DIR__, "results", "baseline_$stamp.csv")
    md_path = joinpath(@__DIR__, "results", "baseline_$stamp.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows; samples=samples, evals=evals)
    println("Wrote $csv_path")
    println("Wrote $md_path")
end

main()
