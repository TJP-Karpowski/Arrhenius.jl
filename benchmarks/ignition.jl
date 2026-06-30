#!/usr/bin/env julia

using Pkg
Pkg.instantiate()

using Arrhenius
using Dates
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using Statistics

const CASES = [
    (
        name = "h2o2_hydrogen_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"),
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        t_end = 1.0e-3,
        composition = ("H2" => 2.0, "O2" => 1.0),
    ),
    (
        name = "h2_li_2004_hydrogen_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2_li_2004.yaml"),
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        t_end = 1.0e-3,
        composition = ("H2" => 2.0, "O2" => 1.0, "N2" => 3.76),
    ),
    (
        name = "gri30_methane_air",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"),
        temperature = 980.0,
        pressure = 15.0 * Arrhenius.one_atm,
        t_end = 2.0e-3,
        composition = ("CH4" => 0.5, "O2" => 1.0, "N2" => 3.76),
    ),
]

function mass_fractions(gas, composition)
    X = zeros(gas.n_species)
    for (species, amount) in composition
        idx = species_index(gas, species)
        idx === nothing && error("Species '$species' is not present in mechanism")
        X[idx] = amount
    end
    X ./= sum(X)
    return X2Y(gas, X, dot(X, gas.MW))
end

function sample!(f; samples::Int)
    times_ns = Vector{Int64}(undef, samples)
    allocations = Vector{Int64}(undef, samples)
    for i in 1:samples
        GC.gc(false)
        allocations[i] = @allocated f()
        t0 = time_ns()
        f()
        times_ns[i] = time_ns() - t0
    end
    return (
        min_ns = minimum(times_ns),
        median_ns = median(times_ns),
        mean_ns = mean(times_ns),
        max_ns = maximum(times_ns),
        alloc_bytes = median(allocations),
    )
end

function typed_reactor(gas, ::Type{T}) where {T}
    return Arrhenius.isobaricReactor(
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
        zero(T),
    )
end

function reactor_rhs(gas, u, p, t)
    state = typed_reactor(gas, eltype(u))
    du = similar(u)
    state(du, u, p, t)
    return du
end

function forwarddiff_jacobian!(J, gas, u, p, t)
    J .= ForwardDiff.jacobian(x -> reactor_rhs(gas, x, p, t), u)
    return nothing
end

function build_problem(case; with_jacobian = false)
    gas = CreateSolution(case.mechanism)
    state = isobaricReactor(gas)
    Y0 = mass_fractions(gas, case.composition)
    u0 = vcat(Y0, case.temperature)
    tspan = (0.0, case.t_end)
    if with_jacobian
        f = ODEFunction(state; jac = (J, u, p, t) -> forwarddiff_jacobian!(J, gas, u, p, t))
        return gas, ODEProblem(f, u0, tspan, (case.pressure,))
    end
    return gas, ODEProblem(state, u0, tspan, (case.pressure,))
end

function benchmark_problem(gas, prob, case, label; samples)

    # One warmup solve compiles the solver path and checks that the case is valid.
    solve(prob, Rosenbrock23(autodiff=false);
        reltol = 1.0e-6,
        abstol = 1.0e-9,
        save_everystep = false,
    )

    stats = sample!(samples=samples) do
        solve(prob, Rosenbrock23(autodiff=false);
            reltol = 1.0e-6,
            abstol = 1.0e-9,
            save_everystep = false,
        )
    end

    return (
        mechanism = case.name,
        species = gas.n_species,
        reactions = gas.n_reactions,
        solver = label,
        t_end = case.t_end,
        reltol = 1.0e-6,
        abstol = 1.0e-9,
        stats...,
    )
end

function benchmark_case(case; samples)
    rows = NamedTuple[]

    gas, prob = build_problem(case; with_jacobian = false)
    push!(rows, benchmark_problem(
        gas,
        prob,
        case,
        "Rosenbrock23(autodiff=false)",
        samples = samples,
    ))

    gas_jac, prob_jac = build_problem(case; with_jacobian = true)
    push!(rows, benchmark_problem(
        gas_jac,
        prob_jac,
        case,
        "Rosenbrock23(autodiff=false, jac=ForwardDiff)",
        samples = samples,
    ))

    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "timestamp,julia_version,mechanism,species,reactions,solver,t_end,reltol,abstol,min_ns,median_ns,mean_ns,max_ns,alloc_bytes")
        timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        for row in rows
            println(io, join((
                timestamp,
                VERSION,
                row.mechanism,
                row.species,
                row.reactions,
                row.solver,
                row.t_end,
                row.reltol,
                row.abstol,
                round(Int, row.min_ns),
                round(Int, row.median_ns),
                round(Int, row.mean_ns),
                round(Int, row.max_ns),
                round(Int, row.alloc_bytes),
            ), ","))
        end
    end
end

function write_markdown(path, rows; samples)
    open(path, "w") do io
        println(io, "# Arrhenius Ignition Baseline")
        println(io)
        println(io, "- Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        println(io, "- Julia: $(VERSION)")
        println(io, "- Samples: $samples")
        println(io, "- Solver variants: `Rosenbrock23(autodiff=false)` with and without a supplied ForwardDiff Jacobian")
        println(io, "- Tolerances: `reltol=1e-6`, `abstol=1e-9`")
        println(io, "- Save policy: `save_everystep=false`")
        println(io)
        println(io, "| Case | Solver | Species | Reactions | t_end | Median solve | Median alloc |")
        println(io, "|---|---|---:|---:|---:|---:|---:|")
        for row in rows
            time_ms = row.median_ns / 1_000_000
            @printf(io, "| %s | `%s` | %d | %d | %.3e | %.3f ms | %d B |\n",
                row.mechanism,
                row.solver,
                row.species,
                row.reactions,
                row.t_end,
                time_ms,
                round(Int, row.alloc_bytes),
            )
        end
    end
end

function main()
    samples = parse(Int, get(ENV, "ARRHENIUS_IGNITION_SAMPLES", "5"))

    rows = NamedTuple[]
    for case in CASES
        println("Benchmarking $(case.name)")
        append!(rows, benchmark_case(case; samples=samples))
    end

    mkpath(joinpath(@__DIR__, "results"))
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    csv_path = joinpath(@__DIR__, "results", "ignition_$stamp.csv")
    md_path = joinpath(@__DIR__, "results", "ignition_$stamp.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows; samples=samples)
    println("Wrote $csv_path")
    println("Wrote $md_path")
end

main()
