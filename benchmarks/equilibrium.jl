#!/usr/bin/env julia

using Arrhenius
using Dates
using Printf
using Statistics

const CASES = [
    (
        name = "h2o2_TP",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"),
        mode = :TP,
        temperature = 2500.0,
        pressure = Arrhenius.one_atm,
        composition = "H2:2,O2:1",
    ),
    (
        name = "h2_li_2004_TP",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2_li_2004.yaml"),
        mode = :TP,
        temperature = 2500.0,
        pressure = Arrhenius.one_atm,
        composition = "H2:2,O2:1,N2:3.76",
    ),
    (
        name = "gri30_methane_air_TP",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"),
        mode = :TP,
        temperature = 2500.0,
        pressure = Arrhenius.one_atm,
        composition = "CH4:1,O2:2,N2:7.52",
    ),
    (
        name = "gri30_methane_air_HP",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "gri30.yaml"),
        mode = :HP,
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        composition = "CH4:1,O2:2,N2:7.52",
    ),
    (
        name = "h2o2_HP",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2o2.yaml"),
        mode = :HP,
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        composition = "H2:2,O2:1",
    ),
    (
        name = "h2_li_2004_HP",
        mechanism = joinpath(@__DIR__, "..", "mechanism", "h2_li_2004.yaml"),
        mode = :HP,
        temperature = 1000.0,
        pressure = Arrhenius.one_atm,
        composition = "H2:2,O2:1,N2:3.76",
    ),
]

function sample!(f; samples::Int)
    result = f()
    times_ns = Vector{Int64}(undef, samples)
    allocations = Vector{Int64}(undef, samples)
    results = Vector{Any}(undef, samples)
    for i in 1:samples
        GC.gc(false)
        allocations[i] = @allocated begin
            results[i] = f()
        end
        GC.gc(false)
        t0 = time_ns()
        results[i] = f()
        times_ns[i] = time_ns() - t0
    end
    return (
        result = results[end] === nothing ? result : results[end],
        min_ns = minimum(times_ns),
        median_ns = median(times_ns),
        mean_ns = mean(times_ns),
        max_ns = maximum(times_ns),
        alloc_bytes = median(allocations),
    )
end

function benchmark_case(case; samples::Int)
    gas = CreateSolution(case.mechanism)
    function run_case_default()
        state = GasState(gas; T = case.temperature, p = case.pressure, X = case.composition)
        result = equilibrate!(state, case.mode)
        return (
            result = result,
            temperature = temperature(state),
            pressure = pressure(state),
            mole_sum = sum(mole_fractions(state)),
        )
    end
    workspace = EquilibriumWorkspace(gas)
    function run_case_workspace()
        state = GasState(gas; T = case.temperature, p = case.pressure, X = case.composition)
        result = equilibrate!(state, workspace, case.mode)
        return (
            result = result,
            temperature = temperature(state),
            pressure = pressure(state),
            mole_sum = sum(mole_fractions(state)),
        )
    end

    rows = NamedTuple[]
    for (variant, runner) in (("default", run_case_default), ("workspace", run_case_workspace))
        stats = sample!(runner; samples = samples)
        output = stats.result
        push!(rows, (
            case = case.name,
            variant = variant,
            mechanism = basename(case.mechanism),
            species = gas.n_species,
            reactions = gas.n_reactions,
            mode = case.mode,
            initial_temperature = case.temperature,
            final_temperature = output.temperature,
            iterations = output.result.iterations,
            residual_norm = output.result.residual_norm,
            mole_sum = output.mole_sum,
            min_ns = stats.min_ns,
            median_ns = stats.median_ns,
            mean_ns = stats.mean_ns,
            max_ns = stats.max_ns,
            alloc_bytes = stats.alloc_bytes,
        ))
    end
    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "timestamp,julia_version,case,variant,mechanism,species,reactions,mode,initial_temperature,final_temperature,iterations,residual_norm,mole_sum,min_ns,median_ns,mean_ns,max_ns,alloc_bytes")
        timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        for row in rows
            println(io, join((
                timestamp,
                VERSION,
                row.case,
                row.variant,
                row.mechanism,
                row.species,
                row.reactions,
                row.mode,
                row.initial_temperature,
                row.final_temperature,
                row.iterations,
                row.residual_norm,
                row.mole_sum,
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
        println(io, "# Arrhenius Equilibrium Benchmark")
        println(io)
        println(io, "- Timestamp: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        println(io, "- Julia: $(VERSION)")
        println(io, "- Samples: $samples")
        println(io)
        println(io, "| Case | Variant | Species | Reactions | Mode | Median solve | Alloc | Iterations | Residual | Final T |")
        println(io, "|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for row in rows
            @printf(io, "| `%s` | `%s` | %d | %d | `%s` | %.3f ms | %d B | %d | %.3e | %.3f K |\n",
                row.case,
                row.variant,
                row.species,
                row.reactions,
                row.mode,
                row.median_ns / 1.0e6,
                round(Int, row.alloc_bytes),
                row.iterations,
                row.residual_norm,
                row.final_temperature,
            )
        end
    end
end

function main()
    samples = parse(Int, get(ENV, "ARRHENIUS_EQUILIBRIUM_SAMPLES", "10"))
    rows = NamedTuple[]
    for case in CASES
        println("Benchmarking $(case.name)")
        append!(rows, benchmark_case(case; samples = samples))
    end

    mkpath(joinpath(@__DIR__, "results"))
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    csv_path = joinpath(@__DIR__, "results", "equilibrium_$stamp.csv")
    md_path = joinpath(@__DIR__, "results", "equilibrium_$stamp.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows; samples = samples)
    println("Wrote $csv_path")
    println("Wrote $md_path")
end

main()
