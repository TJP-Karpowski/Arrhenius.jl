#!/usr/bin/env python3

import argparse
import gc
import statistics
import time

import cantera as ct


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark one Cantera constant-pressure ignition case."
    )
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--pressure", required=True, type=float)
    parser.add_argument("--composition", required=True)
    parser.add_argument("--t-end", required=True, type=float)
    parser.add_argument("--rtol", required=True, type=float)
    parser.add_argument("--atol", required=True, type=float)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--max-step", default=0.0, type=float)
    return parser.parse_args()


def solve_case(args):
    gas = ct.Solution(args.mechanism)
    gas.TPX = args.temperature, args.pressure, args.composition
    reactor = ct.IdealGasConstPressureReactor(gas, energy="on", clone=False)
    network = ct.ReactorNet([reactor])
    network.rtol = args.rtol
    network.atol = args.atol
    if args.max_step > 0.0:
        network.max_time_step = args.max_step
    network.advance(args.t_end)
    return reactor, network


def main():
    args = parse_args()

    warmup_reactor, warmup_network = solve_case(args)
    times_ns = []
    for _ in range(args.samples):
        gc.collect()
        start = time.perf_counter_ns()
        reactor, network = solve_case(args)
        times_ns.append(time.perf_counter_ns() - start)

    print("status=ok")
    print(f"cantera_version={ct.__version__}")
    phase = warmup_reactor.phase
    print(f"species={phase.n_species}")
    print(f"reactions={phase.n_reactions}")
    print(f"retcode=Success")
    print(f"final_temperature={warmup_reactor.T:.17g}")
    print("final_mass_fractions=" + ";".join(f"{y:.17g}" for y in phase.Y))
    print(f"steps={warmup_network.solver_stats.get('steps', 0)}")
    print(f"rhs_evals={warmup_network.solver_stats.get('rhs_evals', 0)}")
    print(f"jac_evals={warmup_network.solver_stats.get('jac_evals', 0)}")
    print(f"min_ns={min(times_ns)}")
    print(f"median_ns={statistics.median(times_ns)}")
    print(f"mean_ns={statistics.mean(times_ns)}")
    print(f"max_ns={max(times_ns)}")


if __name__ == "__main__":
    main()
