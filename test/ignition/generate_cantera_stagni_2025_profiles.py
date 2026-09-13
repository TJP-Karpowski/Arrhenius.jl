#!/usr/bin/env python3
"""Generate Cantera 3.2 ignition profiles for the Stagni 2025 mechanism.

Cantera is intentionally an external reference dependency. A typical run is:

    python -m venv /tmp/arrhenius-stagni-cantera
    /tmp/arrhenius-stagni-cantera/bin/pip install cantera==3.2.0
    /tmp/arrhenius-stagni-cantera/bin/python \
        test/ignition/generate_cantera_stagni_2025_profiles.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import cantera as ct


CASES = (
    {
        "name": "nh3_air_1000K_20bar",
        "temperature": 1000.0,
        "pressure": 2.0e6,
        "composition": "NH3:1,O2:0.75,N2:2.82",
        "t_end": 0.4,
        "focus_start": 0.12,
        "focus_end": 0.23,
    },
    {
        "name": "nh3_h2_air_1000K_10bar",
        "temperature": 1000.0,
        "pressure": 1.0e6,
        "composition": "NH3:0.5,H2:0.5,O2:0.625,N2:2.35",
        "t_end": 0.04,
        "focus_start": 0.007,
        "focus_end": 0.017,
    },
)

RTOL = 1.0e-10
ATOL = 1.0e-18


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repository = here.parents[2]
    parser = argparse.ArgumentParser(
        description="Generate external Cantera ignition profiles for nh3_stagni_2025.yaml."
    )
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=repository / "mechanism" / "nh3_stagni_2025.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here.with_name("cantera_stagni_2025_profiles.csv"),
    )
    return parser.parse_args()


def logarithmic_grid(first: float, last: float, count: int) -> list[float]:
    log_first = math.log(first)
    log_last = math.log(last)
    return [
        math.exp(log_first + i * (log_last - log_first) / (count - 1))
        for i in range(count)
    ]


def linear_grid(first: float, last: float, count: int) -> list[float]:
    return [first + i * (last - first) / (count - 1) for i in range(count)]


def profile_times(case: dict[str, object]) -> list[float]:
    times = [0.0]
    times.extend(logarithmic_grid(1.0e-9, float(case["t_end"]), 71))
    times.extend(
        linear_grid(
            float(case["focus_start"]),
            float(case["focus_end"]),
            61,
        )
    )
    return sorted(set(times))


def mechanism_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_case(mechanism: Path, case: dict[str, object]):
    gas = ct.Solution(str(mechanism))
    gas.TPX = (
        float(case["temperature"]),
        float(case["pressure"]),
        str(case["composition"]),
    )
    reactor = ct.IdealGasConstPressureReactor(gas, energy="on", clone=True)
    network = ct.ReactorNet([reactor])
    network.rtol = RTOL
    network.atol = ATOL

    rows = []
    for time in profile_times(case):
        network.advance(time)
        rows.append((str(case["name"]), time, reactor.T, *reactor.phase.Y))
    return gas.species_names, rows


def main() -> None:
    args = parse_args()
    mechanism = args.mechanism.resolve()
    output = args.output.resolve()

    all_rows = []
    species_names = None
    for case in CASES:
        case_species, rows = solve_case(mechanism, case)
        if species_names is None:
            species_names = case_species
        elif case_species != species_names:
            raise RuntimeError("Species ordering changed between reference cases")
        all_rows.extend(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        stream.write(f"# cantera_version={ct.__version__}\n")
        stream.write(f"# mechanism_sha256={mechanism_sha256(mechanism)}\n")
        stream.write(f"# rtol={RTOL:.17g}\n")
        stream.write(f"# atol={ATOL:.17g}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case", "time_s", "temperature_K", *species_names])
        for row in all_rows:
            writer.writerow(
                [row[0], *(format(float(value), ".17g") for value in row[1:])]
            )

    print(f"Cantera {ct.__version__}: wrote {len(all_rows)} rows to {output}")
    print(f"Mechanism SHA-256: {mechanism_sha256(mechanism)}")


if __name__ == "__main__":
    main()
