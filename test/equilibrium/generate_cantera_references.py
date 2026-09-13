#!/usr/bin/env python3
"""Generate Cantera equilibrium references for Arrhenius integration tests.

Run from the repository root with a Python environment that has Cantera
installed:

    python test/equilibrium/generate_cantera_references.py

The output is deterministic TOML and overwrites
test/equilibrium/cantera_equilibrium_references.toml.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import cantera as ct


PHIS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
PRESSURES_BAR = (0.1, 1.0, 10.0, 100.0)


def phi_pressure_cases(mode: str, temperature: float) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    fuels = (
        ("h2", "h2_li_2004.yaml", lambda phi: f"H2:{2 * phi:g},O2:1,N2:3.76"),
        ("ch4", "gri30.yaml", lambda phi: f"CH4:{phi:g},O2:2,N2:7.52"),
    )
    for fuel, mechanism, composition in fuels:
        for phi in PHIS:
            for pressure_bar in PRESSURES_BAR:
                cases.append(
                    {
                        "name": f"{fuel}_air_phi{phi:g}_{pressure_bar:g}bar_{temperature:g}K_{mode}",
                        "mode": mode,
                        "mechanism": mechanism,
                        "temperature": temperature,
                        "pressure": pressure_bar * 1.0e5,
                        "composition": composition(phi),
                        "fuel": fuel.upper(),
                        "phi": phi,
                        "pressure_bar": pressure_bar,
                    }
                )
    return cases


CASES = phi_pressure_cases("TP", 2500.0) + phi_pressure_cases("HP", 1000.0) + [
    {
        "name": "h2o2_stoich_2500K_1atm",
        "mode": "TP",
        "mechanism": "h2o2.yaml",
        "temperature": 2500.0,
        "pressure": 101325.0,
        "composition": "H2:2,O2:1",
    },
    {
        "name": "h2_li_stoich_air_2500K_1atm",
        "mode": "TP",
        "mechanism": "h2_li_2004.yaml",
        "temperature": 2500.0,
        "pressure": 101325.0,
        "composition": "H2:2,O2:1,N2:3.76",
    },
    {
        "name": "h2o2_stoich_1000K_1atm_HP",
        "mode": "HP",
        "mechanism": "h2o2.yaml",
        "temperature": 1000.0,
        "pressure": 101325.0,
        "composition": "H2:2,O2:1",
    },
    {
        "name": "h2_li_stoich_air_1000K_1atm_HP",
        "mode": "HP",
        "mechanism": "h2_li_2004.yaml",
        "temperature": 1000.0,
        "pressure": 101325.0,
        "composition": "H2:2,O2:1,N2:3.76",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mechanism_path(name: str) -> Path:
    path = repo_root() / "mechanism" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def format_float(value: float) -> str:
    return f"{value:.17e}"


def generate_case(case: dict[str, object]) -> dict[str, object]:
    gas = ct.Solution(str(mechanism_path(str(case["mechanism"]))))
    gas.TPX = case["temperature"], case["pressure"], case["composition"]
    gas.equilibrate(str(case["mode"]), rtol=1.0e-12)

    output = dict(case)
    output["equilibrium_temperature"] = float(gas.T)
    output["species"] = list(gas.species_names)
    output["mole_fractions"] = [float(value) for value in gas.X]
    return output


def write_references(cases: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "[metadata]",
        'source = "Cantera"',
        f'cantera_version = "{ct.__version__}"',
        f'generated = "{date.today().isoformat()}"',
        'description = "Ideal-gas equilibrium references generated with Cantera equilibrate(mode, rtol=1e-12)."',
        "",
    ]

    for case in cases:
        lines.extend(
            [
                "[[case]]",
                f'name = "{case["name"]}"',
            ]
        )
        if case["mode"] != "TP":
            lines.append(f'mode = "{case["mode"]}"')
        for key in ("fuel", "phi", "pressure_bar"):
            if key in case:
                value = case[key]
                lines.append(f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}")
        lines.extend(
            [
                f'mechanism = "{case["mechanism"]}"',
                f'temperature = {float(case["temperature"]):.1f}',
            ]
        )
        if case["mode"] != "TP":
            lines.append(
                f'equilibrium_temperature = {format_float(float(case["equilibrium_temperature"]))}'
            )
        lines.extend(
            [
                f'pressure = {float(case["pressure"]):.1f}',
                f'composition = "{case["composition"]}"',
                "species = ["
                + ", ".join(f'"{species}"' for species in case["species"])
                + "]",
                "mole_fractions = [",
            ]
        )
        lines.extend(f"    {format_float(value)}," for value in case["mole_fractions"])
        lines.extend(["]", ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_path = Path(__file__).resolve().parent / "cantera_equilibrium_references.toml"
    write_references([generate_case(case) for case in CASES], output_path)


if __name__ == "__main__":
    main()
