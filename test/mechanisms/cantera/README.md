This directory contains mechanism fixtures copied from the local Cantera 3.2.0
Python package data directory:

- `air.yaml`
- `gri30_highT.yaml`
- `SiF4_NH3_mec.yaml`
- `nDodecane_Reitz.yaml`

They are used to harden Arrhenius parser, kinetics, reactor RHS, and generated
Jacobian coverage against externally maintained Cantera-format mechanisms. The
tests intentionally use mechanisms that the current ideal-gas gas-phase parser
and reactor implementation support.

Cantera 3.2.0 is distributed under the BSD-3-Clause license. The copied license
notice is stored in `Cantera-License.txt`.

The Cantera `data/example_data` directory contains additional ammonia and
long-hydrocarbon mechanisms, but its README states that Cantera is not the
original author of those mechanisms and is not granting a license to them. Those
files are not vendored here unless their original licensing is audited separately.
