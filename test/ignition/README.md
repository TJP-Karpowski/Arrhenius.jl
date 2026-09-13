# Ignition reference data

`cantera_stagni_2025_profiles.csv` contains adiabatic, constant-pressure
ignition profiles generated with Cantera 3.2.0 from
`mechanism/nh3_stagni_2025.yaml`. The CSV records temperature and all species
mass fractions for these initially 1000 K mixtures:

- stoichiometric NH3/air at 20 bar
- a 50/50 molar NH3/H2 fuel blend in stoichiometric air at 10 bar

The time grids combine logarithmic samples over the full simulation with dense
linear samples across each ignition rise. The test checks the full temperature
and composition matrices, selected species profiles, successful integration,
and the time at which temperature has risen by 400 K.

Cantera is an external reference tool and is not an Arrhenius dependency. To
regenerate the committed CSV in an isolated environment:

```sh
python -m venv /tmp/arrhenius-stagni-cantera
/tmp/arrhenius-stagni-cantera/bin/pip install cantera==3.2.0
/tmp/arrhenius-stagni-cantera/bin/python \
    test/ignition/generate_cantera_stagni_2025_profiles.py
```

The generator writes the Cantera version, solver tolerances, and SHA-256 of the
mechanism into the CSV header. Review any reference diff before committing a
regenerated profile.
