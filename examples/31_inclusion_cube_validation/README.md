# Impenetrable cubic-inclusion validation case

This case checks the geometry and algorithms of the existing axis-aligned,
impenetrable cubic-inclusion model. It is not a gamma-prime raft model and is
not calibrated for quantitative nickel-superalloy or creep-life prediction.

## Frozen case

All case parameters and units are defined once in `case_manifest.json`.
The initial network is `fr_network_seed42.data`; it contains one pinned FCC
Frank-Read source generated with seed 42. The runner verifies its SHA256 before
starting and refuses to overwrite a non-empty result directory.

The periodic cell is a 2.5 micrometre cube. `INCLUSION_A=8e-7` m and
`INCLUSION_VOL_FRAC=0.034` produce one centered 0.8 micrometre cube. Because the
integer array contains one inclusion, the realized volume fraction is 0.032768,
not exactly the requested 0.034. The net channel is 1.7 micrometres in each
direction. Lengths inside ExaDiS are normalized by `burgmag=2.55e-10` m.

Thermal cross-slip and climb are intentionally disabled. No lattice-misfit
stress, elastic contrast, gamma-prime cutting, or plate/raft geometry is present.

## Build and unit test

Use the repository's existing platform-specific compiler, CUDA, FFT, Python and
Kokkos settings. A generic CPU configuration from the repository root is:

```bash
./configure.sh -DEXADIS_BUILD_TESTS=On -DEXADIS_PYTHON_BINDING=On
cmake --build build -j8
./build/tests/unit_tests/test_inclusion
```

If the remote machine already has a validated build configuration, add
`-DEXADIS_BUILD_TESTS=On` to that configuration instead of replacing its
backend/architecture options. Record the complete configure and build commands.

## Run

Choose a new result directory and a log path outside that directory. From the
repository root:

```bash
RESULT_DIR=/absolute/path/to/cube_validation_seed42_run1
RUN_LOG=/absolute/path/to/cube_validation_seed42_run1.log
python examples/31_inclusion_cube_validation/run_cube_validation.py \
  --output-dir "$RESULT_DIR" 2>&1 | tee "$RUN_LOG"
test "${PIPESTATUS[0]}" -eq 0
```

The `PIPESTATUS` check is required: a successful `tee` must not hide a failed
simulation. Do not reuse a non-empty result directory.

The committed input can be regenerated, without overwriting it, by running:

```bash
python examples/31_inclusion_cube_validation/generate_input.py \
  --output /absolute/path/to/fr_network_seed42.generated.data
```

The generator returns status 2 if the bytewise SHA256 differs from the frozen
file; that requires a semantic comparison and dependency-version check before
any replacement is considered.

## Analyze

The analyzer only reads raw results and writes to a new/empty analysis directory:

```bash
python examples/31_inclusion_cube_validation/analyze_cube_validation.py \
  --output-dir "$RESULT_DIR" \
  --log "$RUN_LOG"
```

It writes:

- `analysis/validation_report.json`: geometry, penetration, numerical-output,
  run-status and log checks;
- `analysis/configuration_audit.csv`: per-configuration node, segment,
  constrained-node and closed-loop diagnostics;
- `analysis/stress_strain.png`, `density_strain.png`, and
  `interaction_diagnostics.png` when Matplotlib is available.

The automatic audit fails if it finds a node or non-zero-Burgers segment in the
strict cube interior, a NaN/Inf in the property table, an error/fatal diagnostic,
or an incomplete `run_record.json`. A zero-Burgers surface chord is logged only
as a bypass candidate. A geometry-based `orowan_like_closed_loops` result still
requires visual inspection of the corresponding configuration before it can be
reported as an Orowan loop.

## Evidence required for initial validation

Successful exit alone is insufficient. Preserve and return:

- remote branch and HEAD; compiler, CUDA, Python, Kokkos and FFT versions;
- complete configure/build/test commands and logs;
- complete run command, `case_manifest.json`, `run_record.json`, and run log;
- `stress_strain_dens.dat`, every `config.*.data`, and the analysis outputs;
- runtime, time-step history and termination reason;
- visualizations of the initial state, first contact, surface/edge motion,
  maximum bow-out, bypass, and any residual closed loop.

The model is only "initially validated" after the unit test passes, the complete
case terminates normally and reproducibly, no penetration/topology blocker is
observed, and at least one interaction sequence is physically interpretable.

## Known algorithm boundary

The present surface router handles entry and exit through adjacent cube faces by
inserting their common-edge node. A segment chord through opposite faces would
require two edge nodes; the code now stops with a diagnostic instead of silently
leaving a segment through the inclusion. Such a failure is evidence that the
segment length/time step or routing algorithm must be revisited, not a condition
to suppress.
