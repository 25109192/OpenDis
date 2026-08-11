#!/usr/bin/env python3
"""Run the fixed-seed impenetrable-cube mechanism-validation case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
MANIFEST_PATH = CASE_DIR / "case_manifest.json"


def read_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def load_network(input_path: Path, ExaDisNet, DisNetManager):
    errors = []
    for method_name in ("read_data", "read_paradis"):
        try:
            network = ExaDisNet()
            getattr(network, method_name)(str(input_path))
            managed = DisNetManager(network)
            print(
                f"Loaded {input_path.name} with {method_name}: "
                f"nodes={managed.num_nodes()}, segments={managed.num_segments()}"
            )
            return managed
        except Exception as error:  # retain both loader diagnostics
            errors.append(f"{method_name}: {error}")
    raise RuntimeError("Unable to read initial network: " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="New or empty result directory; existing results are never overwritten.",
    )
    parser.add_argument(
        "--input", type=Path, default=CASE_DIR / "fr_network_seed42.data",
        help="Initial ParaDiS/ExaDiS network file.",
    )
    args = parser.parse_args()

    manifest = read_manifest()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Initial network not found: {input_path}")
    actual_hash = sha256(input_path)
    expected_hash = manifest["input"]["sha256"]
    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            f"Initial-network SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    prepare_output(output_dir)

    inclusion = manifest["cell_and_inclusion"]
    os.environ["INCLUSION_A"] = format(inclusion["cube_edge_m"], ".17g")
    os.environ["INCLUSION_VOL_FRAC"] = format(
        inclusion["requested_volume_fraction"], ".17g"
    )
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES", "threads")

    python_dir = REPO_ROOT / "python"
    sys.path.insert(0, str(python_dir))
    try:
        import numpy as np
        import pyexadis
        from pyexadis_base import (
            CalForce,
            Collision,
            DisNetManager,
            ExaDisNet,
            MobilityLaw,
            Remesh,
            SimulateNetworkPerf,
            TimeIntegration,
            Topology,
        )
    except ImportError as error:
        raise ImportError(
            f"Cannot import the built ExaDiS Python module from {python_dir}: {error}"
        ) from error

    np.random.seed(manifest["random_seed"])
    core = manifest["material_and_core"]
    numerical = manifest["numerics"]
    loading = manifest["loading"]
    output = manifest["output"]
    state = {
        "crystal": "fcc",
        "burgmag": core["burgmag_m"],
        "mu": core["shear_modulus_Pa"],
        "nu": core["poisson_ratio"],
        "a": core["core_radius_b"],
        "maxseg": numerical["max_segment_length_b"],
        "minseg": numerical["min_segment_length_b"],
        "rtol": numerical["relative_tolerance_b"],
        "rann": numerical["annihilation_radius_b"],
        "nextdt": numerical["initial_time_step_s"],
        "maxdt": numerical["maximum_time_step_s"],
    }

    run_record = {
        "case_name": manifest["case_name"],
        "git_head": git_head(),
        "input_file": str(input_path),
        "input_sha256": actual_hash,
        "output_directory": str(output_dir),
        "random_seed": manifest["random_seed"],
        "environment": {
            "INCLUSION_A": os.environ["INCLUSION_A"],
            "INCLUSION_VOL_FRAC": os.environ["INCLUSION_VOL_FRAC"],
            "OMP_PROC_BIND": os.environ["OMP_PROC_BIND"],
            "OMP_PLACES": os.environ["OMP_PLACES"],
        },
        "status": "started",
        "start_unix_time": time.time(),
    }
    shutil.copy2(MANIFEST_PATH, output_dir / "case_manifest.json")
    with (output_dir / "run_record.json").open("w", encoding="utf-8") as stream:
        json.dump(run_record, stream, indent=2, sort_keys=True)
        stream.write("\n")

    initialized = False
    try:
        pyexadis.initialize()
        initialized = True
        network = load_network(input_path, ExaDisNet, DisNetManager)

        force = CalForce(
            force_mode=numerical["force_mode"], state=state,
            Ngrid=numerical["force_grid"][0], cell=network.cell,
        )
        mobility = MobilityLaw(
            mobility_law=numerical["mobility_law"], state=state,
            Medge=numerical["edge_mobility_Pa_inv_s_inv"],
            Mscrew=numerical["screw_mobility_Pa_inv_s_inv"],
            vmax=numerical["maximum_velocity_m_s"],
        )
        integrator = TimeIntegration(
            integrator=numerical["integrator"],
            rgroups=numerical["subcycling_radius_groups_b"],
            state=state, force=force, mobility=mobility,
        )
        collision = Collision(
            collision_mode=numerical["collision_mode"], state=state
        )
        topology = Topology(
            topology_mode=numerical["topology_mode"], state=state,
            force=force, mobility=mobility,
        )
        remesh = Remesh(
            remesh_rule=numerical["remesh_rule"], state=state,
            coarsen_mode=numerical["remesh_coarsen_mode"],
        )

        simulation = SimulateNetworkPerf(
            calforce=force, mobility=mobility, timeint=integrator,
            collision=collision, topology=topology, remesh=remesh,
            cross_slip=None, vis=None,
            loading_mode=loading["mode"], erate=loading["engineering_strain_rate_s_inv"],
            edir=np.asarray(loading["loading_direction"], dtype=float),
            max_strain=loading["maximum_engineering_strain"],
            burgmag=state["burgmag"], state=state,
            print_freq=output["print_frequency_steps"],
            write_freq=output["configuration_frequency_steps"],
            write_dir=str(output_dir), out_props=output["properties"],
        )

        print(json.dumps(run_record, indent=2, sort_keys=True))
        final_state = simulation.run(network, state)
        run_record.update({
            "status": "completed",
            "end_unix_time": time.time(),
            "termination_target": {
                "type": "maximum_engineering_strain",
                "value": loading["maximum_engineering_strain"],
            },
            "final_state": {
                key: float(final_state[key])
                for key in ("strain", "stress", "density", "time", "dt")
                if key in final_state
            },
        })
    except BaseException as error:
        run_record.update({
            "status": "failed",
            "end_unix_time": time.time(),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        })
        raise
    finally:
        with (output_dir / "run_record.json").open("w", encoding="utf-8") as stream:
            json.dump(run_record, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if initialized:
            pyexadis.finalize()

    print(f"Completed cube-validation case in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
