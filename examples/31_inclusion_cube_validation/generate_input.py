#!/usr/bin/env python3
"""Regenerate the fixed-seed Frank-Read source used by this validation case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]


def load_manifest() -> dict:
    with (CASE_DIR / "case_manifest.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def cube_centers(box: float, count: int):
    import numpy as np
    spacing = box / count
    return np.asarray([
        [(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing]
        for i in range(count) for j in range(count) for k in range(count)
    ])


def point_in_cube(point, centers, half: float, safety: float = 0.0) -> bool:
    return any(all(abs(point[i] - center[i]) < half + safety for i in range(3))
               for center in centers)


def segment_hits_cube(p1, p2, centers, half: float) -> bool:
    direction = p2 - p1
    for center in centers:
        enter, leave = 0.0, 1.0
        for axis in range(3):
            low, high = center[axis] - half, center[axis] + half
            if abs(direction[axis]) < 1.0e-12:
                if p1[axis] < low or p1[axis] > high:
                    break
            else:
                t1 = (low - p1[axis]) / direction[axis]
                t2 = (high - p1[axis]) / direction[axis]
                enter = max(enter, min(t1, t2))
                leave = min(leave, max(t1, t2))
            if enter >= leave:
                break
        else:
            return True
    return False


def valid_source(p1, p2, centers, half, n_mid, maxseg, minseg) -> bool:
    if point_in_cube(p1, centers, half, minseg):
        return False
    if point_in_cube(p2, centers, half, minseg):
        return False
    if segment_hits_cube(p1, p2, centers, half):
        return False
    for index in range(1, n_mid + 1):
        fraction = index / (n_mid + 1)
        point = (1.0 - fraction) * p1 + fraction * p2
        if point_in_cube(point, centers, half, maxseg):
            return False
    return True


def fcc_slip_systems():
    import numpy as np
    burgers = np.asarray([
        [0., 1., -1.], [1., 0., -1.], [1., -1., 0.],
        [0., 1., -1.], [1., 0., 1.], [1., 1., 0.],
        [0., 1., 1.], [1., 0., -1.], [1., 1., 0.],
        [0., 1., 1.], [1., 0., 1.], [1., -1., 0.],
    ])
    planes = np.asarray([
        [1., 1., 1.], [1., 1., 1.], [1., 1., 1.],
        [-1., 1., 1.], [-1., 1., 1.], [-1., 1., 1.],
        [1., -1., 1.], [1., -1., 1.], [1., -1., 1.],
        [1., 1., -1.], [1., 1., -1.], [1., 1., -1.],
    ])
    return [(b / np.linalg.norm(b), n / np.linalg.norm(n))
            for b, n in zip(burgers, planes)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=CASE_DIR / "fr_network_seed42.generated.data",
        help="Output file. Existing files are not overwritten.",
    )
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    manifest = load_manifest()
    sys.path.insert(0, str(REPO_ROOT / "python"))
    try:
        import numpy as np
        import pyexadis
        from pyexadis_base import DisNetManager, ExaDisNet
        from pyexadis_utils import insert_frank_read_src, write_data
    except ImportError as error:
        raise ImportError("Build the ExaDiS Python module before regenerating input") from error

    geometry = manifest["cell_and_inclusion"]
    numerical = manifest["numerics"]
    seed = manifest["random_seed"]
    box = geometry["cell_edge_b"]
    edge = geometry["cube_edge_b"]
    target_vf = geometry["requested_volume_fraction"]
    count = max(1, int(round((target_vf * box**3 / edge**3) ** (1.0 / 3.0))))
    centers = cube_centers(box, count)
    half = 0.5 * edge
    n_mid = 8

    np.random.seed(seed)
    initialized = False
    try:
        pyexadis.initialize()
        initialized = True
        cell = pyexadis.Cell(box)
        nodes, segments = [], []
        for attempt in range(1, 100001):
            system_index = np.random.randint(12)
            burgers, plane = fcc_slip_systems()[system_index]
            length = np.random.uniform(1500.0, 2500.0)
            theta = np.random.uniform(0.0, 90.0)
            center = np.random.uniform(0.0, box, size=3)
            b_hat = burgers / np.linalg.norm(burgers)
            transverse = np.cross(plane, b_hat)
            if np.linalg.norm(transverse) < 1.0e-10:
                continue
            transverse /= np.linalg.norm(transverse)
            direction = (np.cos(np.radians(theta)) * b_hat
                         + np.sin(np.radians(theta)) * transverse)
            p1 = (center - 0.5 * length * direction) % box
            p2 = center + 0.5 * length * direction
            for axis in range(3):
                while p2[axis] - p1[axis] > 0.5 * box:
                    p2[axis] -= box
                while p2[axis] - p1[axis] < -0.5 * box:
                    p2[axis] += box
            plane_hits = any(abs(np.dot(cube - center, plane)) < half
                             for cube in centers)
            if not plane_hits or not valid_source(
                p1, p2, centers, half, n_mid,
                numerical["max_segment_length_b"],
                numerical["min_segment_length_b"],
            ):
                continue
            nodes, segments = insert_frank_read_src(
                cell=cell, nodes=nodes, segs=segments,
                burg=burgers, plane=plane, length=length,
                center=center, theta=theta, numnodes=n_mid + 2,
            )
            print(
                f"accepted attempt={attempt}, slip_system={system_index}, "
                f"length={length:.6f} b, theta={theta:.6f} deg"
            )
            break
        else:
            raise RuntimeError("No valid Frank-Read source found in 100000 attempts")

        network = DisNetManager(ExaDisNet(cell, nodes, segments))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_data(network, str(output_path))
        # The upstream ParaDiS writer emits a trailing blank after
        # "domainDecomposition =". Normalize line endings and trailing spaces
        # so regenerated files pass Git whitespace checks deterministically.
        normalized_lines = [
            line.rstrip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(normalized_lines) + "\n")
    finally:
        if initialized:
            pyexadis.finalize()

    actual = hashlib.sha256(output_path.read_bytes()).hexdigest()
    expected = manifest["input"]["sha256"]
    print(f"wrote {output_path}")
    print(f"SHA256={actual}")
    if actual != expected:
        print(
            "WARNING: bytewise hash differs from the frozen input. "
            "Compare network semantics and library versions before replacing it."
        )
        return 2
    print("Frozen-input SHA256 reproduced exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
