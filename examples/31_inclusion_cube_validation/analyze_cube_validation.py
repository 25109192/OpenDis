#!/usr/bin/env python3
"""Audit cube-interaction outputs without modifying raw simulation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Iterable


CASE_DIR = Path(__file__).resolve().parent
INCLUSION_NODE = 9
NODE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*(\d+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+(\d+)\s+(-?\d+)\s*$"
)
ARM_RE = re.compile(
    r"^\s*(\d+)\s*,\s*(\d+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)


def vsub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def vadd(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def vmul(a, value):
    return tuple(a[i] * value for i in range(3))


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm2(a):
    return dot(a, a)


def minimum_image(delta, box):
    return tuple(delta[i] - round(delta[i] / box[i]) * box[i] for i in range(3))


def parse_config(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    minimum = [0.0, 0.0, 0.0]
    maximum = [0.0, 0.0, 0.0]
    for label, target in (("minCoordinates", minimum), ("maxCoordinates", maximum)):
        for index, line in enumerate(lines):
            if line.strip().startswith(label):
                target[:] = [float(lines[index + offset].strip()) for offset in (1, 2, 3)]
                break

    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "nodalData =") + 1
    except StopIteration as error:
        raise ValueError(f"No nodalData section in {path}") from error

    nodes, arms = {}, []
    index = start
    while index < len(lines):
        match = NODE_RE.match(lines[index])
        if not match:
            index += 1
            continue
        tag = (int(match.group(1)), int(match.group(2)))
        nodes[tag] = {
            "position": tuple(float(match.group(i)) for i in (3, 4, 5)),
            "constraint": int(match.group(7)),
        }
        arm_count = int(match.group(6))
        index += 1
        for _ in range(arm_count):
            while index < len(lines) and not ARM_RE.match(lines[index]):
                index += 1
            if index >= len(lines):
                raise ValueError(f"Truncated arm data for node {tag} in {path}")
            arm = ARM_RE.match(lines[index])
            neighbor = (int(arm.group(1)), int(arm.group(2)))
            burgers = tuple(float(arm.group(i)) for i in (3, 4, 5))
            index += 1
            if index >= len(lines):
                raise ValueError(f"Missing plane line for node {tag} in {path}")
            plane_values = lines[index].split()
            if len(plane_values) < 3:
                raise ValueError(f"Invalid plane line for node {tag} in {path}")
            plane = tuple(float(value) for value in plane_values[:3])
            arms.append((tag, neighbor, burgers, plane))
            index += 1

    segments = {}
    for node, neighbor, burgers, plane in arms:
        key = tuple(sorted((node, neighbor)))
        if key not in segments or node == key[0]:
            segments[key] = {"nodes": key, "burgers": burgers, "plane": plane}
    return nodes, list(segments.values()), tuple(maximum[i] - minimum[i] for i in range(3))


def segment_crosses_strict_cube(p1, p2, center, half, box, epsilon):
    local1 = minimum_image(vsub(p1, center), box)
    local2 = vadd(local1, minimum_image(vsub(p2, p1), box))
    direction = vsub(local2, local1)
    enter, leave = 0.0, 1.0
    strict_half = half - epsilon
    for axis in range(3):
        low, high = -strict_half, strict_half
        if abs(direction[axis]) < 1.0e-14:
            if local1[axis] <= low or local1[axis] >= high:
                return False
        else:
            t1 = (low - local1[axis]) / direction[axis]
            t2 = (high - local1[axis]) / direction[axis]
            enter = max(enter, min(t1, t2))
            leave = min(leave, max(t1, t2))
            if enter >= leave:
                return False
    return leave > max(enter, 0.0) and enter < 1.0


def ordered_cycles(nodes, segments, nonzero_only=True):
    adjacency = {tag: [] for tag in nodes}
    for segment in segments:
        if nonzero_only and norm2(segment["burgers"]) < 1.0e-20:
            continue
        first, second = segment["nodes"]
        adjacency[first].append(second)
        adjacency[second].append(first)

    seen, cycles = set(), []
    for start in adjacency:
        if start in seen or not adjacency[start]:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node])
        seen.update(component)
        if not component or any(len(adjacency[node]) != 2 for node in component):
            continue
        cycle = [next(iter(component))]
        previous = None
        current = cycle[0]
        while True:
            candidates = [node for node in adjacency[current] if node != previous]
            following = candidates[0]
            if following == cycle[0]:
                break
            if following in cycle:
                cycle = []
                break
            cycle.append(following)
            previous, current = current, following
        if cycle and len(cycle) == len(component):
            cycles.append(cycle)
    return cycles


def point_in_polygon(point, polygon):
    inside = False
    for i, first in enumerate(polygon):
        second = polygon[(i + 1) % len(polygon)]
        if (first[1] > point[1]) != (second[1] > point[1]):
            x_cross = first[0] + (point[1] - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
            if point[0] < x_cross:
                inside = not inside
    return inside


def cycle_surrounds_center(cycle, nodes, center, half, box):
    points = [nodes[cycle[0]]["position"]]
    for previous, current in zip(cycle, cycle[1:]):
        points.append(vadd(points[-1], minimum_image(
            vsub(nodes[current]["position"], nodes[previous]["position"]), box
        )))
    center_image = vadd(points[0], minimum_image(vsub(center, points[0]), box))
    area_vector = (0.0, 0.0, 0.0)
    for first, second in zip(points, points[1:] + points[:1]):
        area_vector = vadd(area_vector, cross(first, second))
    if norm2(area_vector) < 1.0e-16:
        return False
    drop_axis = max(range(3), key=lambda axis: abs(area_vector[axis]))
    axes = [axis for axis in range(3) if axis != drop_axis]
    polygon = [(point[axes[0]], point[axes[1]]) for point in points]
    projected_center = (center_image[axes[0]], center_image[axes[1]])
    plane_distance_numerator = abs(dot(area_vector, vsub(center_image, points[0])))
    plane_intersects_cube = plane_distance_numerator <= half * sum(abs(value) for value in area_vector)
    return plane_intersects_cube and point_in_polygon(projected_center, polygon)


def natural_config_key(path: Path):
    match = re.search(r"config\.(\d+)\.data$", path.name)
    return int(match.group(1)) if match else -1


def audit_config(path, manifest):
    nodes, segments, box = parse_config(path)
    geometry = manifest["cell_and_inclusion"]
    center = tuple(geometry["expected_centers_b"][0])
    half = 0.5 * geometry["cube_edge_b"]
    epsilon = max(1.0e-6, half * 1.0e-8)
    strict_inside = []
    for tag, node in nodes.items():
        local = minimum_image(vsub(node["position"], center), box)
        if all(abs(value) < half - epsilon for value in local):
            strict_inside.append(tag)
    interior_nonzero_segments = []
    for segment in segments:
        if norm2(segment["burgers"]) < 1.0e-20:
            continue
        first, second = segment["nodes"]
        if segment_crosses_strict_cube(
            nodes[first]["position"], nodes[second]["position"],
            center, half, box, epsilon,
        ):
            interior_nonzero_segments.append([list(first), list(second)])
    cycles = ordered_cycles(nodes, segments)
    surrounding = [cycle for cycle in cycles
                   if cycle_surrounds_center(cycle, nodes, center, half, box)]
    return {
        "file": path.name,
        "step": natural_config_key(path),
        "nodes": len(nodes),
        "segments": len(segments),
        "inclusion_constrained_nodes": sum(
            node["constraint"] == INCLUSION_NODE for node in nodes.values()
        ),
        "strict_interior_nodes": [list(tag) for tag in strict_inside],
        "nonzero_segments_crossing_strict_interior": interior_nonzero_segments,
        "closed_nonzero_components": len(cycles),
        "orowan_like_closed_loops": len(surrounding),
    }


def read_properties(path: Path):
    if not path.is_file():
        return {"present": False, "rows": 0, "nonfinite": []}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = next((line.lstrip("# ").split() for line in lines if line.startswith("#")), [])
    rows, nonfinite = [], []
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        row = {}
        for index, value in enumerate(values):
            number = float(value)
            key = header[index] if index < len(header) else f"column_{index}"
            row[key] = number
            if not math.isfinite(number):
                nonfinite.append({"line": line_number, "field": key, "value": value})
        rows.append(row)
    summary = {"present": True, "rows": len(rows), "fields": header, "nonfinite": nonfinite}
    if rows:
        summary["first"] = rows[0]
        summary["last"] = rows[-1]
        for field in ("dt", "Density", "Nnodes", "Nsegs", "Stress"):
            values = [row[field] for row in rows if field in row and math.isfinite(row[field])]
            if values:
                summary[f"{field}_min"] = min(values)
                summary[f"{field}_max"] = max(values)
    return summary


def inspect_log(path: Path | None):
    if path is None or not path.is_file():
        return {"present": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "present": True,
        "bypass_candidate_messages": text.count("[INCDIAG] new inclusion bypass candidate"),
        "surface_node_insertion_messages": text.count("[INCDIAG] inserted"),
        "opposite_face_fatal_messages": text.count("crossed opposite faces"),
        "nan_or_inf_tokens": len(re.findall(r"(?i)(?<![A-Za-z])(?:nan|[-+]?inf)(?![A-Za-z])", text)),
        "error_lines": [line for line in text.splitlines() if "Error:" in line][:50],
    }


def write_figures(property_path: Path, audits, report_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return {"created": [], "warning": f"matplotlib unavailable: {error}"}

    created = []
    if property_path.is_file():
        lines = property_path.read_text(encoding="utf-8", errors="replace").splitlines()
        header = next((line.lstrip("# ").split() for line in lines if line.startswith("#")), [])
        rows = []
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            rows.append({header[i]: value for i, value in enumerate(values) if i < len(header)})
        strain = [row["Strain"] for row in rows if "Strain" in row]
        if strain and all("Stress" in row for row in rows):
            fig, axis = plt.subplots(figsize=(6.0, 4.0))
            axis.plot(strain, [row["Stress"] / 1.0e6 for row in rows], linewidth=1.5)
            axis.set(xlabel="Engineering strain (-)", ylabel="Axial stress (MPa)")
            axis.grid(True, alpha=0.3)
            fig.tight_layout()
            filename = report_dir / "stress_strain.png"
            fig.savefig(filename, dpi=200)
            plt.close(fig)
            created.append(filename.name)
        if strain and all("Density" in row for row in rows):
            fig, axis = plt.subplots(figsize=(6.0, 4.0))
            axis.plot(strain, [row["Density"] for row in rows], linewidth=1.5)
            axis.set(xlabel="Engineering strain (-)", ylabel=r"Dislocation density (m$^{-2}$)")
            axis.grid(True, alpha=0.3)
            fig.tight_layout()
            filename = report_dir / "density_strain.png"
            fig.savefig(filename, dpi=200)
            plt.close(fig)
            created.append(filename.name)

    if audits:
        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        steps = [audit["step"] for audit in audits]
        axis.step(
            steps, [audit["inclusion_constrained_nodes"] for audit in audits],
            where="post", label="inclusion-constrained nodes",
        )
        axis.step(
            steps, [audit["orowan_like_closed_loops"] for audit in audits],
            where="post", label="Orowan-like closed loops",
        )
        axis.set(xlabel="Simulation step", ylabel="Count")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        filename = report_dir / "interaction_diagnostics.png"
        fig.savefig(filename, dpi=200)
        plt.close(fig)
        created.append(filename.name)
    return {"created": created}


def write_csv(path: Path, audits: Iterable[dict]):
    fields = [
        "step", "file", "nodes", "segments", "inclusion_constrained_nodes",
        "strict_interior_node_count", "interior_nonzero_segment_count",
        "closed_nonzero_components", "orowan_like_closed_loops",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            writer.writerow({
                **{key: audit[key] for key in fields if key in audit},
                "strict_interior_node_count": len(audit["strict_interior_nodes"]),
                "interior_nonzero_segment_count": len(audit["nonzero_segments_crossing_strict_interior"]),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log", type=Path, help="Captured stdout/stderr from the simulation")
    parser.add_argument(
        "--report-dir", type=Path,
        help="New directory for analysis artifacts (default: OUTPUT_DIR/analysis).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    report_dir = (args.report_dir or output_dir / "analysis").resolve()
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty report directory: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "case_manifest.json"
    if not manifest_path.is_file():
        manifest_path = CASE_DIR / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs = sorted(output_dir.glob("config.*.data"), key=natural_config_key)
    audits = [audit_config(path, manifest) for path in configs]
    property_audit = read_properties(output_dir / "stress_strain_dens.dat")
    log_audit = inspect_log(args.log.resolve() if args.log else None)
    figure_audit = write_figures(
        output_dir / "stress_strain_dens.dat", audits, report_dir
    )

    penetration_frames = [audit["step"] for audit in audits
                          if audit["strict_interior_nodes"]
                          or audit["nonzero_segments_crossing_strict_interior"]]
    loop_counts = [audit["orowan_like_closed_loops"] for audit in audits]
    loop_increase_steps = [audits[i]["step"] for i in range(1, len(audits))
                           if loop_counts[i] > loop_counts[i - 1]]
    run_record_path = output_dir / "run_record.json"
    run_record = (json.loads(run_record_path.read_text(encoding="utf-8"))
                  if run_record_path.is_file() else None)
    report = {
        "case_name": manifest["case_name"],
        "configuration_files": len(configs),
        "configuration_audits": audits,
        "penetration_detected": bool(penetration_frames),
        "penetration_steps": penetration_frames,
        "orowan_like_loop_increase_steps": loop_increase_steps,
        "maximum_orowan_like_closed_loops": max(loop_counts, default=0),
        "properties": property_audit,
        "log": log_audit,
        "figures": figure_audit,
        "run_record": run_record,
        "automatic_checks_pass": bool(
            configs
            and property_audit.get("present")
            and property_audit.get("rows", 0) > 0
            and log_audit.get("present")
            and not penetration_frames
            and not property_audit.get("nonfinite")
            and not log_audit.get("opposite_face_fatal_messages", 0)
            and not log_audit.get("nan_or_inf_tokens", 0)
            and not log_audit.get("error_lines", [])
            and run_record is not None
            and run_record.get("status") == "completed"
        ),
        "interpretation_limit": (
            "The loop classifier is a geometry-based diagnostic. Key configurations "
            "must still be inspected visually before calling a structure an Orowan loop."
        ),
    }
    with (report_dir / "validation_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    write_csv(report_dir / "configuration_audit.csv", audits)
    print(json.dumps({
        "configuration_files": len(configs),
        "penetration_detected": report["penetration_detected"],
        "maximum_orowan_like_closed_loops": report["maximum_orowan_like_closed_loops"],
        "property_rows": property_audit["rows"],
        "automatic_checks_pass": report["automatic_checks_pass"],
        "report": str(report_dir / "validation_report.json"),
    }, indent=2))
    return 0 if report["automatic_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
