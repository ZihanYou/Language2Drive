#!/usr/bin/env python3
"""Render Language2Drive decision-route XML as an annotated SVG."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Layer C decision routes and candidate gates."
    )
    parser.add_argument("routes_xml", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.routes_xml.with_suffix(".svg")
    root = ET.parse(args.routes_xml).getroot()
    routes = []
    all_points = []

    for route in root.findall("route"):
        waypoints = [
            {
                "x": float(position.get("x", "0")),
                "y": float(position.get("y", "0")),
            }
            for position in route.findall("./waypoints/position")
        ]
        decision = route.find("decision_point")
        if not waypoints or decision is None:
            continue
        center = decision.find("center")
        candidates = decision.findall("candidate")
        routes.append(
            {
                "id": route.get("id", ""),
                "instruction": (route.findtext("instruction") or "").strip(),
                "expected_exit": decision.get("expected_exit", ""),
                "waypoints": waypoints,
                "center": {
                    "x": float(center.get("x", "0")),
                    "y": float(center.get("y", "0")),
                    "radius": float(center.get("radius_m", "4")),
                },
                "candidates": [
                    {
                        "id": candidate.get("id", ""),
                        "node_id": candidate.get("node_id", ""),
                        "x": float(candidate.get("x", "0")),
                        "y": float(candidate.get("y", "0")),
                        "radius": float(candidate.get("radius_m", "3")),
                    }
                    for candidate in candidates
                ],
            }
        )
        all_points.extend(waypoints)
        all_points.append(routes[-1]["center"])
        all_points.extend(routes[-1]["candidates"])

    if not routes:
        raise ValueError("No route with waypoints and decision_point found")

    min_x = min(point["x"] for point in all_points)
    max_x = max(point["x"] for point in all_points)
    min_y = min(point["y"] for point in all_points)
    max_y = max(point["y"] for point in all_points)
    width, height = 1100, 680
    margin_left, margin_right = 80, 300
    margin_top, margin_bottom = 100, 70
    scale = min(
        (width - margin_left - margin_right) / max(max_x - min_x, 1.0),
        (height - margin_top - margin_bottom) / max(max_y - min_y, 1.0),
    )

    def screen(point: dict[str, float]) -> tuple[float, float]:
        return (
            margin_left + (point["x"] - min_x) * scale,
            height - margin_bottom - (point["y"] - min_y) * scale,
        )

    colors = {"left": "#2563eb", "right": "#dc2626"}
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<defs>",
        (
            '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
            'refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" '
            'fill="context-stroke"/></marker>'
        ),
        "</defs>",
        "<style>",
        "  .title { font: 700 22px sans-serif; fill: #111827; }",
        "  .subtitle { font: 13px sans-serif; fill: #4b5563; }",
        "  .route { fill: none; stroke-width: 4; marker-end: url(#arrow); }",
        "  .shared { fill: none; stroke: #6b7280; stroke-width: 7; opacity: 0.28; }",
        "  .waypoint { fill: white; stroke-width: 2; }",
        "  .center { fill: #f59e0b; fill-opacity: 0.16; stroke: #d97706; stroke-width: 2; }",
        "  .gate { fill-opacity: 0.12; stroke-width: 2; stroke-dasharray: 5 3; }",
        "  .label { font: 600 12px sans-serif; fill: #1f2937; paint-order: stroke; stroke: white; stroke-width: 3px; }",
        "  .legend { font: 13px sans-serif; fill: #374151; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" class="title">Layer C · junction16 目标导航决策 Demo</text>',
        '<text x="30" y="60" class="subtitle">CARLA 世界坐标（米）· 指令仅给目标 · 灰色为共享入口</text>',
    ]

    shared_points = routes[0]["waypoints"][:3]
    shared_svg = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in map(screen, shared_points)
    )
    svg.append(f'<polyline class="shared" points="{shared_svg}"/>')

    for route in routes:
        expected = route["expected_exit"]
        color = colors.get(expected, "#2563eb")
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in map(screen, route["waypoints"])
        )
        svg.append(
            f'<polyline class="route" stroke="{color}" points="{points}">'
            f'<title>{escape(route["instruction"])}</title></polyline>'
        )
        for waypoint in route["waypoints"]:
            x, y = screen(waypoint)
            svg.append(
                f'<circle class="waypoint" stroke="{color}" '
                f'cx="{x:.2f}" cy="{y:.2f}" r="3"/>'
            )
        target_x, target_y = screen(route["waypoints"][-1])
        svg.append(
            f'<text x="{target_x + 8:.2f}" y="{target_y - 8:.2f}" '
            f'class="label">{expected} · {escape(route["instruction"])}</text>'
        )

    center = routes[0]["center"]
    center_x, center_y = screen(center)
    svg.append(
        f'<circle class="center" cx="{center_x:.2f}" cy="{center_y:.2f}" '
        f'r="{center["radius"] * scale:.2f}"/>'
    )
    svg.append(
        f'<text x="{center_x + 8:.2f}" y="{center_y + 18:.2f}" '
        'class="label">decision center · N013</text>'
    )

    for candidate in routes[0]["candidates"]:
        color = colors.get(candidate["id"], "#6b7280")
        x, y = screen(candidate)
        svg.append(
            f'<circle class="gate" stroke="{color}" fill="{color}" '
            f'cx="{x:.2f}" cy="{y:.2f}" '
            f'r="{candidate["radius"] * scale:.2f}"/>'
        )
        svg.append(
            f'<text x="{x + 8:.2f}" y="{y + 18:.2f}" class="label">'
            f'{escape(candidate["id"])} gate · {escape(candidate["node_id"])}</text>'
        )

    legend_x = width - margin_right + 45
    svg.extend(
        [
            f'<g transform="translate({legend_x}, 125)">',
            '<text x="0" y="0" class="label">路线与目标</text>',
        ]
    )
    for index, route in enumerate(routes):
        y = 34 + index * 84
        color = colors.get(route["expected_exit"], "#2563eb")
        svg.extend(
            [
                f'<line x1="0" y1="{y}" x2="34" y2="{y}" stroke="{color}" stroke-width="4"/>',
                f'<text x="44" y="{y + 4}" class="legend">{escape(route["expected_exit"])}</text>',
                f'<text x="0" y="{y + 28}" class="legend">{escape(route["instruction"])}</text>',
            ]
        )
    svg.extend(["</g>", "</svg>"])
    output_path.write_text("\n".join(svg), encoding="utf-8")
    print(f"SVG: {output_path.resolve()}")


if __name__ == "__main__":
    main()
