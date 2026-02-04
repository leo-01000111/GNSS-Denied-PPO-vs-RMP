"""
Environment generator for GNSS-denied robot navigation experiments.

Creates 10,000 scenario folders under envStorage/ with balanced splits:
 - training:   8,000 (all PPO)
 - testing:    1,500 (50% PPO, 50% MPC)
 - comparison:   500 (50% PPO, 50% MPC)

Each scenario directory is named PP_MMM_NNNNN where:
 - PP  : TR (training), TE (testing), CO (comparison)
 - MMM : PPO or MPC
 - NNNNN : zero-padded id within the (purpose, model) group

For every scenario we write:
 - <name>.json : reference trajectory as [[x, y], ...]
 - <name>.yaml : ir-sim compatible world description with robot + obstacles

This script is deterministic only with an explicit seed. Run with Python 3.12+.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "envStorage"

# Constants
WORLD_SIZE = 20.0
ROBOT_RADIUS = 0.2
OBSTACLE_RADIUS = 1.0
NUM_CIRCLE_OBS = 7
NUM_EDGE_OBS = 4
TRAIN_COUNT = 8000
TEST_COUNT = 1500
COMP_COUNT = 500
TEST_HALF = TEST_COUNT // 2  # 750
COMP_HALF = COMP_COUNT // 2  # 250
PATH_SAMPLES = 1000
START = (1.0, 1.0)
GOAL = (19.0, 19.0)


@dataclass(frozen=True)
class ScenarioSpec:
    purpose: str  # TR | TE | CO
    model: str    # PPO | MPC
    index: int    # 1-based within purpose+model

    @property
    def folder(self) -> Path:
        name = f"{self.purpose}_{self.model}_{self.index:05d}"
        return STORAGE / purpose_dir(self.purpose) / name

    @property
    def basename(self) -> str:
        return f"{self.purpose}_{self.model}_{self.index:05d}"


def purpose_dir(purpose: str) -> str:
    return {
        "TR": "training",
        "TE": "testing",
        "CO": "comparison",
    }[purpose]


def generate_specs() -> List[ScenarioSpec]:
    specs: List[ScenarioSpec] = []
    # Training: all PPO
    specs += [ScenarioSpec("TR", "PPO", i) for i in range(1, TRAIN_COUNT + 1)]
    # Testing: split PPO/MPC
    specs += [ScenarioSpec("TE", "PPO", i) for i in range(1, TEST_HALF + 1)]
    specs += [ScenarioSpec("TE", "MPC", i) for i in range(1, TEST_COUNT - TEST_HALF + 1)]
    # Comparison: split PPO/MPC
    specs += [ScenarioSpec("CO", "PPO", i) for i in range(1, COMP_HALF + 1)]
    specs += [ScenarioSpec("CO", "MPC", i) for i in range(1, COMP_COUNT - COMP_HALF + 1)]
    return specs


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def generate_path() -> List[Tuple[float, float]]:
    """Generate a smooth winding path using provided parametric form."""
    b = random.uniform(0.5, 1.25)
    c = random.uniform(0.0, 2 * math.pi)
    t_values = linspace(1.0, 19.0, PATH_SAMPLES)

    def s1(x: float) -> float:
        return 7 * math.sin(math.pi / 20 * x) - 7 * math.sin(math.pi / 20)

    def s2(x: float) -> float:
        return math.sin(b * x - c)

    def S(x: float) -> float:
        return s1(x) * s2(x)

    path: List[Tuple[float, float]] = []
    for t in t_values:
        xr = math.sqrt(2) * (t * math.cos(math.pi / 4) - S(t) * math.sin(math.pi / 4))
        yr = math.sqrt(2) * (t * math.sin(math.pi / 4) + S(t) * math.cos(math.pi / 4))
        path.append((xr, yr))

    # Clip to world bounds with small margin to avoid exact border contact.
    margin = 0.1
    path = [
        (
            max(margin, min(WORLD_SIZE - margin, x)),
            max(margin, min(WORLD_SIZE - margin, y)),
        )
        for x, y in path
    ]
    # Enforce exact start and goal points.
    path[0] = START
    path[-1] = GOAL
    return path


def distance_sq(p1: Sequence[float], p2: Sequence[float]) -> float:
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def path_min_distance(center: Tuple[float, float], path: List[Tuple[float, float]]) -> float:
    return min(math.sqrt(distance_sq(center, p)) for p in path)


def sample_obstacle_positions(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = []
    max_attempts = 5000
    # Clearance: obstacle radius plus a buffer to keep reference path usable.
    clearance = OBSTACLE_RADIUS + ROBOT_RADIUS * 1.2  # 1.24
    no_place_centers = [(0.0, 0.0), (WORLD_SIZE, WORLD_SIZE)]
    no_place_radius = 4.0 + OBSTACLE_RADIUS  # keep centers outside radius 5 of corners

    attempts = 0
    while len(positions) < NUM_CIRCLE_OBS and attempts < max_attempts:
        attempts += 1
        x = random.uniform(1.0, WORLD_SIZE - 1.0)
        y = random.uniform(1.0, WORLD_SIZE - 1.0)
        candidate = (x, y)

        # Corner exclusion zones
        if any(math.sqrt(distance_sq(candidate, c)) < no_place_radius for c in no_place_centers):
            continue

        # Avoid overlapping other obstacles (centers at least 2*radius apart)
        if any(math.sqrt(distance_sq(candidate, other)) < 2 * OBSTACLE_RADIUS for other in positions):
            continue

        # Avoid being too close to start/goal specifically
        if math.sqrt(distance_sq(candidate, START)) < 4.0 or math.sqrt(distance_sq(candidate, GOAL)) < 4.0:
            continue

        # Avoid path collision
        if path_min_distance(candidate, path) < clearance:
            continue

        positions.append(candidate)

    if len(positions) < NUM_CIRCLE_OBS:
        # Fallback: fill remaining with safe grid points away from path
        grid_points = [
            (x, y)
            for x in linspace(1.0, WORLD_SIZE - 1.0, 10)
            for y in linspace(1.0, WORLD_SIZE - 1.0, 10)
        ]
        for gp in grid_points:
            if len(positions) >= NUM_CIRCLE_OBS:
                break
            if any(math.sqrt(distance_sq(gp, other)) < 2 * OBSTACLE_RADIUS for other in positions):
                continue
            if path_min_distance(gp, path) < clearance:
                continue
            if any(math.sqrt(distance_sq(gp, c)) < no_place_radius for c in no_place_centers):
                continue
            positions.append(gp)

    return positions


def build_obstacle_section(circle_states: List[Tuple[float, float]]) -> List[dict]:
    obstacles: List[dict] = []

    # Circles
    obstacles.append(
        {
            "number": len(circle_states),
            "distribution": {"name": "manual"},
            "shape": [{"name": "circle", "radius": OBSTACLE_RADIUS}],
            "state": [[round(x, 3), round(y, 3)] for x, y in circle_states],
        }
    )

    # Border rectangles (lines)
    border_rect = {"name": "rectangle", "length": WORLD_SIZE, "width": 0.2}
    obstacles.append(
        {
            "number": NUM_EDGE_OBS,
            "distribution": {"name": "manual"},
            "shape": [border_rect for _ in range(NUM_EDGE_OBS)],
            "state": [
                [WORLD_SIZE / 2, 0.0, 0.0],  # bottom
                [WORLD_SIZE, WORLD_SIZE / 2, math.pi / 2],  # right
                [WORLD_SIZE / 2, WORLD_SIZE, 0.0],  # top
                [0.0, WORLD_SIZE / 2, math.pi / 2],  # left
            ],
        }
    )
    return obstacles


def build_world_dict(sample_time: float, circle_states: List[Tuple[float, float]]) -> dict:
    return {
        "world": {
            "height": WORLD_SIZE,
            "width": WORLD_SIZE,
            "step_time": 0.1,
            "sample_time": sample_time,
            "offset": [0, 0],
            "collision_mode": "stop",
            "control_mode": "auto",
        },
        "robot": [
            {
                "kinematics": {"name": "diff"},
                "shape": {"name": "circle", "radius": ROBOT_RADIUS},
                "state": [START[0], START[1], 0],
                "goal": [GOAL[0], GOAL[1], 0],
                "color": "g",
                "plot": {
                    "show_trajectory": True,
                    "show_goal": True,
                },
            }
        ],
        "obstacle": build_obstacle_section(circle_states),
    }


def write_yaml(path: Path, data: dict) -> None:
    import yaml  # Local import to avoid dependency unless running generator

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: Path, points: List[Tuple[float, float]]) -> None:
    path.write_text(json.dumps(points, indent=2), encoding="utf-8")


def ensure_dirs() -> None:
    for sub in ("training", "testing", "comparison"):
        (STORAGE / sub).mkdir(parents=True, exist_ok=True)


def main(seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)

    ensure_dirs()
    specs = generate_specs()

    for spec in specs:
        folder = spec.folder
        folder.mkdir(parents=True, exist_ok=True)
        basename = spec.basename

        path_points = generate_path()
        circles = sample_obstacle_positions(path_points)

        sample_time = 0.5 if spec.purpose == "TR" else 0.1
        world = build_world_dict(sample_time, circles)

        write_json(folder / f"{basename}.json", path_points)
        write_yaml(folder / f"{basename}.yaml", world)


if __name__ == "__main__":
    main()
