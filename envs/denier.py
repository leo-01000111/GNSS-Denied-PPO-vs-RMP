"""
Generate GNSS denial zones for every environment in envStorage.

For each scenario folder, read the reference trajectory <name>.json, pick a
center point along the middle 30–70% of the path, choose a random radius in
[1, 5], and write <name>_D.json containing {"x": ..., "y": ..., "r": ...}.

Run: python -m envs.denier
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "envStorage"


def get_zone(path_file: Path) -> dict:
    points = json.loads(path_file.read_text(encoding="utf-8"))
    if not points:
        raise ValueError(f"Empty trajectory: {path_file}")
    start = int(len(points) * 0.3)
    end = max(start + 1, int(len(points) * 0.7))
    idx = random.randint(start, end - 1)
    x, y = points[idx]
    radius = random.uniform(1.0, 5.0)
    return {"x": float(x), "y": float(y), "r": float(radius)}


def process_folder(folder: Path) -> None:
    json_files = list(folder.glob("*.json"))
    base_files = [p for p in json_files if not p.name.endswith("_D.json")]
    for base in base_files:
        zone_file = base.with_name(base.stem + "_D.json")
        if zone_file.exists():
            continue
        zone = get_zone(base)
        zone_file.write_text(json.dumps(zone, indent=2), encoding="utf-8")


def main():
    if not STORAGE.exists():
        raise SystemExit(f"envStorage not found at {STORAGE}")
    for purpose_dir in STORAGE.iterdir():
        if not purpose_dir.is_dir():
            continue
        for scenario_dir in purpose_dir.iterdir():
            if scenario_dir.is_dir():
                process_folder(scenario_dir)


if __name__ == "__main__":
    main()
