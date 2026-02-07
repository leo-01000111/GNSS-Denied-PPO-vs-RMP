"""
Tuner for MPC hyperparameters (ro1, ro2, receding, iter_num).
Runs headless simulations on random TE_MPC_* worlds, minimizes mean path deviation,
then prompts to overwrite mpc_config.yaml (with backup).
"""

from __future__ import annotations

import json
import random
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import yaml
import irsim

from MPCAgent import MPCAgent

ROOT = Path(__file__).resolve().parent
ENV_ROOT = ROOT.parent / "envs" / "envStorage" / "testing"
CONFIG_PATH = ROOT / "mpc_config.yaml"

# Search ranges
RANGES = {
    "ro1": (500, 8000),
    "ro2": (0.5, 5.0),
    "receding": (6, 15),
    "iter_num": (1, 4),
}

STEP_CAP = 1000
PATIENCE = 20
TIME_LIMIT = 1200  # seconds
SAMPLES_PER_ROUND = 3  # worlds per candidate


def load_env_paths() -> List[Path]:
    return list(ENV_ROOT.glob("TE_MPC_*"))


def load_paths(folder: Path):
    base = folder.name
    json_path = folder / f"{base}.json"
    yaml_path = folder / f"{base}.yaml"
    if not yaml_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing env files in {folder}")
    ref_path = json.loads(json_path.read_text(encoding="utf-8"))
    return yaml_path, ref_path


def to_array_path(path_iter) -> List[np.ndarray]:
    arr = []
    pts = list(path_iter)
    n = len(pts)
    for i, p in enumerate(pts):
        x, y = float(p[0]), float(p[1])
        if len(p) >= 3:
            th = float(p[2])
        else:
            nxt = pts[i + 1] if i + 1 < n else pts[i - 1] if i > 0 else p
            dx, dy = float(nxt[0]) - x, float(nxt[1]) - y
            th = float(np.arctan2(dy, dx)) if (dx or dy) else 0.0
        arr.append(np.array([[x], [y], [th]], dtype=float))
    return arr


def nearest_dist(point: np.ndarray, ref_path: List[np.ndarray]) -> float:
    px, py = float(point[0]), float(point[1])
    best = float("inf")
    for wp in ref_path:
        dx = float(wp[0]) - px
        dy = float(wp[1]) - py
        d2 = dx * dx + dy * dy
        if d2 < best:
            best = d2
    return best ** 0.5


def evaluate_candidate(params: Dict[str, Any], worlds: List[Path]) -> Tuple[float, Dict[str, Any]]:
    deviations = []
    for world in worlds:
        print(f"  ▶ running {world.name} ...", end="", flush=True)
        yaml_path, ref_json = load_paths(world)
        env = irsim.make(world_name=str(yaml_path), save_ani=False, display=False, full=False)
        ref_path = to_array_path(ref_json)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tf:
            cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg["controller"].update({k: params[k] for k in ("ro1", "ro2", "receding", "iter_num")})
            yaml.safe_dump(cfg, tf)
            temp_cfg_path = Path(tf.name)
        agent = MPCAgent(env, ref_path, config_path=temp_cfg_path)
        total_dev = 0.0
        steps = 0
        for _ in range(STEP_CAP):
            obs = env.get_obstacle_info_list()
            state = np.array(env.robot.state[0:3], dtype=float).reshape(-1)
            d = nearest_dist(state, ref_path)
            total_dev += d
            steps += 1
            vel, info = agent.control_step(state, obs)
            vec = np.zeros((2, 1), dtype=float)
            flat = np.array(vel, dtype=float).ravel()
            if flat.size > 0:
                vec[0, 0] = flat[0]
            if flat.size > 1:
                vec[1, 0] = flat[1]
            env.step(vec)
            if info.get("arrive"):
                break
        env.end(show_traj=False, show_trail=False, ending_time=0)
        temp_cfg_path.unlink(missing_ok=True)
        deviations.append(total_dev / max(1, steps))
        print(f" done (mean dev {deviations[-1]:.3f}, steps {steps})")
    avg_dev = float(np.mean(deviations)) if deviations else float("inf")
    return avg_dev, {"avg_dev": avg_dev, "worlds": len(worlds)}


def random_params() -> Dict[str, Any]:
    return {
        "ro1": random.uniform(*RANGES["ro1"]),
        "ro2": random.uniform(*RANGES["ro2"]),
        "receding": random.randint(*RANGES["receding"]),
        "iter_num": random.randint(*RANGES["iter_num"]),
    }


def main():
    random.seed()
    worlds = load_env_paths()
    if not worlds:
        print(f"No TE_MPC_* worlds found in {ENV_ROOT}")
        return

    best_score = float("inf")
    best_params = None
    no_improve = 0
    start_time = time.time()
    tried = 0
    print(f"Loaded {len(worlds)} test worlds. Time limit {TIME_LIMIT}s. Patience {PATIENCE}.")

    while time.time() - start_time < TIME_LIMIT:
        params = random_params()
        sample_worlds = random.sample(worlds, k=min(SAMPLES_PER_ROUND, len(worlds)))
        score, info = evaluate_candidate(params, sample_worlds)
        tried += 1
        elapsed = time.time() - start_time
        print(f"Trial {tried}: score={score:.3f} params={params} | elapsed {elapsed:.1f}s")
        if score + 1e-6 < best_score:
            best_score = score
            best_params = params
            no_improve = 0
            print(f"  new best (avg_dev={score:.3f})")
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stop: no improvement")
            break
    if best_params is None:
        print("No successful trials")
        return

    print("\nBest parameters found:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"Avg path deviation: {best_score:.3f}")

    try:
        resp = input("Apply these to mpc_config.yaml? [y/N]: ").strip().lower()
    except EOFError:
        resp = "n"
    if not resp.startswith("y"):
        print("Keeping existing config.")
        return

    backup_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")
    CONFIG_PATH.replace(backup_path)
    cfg = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    cfg["controller"].update({k: best_params[k] for k in ("ro1", "ro2", "receding", "iter_num")})
    CONFIG_PATH.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print(f"Applied. Backup saved at {backup_path}")


if __name__ == "__main__":
    main()
