"""
Batch evaluation for MPC on TE_MPC_* worlds.

Runs headless simulations in succession (GNSS denial always active) and reports:
  1) Success rate (reached goal without collision)
  2) Mean cross-track error (distance to closest *unpassed* reference point)
  3) Mean cross-track error split by in/out of GNSS denial zone
  4) Average speed (mean |v|)
  5) Input variation (normalized total variation), mapped to [0, 1]

Run:
    py main/RMPC/statsAndGraphs.py
or:
    py -m RMPC.statsAndGraphs
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import irsim
import numpy as np

import sys

# Allow running as script from RMPC/ by adding project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from RMPC.MPCAgent import MPCAgent  # type: ignore
from utilities.DENIAL import apply_gnss_denial  # type: ignore

ROOT = Path(__file__).resolve().parent
ENV_ROOT = ROOT.parent / "envs" / "envStorage" / "testing"

GOAL_THRESH = 0.5
STEP_CAP = 2500


def _load_env(folder: Path):
    base = folder.name
    yaml_path = folder / f"{base}.yaml"
    traj_path = folder / f"{base}.json"
    zone_path = folder / f"{base}_D.json"
    if not yaml_path.exists() or not traj_path.exists():
        raise FileNotFoundError(f"Missing yaml/json in {folder}")
    ref = json.loads(traj_path.read_text(encoding="utf-8"))
    zone = json.loads(zone_path.read_text(encoding="utf-8")) if zone_path.exists() else None
    return yaml_path, ref, zone


def _prep_ref(ref_path: list):
    ref = np.asarray(ref_path, dtype=float)
    if ref.ndim != 2 or ref.shape[1] < 2:
        raise ValueError("Reference path must be list of [x, y] points")
    return ref[:, 0].copy(), ref[:, 1].copy()


def _nearest_ref_unpassed(x: float, y: float, ref_x: np.ndarray, ref_y: np.ndarray, last_idx: int):
    # Only search ahead to avoid snapping back to already-passed parts of the path.
    dx = ref_x[last_idx:] - x
    dy = ref_y[last_idx:] - y
    d2 = dx * dx + dy * dy
    off = int(np.argmin(d2))
    idx = last_idx + off
    dist = float(np.sqrt(d2[off]))
    return dist, idx


def _in_zone(x: float, y: float, zone: dict | None) -> bool:
    if zone is None:
        return False
    dx = x - float(zone["x"])
    dy = y - float(zone["y"])
    return (dx * dx + dy * dy) <= float(zone["r"]) ** 2


def _ask_minutes() -> float:
    try:
        s = input("For how many minutes to test? (e.g. 5): ").strip()
    except EOFError:
        return 5.0
    if not s:
        return 5.0
    try:
        m = float(s)
    except ValueError:
        return 5.0
    return max(0.1, m)


def _variation_to_unit(tv_norm: float) -> float:
    # Map [0, inf) -> [0, 1): 1-exp(-x) is smooth and saturates.
    return float(1.0 - np.exp(-tv_norm))


@dataclass
class WorldResult:
    name: str
    success: bool
    steps: int
    final_dist_goal: float
    mean_cte: float
    mean_cte_in: float | None
    mean_cte_out: float | None
    avg_speed: float
    tv_norm: float
    tv_unit: float


def run_world(folder: Path, *, seed: int | None = None) -> WorldResult:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    yaml_path, ref, zone = _load_env(folder)
    ref_x, ref_y = _prep_ref(ref)

    env = irsim.make(world_name=str(yaml_path), save_ani=False, display=False, full=False)
    agent = MPCAgent(env, ref)

    last_ref_idx = 0

    cte_sum = 0.0
    cte_n = 0
    cte_in_sum = 0.0
    cte_in_n = 0
    cte_out_sum = 0.0
    cte_out_n = 0

    v_abs_sum = 0.0
    v_abs_n = 0

    tv_sum = 0.0
    u_abs_sum = 0.0
    prev_u = None

    success = False
    final_dist = float("inf")

    for step in range(STEP_CAP):
        true_state = np.asarray(env.robot.state[0:3], dtype=float).reshape(-1)
        x, y = float(true_state[0]), float(true_state[1])

        # CTE w.r.t. closest unpassed reference point (monotonic index).
        cte, idx = _nearest_ref_unpassed(x, y, ref_x, ref_y, last_ref_idx)
        last_ref_idx = max(last_ref_idx, idx)
        cte_sum += cte
        cte_n += 1

        inside = _in_zone(x, y, zone)
        if inside:
            cte_in_sum += cte
            cte_in_n += 1
        else:
            cte_out_sum += cte
            cte_out_n += 1

        goal = np.asarray(env.robot.goal, dtype=float).reshape(-1)
        final_dist = float(np.hypot(float(goal[0]) - x, float(goal[1]) - y))
        if final_dist < GOAL_THRESH:
            success = True
            break

        # Controller sees noisy state only inside zone, but metrics use true state.
        used_state = true_state.copy()
        if inside:
            used_state = apply_gnss_denial(used_state)

        obs_list = env.get_obstacle_info_list()
        vel, info = agent.control_step(used_state, obs_list)

        flat = np.array(vel, dtype=float).ravel()
        u = np.zeros(2, dtype=float)
        if flat.size > 0:
            u[0] = flat[0]
        if flat.size > 1:
            u[1] = flat[1]

        # Speed metric uses the applied linear command magnitude.
        v_abs_sum += abs(float(u[0]))
        v_abs_n += 1

        # Total variation on inputs.
        if prev_u is not None:
            tv_sum += float(np.linalg.norm(u - prev_u))
        u_abs_sum += float(np.linalg.norm(u))
        prev_u = u

        # Step sim
        env.step(u.reshape(2, 1))

        collided = False
        try:
            collided = bool(getattr(env.robot, "collision_flag", False))
        except Exception:
            collided = False
        if collided or env.done():
            success = False
            break

    env.end(show_traj=False, show_trail=False, ending_time=0)

    mean_cte = cte_sum / max(1, cte_n)
    mean_cte_in = (cte_in_sum / cte_in_n) if cte_in_n > 0 else None
    mean_cte_out = (cte_out_sum / cte_out_n) if cte_out_n > 0 else None
    avg_speed = v_abs_sum / max(1, v_abs_n)
    tv_norm = tv_sum / max(1e-9, u_abs_sum)
    tv_unit = _variation_to_unit(tv_norm)

    return WorldResult(
        name=folder.name,
        success=bool(success),
        steps=step + 1,
        final_dist_goal=float(final_dist),
        mean_cte=float(mean_cte),
        mean_cte_in=float(mean_cte_in) if mean_cte_in is not None else None,
        mean_cte_out=float(mean_cte_out) if mean_cte_out is not None else None,
        avg_speed=float(avg_speed),
        tv_norm=float(tv_norm),
        tv_unit=float(tv_unit),
    )


def main():
    minutes = _ask_minutes()
    deadline = time.time() + 60.0 * minutes

    worlds = sorted(ENV_ROOT.glob("TE_MPC_*"))
    if not worlds:
        raise SystemExit(f"No TE_MPC_* worlds found in {ENV_ROOT}")

    print(f"Running MPC stats for up to {minutes:.2f} minutes on {len(worlds)} worlds (GNSS denial always ON).")

    results: list[WorldResult] = []
    for i, w in enumerate(worlds, start=1):
        if time.time() >= deadline:
            break
        try:
            res = run_world(w)
        except Exception as e:
            print(f"[{i}/{len(worlds)}] {w.name}: ERROR {type(e).__name__}: {e}")
            continue
        results.append(res)
        status = "OK" if res.success else "FAIL"
        print(
            f"[{i}/{len(worlds)}] {w.name}: {status} "
            f"steps={res.steps} dist={res.final_dist_goal:.2f} "
            f"cte={res.mean_cte:.3f} vavg={res.avg_speed:.3f} tv={res.tv_unit:.3f}"
        )

    if not results:
        print("No successful runs.")
        return

    success_rate = 100.0 * (sum(1 for r in results if r.success) / len(results))
    mean_cte = float(np.mean([r.mean_cte for r in results]))
    cte_in_vals = [r.mean_cte_in for r in results if r.mean_cte_in is not None]
    cte_out_vals = [r.mean_cte_out for r in results if r.mean_cte_out is not None]
    mean_cte_in = float(np.mean(cte_in_vals)) if cte_in_vals else float("nan")
    mean_cte_out = float(np.mean(cte_out_vals)) if cte_out_vals else float("nan")
    avg_speed = float(np.mean([r.avg_speed for r in results]))
    tv_unit = float(np.mean([r.tv_unit for r in results]))

    print("\n=== Aggregated Results ===")
    print(f"Worlds evaluated: {len(results)}")
    print(f"Success rate:     {success_rate:.1f}%")
    print(f"Mean CTE:         {mean_cte:.4f}")
    print(f"Mean CTE (in):    {mean_cte_in:.4f}")
    print(f"Mean CTE (out):   {mean_cte_out:.4f}")
    print(f"Average speed:    {avg_speed:.4f}")
    print(f"Input variation:  {tv_unit:.4f}  (0=constant, 1=spiky)")


if __name__ == "__main__":
    main()
