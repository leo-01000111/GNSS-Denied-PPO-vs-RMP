"""
Single rendered rollout of MPC on a random MPC testing environment.

Run:
    python -m RMPC.RenderedRollout

Steps:
 - Select a random TE_MPC_* scenario from envs/envStorage/testing.
 - Ask user whether to activate GNSS denial noise.
 - Load env via irsim.make(config=<yaml>, display=True, save_ani=False, full=False).
 - Draw reference path and denial zone.
 - Run MPC until goal, crash, or step cap; render frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import irsim
import numpy as np

# Allow running as script from RMPC/ by adding project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from RMPC.MPCAgent import MPCAgent  # type: ignore
from utilities.DENIAL import apply_gnss_denial  # type: ignore

ROOT = Path(__file__).resolve().parent
ENV_ROOT = ROOT.parent / "envs" / "envStorage" / "testing"
LOG_ROOT = ROOT / "logs"

STEP_CAP = 8000
GOAL_THRESH = 0.5


def pick_world(name: str | None = None):
    candidates = list(ENV_ROOT.glob("TE_MPC_*"))
    if not candidates:
        raise SystemExit(f"No testing MPC scenarios found in {ENV_ROOT}")
    if name is None:
        return random.choice(candidates)
    for c in candidates:
        if c.name == name:
            return c
    raise SystemExit(f"Scenario '{name}' not found under {ENV_ROOT}")


def load_paths(folder: Path):
    base = folder.name
    json_path = folder / f"{base}.json"
    zone_path = folder / f"{base}_D.json"
    yaml_path = folder / f"{base}.yaml"
    if not yaml_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing env files in {folder}")
    ref_path = json.loads(json_path.read_text(encoding="utf-8"))
    zone = None
    if zone_path.exists():
        zone = json.loads(zone_path.read_text(encoding="utf-8"))
    return yaml_path, ref_path, zone


def ask_gnss_denial() -> bool:
    try:
        answer = input("Activate GNSS denial noise? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer.startswith("y")


def draw_zone(env, zone, active: bool):
    if not zone:
        return
    color = "#c5d5f5" if not active else "#f0a3a3"
    drawn = False
    if hasattr(env, "draw_circle"):
        try:
            env.draw_circle(zone["x"], zone["y"], zone["r"], color=color, alpha=0.35)
            drawn = True
        except Exception:
            drawn = False
    if not drawn:
        try:
            from matplotlib.patches import Circle

            ax = env._env_plot.ax  # type: ignore[attr-defined]
            circ = Circle((zone["x"], zone["y"]), zone["r"], color=color, alpha=0.35, zorder=0)
            ax.add_patch(circ)
            ax.figure.canvas.draw_idle()
        except Exception:
            pass


def to_array_traj(traj):
    """Convert list of points to list of (2,1) numpy arrays for irsim plotting."""
    arr = []
    for p in traj:
        if isinstance(p, np.ndarray):
            if p.shape == (2, 1):
                arr.append(p)
            elif p.shape == (2,):
                arr.append(p.reshape(2, 1))
            elif p.shape == (1, 2):
                arr.append(p.T)
            else:
                arr.append(np.array(p)[0:2].reshape(2, 1))
        else:
            arr.append(np.array([[p[0]], [p[1]]], dtype=float))
    return arr


def add_zone_legend(env, active: bool):
    try:
        ax = env._env_plot.ax  # type: ignore[attr-defined]
        import matplotlib.patches as mpatches

        label = "GNSS denial (active)" if active else "GNSS denial (inactive)"
        color = "#f0a3a3" if active else "#c5d5f5"
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, mpatches.Patch(color=color, alpha=0.35, label=label))
        labels.insert(0, label)
        ax.legend(handles, labels, loc="upper left")
    except Exception:
        pass


def _prep_ref_arrays(ref_path: list):
    """Prepare vectorized reference arrays for quick distance/index queries."""
    ref = np.asarray(ref_path, dtype=float)
    if ref.ndim != 2 or ref.shape[1] < 2:
        raise ValueError("Reference path must be a list of [x, y] points.")
    return ref[:, 0].copy(), ref[:, 1].copy()


def _nearest_ref(x: float, y: float, ref_x: np.ndarray, ref_y: np.ndarray):
    dx = ref_x - x
    dy = ref_y - y
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    return float(np.sqrt(d2[idx])), idx


class RolloutLogger:
    HEADER = [
        "step",
        "t",
        "x_true",
        "y_true",
        "theta_true",
        "x_used",
        "y_used",
        "theta_used",
        "dist_goal",
        "cte",  # cross-track error: nearest ref distance
        "ref_idx",
        "mpc_cur_index",
        "ref_speed",
        "v_cmd",
        "w_cmd",
        "v_clip",
        "w_clip",
        "sat_v",
        "sat_w",
        "in_denial_zone",
        "noise_applied",
        "collision_flag",
        "min_circle_clearance",
        "num_circle_obs",
        "mode",
        "arrive",
        "done",
        "iter_time",
        "resi_pri",
        "resi_dual",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(self.HEADER)

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


def _make_logger(enabled: bool, scenario_name: str, deny: bool) -> tuple[RolloutLogger | None, Path | None]:
    if not enabled:
        return None, None
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = LOG_ROOT / f"{scenario_name}_{'deny' if deny else 'no_deny'}_{ts}.csv"
    return RolloutLogger(out), out


def _write_meta(enabled: bool, csv_path: Path | None, scenario: Path, deny: bool, config_path: Path):
    if not enabled or csv_path is None:
        return
    meta = {
        "scenario": scenario.name,
        "yaml": str((scenario / f"{scenario.name}.yaml").resolve()),
        "deny": bool(deny),
        "goal_thresh": GOAL_THRESH,
        "step_cap": STEP_CAP,
        "config_path": str(config_path.resolve()),
        "config": None,
    }
    try:
        meta["config"] = json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))))
    except Exception:
        meta["config"] = None
    meta_path = csv_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Disable rendering for faster debugging.")
    p.add_argument("--log", action="store_true", help="Write CSV log to RMPC/logs/")
    p.add_argument("--scenario", type=str, default=None, help="Scenario folder name, e.g. TE_MPC_00001.")
    p.add_argument("--deny", choices=["ask", "y", "n"], default="ask", help="GNSS denial switch.")
    p.add_argument("--step-cap", type=int, default=None, help="Override STEP_CAP for this run.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for scenario selection/etc.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.seed is None:
        np.random.seed()
        random.seed()
    else:
        np.random.seed(args.seed)
        random.seed(args.seed)

    scenario = pick_world(args.scenario)
    yaml_path, ref_path, zone = load_paths(scenario)

    if args.deny == "ask":
        deny = ask_gnss_denial()
    else:
        deny = args.deny == "y"

    step_cap = int(args.step_cap) if args.step_cap is not None else STEP_CAP

    env = irsim.make(world_name=str(yaml_path), save_ani=False, display=not args.headless, full=False)
    ref_path_arr = to_array_traj(ref_path)
    if not args.headless:
        env.draw_trajectory(ref_path_arr, traj_type="-k")
        draw_zone(env, zone, active=deny)
        add_zone_legend(env, active=deny)

    agent = MPCAgent(env, ref_path)
    ref_x, ref_y = _prep_ref_arrays(ref_path)

    # optional logging
    logger, log_path = _make_logger(args.log, scenario.name, deny)
    try:
        config_path = Path(__file__).resolve().parent / "mpc_config.yaml"
        _write_meta(args.log, log_path, scenario, deny, config_path)
    except Exception:
        pass

    fallback_printed = False
    for step in range(step_cap):
        obs_list = env.get_obstacle_info_list()
        true_state = np.asarray(env.robot.state[0:3], dtype=float).reshape(-1)
        state = true_state.copy()
        goal_vec = np.asarray(env.robot.goal, dtype=float).flatten()
        gx, gy = float(goal_vec[0]), float(goal_vec[1])
        sx, sy = float(state[0]), float(state[1])
        dist = float(np.hypot(gx - sx, gy - sy))
        in_zone = False
        noise_applied = False
        if zone is not None:
            dx = sx - float(zone["x"])
            dy = sy - float(zone["y"])
            in_zone = dx * dx + dy * dy <= float(zone["r"]) ** 2
        if deny and in_zone:
            state = apply_gnss_denial(state)
            noise_applied = True

        opt_vel, info = agent.control_step(state, obs_list)

        if not fallback_printed:
            mode = info.get("mode", "unknown")
            print(f"Controller mode: {mode}")
            fallback_printed = True

        if dist < GOAL_THRESH or info.get("arrive"):
            print("Goal reached")
            break
        if step % 50 == 0:
            print(f"Step {step}: dist to goal {dist:.2f}")
            if dist < 0.5:
                print("Goal reached")
                break

        opt_traj = info.get("opt_state_list", [])
        if not args.headless:
            try:
                opt_traj = to_array_traj(opt_traj)
                env.draw_trajectory(opt_traj, "r", refresh=True)
            except Exception:
                # fallback: skip drawing if format unexpected
                pass

        flat = np.array(opt_vel, dtype=float).ravel()
        opt_vec = np.zeros((2, 1), dtype=float)
        if flat.size > 0:
            opt_vec[0, 0] = flat[0]
        if flat.size > 1:
            opt_vec[1, 0] = flat[1]
        try:
            vmax = np.array(env.robot.vel_max[:2], dtype=float).reshape(2, 1)
        except Exception:
            vmax = np.array([[1.0], [1.0]])
        opt_vec = np.clip(opt_vec, -vmax, vmax)

        # Log after clipping so we can correlate saturation/loops.
        if logger is not None:
            cte, ref_idx = _nearest_ref(float(true_state[0]), float(true_state[1]), ref_x, ref_y)
            mpc_cur_index = ""
            try:
                if getattr(agent, "mpc", None) is not None:
                    mpc_cur_index = int(getattr(agent.mpc, "cur_index", -1))
            except Exception:
                mpc_cur_index = ""
            mode = info.get("mode", "")
            iter_time = info.get("iteration_time", "")
            resi_pri = info.get("resi_pri", "")
            resi_dual = info.get("resi_dual", "")
            ref_speed = info.get("ref_speed", "")
            v_cmd = float(flat[0]) if flat.size > 0 else 0.0
            w_cmd = float(flat[1]) if flat.size > 1 else 0.0
            v_clip = float(opt_vec[0, 0])
            w_clip = float(opt_vec[1, 0])
            collision_flag = 0
            try:
                collision_flag = int(bool(getattr(env.robot, "collision_flag", False)))
            except Exception:
                collision_flag = 0
            min_clear = ""
            num_circ = 0
            try:
                px, py = float(true_state[0]), float(true_state[1])
                best = float("inf")
                for o in obs_list:
                    if getattr(o, "cone_type", None) != "norm2":
                        continue
                    c = np.asarray(getattr(o, "center", [0, 0]), dtype=float).reshape(-1)
                    r = float(getattr(o, "radius", 0.0))
                    d = float(np.hypot(px - float(c[0]), py - float(c[1]))) - r
                    if d < best:
                        best = d
                    num_circ += 1
                if num_circ > 0:
                    min_clear = best
            except Exception:
                min_clear = ""
                num_circ = 0
            logger.w.writerow(
                [
                    step,
                    step * float(env.step_time),
                    float(true_state[0]),
                    float(true_state[1]),
                    float(true_state[2]) if true_state.size > 2 else 0.0,
                    float(state[0]),
                    float(state[1]),
                    float(state[2]) if state.size > 2 else 0.0,
                    dist,
                    cte,
                    ref_idx,
                    mpc_cur_index,
                    ref_speed,
                    v_cmd,
                    w_cmd,
                    v_clip,
                    w_clip,
                    int(abs(v_cmd - v_clip) > 1e-6),
                    int(abs(w_cmd - w_clip) > 1e-6),
                    int(in_zone),
                    int(noise_applied),
                    collision_flag,
                    min_clear,
                    num_circ,
                    mode,
                    int(bool(info.get("arrive"))),
                    int(bool(env.done())),
                    float(iter_time) if isinstance(iter_time, (int, float, np.floating)) else "",
                    float(resi_pri) if isinstance(resi_pri, (int, float, np.floating)) else "",
                    float(resi_dual) if isinstance(resi_dual, (int, float, np.floating)) else "",
                ]
            )

        env.step(opt_vec)
        if not args.headless:
            env.render(show_traj=True, show_trail=True)
        # Avoid "freeze": collision_mode=stop can pin the robot while env.done() may remain false.
        collided = False
        try:
            collided = bool(getattr(env.robot, "collision_flag", False))
        except Exception:
            collided = False
        if env.done() or info.get("arrive") or collided:
            print("Goal reached" if info.get("arrive") else "Episode done")
            break
    env.end(ani_name=scenario.name, show_traj=not args.headless, show_trail=not args.headless, ending_time=5)
    if logger is not None:
        logger.close()
        print(f"Wrote rollout log: {log_path}")


if __name__ == "__main__":
    main()
