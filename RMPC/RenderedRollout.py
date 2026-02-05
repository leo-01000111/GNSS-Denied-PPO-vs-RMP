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

import json
import random
import sys
from pathlib import Path

import irsim
import numpy as np

# Allow running as script from RMPC/ by adding project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from RMPC.MPCAgent import MPCAgent  # type: ignore
from utilities.DENIAL import apply_gnss_denial  # type: ignore

ROOT = Path(__file__).resolve().parent
ENV_ROOT = ROOT.parent / "envs" / "envStorage" / "testing"

STEP_CAP = 500


def pick_random_world():
    candidates = list((ENV_ROOT).glob("TE_MPC_*"))
    if not candidates:
        raise SystemExit(f"No testing MPC scenarios found in {ENV_ROOT}")
    return random.choice(candidates)


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


def main():
    np.random.seed()
    random.seed()

    scenario = pick_random_world()
    yaml_path, ref_path, zone = load_paths(scenario)

    deny = ask_gnss_denial()

    env = irsim.make(world_name=str(yaml_path), save_ani=False, display=True, full=False)
    ref_path_arr = to_array_traj(ref_path)
    env.draw_trajectory(ref_path_arr, traj_type="-k")
    draw_zone(env, zone, active=deny)
    add_zone_legend(env, active=deny)

    agent = MPCAgent(env, ref_path)

    fallback_printed = False
    for step in range(STEP_CAP):
        obs_list = env.get_obstacle_info_list()
        state = np.array(env.robot.state[0:3], dtype=float)
        if deny:
            state = apply_gnss_denial(state)

        opt_vel, info = agent.control_step(state, obs_list)
        if not fallback_printed:
            mode = info.get("mode", "unknown")
            print(f"Controller mode: {mode}")
            fallback_printed = True
        if step % 50 == 0:
            goal = env.robot.goal
            gx, gy = float(goal[0]), float(goal[1])
            dist = np.hypot(gx - float(env.robot.state[0]), gy - float(env.robot.state[1]))
            print(f"Step {step}: dist to goal {dist:.2f}")
            if dist < 0.5:
                print("Goal reached")
                break

        opt_traj = info.get("opt_state_list", [])
        try:
            opt_traj = to_array_traj(opt_traj)
            env.draw_trajectory(opt_traj, "r", refresh=True)
        except Exception:
            # fallback: skip drawing if format unexpected
            pass

        try:
            vmax = np.array(env.robot.vel_max[:2], dtype=float)
        except Exception:
            vmax = np.array([1.0, 1.0])
        opt_vel = np.clip(opt_vel, -vmax, vmax)

        env.step(opt_vel)
        env.render(show_traj=True, show_trail=True)

        if env.done() or info.get("arrive"):
            print("Goal reached" if info.get("arrive") else "Episode done")
            break
    env.end(ani_name=scenario.name, show_traj=True, show_trail=True, ending_time=5)


if __name__ == "__main__":
    main()
