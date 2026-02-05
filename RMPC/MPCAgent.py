"""
MPC agent wrapper around RDA_planner for reuse in rollouts and batch stats.

Usage (example):
    from RMPC.MPCAgent import MPCAgent, make_env, load_reference_from_npy
    env = make_env(display=True)
    ref_path = load_reference_from_npy(Path("some_ref.npy"))
    agent = MPCAgent(env, ref_path)
    vel, info = agent.control_step(env.robot.state[:3], env.get_obstacle_info_list())
"""

from __future__ import annotations

import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import irsim
from RDA_planner.mpc import MPC

CONFIG_PATH = Path(__file__).resolve().parent / "mpc_config.yaml"


@dataclass
class MPCConfig:
    receding: int
    process_num: int
    iter_num: int
    obstacle_order: bool
    ro1: float
    ro2: float
    max_edge_num: int
    max_obs_num: int
    slack_gain: float
    max_speed: Tuple[float, float]
    max_acce: Tuple[float, float]
    dynamics: str
    sample_time: float | None
    goal_threshold: float

    @staticmethod
    def from_yaml(path: Path) -> "MPCConfig":
        import yaml  # local import

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        ctrl = data.get("controller", {})
        limits = data.get("robot_limits", {})
        integ = data.get("integration", {})
        return MPCConfig(
            receding=int(ctrl.get("receding", 20)),
            process_num=int(ctrl.get("process_num", 4)),
            iter_num=int(ctrl.get("iter_num", 2)),
            obstacle_order=bool(ctrl.get("obstacle_order", True)),
            ro1=float(ctrl.get("ro1", 300)),
            ro2=float(ctrl.get("ro2", 1)),
            max_edge_num=int(ctrl.get("max_edge_num", 4)),
            max_obs_num=int(ctrl.get("max_obs_num", 11)),
            slack_gain=float(ctrl.get("slack_gain", 8)),
            max_speed=tuple(limits.get("max_speed", [10, 2])),
            max_acce=tuple(limits.get("max_acce", [10, 1])),
            dynamics=str(limits.get("dynamics", "diff")),
            sample_time=integ.get("sample_time", None),
            goal_threshold=float(integ.get("goal_threshold", 0.3)),
        )


class MPCAgent:
    """Thin wrapper to build and step the RDA-planner MPC."""

    def __init__(self, env: irsim.Env, ref_path: Iterable, config_path: Path | None = None):
        self.env = env
        self.ref_path = self._to_array_path(ref_path)
        cfg_path = config_path or CONFIG_PATH
        self.cfg = MPCConfig.from_yaml(cfg_path)
        self._fallback = False

        robot_info = env.get_robot_info()
        car_tuple = self._build_car_tuple(robot_info)

        sample_time = self.cfg.sample_time or env.step_time
        try:
            self.mpc = self._build_mpc(car_tuple, sample_time)
        except Exception as e:
            print(f"MPC build failure: {type(e).__name__}: {e}")
            self.mpc = None
            self._fallback = True

    def _build_mpc(self, car_tuple, sample_time):
        """Create MPC, retrying with single-process if multiprocessing is blocked."""
        try:
            return MPC(
                car_tuple,
                self.ref_path,
                receding=self.cfg.receding,
                sample_time=sample_time,
                process_num=self.cfg.process_num,
                iter_num=self.cfg.iter_num,
                obstacle_order=self.cfg.obstacle_order,
                ro1=self.cfg.ro1,
                max_edge_num=self.cfg.max_edge_num,
                max_obs_num=self.cfg.max_obs_num,
                slack_gain=self.cfg.slack_gain,
            )
        except PermissionError:
            # Windows or sandbox may block multiprocessing pipes; fall back.
            return MPC(
                car_tuple,
                self.ref_path,
                receding=self.cfg.receding,
                sample_time=sample_time,
                process_num=1,
                iter_num=self.cfg.iter_num,
                obstacle_order=self.cfg.obstacle_order,
                ro1=self.cfg.ro1,
                max_edge_num=self.cfg.max_edge_num,
                max_obs_num=self.cfg.max_obs_num,
                slack_gain=self.cfg.slack_gain,
            )
        except OverflowError:
            # Solver blew up; retry with lighter settings.
            return MPC(
                car_tuple,
                self.ref_path,
                receding=min(10, self.cfg.receding),
                sample_time=sample_time,
                process_num=1,
                iter_num=1,
                obstacle_order=self.cfg.obstacle_order,
                ro1=self.cfg.ro1,
                max_edge_num=min(2, self.cfg.max_edge_num),
                max_obs_num=min(8, self.cfg.max_obs_num),
                slack_gain=self.cfg.slack_gain,
            )

    def _build_car_tuple(self, robot_info):
        car = namedtuple("car", "G h cone_type wheelbase max_speed max_acce dynamics")
        wheelbase = robot_info.shape[2] if len(robot_info.shape) > 2 else 0
        return car(
            robot_info.G,
            robot_info.h,
            robot_info.cone_type,
            wheelbase,
            list(self.cfg.max_speed),
            list(self.cfg.max_acce),
            self.cfg.dynamics,
        )

    def control_step(self, state: np.ndarray, obstacles) -> Tuple[np.ndarray, dict]:
        """Compute one MPC action given robot state and obstacle list."""
        if self._fallback or self.mpc is None:
            return self._pure_pursuit(state)
        try:
            opt_vel, info = self.mpc.control(state[0:3], 4, obstacles)
            info["mode"] = "mpc"
        except OverflowError:
            self._fallback = True
            self.mpc = None
            return self._pure_pursuit(state)
        except Exception as e:
            print(f"MPC failure: {type(e).__name__}: {e}")
            self._fallback = True
            self.mpc = None
            return self._pure_pursuit(state)
        # Optional online tuning hook
        self.mpc.rda.assign_adjust_parameter(ro1=self.cfg.ro1, ro2=self.cfg.ro2)
        return opt_vel, info

    def _pure_pursuit(self, state):
        idx = getattr(self, "_pp_idx", 0)
        target = self.ref_path[min(idx, len(self.ref_path) - 1)].flatten()
        dx, dy = target[0] - state[0], target[1] - state[1]
        heading = state[2]
        desired_heading = np.arctan2(dy, dx)
        heading_err = desired_heading - heading
        # normalize
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        v = 0.5  # conservative forward speed
        w = max(-1.0, min(1.0, 1.2 * heading_err))  # clamp angular rate
        if np.hypot(dx, dy) < 0.5 and idx < len(self.ref_path) - 1:
            self._pp_idx = idx + 1
        return np.array([float(v), float(w)], dtype=float), {"arrive": False, "mode": "pure_pursuit"}

    def reset(self, ref_path: Iterable | None = None):
        """Reset to a new reference path (optional)."""
        if ref_path is not None:
            self.ref_path = self._to_array_path(ref_path)
        self.mpc.set_ref_path(self.ref_path)

    @staticmethod
    def _to_array_path(path_iter: Iterable) -> list:
        """
        Ensure ref path is list of 3x1 numpy arrays [x,y,theta]^T as expected by MPC.
        If heading is missing, compute it from successive points; last point reuses previous heading.
        """
        pts = list(path_iter)
        arr = []
        n = len(pts)
        for i, p in enumerate(pts):
            if isinstance(p, np.ndarray) and p.shape[0] >= 3:
                if p.shape == (3, 1):
                    arr.append(p)
                elif p.shape == (3,):
                    arr.append(p.reshape(3, 1))
                else:
                    arr.append(np.array(p).reshape(3, 1))
                continue

            x, y = (float(p[0]), float(p[1]))
            if len(p) >= 3:
                theta = float(p[2])
            else:
                nxt = pts[i + 1] if i + 1 < n else pts[i - 1] if i > 0 else p
                dx, dy = float(nxt[0]) - x, float(nxt[1]) - y
                theta = float(np.arctan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0
            arr.append(np.array([[x], [y], [theta]], dtype=float))
        return arr


def load_reference_from_npy(npy_path: Path) -> list:
    return list(np.load(npy_path, allow_pickle=True))


def make_env(display: bool = False):
    """Factory to create an ir-sim environment consistent with RDA-planner usage."""
    return irsim.make(save_ani=False, display=display, full=False)
