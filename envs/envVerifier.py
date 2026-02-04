"""
Small GUI to browse generated environments and visualise them.

Requirements: PyYAML, matplotlib (pip install pyyaml matplotlib)
Run: python -m envs.envVerifier
"""

from __future__ import annotations

import json
import math
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parent
ENV_ROOT = ROOT / "envStorage"

PURPOSE_DIR = {"TR": "training", "TE": "testing", "CO": "comparison"}


def list_ids(purpose: str, model: str) -> list[str]:
    folder = ENV_ROOT / PURPOSE_DIR[purpose]
    if not folder.exists():
        return []
    pat = re.compile(rf"^{re.escape(purpose)}_{re.escape(model)}_(\d{{5}})$")
    ids = []
    for child in folder.iterdir():
        if child.is_dir():
            m = pat.match(child.name)
            if m:
                ids.append(m.group(1))
    ids.sort()
    return ids


def load_env(purpose: str, model: str, idx: str):
    base = f"{purpose}_{model}_{idx}"
    folder = ENV_ROOT / PURPOSE_DIR[purpose] / base
    json_path = folder / f"{base}.json"
    yaml_path = folder / f"{base}.yaml"
    if not json_path.exists() or not yaml_path.exists():
        raise FileNotFoundError(f"Missing files in {folder}")
    with json_path.open("r", encoding="utf-8") as f:
        path = json.load(f)
    with yaml_path.open("r", encoding="utf-8") as f:
        world = yaml.safe_load(f)
    return path, world


def plot_env(path_points, world_dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")

    world = world_dict.get("world", {})
    width = world.get("width", 20)
    height = world.get("height", 20)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title("Environment Visualiser")
    ax.grid(True, linestyle=":", linewidth=0.5)

    # Draw path
    xs, ys = zip(*path_points)
    ax.plot(xs, ys, color="C0", linewidth=1.5, label="reference path")
    ax.scatter(xs[0], ys[0], color="green", s=50, label="start")
    ax.scatter(xs[-1], ys[-1], color="red", s=50, label="goal")

    # Draw obstacles
    obstacles = world_dict.get("obstacle", [])
    for obs_group in obstacles:
        number = obs_group.get("number", 0)
        shapes = obs_group.get("shape", [])
        states = obs_group.get("state", [])
        for i in range(number):
            shape = shapes[min(i, len(shapes) - 1)] if shapes else {}
            state = states[min(i, len(states) - 1)] if states else []
            name = shape.get("name")
            if name == "circle":
                radius = float(shape.get("radius", 1.0))
                x, y = state[:2]
                ax.add_patch(Circle((x, y), radius, color="tomato", alpha=0.5))
            elif name == "rectangle":
                length = float(shape.get("length", 1.0))
                width_rect = float(shape.get("width", 0.2))
                x, y, theta = state[:3]
                rect = Rectangle(
                    (-length / 2, -width_rect / 2),
                    length,
                    width_rect,
                    facecolor="sandybrown",
                    alpha=0.6,
                )
                transform = Affine2D().rotate(theta).translate(x, y) + ax.transData
                rect.set_transform(transform)
                ax.add_patch(rect)
            elif name == "polygon":
                # Basic polygon support; not expected in generated envs.
                verts = shape.get("vertices", [])
                ax.fill([v[0] for v in verts], [v[1] for v in verts], color="gray", alpha=0.5)

    ax.legend(loc="upper left")
    plt.show()


class VerifierUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Environment Verifier")
        self.resizable(False, False)

        self.purpose_var = tk.StringVar(value="TR")
        self.model_var = tk.StringVar(value="PPO")
        self.id_var = tk.StringVar(value="")

        self._build_widgets()
        self._refresh_ids()

    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 6}

        ttk.Label(self, text="Purpose (PP)").grid(row=0, column=0, sticky="w", **pad)
        purpose_box = ttk.Combobox(self, textvariable=self.purpose_var, values=["TR", "TE", "CO"], state="readonly", width=10)
        purpose_box.grid(row=0, column=1, **pad)

        ttk.Label(self, text="Model (MMM)").grid(row=1, column=0, sticky="w", **pad)
        model_box = ttk.Combobox(self, textvariable=self.model_var, values=["PPO", "MPC"], state="readonly", width=10)
        model_box.grid(row=1, column=1, **pad)

        ttk.Label(self, text="ID (NNNNN)").grid(row=2, column=0, sticky="w", **pad)
        self.id_box = ttk.Combobox(self, textvariable=self.id_var, values=[], state="readonly", width=10)
        self.id_box.grid(row=2, column=1, **pad)

        verify_btn = ttk.Button(self, text="VERIFY", command=self._verify)
        verify_btn.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 10))

        # Bind changes
        self.purpose_var.trace_add("write", self._on_selection_change)
        self.model_var.trace_add("write", self._on_selection_change)

    def _on_selection_change(self, *_):
        self._refresh_ids()

    def _refresh_ids(self) -> None:
        ids = list_ids(self.purpose_var.get(), self.model_var.get())
        if not ids:
            self.id_box["values"] = []
            self.id_var.set("")
        else:
            self.id_box["values"] = ids
            # Keep current if still valid, else first available
            if self.id_var.get() not in ids:
                self.id_var.set(ids[0])

    def _verify(self) -> None:
        purpose = self.purpose_var.get()
        model = self.model_var.get()
        idx = self.id_var.get()
        if not idx:
            messagebox.showwarning("Select ID", "No ID available for this purpose/model.")
            return
        try:
            path_points, world_dict = load_env(purpose, model, idx)
            plot_env(path_points, world_dict)
        except FileNotFoundError as e:
            messagebox.showerror("Missing files", str(e))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to load environment: {exc}")


def main():
    if not ENV_ROOT.exists():
        messagebox.showerror("Missing envStorage", f"envStorage not found at {ENV_ROOT}")
        sys.exit(1)
    app = VerifierUI()
    app.mainloop()


if __name__ == "__main__":
    main()
