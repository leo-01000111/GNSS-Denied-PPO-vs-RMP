"""
Utility to inject GNSS-denial-like noise into robot state.

Function:
    apply_gnss_denial(state, noise_std=0.5, bias=0.0)

 - state: sequence or np.array [x, y, heading, ...]
 - noise_std: std dev of zero-mean Gaussian applied to x and y
 - bias: optional constant bias added to x and y

Returns a copy of the state with noise applied to x, y only.
"""

from __future__ import annotations

import numpy as np


def apply_gnss_denial(state, noise_std: float = 0.5, bias: float = 0.0):
    """
    Add GNSS-denial-like perturbation to x, y.

    Parameters
    ----------
    state : sequence or np.ndarray
        Robot state (at least x, y present).
    noise_std : float
        Std dev of Gaussian noise applied to x, y.
    bias : float
        Constant bias added to x, y.
    """
    s = np.array(state, dtype=float)
    original_shape = s.shape
    flat = s.reshape(-1)
    if flat.size < 2:
        return s
    noise = np.random.normal(loc=bias, scale=noise_std, size=2)
    flat[0:2] = flat[0:2] + noise
    # restore original shape
    return flat.reshape(original_shape)
