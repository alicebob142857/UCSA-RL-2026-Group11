"""Linear dynamics utilities for the Group 11 balancing robot simulator.

The model is intentionally small and explicit.  It follows the same physical
idea as a two-stage balancing robot: wheel coordinates, body angle, pole angle,
and their velocities form an 8D state.  The returned matrices use the internal
physics order:

    theta_l, theta_r, body_angle, pole_angle,
    theta_l_dot, theta_r_dot, body_rate, pole_rate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class RobotConstants:
    wheel_radius_m: float = 0.0335
    body_mass_kg: float = 0.9
    pole_mass_kg: float = 0.1
    body_length_m: float = 0.126
    pole_length_m: float = 0.390
    gravity: float = 9.8
    dt: float = 0.01


def continuous_matrices(constants: RobotConstants = RobotConstants()) -> tuple[np.ndarray, np.ndarray]:
    """Build continuous-time A/B matrices from a compact rigid-body model."""
    m_body = constants.body_mass_kg
    m_pole = constants.pole_mass_kg
    radius = constants.wheel_radius_m
    body_half = constants.body_length_m / 2.0
    pole_half = constants.pole_length_m / 2.0
    body_inertia = m_body * constants.body_length_m**2 / 12.0
    pole_inertia = m_pole * constants.pole_length_m**2 / 12.0

    mass_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [
                0.5 * radius * (m_body * body_half + m_pole * constants.body_length_m),
                0.5 * radius * (m_body * body_half + m_pole * constants.body_length_m),
                m_body * body_half**2 + m_pole * constants.body_length_m**2 + body_inertia,
                m_pole * constants.body_length_m * pole_half,
            ],
            [
                0.5 * radius * m_pole * pole_half,
                0.5 * radius * m_pole * pole_half,
                m_pole * constants.body_length_m * pole_half,
                m_pole * pole_half**2 + pole_inertia,
            ],
        ],
        dtype=np.float64,
    )

    forcing = np.zeros((4, 10), dtype=np.float64)
    forcing[0, 8] = 1.0
    forcing[1, 9] = 1.0
    forcing[2, 2] = (m_body * body_half + m_pole * constants.body_length_m) * constants.gravity
    forcing[3, 3] = m_pole * constants.gravity * pole_half

    accel_map = np.linalg.solve(mass_matrix, forcing)

    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros((8, 2), dtype=np.float64)
    a[:4, 4:] = np.eye(4)
    a[4:, :] = accel_map[:, :8]
    b[4:, :] = accel_map[:, 8:]
    return a, b


def discretize(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert continuous dynamics to zero-order-hold discrete dynamics."""
    g, h, _, _, _ = signal.cont2discrete(
        (a, b, np.eye(a.shape[0]), np.zeros((a.shape[0], b.shape[1]))),
        dt,
        method="zoh",
    )
    return g.astype(np.float64), h.astype(np.float64)


def default_discrete_model(constants: RobotConstants = RobotConstants()) -> tuple[np.ndarray, np.ndarray]:
    """Return the nominal discrete dynamics used by the simulator."""
    a, b = continuous_matrices(constants)
    return discretize(a, b, constants.dt)
