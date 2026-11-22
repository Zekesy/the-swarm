from typing import Any
import numpy as np

from irsim.world.object_base import ObjectBase

def limit(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > max_norm:
        return vector / norm * max_norm
    return vector

def get_velocity(member: ObjectBase, vel_min: float) -> np.ndarray:
    """World-frame velocity"""
    velocity = np.asarray(member.velocity_xy, dtype=float).reshape(2)
    if np.linalg.norm(velocity) < 1e-3:
        heading = member.state[2,0]
        velocity = vel_min * np.array([np.cos(heading), np.sin(heading)])
    return velocity

def boundary_recovery(position, velocity, bounds, margin, vel_max):
    """Soft walls: inside `margin` of an edge (or outside the world), steer
    the velocity component toward the inside. No hard mode switch."""

    x_min, y_min, x_max, y_max = bounds
    desired = velocity.copy()
    near_wall = False

    if position[0] < x_min + margin:
        desired[0] = vel_max
        near_wall = True
    elif position[0] > x_max - margin:
        desired[0] = -vel_max
        near_wall = True

    if position[1] < y_min + margin:
        desired[1] = vel_max
        near_wall = True
    elif position[1] > y_max - margin:
        desired[1] = -vel_max
        near_wall = True

    if not near_wall:
        return np.zeros(2)
    return desired - velocity

def clamp_speed(velocity, fallback, vel_min, vel_max):
    speed = np.linalg.norm(velocity)
    if speed > vel_max:
        return velocity / speed * vel_max
    if speed < vel_min:
        direction = velocity if speed > 1e-6 else fallback
        return direction / np.linalg.norm(direction) * vel_min
    return velocity