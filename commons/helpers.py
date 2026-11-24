"""Vector and robot helpers shared by the boids, predator and potential-fields behaviors."""

import numpy as np

from irsim.world.object_base import ObjectBase

EPS = 1e-6


def get_position(obj: ObjectBase) -> np.ndarray:
    """World-frame [x, y] position."""
    return np.asarray(obj.state[:2, 0], dtype=float).reshape(2)


def get_velocity(obj: ObjectBase, vel_min: float) -> np.ndarray:
    """World-frame velocity. A stationary robot gets `vel_min` along its heading."""
    velocity = np.asarray(obj.velocity_xy, dtype=float).reshape(2)
    if np.linalg.norm(velocity) < 1e-3:
        heading = obj.state[2, 0]
        velocity = vel_min * np.array([np.cos(heading), np.sin(heading)])
    return velocity


def with_behavior(objects: list[ObjectBase], name: str) -> list[ObjectBase]:
    """Objects whose group behavior is `name` (e.g. "boids", "predator")."""
    return [obj for obj in objects if obj.group_behavior_dict.get("name") == name]


def limit(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale `vector` down so its length is at most `max_norm`."""
    norm = np.linalg.norm(vector)
    return vector / norm * max_norm if norm > max_norm else vector


def clamp_speed(velocity, fallback, vel_min, vel_max) -> np.ndarray:
    """Keep speed within [vel_min, vel_max]. Uses `fallback`'s direction if `velocity` is ~0."""
    speed = np.linalg.norm(velocity)
    if speed > vel_max:
        return velocity / speed * vel_max
    if speed < vel_min:
        direction = velocity if speed > EPS else fallback
        return direction / np.linalg.norm(direction) * vel_min
    return velocity


def find_neighbors(primary: ObjectBase, members: list[ObjectBase], perception_radius,
                   heading=None, fov_deg=360) -> list[ObjectBase]:
    """Members within `perception_radius` of `primary` (excluding itself).
    If `heading` is given, only members inside the `fov_deg` cone around it count."""
    primary_pos = get_position(primary)

    use_fov = heading is not None and fov_deg < 360 and np.linalg.norm(heading) > EPS
    if use_fov:
        forward = np.asarray(heading, dtype=float).reshape(2)
        forward = forward / np.linalg.norm(forward)
        cos_half_fov = np.cos(np.radians(fov_deg) / 2)

    neighbors = []
    for member in members:
        offset = get_position(member) - primary_pos
        distance = np.linalg.norm(offset)
        if not 0 < distance < perception_radius:
            continue
        if use_fov and np.dot(forward, offset / distance) < cos_half_fov:
            continue
        neighbors.append(member)
    return neighbors


def boundary_recovery(position, velocity, bounds, margin, vel_max) -> np.ndarray:
    """Soft walls: within `margin` of an edge of `bounds` = [x_min, y_min, x_max, y_max],
    steer that velocity component back inside at `vel_max`."""
    x_min, y_min, x_max, y_max = bounds
    desired = velocity.copy()

    for axis, (low, high) in enumerate([(x_min, x_max), (y_min, y_max)]):
        if position[axis] < low + margin:
            desired[axis] = vel_max
        elif position[axis] > high - margin:
            desired[axis] = -vel_max

    return desired - velocity
