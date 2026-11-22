from typing import Any
import numpy as np

from irsim.lib.behavior.behavior_registry import register_behavior
from irsim.world.object_base import ObjectBase




@register_behavior("omni", "potential-fields")
def potential_fields(ego_object: ObjectBase, external_objects, **kwargs: Any) -> np.ndarray:
    primary = ego_object

    vel_max = kwargs.get("velocity_max", 3.0)
    vel_min = kwargs.get("velocity_min", 0.05)

    goal_pos = primary.goal[:2, 0]
    goal_reached_threshold = kwargs.get("goal_threshold", 0.05)
    goal_distance_threshold = kwargs.get("goal_distance_threshold", 25.0)
    
    attractive_gain = kwargs.get("attraction_gain", 6.0)
    repulsion_gain = kwargs.get("repulsion_gain", 3.0)
    influence_range = kwargs.get("influence_range", 5.0)


    primary_pos = primary.state[:2, 0]
    current_velocity = get_velocity(primary, vel_min)

    goal_direction = goal_pos - primary_pos
    goal_distance = np.linalg.norm(goal_direction)

    if goal_distance < goal_reached_threshold:
        return np.zeros((2,1))

    # Calculate pull towards the goal
    attractive_f = attractive_force(primary_pos, goal_pos, attractive_gain, goal_distance_threshold)

    # Calculate repulsive force of all obstacles
    repulsive_f = np.zeros(2)
    for obstacle in external_objects:
        obstacle_pos = obstacle.state[:2, 0]
        away_direction = primary_pos - obstacle_pos
        gap = np.linalg.norm(away_direction) - obstacle.radius - primary.radius

        repulsive_f += repulsive_force(gap, away_direction, repulsion_gain, influence_range)

    total_force = attractive_f + repulsive_f
    new_velocity = clamp_speed(total_force, current_velocity, vel_min, vel_max)

    return primary.vel_world2body(new_velocity.reshape(2, 1))


def attractive_force(primary_pos, goal_pos, attractive_gain, distance_threshold) -> np.ndarray:
    # attracive force
    # direction to goal 
    direction = goal_pos - primary_pos
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        return np.zeros(2)
    
    # Since force grows with distance, using a bowl potential
    # means that hte force will get extremely strong farther out
    # better to use cone for when target is still far 
    # as it has constant strength
    if distance <= distance_threshold: # Bowl potential
        attractive_force = attractive_gain * direction
    else:        # Conic potential
        attractive_force = (
            distance_threshold
            * attractive_gain 
            * direction / distance
        )

    return attractive_force

def repulsive_force(gap, away_direction, gain, influence_range) -> np.ndarray:
    if gap > influence_range:
        return np.zeros(2)
    
    # Puhs away from one specific obstacle
    # So farther away, gap is bigger, force is weaker
    repulsive_force = gain * (1/gap - 1/influence_range) / gap ** 2
    direction = away_direction / np.linalg.norm(away_direction)
    
    return direction * repulsive_force

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