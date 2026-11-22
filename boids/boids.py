from typing import Any
import numpy as np

from irsim.lib.behavior.behavior_registry import register_group_behavior
from irsim.world.object_base import ObjectBase

from .helpers import limit, get_velocity, boundary_recovery, clamp_speed

@register_group_behavior("omni", "boids")
def boids(members: list[ObjectBase], **kwargs: Any) -> list[np.ndarray]:
    """
    """
    if not members:
        return []

    perception_radius = kwargs.get("perception_radius", 5.0)
    vel_max = kwargs.get("velocity_max", 3.0)
    vel_min = kwargs.get("velocity_min", 1.0)  # birds never stop
    max_force = kwargs.get("max_force", 0.3)   # max velocity change per step
    
    separation_weight = kwargs.get("separation_weight", 0.5)
    alignment_weight = kwargs.get("alignment_weight", 1.0)
    cohesion_weight = kwargs.get("cohesion_weight", 1.5)
    safe_distance = kwargs.get("safe_distance", 2.0)

    goal_position = kwargs.get("goal_position", [25.0, 25.0])
    goal_radius = kwargs.get("goal_radius", 6.0)
    goal_weight = kwargs.get("goal_weight", 1.0)

    cruise_speed = kwargs.get("cruise_speed", 1.0)
    cruise_weight = kwargs.get("cruise_weight", 0.5)
    fov = kwargs.get("field_of_view", 270.0)  # degrees; 270 = not seeing back blindpsot

    # [x_min, y_min, x_max, y_max]: match your world's width/height/offset
    bounds = kwargs.get("bounds", [0.0, 0.0, 50.0, 50.0])
    boundary_margin = kwargs.get("boundary_margin", 5.0)

    # Predator avoidance
    flee_radius = kwargs.get("flee_radius", 8.0)
    flee_force = kwargs.get("flee_force", 0.8)
    wall_force = kwargs.get("wall_force", 1.0)

    predators = [
        obj for obj in kwargs.get("external_objects", [])
        if obj.group_behavior_dict.get("name") == "predator"
    ]

    velocities = []
    for member in members:
        current_pos = np.asarray(member.state[:2, 0], dtype=float).reshape(2)
        current_velocity = get_velocity(member, vel_min)

        # Separation sees all birds since bird behind you can sitll hit you
        # alignment and cohesion only use birds in arc in front
        close_birds = find_neighbors(member, members, perception_radius)
        front_birds = find_neighbors(member, members, perception_radius, heading=current_velocity, fov_deg=fov)

        # flocking rules (steering vectors)
        flock_steer = (
            cruising(current_velocity, cruise_speed) * cruise_weight
            + goal_attraction(current_pos, current_velocity, goal_position, goal_radius, cruise_speed) * goal_weight
        )

        if close_birds:
            flock_steer += separation(member, close_birds, safe_distance) * separation_weight
        
        if front_birds:
            flock_steer += (
                alignment(current_velocity, front_birds) * alignment_weight
                + cohesion(current_pos, current_velocity, front_birds, perception_radius) * cohesion_weight
            )
        
        wall_steer = boundary_recovery(current_pos, current_velocity, bounds, boundary_margin, vel_max)

        flee_steer = flee(current_pos, current_velocity, predators, flee_radius, vel_max)

        new_velocity = (
            current_velocity 
            + limit(flock_steer, max_force)
            + limit(wall_steer, wall_force)
            + limit(flee_steer, flee_force)
        )

        new_velocity = clamp_speed(new_velocity, current_velocity, vel_min, vel_max)
        velocities.append(member.vel_world2body(new_velocity.reshape(2, 1)))

    return velocities


##### Helpers
def find_neighbors(
    primary: ObjectBase, 
    members: list[ObjectBase], 
    perception_radius=5.0,
    heading=None,
    fov_deg=360
) -> list[ObjectBase]:
    """Neighbors within perception_radius. If `heading` (a world-frame direction,
    e.g. the bird's velocity) is given, only birds inside the field of view
    `fov_deg` centered on that heading count. 180 = only birds in front."""

    # omni -> returns [x,y theta]
    primary_pos = primary.state[:2, 0]

    use_fov = heading is not None and fov_deg < 360 and np.linalg.norm(heading) > 1e-6

    if use_fov:
        forward = np.asarray(heading, dtype=float).reshape(2)
        forward = forward / np.linalg.norm(forward)
        cos_half_fov = np.cos(np.radians(fov_deg) / 2.0)
    
    neighbors = []
    for member in members:
        member_pos = member.state[:2, 0]
        offset = member_pos - primary_pos
        distance = np.linalg.norm(primary_pos - member_pos)

        if not (0 < distance < perception_radius):
            continue
        
        # angle between the facing direction and the direction to the neighbor
        if use_fov and np.dot(forward, offset / distance) < cos_half_fov:
            continue
        
        neighbors.append(member)

    return neighbors


##### Rules
def separation(primary: ObjectBase, neighbors: list[ObjectBase], safe_distance=2.0) -> np.ndarray:
    repulsion_vector = np.zeros(2)
    primary_pos = primary.state[:2, 0]

    for neighbor in neighbors:
        neighbor_pos = neighbor.state[:2, 0]
        difference = primary_pos - neighbor_pos
        distance = np.linalg.norm(difference)

        if distance < 1e-6:
            continue
    
        if distance < safe_distance:
            direction = difference / distance 
            urgency = (safe_distance - distance) / safe_distance
            repulsion_vector += direction * urgency / distance

    return repulsion_vector

def alignment(velocity: np.ndarray, neighbors: list[ObjectBase]) -> np.ndarray:
    avg_velocity = np.mean(
        [np.asarray(n.velocity_xy, dtype=float).reshape(2) for n in neighbors], axis=0
    )
    return avg_velocity - velocity

def cohesion(position, velocity, neighbors: list[ObjectBase], perception_radius) -> np.ndarray:
    """Pull toward the neighbors' center. This is a pure position pull, with no
    '- velocity' term, so it bends the path without slowing the bird down.
    Strength grows from 0 (at the center) to 1 (at the edge of perception)."""
     
    avg_neighbor_pos = np.mean([n.state[:2, 0] for n in neighbors], axis=0)

    direction = avg_neighbor_pos - position
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        return np.zeros(2)
    
    strength = min(1.0, distance / perception_radius)
    return strength * direction / distance


def cruising(velocity, cruise_speed):
    """Keep the bird at cruise_speed in the direction it is ALREADY moving.
    (state[2] is not usable: omni robots never change their theta.)"""
    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        return np.zeros(2)
    return velocity / speed * cruise_speed - velocity

def goal_attraction(position, velocity, goal_position, goal_radius, cruise_speed):
    direction = np.asarray(goal_position, dtype=float) - position
    distance = np.linalg.norm(direction)

    # Inside the central region: no additional attraction
    if distance <= goal_radius:
        return np.zeros(2)

    # Outside: steer toward the goal, gently at first and stronger further out
    strength = min(1.0, (distance - goal_radius) / goal_radius)
    desired_velocity = direction / distance * cruise_speed
    return strength * (desired_velocity - velocity)

def flee(position, velocity, predators: list[ObjectBase], flee_radius, vel_max):
    """Steer away from every predator inside flee_radius, at full speed.
    The closer the predator, the stronger the push (urgency 0 -> 1)."""
    steer = np.zeros(2)
    for predator in predators:
        predator_pos = np.asarray(predator.state[:2, 0], dtype=float).reshape(2)
        away = position - predator_pos # arrow pointing away from predator
        distance = np.linalg.norm(away)

        if distance < 1e-6 or distance > flee_radius:
            continue

        urgency = (flee_radius - distance) / flee_radius
        desired_velocity = away / distance * vel_max
        steer += urgency * (desired_velocity - velocity)
    
    return steer

