from typing import Any
import numpy as np

from irsim.lib.behavior.behavior_registry import register_group_behavior
from irsim.world.object_base import ObjectBase

from .helpers import limit, get_velocity, boundary_recovery, clamp_speed

##### Predator
@register_group_behavior("omni", "predator")
def predator(members: list[ObjectBase], **kwargs: Any) -> list[np.ndarray]:
    if not members:
        return []
    
    vel_max = kwargs.get("velocity_max", 3.0)
    vel_min = kwargs.get("velocity_min", 1.0)
    max_force = kwargs.get("max_force", 0.3)
    perception_radius = kwargs.get("perception_radius", 8.0)

    bounds = kwargs.get("bounds", [0.0, 0.0, 50.0, 50.0])
    boundary_margin = kwargs.get("boundary_margin", 0.0)

    hunt_radius = kwargs.get("hunt_radius", 8.0)
    center = np.asarray(kwargs.get("center_position", [25,25]), dtype=float)
    patrol_speed = kwargs.get("patrol_speed", 1.5)

    # Spacing between predators
    perception_radius = kwargs.get("perception_radius", 8.0)  # how far it sees other predators
    separation_weight = kwargs.get("separation_weight", 3.0)
    separation_force = kwargs.get("separation_force", 0.3)    # own budget

    prey = [
        obj for obj in kwargs.get("external_objects", [])
        if obj.group_behavior_dict.get("name") == "boids"
    ]

    predator_positions = np.array(
        [np.asarray(m.state[:2, 0], dtype=float).reshape(2) for m in members]
    )

    prey_positions = np.array(
        [np.asarray(p.state[:2, 0], dtype=float).reshape(2) for p in prey]
    ).reshape(-1, 2)

    targets = assign_targets(predator_positions, prey_positions, hunt_radius)

    velocities = []
    for i, member in enumerate(members):
        current_pos = predator_positions[i]
        current_velocity = get_velocity(member, vel_min)

        target_index = targets[i]
        if target_index is not None:
            # HUNT: full speed toward its own assigned boid
            to_target = prey_positions[target_index] - current_pos
            distance = np.linalg.norm(to_target)
            desired_velocity = (
                to_target / distance * vel_max if distance > 1e-6 else current_velocity
            )
        else:
            # PATROL: head to the center, slowing down as it arrives
            to_center = center - current_pos
            distance = np.linalg.norm(to_center)
            if distance > 1e-6:
                speed = min(patrol_speed, distance)  # slows down near center
                desired_velocity = to_center / distance * speed
            else:
                desired_velocity = current_velocity
            
        chase_steer = desired_velocity - current_velocity
        wall_steer = boundary_recovery(current_pos, current_velocity, bounds, boundary_margin, vel_max)

        neighbors = find_neighbors(member, members, perception_radius)
        separation_steer = separation(member, neighbors) * separation_weight

        new_velocity = (
            current_velocity 
            + limit(chase_steer, max_force)
            + limit(wall_steer, max_force)
            + limit(separation_steer, separation_force)
        )

        new_velocity = clamp_speed(new_velocity, current_velocity, vel_min, vel_max)
        velocities.append(member.vel_world2body(new_velocity.reshape(2, 1)))

    return velocities

def find_neighbors(primary: ObjectBase, members:list[ObjectBase], perception_radius) -> list[ObjectBase]:
    primary_pos = primary.state[:2, 0]

    neighbors = []
    for member in members:
        member_pos = member.state[:2, 0]
        distance = np.linalg.norm(primary_pos - member_pos)

        if not (0 < distance < perception_radius):
            continue
        
        neighbors.append(member)

    return neighbors

def assign_targets(predator_positions: np.ndarray, prey_positions: np.ndarray, hunt_radius) -> list:
    """Greedy one-to-one matching of predators to boids.

    Repeatedly take the closest (predator, boid) pair that is still free and
    within hunt_radius, then remove both from the pool. Returns, for each
    predator, the index of its boid, or None if it has no target.
    """
    n_predators = len(predator_positions)
    targets = [None] * n_predators
    if len(prey_positions) == 0:
        return targets

    # distances[i, j] = distance from predator i to boid j
    distances = np.linalg.norm(
        predator_positions[:, None, :] - prey_positions[None, :, :], axis=2
    )
    distances[distances >= hunt_radius] = np.inf  # out of range: not allowed

    for _ in range(min(n_predators, len(prey_positions))):
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        if np.isinf(distances[i, j]):
            break                      # no valid pairs left
        targets[i] = j
        distances[i, :] = np.inf       # this predator is busy
        distances[:, j] = np.inf       # this boid is taken
    return targets

def separation(primary: ObjectBase, neighbors: list[ObjectBase]) -> np.ndarray:
    if not neighbors:
        return np.zeros(2)
    
    repulsion_vector = np.zeros(2)
    primary_pos = primary.state[:2, 0]

    for neighbor in neighbors:
        neighbor_pos = neighbor.state[:2, 0]
        difference = primary_pos - neighbor_pos
        distance = np.linalg.norm(difference)

        if distance < 1e-6:
            continue
        
        direction = difference / distance 
        repulsion_vector += direction / distance
    
    return repulsion_vector
