# the-swarm

A personal learning repo for **multi-agent coordination**: implementing
algorithms from scratch, seeing them run, and understanding the math behind
them. Simulations run in [IR-SIM](https://github.com/hanruihua/ir-sim), a
lightweight Python robot simulator.

The goal isn't production code. It's to build intuition for how simple local
rules, planning and coordination let groups of agents do things no single
agent could, and to keep my math (vectors, trig, linear algebra) sharp along
the way.

## What's here

| Module | What it does | Key idea |
| --- | --- | --- |
| `boids/boids.py` | Reynolds flocking: separation, alignment, cohesion, field of view, soft walls, fleeing predators | Simple local rules produce emergent group behavior |
| `boids/predator.py` | Predators that hunt boids in range, patrol to the center otherwise, and never share a target | Greedy one-to-one task allocation |
| `boids/helpers.py` | Shared steering utilities (`limit`, `clamp_speed`, soft walls) | Steering = desired velocity − current velocity |

## Running

```bash
pip install ir-sim
python quick_start.py
```

## Lessons learned

- **Steer, don't replace.** New velocity = current + limited steering. Returning
  a steering vector as the velocity (`v_next = desired − v`) makes agents
  oscillate every step.
- **Keep the per-step change below the minimum speed** so agents can only turn, never reverse.
- **Prioritize with separate budgets.** Walls > fleeing > flocking, so an
  important rule can't be outvoted.
- **IR-SIM omni robots take body-frame velocity commands.** Do the math in the
  world frame, then convert once with `vel_world2body`.
- **Omni robots never change theta.** Use the velocity direction as the heading.
- **Set `vel_max`/`vel_min` in the YAML.** The default omni limit is 1 per axis.

## Resources

- Craig Reynolds, *Steering Behaviors for Autonomous Characters*
- Daniel Shiffman, *The Nature of Code* (autonomous agents chapter)
- 3Blue1Brown, *Essence of Linear Algebra*; Freya Holmér, *Math for Game Devs*
- Red Blob Games: A\* and path finding articles
- Lynch & Park, *Modern Robotics*
- PythonRobotics (Atsushi Sakai) on GitHub