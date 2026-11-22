import irsim
import matplotlib.pyplot as plt


env = irsim.make("robot_world.yaml")
env.load_behavior("boids.boids")      # boids/boids.py
env.load_behavior("boids.predator")   # boids/predator.py

# Draw the central goal region
goal_x, goal_y = 25.0, 25.0
goal_radius = 3.0

circle = plt.Circle(
    (goal_x, goal_y),
    goal_radius,
    color="green",
    fill=False,
    linestyle="--",
    linewidth=2,
    label="Goal region"
)

ax = plt.gca()
ax.add_patch(circle)
ax.legend()

for _ in range(1000):
    env.step()
    env.render()

    if env.done():
        break

env.end(5)