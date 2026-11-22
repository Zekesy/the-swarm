import irsim
import matplotlib.pyplot as plt

env = irsim.make("potential_fields/robot_world.yaml")
env.load_behavior("potential_fields.potential_fields")

for _ in range(1000):
    env.step()
    env.render()

    if env.done():
        break

env.end(5)