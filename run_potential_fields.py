import sys
import irsim

world = sys.argv[1] if len(sys.argv) > 1 else "mixed_obstacles"
env = irsim.make(f"potential_fields/worlds/{world}.yaml")
env.load_behavior("potential_fields.potential_fields")

for _ in range(1000):
    env.step()
    env.render()

    if env.done():
        break

env.end(5)