import gymnasium as gym
import math

env = gym.make("CartPole-v1").unwrapped
env.theta_threshold_radians = 20 * math.pi
env.x_threshold = 200.0

obs, _ = env.reset()
terminated = False
truncated = False

for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(1)
    if terminated or truncated:
        print(f"Terminated at {i}")
        break

print(f"Finished 1000 steps without terminating. Final obs: {obs}")
