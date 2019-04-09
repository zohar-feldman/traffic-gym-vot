import gym
import gym_traffic_vot
from tqdm import tqdm

env = gym.make("traffic-vot-simple-v0")
for episode in tqdm(range(10)):
  episode_losses = []
  observation = env.reset()
  total_reward = 0.0
  for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    total_reward += reward
    if done:
      observation = env.reset()

env.close()