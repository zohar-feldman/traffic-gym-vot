import gym
import gym_traffic_vot
from tqdm import tqdm

env = gym.make("traffic-vot-simple-v0")
phases = [(0, 40), (2, 5), (1, 40), (3, 5)]
for episode in tqdm(range(10)):
    phase_time = 0
    phase = 0
    episode_losses = []
    observation = env.reset()
    total_reward = 0.0
    for _ in range(1000):
        env.render()
        if phase_time == phases[phase][1]:
            phase = (phase + 1) % 4
            phase_time = 0
        # action = env.action_space.sample() # your agent here (this takes random actions)
        action = phases[phase][0]
        observation, reward, done, info = env.step(action)
        total_reward += reward
        phase_time += 1
        if done:
            observation = env.reset()
    print('episode reward={}'.format(total_reward))
env.close()