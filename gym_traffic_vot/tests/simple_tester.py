import gym
import gym_traffic_vot
from tqdm import tqdm
from scipy.stats import sem, t
from scipy import mean

env = gym.make("traffic-vot-simple-v0")
phases = [(0, 40), (2, 5), (1, 40), (3, 5)]
episode_rewards = []
confidence = 0.95

while True:
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
    episode_rewards.append(total_reward)
    if 0 == len(episode_rewards) % 10:
        n = len(episode_rewards)
        m = mean(episode_rewards)
        std_err = sem(episode_rewards)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        print('Avg. reward of last {} episodes is [{:.2f}-{:.2f}]'.format(n, m - h, m + h))
        if h < abs(m * 0.01):
            break

env.close()