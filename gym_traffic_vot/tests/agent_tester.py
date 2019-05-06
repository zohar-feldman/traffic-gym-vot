import gym
import gym_traffic_vot
from tqdm import tqdm
import sys
import tensorflow as tf
from baselines.deepq.models import build_q_func
from scipy.stats import sem, t
from scipy import mean

env = gym.make("traffic-social-simple-v0")
sess = tf.Session()

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
        action = env.env.sess.run([env.env.q_action], {env.env.obs_t_input.get(): [observation]})
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