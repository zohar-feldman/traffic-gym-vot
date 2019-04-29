import gym
import gym_traffic_vot
from tqdm import tqdm
import sys
import tensorflow as tf
from baselines.deepq.models import build_q_func

env = gym.make("traffic-social-simple-v0")
sess = tf.Session()
# q_func = build_q_func('mlp')
# with tf.variable_scope('deepq_play', reuse=tf.AUTO_REUSE):
#     self.obs_t_input = ObservationInput(self.observation_space, name="obs_t")
#     self.q_t = q_func(self.obs_t_input.get(), self.action_space.n, scope="q_func")
#     self.v_t = tf.squeeze(tf.reduce_max(self.q_t, axis=1))
#     self.q_action = tf.squeeze(tf.argmax(self.q_t, axis=1))
#     tf.initialize_all_variables().run(session=self.sess)

for episode in tqdm(range(10)):
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
env.close()