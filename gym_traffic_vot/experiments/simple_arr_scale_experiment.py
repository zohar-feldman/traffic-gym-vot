import gym
import gym_traffic_vot
from gym_traffic_vot.tests import run
import numpy as np
from baselines import logger
import os.path as osp
import sys
from scipy.stats import sem, t
from scipy import mean
import os
import tensorflow as tf
confidence = 0.95

# run.main(alg='deepq', env='traffic-vot-simple-v0', num_timesteps=2e6, print_freq=100, exploration_fraction=0.5, exploration_final_eps=0.1 ,save_path='C:/Users/FEZ1TV/PycharmProjects/gym-traffic-vot/models/test')

def sample_stats(sample, confidence):
    n = len(sample)
    m = mean(sample)
    std_err = sem(sample)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    return m, std_err, m - h, m + h

phases = [(0, 40), (2, 5), (1, 40), (3, 5)]

def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = run.common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = run.parse_cmdline_kwargs(unknown_args)
    # a_start = extra_args['a_start']
    # a_end = extra_args['a_end']
    # extra_args.pop('a_start')
    # extra_args.pop('a_end')
    for a in np.arange(0.02, 0.17, 0.04):
        for s in [1,2,4,8]:
            res_path = '~/projects/traffic-gym-vot/resutls/simple_{}_{}_deepq.txt'.format(round(a, 2), round(s, 2))
            env_name = 'traffic-vot-simple-v0'

            env = gym.make(env_name, network='simple', arrival_rate=a, scale=s)

            logger.log("Running trained model")
            obs = env.reset()


            episode_rew = 0
            episode_rewards = []
            dir_name = os.path.dirname(osp.expanduser(res_path))
            os.makedirs(dir_name, exist_ok=True)
            f = open(osp.expanduser(res_path), 'w+')
            phase_time = 0
            phase = 0
            while True and len(episode_rewards) < 300:

                if phase_time == phases[phase][1]:
                    phase = (phase + 1) % 4
                    phase_time = 0
                # action = env.action_space.sample() # your agent here (this takes random actions)
                action = phases[phase][0]
                obs, rew, done, _ = env.step(action)
                episode_rew += rew
                phase_time += 1
                env.render()
                if done:
                    phase_time = 0
                    phase = 0
                    episode_rewards.append(episode_rew)
                    f.write("episode_rew={}\n".format(episode_rew))
                    episode_rew = 0
                    if 0 == len(episode_rewards) % 10:
                        n = len(episode_rewards)
                        m, std_err, lb, ub = sample_stats(episode_rewards, confidence)

                        if ub - lb < abs(m * 0.02):
                            break
                    obs = env.reset()
            m, std_err, lb, ub = sample_stats(episode_rewards, confidence)
            f.write('********************\n')
            f.write('num episodes:{}\n'.format(len(episode_rewards)))
            f.write('mean:{}\n'.format(m))
            f.write('lb:{}\n'.format(lb))
            f.write('ub:{}\n'.format(ub))
            f.close()
            env.close()
            tf.get_variable_scope().reuse_variables()

if __name__ == '__main__':
    main(sys.argv)