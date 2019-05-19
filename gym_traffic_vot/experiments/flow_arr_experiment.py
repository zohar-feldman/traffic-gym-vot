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


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = run.common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = run.parse_cmdline_kwargs(unknown_args)
    for a in np.arange(0.02, 0.17, 0.02):
        for s in [1,2,4,8]:
            save_path = osp.join('~{}projects'.format(os.sep),'traffic-gym-vot','models','flow_{}_{}_deepq'.format(a, s))
            res_path = osp.join('~{}projects'.format(os.sep),'traffic-gym-vot','resutls','flow_{}_{}_deepq.txt'.format(a, s))
            env_name = 'traffic-flow-simple-v0'

            env = gym.make(env_name, network='simple', arrival_rate=a, scale=s)
            if args.network:
                extra_args['network'] = args.network
            else:
                if extra_args.get('network') is None:
                    extra_args['network'] = 'mlp'

            print('Training {} on {} with arguments \n{}'.format(args.alg, env_name, extra_args))

            learn = run.get_learn_function(args.alg)
            model = learn(
                env=env,
                seed=args.seed,
                total_timesteps=int(args.num_timesteps),
                **extra_args
            )

            save_path = osp.expanduser(save_path)
            model.save(save_path)

            logger.log("Running trained model")
            obs = env.reset()

            state = model.initial_state if hasattr(model, 'initial_state') else None
            dones = np.zeros((1,))

            episode_rew = 0
            episode_rewards = []
            dir_name = os.path.dirname(osp.expanduser(res_path))
            os.makedirs(dir_name, exist_ok=True)
            f = open(osp.expanduser(res_path), 'w+')
            while True and len(episode_rewards) < 1000:
                if state is not None:
                    actions, _, state, _ = model.step(obs,S=state, M=dones)
                else:
                    actions, _, _, _ = model.step(obs)

                obs, rew, done, _ = env.step(actions[0])
                episode_rew += rew
                env.render()
                done = done.any() if isinstance(done, np.ndarray) else done
                if done:
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