import gym
import gym_traffic_vot
from gym_traffic_vot.tests import run
import numpy as np
from baselines import logger
import os.path as osp
import sys
from scipy.stats import sem, t
from scipy import mean

confidence = 0.95

# run.main(alg='deepq', env='traffic-vot-simple-v0', num_timesteps=2e6, print_freq=100, exploration_fraction=0.5, exploration_final_eps=0.1 ,save_path='C:/Users/FEZ1TV/PycharmProjects/gym-traffic-vot/models/test')



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = run.common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = run.parse_cmdline_kwargs(unknown_args)
    save_path = 'C:/Users/FEZ1TV/PycharmProjects/gym-traffic-vot/models/test'

    env_name = 'traffic-vot-simple-v0'

    env = gym.make(env_name, network='simple', arrival_rate=0.1, scale=4)
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
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            episode_rewards.append(episode_rew)
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            if 0 == len(episode_rewards) % 10:
                n = len(episode_rewards)
                m = mean(episode_rewards)
                std_err = sem(episode_rewards)
                h = std_err * t.ppf((1 + confidence) / 2, n - 1)
                print('Avg. reward of last {} episodes is [{:.2f}-{:.2f}]'.format(n, m - h, m + h))
                if h < abs(m * 0.01):
                    break
            obs = env.reset()

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)