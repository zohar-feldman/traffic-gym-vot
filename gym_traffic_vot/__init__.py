from gym.envs.registration import register

register(
    id='traffic-vot-simple-v0',
    entry_point='gym_traffic_vot.envs:TrafficVotEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli", 'vots':True},
    nondeterministic=True
)

register(
    id='traffic-flow-simple-v0',
    entry_point='gym_traffic_vot.envs:TrafficVotEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli"},
    nondeterministic=True
)

register(
    id='traffic-vot-simple-gui-v0',
    entry_point='gym_traffic_vot.envs:TrafficVotEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui"},
    nondeterministic=True
)

register(
    id='traffic-social-simple-v0',
    entry_point='gym_traffic_vot.envs:TrafficSocialEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli"},
    nondeterministic=True
)