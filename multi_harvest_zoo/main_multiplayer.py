from multi_harvest_zoo.environment import multi_harvest_zoo

env_config = {
    "level": 'deterministic_room',
    "num_agents": 2,
    "seed": 1,
    "record": False,
    "max_steps": 100,
    "reward_scheme": "scheme_1"
}


done = False

parallel_env = multi_harvest_zoo.parallel_env(level=env_config["level"], num_agents=env_config["num_agents"],
                                              record=env_config["record"], max_steps=env_config["max_steps"],
                                              reward_scheme=env_config["reward_scheme"])

observations = parallel_env.reset()

action_space = parallel_env.action_spaces


def policy(a, b):
    return action_space["player_0"].sample()


for step in range(env_config["max_steps"]):

    actions = {agent: policy(observations[agent], agent) for agent in parallel_env.agents}
    observations, rewards, dones, infos = parallel_env.step(actions)

print('done')
