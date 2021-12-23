from multi_harvest_zoo.environment.game.game import Game

from multi_harvest_zoo.environment import multi_harvest_zoo

n_agents = 2
num_humans = 1
max_steps = 100
render = False

level = 'deterministic_room'
reward_scheme = "scheme_1"
seed = 1
record = False

parallel_env = multi_harvest_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_steps,
                                              reward_scheme=reward_scheme)

action_spaces = parallel_env.action_spaces
player_2_action_space = action_spaces["player_1"]


class GatheringAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation) -> int:
        return self.action_space.sample()


gathering_agent = GatheringAgent(player_2_action_space)

game = Game(parallel_env, num_humans, [gathering_agent], max_steps)
store = game.on_execute()

print("done")
