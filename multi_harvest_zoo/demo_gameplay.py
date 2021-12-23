from multi_harvest_zoo.environment.game.game import Game

from multi_harvest_zoo.environment import multi_harvest_zoo


n_agents = 1
num_humans = 1
max_steps = 100
render = False

level = 'deterministic_room'
reward_scheme = "scheme_1"
seed = 1
record = False

parallel_env = multi_harvest_zoo.parallel_env(level=level, num_agents=n_agents, record=record, max_steps=max_steps,
                                              reward_scheme=reward_scheme)

game = Game(parallel_env, num_humans, [], max_steps)
store = game.on_execute()

print("done")

