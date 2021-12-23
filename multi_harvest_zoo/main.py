import gym

level = "open_room"
seed = 1
record = False
max_steps = 100
reward_scheme = "scheme_1"
env = gym.envs.make("multi_harvest_zoo:multiHarvestEnv-v0", level=level, record=record, max_steps=max_steps,
                    reward_scheme=reward_scheme)

obs = env.reset()

action_space = env.action_space

done = False

while not done:

    action = action_space.sample()
    observation, reward, done, info = env.step(action)

print('done')
