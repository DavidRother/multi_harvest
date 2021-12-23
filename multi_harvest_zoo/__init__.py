from gym.envs.registration import register

register(id="multiHarvestEnv-v0",
         entry_point="multi_harvest_zoo.environment:GymMultiHarvestEnvironment")
register(id="multiHarvestZooEnv-v0",
         entry_point="multi_harvest_zoo.environment:MultiHarvestZooEnvironment")
