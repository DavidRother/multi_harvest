from gym.envs.registration import register

register(id="gatheringEnv-v0",
         entry_point="gathering_zoo.environment:GymGatheringEnvironment")
register(id="gatheringZooEnv-v0",
         entry_point="gathering_zoo.environment:GatheringZooEnvironment")
