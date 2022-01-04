# Other core modules
import copy

from multi_harvest_zoo.multi_harvest_world.multi_harvest_world import MultiHarvestWorld
from multi_harvest_zoo.multi_harvest_world.actions import ACTIONS
from multi_harvest_zoo.multi_harvest_world.world_objects import *


import numpy as np
from collections import namedtuple, defaultdict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from pathlib import Path
import os.path
import gym
import json


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")
COLORS = ['blue', 'magenta', 'yellow', 'green']


def env(level, num_agents, record, max_steps, reward_scheme, obs_spaces=None, n_freeze=3, n_mature=5):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = MultiHarvestEnvironment(level, num_agents, record, max_steps, reward_scheme, obs_spaces, n_freeze,
                                       n_mature)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class MultiHarvestEnvironment(AECEnv):
    """Environment object for Overcooked."""

    metadata = {'render.modes': ['human'], 'name': "cooking_zoo"}

    def __init__(self, level, num_agents, record, max_steps, reward_scheme, obs_spaces=None, n_freeze=3, n_mature=5):
        super().__init__()

        obs_spaces = obs_spaces or ["numeric"]
        self.allowed_obs_spaces = ["symbolic", "numeric"]
        assert len(set(obs_spaces + self.allowed_obs_spaces)) == 2, \
            f"Selected invalid obs spaces. Allowed {self.allowed_obs_spaces}"
        assert len(obs_spaces) != 0, f"Please select an observation space from: {self.allowed_obs_spaces}"
        self.obs_spaces = obs_spaces
        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]

        self.level = level
        self.record = record
        self.max_steps = max_steps
        self.t = 0
        self.filename = ""
        self.set_filename()
        self.world = MultiHarvestWorld(n_freeze, n_mature)
        self.game = None

        self.termination_info = ""
        self.world.load_level(level=self.level, num_agents=num_agents)
        self.graph_representation_length = sum([tup[1] for tup in GAME_CLASSES_STATE_LENGTH]) + 2 * self.num_agents

        numeric_obs_space = {"tensor": gym.spaces.Box(low=0, high=10, shape=(self.world.width, self.world.height,
                                                                             self.graph_representation_length),
                                                      dtype=np.int32)}
        self.observation_spaces = {agent: gym.spaces.Dict(numeric_obs_space) for agent in self.possible_agents}
        self.action_spaces = {agent: gym.spaces.Discrete(len(ACTIONS)) for agent in self.possible_agents}
        self.has_reset = True

        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))
        self.scheme_name = reward_scheme
        self.reward_type = "default"
        self.collectors_bonus = 0
        self.reward_scheme = self.load_reward_scheme(self.scheme_name)

    def set_filename(self):
        self.filename = f"{self.level}_agents{self.num_agents}"

    def state(self):
        pass

    def reset(self):
        self.world = MultiHarvestWorld()
        self.t = 0

        # For tracking data during an episode.
        self.termination_info = ""

        # Load world & distances.
        self.world.load_level(level=self.level, num_agents=self.num_agents)

        # if self.record:
        #     self.game = GameImage(
        #         filename=self.filename,
        #         world=self.world,
        #         record=self.record)
        #     self.game.on_init()
        #     self.game.save_image_obs(self.t)
        # else:
        #     self.game = None

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))

        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.reward_scheme = self.load_reward_scheme(self.scheme_name)

    def close(self):
        return

    def step(self, action):
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        # translated_actions = [action_translation_dict[actions[f"player_{idx}"]] for idx in range(len(actions))]
        self.world.perform_agent_actions(self.world.agents, actions)

        # Visualize.
        if self.record:
            self.game.on_render()

        if self.record:
            self.game.save_image_obs(self.t)

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        for agent in self.agents:
            self.current_tensor_observation[agent] = self.get_tensor_representation(agent)

        done, rewards, goals = self.compute_rewards()

        info = {"t": self.t, "termination_info": self.termination_info}

        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done
            self.rewards[agent] = rewards[idx]
            info["rewards_others"] = rewards[:idx] + rewards[idx+1:]
            self.infos[agent] = info

    def observe(self, agent):
        observation = []
        if "numeric" in self.obs_spaces:
            num_observation = {'tensor': self.current_tensor_observation[agent]}
            observation.append(num_observation)
        if "symbolic" in self.obs_spaces:
            objects = defaultdict(list)
            objects.update(self.world.world_objects)
            objects["Agent"] = self.world.agents
            sym_observation = copy.deepcopy(objects)
            observation.append(sym_observation)
        returned_observation = observation if not len(observation) == 1 else observation[0]
        return returned_observation

    def compute_rewards(self):
        done = False
        rewards = [0] * self.num_agents
        open_goals = [[0]] * self.num_agents
        # Done if the episode maxes out
        if self.t >= self.max_steps and self.max_steps:
            self.termination_info = f"Terminating because passed {self.max_steps} timesteps"
            done = True

        for idx, agent in enumerate(self.world.agents):
            crops_collected = len(self.world.last_collected)
            if self.reward_type == "joint":
                for collector in self.world.agents:
                    if collector in self.world.last_collected:
                        crop = self.world.last_collected[collector]
                        if crop.age < crop.age_threshold:
                            bonus = self.collectors_bonus if collector == agent else 0
                            rewards[idx] += self.reward_scheme[agent]["young"][crop.color] + bonus
                        else:
                            rewards[idx] += self.reward_scheme[agent]["mature"][crop.color]
            else:
                if agent in self.world.last_collected:
                    crop = self.world.last_collected[agent]
                    if crop.age < crop.age_threshold:
                        rewards[idx] += self.reward_scheme[agent]["young"][crop.color] + self.collectors_bonus
                    else:
                        rewards[idx] += self.reward_scheme[agent]["mature"][crop.color]
            rewards[idx] = rewards[idx] / max(crops_collected**2, 1)

        if not done:
            done = True
            for crop in self.world.world_objects["Crop"]:
                for agent in self.world.agents:
                    if self.reward_scheme[agent]["young"][crop.color] > 0 or \
                            self.reward_scheme[agent]["mature"][crop.color] > 0:
                        done = False

        return done, rewards, open_goals

    def get_tensor_representation(self, agent):
        tensor = np.zeros(
            (self.world.width, self.world.height, self.graph_representation_length))
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        idx = 0
        for color in range(2):
            for crop in self.world.world_objects["Crop"]:
                if crop.color == color:
                    x, y = crop.location
                    if crop.age < crop.age_threshold:
                        tensor[x, y, idx] = 1
                    else:
                        tensor[x, y, idx + 1] = 1
            idx += 2

        ego_agent = self.world_agent_mapping[agent]
        x, y = ego_agent.location
        # location map for all agents, location maps for separate agent and four orientation maps shared
        # between all agents
        tensor[x, y, idx] = 1
        tensor[x, y, idx + 1] = 1
        tensor[x, y, idx + 2] = 1 if ego_agent.freeze_timer else 0
        tensor[x, y, idx + self.num_agents * 2 + ego_agent.orientation] = 1

        agent_idx = 1
        for world_agent in self.world.agents:
            if agent != world_agent:
                x, y = world_agent.location
                # location map for all agents, location maps for separate agent and four orientation maps shared
                # between all agents
                tensor[x, y, idx] = 1
                tensor[x, y, idx + 2 * agent_idx + 1] = 1
                tensor[x, y, idx + 2 * agent_idx + 2] = 1 if world_agent.freeze_timer else 0
                tensor[x, y, idx + self.num_agents * 2 + world_agent.orientation] = 1
                agent_idx += 1

        return tensor

    def load_reward_scheme(self, scheme_name):
        my_path = os.path.realpath(__file__)
        dir_name = os.path.dirname(my_path)
        path = Path(dir_name)
        parent = path.parent / f"utils/reward_schemes/{scheme_name}.json"
        with open(parent) as json_file:
            reward_object = json.load(json_file)
            json_file.close()
        assert len(reward_object) >= len(self.world.agents)
        reward_scheme = {}
        for agent, scheme in zip(self.world.agents, reward_object["allocation"]):
            reward_scheme[agent] = scheme
        self.reward_type = reward_object["type"]
        self.collectors_bonus = reward_object["collectors_bonus"]
        return reward_scheme

    def get_agent_names(self):
        return [agent.name for agent in self.world.agents]

    def render(self, mode='human'):
        pass
