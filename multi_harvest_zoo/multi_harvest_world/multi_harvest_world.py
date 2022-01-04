from collections import defaultdict
from multi_harvest_zoo.multi_harvest_world.world_objects import *
from multi_harvest_zoo.multi_harvest_world.actions import *

from pathlib import Path
import os.path
import json
import random


class MultiHarvestWorld:

    COLORS = ['blue', 'magenta', 'yellow', 'green']

    SymbolToClass = {
        ' ': Floor,
        'x': Crop,
        'o': Agent
    }

    # AGENT_ACTIONS: 0: Noop, 1: Left, 2: right, 3: down, 4: up, 5: interact

    def __init__(self, n_freeze=3, n_mature=5):
        self.agents = []
        self.width = 0
        self.height = 0
        self.world_objects = defaultdict(list)
        self.num_colors = 0
        self.crops_bags = defaultdict(list)
        self.last_collected = {}
        self.marked_squares = []
        self.n_freeze = n_freeze
        self.n_mature = n_mature

    def add_object(self, obj):
        self.world_objects[type(obj).__name__].append(obj)

    def delete_object(self, obj):
        self.world_objects[type(obj).__name__].remove(obj)

    def get_object_list(self):
        object_list = []
        for value in self.world_objects.values():
            object_list.extend(value)
        return object_list

    def perform_agent_actions(self, agents, actions):
        for idx, agent in enumerate(agents):
            if agent.freeze_timer > 0:
                actions[idx] = NO_OP
        for agent, action in zip(agents, actions):
            if action in WALK_ACTIONS:
                agent.change_orientation(action)
        self.last_collected = {}
        self.marked_squares = []
        cleaned_actions = self.check_inbounds(agents, actions)
        collision_actions = self.check_collisions(agents, cleaned_actions)
        for agent, action in zip(agents, collision_actions):
            self.perform_agent_action(agent, action)
        self.resolve_after_effects()

    def perform_agent_action(self, agent: Agent, action):
        if action in WALK_ACTIONS:
            self.resolve_walking_action(agent, action)
        if action is SHOOT:
            self.resolve_shoot_action(agent)
        # if action in INTERACT_ACTIONS:
        #     self.resolve_interaction(agent, action)

    def resolve_after_effects(self):
        for location in self.marked_squares:
            for agent in self.agents:
                if agent.location == location and agent.freeze_timer == 0:
                    agent.freeze_timer = self.n_freeze
        for agent in self.agents:
            agent.freeze_timer = max(agent.freeze_timer - 1, 0)
        for crop in self.world_objects["Crop"]:
            crop.age += 1
        if self.last_collected:
            self.respawn_fresh_crop()

    def respawn_fresh_crop(self):
        self.world_objects["Crop"] = []
        for idx in range(0, 2):
            collision = True
            location = (0, 0)
            while collision:
                location = (random.sample(list(range(0, self.width)), 1)[0],
                            random.sample(list(range(0, self.height)), 1)[0])
                collision = any([agent.location == location for agent in self.agents])
                collision = collision or any([crop.location == location for crop in self.world_objects["Crop"]])
            self.world_objects["Crop"].append(Crop(location, idx, age_threshold=self.n_mature))

    def resolve_shoot_action(self, agent: Agent):
        if agent.orientation == 1:  # left
            self.marked_squares = [(loc, agent.location[1]) for loc in range(0, agent.location[0])]
        if agent.orientation == 2:  # right
            self.marked_squares = [(loc, agent.location[1]) for loc in range(agent.location[0] + 1, self.width)]
        if agent.orientation == 3:  # down
            self.marked_squares = [(agent.location[0], loc) for loc in range(agent.location[1] + 1, self.height)]
        if agent.orientation == 4:  # up
            self.marked_squares = [(agent.location[0], loc) for loc in range(0, agent.location[1])]

    def resolve_walking_action(self, agent: Agent, action):
        target_location = self.get_target_location(agent, action)
        if self.square_walkable(target_location):
            agent.move_to(target_location)
            crop = self.get_crop(agent.location)
            if crop:
                self.last_collected[agent] = crop
                self.world_objects["Crop"].remove(crop)
                self.crops_bags[agent].append(crop)

    def get_crop(self, position):
        crops = [crop for crop in self.world_objects["Crop"] if crop.location == position]
        if crops:
            return crops[0]
        else:
            return None

    def check_inbounds(self, agents, actions):
        cleaned_actions = []
        for agent, action in zip(agents, actions):
            if action == NO_OP or action == SHOOT:
                cleaned_actions.append(action)
                continue
            target_location = self.get_target_location(agent, action)
            if target_location[0] > self.width - 1 or target_location[0] < 0:
                action = 0
            if target_location[1] > self.height - 1 or target_location[1] < 0:
                action = 0
            cleaned_actions.append(action)
        return cleaned_actions

    def check_collisions(self, agents, actions):
        collision_actions = []
        target_locations = []
        walkable = []
        for agent, action in zip(agents, actions):
            target_location = self.get_target_location(agent, action)
            target_walkable = self.square_walkable(target_location)
            end_location = target_location if target_walkable else agent.location
            target_locations.append(end_location)
            walkable.append(target_walkable)
        for idx, (action, target_location, target_walkable) in enumerate(zip(actions, target_locations, walkable)):
            if target_location in target_locations[:idx] + target_locations[idx+1:] and target_walkable:
                collision_actions.append(0)
            else:
                collision_actions.append(action)
        return collision_actions

    @staticmethod
    def get_target_location(agent, action):
        if action == 1:
            target_location = (agent.location[0] - 1, agent.location[1])
        elif action == 2:
            target_location = (agent.location[0] + 1, agent.location[1])
        elif action == 3:
            target_location = (agent.location[0], agent.location[1] + 1)
        elif action == 4:
            target_location = (agent.location[0], agent.location[1] - 1)
        else:
            target_location = (agent.location[0], agent.location[1])
        return target_location

    def square_walkable(self, location):
        objects = self.get_objects_at(location, StaticObject)
        if len(objects) != 1:
            raise Exception(f"Not exactly one static object at location: {location}")
        return objects[0].walkable

    def get_objects_at(self, location, object_type=object):
        located_objects = []
        for obj_class_string, objects in self.world_objects.items():
            obj_class = StringToClass[obj_class_string]
            if not issubclass(obj_class, object_type):
                continue
            for obj in objects:
                if obj.location == location:
                    located_objects.append(obj)
        return located_objects

    def load_new_style_level(self, level_name, num_agents):
        my_path = os.path.realpath(__file__)
        dir_name = os.path.dirname(my_path)
        path = Path(dir_name)
        parent = path.parent / f"utils/level/{level_name}.json"
        with open(parent) as json_file:
            level_object = json.load(json_file)
            json_file.close()
        self.num_colors = level_object["NUMBER_COLORS"]
        self.parse_level_layout(level_object)
        self.parse_static_objects(level_object)
        self.parse_dynamic_objects(level_object)
        self.parse_agents(level_object, num_agents)

    def parse_level_layout(self, level_object):
        level_layout = level_object["LEVEL_LAYOUT"]
        x = 0
        y = 0
        for y, line in enumerate(iter(level_layout.splitlines())):
            for x, char in enumerate(line):
                if char == "-":
                    counter = Counter(location=(x, y))
                    self.add_object(counter)
                else:
                    floor = Floor(location=(x, y))
                    self.add_object(floor)
        self.width = x + 1
        self.height = y + 1

    def parse_static_objects(self, level_object):
        static_objects = level_object["STATIC_OBJECTS"]
        for static_object in static_objects:
            name = list(static_object.keys())[0]
            for idx in range(static_object[name]["COUNT"]):
                time_out = 0
                while True:
                    x = random.sample(static_object[name]["X_POSITION"], 1)[0]
                    y = random.sample(static_object[name]["Y_POSITION"], 1)[0]
                    if x < 0 or y < 0 or x > self.width or y > self.height:
                        raise ValueError(f"Position {x} {y} of object {name} is out of bounds set by the level layout!")
                    static_objects_loc = self.get_objects_at((x, y), StaticObject)

                    counter = [obj for obj in static_objects_loc if isinstance(obj, (Counter, Floor))]
                    if counter:
                        if len(counter) != 1:
                            raise ValueError("Too many counter in one place detected during initialization")
                        self.delete_object(counter[0])
                        obj = StringToClass[name](location=(x, y))
                        self.add_object(obj)
                        break
                    else:
                        time_out += 1
                        if time_out > 100:
                            raise ValueError(f"Can't find valid position for object: "
                                             f"{static_object} in {time_out} steps")
                        continue

    def parse_dynamic_objects(self, level_object):
        dynamic_objects = level_object["DYNAMIC_OBJECTS"]
        for dynamic_object in dynamic_objects:
            name = list(dynamic_object.keys())[0]
            for idx in range(dynamic_object[name]["COUNT"]):
                time_out = 0
                while True:
                    x = random.sample(dynamic_object[name]["X_POSITION"], 1)[0]
                    y = random.sample(dynamic_object[name]["Y_POSITION"], 1)[0]
                    color = dynamic_object[name]["COLOR"]
                    if x < 0 or y < 0 or x > self.width or y > self.height:
                        raise ValueError(f"Position {x} {y} of object {name} is out of bounds set by the level layout!")
                    dynamic_objects_loc = self.get_objects_at((x, y), DynamicObject)

                    if not dynamic_objects_loc:
                        obj = StringToClass[name](location=(x, y), color=color, age_threshold=self.n_mature)
                        self.add_object(obj)
                        break
                    else:
                        time_out += 1
                        if time_out > 1000:
                            raise ValueError(f"Can't find valid position for object: "
                                             f"{dynamic_object} in {time_out} steps")
                        continue

    def parse_agents(self, level_object, num_agents):
        agent_objects = level_object["AGENTS"]
        agent_idx = 0
        for agent_object in agent_objects:
            for idx in range(agent_object["MAX_COUNT"]):
                agent_idx += 1
                if agent_idx > num_agents:
                    return
                time_out = 0
                while True:
                    x = random.sample(agent_object["X_POSITION"], 1)[0]
                    y = random.sample(agent_object["Y_POSITION"], 1)[0]
                    if x < 0 or y < 0 or x > self.width or y > self.height:
                        raise ValueError(f"Position {x} {y} of agent is out of bounds set by the level layout!")
                    static_objects_loc = self.get_objects_at((x, y), Floor)
                    if not any([(x, y) == agent.location for agent in self.agents]) and static_objects_loc:
                        agent = Agent((int(x), int(y)), self.COLORS[len(self.agents)],
                                      'agent-' + str(len(self.agents) + 1))
                        self.agents.append(agent)
                        break
                    else:
                        time_out += 1
                        if time_out > 100:
                            raise ValueError(f"Can't find valid position for agent: {agent_object} in {time_out} steps")

    def load_level(self, level, num_agents):
        self.load_new_style_level(level, num_agents)
