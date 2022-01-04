from multi_harvest_zoo.multi_harvest_world.abstract_classes import *
from typing import List
from collections import defaultdict


class Floor(StaticObject):

    def __init__(self, location):
        super().__init__(location, True)

    def accepts(self, dynamic_objects) -> bool:
        return True

    def file_name(self) -> str:
        return "floor"


class Counter(StaticObject):

    def __init__(self, location):
        super().__init__(location, False)

    def accepts(self, dynamic_objects) -> bool:
        return False

    def file_name(self) -> str:
        return "counter"


class Crop(DynamicObject):

    def __init__(self, location, color, age_threshold=5):
        super().__init__(location)
        self.color = color
        self.age = 0
        self.age_threshold = age_threshold

    def file_name(self) -> str:
        age = "young" if self.age < self.age_threshold else "mature"
        return f"crop_{age}_{self.color}"


class Agent(Object):

    def __init__(self, location, color, name):
        super().__init__(location, False, False)
        self.color = color
        self.name = name
        self.orientation = 1
        self.freeze_timer = 0

    def move_to(self, new_location):
        self.location = new_location

    def change_orientation(self, new_orientation):
        self.orientation = new_orientation

    def file_name(self) -> str:
        pass


StringToClass = {
    "Floor": Floor,
    "Counter": Counter,
    "Crop": Crop
}

ClassToString = {
    Floor: "Floor",
    Counter: "Counter",
    Crop: "Crop"
}

GAME_CLASSES = [Floor, Counter, Agent, Crop]
GAME_CLASSES_STATE_LENGTH = [(Crop, 4), (Agent, 5)]
