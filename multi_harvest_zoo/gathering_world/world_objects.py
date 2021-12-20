from gathering_zoo.gathering_world.abstract_classes import *
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


class Chip(DynamicObject):

    def __init__(self, location, color):
        super().__init__(location)
        self.color = color

    def file_name(self) -> str:
        return f"chip_{self.color}"


class Agent(Object):

    def __init__(self, location, color, name):
        super().__init__(location, False, False)
        self.color = color
        self.name = name
        self.orientation = 0

    def move_to(self, new_location):
        self.location = new_location

    def file_name(self) -> str:
        pass


StringToClass = {
    "Floor": Floor,
    "Counter": Counter,
    "Chip": Chip
}

ClassToString = {
    Floor: "Floor",
    Counter: "Counter",
    Chip: "Chip"
}


