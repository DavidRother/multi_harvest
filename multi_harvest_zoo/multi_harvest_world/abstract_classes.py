from abc import abstractmethod, ABC


class Object(ABC):

    def __init__(self, location, movable, walkable):
        self.location = location
        self.movable = movable  # you can pick this one up
        self.walkable = walkable  # you can walk on it

    def name(self) -> str:
        return type(self).__name__

    def move_to(self, new_location):
        self.location = new_location

    @abstractmethod
    def file_name(self) -> str:
        pass


class StaticObject(Object):

    def __init__(self, location, walkable):
        super().__init__(location, False, walkable)

    def move_to(self, new_location):
        raise Exception(f"Can't move static object {self.name()}")

    @abstractmethod
    def accepts(self, dynamic_objects) -> bool:
        pass


class DynamicObject(Object, ABC):

    def __init__(self, location):
        super().__init__(location, True, True)


ABSTRACT_GAME_CLASSES = (DynamicObject, StaticObject)
