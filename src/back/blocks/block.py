from abc import abstractmethod


class Block:
    def __init__(self, color: str = "red"):
        self.color = color

    @abstractmethod
    def list():
        raise NotImplementedError
