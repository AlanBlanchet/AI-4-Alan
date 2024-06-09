from ..registry.registers import MODEL
from .block import Block


class Model(Block):
    def __init__(self):
        super().__init__((120, 40, 180))

    def list():
        return MODEL
