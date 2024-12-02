from abc import abstractmethod

from .pretrained import Pretrained


class Backbone(Pretrained):
    @abstractmethod
    def features(self): ...
