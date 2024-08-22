from abc import ABC, abstractmethod
from typing import AnyStr, Literal, Union


class ISSD(ABC):
    """
    Layer definition for the SSD detection boxes
    """

    @abstractmethod
    def ssd_compat(
        self,
    ) -> dict[Union[Literal["features"], AnyStr], list[int]]:
        """
        Make your model compatible with the SSD detection boxes
        """
        ...


class ITimm(ABC):
    """
    Layer definition for the Timm models
    """

    @abstractmethod
    def timm_compat(self) -> dict[str, AnyStr]:
        """
        Make your model compatible with the Timm models
        """
        ...
