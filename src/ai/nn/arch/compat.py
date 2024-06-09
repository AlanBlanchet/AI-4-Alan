from abc import ABC, abstractmethod
from typing import AnyStr, Literal, Union


class ISSD(ABC):
    """
    Layer definition for the SSD detection boxes
    """

    @abstractmethod
    def ssd_compat(
        self,
    ) -> dict[Union[Literal["features"], AnyStr], list[int]]: ...
