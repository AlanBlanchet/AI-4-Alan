from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

from torch.utils.data import Dataset

from ..configs.base import Base

if TYPE_CHECKING:
    from ..task.classification.label_map import LabelMap


class BaseTorchDataset(Dataset): ...


class BaseDataset(Base):
    log_name: ClassVar[str] = "dataset"

    @property
    def name() -> str: ...

    @abstractmethod
    def train(self) -> BaseTorchDataset: ...

    def val(self) -> BaseTorchDataset:
        raise NotImplementedError

    def test(self) -> BaseTorchDataset:
        raise NotImplementedError

    @cached_property
    def _labels(self) -> list[str]: ...

    # TODO Refactor
    @property
    def compatible_tasks(self) -> dict[str, Any]: ...

    @abstractmethod
    def prepare(self, label_map: LabelMap): ...

    # TODO Refactor
    # def compatible_tasks_flat(self) -> list[tuple[str, list[Task]]]:
    #     return self._compatible_tasks_flat_recursive(self.compatible_tasks)

    # TODO Refactor
    # def _compatible_tasks_flat_recursive(self, tasks: dict[str, Any]):
    #     for key, value in tasks.items():
    #         if isinstance(value, dict):
    #             flat = self._compatible_tasks_flat_recursive(value)
    #             for k, v in flat:
    #                 yield f"{key}.{k}", v
    #         else:
    #             yield key, value

    # TODO Refactor
    # def print_compatible_tasks(self):
    #     table = Table(
    #         "Dataset keys",
    #         "Compatible tasks",
    #         title=f"Task compatibility for {self.source}'s {self.name} dataset",
    #     )

    #     for key, tasks in self.compatible_tasks_flat():
    #         table.add_row(key, ", ".join([task.name for task in tasks]))

    #     cls = console.Console(record=True)
    #     with cls.capture() as capture:
    #         cls.print(table)
    #     self.log("\n", capture.get())

    def example(self):
        return next(self.val().__iter__())


# class CompatibilityOutput(BaseModel):
#     tasks: list[Task]
#     value: Any
