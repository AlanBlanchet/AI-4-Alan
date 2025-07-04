from __future__ import annotations

from ..configs import Base
from ..nn.compat.module import Module


class Train(Base):
    model: Module


# class Runner(Base):
#     log_name: ClassVar[str] = "runner"
#     color: ClassVar[str] = "green"

#     # Shared
#     task: Task

#     lightning: dict = {}
#     checkpoint: Path = None

#     # @field_validator("checkpoint", mode="before")
#     # @classmethod
#     # def validate_checkpoint(cls, v, others):
#     #     # TODO change this
#     #     if isinstance(v, bool):
#     #         return others.data["task"].run_p / "checkpoints"
#     #     return v

#     @property
#     def dataset(self):
#         return self.task.datasets

#     @cached_property
#     def trainer(self) -> Trainer:
#         lightning = self.lightning

#         if hasattr(self.task, "logger"):
#             logger = self.task.logger
#             if logger:
#                 lightning["logger"] = [logger]

#         # Training configuration
#         # TODO add as config
#         torch.set_float32_matmul_precision("medium")

#         # profiler_p = self.task.run_p / "profiler"
#         # profiler_p.mkdir(exist_ok=True)
#         # profiler = PyTorchProfiler(
#         #     dirpath=profiler_p,
#         #     output_filename="profiler.json",
#         #     export_to_chrome=True,
#         # )

#         return Trainer(
#             **lightning,
#             default_root_dir=self.task.run_p,
#             callbacks=[CustomTQDMProgressBar()],
#             num_sanity_val_steps=0,
#             # profiler=profiler,
#         )

#     def __call__(self):
#         self.task.save_config()

#         self.info("Dataset example sample :\n", self.dataset.example)
#         self.info("Dataset :\n", self.dataset)

#         if self.checkpoint:
#             self.task.load(self.checkpoint)

#         if self.val_only:
#             self.info("Validating with pytorch-lightning")
#             self.trainer.validate(model=self.task, datamodule=self.task.datamodule)
#         else:
#             self.info("Training with pytorch-lightning")
#             self.trainer.fit(model=self.task, datamodule=self.task.datamodule)

#     @property
#     def val_only(self):
#         return self.action == ActionEnum.val

#     @field_validator("action", mode="before")
#     def _flexible_action(cls, v):
#         match v:
#             case "train":
#                 return ActionEnum.fit
#             case "eval":
#                 return ActionEnum.val
#             case _:
#                 return ActionEnum[v]
