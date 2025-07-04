# from copy import deepcopy
# from pathlib import Path

# from .configs import Base
# from .dataset import Dataset, Datasets
# from .task.task import Task
# from .train.runner import Runner
# from .utils.env import AIEnv


# class Main(Base):
#     config: dict
#     datasets: Datasets
#     task: Task
#     run: Runner

#     @classmethod
#     def configure(self, config):
#         original = deepcopy(config["config"])
#         passed_config = deepcopy(config["config"])
#         config = config["config"]
#         config["config"] = original
#         # Datasets
#         datasets = Dataset.from_config(config["datasets"], passed_config)
#         if not isinstance(datasets, list):
#             datasets = [datasets]
#         config["datasets"] = Datasets(datasets=datasets)
#         # Task
#         config["task"]["datasets"] = config["datasets"]
#         config["task"] = Task.from_config(config["task"], passed_config)
#         # Runner
#         config["run"]["task"] = config["task"]
#         config["run"] = Runner(**config["run"])
#         return config

#     # @validator("datasets")
#     # def validate_datasets(cls, _, values):
#     #     config = deepcopy(values["config"])
#     #     datasets = Dataset.from_config(config["datasets"], config)
#     #     if not isinstance(datasets, list):
#     #         datasets = [datasets]
#     #     return Datasets(datasets=datasets)

#     # @validator("task")
#     # def validate_task(cls, _, values):
#     #     config = deepcopy(values["config"])
#     #     config["task"]["datasets"] = values["datasets"]
#     #     return Task.from_config(config["task"], config)

#     # @validator("run")
#     # def validate_run(cls, _, values):
#     #     config = deepcopy(values["config"])
#     #     config["run"]["task"] = values["task"]
#     #     return Runner(**config["run"])

#     def __call__(self):
#         # Save this root config in the instances so they can communicate
#         self.task._root_config = self
#         self.datasets._root_config = self
#         self.run._root_config = self
#         # Run the pipeline
#         self.run()


# def run_task(
#     config: Path,
#     extra_params={},
#     default_run=AIEnv.run_configs_p / "default.yml",
# ):
#     # Resolve config
#     config = AIEnv.resolve_config(
#         config, extra_params=extra_params, default_run=default_run
#     )

#     main = Main(config=config)
#     main()
