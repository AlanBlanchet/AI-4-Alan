# DS = DatasetDict | Dataset


# task_map = dict(
#     image=["classification", "detection"],
#     bboxes=["detection"],
#     classes=["classification", "detection"],
# )


# def get_compatible_tasks(dataset: DS):
#     if isinstance(dataset, DatasetDict):
#         key = list(dataset.keys())[0]
#         dataset = dataset[key]

#     return recursive_task_detect(dataset.features)


# def recursive_task_detect(feature: dict[str, Any]):
#     tasks: dict[str, CompatibilityOutput | dict] = {}

#     for key, value in feature.items():
#         if isinstance(value, Sequence):
#             feat = value.feature
#             if isinstance(feat, dict):
#                 tasks[key] = recursive_task_detect(feat)
#             else:
#                 valid_tasks = Task.get_all_valid_tasks(key, feat.dtype)
#                 tasks[key] = CompatibilityOutput(tasks=valid_tasks, value=feat)
#         else:
#             valid_tasks = Task.get_all_valid_tasks(key, value.dtype)
#             tasks[key] = CompatibilityOutput(tasks=valid_tasks, value=value)

#     return tasks
