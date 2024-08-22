from .registry import Registry

SOURCE = Registry(name="SOURCE", root=["ai.dataset", "ai.task", "ai.nn"])

MODEL = Registry(name="MODEL", root="ai.nn.arch")

REGISTERS = [MODEL, SOURCE]
