import inspect
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import ClassVar

import rich
import rich.table
import yaml
from pydantic import BaseModel
from rich import console

from ..utils.env import AIEnv


class Registry(BaseModel):
    cache_p: ClassVar[Path] = AIEnv.cache_p / "registry"
    name: str
    root: str | list[str]
    _registry = {}

    def model_post_init(self, _):
        self._load()

    def _load(self):
        if self.cache_file_p.exists():
            self._registry = yaml.safe_load(self.cache_file_p.read_text())

    @property
    def cache_file_p(self):
        return Registry.cache_p / f"{self.name}.yaml"

    @property
    def names(self):
        return list(map(lambda x: x["name"], self._registry.values()))

    def _unique_name(self, module):
        return module.__module__ + "." + module.__name__

    def _clear(self):
        for item in self._registry.values():
            item["children"] = {}

    def _resolve_index(self):
        self._clear()
        registry = deepcopy(self._registry)
        for module_name, item in self._registry.items():
            module = __import__(item["module"], fromlist=[item["name"]])
            if item["name"] not in module.__dict__:
                print(f"Removing old module {module_name}")
                del registry[module_name]
                continue

            cls = module.__dict__[item["name"]]
            bases = inspect.getmro(cls)[1:]

            for base in bases:
                base_name = self._unique_name(base)
                if base_name in self._registry:
                    registry[base_name]["children"].update({module_name: len(bases)})
                    registry[module_name]["parents"].add(base_name)
        for item in registry.values():
            children: set = item["children"]
            item["children"] = dict(sorted(children.items(), key=lambda x: x[1]))
        self._registry = registry

    def _register(self, module, **kwargs: dict):
        module_name = self._unique_name(module)
        self._registry[module_name] = dict(
            module=module.__module__,
            name=module.__name__,
            children={},
            parents=set(),
            meta=dict(**kwargs),
        )
        return module

    def _scope(self):
        python_files = AIEnv.ai_p.rglob("*.py")
        for file in python_files:
            if file.stem == "__init__":
                continue

            module = AIEnv.path2module(file)
            try:
                import_module(module)
            except Exception as e:
                raise ValueError(f"Error importing {module}") from e
        # # Get all root modules into scope
        # if isinstance(self.root, list):
        #     for root in self.root:
        #         import_module(root)
        # else:
        #     import_module(self.root)

    def calculate_index(self):
        self._scope()
        # Resolve their index
        self._resolve_index()
        registry = deepcopy(self._registry)
        for item in registry.values():
            item["parents"] = list(item["parents"])
        self.cache_file_p.write_text(yaml.dump(registry))
        return self

    def register(self, module=None, **kwargs: dict):
        if module is not None:
            return self._register(module)

        def _register(module):
            self._register(module, **kwargs)
            return module

        return _register

    def __call__(self, module=None, **kwargs: dict):
        return self.register(module, **kwargs)

    def get(self, name):
        return self._registry[name]

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __contains__(self, name):
        return name in self._registry

    def __delitem__(self, name):
        del self._registry[name]

    def __getitem__(self, name):
        item = self._get_by_name(name)

        if item is None:
            raise KeyError(f"Module {name} not found in registry")

        module = item["module"]

        # Get the Module from the name
        return getattr(import_module(module), name)

    def get_info(self, name):
        return self._get_by_name(name)

    def _get_by_name(self, name):
        name = name.lower()
        for item in self._registry.values():
            if item["name"].lower() == name:
                return item
        return None

    def _get_resolved(self):
        parents = filter(
            lambda k: len(self._registry[k]["parents"]) == 0, self._registry
        )
        return {k: self._registry[k] for k in parents}

    def __repr__(self):
        resolved = self._get_resolved()
        headers = [f"{self.name.lower().capitalize()}"]

        if any([len(item["children"]) > 0 for item in resolved.values()]):
            headers.append("children")

        table = rich.table.Table(
            *headers,
            title=f"Registry {self.name}",
        )

        for item in resolved.values():
            row = [item["name"]]

            if len(item["children"]) > 0:
                row.append(
                    ", ".join(
                        [self._registry[child]["name"] for child in item["children"]]
                    )
                )

            table.add_row(*row)

        cls = console.Console(record=True)
        with cls.capture() as capture:
            cls.print(table)
        return capture.get()

    def __str__(self):
        return self.__repr__()


Registry.cache_p.mkdir(exist_ok=True, parents=True)

REGISTER = Registry(name="REGISTER", root="ai")
