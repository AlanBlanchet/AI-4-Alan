import inspect
from copy import deepcopy

import rich
import rich.table
import yaml
from rich import console

from ..utils.paths import AIPaths


class Registry:
    def __init__(self, name: str):
        self.cache_p = AIPaths.cache_p / "registry"
        self.cache_p.mkdir(exist_ok=True, parents=True)
        self.cache_file_p = self.cache_p / f"{name}.yaml"
        self._name = name
        self._registry = {}
        self._load()

    def _load(self):
        if self.cache_file_p.exists():
            self._registry = yaml.safe_load(self.cache_file_p.read_text())

    @property
    def name(self):
        return self._name

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

    def _register(self, module):
        module_name = self._unique_name(module)
        self._registry[module_name] = {
            "module": module.__module__,
            "name": module.__name__,
            "children": {},
            "parents": set(),
        }
        return module

    def calculate_index(self):
        self._resolve_index()
        registry = deepcopy(self._registry)
        for item in registry.values():
            # item["children"] = list(item["children"])
            item["parents"] = list(item["parents"])
        self.cache_file_p.write_text(yaml.dump(registry))
        return self

    def register(self, module=None):
        if module is not None:
            return self._register(module)

        def _register(module):
            self._register(module)
            return module

        return _register

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
        return self._get_by_name(name)

    def _get_by_name(self, name):
        for item in self._registry.values():
            if item["name"] == name:
                return item
        return None

    def _get_resolved(self):
        parents = filter(
            lambda k: len(self._registry[k]["parents"]) == 0, self._registry
        )
        return {k: self._registry[k] for k in parents}

    def __repr__(self):
        resolved = self._get_resolved()
        table = rich.table.Table(
            f"{self._name}", "children", title=f"Registry {self.name}"
        )

        for item in resolved.values():
            table.add_row(
                item["name"],
                ", ".join(
                    [self._registry[child]["name"] for child in item["children"]]
                ),
            )

        cls = console.Console(record=True)
        with cls.capture() as capture:
            cls.print(table)
        return capture.get()

    def __str__(self):
        return self.__repr__()
