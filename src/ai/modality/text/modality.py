from typing import Any

from ...utils.types import is_list
from ..modality import Modality


class Text(Modality):
    def item_accept(self, key: str, value: Any):
        if is_list(value):
            return self.item_accept(key, value[0])
        return isinstance(value, str)
