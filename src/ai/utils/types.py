from typing import Literal, get_args

FORMATS_TYPE = Literal["project", "pytorch"]
FORMATS: list[FORMATS_TYPE] = list(get_args(FORMATS_TYPE))
