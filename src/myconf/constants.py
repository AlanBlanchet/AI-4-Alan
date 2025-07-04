"""Constants used throughout MyConf"""

# Metaclass definition constants
MANUAL_DEFINITIONS = [
    "__module__",
    "__qualname__",
    "__firstlineno__",
    "__annotations__",
    "__static_attributes__",
]

MANUAL_BASE_DEFINITIONS = [
    "__init__",
    "__setattr__",
    "__str__",
    "__repr__",
    "__classcell__",
    "__init_subclass__",
]

# MyConf special attributes
MYCONF_PROPERTIES_ATTR = "_myconf_properties"
MYCONF_METHODS_ATTR = "_myconf_methods"
MYCONF_ANNOTATIONS_ATTR = "_myconf_annotations"
PARSERS_ATTR = "_parsers"

# Parameter inspection constants
SELF_PARAM = "self"
VAR_POSITIONAL = "VAR_POSITIONAL"
VAR_KEYWORD = "VAR_KEYWORD"

# Common type names for error messages
TYPE_ERROR_MSG = "Could not convert {value} to {cls}"
MISSING_ANNOTATION_MSG = "Missing type annotation for property '{prop}'. Please add a type annotation to help MyConf parse and validate the type."

# File extensions and formats
JSON_INDENT = 2
DEFAULT_FMT = 5
