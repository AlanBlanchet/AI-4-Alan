from ....configs.main import BackboneConfig
from ....configs.models import ClassificationConfig, PretrainedConfig


class DETRConfig(PretrainedConfig, ClassificationConfig):
    backbone: BackboneConfig
    hidden_dim: int = 256
    num_queries: int = 100
