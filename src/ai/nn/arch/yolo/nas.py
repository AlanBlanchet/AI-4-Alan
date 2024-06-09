import torch.nn as nn


class YoloNAS(nn.Module):
    """
    YOLO Neural Architecture Search

    Using AutoNAC (Automated Neural Architecture Construction)
    """

    def __init__(self):
        super().__init__()
