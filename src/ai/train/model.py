import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule

from ..dataset.base_dataset import DetBaseDataset
from ..utils.anchor import AnchorManager
from ..utils.arch import get_arch_module


class AIModule(LightningModule):
    def __init__(self, model_name: str, dataset: DetBaseDataset):
        super().__init__()

        model_cls = get_arch_module(model_name)
        self.model: nn.Module = model_cls(num_classes=len(dataset.label_map))

        self.anchor_manager = AnchorManager(
            [38, 19, 10, 5, 3, 1],
            [4, 6, 6, 6, 4, 4],
            background_id=dataset.label_map["background"],
        )
        self.dataset = dataset

        self.conf_loss = nn.CrossEntropyLoss()
        self.loc_loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        image = batch["image"]
        objects = batch["objects"]
        mask = objects["mask"]
        gt_locs = objects["bbox"]
        label_ids = objects["category"]

        # Forward pass
        out = self.model(image)
        locs = self.anchor_manager.decode(out["location"])
        scores = out["confidence"]

        # Loss preperation
        gt_locs, gt_labels, pos_mask = self.anchor_manager.encode(
            gt_locs, label_ids, mask
        )

        # Flatten tensors for loss calculation
        locs_f = locs.view(-1, 4)
        scores_f = scores.view(-1, scores.shape[-1])
        gt_locs_f = gt_locs.view(-1, 4)
        gt_labels_f = gt_labels.view(-1)
        pos_mask = pos_mask.view(-1)

        # Detection loss
        loc_loss = self.loc_loss(locs_f[pos_mask], gt_locs_f[pos_mask])
        conf_loss = self.conf_loss(scores_f[pos_mask], gt_labels_f[pos_mask])

        loss = loc_loss + conf_loss

        self.log("loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, batch): ...

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=1e-3)
