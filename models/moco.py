# although we have the original CLAIRE-code, we can use the MoCo-code from lightly as a comparison.
import copy

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule

from utils.train_utils import *
from models.model_utils import get_backbone

import lightning as pl


class MoCo(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, out_dim, memory_bank_size, max_epochs=50):
        super().__init__()
        self.max_epochs = max_epochs
        self.backbone = get_backbone(in_dim, hidden_dim)
        self.projection_head = MoCoProjectionHead(hidden_dim, hidden_dim, out_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=(memory_bank_size, out_dim))

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query
    
    def predict(self, x):
        with torch.no_grad():
            return self(x)

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        return loss
    
    def configure_optimizers(self):
        return optimizer_builder(self)

"""    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""