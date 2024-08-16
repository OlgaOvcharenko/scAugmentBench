import torch
import torchvision
from torch import nn

from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone

import lightning as pl


class VICReg(pl.LightningModule):
    
    def __init__(self, in_dim, hidden_dim, out_dim, reg_lambda=1.0, reg_alpha=0.3, reg_beta=1.0, **kwargs):
        super().__init__()
        self.backbone = get_backbone(in_dim, hidden_dim)
        self.projection_head = VICRegProjectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
            num_layers=2,
        )
        self.criterion = VICRegLoss(reg_lambda, reg_alpha, reg_beta)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
    
    def predict(self, x):
        with torch.no_grad():
            return self(x)

    def training_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        return optimizer_builder(self)