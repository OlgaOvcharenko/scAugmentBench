import torch
import torch.nn as nn

import lightly
from lightly.loss.barlow_twins_loss import BarlowTwinsLoss
from lightly.models.modules.nn_memory_bank import NNMemoryBankModule
from lightly.models.modules.heads import BarlowTwinsProjectionHead

from utils.train_utils import optimizer_builder
from models.model_utils import get_backbone

import lightning as pl


class BarlowTwins(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, out_dim, **kwargs):
        super().__init__()
        self.backbone = get_backbone(in_dim, hidden_dim, **kwargs)
        self.projection_head = BarlowTwinsProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.criterion = BarlowTwinsLoss()

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
        # TODO: symmetrize the loss?
        loss = self.criterion(z0, z1)
        return loss
    
    def configure_optimizers(self):
        return optimizer_builder(self)
    

"""    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""