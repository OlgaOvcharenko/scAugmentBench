import torch
import torch.nn as nn

import lightly
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules.nn_memory_bank import NNMemoryBankModule
from lightly.models.modules.heads import NNCLRPredictionHead, NNCLRProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone

import lightning as pl


class NNCLR(pl.LightningModule):

    def __init__(self, in_dim, hidden_dim, hidden_dim_2, out_dim, memory_bank_size=4096, **kwargs):
        super().__init__()
        self.backbone = get_backbone(in_dim, hidden_dim, **kwargs)
        self.projection_head = NNCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.prediction_head = NNCLRPredictionHead(out_dim, hidden_dim_2, out_dim)
        self.memory_bank = NNMemoryBankModule(size=(memory_bank_size, out_dim))

        self.criterion = NTXentLoss()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p
    
    def predict(self, x):
        with torch.no_grad():
            z, p = self(x)
        return z # not p!!

    def training_step(self, batch, batch_idx):
        x0, x1 = batch[0]
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        # TODO: symmetrize the loss?
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        return optimizer_builder(self)
    
    """def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""
    