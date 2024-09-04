# although we have the original CLEAR code, we can use the SIMCLR code from lightly as a comparison.
import torch
from torch import nn

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone_deep

import lightning as pl


class SimCLR(pl.LightningModule):
    
    def __init__(self, in_dim, hidden_dim, factor, **kwargs):
        super().__init__()
        out_dim=hidden_dim//factor
        self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
    
    def predict(self, x):
        with torch.no_grad():
            #return self(x)
            z = self.backbone(x)
            return z

    def training_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        # TODO: symmetrize the loss?
        loss = self.criterion(z0, z1)
        return loss
    
    def configure_optimizers(self):
        return optimizer_builder(self)
    
    """def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""