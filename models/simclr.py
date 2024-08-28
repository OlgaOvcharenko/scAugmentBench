# although we have the original CLEAR code, we can use the SIMCLR code from lightly as a comparison.
import torch
from torch import nn

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone

import lightning as pl


class SimCLR(pl.LightningModule):
    
    def __init__(self, in_dim, hidden_dim, multimodal, out_dim, in_dim2=0, integrate=None, **kwargs):
        super().__init__()

        self.multimodal = multimodal
        self.integrate = integrate
        if self.multimodal:
            self.backbone1 = get_backbone(in_dim, hidden_dim, **kwargs)
            self.projection_head1 = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)

            self.backbone2 = get_backbone(in_dim2, hidden_dim, **kwargs)
            self.projection_head2 = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)

            self.criterion = NTXentLoss()
        else:
            self.backbone = get_backbone(in_dim, hidden_dim, **kwargs)
            self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
            self.criterion = NTXentLoss()

    def forward(self, x):
        if self.multimodal:
            x1 = self.backbone1(x[0]).flatten(start_dim=1)
            z1 = self.projection_head1(x1)

            x2 = self.backbone2(x[1]).flatten(start_dim=1)
            z2 = self.projection_head2(x2)
            return z1, z2

        else:
            x = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(x)
            return z
    
    def predict(self, x):
        with torch.no_grad():
            return self(x)

    def training_step(self, batch, batch_index):
        if self.multimodal:
            x1_1, x2_1, x1_2, x2_2 = batch[0]
            x0 = [x1_1, x1_2]
            x1 = [x2_1, x2_2]
            
            z1_0, z1_1 = self.forward(x0) # RNA
            z2_0, z2_1 = self.forward(x1) # Protein

            if self.integrate == "add":
                z0 = z1_0 + z2_0
                z1 = z1_1 + z2_1
            elif self.integrate == "mean":
                z0 = (z1_0 + z2_0) / 2
                z1 = (z1_1 + z2_1) / 2
            elif self.integrate == "concat":
                z0 = torch.cat((z1_0, z1_1), 1)
                z1 = torch.cat((z2_0, z2_1), 1)
            else:
                raise Exception("Invalid integration method.")

            # TODO: symmetrize the loss?
            loss = self.criterion(z0, z1)

        else:
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