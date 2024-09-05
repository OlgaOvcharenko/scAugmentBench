import torch
import torch.nn as nn

import lightly
from lightly.loss.barlow_twins_loss import BarlowTwinsLoss
from lightly.models.modules.nn_memory_bank import NNMemoryBankModule
from lightly.models.modules.heads import BarlowTwinsProjectionHead

from utils.train_utils import optimizer_builder
from models.model_utils import *

import lightning as pl
import numpy as np


class BarlowTwins(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, multimodal, factor, in_dim2=0, integrate=None, predict_only_rna=False, predict_projection=False, **kwargs):
        super().__init__()

        self.multimodal = multimodal
        self.predict_projection = predict_projection
        
        out_dim = factor*hidden_dim
        
        if self.multimodal:
            self.integrate = integrate
            self.predict_only_rna = predict_only_rna
            self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
            self.projection_head = BarlowTwinsProjectionHead(hidden_dim, out_dim, out_dim)

            self.backbone2 = get_backbone_deep(in_dim2, hidden_dim, **kwargs)
            self.projection_head2 = BarlowTwinsProjectionHead(hidden_dim, out_dim, out_dim)

            if self.integrate == 'clip':
                self.loss_img = nn.CrossEntropyLoss()
                self.loss_txt = nn.CrossEntropyLoss()
            else:
                self.criterion = BarlowTwinsLoss()
        else:
            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
            self.projection_head = BarlowTwinsProjectionHead(hidden_dim, out_dim, out_dim)
            self.criterion = BarlowTwinsLoss()

    def forward(self, x):
        if self.multimodal:
            x1 = self.backbone(x[0]).flatten(start_dim=1)
            z1 = self.projection_head(x1)

            x2 = self.backbone2(x[1]).flatten(start_dim=1)
            z2 = self.projection_head2(x2)
            return z1, z2

        else:
            x = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(x)
            return z
    
    def predict(self, x):
        with torch.no_grad():
            if self.multimodal:
                if self.predict_projection:
                    z1_0, z1_1 = self(x)  
                else:
                    z1_0, z1_1 = self.backbone(x[0]), self.backbone2(x[1])

                if self.predict_only_rna:
                    return z1_0

                if self.integrate == "add":
                    z0 = z1_0 + z1_1
                elif self.integrate == "mean":
                    z0 = (z1_0 + z1_1) / 2
                else:
                    z0 = torch.cat((z1_0, z1_1), 1)
                return z0
            else:
                return self(x) if self.predict_projection else self.backbone(x)
    
    def predict_separate(self, x):
        with torch.no_grad():
            if self.multimodal:
                if self.predict_projection:
                    z1_0, z1_1 = self(x)  
                else:
                    z1_0, z1_1 = self.backbone(x[0]), self.backbone2(x[1])

                if self.predict_only_rna:
                    raise Exception("Invalid path")

                if self.integrate == "add":
                    z0 = z1_0 + z1_1
                elif self.integrate == "mean":
                    z0 = (z1_0 + z1_1) / 2
                else:
                    z0 = torch.cat((z1_0, z1_1), 1)
                return z0, z1_0, z1_1
            else:
                raise Exception("Invalid path")

    def training_step(self, batch, batch_index):
        if self.multimodal:
            x1_1, x2_1, x1_2, x2_2 = batch[0]
            x0 = [x1_1, x1_2]
            x1 = [x2_1, x2_2]
            
            z1_0, z1_1 = self.forward(x0) # RNA, Protein
            z2_0, z2_1 = self.forward(x1) # RNA, Protein

            if self.integrate == "add":
                z0 = z1_0 + z1_1
                z1 = z2_0 + z2_1
            elif self.integrate == "mean":
                z0 = (z1_0 + z1_1) / 2
                z1 = (z2_0 + z2_1) / 2
            elif self.integrate == "concat":
                z0 = torch.cat((z1_0, z1_1), 1)
                z1 = torch.cat((z2_0, z2_1), 1)
            elif self.integrate == "clip":
                logit_scale = self.temperature.exp()
                
                # FIXME 0.5 * ()
                loss = clip_loss(z1_0, z1_1, logit_scale, self.loss_img, self.loss_txt) + clip_loss(z2_0, z2_1, logit_scale, self.loss_img, self.loss_txt)
                return loss

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
    

"""    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""