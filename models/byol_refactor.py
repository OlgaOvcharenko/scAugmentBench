
import copy

import torch
import torchvision
from torch import nn
import lightning as pl

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.loss.sym_neg_cos_sim_loss import SymNegCosineSimilarityLoss

from models.model_utils import *
from utils.train_utils import *


class BYOL(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, factor, **kwargs):
        super().__init__()
        self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
        hidden_dim_2 = factor*hidden_dim
        out_dim = hidden_dim
        
        self.projection_head = BYOLProjectionHead(hidden_dim, hidden_dim_2, out_dim)
        self.prediction_head = BYOLPredictionHead(out_dim, hidden_dim_2, out_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        self.criterion = SymNegCosineSimilarityLoss()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
    def training_step(self, batch, batch_idx):
        #(x0, x1), (id0, id1) = batch
        x0, x1 = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        # TODO: symmetrize the outputs of byol and calculate the loss
        loss = self.criterion((z0, p0), (z1, p1))
        self.log('train_loss_ssl', loss)
        return loss
    
    def predict(self, x):
        with torch.no_grad():
            z = self.backbone(x)
            return z
    
    def configure_optimizers(self):
        return optimizer_builder(self)