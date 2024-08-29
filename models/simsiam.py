import torch
import torch.nn as nn

import lightly
#from lightly.loss.neg_cos_sim_loss import NegativeCosineSimilarity
from lightly.loss.negative_cosine_similarity import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone

import lightning as pl


class SimSiam(pl.LightningModule):

    def __init__(self, in_dim, hidden_dim, hidden_dim_2, out_dim, **kwargs):
        super().__init__()
        assert hidden_dim_2 <= hidden_dim, "hidden dim of prediction head should not be too large!"
        self.backbone = get_backbone(in_dim, hidden_dim, **kwargs)
        #self.projection_head = SimSiamProjectionHead(hidden_dim, hidden_dim//2, out_dim) # TODO: How to choose dimensions here?
        self.projection_head = SimSiamProjectionHead(hidden_dim, hidden_dim_2, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, hidden_dim_2, out_dim)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p
    
    def predict(self, x):
        with torch.no_grad():
            z = self.backbone(x)
            return z
            #return self(x)[0] # TODO: or [1]??

    def training_step(self, batch, batch_idx):
        x0, x1 = batch[0]
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        # TODO: symmetrize the loss? --> Using the first term below is symmetry! 
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss
    
    def configure_optimizers(self):
        return optimizer_builder(self)
    
    """def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""