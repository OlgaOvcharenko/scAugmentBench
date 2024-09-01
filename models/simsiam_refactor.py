import torch
import torch.nn as nn

import lightly
#from lightly.loss.neg_cos_sim_loss import NegativeCosineSimilarity
from lightly.loss.negative_cosine_similarity import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone, get_backbone_deep, clip_loss

import lightning as pl


class SimSiam(pl.LightningModule):

    def __init__(self, in_dim, hidden_dim, multimodal, factor, hidden_dim_2, out_dim, in_dim2=0, integrate=None, only_rna=False, predict_projection=False, **kwargs):
        super().__init__()
        assert hidden_dim_2 <= hidden_dim, "hidden dim of prediction head should not be too large!"
        
        self.multimodal = multimodal
        self.predict_projection = predict_projection

        hidden_dim_2 = hidden_dim//(2*factor)
        out_dim = hidden_dim//factor
        
        if self.multimodal:
            self.integrate = integrate
            self.predict_only_rna = only_rna

            self.temperature = 1.0 # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs) 
            self.projection_head = SimSiamProjectionHead(hidden_dim, hidden_dim, out_dim)
            self.prediction_head = SimSiamPredictionHead(out_dim, hidden_dim_2, out_dim)
            
            self.backbone2 = get_backbone_deep(in_dim2, hidden_dim, **kwargs)
            self.projection_head2 = SimSiamProjectionHead(hidden_dim, hidden_dim, out_dim)
            self.prediction_head2 = SimSiamPredictionHead(out_dim, hidden_dim_2, out_dim)
            self.criterion = NegativeCosineSimilarity()

        else:
            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
            self.projection_head = SimSiamProjectionHead(hidden_dim, hidden_dim, out_dim)
            self.prediction_head = SimSiamPredictionHead(out_dim, hidden_dim_2, out_dim)
            self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        if self.multimodal:
            f = self.backbone(x[0]).flatten(start_dim=1)
            z = self.projection_head(f)
            p = self.prediction_head(z)
            z = z.detach()

            f2 = self.backbone2(x[1]).flatten(start_dim=1)
            z2 = self.projection_head2(f2)
            p2 = self.prediction_head2(z2)
            z2 = z2.detach()
            return z, p, z2, p2
        else:
            f = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(f)
            p = self.prediction_head(z)
            z = z.detach()
            return z, p
    
    def predict(self, x):
        with torch.no_grad():
            if self.multimodal:
                if self.predict_projection:
                    z1_0, _, z1_1, _ = self(x)
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
                return self(x)[0] if self.predict_projection else self.backbone(x)

    def training_step(self, batch, batch_idx):
        if self.multimodal:
            x1_1, x2_1, x1_2, x2_2 = batch[0]
            x0 = [x1_1, x1_2]
            x1 = [x2_1, x2_2]
            
            z1_0, p1_0, z1_1, p1_1 = self.forward(x0) # RNA, Protein
            z2_0, p2_0, z2_1, p2_1 = self.forward(x1)

            if self.integrate == "add":
                z0 = z1_0 + z1_1
                p0 = p1_0 + p1_1
                z1 = z2_0 + z2_1
                p1 = p2_0 + p2_1
            elif self.integrate == "mean":
                z0 = (z1_0 + z1_1) / 2
                p0 = (p1_0 + p1_1) / 2
                z1 = (z2_0 + z2_1) / 2
                p1 = (p2_0 + p2_1) / 2
            elif self.integrate == "concat":
                z0 = torch.cat((z1_0, z1_1), 1)
                z1 = torch.cat((z2_0, z2_1), 1)
                p0 = torch.cat((p1_0, p1_1), 1)
                p1 = torch.cat((p2_0, p2_1), 1)
            elif self.integrate == "clip":
                # FIXME does it make sense?
                # loss = 0.25 * (clip_loss(z1_0, p2_1) + clip_loss(z2_1, p1_0) + clip_loss(z1_1, p2_0) + clip_loss(z2_0, p1_1))
                loss = 0.25 * (clip_loss(z1_0, z1_1) + clip_loss(p1_0, p1_1) + clip_loss(z2_0, z2_1) + clip_loss(p2_0, p2_1))
                return loss

            else:
                raise Exception("Invalid integration method.")
            
            loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

        else:
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