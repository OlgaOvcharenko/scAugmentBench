import torch
from torch import nn

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone

import lightning as pl


def student_backbone(in_dim: int, out_dim: int, **kwargs):
    """
    Returns a single-layer encoder to extract features from gene expression profiles.

    In our benchmark study, this architecture is shared between all models.
    """
    modules = [
        nn.BatchNorm1d(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.BatchNorm1d(out_dim),
    ]
    return nn.Sequential(*modules)


class AttentionWithContext(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        # the u needs to be initialized. 
        self.u = nn.Parameter(torch.Tensor(in_dim))
        nn.init.xavier_uniform_(self.u.unsqueeze(0))

    def forward(self, x):
        #u_t = tanh(W h_t + b)
        u_t = torch.tanh(self.l1(x))
        a_t = torch.matmul(u_t, self.u)
        a = torch.exp(a_t)
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-10)
        a = a.unsqueeze(-1)
        weighted_input = x * a
        return weighted_input, a

    

class Concerto(pl.LightningModule):
    
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2, vocabulary_size=4000, **kwargs):
        super().__init__()
        #self.backbone = get_backbone(in_dim, hidden_dim, , **kwargs)
        print(f"Attention with vocabulary-size {vocabulary_size}")
        self.student_encoder = student_backbone(in_dim, hidden_dim, **kwargs)
        self.projection_head_student = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim, num_layers=1)
        
        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=out_dim)
        self.attention = AttentionWithContext(out_dim, out_dim)
        self.projection_head_teacher = SimCLRProjectionHead(out_dim, out_dim, out_dim, num_layers=1)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.criterion = NTXentLoss(temperature=0.2)

    def forward_teacher(self, x):
        # TODO.
        #sparse_value = torch.unsqueeze(x, 2)
        x = self.bn1(x)
        torch.nn.functional.relu_(x)
        x = self.embedding_layer(x.long())
        x, a = self.attention(x)
        # attention returned nans because u was uninitialized...

        x = torch.tanh(torch.sum(x, axis=1))
        x = self.bn2(x)
        x = self.dropout(x)
        z = self.projection_head_teacher(x)
        return z
    
    def forward(self, x):
        x = self.student_encoder(x).flatten(start_dim=1)
        x = self.dropout(x)
        z = self.projection_head_student(x)
        return z
    
    def predict(self, x):
        with torch.no_grad():
            return self(x)
    
    def training_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward_teacher(x1)
        # TODO: symmetrize the loss?
        loss = self.criterion(z0, z1)
        return loss
    
    def configure_optimizers(self):
        return optimizer_builder(self)