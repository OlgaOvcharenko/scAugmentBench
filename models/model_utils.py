import torch
from torch import nn

def get_backbone(in_dim: int, out_dim: int, dropout:float=0, **kwargs):
    """
    Returns a single-layer encoder to extract features from gene expression profiles.

    In our benchmark study, this architecture is shared between all models.
    """
    print(f"Backbone built with dropout {dropout}")

    modules = [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout)
    ]
    return nn.Sequential(*modules)
