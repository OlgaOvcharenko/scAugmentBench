import torch
from torch import nn

def get_backbone(in_dim: int, out_dim: int):
    """
    Returns a single-layer encoder to extract features from gene expression profiles.

    In our benchmark study, this architecture is shared between all models.
    """

    modules = [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    ]
    return nn.Sequential(*modules)
