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


def get_backbone_deep(in_dim: int, encoder_out_dim: int, dropout:float=0, **kwargs):
    """
    Returns a single-layer encoder to extract features from gene expression profiles.

    In our benchmark study, this architecture is shared between all models.
    """
    print(f"Backbone built with dropout {dropout}")

    modules = [
        nn.Linear(in_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, encoder_out_dim),
        nn.BatchNorm1d(encoder_out_dim),
        nn.Dropout(p=dropout),
    ]
    return nn.Sequential(*modules)
