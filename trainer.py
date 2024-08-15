import logging

import numpy as np
import scanpy as sc
import wandb

import torch
import lightning as pl

from models.byol import BYOLModule
from models.barlowtwins import BarlowTwins
from models.moco import MoCo
from models.vicreg import VICReg
from models.simclr import SimCLR
from models.simsiam import SimSiam # which should we use? z or p? LOOK @ PAPER. --> zzzzz
from models.nnclr import NNCLR # which should we use? z or p? LOOK @ PAPER. --> zzzzz
#from models.dino import *

_model_dict = {"BYOL": BYOLModule, "BarlowTwins": BarlowTwins, "MoCo": MoCo, "VICReg": VICReg, "SimCLR": SimCLR, "SimSiam": SimSiam, "NNCLR": NNCLR}

def train_model(dataset, model_config, random_seed, batch_size, 
                num_workers, n_epochs):
    config = dict(config)
    model_name = config.pop("model")

    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True)
    
    model = _model_dict[str(model_name)](**model_config)
    trainer = pl.Trainer(max_epochs=n_epochs, accelerator="gpu")
    trainer.fit(
        model,
        train_loader,
    )

    return model