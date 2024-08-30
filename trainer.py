import logging

import numpy as np
import scanpy as sc
import wandb
import time

import torch
import lightning as pl
import hydra

from models.byol_refactor import BYOL
from models.barlowtwins_refactor import BarlowTwins
from models.moco_refactor import MoCo
from models.vicreg_refactor import VICReg
from models.simclr_refactor import SimCLR
from models.simsiam_refactor import SimSiam
from models.nnclr_refactor import NNCLR
from models.concerto import Concerto
#from models.dino import *
from evaluator import infer_embedding
from anndata.experimental.pytorch import AnnLoader
from data.dataset import OurDataset

from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import os

_model_dict = {"BYOL": BYOL, "BarlowTwins": BarlowTwins, "MoCo": MoCo, "VICReg": VICReg, "SimCLR": SimCLR, "SimSiam": SimSiam, "NNCLR": NNCLR, "Concerto": Concerto}


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def train_model(dataset, model_config, random_seed, batch_size, 
                num_workers, n_epochs, logger, ckpt_dir, cfg=None):
    model_name = model_config["model"]

    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True)
    
    logger.info(f".. Dataloader ready. Now build {model_name}")

    model = _model_dict[str(model_name)](**model_config)
    
    print(model_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=n_epochs, accelerator=device, default_root_dir=ckpt_dir, callbacks=[CheckpointEveryNSteps(save_step_frequency=25)]) # cpu works for smaller tasks!!
    logger.info(f".. Model ready. Now train on {device}.")
    
    try:
        trainer.fit(
            model,
            train_loader,
        )
        logger.info(f".. Training done.")
    except Exception as error:
        # handle the exception
        logger.info(".. An exception occurred while training:", error)
    
    return model

def inference(model, val_loader):
    outs = []
    for x in val_loader:
        with torch.no_grad():
            outs.append(model.predict(x.layers['counts']))
    
    embedding = torch.concat(outs)
    embedding = np.array(embedding)
    return embedding

def train_clf(encoder, train_adata, val_adata, batch_size=256, num_workers=12, ctype_key='CellType'):
    train_loader = torch.utils.data.DataLoader(
        dataset=OurDataset(adata=train_adata, transforms=None, valid_ids=None),
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False, 
        drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=OurDataset(adata=val_adata, transforms=None, valid_ids=None),
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False, 
        drop_last=False
    )

    start = time.time()
    train_X, val_X = infer_embedding(encoder, train_loader), infer_embedding(encoder, val_loader)
    train_y = train_adata.obs[ctype_key]
    val_y = val_adata.obs[ctype_key]
    
    clf = KNeighborsClassifier(n_neighbors=11)
    clf = clf.fit(train_X, train_y)
    run_time = time.time() - start
    
    y_pred = clf.predict(val_X)
    
    maavg_f1 = f1_score(val_y, y_pred, average='macro')
    accuracy = accuracy_score(val_y, y_pred)
    return clf, maavg_f1, accuracy, run_time