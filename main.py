import logging
import os
import pathlib
import time

import hydra
from omegaconf import DictConfig
from collections.abc import MutableMapping
import wandb

import scanpy as sc
import pandas as pd

from trainer import train_model
from evaluator import evaluate_model
from data.graph_augmentation_prep import *
from data.dataset import OurDataset
from data.augmentations import * #get_transforms, augmentations
from data.graph_augmentation_prep import * # builders for mnn and bbknn augmentation.

import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
import numpy as np
import random
import lightning as pl


_LOGGER = logging.getLogger(__name__)
_celltype_key = "CellType" #cfg["data"]["celltype_key"]
_batch_key = "batchlb" #cfg["data"]["batch_key"]


def load_data(config) -> sc.AnnData:
    # config["augmentation"]
    data_path = config["data"]["data_path"]
    _LOGGER.info(f"Loading data from {data_path}")

    augmentation_config = config["augmentation"] # using config["augmentation"]
    
    pm = PreProcessingModule(data_path, select_hvg=config["data"]["n_hvgs"], scale=False)
    #prepare_bbknn()

    augmentation_list = get_augmentation_list(augmentation_config, X=pm.adata.X)
    transforms = Compose(augmentation_list)
    
    train_dataset = OurDataset(adata=pm.adata,
                               transforms=transforms, 
                               valid_ids=None
                               )
    val_dataset = OurDataset(adata=pm.adata,
                             transforms=None,
                             valid_ids=None
                             )
    
    _LOGGER.info("Finished loading data.....")
    
    return train_dataset, val_dataset, pm.adata


def reset_random_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    # Old # No determinism as nn.Upsample has no deterministic implementation
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _LOGGER.info(f"Set random seed to {seed}")    


def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    """wandb.init(
        project=cfg.logging.project,
        reinit=True,
        config=flatten(dict(cfg)),
        entity=cfg.logging.entity,
        mode=cfg.logging.mode,
        tags=cfg.model.get("tag", None),
    )"""

    results_dir = pathlib.Path(cfg["results_dir"])
    _LOGGER.info(f"Results stored in {results_dir}")

    """
    Makes sure that we don't have to train models multiple times given a config file.
    """
    if os.path.exists(results_dir.joinpath("embedding.npz")):
        _LOGGER.info(f"Embedding at {results_dir} already exists.")
        return
    
    random_seed = cfg["random_seed"]
    if torch.cuda.is_available():
        reset_random_seeds(random_seed)
    
    train_dataset, val_dataset, ad = load_data(cfg)

    _LOGGER.info(f"Start training ({cfg['model']['model']})")
    _LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")

    start = time.time()
    model = train_model(dataset=train_dataset, 
                        model_config=cfg["model"],
                        random_seed=random_seed, 
                        batch_size=256,
                        num_workers=14,
                        n_epochs=200,
                        logger=_LOGGER)
    run_time = time.time() - start
    _LOGGER.info(f"Training of the model took {round(run_time, 3)} seconds.")
    
    results, embedding = evaluate_model(model=model,
                                        dataset=val_dataset,
                                        adata=ad,
                                        batch_size=256,
                                        num_workers=14,
                                        )
    #_LOGGER.info(f"Results:\n{results}")
    
    results.to_csv(results_dir.joinpath("evaluation_metrics.csv"), index=None)
    np.savez_compressed(results_dir.joinpath("embedding.npz"), embedding)

if __name__ == "__main__":
    main()