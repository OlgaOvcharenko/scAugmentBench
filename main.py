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

import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np

_LOGGER = logging.getLogger(__name__)
_celltype_key = "CellType" #cfg["data"]["celltype_key"]
_batch_key = "batchlb" #cfg["data"]["batch_key"]


def load_data(config) -> sc.AnnData:
    # config["augmentation"]
    _LOGGER.info("Loading data...")
    print(config)
    data_path = config["data"]["data_path"]
    augment_list = config["augmentation"] # using config["augmentation"]
    print(augment_list)
    return None, None


def reset_random_seeds(seed):
    random.seed(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    # Old # No determinism as nn.Upsample has no deterministic implementation
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
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
    print(cfg)
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

    random_seed = cfg["random_seed"]
    reset_random_seeds(random_seed)
    
    train_dataset, val_dataset = load_data(cfg)
    print(results_dir)
    with open(f"{results_dir}/test.txt", 'w') as f:
        f.write("AA")

    """_LOGGER.info("Start training model")
    start = time.time()
    model = train_model(
        train_dataset,
        model_config=cfg["model"],
        batch_key=_batch_key,
        celltype_key=_celltype_key,
        random_seed=random_seed,
    )
    run_time = time.time() - start"""

    print("evaluation to be implemented.")
    """_LOGGER.info("Evaluate model")
    results = evaluate_model(val_dataset,
                                     model)
    _LOGGER.info("logging results.")
    _LOGGER.info(results)
    #wandb.log(results)
    results.to_csv(results_dir.joinpath("evaluation_metrics.csv"), index=None)"""

if __name__ == "__main__":
    main()