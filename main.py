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

from trainer import train_model, train_clf
from evaluator import evaluate_model
from data.graph_augmentation_prep import *
from data.dataset import OurDataset
from data.augmentations import * #get_transforms, augmentations
from data.graph_augmentation_prep import * # builders for mnn and bbknn augmentation.

import torch
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
    
    #prepare_bbknn()
    if config['augmentation']['bbknn']['apply_prob'] > 0:
        _LOGGER.info("Preprocessing with bbknn.")
        pm = BbknnAugment(data_path, select_hvg=config["data"]["n_hvgs"], 
                          scale=False, knn=augmentation_config['bbknn']['knn'],
                          exclude_fn=False, trim_val=None, holdout_batch=config["data"]["holdout_batch"]
                          )
        augmentation_list = get_augmentation_list(augmentation_config, X=pm.adata.X, nns=pm.nns, input_shape=(1, len(pm.gname)))
    elif config['augmentation']['mnn']['apply_prob'] > 0:
        _LOGGER.info("Preprocessing with mnn.")
        pm = ClaireAugment(data_path, select_hvg=config["data"]["n_hvgs"],
                           scale=False, knn=augmentation_config['mnn']['knn'],
                           exclude_fn=False, filtering=True, holdout_batch=config["data"]["holdout_batch"])
        augmentation_list = get_augmentation_list(augmentation_config, X=pm.adata.X, nns=pm.nns, mnn_dict=pm.mnn_dict, input_shape=(1, len(pm.gname)))
    else:
        _LOGGER.info("Preprocessing without bbknn.")
        pm = PreProcessingModule(data_path, select_hvg=config["data"]["n_hvgs"], 
                                 scale=False, holdout_batch=config["data"]["holdout_batch"])
        augmentation_list = get_augmentation_list(augmentation_config, X=pm.adata.X, input_shape=(1, len(pm.gname)))
        _LOGGER.info("Augmentations generated.")
    
    #_LOGGER.info(f"Augmentation list: {augmentation_list}")
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
    The check below makes sure that we don't have to train models multiple times given a config file.
    """
    if os.path.exists(results_dir.joinpath("embedding.npz")):
        _LOGGER.info(f"Embedding at {results_dir} already exists.")
        return
    
    random_seed = cfg["random_seed"]
    if torch.cuda.is_available():
        reset_random_seeds(random_seed)
        _LOGGER.info(f"Successfully reset seeds.")
        print(cfg)

    train_dataset, val_dataset, ad = load_data(cfg)

    _LOGGER.info(f"Start training ({cfg['model']['model']})")
    _LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    
    cfg['model']['in_dim'] = train_dataset.n_genes

    start = time.time()
    model = train_model(dataset=train_dataset, 
                        model_config=cfg["model"],
                        random_seed=random_seed, 
                        batch_size=cfg["model"]["training"]["batch_size"],
                        num_workers=14,
                        n_epochs=cfg["model"]["training"]["max_epochs"],
                        logger=_LOGGER)
    run_time = time.time() - start
    _LOGGER.info(f"Training of the model took {round(run_time, 3)} seconds.")
    
    if cfg["debug"] is True:
        pass
    elif cfg["data"]["holdout_batch"] is None:
        _LOGGER.info("Running SCIB-Benchmark Evaluation.")
        results, embedding = evaluate_model(model=model,
                                            dataset=val_dataset,
                                            adata=ad,
                                            batch_size=cfg["model"]["training"]["batch_size"],
                                            num_workers=14,
                                            logger=_LOGGER,
                                            embedding_save_path=results_dir.joinpath("embedding.npz")
                                        )
        try:
            results.to_csv(results_dir.joinpath("evaluation_metrics.csv"), index=None)
        except:
            _LOGGER.info("Something went wrong with the benchmark.")
    
    elif cfg["data"]["holdout_batch"] is not None:
        _LOGGER.info("Running QR-Mapper-Inference.")
        _LOGGER.info(f"Results of QR-Mapper will be saved in {results_dir}")
        # load total adata, and get holdout-subset as Val_X and Y for clf-training
        pm = PreProcessingModule(cfg["data"]["data_path"], select_hvg=cfg["data"]["n_hvgs"], 
                                 scale=False, holdout_batch=None)
        if type(cfg["data"]["holdout_batch"]) == str:
            fltr = pm.adata.obs['batchlb']==cfg["data"]["holdout_batch"]
        else:
            fltr = [pm.adata.obs['batchlb'][i] in cfg["data"]["holdout_batch"] for i in range(len(pm.adata))]
        
        train_adata = ad
        val_adata = pm.adata[fltr]
        clf, maavg_f1, acc, run_time = train_clf(model, train_adata, val_adata, ctype_key='CellType')
        
        print(f"MaAVG-F1: {maavg_f1}\nAccuracy: {acc}")
        _LOGGER.info(f"Finished Training of the QR-Mapper in {run_time} seconds.")


if __name__ == "__main__":
    main()