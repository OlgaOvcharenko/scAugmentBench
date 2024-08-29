from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch
import torch.nn as nn
import numpy as np


class OurDataset(Dataset):

    def __init__(self, adata, transforms=None, valid_ids=None):
        super().__init__()
        self.transforms = transforms
        self.adata = adata
        self.X = torch.tensor(adata.X.toarray()) # this may be a caveat for large matrices; there, move to forward function: x = torch.tensor(self.X[index].toarray())
        if valid_ids is None:
            self.valid_cellidx = np.arange(len(adata))
        else:
            self.valid_cellidx = valid_ids
        self.n_samples = len(self.valid_cellidx)
        self.n_genes = self.X.shape[1]

    """
    TODO: How to handle when we remove cells during filtering?
    """
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #index = self.valid_cellidx[i]
        x = self.X[index].unsqueeze(0)
        if self.transforms is not None:
            """
            Can we optimize here? Remove the cell-ids-stuff? Remove the squeeze stuff?
            """
            #out_dict = self.transforms({'x1': x, 'x2': x, 'cell_ids': torch.from_numpy(np.array([index]))})
            out_dict = self.transforms({'x1': x, 'x2': x, 'cell_ids': index})
            return [out_dict['x1'].squeeze().float(), out_dict['x2'].squeeze().float()], [index, index] #out_dict['x1'], out_dict['x2'], index
        else:
            return x.squeeze().float(), index