import copy

import torch
import torchvision
from torch import nn

from lightly.loss.dino_loss import DINOLoss
from lightly.models.modules.heads import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
#from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule

import lightning as pl

from utils.train_utils import optimizer_builder
from models.model_utils import get_backbone



class DINO(pl.LightningModule):
    
    def __init__(self, in_dim, hidden_dim, out_dim, bsize):
        super().__init__()
        self.student_backbone = get_backbone(in_dim, hidden_dim)
        #bottleneck = hidden_dim//4 # TODO: Should this be its own parameter?
        self.student_head = DINOProjectionHead(hidden_dim, hidden_dim, bsize, out_dim, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOProjectionHead(hidden_dim, hidden_dim, out_dim, out_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
        self.criterion = DINOLoss(
            output_dim=out_dim,
            #warmup_teacher_temp_epochs=5,
            )
        self.curr_epoch = 0

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z
    
    def predict(self, x):
        with torch.no_grad():
            return self(x)
    
    def training_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion([z0], [z1], epoch=self.curr_epoch)
        return loss
    
    def on_train_epoch_end(self):
        self.curr_epoch += 1
        #self.training_step_outputs.clear()  # free memory
    
    def configure_optimizers(self):
        return optimizer_builder(self)