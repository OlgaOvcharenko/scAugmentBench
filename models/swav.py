import torch
import torchvision
from torch import nn

from lightly.loss.swav_loss import SwaVLoss
from  lightly.models.modules.heads import SwaVProjectionHead, SwaVPrototypes
from  lightly.models.modules.nn_memory_bank import NNMemoryBankModule

from models.model_utils import get_backbone


class SwaV(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, memory_bank_size=4096, **kwargs):
        super().__init__()
        self.backbone = get_backbone(in_dim, hidden_dim, **kwargs)
        self.projection_head = SwaVProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.prototypes = SwaVPrototypes(out_dim, hidden_dim, 1) # last param is #steps during which the "prototypes" are fixed.


        self.start_queue_at_epoch = 2
        self.queues = nn.ModuleList(
            [NNMemoryBankModule(size=(memory_bank_size, out_dim)) for _ in range(2)]
        )

    def forward(self, high_resolution, low_resolution, epoch):
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [
            self.prototypes(x, epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features, epoch)

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features
    
    def predict(self, x):
        return self()

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features, epoch):
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if self.start_queue_at_epoch > 0 and epoch < self.start_queue_at_epoch:
            return None

        # Assign prototypes
        queue_prototypes = [self.prototypes(x, epoch) for x in queue_features]
        return queue_prototypes
    
    """def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""