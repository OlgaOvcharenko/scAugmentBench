import torch
from torch import nn

class DomainSpecificBatchNorm(nn.Module):

    def __init__(self, num_features, num_classes):
        super(DomainSpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, domain_label):
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label

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

def get_backbone_deep(in_dim: int, encoder_out_dim: int, dropout:float=0, dsbn=False, num_domains=1, **kwargs):
    """
    Returns a single-layer encoder to extract features from gene expression profiles.

    In our benchmark study, this architecture is shared between all models.
    """
    print(f"Backbone built with dropout {dropout}")

    # if dsbn:
    #     modules = [
    #         nn.Linear(in_dim, 128),
    #         DomainSpecificBatchNorm(128, num_domains),
    #         nn.ReLU(),
    #         nn.Linear(128, encoder_out_dim),
    #         DomainSpecificBatchNorm(128, num_domains),
    #         nn.Dropout(p=dropout),
    #     ]
    # else:
    modules = [
        nn.Linear(in_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, encoder_out_dim),
        nn.BatchNorm1d(encoder_out_dim),
        nn.Dropout(p=dropout),
    ]
    return nn.Sequential(*modules)

def clip_loss(image_features, text_features, logit_scale, loss_img, loss_txt):
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ground_truth = torch.arange(len(image_features), dtype=torch.long, device=device)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2

    return total_loss

def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss
    
