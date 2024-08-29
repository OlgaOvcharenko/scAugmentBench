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

def clip_loss(image_embeddings, text_embeddings):
    # FIXME needed or not
    # image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    # text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    logits = (text_embeddings @ image_embeddings.T) / self.temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
    )
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (images_loss + texts_loss) / 2.0
    return loss.mean()

def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss
    