import torch
import numpy as np


def infer_embedding(model, val_loader):
    outs = []
    for x in val_loader:
        with torch.no_grad():
            outs.append(model.predict(x[0]))

    embedding = torch.concat(outs)
    print(f"Embedding-Shape: {embedding.shape}")
    embedding = np.array(embedding)
    return embedding


def evaluate_model(model, dataset, batch_size, num_workers):
    val_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False)
    embedding = infer_embedding(model, val_loader)
    em = EvaluationModule(db, c.adata, model_indices=range(len(db.configs)))
    em.run_evaluation()
    return em.display_table()