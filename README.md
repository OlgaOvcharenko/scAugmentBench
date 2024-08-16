# Benchmarking Single-Cell Augmentations for Contrastive SSL

This repository implements a benchmark of augmentations for single-cell RNA-seq data with contrastive learning.

We evaluate augmentations across three common model architectures.

All architectures share the same encoder architecture. They differ in various details, such as:
- the employed loss function
- usage of a memory bank
- implementation of nearest-neighbor-embeddings
- usage of a projector
- usage of a predictor

## Implementation Details

We use hydra to schedule experiments (see _conf_ folder) and lightly to define the neural networks (see _model_ folder).
Model training is performed with pytorch-lightning, the ADAM optimizer and a constant learning rate 1e-4.

## Scope

Goal of this project is to advertise further research on contrastive self-supervised learning for cell representation learning.
Our work shows that current methods are able to correct for batch effects, improving performance on downstream tasks.

## Limitations

This work is a first step towards a wider application of CL for cell representation learning. We note that there are parameters and architectural
choices that were not considered during this work due to computational constraints. For the presented models, there are many parameters (and hyperparameters)
that could be improved upon. This is left as future work.