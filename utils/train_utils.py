import torch


def optimizer_builder(model, lr=1e-4, weight_decay=1e-6):
    #optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    optim= torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay,) # momentum=momentum
    scheduler=torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    return [optim], [scheduler]
