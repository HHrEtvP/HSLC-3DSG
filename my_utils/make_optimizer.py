import torch


def build_optimizer(cfg, model, base_lr, weight_decay):
    params = []
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        params += [{"params": [v], "lr": base_lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=weight_decay)
    return optimizer
