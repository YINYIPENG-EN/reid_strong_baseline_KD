# encoding: utf-8
import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():  # key为对应的层
        if not value.requires_grad:  # 是否求导
            continue
        lr = cfg.BASE_LR  # 初始学习率
        weight_decay = cfg.WEIGHT_DECAY  # 权重衰减
        if "bias" in key:  # 判断是否有bias
            lr = cfg.BASE_LR * cfg.BIAS_LR_FACTOR
            weight_decay = cfg.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.OPTIMIZER_NAME == 'SGD':  # 优化器类型，随机梯度下降
        optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params, momentum=cfg.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params)  # Adam
    return optimizer
