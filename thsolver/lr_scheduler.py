import math
from bisect import bisect_right
import torch.optim.lr_scheduler as LR


def multi_step(optimizer, flags):
    return LR.MultiStepLR(optimizer, flags.milestones, flags.gamma)


def cos(optimizer, flags):
    return LR.CosineAnnealingLR(optimizer, flags.max_epoch, eta_min=flags.lr_min)


def poly(optimizer, flags):
    lr_lambda = lambda epoch: (1 - epoch / flags.max_epoch) ** flags.lr_power
    return LR.LambdaLR(optimizer, lr_lambda)


def constant(optimizer, flags):
    lr_lambda = lambda epoch: 1
    return LR.LambdaLR(optimizer, lr_lambda)


def cos_warmup(optimizer, flags):
    def lr_lambda(epoch):
        warmup = flags.warmup_epoch
        warmup_init = flags.warmup_init
        if epoch <= warmup:
            return (1 - warmup_init) * epoch / warmup + warmup_init
        lr_min = flags.lr_min
        ratio = (epoch - warmup) / (flags.max_epoch - warmup)
        return lr_min + 0.5 * (1.0 - lr_min) * (1 + math.cos(math.pi * ratio))
    return LR.LambdaLR(optimizer, lr_lambda)


def poly_warmup(optimizer, flags):
    def lr_lambda(epoch):
        warmup = flags.warmup_epoch
        warmup_init = flags.warmup_init
        if epoch <= warmup:
            return (1 - warmup_init) * epoch / warmup + warmup_init
        ratio = (epoch - warmup) / (flags.max_epoch - warmup)
        return (1 - ratio) ** flags.lr_power
    return LR.LambdaLR(optimizer, lr_lambda)


def step_warmup(optimizer, flags):
    def lr_lambda(epoch):
        warmup = flags.warmup_epoch
        warmup_init = flags.warmup_init
        if epoch <= warmup:
            return (1 - warmup_init) * epoch / warmup + warmup_init
        milestones = sorted(flags.milestones)
        return flags.gamma ** bisect_right(milestones, epoch)
    return LR.LambdaLR(optimizer, lr_lambda)


def get_lr_scheduler(optimizer, flags):
    lr_dict = {
        'step': multi_step,
        'cos': cos,
        'poly': poly,
        'constant': constant,
        'cos_warmup': cos_warmup,
        'poly_warmup': poly_warmup,
        'step_warmup': step_warmup,
    }
    lr_func = lr_dict[flags.lr_type]
    return lr_func(optimizer, flags)
