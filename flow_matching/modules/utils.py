import torch
import torch.nn as nn

FP16_MODULES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.Linear,
)


def convert_module_to_f16(layer: nn.Module) -> None:
    if isinstance(layer, FP16_MODULES):
        for p in layer.parameters():
            p.data = p.data.half()


def convert_module_to_f32(layer: nn.Module) -> None:
    if isinstance(layer, FP16_MODULES):
        for p in layer.parameters():
            p.data = p.data.float()


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module
