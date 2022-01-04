# Author: Jonathan Donzallaz

import pl_bolts.models.vision.unet as plt_unet
import torch
#from solarnet.models.simple_cnn import SimpleCNN
from torch import nn
from torchvision import models

MODELS_OUTPUT_SIZES = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "squeezenet1_1": 492032,
}


def get_backbone(name: str, channels: int, **kwargs):
    """
    Build a backbone model and return it with its output size.

    :param name: The name of the backbone model to build
    :param channels: The number of channel for the first layer of the model
    :return: a tuple with a nn.Module and its output size
    """


    if name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        backbone = getattr(models, name)()
        backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.fc = nn.Identity()
        output_size = MODELS_OUTPUT_SIZES[name]

    else:
        raise RuntimeError(f"Backbone model {name} unsupported.")

    return backbone, output_size
