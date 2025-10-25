"""
Model definitions for the project.

For simplicity, we expose just one function: build_convnext_tiny, which creates
the ConvNeXt-Tiny model using the timm library. This is a small, modern CNN
that works well on 2D images and is available with ImageNet pretraining.

We keep this file short and heavily commented so it is easy to read.
"""

# ========= IMPORTS =========
import timm
import torch.nn as nn



# We use the ConvNeXt-Tiny model from the timm library. It is small, fast to
# train, and has good performance. The 'pretrained' flag loads ImageNet weights
# which usually improves results when fine-tuning on medical images.

def build_convnext_tiny(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create a ConvNeXt-Tiny model.

    Args:
      num_classes: number of output classes for classification (2 for AD/CN)
      pretrained: whether to load ImageNet-pretrained weights

    Returns:
      A torch.nn.Module that predicts logits of shape [batch_size, num_classes].
    """
    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


