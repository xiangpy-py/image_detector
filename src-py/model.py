import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import MODEL_ARCH, RANDOM_SEED


class SEAttention(nn.Module):
    """Squeeze-and-Excitation 注意力模块。

    对输入特征图做通道级别的注意力加权，提升对关键区域的关注能力。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetWithAttention(nn.Module):
    """ResNet-50 + SE Attention + Dropout 包装器。"""

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # 提取除 fc 外的所有层
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # 在 layer4 后添加 SE 注意力
        self.se = SEAttention(2048, reduction=16)

        # 添加 Dropout + 分类头
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(2048, 1)

        # 初始化新层的权重
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def _build_efficientnet_b0(pretrained=True, dropout=0.3):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(in_features, 1),
    )
    nn.init.xavier_uniform_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)
    return model


def _build_efficientnet_b4(pretrained=True, dropout=0.3):
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(in_features, 1),
    )
    nn.init.xavier_uniform_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)
    return model


def _build_convnext_tiny(pretrained=True, dropout=0.3):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],  # Flatten
        model.classifier[1],  # LayerNorm
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(in_features, 1),
    )
    nn.init.xavier_uniform_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)
    return model


def build_model(pretrained=True, arch=None, dropout=None):
    """构建模型，支持多种架构选择。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重
        arch: 模型架构名称，None 则使用 config.MODEL_ARCH
        dropout: Dropout 比率，None 则使用 config.DROPOUT_RATE

    Returns:
        nn.Module: 构建好的模型
    """
    from config import DROPOUT_RATE, MODEL_ARCH

    arch = arch or MODEL_ARCH
    dropout = dropout if dropout is not None else DROPOUT_RATE

    if arch == "resnet50":
        model = ResNetWithAttention(pretrained=pretrained, dropout=dropout)
    elif arch == "efficientnet_b0":
        model = _build_efficientnet_b0(pretrained=pretrained, dropout=dropout)
    elif arch == "efficientnet_b4":
        model = _build_efficientnet_b4(pretrained=pretrained, dropout=dropout)
    elif arch == "convnext_tiny":
        model = _build_convnext_tiny(pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"不支持的模型架构: {arch}")

    return model


def set_seed(seed=RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
