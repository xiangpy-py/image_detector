"""测试 model 模块。"""

import torch

from model import build_model, set_seed


def test_build_model_output_dim():
    model = build_model(pretrained=False)
    # 检查最后一层输出维度是否为 1
    assert model.fc.out_features == 1


def test_build_model_forward():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_set_seed_determinism():
    """测试设置相同种子后随机数一致。"""
    set_seed(123)
    r1 = torch.rand(5)

    set_seed(123)
    r2 = torch.rand(5)

    assert torch.allclose(r1, r2)


def test_set_seed_different_seeds():
    """测试不同种子产生不同随机数。"""
    set_seed(123)
    r1 = torch.rand(5)

    set_seed(456)
    r2 = torch.rand(5)

    assert not torch.allclose(r1, r2)
