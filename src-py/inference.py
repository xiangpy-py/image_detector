"""模型推理：加载训练好的模型、对单张或多张图像做预测（含可选 TTA）。"""

import torch

from config import CLASS_NAMES, IMG_SIZE, MODELS_DIR
from image_process import preprocess_image_path
from model import build_model
from model_utils import find_model_path
from threshold_tuner import load_threshold


# TTA 变换列表：原图 + 水平翻转 + 两种轻微旋转。
# 注意：所有变换作用于已 Normalize 的 224×224 tensor，
# 几何变换在归一化后做是安全的（线性变换），但要注意：
# 1. flip/rotate 只影响张量的空间维度，通道统计不变；
# 2. 旋转 5° 在 224 上等价于大约 19 像素的偏移，肉眼几乎不可见，
#    但对模型决策边界有一定扰动，能起到 TTA 的作用。
def _tta_augment(tensor: torch.Tensor) -> list[torch.Tensor]:
    """对单张预处理后的 tensor 应用 TTA，返回一组变体。"""
    return [
        tensor,
        torch.flip(tensor, dims=[-1]),  # 水平翻转
        torch.rot90(tensor, k=1, dims=[-2, -1]),  # 逆时针 90°
        torch.rot90(tensor, k=-1, dims=[-2, -1]),  # 顺时针 90°
    ]


def _infer_arch_from_state_dict(state_dict) -> str | None:
    """从 state_dict 的 key 推断模型架构。

    返回: 架构名（与 model.build_model 接受的 arch 参数一致）或 None。
    """
    keys = list(state_dict.keys())

    if any("features.denseblock" in k for k in keys):
        return "densenet121"
    if any(k.startswith("se.") for k in keys) or any("layer4.2." in k for k in keys):
        return "resnet50"
    if any("_expand_conv" in k for k in keys):
        # EfficientNet 通用特征；通过分类头 in_features 进一步区分 B0/B4
        head_weight = next(
            (k for k in keys if k.startswith("classifier.") and k.endswith(".weight")),
            None,
        )
        if head_weight and head_weight in state_dict:
            in_features = state_dict[head_weight].shape[1]
            if in_features == 1280:
                return "efficientnet_b0"
            if in_features == 1792:
                return "efficientnet_b4"
        return "efficientnet_b0"
    if any(k.startswith("stages") for k in keys):
        return "convnext_tiny"
    return None


def load_trained_model(model_path=None, device=None, use_ema=False):
    """加载训练好的模型。

    Args:
        model_path: 模型文件路径，None 时自动从 MODELS_DIR 找最新
        device: torch device，None 时自动选 cuda/cpu
        use_ema: 若 checkpoint 含 ema_state_dict，是否加载 EMA 权重

    Returns:
        (model, device) 元组
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = find_model_path()

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    arch = checkpoint.get("arch")
    if arch is None:
        arch = _infer_arch_from_state_dict(checkpoint["model_state_dict"])

    if use_ema and checkpoint.get("ema_state_dict"):
        ema_arch = checkpoint.get("arch") or _infer_arch_from_state_dict(
            checkpoint["ema_state_dict"]
        )
        ema_model = build_model(pretrained=False, arch=ema_arch)
        ema_model.load_state_dict(
            checkpoint["ema_state_dict"], strict=(ema_arch is not None)
        )
        ema_model.to(device)
        ema_model.eval()
        return ema_model, device

    model = build_model(pretrained=False, arch=arch)
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=(arch is not None)
    )
    model.to(device)
    model.eval()
    return model, device


def predict(image_path, model=None, device=None, threshold=None, use_tta=False):
    """单张图像预测。

    Args:
        image_path: 图像路径
        model: 模型实例，None 则自动加载
        device: torch device
        threshold: 分类阈值，None 则加载保存的最优阈值
        use_tta: 是否使用 Test-Time Augmentation

    Returns:
        (class_name, confidence, prob)
    """
    if model is None:
        model, device = load_trained_model(device=device)

    if threshold is None:
        threshold = load_threshold()

    tensor = preprocess_image_path(image_path, size=IMG_SIZE)

    if use_tta:
        variants = _tta_augment(tensor)
        batch = torch.stack(variants, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        prob = float(probs.mean())
    else:
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            prob = float(torch.sigmoid(logit).item())

    label = 1 if prob >= threshold else 0
    class_name = CLASS_NAMES[label]
    confidence = prob if label == 1 else 1 - prob
    return class_name, confidence, prob


def _infer_batch(batch_paths, model, device, use_tta=False):
    """推理单个 batch，确保 GPU 张量在该函数返回后立即释放。"""
    if use_tta:
        all_probs = []
        for path in batch_paths:
            _, _, prob = predict(path, model, device, use_tta=True)
            all_probs.append(prob)
        return all_probs

    batch = torch.stack(
        [preprocess_image_path(path, size=IMG_SIZE) for path in batch_paths]
    ).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    # 返回 CPU numpy 后，局部 GPU 张量随函数返回失去引用，可被 GC 回收
    return probs


def predict_batch(image_paths, model=None, device=None, threshold=None, batch_size=32, use_tta=False):
    if model is None:
        model, device = load_trained_model(device=device)

    if threshold is None:
        threshold = load_threshold()

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        probs = _infer_batch(batch_paths, model, device, use_tta=use_tta)

        for path, prob in zip(batch_paths, probs):
            label = 1 if prob >= threshold else 0
            class_name = CLASS_NAMES[label]
            confidence = prob if label == 1 else 1 - prob
            results.append((path, class_name, float(confidence), float(prob)))

    return results
