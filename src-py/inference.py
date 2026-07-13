import torch
from torchvision import transforms

from config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MODELS_DIR
from image_process import preprocess_image_path
from model import build_model
from threshold_tuner import load_threshold


# TTA 变换列表：原图 + 水平翻转 + 轻微旋转
tta_transforms = [
    lambda img: img,  # 原图
    transforms.RandomHorizontalFlip(p=1.0),  # 水平翻转
    lambda img: transforms.functional.rotate(img, angle=5),  # 顺时针 5°
    lambda img: transforms.functional.rotate(img, angle=-5),  # 逆时针 5°
]


def load_trained_model(model_path=None, device=None, use_ema=False):
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pth"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 如果 checkpoint 包含 EMA 且用户要求使用 EMA
    if use_ema and checkpoint.get("ema_state_dict"):
        from model import build_model
        ema_model = build_model(pretrained=False)
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
        ema_model.to(device)
        ema_model.eval()
        return ema_model, device

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
        probs = []
        with torch.no_grad():
            for tfm in tta_transforms:
                aug_tensor = tfm(tensor.clone()).unsqueeze(0).to(device)
                logit = model(aug_tensor)
                prob = torch.sigmoid(logit).item()
                probs.append(prob)
        prob = sum(probs) / len(probs)
    else:
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()

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
