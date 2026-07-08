import torch

from config import CLASS_NAMES, IMG_SIZE, MODELS_DIR
from image_process import preprocess_image_path
from model import build_model
from threshold_tuner import load_threshold


def load_trained_model(model_path=None, device=None):
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pth"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def predict(image_path, model=None, device=None, threshold=None):
    if model is None:
        model, device = load_trained_model(device=device)

    if threshold is None:
        threshold = load_threshold()

    tensor = preprocess_image_path(image_path, size=IMG_SIZE).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    label = 1 if prob >= threshold else 0
    class_name = CLASS_NAMES[label]
    confidence = prob if label == 1 else 1 - prob
    return class_name, confidence, prob


def predict_batch(image_paths, model=None, device=None, threshold=None, batch_size=32):
    if model is None:
        model, device = load_trained_model(device=device)

    if threshold is None:
        threshold = load_threshold()

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = [
            preprocess_image_path(path, size=IMG_SIZE) for path in batch_paths
        ]
        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        for path, prob in zip(batch_paths, probs):
            label = 1 if prob >= threshold else 0
            class_name = CLASS_NAMES[label]
            confidence = prob if label == 1 else 1 - prob
            results.append((path, class_name, float(confidence), float(prob)))

    return results
