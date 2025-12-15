#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torchvision as tv
from PIL import Image

def load_model(ckpt_path: Path, device: torch.device):

    ckpt = torch.load(ckpt_path, map_location=device)

    model = tv.models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 2)

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    class_names = ckpt.get("class_names", ["OK", "stress"])
    return model, class_names

def preprocess(img_path: Path, img_size: int = 224) -> torch.Tensor:

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tf = tv.transforms.Compose(
        [
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(img_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std),
        ]
    )

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    return x

def main():
    ap = argparse.ArgumentParser(description="Leaf health binary inference (OK vs stress)")
    ap.add_argument(
        "--ckpt",
        default="models/leaf_binary.pt",
        help="Path to model checkpoint (default: models/leaf_binary.pt)",
    )
    ap.add_argument("image", help="Path to input image")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    img_path = Path(args.image).expanduser().resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    device = torch.device("cpu")

    try:
        torch.set_num_threads(2)
    except Exception:
        pass

    model, class_names = load_model(ckpt_path, device)
    x = preprocess(img_path).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_name = class_names[pred_idx]

    if "stress" in class_names:
        stress_idx = class_names.index("stress")
    else:
        stress_idx = 1
    stress_prob = float(probs[stress_idx])

    result = {
        "pred_class": pred_name,
        "probabilities": {
            class_names[0]: float(probs[0]),
            class_names[1]: float(probs[1]),
        },
        "p_stress": stress_prob,
        "image": str(img_path),
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
