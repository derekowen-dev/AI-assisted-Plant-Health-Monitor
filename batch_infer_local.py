import argparse
from pathlib import Path

import torch
import torchvision as tv
from PIL import Image

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = tv.models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 2)

    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    class_names = ckpt.get("class_names", ["OK", "stress"])
    return model, class_names

def preprocess_image(img_path: Path, img_size: int = 224) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    transform = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean, std),
    ])

    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(
        description="Run leaf stress model on one or more images."
    )
    parser.add_argument(
        "--ckpt",
        default="models/leaf_binary.pt",
        help="Path to checkpoint file (default: models/leaf_binary.pt)",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image file(s) to run inference on",
    )

    args = parser.parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(ckpt_path, device)

    print(f"Using device: {device}")
    print(f"Classes: {class_names}")
    print("-" * 60)

    for img_str in args.images:
        img_path = Path(img_str)
        if not img_path.exists():
            print(f"[SKIP] {img_path} (file not found)")
            continue

        x = preprocess_image(img_path).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        pred_name = class_names[pred_idx]

        if "stress" in class_names:
            stress_idx = class_names.index("stress")
        else:
            stress_idx = 1

        p_stress = float(probs[stress_idx])
        p_ok = float(1.0 - p_stress)

        print(f"Image: {img_path.name}")
        print(f"  Predicted class: {pred_name}")
        print(f"  P(OK):     {p_ok:.3f}")
        print(f"  P(stress): {p_stress:.3f}")
        print("-" * 60)

if __name__ == "__main__":
    main()