import io
from pathlib import Path
from flask import Flask, request, jsonify
import torch, torchvision as tv
from PIL import Image

CKPT_PATH = Path("models/leaf_binary.pt")

def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = tv.models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    class_names = ckpt.get("class_names", ["OK", "stress"])
    return model, class_names

def preprocess_pil(img, img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean, std),
    ])
    return tf(img).unsqueeze(0)

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, CLASS_NAMES = load_model(DEVICE)

@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    f = request.files["image"]
    img_bytes = f.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preprocess_pil(img).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    pred_name = CLASS_NAMES[pred_idx]

    stress_index = CLASS_NAMES.index("stress") if "stress" in CLASS_NAMES else 1
    stress_prob = float(probs[stress_index])

    return jsonify({
        "pred_class": pred_name,
        "probabilities": {
            CLASS_NAMES[0]: float(probs[0]),
            CLASS_NAMES[1]: float(probs[1]),
        },
        "p_stress": stress_prob,
    })

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=6000)
