import os, pathlib, re
from datasets import load_dataset
from PIL import Image

OUT_DIR = pathlib.Path("./pp2021_binary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_safe_name(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(txt))

def has_healthy(label_names) -> bool:
    if isinstance(label_names, (list, tuple)):
        return any(re.search(r"healthy", ln, re.I) for ln in label_names)
    return bool(re.search(r"healthy", str(label_names), re.I))

def get_label_names(example, features):
    if "labels" in example:
        ids = example["labels"]
        names = features["labels"].feature.names
        return [names[i] for i in ids]
    elif "label" in example:
        idx = example["label"]
        names = features["label"].names
        return names[idx]
    else:
        return example.get("category", "unknown")

def export_split(ds_split, split_name):
    print(f"Exporting split: {split_name} ({len(ds_split)} samples)")
    feats = ds_split.features
    ok_dir     = OUT_DIR / split_name / "OK"
    stress_dir = OUT_DIR / split_name / "stress"
    ok_dir.mkdir(parents=True, exist_ok=True)
    stress_dir.mkdir(parents=True, exist_ok=True)
    
    for i, ex in enumerate(ds_split):
        img = ex["image"]
        labels = get_label_names(ex, feats)
        target = "OK" if has_healthy(labels) else "stress"
        base = to_safe_name(ex.get("image_id", f"{split_name}_{i}"))
        fp = (ok_dir if target == "OK" else stress_dir) / f"{base}.jpg"
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(fp, format="JPEG", quality=95)
    print(f"Done: {split_name} → {ok_dir} , {stress_dir}")

def main():
    ds = load_dataset("timm/plant-pathology-2021")

    split_names = [k for k in ds.keys()]
    print("Found splits:", split_names)

    if set(split_names) == {"train"}:
        ds = ds["train"].train_test_split(test_size=0.15, seed=42, stratify_by_column="label" if "label" in ds["train"].features else None)
        ds = {"train": ds["train"], "validation": ds["test"]}
        split_names = ["train", "validation"]

    for s in split_names:
        export_split(ds[s], s)

    print("\nExport complete. Folder layout:")
    print(OUT_DIR.resolve())

if __name__ == "__main__":
    main()
