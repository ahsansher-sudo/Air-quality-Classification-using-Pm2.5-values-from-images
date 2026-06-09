# ==============================================================
# PHASE 4 - CLASSIFICATION EVALUATION & BASELINES
# IAQ Image Classification (DINOv3 + Temp/Humidity fusion)
# ==============================================================
# Evaluates the trained multimodal IAQ classifier on the held-out
# TEST set and compares it against three reference baselines:
#
#   1. Majority-class   -- always predict the most common training class
#   2. Random-init head -- same architecture, untrained weights
#   3. ResNet50 backbone-- swap DINOv3 for ImageNet ResNet50, same
#                          fusion + training recipe (fair backbone test)
#
# Metric set (classification): accuracy, macro-F1, weighted-F1,
# plus a full per-class precision/recall/F1 report and confusion
# matrices.
#
# Requires (produced by "Model Training Classification.py"):
#   outputs/feature_cache/test_features_cls_v2.pt
#   outputs/feature_cache/train_features_cls_v2.pt
#   outputs/tab_scaler_cls_v2.pt
#   outputs/best_cls_model_v2.pth
# ==============================================================

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)


# ==============================================================
# SECTION 1: CONFIGURATION
# ==============================================================

BASE_DIR   = Path(r"d:\Ahsan DS project\Dataset")
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR  = OUTPUT_DIR / "feature_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 16
EPOCHS      = 50           # for training the ResNet50 baseline head
LR          = 1e-4
NUM_CLASSES = 4
TAB_DIM     = 3
CLASS_NAMES = ["Good", "Moderate", "Unhealthy-SG", "Unhealthy"]
FAKE_TAB_DATE   = "7_24_data"        # no real sensors -> tab features zeroed
PM25_THRESHOLDS = [12.0, 35.4, 55.4]

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ==============================================================
# SECTION 2: IAQ LABEL CONVERSION
# PM2.5 reading -> class index (0-3) via EPA breakpoints.
# Used only to build labels, never fed to the model.
# ==============================================================

def pm25_to_class(pm25: float) -> int:
    for i, thresh in enumerate(PM25_THRESHOLDS):
        if pm25 <= thresh:
            return i
    return len(PM25_THRESHOLDS)


# ==============================================================
# SECTION 3: MODEL ARCHITECTURE (must match the training script)
# Multimodal head: fuses image features [img_dim] with the 3 tab
# features, then 3 linear layers -> class logits.
# img_dim defaults to 768 (DINOv3); the ResNet50 baseline uses 2048.
# ==============================================================

class MultimodalClassHead(nn.Module):
    def __init__(self, num_classes: int = 4, img_dim: int = 768, tab_dim: int = 3):
        super().__init__()
        fusion_dim = img_dim + tab_dim
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, img_feat: torch.Tensor, tab_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([img_feat, tab_feat], dim=1))


# ==============================================================
# SECTION 4: CLASSIFICATION METRICS HELPER
# Returns accuracy, macro-F1 and weighted-F1 in one dict.
# ==============================================================

def clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "macro_f1":    f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                average="weighted", zero_division=0),
    }


# ==============================================================
# SECTION 5: TABULAR NORMALISATION (apply saved train-set scaler)
# Standardises temp/humidity using the mean/std saved during
# training. Only rows with a real sensor (mask==1) are scaled;
# zeroed 7_24 rows are left untouched.
# ==============================================================

def apply_tab_norm(tab: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    out  = tab.clone()
    mask = tab[:, 2] == 1.0
    out[mask, :2] = (tab[mask, :2] - mean) / std
    return out


# ==============================================================
# SECTION 6: RESNET50 BASELINE -- FEATURE EXTRACTION
# Runs each image through ImageNet-pretrained ResNet50 (2048-dim
# avg-pool output) and builds matching tab features + labels, so
# the same multimodal head can be trained on a different backbone.
# ==============================================================

@torch.no_grad()
def extract_resnet_features(df: pd.DataFrame, device: str) -> tuple:
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()                  # drop classifier -> 2048-dim features
    resnet.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats, tabs, labels = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ResNet50 features"):
        img    = Image.open(row["image_path"]).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        feats.append(resnet(tensor).squeeze(0).cpu())

        if row["date"] == FAKE_TAB_DATE:
            tabs.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        else:
            tabs.append(torch.tensor([row["temperature"], row["humidity"], 1.0],
                                     dtype=torch.float32))
        labels.append(torch.tensor(pm25_to_class(row["pm25"]), dtype=torch.long))

    del resnet
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(feats), torch.stack(tabs), torch.stack(labels)


# ==============================================================
# SECTION 7: MAIN EVALUATION PIPELINE
# ==============================================================

def main():

    # ── Check required artefacts exist ────────────────────────
    test_cache  = CACHE_DIR / "test_features_cls_v2.pt"
    scaler_path = OUTPUT_DIR / "tab_scaler_cls_v2.pt"
    model_path  = OUTPUT_DIR / "best_cls_model_v2.pth"
    for p in (test_cache, scaler_path, model_path):
        if not p.exists():
            print(f"\nERROR: missing {p.name}. Run 'Model Training Classification.py' first.")
            sys.exit(1)

    train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
    test_df  = pd.read_csv(OUTPUT_DIR / "test.csv")

    # ── Load cached DINOv3 TEST features + saved scaler ───────
    sd = torch.load(test_cache, weights_only=True)
    test_img, test_tab, test_labels = sd["img"], sd["tab"], sd["labels"]
    scaler = torch.load(scaler_path, weights_only=True)
    test_tab_n = apply_tab_norm(test_tab, scaler["mean"], scaler["std"])
    y_true = test_labels.numpy()
    print(f"\nTest features: {test_img.shape}   ({len(test_df)} images)")

    test_loader = DataLoader(
        TensorDataset(test_img, test_tab_n, test_labels),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    results = {}          # model name -> metrics dict
    pred_store = {}       # model name -> predictions (for confusion matrices)

    # ── 1. Trained DINOv3 multimodal classifier ───────────────
    print("\n" + "=" * 60)
    print("Evaluating trained DINOv3 classifier ...")
    ckpt  = torch.load(model_path, weights_only=True)
    model = MultimodalClassHead(num_classes=NUM_CLASSES, img_dim=768, tab_dim=TAB_DIM).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for img_f, tab_f, _ in test_loader:
            preds.append(model(img_f.to(DEVICE), tab_f.to(DEVICE)).argmax(1).cpu())
    y_dinov3 = torch.cat(preds).numpy()
    results["DINOv3 (ours)"]   = clf_metrics(y_true, y_dinov3)
    pred_store["DINOv3 (ours)"] = y_dinov3

    print(f"\n-- DINOv3 Classification Report ----------------------")
    print(classification_report(y_true, y_dinov3, labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES, zero_division=0))

    # ── 2. Baseline A: Majority class ─────────────────────────
    train_classes = train_df["pm25"].apply(pm25_to_class)
    majority_class = int(train_classes.mode()[0])
    y_majority = np.full_like(y_true, majority_class)
    results["Majority class"]   = clf_metrics(y_true, y_majority)
    pred_store["Majority class"] = y_majority
    print(f"Majority class in training set: {majority_class} ({CLASS_NAMES[majority_class]})")

    # ── 3. Baseline B: Random-init head (DINOv3 features) ─────
    rand_head = MultimodalClassHead(num_classes=NUM_CLASSES, img_dim=768, tab_dim=TAB_DIM).to(DEVICE)
    rand_head.eval()
    preds = []
    with torch.no_grad():
        for img_f, tab_f, _ in test_loader:
            preds.append(rand_head(img_f.to(DEVICE), tab_f.to(DEVICE)).argmax(1).cpu())
    y_rand = torch.cat(preds).numpy()
    results["Random-init head"]   = clf_metrics(y_true, y_rand)
    pred_store["Random-init head"] = y_rand

    # ── 4. Baseline C: ResNet50 backbone + trained head ───────
    print("\n" + "=" * 60)
    print("Building ResNet50 baseline (extract -> train head -> evaluate) ...")

    rtr_path = CACHE_DIR / "resnet_train_features_cls.pt"
    rte_path = CACHE_DIR / "resnet_test_features_cls.pt"
    if rtr_path.exists() and rte_path.exists():
        d = torch.load(rtr_path, weights_only=True)
        r_train_img, r_train_tab, r_train_labels = d["img"], d["tab"], d["labels"]
        d = torch.load(rte_path, weights_only=True)
        r_test_img,  r_test_tab,  r_test_labels  = d["img"], d["tab"], d["labels"]
    else:
        r_train_img, r_train_tab, r_train_labels = extract_resnet_features(train_df, DEVICE)
        r_test_img,  r_test_tab,  r_test_labels  = extract_resnet_features(test_df, DEVICE)
        torch.save({"img": r_train_img, "tab": r_train_tab, "labels": r_train_labels}, rtr_path)
        torch.save({"img": r_test_img,  "tab": r_test_tab,  "labels": r_test_labels},  rte_path)

    # Reuse the same saved temp/humidity scaler (tab values are backbone-independent)
    r_train_tab_n = apply_tab_norm(r_train_tab, scaler["mean"], scaler["std"])
    r_test_tab_n  = apply_tab_norm(r_test_tab,  scaler["mean"], scaler["std"])

    # Same class-weighted CrossEntropy recipe as the DINOv3 head
    counts = torch.bincount(r_train_labels, minlength=NUM_CLASSES).float()
    class_weights = (counts.sum() / (NUM_CLASSES * counts.clamp(min=1))).to(DEVICE)

    resnet_head = MultimodalClassHead(num_classes=NUM_CLASSES, img_dim=2048, tab_dim=TAB_DIM).to(DEVICE)
    opt  = torch.optim.Adam(resnet_head.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    r_train_loader = DataLoader(
        TensorDataset(r_train_img, r_train_tab_n, r_train_labels),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    for _ in tqdm(range(EPOCHS), desc="ResNet head"):
        resnet_head.train()
        for img_f, tab_f, labels in r_train_loader:
            img_f, tab_f, labels = img_f.to(DEVICE), tab_f.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = crit(resnet_head(img_f, tab_f), labels)
            loss.backward()
            opt.step()

    resnet_head.eval()
    r_test_loader = DataLoader(
        TensorDataset(r_test_img, r_test_tab_n, r_test_labels),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for img_f, tab_f, _ in r_test_loader:
            preds.append(resnet_head(img_f.to(DEVICE), tab_f.to(DEVICE)).argmax(1).cpu())
    y_resnet = torch.cat(preds).numpy()
    results["ResNet50"]   = clf_metrics(y_true, y_resnet)
    pred_store["ResNet50"] = y_resnet

    # ── 5. Results table ──────────────────────────────────────
    order = ["DINOv3 (ours)", "ResNet50", "Random-init head", "Majority class"]
    results_df = pd.DataFrame([
        {"Model": m,
         "Accuracy":    results[m]["accuracy"],
         "Macro-F1":    results[m]["macro_f1"],
         "Weighted-F1": results[m]["weighted_f1"]}
        for m in order
    ]).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY -- Test Set (sorted by accuracy)")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_df.to_csv(OUTPUT_DIR / "classification_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'classification_results.csv'}")

    # ── 6. Confusion matrices: ours vs ResNet50 ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ["DINOv3 (ours)", "ResNet50"]):
        cm = confusion_matrix(y_true, pred_store[name], labels=list(range(NUM_CLASSES)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(f"{name}\nacc={results[name]['accuracy']:.3f}  "
                     f"macro-F1={results[name]['macro_f1']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.suptitle("Confusion Matrices -- Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eval_confusion_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'eval_confusion_comparison.png'}")

    # ── 7. Bar chart: accuracy + macro-F1 across models ───────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    palette = ["#2A7CB8", "#E07B39", "#E04040", "#999999"]
    for ax, metric, title in [(axes[0], "Accuracy", "Accuracy"),
                              (axes[1], "Macro-F1", "Macro-F1")]:
        vals   = results_df[metric].tolist()
        labels = results_df["Model"].tolist()
        bars   = ax.bar(labels, vals, color=palette[:len(labels)], edgecolor="white", width=0.6)
        ax.set_title(title, fontweight="bold"); ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("Model Comparison -- HVAQ Test Set (IAQ Classification)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eval_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'eval_model_comparison.png'}")

    print("\n" + "=" * 60)
    print("Phase 4 Complete")
    print("=" * 60)
    best = results_df.iloc[0]
    print(f"  Best model    : {best['Model']}")
    print(f"  Test accuracy : {best['Accuracy']:.4f}")
    print(f"  Macro-F1      : {best['Macro-F1']:.4f}")
    return results_df


if __name__ == "__main__":
    main()
