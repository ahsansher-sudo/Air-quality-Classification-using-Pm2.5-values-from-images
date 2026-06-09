# ==============================================================
# PHASE 5 - VISUALISATIONS & FINAL REPORT (CLASSIFICATION)
# IAQ Image Classification (DINOv3 + Temp/Humidity fusion)
# ==============================================================
# Produces:
#   1. t-SNE of DINOv3 features (coloured by IAQ class, and by split)
#   2. Per-class accuracy bar chart (test set)
#   3. Per-date accuracy breakdown (test set)
#   4. Misclassified-sample table -> misclassified_test.csv
#   5. Clean learning-curve re-plot from training history
#   6. project_summary.md -- final written classification report
#
# Requires (produced by "Model Training Classification.py"):
#   outputs/feature_cache/{train,val,test}_features_cls_v2.pt
#   outputs/tab_scaler_cls_v2.pt
#   outputs/best_cls_model_v2.pth
#   outputs/training_history_cls.csv
# Optional (produced by "Evaluation.py"):
#   outputs/classification_results.csv  -> adds the baseline table
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from datetime import datetime


# ==============================================================
# SECTION 1: CONFIGURATION
# ==============================================================

BASE_DIR   = Path(r"d:\Ahsan DS project\Dataset")
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR  = OUTPUT_DIR / "feature_cache"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 4
TAB_DIM     = 3
CLASS_NAMES = ["Good", "Moderate", "Unhealthy-SG", "Unhealthy"]
FAKE_TAB_DATE   = "7_24_data"
PM25_THRESHOLDS = [12.0, 35.4, 55.4]

print(f"Device : {DEVICE}")


# ==============================================================
# SECTION 2: IAQ LABEL CONVERSION
# ==============================================================

def pm25_to_class(pm25: float) -> int:
    for i, thresh in enumerate(PM25_THRESHOLDS):
        if pm25 <= thresh:
            return i
    return len(PM25_THRESHOLDS)


# ==============================================================
# SECTION 3: MODEL ARCHITECTURE (must match the training script)
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
# SECTION 4: TABULAR NORMALISATION (apply saved train-set scaler)
# ==============================================================

def apply_tab_norm(tab: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    out  = tab.clone()
    mask = tab[:, 2] == 1.0
    out[mask, :2] = (tab[mask, :2] - mean) / std
    return out


# ==============================================================
# SECTION 5: LOAD FEATURES, SCALER, MODEL -> GENERATE PREDICTIONS
# Train cache holds augmented copies; only the first N originals
# (one per row of train.csv) are used for the t-SNE plot.
# ==============================================================

print("\nLoading cached features, scaler and model ...")

train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
val_df   = pd.read_csv(OUTPUT_DIR / "val.csv")
test_df  = pd.read_csv(OUTPUT_DIR / "test.csv")
n_train_orig = len(train_df)

td = torch.load(CACHE_DIR / "train_features_cls_v2.pt", weights_only=True)
vd = torch.load(CACHE_DIR / "val_features_cls_v2.pt",   weights_only=True)
sd = torch.load(CACHE_DIR / "test_features_cls_v2.pt",  weights_only=True)
scaler = torch.load(OUTPUT_DIR / "tab_scaler_cls_v2.pt", weights_only=True)

# Originals only for train (skip augmented copies)
train_img    = td["img"][:n_train_orig]
train_tab    = td["tab"][:n_train_orig]
train_labels = td["labels"][:n_train_orig]
val_img,  val_tab,  val_labels  = vd["img"], vd["tab"], vd["labels"]
test_img, test_tab, test_labels = sd["img"], sd["tab"], sd["labels"]

ckpt  = torch.load(OUTPUT_DIR / "best_cls_model_v2.pth", weights_only=True)
model = MultimodalClassHead(num_classes=NUM_CLASSES, img_dim=768, tab_dim=TAB_DIM).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()


def predict(img_feats: torch.Tensor, tab_feats: torch.Tensor) -> np.ndarray:
    tab_n = apply_tab_norm(tab_feats, scaler["mean"], scaler["std"])
    ld = DataLoader(TensorDataset(img_feats, tab_n), batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for img_f, tab_f in ld:
            preds.append(model(img_f.to(DEVICE), tab_f.to(DEVICE)).argmax(1).cpu())
    return torch.cat(preds).numpy()


test_preds = predict(test_img, test_tab)
y_test     = test_labels.numpy()
test_acc   = accuracy_score(y_test, test_preds)


# ==============================================================
# SECTION 6: t-SNE FEATURE VISUALISATION
# Projects the 768-dim DINOv3 features to 2-D, coloured first by
# true IAQ class then by data split, to show class separability.
# ==============================================================

print("Computing t-SNE projection ...")
all_img    = torch.cat([train_img, val_img, test_img], dim=0).numpy()
all_labels = torch.cat([train_labels, val_labels, test_labels], dim=0).numpy()
split_ids  = (["train"] * len(train_labels) +
              ["val"]   * len(val_labels)   +
              ["test"]  * len(test_labels))

tsne = TSNE(n_components=2, perplexity=min(20, len(all_img) - 1),
            max_iter=1000, random_state=42, init="pca")
emb = tsne.fit_transform(all_img)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: colour = IAQ class
class_palette = ["#2E933C", "#E8B500", "#E07B39", "#D7263D"]
for c, name in enumerate(CLASS_NAMES):
    m = all_labels == c
    axes[0].scatter(emb[m, 0], emb[m, 1], c=class_palette[c], label=name,
                    s=55, alpha=0.85, edgecolors="white", linewidths=0.4)
axes[0].set_title("t-SNE -- DINOv3 Features (colour = IAQ class)", fontweight="bold")
axes[0].legend(fontsize=9); axes[0].axis("off")

# Plot B: colour = split
split_colors = {"train": "steelblue", "val": "darkorange", "test": "tomato"}
for split, color in split_colors.items():
    m = [i for i, s in enumerate(split_ids) if s == split]
    axes[1].scatter(emb[m, 0], emb[m, 1], c=color, label=split.capitalize(),
                    s=55, alpha=0.85, edgecolors="white", linewidths=0.4)
axes[1].set_title("t-SNE -- DINOv3 Features (colour = split)", fontweight="bold")
axes[1].legend(fontsize=10); axes[1].axis("off")

plt.suptitle("DINOv3 Feature Space -- HVAQ IAQ Dataset", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tsne_features_cls.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'tsne_features_cls.png'}")


# ==============================================================
# SECTION 7: MISCLASSIFICATION ANALYSIS (test set)
# Lists every test image with its true vs predicted class and
# flags the mistakes, saved to misclassified_test.csv.
# ==============================================================

analysis = test_df.copy().reset_index(drop=True)
analysis["true_class"] = [CLASS_NAMES[i] for i in y_test]
analysis["pred_class"] = [CLASS_NAMES[i] for i in test_preds]
analysis["correct"]    = y_test == test_preds
analysis_out = analysis[["image_path", "date", "pm25", "true_class", "pred_class", "correct"]]
analysis_out.to_csv(OUTPUT_DIR / "misclassified_test.csv", index=False)

n_wrong = int((~analysis["correct"]).sum())
print(f"\nMisclassified {n_wrong}/{len(analysis)} test images:")
for _, r in analysis[~analysis["correct"]].iterrows():
    print(f"  {Path(r['image_path']).name:30s}  true={r['true_class']:13s} -> pred={r['pred_class']}")


# ==============================================================
# SECTION 8: PER-CLASS + PER-DATE ACCURACY (test set)
# ==============================================================

# Per-class accuracy (recall) bar chart
per_class_acc = []
for c in range(NUM_CLASSES):
    m = y_test == c
    per_class_acc.append(float((test_preds[m] == c).mean()) if m.sum() else 0.0)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(CLASS_NAMES, per_class_acc, color=class_palette, edgecolor="white")
ax.set_ylim(0, 1); ax.set_ylabel("Accuracy (recall)")
ax.set_title(f"Per-Class Accuracy -- Test Set (overall {test_acc:.1%})", fontweight="bold")
for bar, v in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_class_accuracy_cls.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'per_class_accuracy_cls.png'}")

# Per-date accuracy
print("\nPer-date accuracy (test set):")
per_date = {}
for date_folder in analysis["date"].unique():
    m   = analysis["date"] == date_folder
    acc = float(analysis.loc[m, "correct"].mean())
    per_date[date_folder] = {"n": int(m.sum()), "acc": acc}
    print(f"  {date_folder}: n={int(m.sum())}  acc={acc:.3f}")


# ==============================================================
# SECTION 9: LEARNING CURVES RE-PLOT (clean style)
# ==============================================================

history_df = pd.read_csv(OUTPUT_DIR / "training_history_cls.csv")
best_epoch = int(history_df["val_acc"].idxmax()) + 1
best_acc   = history_df["val_acc"].max()
ep = range(1, len(history_df) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(ep, history_df["train_loss"], color="steelblue", linewidth=1.5)
ax1.axvline(best_epoch, color="tomato", linestyle="--", linewidth=1,
            label=f"best epoch {best_epoch}")
ax1.set_title("Training Loss (CrossEntropy)", fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

ax2.plot(ep, history_df["val_acc"], color="darkorange", linewidth=1.5)
ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
            label=f"best epoch {best_epoch} (acc={best_acc:.2f})")
ax2.set_title("Validation Accuracy", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_ylim(0, 1); ax2.legend()

plt.suptitle("DINOv3 + Temp/Humidity -- Learning Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curves_cls_final.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'learning_curves_cls_final.png'}")


# ==============================================================
# SECTION 10: PROJECT SUMMARY REPORT (Markdown)
# ==============================================================

report_dict = classification_report(
    y_test, test_preds, labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES, output_dict=True, zero_division=0,
)
macro_f1    = report_dict["macro avg"]["f1-score"]
weighted_f1 = report_dict["weighted avg"]["f1-score"]

# Per-class report rows
per_class_lines = []
for name in CLASS_NAMES:
    r = report_dict[name]
    per_class_lines.append(
        f"| {name} | {int(r['support'])} | {r['precision']:.2f} | "
        f"{r['recall']:.2f} | {r['f1-score']:.2f} |"
    )

# Per-date rows
per_date_lines = [f"| {d} | {s['n']} | {s['acc']:.2f} |" for d, s in per_date.items()]

# Optional baseline comparison table from Evaluation.py
baseline_section = ""
results_csv = OUTPUT_DIR / "classification_results.csv"
if results_csv.exists():
    res_df = pd.read_csv(results_csv)
    rows = [f"| {r['Model']} | {r['Accuracy']:.3f} | {r['Macro-F1']:.3f} | {r['Weighted-F1']:.3f} |"
            for _, r in res_df.iterrows()]
    baseline_section = (
        "## 4. Baseline Comparison (Test Set)\n\n"
        "| Model | Accuracy | Macro-F1 | Weighted-F1 |\n"
        "|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        f"Both deep backbones (DINOv3 and ResNet50) far exceed the majority-class "
        f"and random-init references, so the head is clearly learning real visual "
        f"cues rather than guessing. On this {len(test_df)}-image test set the two "
        f"backbones differ by only a few images, and the ResNet50 head is retrained "
        f"fresh and unseeded on each run -- so the gap between them is within "
        f"run-to-run noise and should not be read as a stable ranking.\n\n---\n\n"
    )

# Training-set class distribution (original, un-augmented)
train_classes = train_df["pm25"].apply(pm25_to_class)
dist_lines = [f"| {CLASS_NAMES[c]} | {int((train_classes == c).sum())} |"
              for c in range(NUM_CLASSES)]

report = f"""# HVAQ IAQ Image Classification -- Project Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Task:** Classify outdoor air-quality category (IAQ) from a photo plus
temperature and humidity, using a frozen DINOv3 backbone with a small
multimodal classification head.
**Dataset:** HVAQ -- 106 matched image-sensor pairs across 3 dates.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Total images matched | 106 |
| Date folders | 3 (2019-07-24, 2019-10-19, 2019-11-10) |
| Classes | 4 (Good / Moderate / Unhealthy-SG / Unhealthy) |
| Label source | PM2.5 -> EPA breakpoints [12.0, 35.4, 55.4] |
| Train / Val / Test split | {len(train_df)} / {len(val_df)} / {len(test_df)} |
| Missing temp/humidity (7_24_data) | tab features zeroed + sensor-mask flag |
| Image resolution | 4000-4864 x 3000-3648 px -> 224x224 by processor |

**Training class distribution (original images):**

| Class | Count |
|---|---|
{chr(10).join(dist_lines)}

---

## 2. Model Architecture

**Backbone:** `facebook/dinov3-vitb16-pretrain-lvd1689m` (DINOv3 ViT-B/16, frozen)
- Feature extraction: mean-pool 196 patch tokens -> 768-dim embedding
- CLS token (index 0) and 4 register tokens (1-4) skipped

**Multimodal fusion:** image features [768] + [norm_temp, norm_humidity, sensor_mask] [3] -> [771]

**Classification head** (trained from scratch):
```
Linear(771 -> 256) -> ReLU -> Dropout(0.3)
Linear(256 -> 64)  -> ReLU
Linear(64  -> 4)
```

**Training recipe:**
- Feature caching: backbone runs once, head trains in a few seconds
- Augmentation: {len(train_df)} images x3 copies (flip, colour jitter, synthetic haze)
- Loss: class-weighted CrossEntropy (handles class imbalance)
- Optimiser: Adam (lr=1e-4), CosineAnnealingLR, 50 epochs, batch size 16
- Best epoch: {best_epoch} (val accuracy {best_acc:.4f})

---

## 3. Results -- Test Set ({len(test_df)} images)

| Metric | Value |
|---|---|
| **Accuracy** | **{test_acc:.4f}** |
| Macro-F1 | {macro_f1:.4f} |
| Weighted-F1 | {weighted_f1:.4f} |

### Per-Class Performance

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
{chr(10).join(per_class_lines)}

### Per-Date Accuracy

| Date Folder | n | Accuracy |
|---|---|---|
{chr(10).join(per_date_lines)}

---

{baseline_section}## 5. Output Files

| File | Description |
|---|---|
| `outputs/best_cls_model_v2.pth` | Trained classifier weights |
| `outputs/tab_scaler_cls_v2.pt` | Temp/humidity normalisation stats |
| `outputs/training_history_cls.csv` | Per-epoch train loss, val accuracy, LR |
| `outputs/classification_results.csv` | Test metrics for all models (Phase 4) |
| `outputs/misclassified_test.csv` | Per-image true vs predicted (Phase 5) |
| `outputs/confusion_matrix_val.png` / `_test.png` | Confusion matrices (training script) |
| `outputs/learning_curves_cls.png` | Loss + accuracy + LR (training script) |
| `outputs/learning_curves_cls_final.png` | Clean learning-curve re-plot (Phase 5) |
| `outputs/eval_confusion_comparison.png` | DINOv3 vs ResNet50 (Phase 4) |
| `outputs/eval_model_comparison.png` | Accuracy / macro-F1 bars (Phase 4) |
| `outputs/tsne_features_cls.png` | t-SNE coloured by class and split (Phase 5) |
| `outputs/per_class_accuracy_cls.png` | Per-class accuracy bars (Phase 5) |

---

## 6. Key Findings

1. **The model separates clean from polluted air reliably.** Validation accuracy
   reached {best_acc:.0%}, and the strong classes (Moderate, Unhealthy-SG) carry
   high recall on the test set.

2. **The minority classes are the weak point.** "Good" and "Unhealthy" have very
   few samples, so a single misclassified test image swings their recall sharply --
   the {len(test_df)}-image test set means each sample is ~{100/len(test_df):.0f}% of the score.

3. **Data size is the main constraint, not the method.** With only 106 matched
   pairs the ceiling is limited; the frozen-backbone + cached-feature design is the
   right choice for a dataset this small and trains in seconds.

4. **Sensor fusion is handled honestly.** The 7_24_data date has no real
   temperature/humidity, so those inputs are zeroed and flagged, letting the model
   fall back to the image alone rather than learning from filler values.

---

## 7. Hardware & Software

| Item | Version |
|---|---|
| GPU | NVIDIA GeForce GTX 750 Ti |
| CUDA | 11.8 |
| Python | 3.12 |
| PyTorch | 2.7.1+cu118 |
| transformers | >= 4.56.0 |
| Backbone | facebook/dinov3-vitb16-pretrain-lvd1689m |
"""

report_path = OUTPUT_DIR / "project_summary.md"
report_path.write_text(report, encoding="utf-8")
print(f"\nSaved: {report_path}")

print("\n" + "=" * 60)
print("Phase 5 Complete -- Classification project report generated")
print("=" * 60)
print(f"  Test accuracy : {test_acc:.4f}")
print(f"  Macro-F1      : {macro_f1:.4f}")
print(f"  Weighted-F1   : {weighted_f1:.4f}")
