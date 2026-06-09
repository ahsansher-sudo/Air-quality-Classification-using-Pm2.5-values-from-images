# ==============================================================
# IAQ IMAGE CLASSIFICATION USING DINOv3 + SENSOR FUSION
# ==============================================================
# Goal:
#   Predict Indoor/Outdoor Air Quality category (Good / Moderate /
#   Unhealthy-SG / Unhealthy) from a photo + temperature + humidity.
#
# How it works:
#   1. A frozen DINOv3 backbone reads each image and produces a
#      768-dimensional visual summary (done once, cached to disk).
#   2. Temperature and humidity are normalised and joined to the
#      image features, giving a 770-dimensional combined input.
#   3. A small classification head (3 Linear layers) is trained on
#      these combined features to predict the IAQ category.
#   4. IAQ labels come from PM2.5 sensor readings using EPA breakpoints
#      -- PM2.5 is NOT fed into the model at inference time.
#
# IAQ Classes (EPA PM2.5 breakpoints):
#   0 - Good           PM2.5 <= 12.0 ug/m3
#   1 - Moderate       PM2.5 <= 35.4 ug/m3
#   2 - Unhealthy-SG   PM2.5 <= 55.4 ug/m3
#   3 - Unhealthy      PM2.5  > 55.4 ug/m3
# ==============================================================

import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ==============================================================
# SECTION 1: CONFIGURATION
# All hyperparameters and paths in one place
# ==============================================================

BASE_DIR   = Path(r"d:\Ahsan DS project\Dataset")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ID    = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # frozen DINOv3 backbone
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AUG_COPIES  = 3       # original + 2 augmented copies per training image = 222 samples
NUM_CLASSES = 4
TAB_DIM     = 3       # [normalised_temp, normalised_humidity, has_real_sensor_flag]
CLASS_NAMES = ["Good", "Moderate", "Unhealthy-SG", "Unhealthy"]
FAKE_TAB_DATE = "7_24_data"  # this date folder has no real sensors; tab features zeroed

PM25_THRESHOLDS = [12.0, 35.4, 55.4]  # upper boundary for classes 0, 1, 2

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ==============================================================
# SECTION 2: TRAINING AUGMENTATION PIPELINE
# Applied to images before DINOv3 feature extraction.
# Creates varied copies of each training image so the model
# does not memorise exact pixel values.
# Val and test images are NOT augmented.
# ==============================================================

class HazeAugment:
    """
    Blends the image with a white overlay to simulate atmospheric haze.
    This is domain-specific: higher PM2.5 makes outdoor scenes hazier,
    so synthetic haze creates more realistic high-pollution training views.
    """
    def __init__(self, p: float = 0.5, intensity: tuple = (0.05, 0.35)):
        self.p = p
        self.lo, self.hi = intensity

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        alpha = random.uniform(self.lo, self.hi)
        arr   = np.array(img, dtype=np.float32)
        arr   = arr * (1.0 - alpha) + 255.0 * alpha
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


TRAIN_AUG = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    HazeAugment(p=0.5, intensity=(0.05, 0.35)),
])


# ==============================================================
# SECTION 3: IAQ LABEL CONVERSION
# Converts a raw PM2.5 reading into a class index (0-3)
# using standard EPA air quality breakpoints.
# This is used only to BUILD labels -- not at inference time.
# ==============================================================

def pm25_to_class(pm25: float) -> int:
    for i, thresh in enumerate(PM25_THRESHOLDS):
        if pm25 <= thresh:
            return i
    return len(PM25_THRESHOLDS)


# ==============================================================
# SECTION 4: MODEL ARCHITECTURE -- MULTIMODAL CLASSIFICATION HEAD
# Takes two inputs and fuses them before classifying:
#   - img_feat  [B, 768]: visual summary from frozen DINOv3
#   - tab_feat  [B,   3]: (norm_temp, norm_humidity, sensor_mask)
# Concatenated to [B, 771] then passed through 3 linear layers.
# The sensor_mask=0 for images with no real sensor data (7_24),
# so the model learns to rely on the image alone for those.
# ==============================================================

class MultimodalClassHead(nn.Module):
    def __init__(self, num_classes: int = 4, tab_dim: int = 3):
        super().__init__()
        fusion_dim = 768 + tab_dim
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
# SECTION 5: DINOV3 FEATURE EXTRACTION
# Runs each image through the frozen DINOv3 backbone once and
# saves the 768-dim patch-token mean to disk as a .pt cache file.
# On subsequent runs the backbone is skipped entirely -- only
# the cached tensors are loaded, keeping training under 3 seconds.
#
# Also reads temperature, humidity, and sensor availability flag
# from the CSV row for each image.
# ==============================================================

@torch.no_grad()
def extract_features(
    df: pd.DataFrame,
    processor,
    backbone,
    device: str,
    augment: bool = False,
    aug_copies: int = 1,
) -> tuple:
    """
    Returns:
        img_feats    [N * aug_copies, 768]  -- DINOv3 patch-mean embedding
        tab_feats    [N * aug_copies,   3]  -- (temp, humidity, sensor_mask) raw
        class_labels [N * aug_copies]       -- IAQ class index (long)
    """
    backbone.eval()
    all_img, all_tab, all_labels = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        image = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(pm25_to_class(row["pm25"]), dtype=torch.long)

        # Zero out tab features for rows with no real sensors (7_24_data)
        if row["date"] == FAKE_TAB_DATE:
            tab = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            tab = torch.tensor(
                [row["temperature"], row["humidity"], 1.0], dtype=torch.float32
            )

        for copy in range(aug_copies):
            img    = TRAIN_AUG(image) if (augment and copy > 0) else image
            inputs = processor(images=img, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            out    = backbone(pixels)
            # Skip CLS token (index 0) and 4 register tokens (indices 1-4)
            # Mean-pool the 196 patch tokens starting at index 5
            feat   = out.last_hidden_state[:, 5:, :].mean(dim=1)  # [1, 768]
            all_img.append(feat.squeeze(0).cpu())
            all_tab.append(tab)
            all_labels.append(label)

    return torch.stack(all_img), torch.stack(all_tab), torch.stack(all_labels)


# ==============================================================
# SECTION 6: TABULAR FEATURE NORMALISATION
# Scales temperature and humidity to zero-mean, unit-variance
# so they contribute equally to the model alongside image features.
#
# Stats are computed from REAL sensor rows only (not 7_24 zeroed rows).
# The same train-set stats are applied to val and test -- never
# computing stats from val/test to avoid data leakage.
# ==============================================================

def normalise_tab(train_tab: torch.Tensor, *other_tabs: torch.Tensor) -> tuple:
    real_rows  = train_tab[:, 2] == 1.0        # rows with real sensor data
    tab_mean   = train_tab[real_rows, :2].mean(0)
    tab_std    = train_tab[real_rows, :2].std(0).clamp(min=1e-6)

    def _apply(tab: torch.Tensor) -> torch.Tensor:
        out  = tab.clone()
        mask = tab[:, 2] == 1.0
        out[mask, :2] = (tab[mask, :2] - tab_mean) / tab_std
        return out

    return (tab_mean, tab_std) + tuple(_apply(t) for t in (train_tab, *other_tabs))


# ==============================================================
# SECTION 7: EVALUATION -- CONFUSION MATRIX + CLASSIFICATION REPORT
# Runs the best saved model on a DataLoader, prints per-class
# precision / recall / F1, and saves a confusion matrix PNG.
# ==============================================================

def evaluate(model, loader, device, split_name: str, output_dir: Path) -> float:
    model.eval()
    all_logits, all_true = [], []
    with torch.no_grad():
        for img_f, tab_f, labels in loader:
            logits = model(img_f.to(device), tab_f.to(device)).cpu()
            all_logits.append(logits)
            all_true.append(labels)
    all_logits = torch.cat(all_logits)
    all_true   = torch.cat(all_true)
    all_preds  = all_logits.argmax(1)

    acc = float((all_preds == all_true).float().mean())
    print(f"\n-- {split_name} Classification Report -------------------")
    print(classification_report(
        all_true, all_preds,
        target_names=CLASS_NAMES,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    ))

    cm = confusion_matrix(all_true, all_preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- {split_name} Set")
    plt.tight_layout()
    out_path = output_dir / f"confusion_matrix_{split_name.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return acc


# ==============================================================
# SECTION 8: MAIN TRAINING PIPELINE
# Orchestrates the full workflow:
#   Load data -> extract/load features -> normalise ->
#   compute class weights -> train head -> evaluate on val + test
# ==============================================================

def main():

    # ── Load the three CSV splits ─────────────────────────────
    train_df = pd.read_csv(OUTPUT_DIR / "train.csv")
    val_df   = pd.read_csv(OUTPUT_DIR / "val.csv")
    test_df  = pd.read_csv(OUTPUT_DIR / "test.csv")
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        classes = df["pm25"].apply(pm25_to_class)
        real_n  = (df["date"] != FAKE_TAB_DATE).sum()
        print(f"\n{split_name}  (real sensors: {real_n}/{len(df)})")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  Class {c} ({name:15s}): {int((classes == c).sum()):3d}")

    # ── Load cached DINOv3 features (or extract once if missing) ──
    cache_dir   = OUTPUT_DIR / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    train_cache = cache_dir / "train_features_cls_v2.pt"
    val_cache   = cache_dir / "val_features_cls_v2.pt"
    test_cache  = cache_dir / "test_features_cls_v2.pt"

    if train_cache.exists() and val_cache.exists() and test_cache.exists():
        print("\nLoading cached DINOv3 features ...")
        td = torch.load(train_cache, weights_only=True)
        vd = torch.load(val_cache,   weights_only=True)
        sd = torch.load(test_cache,  weights_only=True)
        train_img, train_tab, train_labels = td["img"], td["tab"], td["labels"]
        val_img,   val_tab,   val_labels   = vd["img"], vd["tab"], vd["labels"]
        test_img,  test_tab,  test_labels  = sd["img"], sd["tab"], sd["labels"]
    else:
        print(f"\nLoading DINOv3 backbone: {MODEL_ID}")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        backbone  = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
        print("Extracting features -- this runs once then is cached ...")

        t0 = time.time()
        train_img, train_tab, train_labels = extract_features(
            train_df, processor, backbone, DEVICE, augment=True, aug_copies=AUG_COPIES
        )
        val_img, val_tab, val_labels = extract_features(
            val_df, processor, backbone, DEVICE, augment=False, aug_copies=1
        )
        test_img, test_tab, test_labels = extract_features(
            test_df, processor, backbone, DEVICE, augment=False, aug_copies=1
        )
        print(f"Extraction complete in {time.time() - t0:.1f}s")

        torch.save({"img": train_img, "tab": train_tab, "labels": train_labels}, train_cache)
        torch.save({"img": val_img,   "tab": val_tab,   "labels": val_labels},   val_cache)
        torch.save({"img": test_img,  "tab": test_tab,  "labels": test_labels},  test_cache)
        print(f"Features cached to {cache_dir}")

        del backbone
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTrain: {train_img.shape}  Val: {val_img.shape}  Test: {test_img.shape}")

    # ── Normalise temperature and humidity ────────────────────
    tab_mean, tab_std, train_tab_n, val_tab_n, test_tab_n = normalise_tab(
        train_tab, val_tab, test_tab
    )
    torch.save({"mean": tab_mean, "std": tab_std}, OUTPUT_DIR / "tab_scaler_cls_v2.pt")
    print(f"\nTemp/Humidity normalisation -- mean: {tab_mean.tolist()}  std: {tab_std.tolist()}")

    # ── Compute class weights to handle imbalanced data ───────
    counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    class_weights = (counts.sum() / (NUM_CLASSES * counts.clamp(min=1))).to(DEVICE)
    print(f"\nClass sample counts : {counts.int().tolist()}")
    print(f"Class weights       : {[f'{w:.3f}' for w in class_weights.cpu().tolist()]}")

    # ── Build DataLoaders ─────────────────────────────────────
    train_loader = DataLoader(TensorDataset(train_img, train_tab_n, train_labels),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_img,   val_tab_n,   val_labels),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(test_img,  test_tab_n,  test_labels),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── Initialise model, loss, optimiser, LR scheduler ───────
    model     = MultimodalClassHead(num_classes=NUM_CLASSES, tab_dim=TAB_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {EPOCHS} epochs ...")
    print("-" * 60)

    # ── Training loop ─────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0
    history      = {"train_loss": [], "val_acc": [], "lr": []}
    start = time.time()

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):

        # Forward pass + backprop on training set
        model.train()
        train_loss = 0.0
        for img_f, tab_f, labels in train_loader:
            img_f  = img_f.to(DEVICE)
            tab_f  = tab_f.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(img_f, tab_f), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)

        # Measure accuracy on validation set (no gradient update)
        model.eval()
        all_logits, all_true = [], []
        with torch.no_grad():
            for img_f, tab_f, labels in val_loader:
                all_logits.append(model(img_f.to(DEVICE), tab_f.to(DEVICE)).cpu())
                all_true.append(labels)
        all_logits = torch.cat(all_logits)
        all_true   = torch.cat(all_true)
        val_acc    = float((all_logits.argmax(1) == all_true).float().mean())

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Save checkpoint when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "tab_mean":   tab_mean,
                "tab_std":    tab_std,
            }, OUTPUT_DIR / "best_cls_model_v2.pth")

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(
                f"  Epoch {epoch:>2}/{EPOCHS}  "
                f"loss={train_loss:.4f}  "
                f"val_acc={val_acc:.3f}  "
                f"lr={current_lr:.2e}  "
                f"[{time.time()-start:.1f}s]"
            )

    print(f"\nTraining complete -- best val accuracy {best_val_acc:.4f} at epoch {best_epoch}")

    # ── Load best checkpoint and evaluate on val and test ─────
    ckpt = torch.load(OUTPUT_DIR / "best_cls_model_v2.pth", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])

    val_acc_final  = evaluate(model, val_loader,  DEVICE, "Val",  OUTPUT_DIR)
    test_acc_final = evaluate(model, test_loader, DEVICE, "Test", OUTPUT_DIR)

    print(f"\nFinal Results")
    print(f"  Val  accuracy : {val_acc_final:.4f}")
    print(f"  Test accuracy : {test_acc_final:.4f}")

    # ── Save learning curves ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ep = range(1, EPOCHS + 1)

    axes[0].plot(ep, history["train_loss"], color="steelblue", linewidth=1.5)
    axes[0].axvline(best_epoch, color="tomato", linestyle="--", linewidth=1,
                    label=f"best epoch {best_epoch}")
    axes[0].set_title("Training Loss (CrossEntropy)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

    axes[1].plot(ep, history["val_acc"], color="darkorange", linewidth=1.5)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1)

    axes[2].plot(ep, history["lr"], color="mediumseagreen", linewidth=1.5)
    axes[2].set_title("Learning Rate (Cosine Decay)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")

    plt.suptitle("DINOv3 + Temp/Humidity -- Multimodal IAQ Classifier",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves_cls.png", dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history_cls.csv", index=False)

    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"  best_cls_model_v2.pth      -- model weights")
    print(f"  tab_scaler_cls_v2.pt       -- temp/humidity normalisation stats")
    print(f"  confusion_matrix_val.png   -- val set confusion matrix")
    print(f"  confusion_matrix_test.png  -- test set confusion matrix")
    print(f"  learning_curves_cls.png    -- loss + accuracy + LR curves")
    print(f"  training_history_cls.csv   -- per-epoch metrics")


if __name__ == "__main__":
    main()
