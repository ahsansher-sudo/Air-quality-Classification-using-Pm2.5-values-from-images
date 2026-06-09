"""
Phase 2 - Data Preparation
HVAQ Dataset: PM2.5 Estimation with DINOv2

Strategy:
- Match each image to sensor readings within ±30 s window
- Average PM2.5/PM10/temp/humidity across all sensors at that timestamp
- 7_24_data has no temp/humidity -> fill with global mean from other dates
- Split 70 / 15 / 15 stratified by date folder
"""

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE_DIR   = Path(r"d:\Ahsan DS project\Dataset")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATE_FOLDERS = {
    "7_24_data":  "2019-07-24",
    "10_19_data": "2019-10-19",
    "11_10_data": "2019-11-10",
}
TOLERANCE_SEC = 30     # ±30 s matching window
PM25_MAX_VALID = 500   # outlier ceiling (µg/m³)

# ─────────────────────────────────────────
# HELPERS: timestamp parsers
# ─────────────────────────────────────────

def parse_img_timestamp(filename: str, date_str: str, folder: str) -> datetime | None:
    stem = Path(filename).stem
    try:
        if folder == "7_24_data":
            # stem = "152338"  ->  HH MM SS
            t = datetime.strptime(f"{date_str} {stem}", "%Y-%m-%d %H%M%S")

        elif folder == "10_19_data":
            # stem = "2019-10-19 104440"
            parts = stem.split(" ")
            t = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H%M%S")

        elif folder == "11_10_data":
            # stem = "IMG_20191110_103543"
            m = re.search(r"(\d{8})_(\d{6})", stem)
            if not m:
                return None
            t = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")

        else:
            return None
        return t

    except ValueError:
        return None


def load_sensor_df(folder: str, date_str: str) -> pd.DataFrame:
    """Load all sensor CSVs for a date folder into one DataFrame with parsed timestamps."""
    csvs = sorted((BASE_DIR / folder).glob("*.csv"))
    frames = []
    for csv in csvs:
        df = pd.read_csv(csv)
        loc_name = csv.stem   # e.g. "location1"

        # parse the 'time' column
        if folder == "7_24_data":
            # time is HH:MM:SS only
            df["timestamp"] = pd.to_datetime(
                date_str + " " + df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )
            df["temperature"] = np.nan
            df["humidity"]    = np.nan
        else:
            df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")

        df = df.dropna(subset=["timestamp"])
        df["location"] = loc_name
        frames.append(df[["timestamp", "pm2.5", "pm10", "temperature", "humidity", "location"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


# ─────────────────────────────────────────
# STEP 1: Build master dataset
# ─────────────────────────────────────────
print("=" * 60)
print("STEP 1: Building master dataset")
print("=" * 60)

records = []

for folder, date_str in DATE_FOLDERS.items():
    print(f"\nProcessing {folder} ...")
    img_dir = BASE_DIR / folder / "pictures"
    images  = sorted(img_dir.glob("*.[jJ][pP][gG]")) + sorted(img_dir.glob("*.jpeg"))

    sensor_df = load_sensor_df(folder, date_str)
    print(f"  Loaded {len(sensor_df):,} sensor rows from {sensor_df['location'].nunique()} sensors")

    matched = 0
    for img_path in images:
        ts = parse_img_timestamp(img_path.name, date_str, folder)
        if ts is None:
            print(f"  WARNING: could not parse timestamp for {img_path.name}")
            continue

        lo = ts - timedelta(seconds=TOLERANCE_SEC)
        hi = ts + timedelta(seconds=TOLERANCE_SEC)
        window = sensor_df[(sensor_df["timestamp"] >= lo) & (sensor_df["timestamp"] <= hi)]

        if window.empty:
            print(f"  No sensor data within ±{TOLERANCE_SEC}s of {img_path.name} ({ts})")
            continue

        row = {
            "image_path":  str(img_path),
            "timestamp":   ts,
            "pm25":        window["pm2.5"].mean(),
            "pm10":        window["pm10"].mean(),
            "temperature": window["temperature"].mean(),   # NaN for 7_24
            "humidity":    window["humidity"].mean(),       # NaN for 7_24
            "date":        folder,
            "num_sensors": window["location"].nunique(),
        }
        records.append(row)
        matched += 1

    print(f"  Matched {matched}/{len(images)} images")

master = pd.DataFrame(records)
print(f"\nMaster dataset: {len(master)} rows, {master.columns.tolist()}")

# ─────────────────────────────────────────
# STEP 2: Fill temp/humidity NaN with global mean
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Filling missing temperature/humidity (7_24_data)")
print("=" * 60)

global_temp_mean = master.loc[master["date"] != "7_24_data", "temperature"].mean()
global_hum_mean  = master.loc[master["date"] != "7_24_data", "humidity"].mean()

n_temp_missing = master["temperature"].isna().sum()
n_hum_missing  = master["humidity"].isna().sum()

master["temperature"] = master["temperature"].fillna(global_temp_mean)
master["humidity"]    = master["humidity"].fillna(global_hum_mean)

print(f"  Global mean temperature (from 10_19 + 11_10): {global_temp_mean:.2f} °C")
print(f"  Global mean humidity    (from 10_19 + 11_10): {global_hum_mean:.2f} %")
print(f"  Filled {n_temp_missing} temperature NaNs")
print(f"  Filled {n_hum_missing}  humidity NaNs")

# ─────────────────────────────────────────
# STEP 3: Data quality checks
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Data quality checks")
print("=" * 60)

before = len(master)

# 3a) Drop rows with missing PM2.5
n_miss_pm25 = master["pm25"].isna().sum()
master = master.dropna(subset=["pm25"])
print(f"  Removed {n_miss_pm25} rows with missing PM2.5")

# 3b) Outlier check: PM2.5 > PM25_MAX_VALID
n_outliers = (master["pm25"] > PM25_MAX_VALID).sum()
master = master[master["pm25"] <= PM25_MAX_VALID]
print(f"  Removed {n_outliers} rows with PM2.5 > {PM25_MAX_VALID}")

# 3c) Verify all images exist and are readable
bad_imgs = []
for _, row in master.iterrows():
    try:
        with Image.open(row["image_path"]) as img:
            img.verify()
    except Exception as e:
        bad_imgs.append((row["image_path"], str(e)))

if bad_imgs:
    print(f"  WARNING: {len(bad_imgs)} unreadable images — removing them")
    bad_paths = {b[0] for b in bad_imgs}
    master = master[~master["image_path"].isin(bad_paths)]
else:
    print(f"  All {len(master)} images verified as readable")

print(f"\n  Dataset size: {before} -> {len(master)} rows (after quality checks)")

# PM2.5 summary
print(f"\n  PM2.5 stats (final):")
print(f"    min    : {master['pm25'].min():.1f}")
print(f"    max    : {master['pm25'].max():.1f}")
print(f"    mean   : {master['pm25'].mean():.1f}")
print(f"    std    : {master['pm25'].std():.1f}")
print(f"    median : {master['pm25'].median():.1f}")

print(f"\n  Per-date image counts:")
for d, grp in master.groupby("date"):
    print(f"    {d}: {len(grp)} images  PM2.5 {grp['pm25'].min():.1f}–{grp['pm25'].max():.1f}")

# PM2.5 histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(master["pm25"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.set_xlabel("PM2.5 (µg/m³)")
ax.set_ylabel("Number of images")
ax.set_title("PM2.5 Distribution — Master Dataset (per-image averages)")
ax.axvline(master["pm25"].mean(), color="tomato", linestyle="--", linewidth=1.5,
           label=f"mean={master['pm25'].mean():.1f}")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "master_pm25_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUTPUT_DIR / 'master_pm25_histogram.png'}")

# ─────────────────────────────────────────
# STEP 4: Train / Val / Test split (70/15/15 stratified by date)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Train / Val / Test split (70/15/15)")
print("=" * 60)

train_frames, val_frames, test_frames = [], [], []

for date, grp in master.groupby("date"):
    grp = grp.sample(frac=1, random_state=42)   # shuffle within date
    n = len(grp)
    n_train = int(round(n * 0.70))
    n_val   = int(round(n * 0.15))
    # test gets the remainder so totals always add up
    train_frames.append(grp.iloc[:n_train])
    val_frames.append(grp.iloc[n_train : n_train + n_val])
    test_frames.append(grp.iloc[n_train + n_val :])
    print(f"  {date}: {n} total -> {n_train} train / {n_val} val / {n - n_train - n_val} test")

train_df = pd.concat(train_frames).reset_index(drop=True)
val_df   = pd.concat(val_frames).reset_index(drop=True)
test_df  = pd.concat(test_frames).reset_index(drop=True)

print(f"\n  Totals: {len(train_df)} train | {len(val_df)} val | {len(test_df)} test")
print(f"  Ratios: {len(train_df)/len(master)*100:.1f}% / "
      f"{len(val_df)/len(master)*100:.1f}% / "
      f"{len(test_df)/len(master)*100:.1f}%")

# ─────────────────────────────────────────
# STEP 5: Save all outputs
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Saving outputs")
print("=" * 60)

master.to_csv(OUTPUT_DIR / "master_dataset.csv", index=False)
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR  / "val.csv",   index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv",  index=False)

print(f"  master_dataset.csv  ({len(master)} rows)")
print(f"  train.csv           ({len(train_df)} rows)")
print(f"  val.csv             ({len(val_df)} rows)")
print(f"  test.csv            ({len(test_df)} rows)")
print(f"\n  All saved to: {OUTPUT_DIR}")

# ─────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2 Complete")
print("=" * 60)
print(f"  Total matched images : {len(master)}")
print(f"  PM2.5 range          : {master['pm25'].min():.1f} - {master['pm25'].max():.1f} µg/m³")
print(f"  Temperature fill     : {global_temp_mean:.2f} °C (applied to {n_temp_missing} rows)")
print(f"  Humidity fill        : {global_hum_mean:.2f} % (applied to {n_hum_missing} rows)")
print(f"  Split                : {len(train_df)} / {len(val_df)} / {len(test_df)}  (train/val/test)")
print("=" * 60)
