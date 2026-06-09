"""
Phase 1 - Data Exploration & Organization
HVAQ Dataset: PM2.5 Estimation with DINOv2
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE_DIR = Path(r"d:\Ahsan DS project\Dataset")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATE_FOLDERS = {
    "7_24_data":  "2019-07-24",
    "10_19_data": "2019-10-19",
    "11_10_data": "2019-11-10",
}

# ─────────────────────────────────────────
# SECTION 1: Dataset Structure
# ─────────────────────────────────────────
print("=" * 60)
print("SECTION 1: Dataset Structure")
print("=" * 60)

structure = {}
for folder, date_str in DATE_FOLDERS.items():
    folder_path = BASE_DIR / folder
    img_dir = folder_path / "pictures"
    images = list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.jpeg"))
    csvs = list(folder_path.glob("*.csv"))
    structure[folder] = {
        "date": date_str,
        "image_count": len(images),
        "csv_count": len(csvs),
        "csv_names": sorted([c.name for c in csvs]),
        "image_paths": images,
    }
    print(f"\n{folder}  ({date_str})")
    print(f"  Images : {len(images)}")
    print(f"  CSVs   : {len(csvs)}  ->  {', '.join(sorted([c.name for c in csvs]))}")

total_imgs = sum(v["image_count"] for v in structure.values())
print(f"\nTotal images across all dates: {total_imgs}")

# ─────────────────────────────────────────
# SECTION 2: Image Filename Patterns
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Image Filename Patterns")
print("=" * 60)

patterns = {
    "7_24_data":  "HHMMSS.JPG              (e.g. 152338.JPG — time only, no date)",
    "10_19_data": "2019-10-19 HHMMSS.JPG   (e.g. 2019-10-19 104440.JPG)",
    "11_10_data": "IMG_YYYYMMDD_HHMMSS.jpg (e.g. IMG_20191110_103543.jpg)",
}
for folder, pattern in patterns.items():
    print(f"  {folder}: {pattern}")

print("\nNote: 7_24_data filenames contain only HHMMSS — date must be inferred from folder name.")

# ─────────────────────────────────────────
# SECTION 3: CSV Format Examination
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: CSV Format (sample from each date)")
print("=" * 60)

csv_samples = {}
for folder in DATE_FOLDERS:
    csv_file = sorted((BASE_DIR / folder).glob("*.csv"))[0]
    df = pd.read_csv(csv_file)
    csv_samples[folder] = df
    print(f"\n--- {folder} / {csv_file.name} ---")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows   : {len(df)}")
    print(df.head(5).to_string(index=False))

print("\nWARNING: 7_24_data CSVs have NO temperature or humidity columns.")
print("   Only pm2.5, pm10, time are available for that date.")

# ─────────────────────────────────────────
# SECTION 4: PM2.5 Statistics per Date
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: PM2.5 Statistics (all sensors combined)")
print("=" * 60)

pm25_stats = {}
for folder in DATE_FOLDERS:
    csvs = sorted((BASE_DIR / folder).glob("*.csv"))
    all_pm25 = []
    for csv in csvs:
        df = pd.read_csv(csv)
        all_pm25.extend(df["pm2.5"].dropna().tolist())
    arr = np.array(all_pm25)
    pm25_stats[folder] = {
        "count": len(arr),
        "min": arr.min(),
        "max": arr.max(),
        "mean": arr.mean(),
        "std": arr.std(),
        "median": np.median(arr),
    }
    s = pm25_stats[folder]
    print(f"\n{folder}:")
    print(f"  Total readings : {s['count']:,}")
    print(f"  PM2.5  min     : {s['min']:.1f}")
    print(f"         max     : {s['max']:.1f}")
    print(f"         mean    : {s['mean']:.1f}")
    print(f"         std     : {s['std']:.1f}")
    print(f"         median  : {s['median']:.1f}")

total_readings = sum(v["count"] for v in pm25_stats.values())
print(f"\nTotal sensor readings (all dates): {total_readings:,}")

# ─────────────────────────────────────────
# SECTION 5: Sample Image Properties
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5: Sample Image Properties")
print("=" * 60)

sample_imgs = {
    "7_24_data":  BASE_DIR / "7_24_data"  / "pictures" / "152338.JPG",
    "10_19_data": BASE_DIR / "10_19_data" / "pictures" / "2019-10-19 104440.JPG",
    "11_10_data": BASE_DIR / "11_10_data" / "pictures" / "IMG_20191110_103543.jpg",
}
for folder, img_path in sample_imgs.items():
    img = Image.open(img_path)
    size_mb = img_path.stat().st_size / 1024**2
    print(f"  {folder}: {img.width}x{img.height}  mode={img.mode}  size={size_mb:.1f}MB  format={img.format}")
    img.close()

print("\nNote: All images are JPEG. DINOv2 expects 224x224 — significant downscaling needed.")

# ─────────────────────────────────────────
# SECTION 6: Visualizations
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6: Generating visualizations …")
print("=" * 60)

# 6a) PM2.5 distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("PM2.5 Distribution per Date", fontsize=14, fontweight="bold")
colors = ["steelblue", "tomato", "seagreen"]

for ax, (folder, stats), color in zip(axes, pm25_stats.items(), colors):
    csvs = sorted((BASE_DIR / folder).glob("*.csv"))
    all_pm25 = []
    for csv in csvs:
        df = pd.read_csv(csv)
        all_pm25.extend(df["pm2.5"].dropna().tolist())
    ax.hist(all_pm25, bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(folder)
    ax.set_xlabel("PM2.5 (µg/m³)")
    ax.set_ylabel("Frequency")
    ax.axvline(np.mean(all_pm25), color="black", linestyle="--", linewidth=1.2, label=f"mean={np.mean(all_pm25):.1f}")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pm25_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR / 'pm25_distributions.png'}")

# 6b) Sample images from each date
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Sample Images per Date", fontsize=14, fontweight="bold")
for ax, (folder, img_path) in zip(axes, sample_imgs.items()):
    img = Image.open(img_path)
    img_resized = img.resize((640, 480))
    ax.imshow(img_resized)
    ax.set_title(folder)
    ax.axis("off")
    img.close()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR / 'sample_images.png'}")

# ─────────────────────────────────────────
# SECTION 7: Data Inventory Report
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7: Data Inventory Report")
print("=" * 60)

report_lines = [
    "# HVAQ Dataset Inventory Report",
    "",
    "## Overview",
    f"- Total date folders    : 3",
    f"- Total images          : {total_imgs}",
    f"- Total sensor readings : {total_readings:,}",
    "",
    "## Per-Date Summary",
    "",
    "| Date Folder | Date       | Images | Sensor CSVs | PM2.5 Min | PM2.5 Max | PM2.5 Mean | Temp/Hum |",
    "|-------------|------------|--------|-------------|-----------|-----------|------------|----------|",
]

has_temp = {"7_24_data": "No", "10_19_data": "Yes", "11_10_data": "Yes"}
for folder, date_str in DATE_FOLDERS.items():
    s = pm25_stats[folder]
    n_imgs = structure[folder]["image_count"]
    n_csvs = structure[folder]["csv_count"]
    report_lines.append(
        f"| {folder:<11} | {date_str} | {n_imgs:>6} | {n_csvs:>11} | "
        f"{s['min']:>9.1f} | {s['max']:>9.1f} | {s['mean']:>10.1f} | {has_temp[folder]:>8} |"
    )

report_lines += [
    "",
    "## Image Properties",
    "- 7_24_data   : 4864×3648 px, JPEG, ~7-8 MB each",
    "- 10_19_data  : 4864×3648 px, JPEG, ~7-8 MB each",
    "- 11_10_data  : 4000×3000 px, JPEG, ~2.7-4 MB each",
    "",
    "## Filename → Timestamp Mapping",
    "- 7_24_data  : `HHMMSS.JPG`               → parse as time-only, combine with folder date",
    "- 10_19_data : `2019-10-19 HHMMSS.JPG`    → parse date + time directly",
    "- 11_10_data : `IMG_YYYYMMDD_HHMMSS.jpg`  → parse YYYYMMDD_HHMMSS",
    "",
    "## CSV Columns",
    "- 7_24_data  : pm2.5, pm10, time  (**no temperature, no humidity**)",
    "- 10_19_data : pm2.5, pm10, time, temperature, humidity",
    "- 11_10_data : pm2.5, pm10, time, temperature, humidity",
    "",
    "## Data Concerns",
    "1. PM2.5 max of 389.2 in 10_19_data — potential outliers (very high but plausible in heavy pollution)",
    "2. 7_24_data missing temperature/humidity — will need NaN fill or exclusion in those feature columns",
    "3. 7_24_data has only 5 sensor locations (1,5,6,7,8) vs 9-10 in other dates",
    "4. 10_19_data missing location9 (9 sensors)",
    "5. Sensor readings are ~1 Hz; images are sparse (~30-40 per day) — matching strategy needed",
    "6. Images are 4864×3648 — must resize to 224×224 for DINOv2 input",
]

report_text = "\n".join(report_lines)
report_path = OUTPUT_DIR / "data_inventory.md"
report_path.write_text(report_text, encoding="utf-8")
print(report_text)
print(f"\nSaved inventory report: {report_path}")
print("\n" + "=" * 60)
print("Phase 1 Complete. Review the findings above before proceeding.")
print("=" * 60)
