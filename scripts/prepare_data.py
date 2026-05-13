import os
from pathlib import Path
import shutil
import yaml

print("🚀 PCB Defect Detection - Data Preparation")

# ========================= CONFIG =========================
DATASET_ROOT = Path("dataset")
PCB_DATASET = DATASET_ROOT / "PCB_DATASET"
OUTPUT_DIR = PCB_DATASET / "output"

IMG_SIZE = 800
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# Create directories
for dir in [OUTPUT_DIR / "images/train", OUTPUT_DIR / "images/val", 
            OUTPUT_DIR / "images/test", OUTPUT_DIR / "labels/train", 
            OUTPUT_DIR / "labels/val", OUTPUT_DIR / "labels/test"]:
    dir.mkdir(parents=True, exist_ok=True)

print(f"✅ Directories created in {OUTPUT_DIR}")

# Create data.yaml
data_yaml = {
    'train': str(OUTPUT_DIR / "images/train").replace("\\", "/"),
    'val': str(OUTPUT_DIR / "images/val").replace("\\", "/"),
    'test': str(OUTPUT_DIR / "images/test").replace("\\", "/"),
    'nc': len(CLASSES),
    'names': CLASSES
}

with open(PCB_DATASET / "data.yaml", 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("✅ data.yaml created successfully!")
print("\n📋 Next Steps:")
print("1. Download dataset from: https://www.kaggle.com/datasets/akhatova/pcb-defects")
print("2. Extract the zip file")
print("3. Copy the 'PCB_DATASET' folder into the 'dataset' folder")
print("4. Run preprocessing notebook or script to resize images and convert annotations")

if __name__ == "__main__":
    print("\n✅ Preparation completed!")