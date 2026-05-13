# scripts/train.py
from pathlib import Path
import subprocess

print("🚀 Starting PCB Defect Detection Training")

# Run your main notebook commands or ultralytics training
print("Please run the training from the Jupyter Notebook for now.")
print("Notebook location: notebooks/pcb_defect_full_code.ipynb")

# Example command (uncomment when ready):
# subprocess.run(["yolo", "train", "model=yolov8n.pt", "data=dataset/PCB_DATASET/data.yaml", 
#                 "epochs=50", "imgsz=800", "batch=4"])