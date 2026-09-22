"""EvalTrack scorer wrapping InspectAI's defect detector.

Free and local, no API key: runs the fine-tuned YOLOv8n model (or a
different weights file via WEIGHTS_OVERRIDE, used below to simulate a
regression) against the real NEU-DET held-out test set and reports mAP50
as "accuracy" plus mean per-image inference latency.
"""
import os
import time

from ultralytics import YOLO

DATA_YAML = os.environ.get("EVALTRACK_DATA_YAML", "data.yaml")
WEIGHTS = os.environ.get("EVALTRACK_WEIGHTS", "weights/neu_det_yolov8n.pt")


def run() -> dict:
    model = YOLO(WEIGHTS)

    start = time.time()
    metrics = model.val(data=DATA_YAML, imgsz=224, split="val", verbose=False)
    elapsed_ms = (time.time() - start) * 1000

    n_images = 180  # NEU-DET held-out test split, see InspectAI/finetune/README.md
    return {
        "accuracy": float(metrics.box.map50),
        "latency_ms": elapsed_ms / n_images,
        "n": n_images,
    }
