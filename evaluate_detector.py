"""Đánh giá detector trên test split chưa dùng để tinh chỉnh mô hình."""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/detector_test_metrics.json"))
    args = parser.parse_args()
    metrics = YOLO(str(args.weights)).val(data=str(args.data), split="test", imgsz=640, batch=16)
    class_ids = np.asarray(metrics.box.ap_class_index, dtype=int)
    precision = np.asarray(metrics.box.p, dtype=float)
    recall = np.asarray(metrics.box.r, dtype=float)
    ap50 = np.asarray(metrics.box.ap50, dtype=float)
    ap = np.asarray(metrics.box.ap, dtype=float)
    per_class = [
        {
            "class_id": int(class_id),
            "class": metrics.names[int(class_id)],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "mAP50": float(ap50[index]),
            "mAP50_95": float(ap[index]),
        }
        for index, class_id in enumerate(class_ids)
    ]
    missing_classes = sorted(set(metrics.names) - set(map(int, class_ids)))
    if missing_classes:
        raise RuntimeError(f"Test split không có dữ liệu đánh giá cho class: {missing_classes}")

    payload = {
        "weights": str(args.weights), "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map), "per_class": per_class,
    }
    write_json(args.output, payload)
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
