"""Huấn luyện YOLOv8 để phát hiện một class biển số xe."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data), epochs=args.epochs, patience=15, imgsz=640, batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu", workers=2, seed=42,
        deterministic=True, project=str(args.runs), name=f"yolov8n_grouped_{datetime.now():%Y%m%d_%H%M%S}",
        pretrained=True, optimizer="auto", close_mosaic=10, degrees=3.0,
        translate=0.10, scale=0.40, perspective=0.0005, fliplr=0.0, flipud=0.0,
    )
    print(f"Trọng số tốt nhất: {model.trainer.best}")


if __name__ == "__main__":
    main()
