"""Huấn luyện YOLOv8 để phát hiện một class biển số xe."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config import TrainingConfig
from src.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    config = TrainingConfig.from_yaml(args.config).override(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
    )
    run_name = f"yolov8n_grouped_{datetime.now():%Y%m%d_%H%M%S}"
    model = YOLO(config.model)
    model.train(
        data=str(args.data), epochs=config.epochs, patience=config.patience,
        imgsz=config.image_size, batch=config.batch,
        device=0 if torch.cuda.is_available() else "cpu", workers=config.workers,
        seed=config.seed, deterministic=True, project=str(args.runs), name=run_name,
        pretrained=True, optimizer="auto", close_mosaic=10, degrees=3.0,
        translate=0.10, scale=0.40, perspective=0.0005, fliplr=0.0, flipud=0.0,
    )
    metadata = {
        "run_name": run_name, "data": str(args.data.resolve()), "model": config.model,
        "epochs": config.epochs, "batch": config.batch, "seed": config.seed,
        "best_weights": str(model.trainer.best),
    }
    write_json(Path(model.trainer.save_dir) / "experiment.json", metadata)
    print(f"Trọng số tốt nhất: {model.trainer.best}")


if __name__ == "__main__":
    main()
