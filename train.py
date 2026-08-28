"""Huấn luyện YOLOv8 để phát hiện một class biển số xe."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = args.model or config["model"]
    epochs = args.epochs or int(config["epochs"])
    batch = args.batch or int(config["batch"])
    run_name = f"yolov8n_grouped_{datetime.now():%Y%m%d_%H%M%S}"
    model = YOLO(model_name)
    model.train(
        data=str(args.data), epochs=epochs, patience=int(config["patience"]),
        imgsz=int(config["imgsz"]), batch=batch,
        device=0 if torch.cuda.is_available() else "cpu", workers=int(config["workers"]),
        seed=int(config["seed"]), deterministic=True, project=str(args.runs), name=run_name,
        pretrained=True, optimizer="auto", close_mosaic=10, degrees=3.0,
        translate=0.10, scale=0.40, perspective=0.0005, fliplr=0.0, flipud=0.0,
    )
    metadata = {
        "run_name": run_name, "data": str(args.data.resolve()), "model": model_name,
        "epochs": epochs, "batch": batch, "seed": int(config["seed"]),
        "best_weights": str(model.trainer.best),
    }
    metadata_path = Path(model.trainer.save_dir) / "experiment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Trọng số tốt nhất: {model.trainer.best}")


if __name__ == "__main__":
    main()
