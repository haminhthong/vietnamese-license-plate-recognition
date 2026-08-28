"""Pipeline nhận diện hoàn chỉnh gồm YOLO và EasyOCR."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .ocr import read_plate


def crop_with_padding(image: np.ndarray, box, padding_ratio: float = 0.05):
    x1, y1, x2, y2 = map(int, box)
    height, width = image.shape[:2]
    pad_x, pad_y = int((x2 - x1) * padding_ratio), int((y2 - y1) * padding_ratio)
    padded = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y))
    px1, py1, px2, py2 = padded
    return image[py1:py2, px1:px2], padded


class LicensePlateRecognizer:
    def __init__(self, weights: str | Path, gpu: bool | None = None):
        import easyocr
        import torch
        from ultralytics import YOLO

        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        self.detector = YOLO(str(weights))
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        self.last_latency_ms = 0.0

    def predict(self, image_bgr: np.ndarray, confidence: float = 0.25) -> list[dict[str, Any]]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào rỗng")
        started_at = perf_counter()
        result = self.detector.predict(source=image_bgr, conf=confidence, iou=0.60, imgsz=640, verbose=False)[0]
        predictions = []
        if result.boxes is None:
            return predictions
        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            crop, padded_box = crop_with_padding(image_bgr, xyxy)
            if crop.size:
                class_id = int(box.cls.item())
                predictions.append({
                    "box": tuple(map(int, xyxy)), "padded_box": padded_box,
                    "class_id": class_id,
                    "detector_class": result.names.get(class_id, "unknown"),
                    "detection_confidence": float(box.conf.item()),
                    **read_plate(self.reader, crop),
                })
        elapsed_ms = (perf_counter() - started_at) * 1_000
        self.last_latency_ms = elapsed_ms
        for prediction in predictions:
            prediction["pipeline_latency_ms"] = elapsed_ms
        return predictions

    def predict_file(self, image_path: str | Path, output_path: str | Path | None = None):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
        predictions = self.predict(image)
        if output_path:
            annotated = image.copy()
            for item in predictions:
                x1, y1, x2, y2 = item["box"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{item['text'] or '[unreadable]'} {item['detection_confidence']:.2f}"
                cv2.putText(annotated, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), annotated):
                raise OSError(f"Không ghi được ảnh kết quả: {output_path}")
        return predictions
