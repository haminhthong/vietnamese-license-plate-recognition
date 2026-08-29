"""Pipeline nhận diện hoàn chỉnh gồm YOLO và EasyOCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .ocr import read_plate

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class RecognitionConfig:
    """Các tham số dùng thống nhất cho detector và bước cắt ảnh."""

    detection_confidence: float = 0.25
    nms_iou: float = 0.60
    image_size: int = 640
    padding_ratio: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError("detection_confidence phải nằm trong [0, 1]")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou phải nằm trong [0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size phải lớn hơn 0")
        if self.padding_ratio < 0:
            raise ValueError("padding_ratio không được âm")


def crop_with_padding(image: np.ndarray, box: Box, padding_ratio: float = 0.05) -> tuple[np.ndarray, Box]:
    """Cắt vùng ảnh và thêm khoảng đệm nhưng không vượt biên ảnh."""
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    if padding_ratio < 0:
        raise ValueError("padding_ratio không được âm")
    x1, y1, x2, y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Bounding box không hợp lệ: {box}")
    height, width = image.shape[:2]
    pad_x, pad_y = int((x2 - x1) * padding_ratio), int((y2 - y1) * padding_ratio)
    padded: Box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    px1, py1, px2, py2 = padded
    return image[py1:py2, px1:px2], padded


class LicensePlateRecognizer:
    """Điều phối detector, chỉnh ảnh và OCR trong một pipeline duy nhất."""

    def __init__(
        self,
        weights: str | Path,
        gpu: bool | None = None,
        config: RecognitionConfig | None = None,
    ) -> None:
        import easyocr
        import torch
        from ultralytics import YOLO

        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        self.config = config or RecognitionConfig()
        self.detector = YOLO(str(weights))
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        self.last_latency_ms = 0.0

    def predict(self, image_bgr: np.ndarray, confidence: float | None = None) -> list[dict[str, Any]]:
        """Trả về danh sách biển số phát hiện được trên một ảnh BGR."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào rỗng")
        detection_confidence = self.config.detection_confidence if confidence is None else confidence
        if not 0 <= detection_confidence <= 1:
            raise ValueError("confidence phải nằm trong [0, 1]")

        started_at = perf_counter()
        result = self.detector.predict(
            source=image_bgr,
            conf=detection_confidence,
            iou=self.config.nms_iou,
            imgsz=self.config.image_size,
            verbose=False,
        )[0]
        predictions: list[dict[str, Any]] = []
        boxes = [] if result.boxes is None else result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            detected_box: Box = tuple(map(int, xyxy))
            crop, padded_box = crop_with_padding(image_bgr, detected_box, self.config.padding_ratio)
            if crop.size == 0:
                continue
            class_id = int(box.cls.item())
            predictions.append({
                "box": detected_box,
                "padded_box": padded_box,
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

    def predict_file(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Đọc ảnh từ ổ đĩa, nhận diện và tùy chọn lưu ảnh minh họa."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
        predictions = self.predict(image)
        if output_path:
            annotated = draw_predictions(image, predictions)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), annotated):
                raise OSError(f"Không ghi được ảnh kết quả: {output_path}")
        return predictions


def draw_predictions(image: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    """Vẽ bounding box và chuỗi biển số lên bản sao của ảnh."""
    annotated = image.copy()
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = prediction["text"] or "[không đọc được]"
        label = f"{text} {prediction['detection_confidence']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return annotated
