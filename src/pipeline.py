"""Pipeline điều phối quy trình nhận diện biển số xe hoàn chỉnh (YOLOv8 + EasyOCR + Post-processing).

Module này kết nối các thành phần phát hiện đối tượng, cắt vùng ảnh có đệm, nắn góc phối cảnh,
nhận dạng ký tự và ghi đè kết quả lên ảnh minh họa.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .config import RecognitionConfig
from .io_utils import read_image
from .ocr import read_plate

Box = tuple[int, int, int, int]


def crop_with_padding(image: np.ndarray, box: Box, padding_ratio: float = 0.05) -> tuple[np.ndarray, Box]:
    """Cắt vùng ảnh chứa biển số từ Bounding Box và mở rộng lề đệm nhưng đảm bảo không vượt quá biên ảnh.

    Args:
        image (np.ndarray): Ảnh BGR gốc.
        box (Box): Tọa độ (x1, y1, x2, y2).
        padding_ratio (float): Tỷ lệ đệm (mặc định: 0.05).

    Returns:
        tuple[np.ndarray, Box]: Cặp (ảnh_crop_đã_đệm, tọa_độ_box_đã_đệm).

    Raises:
        ValueError: Nếu ảnh rỗng, padding_ratio âm hoặc tọa độ box không hợp lệ.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào bị rỗng.")
    if padding_ratio < 0:
        raise ValueError("Tham số 'padding_ratio' không được nhỏ hơn 0.")
    x1, y1, x2, y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Tọa độ Bounding box không hợp lệ: {box}")
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
    """Lớp điều phối chính thực thi pipeline end-to-end từ ảnh đầu vào đến kết quả biển số.

    Args:
        weights (str | Path): Đường dẫn đến tệp trọng số YOLOv8 (.pt).
        gpu (bool | None): Ép buộc sử dụng GPU (True), CPU (False) hoặc tự động phát hiện (None).
        config (RecognitionConfig | None): Cấu hình tùy chọn cho pipeline.
    """

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
        """Dự đoán và nhận diện toàn bộ biển số xe xuất hiện trong ảnh BGR.

        Args:
            image_bgr (np.ndarray): Mảng ảnh BGR (OpenCV format).
            confidence (float | None): Độ tin cậy đè tùy chọn.

        Returns:
            list[dict[str, Any]]: Danh sách kết quả nhận diện từng biển số.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào bị rỗng.")
        detection_confidence = self.config.detection_confidence if confidence is None else confidence
        if not 0 <= detection_confidence <= 1:
            raise ValueError("Tham số 'confidence' phải nằm trong khoảng [0, 1].")

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
                **read_plate(self.reader, crop, config=self.config),
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
        """Đọc ảnh từ đĩa cứng, chạy dự đoán và tùy chọn lưu ảnh vẽ kết quả.

        Args:
            image_path (str | Path): Đường dẫn tệp ảnh nguồn.
            output_path (str | Path | None): Đường dẫn xuất ảnh kết quả minh họa (nếu có).

        Returns:
            list[dict[str, Any]]: Danh sách dự đoán biển số.
        """
        image = read_image(image_path, "tệp ảnh từ đĩa")
        predictions = self.predict(image)
        if output_path:
            annotated = draw_predictions(image, predictions)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), annotated):
                raise OSError(f"Không thể ghi ảnh kết quả ra tệp: {output_path}")
        return predictions


def draw_predictions(image: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    """Vẽ Bounding Box màu xanh lục và chuỗi ký tự biển số nhận dạng lên bản sao của ảnh gốc.

    Args:
        image (np.ndarray): Ảnh BGR gốc.
        predictions (list[dict[str, Any]]): Danh sách kết quả dự đoán từ pipeline.

    Returns:
        np.ndarray: Ảnh mới đã được vẽ trực quan kết quả.
    """
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
