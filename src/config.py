"""Quản lý và kiểm tra hợp lệ cấu hình huấn luyện mô hình YOLOv8.

Module này định nghĩa lớp dataclass `TrainingConfig` hỗ trợ nạp tệp cấu hình YAML,
kiểm tra tính hợp lệ của các tham số và áp dụng các thông số đè từ dòng lệnh (CLI).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """Cấu hình thông số huấn luyện cho mô hình phát hiện biển số YOLOv8.

    Attributes:
        model (str): Tên hoặc đường dẫn đến trọng số khởi tạo YOLO (ví dụ: 'yolov8n.pt').
        epochs (int): Số lượng lượt huấn luyện tối đa (mặc định: 60).
        batch (int): Kích thước lô dữ liệu (mặc định: 16).
        image_size (int): Kích thước ảnh đầu vào độ phân giải vuông (mặc định: 640).
        patience (int): Số lượng epoch chờ dừng sớm nếu kết quả validation không tăng (mặc định: 15).
        seed (int): Hạt giống ngẫu nhiên đảm bảo khả năng tái lập kết quả (mặc định: 42).
        workers (int): Số lượng tiến trình con nạp dữ liệu (mặc định: 2).
    """

    model: str = "yolov8n.pt"
    epochs: int = 60
    batch: int = 16
    image_size: int = 640
    patience: int = 15
    seed: int = 42
    workers: int = 2

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của các trường dữ liệu sau khi khởi tạo đối tượng."""
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Tham số 'model' phải là chuỗi không rỗng.")

        positive_fields = {
            "epochs": self.epochs,
            "batch": self.batch,
            "image_size": self.image_size,
        }
        for field_name, value in positive_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"Tham số '{field_name}' phải là số nguyên (int).")
            if value <= 0:
                raise ValueError(f"Tham số '{field_name}' phải là số nguyên dương (> 0).")

        if not isinstance(self.patience, int) or isinstance(self.patience, bool):
            raise TypeError("Tham số 'patience' phải là số nguyên (int).")
        if self.patience < 0:
            raise ValueError("Tham số 'patience' không được nhỏ hơn 0.")

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("Tham số 'seed' phải là số nguyên (int).")

        if not isinstance(self.workers, int) or isinstance(self.workers, bool):
            raise TypeError("Tham số 'workers' phải là số nguyên (int).")
        if self.workers < 0:
            raise ValueError("Tham số 'workers' không được nhỏ hơn 0.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Đọc và nạp cấu hình từ tệp YAML với cơ chế kiểm tra khóa nghiêm ngặt.

        Args:
            path (str | Path): Đường dẫn tới tệp cấu hình YAML.

        Returns:
            TrainingConfig: Đối tượng cấu hình huấn luyện đã được kiểm tra hợp lệ.

        Raises:
            FileNotFoundError: Nếu không tìm thấy tệp YAML.
            ValueError: Nếu tệp không chứa dictionary hoặc chứa khóa không hợp lệ.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình YAML: {config_path}")

        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tệp cấu hình huấn luyện phải có định dạng YAML dictionary/mapping.")

        # Hỗ trợ đồng nhất alias giữa 'imgsz' và 'image_size'
        aliases = {"imgsz": "image_size"}
        if "imgsz" in payload and "image_size" in payload:
            raise ValueError("Chỉ được khai báo một trong hai khóa 'imgsz' hoặc 'image_size'.")

        normalized = {aliases.get(key, key): value for key, value in payload.items()}
        allowed = set(cls.__dataclass_fields__)
        if unknown := set(normalized) - allowed:
            raise ValueError(f"Tệp cấu hình chứa các khóa không được hỗ trợ: {sorted(unknown)}")

        return cls(**normalized)

    def override(
        self,
        *,
        model: str | None = None,
        epochs: int | None = None,
        batch: int | None = None,
    ) -> TrainingConfig:
        """Đề xuất cấu hình mới bằng cách ghi đè thông số từ CLI mà không sửa đối tượng gốc.

        Args:
            model (str | None): Tên/đường dẫn mô hình đè.
            epochs (int | None): Số epoch đè.
            batch (int | None): Kích thước batch đè.

        Returns:
            TrainingConfig: Đối tượng cấu hình mới đã được cập nhật giá trị.
        """
        return replace(
            self,
            model=self.model if model is None else model,
            epochs=self.epochs if epochs is None else epochs,
            batch=self.batch if batch is None else batch,
        )


@dataclass(frozen=True)
class RecognitionConfig:
    """Các thông số cấu hình thống nhất cho phát hiện, nắn góc, OCR và hậu xử lý.

    Attributes:
        detection_confidence (float): Ngưỡng độ tin cậy tối thiểu của YOLOv8 (mặc định: 0.25).
        nms_iou (float): Ngưỡng NMS IoU loại bỏ bounding box trùng lặp (mặc định: 0.60).
        image_size (int): Kích thước ảnh resize đầu vào mô hình YOLO (mặc định: 640).
        padding_ratio (float): Tỷ lệ đệm mở rộng lề khi cắt biển số (mặc định: 0.05 tức 5%).
        ocr_minimum_confidence (float): Độ tin cậy OCR tối thiểu cho từng token (mặc định: 0.20).
        wide_ratio_threshold (float): Ngưỡng tỷ lệ aspect ratio để phân biệt biển 1 dòng / 2 dòng (mặc định: 2.20).
        valid_format_bonus (float): Điểm cộng khi kết quả OCR khớp mẫu biển số Việt Nam (mặc định: 0.20).
        correction_penalty (float): Điểm phạt cho mỗi ký tự phải hiệu chỉnh (mặc định: 0.07).
        max_correction_cost (float): Giới hạn chi phí hiệu chỉnh tối đa (mặc định: 1.0).
        enable_rectification (bool): Bật nắn góc phối cảnh (mặc định: True).
        enable_preprocessing_variants (bool): Bật 4 biến thể xử lý ảnh (Gray, CLAHE, Otsu, Adaptive) (mặc định: True).
        enable_template_correction (bool): Bật hậu xử lý khớp mẫu (mặc định: True).
        single_variant_mode (str | None): Tùy chọn chỉ chạy một biến thể (crop/gray/clahe/otsu/adaptive).
    """

    detection_confidence: float = 0.25
    nms_iou: float = 0.60
    image_size: int = 640
    padding_ratio: float = 0.05
    ocr_minimum_confidence: float = 0.20
    wide_ratio_threshold: float = 2.20
    valid_format_bonus: float = 0.20
    correction_penalty: float = 0.07
    max_correction_cost: float = 1.0
    enable_rectification: bool = True
    enable_preprocessing_variants: bool = True
    enable_template_correction: bool = True
    single_variant_mode: str | None = None

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của các giá trị tham số cấu hình."""
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError("Tham số 'detection_confidence' phải nằm trong khoảng [0, 1].")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("Tham số 'nms_iou' phải nằm trong khoảng [0, 1].")
        if self.image_size <= 0:
            raise ValueError("Tham số 'image_size' phải lớn hơn 0.")
        if self.padding_ratio < 0:
            raise ValueError("Tham số 'padding_ratio' không được nhỏ hơn 0.")
        if not 0 <= self.ocr_minimum_confidence <= 1:
            raise ValueError("Tham số 'ocr_minimum_confidence' phải nằm trong khoảng [0, 1].")
        if self.wide_ratio_threshold <= 0:
            raise ValueError("Tham số 'wide_ratio_threshold' phải lớn hơn 0.")
        if self.valid_format_bonus < 0:
            raise ValueError("Tham số 'valid_format_bonus' không được nhỏ hơn 0.")
        if self.correction_penalty < 0:
            raise ValueError("Tham số 'correction_penalty' không được nhỏ hơn 0.")
        if self.max_correction_cost < 0:
            raise ValueError("Tham số 'max_correction_cost' không được nhỏ hơn 0.")
        valid_variants = {None, "crop", "gray", "clahe", "otsu", "adaptive"}
        if self.single_variant_mode not in valid_variants:
            raise ValueError(
                "Tham số 'single_variant_mode' phải là một trong: crop, gray, clahe, otsu, adaptive hoặc None."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecognitionConfig:
        """Đọc và nạp cấu hình từ tệp YAML."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình YAML: {config_path}")

        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tệp cấu hình nhận diện phải có định dạng YAML dictionary/mapping.")

        allowed = set(cls.__dataclass_fields__)
        if unknown := set(payload) - allowed:
            raise ValueError(f"Tệp cấu hình chứa các khóa không được hỗ trợ: {sorted(unknown)}")

        return cls(**payload)
