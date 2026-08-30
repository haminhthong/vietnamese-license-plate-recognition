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
