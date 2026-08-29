"""Đọc và kiểm tra cấu hình huấn luyện."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """Cấu hình tối thiểu cho một lần huấn luyện YOLO."""

    model: str = "yolov8n.pt"
    epochs: int = 60
    batch: int = 16
    image_size: int = 640
    patience: int = 15
    seed: int = 42
    workers: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model phải là chuỗi không rỗng")
        positive_fields = {
            "epochs": self.epochs,
            "batch": self.batch,
            "image_size": self.image_size,
        }
        for field_name, value in positive_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} phải là số nguyên")
            if value <= 0:
                raise ValueError(f"{field_name} phải lớn hơn 0")
        if not isinstance(self.patience, int) or isinstance(self.patience, bool):
            raise TypeError("patience phải là số nguyên")
        if self.patience < 0:
            raise ValueError("patience không được âm")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed phải là số nguyên")
        if not isinstance(self.workers, int) or isinstance(self.workers, bool):
            raise TypeError("workers phải là số nguyên")
        if self.workers < 0:
            raise ValueError("workers không được âm")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Tạo cấu hình từ YAML và báo lỗi rõ khi thiếu hoặc thừa khóa."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình: {config_path}")
        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Cấu hình huấn luyện phải là một YAML mapping")
        aliases = {"imgsz": "image_size"}
        if "imgsz" in payload and "image_size" in payload:
            raise ValueError("Chỉ dùng một trong hai khóa imgsz hoặc image_size")
        normalized = {aliases.get(key, key): value for key, value in payload.items()}
        allowed = set(cls.__dataclass_fields__)
        if unknown := set(normalized) - allowed:
            raise ValueError(f"Khóa cấu hình không hỗ trợ: {sorted(unknown)}")
        return cls(**normalized)

    def override(
        self,
        *,
        model: str | None = None,
        epochs: int | None = None,
        batch: int | None = None,
    ) -> "TrainingConfig":
        """Áp dụng giá trị CLI mà không thay đổi đối tượng cấu hình gốc."""
        return replace(
            self,
            model=self.model if model is None else model,
            epochs=self.epochs if epochs is None else epochs,
            batch=self.batch if batch is None else batch,
        )
