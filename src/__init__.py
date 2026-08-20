"""Các thành phần dùng chung cho hệ thống nhận diện biển số xe Việt Nam."""

from .ocr import infer_plate_layout, normalize_plate_text, validate_and_correct_plate

__all__ = ["infer_plate_layout", "normalize_plate_text", "validate_and_correct_plate"]
