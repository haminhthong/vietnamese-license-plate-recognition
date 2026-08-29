"""Các hàm đọc, kiểm tra và ghi artifact dùng chung."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def require_columns(frame: pd.DataFrame, required: set[str], source_name: str) -> None:
    """Báo lỗi nếu bảng thiếu cột bắt buộc."""
    if missing := required - set(frame.columns):
        raise ValueError(f"{source_name} thiếu cột bắt buộc: {sorted(missing)}")


def require_non_empty_text(frame: pd.DataFrame, column: str, source_name: str) -> None:
    """Báo lỗi nếu bảng rỗng hoặc cột văn bản có giá trị trống."""
    if frame.empty:
        raise ValueError(f"{source_name} không có dữ liệu")
    values = frame[column].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"Cột {column} trong {source_name} không được rỗng")


def resolve_relative_path(path: str | Path, base_directory: Path) -> Path:
    """Giữ nguyên đường dẫn tuyệt đối hoặc ghép đường dẫn tương đối với thư mục gốc."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_directory / candidate


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Ghi JSON UTF-8 và tự tạo thư mục cha."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
