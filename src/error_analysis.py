"""Phân loại lỗi và xuất báo cáo phân tích lỗi (Error Analysis) cho pipeline nhận diện biển số."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import levenshtein_distance
from .ocr import normalize_plate_text


def classify_error(
    ground_truth: str,
    raw_pred: str,
    corrected_pred: str,
    detected: bool,
    iou: float,
    iou_threshold: float = 0.5,
) -> str:
    """Phân loại nhóm nguyên nhân lỗi cho từng mẫu nhận diện.

    Các nhóm lỗi:
    - correct: Dự đoán đúng 100%
    - false_negative: Detector bỏ sót biển số
    - wrong_box: Bounding box có IoU thấp (< iou_threshold)
    - missing_chars: OCR thiếu ký tự so với ground truth
    - extra_chars: OCR thừa ký tự so với ground truth
    - char_confusion: OCR nhầm lẫn giữa các ký tự có cùng độ dài
    - correction_error: Hậu xử lý template làm kết quả tệ hơn (raw_pred khớp nhưng corrected_pred sai)
    - ocr_error: Các lỗi OCR tổng hợp khác
    """
    gt_norm = normalize_plate_text(ground_truth)
    raw_norm = normalize_plate_text(raw_pred)
    corr_norm = normalize_plate_text(corrected_pred)

    if iou < iou_threshold:
        return "wrong_box" if iou > 0 else "false_negative"
    if not detected:
        return "false_negative"
    if corr_norm == gt_norm:
        return "correct"
    if raw_norm == gt_norm and corr_norm != gt_norm:
        return "correction_error"
    if len(corr_norm) < len(gt_norm):
        return "missing_chars"
    if len(corr_norm) > len(gt_norm):
        return "extra_chars"
    if len(corr_norm) == len(gt_norm) and levenshtein_distance(gt_norm, corr_norm) > 0:
        return "char_confusion"
    return "ocr_error"


def generate_error_analysis_report(
    records: list[dict[str, Any]],
    output_csv: Path | str,
) -> pd.DataFrame:
    """Xuất DataFrame báo cáo phân loại lỗi và lưu ra tệp CSV."""
    df = pd.DataFrame(records)
    if "error_type" not in df.columns:
        df["error_type"] = [
            classify_error(
                row["ground_truth"],
                row["raw_prediction"],
                row["corrected_prediction"],
                row["detected"],
                row.get("iou", 1.0),
            )
            for _, row in df.iterrows()
        ]
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
