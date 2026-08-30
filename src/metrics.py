"""Các công thức đo lường và đánh giá hiệu năng cho mô hình phát hiện (Detection) và nhận dạng ký tự (OCR).

Module này định nghĩa:
1. `levenshtein_distance`: Khoảng cách chỉnh sửa tối thiểu giữa hai chuỗi văn bản.
2. `box_iou`: Chỉ số Intersection-over-Union giữa hai khung hình chữ nhật.
3. `summarize_ocr`: Thống kê Exact Plate Accuracy, CER (Character Error Rate) và Character Accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .ocr import normalize_plate_text


def levenshtein_distance(source: str, target: str) -> int:
    """Tính khoảng cách chỉnh sửa Levenshtein giữa chuỗi nguồn và chuỗi đích sau khi chuẩn hóa.

    Args:
        source (str): Chuỗi ký tự biển số gốc (nhãn thực tế).
        target (str): Chuỗi ký tự biển số dự đoán.

    Returns:
        int: Số phép thay thế, thêm, hoặc xóa ký tự tối thiểu để biến chuỗi source thành target.
    """
    source, target = normalize_plate_text(source), normalize_plate_text(target)
    previous = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, 1):
        current = [source_index]
        for target_index, target_character in enumerate(target, 1):
            current.append(min(
                current[target_index - 1] + 1,
                previous[target_index] + 1,
                previous[target_index - 1] + (source_character != target_character),
            ))
        previous = current
    return previous[-1]


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Tính chỉ số Giao trên Hợp (Intersection over Union - IoU) giữa hai Bounding Box dạng (x1, y1, x2, y2).

    Args:
        box_a: Bounding box thứ nhất.
        box_b: Bounding box thứ hai.

    Returns:
        float: Giá trị IoU nằm trong khoảng [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def summarize_ocr(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Tổng hợp các chỉ số đánh giá độ chính xác nhận dạng OCR trên một tập cặp mẫu (Ground Truth, Prediction).

    Args:
        pairs (Iterable[tuple[str, str]]): Tập các cặp (nhãn_thực_tế, chuỗi_dự_đoán).

    Returns:
        dict[str, Any]: Dictionary chứa các chỉ số:
            - samples: Số lượng mẫu đánh giá.
            - exact_plate_accuracy: Tỷ lệ biển số đọc chính xác 100%.
            - cer: Character Error Rate (Tổng số phép sửa / Tổng số ký tự).
            - character_accuracy: Độ chính xác cấp ký tự max(0, 1 - CER).

    Raises:
        ValueError: Nếu tập mẫu rỗng hoặc chuỗi nhãn thực tế bị rỗng.
    """
    normalized = [(normalize_plate_text(truth), normalize_plate_text(prediction)) for truth, prediction in pairs]
    if not normalized:
        raise ValueError("Không có mẫu dữ liệu OCR nào để đánh giá.")
    total_characters = sum(len(truth) for truth, _ in normalized)
    if total_characters == 0:
        raise ValueError("Chuỗi nhãn OCR thực tế không được để rỗng hoàn toàn.")
    total_edits = sum(levenshtein_distance(truth, prediction) for truth, prediction in normalized)
    exact_matches = sum(truth == prediction for truth, prediction in normalized)
    cer = total_edits / total_characters
    return {
        "samples": len(normalized),
        "exact_plate_accuracy": exact_matches / len(normalized),
        "cer": cer,
        "character_accuracy": max(0.0, 1.0 - cer),
    }
