"""Các phép đo hỗ trợ đánh giá detection và OCR."""

from collections.abc import Iterable

from .ocr import normalize_plate_text


def levenshtein_distance(source: str, target: str) -> int:
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


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def summarize_ocr(pairs: Iterable[tuple[str, str]]) -> dict[str, float | int]:
    normalized = [(normalize_plate_text(truth), normalize_plate_text(prediction)) for truth, prediction in pairs]
    if not normalized:
        raise ValueError("Không có mẫu OCR để đánh giá")
    total_characters = sum(len(truth) for truth, _ in normalized)
    if total_characters == 0:
        raise ValueError("Nhãn OCR không được để trống")
    total_edits = sum(levenshtein_distance(truth, prediction) for truth, prediction in normalized)
    exact_matches = sum(truth == prediction for truth, prediction in normalized)
    cer = total_edits / total_characters
    return {"samples": len(normalized), "exact_plate_accuracy": exact_matches / len(normalized), "cer": cer, "character_accuracy": max(0.0, 1.0 - cer)}
