"""Tiền xử lý OCR và chuẩn hóa kết quả theo định dạng biển số Việt Nam."""

from __future__ import annotations

from typing import Any

import numpy as np

from .rectification import rectify_plate

ASCII_DIGITS = "0123456789"
ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OCR_ALLOWLIST = f"{ASCII_DIGITS}{ASCII_LETTERS}-."
VALID_LAYOUTS = {"1_line", "2_line"}

DIGIT_SUBSTITUTIONS = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
    "J": "3", "A": "4", "S": "5", "G": "6", "B": "8",
}
LETTER_SUBSTITUTIONS = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "8": "B"}

# D là chữ số, L là chữ cái Latin viết hoa. Có thể bổ sung mẫu cho các loại biển đặc biệt.
PLATE_TEMPLATES = {
    7: ["DDLDDDD"],
    8: ["DDLDDDDD"],
    9: ["DDLDDDDDD", "DDLLDDDDD"],
    10: ["DDLLDDDDDD"],
}


def normalize_plate_text(text: str) -> str:
    """Chuẩn hóa về chữ Latin viết hoa và chữ số ASCII."""
    allowed = set(ASCII_DIGITS + ASCII_LETTERS)
    return "".join(character for character in str(text).upper() if character in allowed)


def fit_plate_template(raw_text: str, template: str) -> dict[str, Any] | None:
    raw_text = normalize_plate_text(raw_text)
    if len(raw_text) != len(template):
        return None
    output: list[str] = []
    correction_cost = 0.0
    for character, expected_type in zip(raw_text, template, strict=True):
        substitutions = DIGIT_SUBSTITUTIONS if expected_type == "D" else LETTER_SUBSTITUTIONS
        is_valid = character in ASCII_DIGITS if expected_type == "D" else character in ASCII_LETTERS
        if is_valid:
            output.append(character)
        elif character in substitutions:
            output.append(substitutions[character])
            correction_cost += 1.0
        else:
            return None
    return {"text": "".join(output), "template": template, "correction_cost": correction_cost}


def validate_and_correct_plate(raw_text: str) -> dict[str, Any]:
    normalized = normalize_plate_text(raw_text)
    candidates = [
        result
        for template in PLATE_TEMPLATES.get(len(normalized), [])
        if (result := fit_plate_template(normalized, template)) is not None
    ]
    if not candidates:
        return {
            "raw_text": normalized, "text": normalized, "format_valid": False,
            "template": None, "correction_cost": 0.0,
        }
    best = min(candidates, key=lambda item: item["correction_cost"])
    return {"raw_text": normalized, "format_valid": True, **best}


def _token_geometry(bbox: list[list[float]]) -> dict[str, float]:
    points = np.asarray(bbox, dtype=float)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return {
        "center_x": float((min_x + max_x) / 2),
        "center_y": float((min_y + max_y) / 2),
        "height": float(max(1.0, max_y - min_y)),
    }


def order_ocr_tokens(ocr_results: list, layout: str, minimum_confidence: float = 0.20) -> tuple[str, float]:
    """Lọc và sắp xếp token OCR theo bố cục biển số."""
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence phải nằm trong [0, 1]")
    tokens = []
    for bbox, text, confidence in ocr_results:
        normalized = normalize_plate_text(text)
        if normalized and float(confidence) >= minimum_confidence:
            tokens.append({
                "text": normalized, "confidence": float(confidence),
                **_token_geometry(bbox),
            })
    if not tokens:
        return "", 0.0

    median_height = float(np.median([token["height"] for token in tokens]))
    tokens = [token for token in tokens if token["height"] >= 0.45 * median_height]
    if not tokens:
        return "", 0.0

    if layout == "2_line" and len(tokens) >= 2:
        ordered = _order_two_line_tokens(tokens, median_height)
    else:
        ordered = sorted(tokens, key=lambda token: token["center_x"])

    text = "".join(token["text"] for token in ordered)
    confidence = float(np.average(
        [token["confidence"] for token in ordered],
        weights=[max(1, len(token["text"])) for token in ordered],
    ))
    return text, confidence


def _order_two_line_tokens(tokens: list[dict[str, Any]], median_height: float) -> list[dict[str, Any]]:
    """Tách token thành hai hàng rồi đọc từ trái sang phải, trên xuống dưới."""
    sorted_by_y = sorted(tokens, key=lambda token: token["center_y"])
    gaps = [
        sorted_by_y[index + 1]["center_y"] - sorted_by_y[index]["center_y"]
        for index in range(len(sorted_by_y) - 1)
    ]
    largest_gap = max(gaps, default=0.0)
    if largest_gap >= 0.25 * median_height:
        split_index = int(np.argmax(gaps)) + 1
        rows = [sorted_by_y[:split_index], sorted_by_y[split_index:]]
    else:
        median_y = float(np.median([token["center_y"] for token in tokens]))
        rows = [
            [token for token in tokens if token["center_y"] <= median_y],
            [token for token in tokens if token["center_y"] > median_y],
        ]
    non_empty_rows = [row for row in rows if row]
    non_empty_rows.sort(key=lambda row: np.mean([token["center_y"] for token in row]))
    return [
        token
        for row in non_empty_rows
        for token in sorted(row, key=lambda token: token["center_x"])
    ]


def preprocess_plate_variants(crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Tạo các biến thể ảnh để OCR chọn kết quả tốt nhất."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return {}
    height, width = crop_bgr.shape[:2]
    scale = float(np.clip(180 / max(height, width), 2.0, 4.0))
    enlarged = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    denoised = cv2.bilateralFilter(gray, 9, 55, 55)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    return {"gray": gray, "clahe": clahe, "otsu": otsu, "adaptive": adaptive}


def infer_plate_layout(crop_bgr: np.ndarray, wide_ratio_threshold: float = 2.20) -> str:
    """Ước lượng biển một hoặc hai dòng bằng tỷ lệ khung ảnh."""
    if wide_ratio_threshold <= 0:
        raise ValueError("wide_ratio_threshold phải lớn hơn 0")
    if crop_bgr is None or crop_bgr.size == 0:
        return "1_line"
    height, width = crop_bgr.shape[:2]
    return "1_line" if width / max(1, height) >= wide_ratio_threshold else "2_line"


def read_plate(reader: Any, crop_bgr: np.ndarray, layout: str = "auto") -> dict[str, Any]:
    """Chỉnh phối cảnh, chạy nhiều biến thể OCR và chọn ứng viên tốt nhất."""
    if layout != "auto" and layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    rectified_crop, rectified = rectify_plate(crop_bgr)
    resolved_layout = layout if layout in VALID_LAYOUTS else infer_plate_layout(rectified_crop)
    candidates = []
    for variant_name, processed in preprocess_plate_variants(rectified_crop).items():
        results = reader.readtext(
            processed, detail=1, paragraph=False,
            allowlist=OCR_ALLOWLIST,
        )
        raw_text, confidence = order_ocr_tokens(results, resolved_layout)
        validated = validate_and_correct_plate(raw_text)
        score = confidence + (0.20 if validated["format_valid"] else 0.0) - 0.07 * validated["correction_cost"]
        candidates.append({
            **validated, "ocr_confidence": confidence, "layout": resolved_layout, "rectified": rectified,
            "variant": variant_name, "score": score,
        })
    if candidates:
        return max(candidates, key=lambda item: item["score"])
    return {
        "raw_text": "", "text": "", "format_valid": False, "template": None,
        "correction_cost": 0.0, "ocr_confidence": 0.0, "layout": resolved_layout,
        "variant": None, "score": 0.0, "rectified": rectified,
    }
