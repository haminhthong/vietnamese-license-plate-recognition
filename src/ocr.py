"""Tiền xử lý OCR và chuẩn hóa kết quả theo định dạng biển số Việt Nam."""

from __future__ import annotations

from typing import Any

import numpy as np

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
    return "".join(character for character in str(text).upper() if character.isalnum())


def fit_plate_template(raw_text: str, template: str) -> dict[str, Any] | None:
    raw_text = normalize_plate_text(raw_text)
    if len(raw_text) != len(template):
        return None
    output: list[str] = []
    correction_cost = 0.0
    for character, expected_type in zip(raw_text, template):
        substitutions = DIGIT_SUBSTITUTIONS if expected_type == "D" else LETTER_SUBSTITUTIONS
        is_valid = character in "0123456789" if expected_type == "D" else character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
        sorted_y = sorted(tokens, key=lambda token: token["center_y"])
        gaps = [sorted_y[i + 1]["center_y"] - sorted_y[i]["center_y"] for i in range(len(sorted_y) - 1)]
        split_index = int(np.argmax(gaps)) + 1 if gaps else len(sorted_y)
        if gaps and max(gaps) >= 0.25 * median_height:
            rows = [sorted_y[:split_index], sorted_y[split_index:]]
        else:
            median_y = float(np.median([token["center_y"] for token in tokens]))
            rows = [
                [token for token in tokens if token["center_y"] <= median_y],
                [token for token in tokens if token["center_y"] > median_y],
            ]
        rows = [row for row in rows if row]
        rows.sort(key=lambda row: np.mean([token["center_y"] for token in row]))
        ordered = [token for row in rows for token in sorted(row, key=lambda token: token["center_x"])]
    else:
        ordered = sorted(tokens, key=lambda token: token["center_x"])

    text = "".join(token["text"] for token in ordered)
    confidence = float(np.average(
        [token["confidence"] for token in ordered],
        weights=[max(1, len(token["text"])) for token in ordered],
    ))
    return text, confidence


def preprocess_plate_variants(crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
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
    if crop_bgr is None or crop_bgr.size == 0:
        return "1_line"
    height, width = crop_bgr.shape[:2]
    return "1_line" if width / max(1, height) >= wide_ratio_threshold else "2_line"


def read_plate(reader: Any, crop_bgr: np.ndarray, layout: str = "auto") -> dict[str, Any]:
    resolved_layout = layout if layout in {"1_line", "2_line"} else infer_plate_layout(crop_bgr)
    candidates = []
    for variant_name, processed in preprocess_plate_variants(crop_bgr).items():
        results = reader.readtext(
            processed, detail=1, paragraph=False,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.",
        )
        raw_text, confidence = order_ocr_tokens(results, resolved_layout)
        validated = validate_and_correct_plate(raw_text)
        score = confidence + (0.20 if validated["format_valid"] else 0.0) - 0.07 * validated["correction_cost"]
        candidates.append({
            **validated, "ocr_confidence": confidence, "layout": resolved_layout,
            "variant": variant_name, "score": score,
        })
    if candidates:
        return max(candidates, key=lambda item: item["score"])
    return {
        "raw_text": "", "text": "", "format_valid": False, "template": None,
        "correction_cost": 0.0, "ocr_confidence": 0.0, "layout": resolved_layout,
        "variant": None, "score": 0.0,
    }
