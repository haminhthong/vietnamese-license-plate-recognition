"""Tiền xử lý ảnh biển số, trích xuất OCR và hậu xử lý khớp mẫu định dạng biển số xe Việt Nam.

Module này chịu trách nhiệm:
1. Xử lý nhiều biến thể ảnh (Gray, CLAHE, Otsu, Adaptive Threshold) để EasyOCR có tỷ lệ đọc cao nhất.
2. Xác định bố cục biển 1 dòng (ô tô dài) hoặc 2 dòng (xe máy, ô tô ngắn) dựa trên aspect ratio.
3. Sắp xếp các token nhận dạng được theo đúng thứ tự hình học (từ trên xuống dưới, từ trái sang phải).
4. Chuẩn hóa chuỗi ký tự, thay thế các lỗi nhận dạng phổ biến (nhầm lẫn giữa 'O' và '0', 'B' và '8', 'I' và '1', v.v.).
5. Khớp các mẫu định dạng biển số tiêu chuẩn Việt Nam để sửa lỗi tối ưu.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import RecognitionConfig
from .rectification import rectify_plate

# Bộ ký tự hợp lệ và phép thay thế chuẩn hóa
ASCII_DIGITS = "0123456789"
ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OCR_ALLOWLIST = f"{ASCII_DIGITS}{ASCII_LETTERS}-."
VALID_LAYOUTS = {"1_line", "2_line"}

# Từ điển thay thế ký tự nhầm lẫn giữa chữ cái và chữ số
DIGIT_SUBSTITUTIONS = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
    "J": "3", "A": "4", "S": "5", "G": "6", "B": "8",
}
LETTER_SUBSTITUTIONS = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "8": "B"}

# Các mẫu biển số Việt Nam tiêu chuẩn (D = Digit/Chữ số, L = Letter/Chữ cái Latin)
PLATE_TEMPLATES = {
    7: ["DDLDDDD"],                  # Ví dụ: 51F1234 (7 ký tự)
    8: ["DDLDDDDD"],                 # Ví dụ: 51F12345 (8 ký tự)
    9: ["DDLDDDDDD", "DDLLDDDDD"],   # Ví dụ: 51F123456 hoặc 51AB12345 (9 ký tự)
    10: ["DDLLDDDDDD"],              # Ví dụ: 51AB123456 (10 ký tự)
}


def normalize_plate_text(text: str) -> str:
    """Chuẩn hóa chuỗi ký tự: Chuyển thành viết hoa và loại bỏ các ký tự không thuộc ASCII chữ cái/số.

    Args:
        text (str): Chuỗi đầu vào.

    Returns:
        str: Chuỗi chỉ chứa chữ cái Latin viết hoa A-Z và chữ số 0-9.
    """
    allowed = set(ASCII_DIGITS + ASCII_LETTERS)
    return "".join(character for character in str(text).upper() if character in allowed)


def fit_plate_template(raw_text: str, template: str) -> dict[str, Any] | None:
    """Khớp chuỗi văn bản nhận dạng với một template mẫu và tính toán chi phí sửa lỗi (correction cost).

    Args:
        raw_text (str): Chuỗi OCR thô đã chuẩn hóa.
        template (str): Chuỗi mẫu quy định dạng 'DDLDDDD' (D: Chữ số, L: Chữ cái).

    Returns:
        dict[str, Any] | None: Kết quả sau khi sửa lỗi hoặc None nếu không khớp độ dài.
    """
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


def validate_and_correct_plate(
    raw_text: str,
    enable_correction: bool = True,
    max_cost: float = 1.0,
) -> dict[str, Any]:
    """Kiểm tra tính hợp lệ và tự động hiệu chỉnh kết quả OCR theo các quy tắc định dạng biển số xe.

    Args:
        raw_text (str): Chuỗi ký tự nhận dạng thô.
        enable_correction (bool): Có bật tự động sửa ký tự nhầm lẫn hay không.
        max_cost (float): Giới hạn chi phí hiệu chỉnh tối đa.

    Returns:
        dict[str, Any]: Kết quả chi tiết bao gồm chuỗi thô, chuỗi đã sửa, cờ format_valid, cờ correction_applied,
                        cờ needs_manual_review và chi phí sửa.
    """
    normalized = normalize_plate_text(raw_text)
    candidates = [
        result
        for template in PLATE_TEMPLATES.get(len(normalized), [])
        if (result := fit_plate_template(normalized, template)) is not None
    ]
    if not candidates:
        return {
            "raw_text": normalized,
            "text": normalized,
            "format_valid": False,
            "template": None,
            "correction_cost": 0.0,
            "correction_applied": False,
            "needs_manual_review": True,
        }

    best = min(candidates, key=lambda item: item["correction_cost"])
    if not enable_correction:
        # Nếu tắt hiệu chỉnh template, giữ nguyên raw_text nhưng thông báo nếu raw_text vốn đã chuẩn format (cost == 0)
        is_exact = best["correction_cost"] == 0.0
        return {
            "raw_text": normalized,
            "text": normalized,
            "format_valid": is_exact,
            "template": best["template"] if is_exact else None,
            "correction_cost": 0.0,
            "correction_applied": False,
            "needs_manual_review": not is_exact,
        }

    correction_applied = best["text"] != normalized
    needs_manual_review = best["correction_cost"] > max_cost
    return {
        "raw_text": normalized,
        "format_valid": True,
        "correction_applied": correction_applied,
        "needs_manual_review": needs_manual_review,
        **best,
    }


def _token_geometry(bbox: list[list[float]]) -> dict[str, float]:
    """Trích xuất thông tin tọa độ tâm và chiều cao của bounding box token OCR."""
    points = np.asarray(bbox, dtype=float)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return {
        "center_x": float((min_x + max_x) / 2),
        "center_y": float((min_y + max_y) / 2),
        "height": float(max(1.0, max_y - min_y)),
    }


def order_ocr_tokens(ocr_results: list, layout: str, minimum_confidence: float = 0.20) -> tuple[str, float]:
    """Lọc các token nhiễu và sắp xếp theo đúng thứ tự đọc dựa trên bố cục 1 dòng hoặc 2 dòng.

    Args:
        ocr_results (list): Kết quả trả về từ EasyOCR.
        layout (str): Bố cục biển số ('1_line' hoặc '2_line').
        minimum_confidence (float): Độ tin cậy tối thiểu để giữ lại token.

    Returns:
        tuple[str, float]: Cặp (chuỗi_ký_tự_hoàn_chỉnh, độ_tin_cậy_trung_bình).
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence phải nằm trong khoảng [0, 1]")
    tokens = []
    for bbox, text, confidence in ocr_results:
        normalized = normalize_plate_text(text)
        if normalized and float(confidence) >= minimum_confidence:
            tokens.append({
                "text": normalized,
                "confidence": float(confidence),
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
    """Phân tách các token thành 2 hàng (trên/dưới) và sắp xếp từng hàng từ trái sang phải."""
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
    """Tạo ra 4 biến thể ảnh tiền xử lý (Gray, CLAHE, Otsu, Adaptive Threshold) để tăng khả năng đọc OCR.

    Args:
        crop_bgr (np.ndarray): Ảnh crop vùng biển số gốc.

    Returns:
        dict[str, np.ndarray]: Từ điển chứa các ảnh biến thể.
    """
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
    """Ước lượng tự động biển số là 1 dòng hay 2 dòng dựa trên tỷ lệ chiều rộng / chiều cao.

    Args:
        crop_bgr (np.ndarray): Ảnh crop biển số.
        wide_ratio_threshold (float): Ngưỡng tỷ lệ khung hình (mặc định: 2.20).

    Returns:
        str: '1_line' nếu là biển dài hoặc '2_line' nếu là biển vuông/gần vuông.
    """
    if wide_ratio_threshold <= 0:
        raise ValueError("wide_ratio_threshold phải lớn hơn 0")
    if crop_bgr is None or crop_bgr.size == 0:
        return "1_line"
    height, width = crop_bgr.shape[:2]
    return "1_line" if width / max(1, height) >= wide_ratio_threshold else "2_line"


def read_plate(
    reader: Any,
    crop_bgr: np.ndarray,
    layout: str = "auto",
    config: RecognitionConfig | None = None,
) -> dict[str, Any]:
    """Thực hiện quy trình đọc biển số hoàn chỉnh: Nắn góc, tiền xử lý đa biến thể, OCR và sửa lỗi theo mẫu.

    Args:
        reader (Any): Đối tượng EasyOCR Reader.
        crop_bgr (np.ndarray): Ảnh crop biển số BGR.
        layout (str): Bố cục ('auto', '1_line', '2_line').
        config (RecognitionConfig | None): Cấu hình nhận diện.

    Returns:
        dict[str, Any]: Thông tin chi tiết kết quả đọc biển số tốt nhất.
    """
    if layout != "auto" and layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    cfg = config or RecognitionConfig()

    if cfg.enable_rectification:
        target_crop, rectified = rectify_plate(crop_bgr)
    else:
        target_crop, rectified = crop_bgr, False

    resolved_layout = layout if layout in VALID_LAYOUTS else infer_plate_layout(target_crop, cfg.wide_ratio_threshold)

    # Phân định biến thể ảnh sẽ nạp vào EasyOCR dựa theo cấu hình ablation
    if cfg.single_variant_mode == "crop":
        variants = {"crop": target_crop}
    elif cfg.single_variant_mode == "gray":
        all_vars = preprocess_plate_variants(target_crop)
        variants = {"gray": all_vars.get("gray", target_crop)}
    elif cfg.single_variant_mode and cfg.single_variant_mode in {"clahe", "otsu", "adaptive"}:
        all_vars = preprocess_plate_variants(target_crop)
        variants = {cfg.single_variant_mode: all_vars.get(cfg.single_variant_mode, target_crop)}
    elif not cfg.enable_preprocessing_variants:
        variants = {"crop": target_crop}
    else:
        variants = preprocess_plate_variants(target_crop)

    candidates = []
    for variant_name, processed in variants.items():
        results = reader.readtext(
            processed, detail=1, paragraph=False,
            allowlist=OCR_ALLOWLIST,
        )
        raw_text, confidence = order_ocr_tokens(
            results, resolved_layout, minimum_confidence=cfg.ocr_minimum_confidence
        )
        validated = validate_and_correct_plate(
            raw_text,
            enable_correction=cfg.enable_template_correction,
            max_cost=cfg.max_correction_cost,
        )
        score = (
            confidence
            + (cfg.valid_format_bonus if validated["format_valid"] else 0.0)
            - cfg.correction_penalty * validated["correction_cost"]
        )
        candidates.append({
            **validated,
            "ocr_confidence": confidence,
            "layout": resolved_layout,
            "rectified": rectified,
            "variant": variant_name,
            "score": score,
        })
    if candidates:
        return max(candidates, key=lambda item: item["score"])
    return {
        "raw_text": "", "text": "", "format_valid": False, "template": None,
        "correction_cost": 0.0, "correction_applied": False, "needs_manual_review": True,
        "ocr_confidence": 0.0, "layout": resolved_layout,
        "variant": None, "score": 0.0, "rectified": rectified,
    }
