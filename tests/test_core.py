"""Kiểm thử đơn vị tự động (Unit Tests) cho các module cốt lõi trong package src."""

import numpy as np
import pandas as pd
import pytest

from src.config import TrainingConfig
from src.io_utils import (
    read_image,
    require_columns,
    require_non_empty_text,
    resolve_relative_path,
    write_json,
)
from src.metrics import box_iou, levenshtein_distance, summarize_ocr
from src.ocr import infer_plate_layout, normalize_plate_text, order_ocr_tokens, validate_and_correct_plate
from src.pipeline import RecognitionConfig, crop_with_padding
from src.rectification import order_points, rectify_plate


def test_post_processing_keeps_raw_result_visible():
    """Kiểm tra quy tắc chuẩn hóa và tự động sửa lỗi OCR theo template mẫu."""
    result = validate_and_correct_plate("51FI2345")
    assert result["raw_text"] == "51FI2345"
    assert result["text"] == "51F12345"
    assert result["format_valid"] is True
    assert result["correction_cost"] == 1.0
    assert normalize_plate_text(" 51F-123.45 ") == "51F12345"
    assert normalize_plate_text("51À12345") == "5112345"


def test_levenshtein_distance_and_ocr_metrics():
    """Kiểm tra khoảng cách Levenshtein và các thống kê chỉ số OCR (CER, Accuracy)."""
    assert levenshtein_distance("51F12345", "51F12345") == 0
    assert levenshtein_distance("51F12345", "51F12346") == 1

    metrics = summarize_ocr([("51F12345", "51F12345"), ("30A12345", "30A12346")])
    assert metrics["samples"] == 2
    assert metrics["exact_plate_accuracy"] == 0.5
    assert metrics["cer"] == 1 / 16
    assert metrics["character_accuracy"] == 1.0 - (1 / 16)


def test_geometry_and_rectification_helpers():
    """Kiểm tra thuật toán sắp xếp 4 đỉnh góc, biến đổi phối cảnh và tính IoU."""
    points = order_points(np.array([[10, 10], [0, 0], [0, 10], [10, 0]]))
    assert points.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]

    blank = np.zeros((50, 120, 3), dtype=np.uint8)
    output, changed = rectify_plate(blank)
    assert output.shape == blank.shape and changed is False
    assert infer_plate_layout(blank) == "1_line"
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_pipeline_configuration_and_crop_validation():
    """Kiểm tra xác thực tham số RecognitionConfig và cắt ảnh có khoảng đệm."""
    RecognitionConfig(detection_confidence=0.4, padding_ratio=0.1)
    with pytest.raises(ValueError):
        RecognitionConfig(detection_confidence=1.1)

    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop, padded_box = crop_with_padding(image, (5, 5, 15, 15), padding_ratio=0.5)
    assert crop.shape[:2] == (20, 20)
    assert padded_box == (0, 0, 20, 20)

    with pytest.raises(ValueError):
        crop_with_padding(image, (10, 10, 5, 5))


def test_rejects_invalid_ocr_options():
    """Kiểm tra xử lý ngoại lệ khi tham số OCR hoặc bố cục không hợp lệ."""
    with pytest.raises(ValueError):
        infer_plate_layout(np.zeros((10, 10, 3)), wide_ratio_threshold=0)
    with pytest.raises(ValueError):
        order_ocr_tokens([], "unknown")


def test_training_config_and_table_validation(tmp_path):
    """Kiểm tra nạp cấu hình từ tệp YAML và các hàm kiểm tra ràng buộc cột DataFrame."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text("model: yolov8n.pt\nepochs: 5\nbatch: 2\nimgsz: 320\n", encoding="utf-8")
    config = TrainingConfig.from_yaml(config_path)
    assert config.image_size == 320
    assert config.override(epochs=10).epochs == 10

    frame = pd.DataFrame({"plate_text": ["51F12345"]})
    require_columns(frame, {"plate_text"}, "test")
    require_non_empty_text(frame, "plate_text", "test")

    with pytest.raises(ValueError):
        require_columns(frame, {"crop_path"}, "test")


def test_io_utils_helpers(tmp_path):
    """Kiểm tra các tiện ích I/O ghi JSON và giải quyết đường dẫn tương đối."""
    json_path = tmp_path / "sub" / "output.json"
    written = write_json(json_path, {"status": "success"})
    assert written.is_file()

    base_dir = tmp_path / "project"
    resolved = resolve_relative_path("data/test.jpg", base_dir)
    assert resolved == base_dir / "data" / "test.jpg"

    with pytest.raises(ValueError, match="Không đọc được ảnh kiểm thử"):
        read_image(tmp_path / "missing.jpg", "ảnh kiểm thử")


def test_error_analysis_classification():
    """Kiểm tra logic phân loại các nhóm lỗi trong module error_analysis."""
    from src.error_analysis import classify_error

    assert classify_error("51F12345", "51F12345", "51F12345", detected=True, iou=0.8) == "correct"
    assert classify_error("51F12345", "", "", detected=False, iou=0.0) == "false_negative"
    assert classify_error("51F12345", "", "", detected=False, iou=0.3) == "wrong_box"
    assert classify_error("51F12345", "51F12345", "51F12345", detected=True, iou=0.3) == "wrong_box"
    assert classify_error("51F12345", "51F1234", "51F1234", detected=True, iou=0.8) == "missing_chars"
    assert classify_error("51F12345", "51F123456", "51F123456", detected=True, iou=0.8) == "extra_chars"
    assert classify_error("51F12345", "51F12346", "51F12346", detected=True, iou=0.8) == "char_confusion"
    assert classify_error("51F12345", "51F12345", "51F12346", detected=True, iou=0.8) == "correction_error"


def test_phash_helpers_and_dataset_audit():
    """Kiểm tra thuật toán pHash và khoảng cách Hamming giữa hai chuỗi băm."""
    from src.dataset import phash_hamming_distance

    assert phash_hamming_distance("0000000000000000", "0000000000000000") == 0
    assert phash_hamming_distance("ffffffffffffffff", "0000000000000000") == 64
    assert phash_hamming_distance("000000000000000f", "0000000000000000") == 4


def test_recognition_config_from_yaml_and_validation(tmp_path):
    """Kiểm tra nạp RecognitionConfig từ tệp YAML và các giới hạn hợp lệ."""
    yaml_file = tmp_path / "rec.yaml"
    yaml_file.write_text("detection_confidence: 0.35\nnms_iou: 0.5\nimage_size: 320\n", encoding="utf-8")
    config = RecognitionConfig.from_yaml(yaml_file)
    assert config.detection_confidence == 0.35
    assert config.nms_iou == 0.5
    assert config.image_size == 320

    with pytest.raises(ValueError):
        RecognitionConfig(ocr_minimum_confidence=-0.1)
    with pytest.raises(ValueError):
        RecognitionConfig(correction_penalty=-0.1)
    with pytest.raises(ValueError):
        RecognitionConfig(single_variant_mode="khong_hop_le")
