import numpy as np
import pandas as pd
import pytest

from src.config import TrainingConfig
from src.io_utils import require_columns, require_non_empty_text
from src.metrics import box_iou, summarize_ocr
from src.ocr import infer_plate_layout, normalize_plate_text, order_ocr_tokens, validate_and_correct_plate
from src.pipeline import RecognitionConfig, crop_with_padding
from src.rectification import order_points, rectify_plate


def test_post_processing_keeps_raw_result_visible():
    result = validate_and_correct_plate("51FI2345")
    assert result["raw_text"] == "51FI2345"
    assert result["text"] == "51F12345"
    assert result["correction_cost"] == 1.0
    assert normalize_plate_text(" 51F-123.45 ") == "51F12345"
    assert normalize_plate_text("51À12345") == "5112345"


def test_ocr_metrics():
    metrics = summarize_ocr([("51F12345", "51F12345"), ("30A12345", "30A12346")])
    assert metrics["exact_plate_accuracy"] == 0.5
    assert metrics["cer"] == 1 / 16


def test_geometry_helpers():
    points = order_points(np.array([[10, 10], [0, 0], [0, 10], [10, 0]]))
    assert points.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]
    blank = np.zeros((50, 120, 3), dtype=np.uint8)
    output, changed = rectify_plate(blank)
    assert output.shape == blank.shape and changed is False
    assert infer_plate_layout(blank) == "1_line"
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_pipeline_configuration_and_crop_validation():
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
    with pytest.raises(ValueError):
        infer_plate_layout(np.zeros((10, 10, 3)), wide_ratio_threshold=0)
    with pytest.raises(ValueError):
        order_ocr_tokens([], "unknown")


def test_training_config_and_table_validation(tmp_path):
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
