import numpy as np

from src.metrics import box_iou, summarize_ocr
from src.ocr import infer_plate_layout, validate_and_correct_plate
from src.rectification import order_points, rectify_plate


def test_post_processing_keeps_raw_result_visible():
    result = validate_and_correct_plate("51FI2345")
    assert result["raw_text"] == "51FI2345"
    assert result["text"] == "51F12345"
    assert result["correction_cost"] == 1.0


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
