"""Đánh giá toàn diện cả hai bước (Phát hiện biển số + Nhận dạng ký tự OCR) trên ảnh thực tế kèm tọa độ và chuỗi phiên âm ground truth."""

import argparse
import logging
from pathlib import Path

import cv2
import pandas as pd

from src.io_utils import require_columns, require_non_empty_text, resolve_relative_path, write_json
from src.metrics import box_iou, summarize_ocr
from src.pipeline import LicensePlateRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Đánh giá pipeline End-to-End tính toán Recall@IoU, Accuracy OCR và latency trung bình."""
    parser = argparse.ArgumentParser(description="Đánh giá pipeline nhận diện biển số xe End-to-End.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn trọng số YOLOv8 (.pt)")
    parser.add_argument("--annotations", type=Path, required=True, help="Tệp CSV ground truth (image_path, x1, y1, x2, y2, plate_text)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/end_to_end_metrics.json"), help="Tệp JSON xuất kết quả")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Ngưỡng IoU coi như khớp bounding box")
    parser.add_argument("--cpu", action="store_true", help="Ép buộc thực thi trên CPU")
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations)
    require_columns(frame, {"image_path", "x1", "y1", "x2", "y2", "plate_text"}, "Annotation end-to-end")
    require_non_empty_text(frame, "plate_text", "Annotation end-to-end")

    logger.info("Khởi tạo pipeline LicensePlateRecognizer...")
    recognizer = LicensePlateRecognizer(args.weights, gpu=False if args.cpu else None)
    base_directory = args.annotations.resolve().parent
    records, latencies = [], []

    for image_name, ground_truths in frame.groupby("image_path", sort=False):
        image_path = resolve_relative_path(image_name, base_directory)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không đọc được tệp ảnh: {image_path}")
        predictions = recognizer.predict(image)
        latencies.append(recognizer.last_latency_ms)
        unused = set(range(len(predictions)))
        for truth in ground_truths.itertuples(index=False):
            truth_box = (truth.x1, truth.y1, truth.x2, truth.y2)
            candidates = [(index, box_iou(truth_box, predictions[index]["box"])) for index in unused]
            prediction_index, best_iou = max(candidates, key=lambda item: item[1], default=(None, 0.0))
            matched = prediction_index is not None and best_iou >= args.iou_threshold
            prediction = predictions[prediction_index] if matched else {"raw_text": "", "text": ""}
            if matched:
                unused.remove(prediction_index)
            records.append({
                "image_path": str(image_path),
                "ground_truth": str(truth.plate_text),
                "raw_prediction": prediction["raw_text"],
                "corrected_prediction": prediction["text"],
                "detected": matched,
                "iou": best_iou,
            })

    predictions_frame = pd.DataFrame(records)
    payload = {
        "detection_recall_at_iou": float(predictions_frame["detected"].mean()),
        "iou_threshold": args.iou_threshold,
        "raw": summarize_ocr(zip(predictions_frame["ground_truth"], predictions_frame["raw_prediction"], strict=True)),
        "corrected": summarize_ocr(
            zip(predictions_frame["ground_truth"], predictions_frame["corrected_prediction"], strict=True)
        ),
        "mean_pipeline_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }
    write_json(args.output, payload)
    predictions_frame.to_csv(args.output.with_suffix(".predictions.csv"), index=False)

    logger.info("Báo cáo đánh giá End-to-End đã lưu tại: %s", args.output)
    logger.info("Detection Recall@IoU>=%.2f: %.2f%% | Độ trễ trung bình: %.1f ms", args.iou_threshold, payload["detection_recall_at_iou"] * 100, payload["mean_pipeline_latency_ms"])


if __name__ == "__main__":
    main()
