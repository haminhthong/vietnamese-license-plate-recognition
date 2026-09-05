"""Đánh giá toàn diện hai bước (Phát hiện biển số + Nhận dạng ký tự OCR) trên ảnh thực tế kèm tọa độ và chuỗi phiên âm ground truth."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.error_analysis import classify_error, generate_error_analysis_report
from src.io_utils import (
    read_image,
    require_columns,
    require_non_empty_text,
    resolve_relative_path,
    write_json,
)
from src.metrics import (
    bootstrap_confidence_intervals,
    compute_confusion_matrix,
    compute_positional_accuracy,
    compute_postprocessing_gain_harm,
    match_ground_truth_boxes,
    summarize_latencies,
    summarize_ocr,
)
from src.pipeline import LicensePlateRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Đánh giá pipeline End-to-End tính toán Recall@IoU, Accuracy OCR, Post-processing Gain/Harm và 95% Bootstrap CI."""
    parser = argparse.ArgumentParser(description="Đánh giá pipeline nhận diện biển số xe End-to-End.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn trọng số YOLOv8 (.pt)")
    parser.add_argument("--annotations", type=Path, required=True, help="Tệp CSV ground truth (image_path, x1, y1, x2, y2, plate_text)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/end_to_end_metrics.json"), help="Tệp JSON xuất kết quả")
    parser.add_argument("--error-analysis-output", type=Path, default=Path("artifacts/error_analysis.csv"), help="Tệp CSV phân tích lỗi")
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
        image = read_image(image_path)
        predictions = recognizer.predict(image)
        latencies.append(recognizer.last_latency_ms)

        for match in match_ground_truth_boxes(predictions, ground_truths, iou_threshold=args.iou_threshold):
            truth, prediction, matched, best_iou = match["truth"], match["prediction"], match["matched"], match["iou"]
            error_type = classify_error(
                ground_truth=str(truth.plate_text),
                raw_pred=prediction.get("raw_text", ""),
                corrected_pred=prediction.get("text", ""),
                detected=matched,
                iou=best_iou,
                iou_threshold=args.iou_threshold,
            )
            records.append({
                "image_path": str(image_path),
                "ground_truth": str(truth.plate_text),
                "raw_prediction": prediction.get("raw_text", ""),
                "corrected_prediction": prediction.get("text", ""),
                "detected": matched,
                "iou": best_iou,
                "error_type": error_type,
            })

    predictions_frame = pd.DataFrame(records)
    latency_stats = summarize_latencies(latencies)

    raw_pairs = list(zip(predictions_frame["ground_truth"], predictions_frame["raw_prediction"], strict=True))
    corr_pairs = list(zip(predictions_frame["ground_truth"], predictions_frame["corrected_prediction"], strict=True))

    raw_summary = summarize_ocr(raw_pairs)
    corr_summary = summarize_ocr(corr_pairs)
    post_metrics = compute_postprocessing_gain_harm(records)
    pos_accuracy = compute_positional_accuracy(corr_pairs)
    confusion_matrix = compute_confusion_matrix(corr_pairs)
    bootstrap_cis = bootstrap_confidence_intervals(corr_pairs)

    total_gt_plates = len(predictions_frame)
    detected_count = int(predictions_frame["detected"].sum())
    exact_e2e_count = sum(r["detected"] and r["ground_truth"] == r["corrected_prediction"] for r in records)

    payload = {
        "total_ground_truth_plates": total_gt_plates,
        "detection_recall_at_iou": float(detected_count / max(1, total_gt_plates)),
        "conditional_ocr_exact_accuracy": float(exact_e2e_count / max(1, detected_count)),
        "end_to_end_exact_recall": float(exact_e2e_count / max(1, total_gt_plates)),
        "iou_threshold": args.iou_threshold,
        "raw": raw_summary,
        "corrected": corr_summary,
        "postprocessing_eval": post_metrics,
        "positional_accuracy": pos_accuracy,
        "character_confusion_matrix": confusion_matrix,
        "confidence_intervals": bootstrap_cis,
        **latency_stats,
        "error_summary": predictions_frame["error_type"].value_counts().to_dict(),
    }

    write_json(args.output, payload)
    generate_error_analysis_report(records, args.error_analysis_output)
    predictions_frame.to_csv(args.output.with_suffix(".predictions.csv"), index=False)

    logger.info("Báo cáo đánh giá End-to-End đã lưu tại: %s", args.output)
    logger.info("Báo cáo phân tích lỗi đã lưu tại: %s", args.error_analysis_output)
    logger.info(
        "Detector Recall@IoU>=%.2f: %.2f%% | E2E Exact Recall: %.2f%% | Latency Mean: %.1f ms | P95: %.1f ms",
        args.iou_threshold,
        payload["detection_recall_at_iou"] * 100,
        payload["end_to_end_exact_recall"] * 100,
        payload["mean_latency_ms"],
        payload["p95_latency_ms"],
    )


if __name__ == "__main__":
    main()
