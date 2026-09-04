"""Script đánh giá ablation benchmark thử nghiệm vai trò từng thành phần (Preprocessing, Rectification, Post-processing)."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import RecognitionConfig
from src.io_utils import (
    read_image,
    require_columns,
    require_non_empty_text,
    resolve_relative_path,
    write_json,
)
from src.metrics import match_ground_truth_boxes, summarize_latencies, summarize_ocr
from src.pipeline import LicensePlateRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ABLATION_CONFIGS = {
    "B0_crop_goc": RecognitionConfig(
        single_variant_mode="crop",
        enable_rectification=False,
        enable_template_correction=False,
    ),
    "B1_gray": RecognitionConfig(
        single_variant_mode="gray",
        enable_rectification=False,
        enable_template_correction=False,
    ),
    "B2_4_bien_the": RecognitionConfig(
        enable_preprocessing_variants=True,
        enable_rectification=False,
        enable_template_correction=False,
    ),
    "B3_4_bien_the_nan_anh": RecognitionConfig(
        enable_preprocessing_variants=True,
        enable_rectification=True,
        enable_template_correction=False,
    ),
    "Final_day_du": RecognitionConfig(
        enable_preprocessing_variants=True,
        enable_rectification=True,
        enable_template_correction=True,
    ),
}


def run_ablation_evaluation(
    weights: Path,
    annotations: Path,
    iou_threshold: float = 0.5,
    cpu: bool = False,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Chạy đánh giá ablation trên tất cả 5 mốc cấu hình pipeline."""
    frame = pd.read_csv(annotations)
    require_columns(frame, {"image_path", "x1", "y1", "x2", "y2", "plate_text"}, "Annotation ablation")
    require_non_empty_text(frame, "plate_text", "Annotation ablation")

    base_directory = annotations.resolve().parent
    image_paths = frame["image_path"].unique()
    images = {
        img_name: read_image(resolve_relative_path(img_name, base_directory))
        for img_name in image_paths
    }

    results = {}
    rows = []

    for name, config in ABLATION_CONFIGS.items():
        logger.info("Chạy ablation cấu hình: %s...", name)
        recognizer = LicensePlateRecognizer(weights, gpu=False if cpu else None, config=config)
        records, latencies = [], []

        for image_name, ground_truths in frame.groupby("image_path", sort=False):
            image = images[image_name]
            predictions = recognizer.predict(image)
            latencies.append(recognizer.last_latency_ms)

            for match in match_ground_truth_boxes(predictions, ground_truths, iou_threshold=iou_threshold):
                truth, prediction, matched = match["truth"], match["prediction"], match["matched"]
                records.append({
                    "ground_truth": str(truth.plate_text),
                    "prediction": prediction["text"] if config.enable_template_correction else prediction["raw_text"],
                    "detected": matched,
                })

        del recognizer

        pred_df = pd.DataFrame(records)
        ocr_stats = summarize_ocr(zip(pred_df["ground_truth"], pred_df["prediction"], strict=True))
        latency_stats = summarize_latencies(latencies)
        recall = float(pred_df["detected"].mean())

        config_result = {
            "name": name,
            "detection_recall": recall,
            **ocr_stats,
            **latency_stats,
        }
        results[name] = config_result
        rows.append({
            "Cấu hình": name,
            "Exact Accuracy": f"{ocr_stats['exact_plate_accuracy'] * 100:.2f}%",
            "CER": f"{ocr_stats['cer']:.4f}",
            "Latency Mean (ms)": f"{latency_stats['mean_latency_ms']:.1f}",
            "Latency P95 (ms)": f"{latency_stats['p95_latency_ms']:.1f}",
            "Detection Recall": f"{recall * 100:.2f}%",
        })

    return results, pd.DataFrame(rows)


def main() -> None:
    """Thực thi benchmark ablation và xuất kết quả báo cáo."""
    parser = argparse.ArgumentParser(description="Chạy ablation benchmark cho pipeline VLPR.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn trọng số YOLOv8 (.pt)")
    parser.add_argument("--annotations", type=Path, required=True, help="Tệp CSV ground truth end-to-end")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ablation_metrics.json"), help="Tệp JSON kết quả")
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/ablation.csv"), help="Tệp CSV bảng ablation")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Ngưỡng IoU")
    parser.add_argument("--cpu", action="store_true", help="Ép buộc thực thi trên CPU")
    args = parser.parse_args()

    metrics, table_df = run_ablation_evaluation(args.weights, args.annotations, args.iou_threshold, args.cpu)
    write_json(args.output, metrics)
    table_df.to_csv(args.output_csv, index=False)

    logger.info("Hoàn tất benchmark ablation! Kết quả đã lưu tại:")
    logger.info("JSON: %s", args.output)
    logger.info("CSV:  %s", args.output_csv)
    print("\n--- BẢNG BÁO CÁO ABLATION ---")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
