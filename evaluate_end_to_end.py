"""Đánh giá detection và OCR hoàn chỉnh trên annotation đã phiên âm."""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from src.metrics import box_iou, summarize_ocr
from src.pipeline import LicensePlateRecognizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="CSV gồm image_path,x1,y1,x2,y2,plate_text")
    parser.add_argument("--output", type=Path, default=Path("artifacts/end_to_end_metrics.json"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.annotations)
    required = {"image_path", "x1", "y1", "x2", "y2", "plate_text"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
    if frame.empty or frame["plate_text"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Annotation end-to-end không được rỗng")

    recognizer = LicensePlateRecognizer(args.weights, gpu=False if args.cpu else None)
    base_directory = args.annotations.resolve().parent
    records, latencies = [], []
    for image_name, ground_truths in frame.groupby("image_path", sort=False):
        image_path = Path(image_name)
        image_path = image_path if image_path.is_absolute() else base_directory / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
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
                "image_path": str(image_path), "ground_truth": str(truth.plate_text),
                "raw_prediction": prediction["raw_text"], "corrected_prediction": prediction["text"],
                "detected": matched, "iou": best_iou,
            })

    predictions_frame = pd.DataFrame(records)
    payload = {
        "detection_recall_at_iou": float(predictions_frame["detected"].mean()),
        "iou_threshold": args.iou_threshold,
        "raw": summarize_ocr(zip(predictions_frame["ground_truth"], predictions_frame["raw_prediction"])),
        "corrected": summarize_ocr(zip(predictions_frame["ground_truth"], predictions_frame["corrected_prediction"])),
        "mean_pipeline_latency_ms": sum(latencies) / len(latencies),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    predictions_frame.to_csv(args.output.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
