"""Đánh giá OCR thô và OCR sau hậu xử lý từ tệp CSV ground truth."""

import argparse
import json
from pathlib import Path

import cv2
import easyocr
import pandas as pd
import torch

from src.metrics import summarize_ocr
from src.ocr import read_plate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True, help="CSV gồm crop_path, plate_text và layout tùy chọn")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ocr_metrics.json"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.annotations, dtype=str).fillna("")
    required = {"crop_path", "plate_text"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
    if frame.empty or (frame["plate_text"].str.strip() == "").any():
        raise ValueError("Ground truth OCR không được rỗng")

    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available() and not args.cpu)
    raw_pairs, corrected_pairs, rows = [], [], []
    base_directory = args.annotations.resolve().parent
    for row in frame.itertuples(index=False):
        crop_path = Path(row.crop_path)
        crop_path = crop_path if crop_path.is_absolute() else base_directory / crop_path
        image = cv2.imread(str(crop_path))
        if image is None:
            raise ValueError(f"Không đọc được crop: {crop_path}")
        result = read_plate(reader, image, getattr(row, "layout", "auto") or "auto")
        raw_pairs.append((row.plate_text, result["raw_text"]))
        corrected_pairs.append((row.plate_text, result["text"]))
        rows.append({"crop_path": str(crop_path), "ground_truth": row.plate_text, **result})

    payload = {"raw": summarize_ocr(raw_pairs), "corrected": summarize_ocr(corrected_pairs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
