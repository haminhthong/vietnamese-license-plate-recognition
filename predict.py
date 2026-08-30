"""Nhận diện biển số xe trên một tệp ảnh đầu vào."""

import argparse
import json
import logging
from pathlib import Path

from src.pipeline import LicensePlateRecognizer, RecognitionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Thực thi dự đoán biển số xe trên ảnh truyền qua CLI và xuất tệp kết quả."""
    parser = argparse.ArgumentParser(description="Nhận diện biển số xe từ một tệp ảnh.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn tới tệp trọng số YOLOv8 (.pt)")
    parser.add_argument("--source", type=Path, required=True, help="Đường dẫn tệp ảnh đầu vào")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.jpg"), help="Đường dẫn tệp ảnh kết quả minh họa")
    parser.add_argument("--cpu", action="store_true", help="Ép buộc thực thi trên CPU")
    parser.add_argument("--confidence", type=float, default=0.25, help="Ngưỡng độ tin cậy phát hiện biển số")
    args = parser.parse_args()

    config = RecognitionConfig(detection_confidence=args.confidence)
    recognizer = LicensePlateRecognizer(
        args.weights,
        gpu=False if args.cpu else None,
        config=config,
    )
    logger.info("Đang thực hiện nhận diện biển số cho tệp: %s", args.source)
    results = recognizer.predict_file(args.source, args.output)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info("Ảnh kết quả minh họa đã lưu tại: %s", args.output)


if __name__ == "__main__":
    main()
