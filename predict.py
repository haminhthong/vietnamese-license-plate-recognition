"""Nhận diện biển số trên một ảnh đầu vào."""

import argparse
import json
from pathlib import Path

from src.pipeline import LicensePlateRecognizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.jpg"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    recognizer = LicensePlateRecognizer(args.weights, gpu=False if args.cpu else None)
    results = recognizer.predict_file(args.source, args.output)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Ảnh kết quả: {args.output}")


if __name__ == "__main__":
    main()
