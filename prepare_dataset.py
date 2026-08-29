"""Kiểm tra dữ liệu YOLO và tạo split không rò rỉ giữa các nhóm nguồn."""

import argparse
from pathlib import Path

from src.dataset import audit_manifest, collect_manifest, find_group_safe_split, materialize_split
from src.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Thư mục chứa train/valid/test")
    parser.add_argument("--output", type=Path, default=Path("dataset/grouped"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-output", type=Path, default=Path("artifacts/dataset_audit.json"))
    args = parser.parse_args()
    manifest = collect_manifest(args.source)
    audit = audit_manifest(manifest)
    write_json(args.audit_output, audit)
    print(args.audit_output.read_text(encoding="utf-8"))
    split = find_group_safe_split(manifest, seed=args.seed)
    yaml_path = materialize_split(split, args.output)
    print(split.groupby("split").agg(images=("image_path", "count"), objects=("n_objects", "sum")))
    print(f"Tệp cấu hình dữ liệu: {yaml_path}")


if __name__ == "__main__":
    main()
