"""Kiểm tra dữ liệu YOLO và chia tập theo nhóm nguồn để tránh rò rỉ dữ liệu."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

CLASS_NAMES = {0: "license_plate"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def infer_source_group(stem: str) -> str:
    match = re.match(r"^(.+?)[_-](\d+)$", stem)
    return match.group(1).lower() if match else f"standalone::{stem.lower()}"


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Nhãn YOLO không hợp lệ tại {label_path}:{line_number}")
        class_value = float(parts[0])
        if not class_value.is_integer():
            raise ValueError(f"Class ID phải là số nguyên tại {label_path}:{line_number}")
        class_id = int(class_value)
        coordinates = tuple(map(float, parts[1:]))
        _, _, width, height = coordinates
        if (
            class_id not in CLASS_NAMES
            or not all(0 <= value <= 1 for value in coordinates)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(f"Annotation không hợp lệ tại {label_path}:{line_number}")
        rows.append((class_id, *coordinates))
    return rows


def collect_manifest(root: Path) -> pd.DataFrame:
    import cv2

    required_directories = [
        root / split / subdirectory
        for split in ("train", "valid", "test")
        for subdirectory in ("images", "labels")
    ]
    missing_directories = [path for path in required_directories if not path.is_dir()]
    if missing_directories:
        raise FileNotFoundError(f"Thiếu thư mục dữ liệu: {missing_directories[0]}")

    records = []
    missing_labels: list[Path] = []
    unreadable_images: list[Path] = []
    for original_split in ("train", "valid", "test"):
        image_dir, label_dir = root / original_split / "images", root / original_split / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path)
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                unreadable_images.append(image_path)
                continue
            labels = parse_label_file(label_path)
            counts = Counter(item[0] for item in labels)
            records.append({
                "original_split": original_split, "image_path": str(image_path),
                "label_path": str(label_path), "image_name": image_path.name,
                "group_id": infer_source_group(image_path.stem), "n_objects": len(labels),
                "class_0_count": counts.get(0, 0), "md5": file_md5(image_path),
            })
    if not records:
        raise RuntimeError(f"Không tìm thấy cặp ảnh/nhãn YOLO hợp lệ trong: {root}")
    if missing_labels:
        preview = ", ".join(str(path) for path in missing_labels[:3])
        raise ValueError(f"Có {len(missing_labels)} ảnh thiếu nhãn. Ví dụ: {preview}")
    if unreadable_images:
        preview = ", ".join(str(path) for path in unreadable_images[:3])
        raise ValueError(f"Có {len(unreadable_images)} ảnh không đọc được. Ví dụ: {preview}")
    return _merge_groups_connected_by_hash(pd.DataFrame(records))


def _merge_groups_connected_by_hash(frame: pd.DataFrame) -> pd.DataFrame:
    """Gộp các nhóm nguồn có chung ảnh để ảnh trùng không nằm ở hai split."""
    parent = {group_id: group_id for group_id in frame["group_id"].unique()}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for groups in frame.groupby("md5")["group_id"].unique():
        first = groups[0]
        for group_id in groups[1:]:
            union(first, group_id)

    result = frame.copy()
    result["group_id"] = result["group_id"].map(find)
    return result


def audit_manifest(frame: pd.DataFrame) -> dict:
    crossing_groups = frame.groupby("group_id")["original_split"].nunique()
    crossing_hashes = frame.groupby("md5")["original_split"].nunique()
    return {
        "images": len(frame), "objects": int(frame["n_objects"].sum()),
        "groups": int(frame["group_id"].nunique()),
        "groups_crossing_splits": int((crossing_groups > 1).sum()),
        "duplicate_hashes_crossing_splits": int((crossing_hashes > 1).sum()),
    }


def _score(frame: pd.DataFrame) -> float:
    total_images = len(frame)
    total_objects = frame["class_0_count"].sum()
    score = 0.0
    for split, target in TARGET_RATIOS.items():
        subset = frame[frame["split"] == split]
        score += 8.0 * abs(len(subset) / total_images - target)
        count = subset["class_0_count"].sum()
        score += 1_000.0 if count == 0 else 2.0 * abs(count / total_objects - target)
    return score


def find_group_safe_split(frame: pd.DataFrame, seed: int = 42, trials: int = 1000) -> pd.DataFrame:
    best, best_score = None, float("inf")
    for trial in range(trials):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + trial)
        _, temp_idx = next(outer.split(frame, groups=frame["group_id"]))
        temp = frame.iloc[temp_idx]
        if temp["group_id"].nunique() < 2:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 10_000 + trial)
        val_rel, test_rel = next(inner.split(temp, groups=temp["group_id"]))
        candidate = frame.copy()
        candidate["split"] = "train"
        candidate.loc[temp.index[val_rel], "split"] = "val"
        candidate.loc[temp.index[test_rel], "split"] = "test"
        candidate_score = _score(candidate)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score
    if best is None or best_score >= 1_000:
        raise RuntimeError("Không thể tạo split theo nhóm mà vẫn có đủ class trong mỗi tập")
    assert best.groupby("group_id")["split"].nunique().max() == 1
    return best


def materialize_split(frame: pd.DataFrame, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"Thư mục đích đã tồn tại: {destination}")
    duplicate_names = set(frame.loc[frame["image_name"].duplicated(keep=False), "image_name"])
    destination_names = []
    for row in frame.itertuples():
        source_image, source_label = Path(row.image_path), Path(row.label_path)
        name = f"{row.original_split}__{source_image.name}" if row.image_name in duplicate_names else source_image.name
        image_dir, label_dir = destination / row.split / "images", destination / row.split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_dir / name)
        shutil.copy2(source_label, label_dir / f"{Path(name).stem}.txt")
        destination_names.append(name)
    output = frame.copy()
    output["destination_name"] = destination_names
    output.to_csv(destination / "split_manifest.csv", index=False)
    yaml_path = destination.parent / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(destination.resolve()), "train": "train/images", "val": "val/images",
        "test": "test/images", "names": CLASS_NAMES,
    }, sort_keys=False), encoding="utf-8")
    return yaml_path
