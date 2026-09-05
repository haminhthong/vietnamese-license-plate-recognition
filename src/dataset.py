"""Kiểm tra dữ liệu YOLOv8 và chia tập train/val/test theo nhóm nguồn để tránh rò rỉ dữ liệu (Data Leakage).

Module này hỗ trợ:
1. Trích xuất tên nhóm nguồn (Video Frame Stem) từ tên tệp ảnh.
2. Tính mã băm MD5 để phát hiện ảnh trùng lặp nội dung.
3. Sử dụng cấu trúc dữ liệu Union-Find (Disjoint Set Union) để gộp các nhóm có chung ảnh trùng hash.
4. Thuật toán `GroupShuffleSplit` để phân chia dữ liệu cân bằng theo tỷ lệ 70% Train / 15% Val / 15% Test.
5. Tạo thư mục dữ liệu đích và file `data.yaml` tiêu chuẩn cho Ultralytics YOLOv8.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

# Định nghĩa danh sách các class và định dạng ảnh hỗ trợ
CLASS_NAMES = {0: "license_plate"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def infer_source_group(stem: str) -> str:
    """Suy luận nhóm nguồn gốc từ tên tệp ảnh (ví dụ: 'cam01_001.jpg' -> 'cam01').

    Args:
        stem (str): Tên tệp ảnh không kèm đuôi mở rộng.

    Returns:
        str: Nhận diện tiền tố làm mã nhóm, hoặc tạo nhóm đơn lẻ nếu không có mẫu tiền tố.
    """
    match = re.match(r"^(.+?)[_-](\d+)$", stem)
    return match.group(1).lower() if match else f"standalone::{stem.lower()}"


def compute_phash(path: Path) -> str:
    """Tính mã băm cảm nhận pHash (Perceptual Hash) 64-bit từ ảnh xám để tìm ảnh gần trùng lặp.

    Args:
        path (Path): Đường dẫn tệp ảnh.

    Returns:
        str: Chuỗi 16 ký tự hex đại diện pHash.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ""
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    dct_low = dct[:8, :8]
    med = float(np.median(dct_low))
    bits = (dct_low > med).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def phash_hamming_distance(hash1: str, hash2: str) -> int:
    """Tính khoảng cách Hamming giữa hai chuỗi pHash."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    return bin(val1 ^ val2).count("1")


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Tính mã băm MD5 của tệp để phát hiện ảnh bị trùng lặp nội dung binary.

    Args:
        path (Path): Đường dẫn tới tệp cần tính hash.
        chunk_size (int): Kích thước khối đọc dữ liệu (mặc định: 1 MB).

    Returns:
        str: Chuỗi hex digest đại diện cho mã MD5.
    """
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Phân tích tệp nhãn định dạng YOLO txt (class_id, x_center, y_center, width, height).

    Args:
        label_path (Path): Đường dẫn tệp nhãn txt.

    Returns:
        list[tuple[int, float, float, float, float]]: Danh sách các bounding box hợp lệ.

    Raises:
        ValueError: Nếu định dạng nhãn sai, tọa độ nằm ngoài [0, 1] hoặc class_id không hỗ trợ.
    """
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
    """Thu thập toàn bộ thông tin ảnh, nhãn, mã băm MD5, pHash và nhóm nguồn từ thư mục dữ liệu YOLO.

    Args:
        root (Path): Đường dẫn thư mục dữ liệu gốc chứa train/valid/test.

    Returns:
        pd.DataFrame: Bảng manifest chứa đầy đủ thuộc tính từng ảnh.

    Raises:
        FileNotFoundError: Nếu thiếu các thư mục con bắt buộc.
        RuntimeError: Nếu không tìm thấy cặp ảnh/nhãn hợp lệ.
        ValueError: Nếu phát hiện ảnh thiếu nhãn hoặc ảnh không đọc được.
    """
    import cv2

    required_directories = [
        root / split / subdirectory
        for split in ("train", "valid", "test")
        for subdirectory in ("images", "labels")
    ]
    missing_directories = [path for path in required_directories if not path.is_dir()]
    if missing_directories:
        raise FileNotFoundError(f"Thiếu thư mục dữ liệu bắt buộc: {missing_directories[0]}")

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
                "original_split": original_split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_name": image_path.name,
                "group_id": infer_source_group(image_path.stem),
                "n_objects": len(labels),
                "class_0_count": counts.get(0, 0),
                "md5": file_md5(image_path),
                "phash": compute_phash(image_path),
            })

    if not records:
        raise RuntimeError(f"Không tìm thấy cặp ảnh/nhãn YOLO hợp lệ trong thư mục: {root}")
    if missing_labels:
        preview = ", ".join(str(path) for path in missing_labels[:3])
        raise ValueError(f"Có {len(missing_labels)} ảnh thiếu tệp nhãn. Ví dụ: {preview}")
    if unreadable_images:
        preview = ", ".join(str(path) for path in unreadable_images[:3])
        raise ValueError(f"Có {len(unreadable_images)} ảnh lỗi không đọc được. Ví dụ: {preview}")

    return _merge_groups_connected_by_hash(pd.DataFrame(records))


def _merge_groups_connected_by_hash(frame: pd.DataFrame) -> pd.DataFrame:
    """Sử dụng thuật toán Disjoint-Set Union (DSU) để gộp các nhóm nguồn có chung ảnh trùng MD5 hoặc pHash gần kề.

    Điều này đảm bảo hai ảnh giống hệt hoặc gần trùng không bao giờ bị rơi vào hai tập split khác nhau.
    """
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

    # Gộp theo trùng mã MD5
    for groups in frame.groupby("md5")["group_id"].unique():
        first = groups[0]
        for group_id in groups[1:]:
            union(first, group_id)

    # Gộp theo pHash khoảng cách Hamming <= 4
    phashes = frame[["group_id", "phash"]].drop_duplicates().to_dict(orient="records")
    for index1 in range(len(phashes)):
        for index2 in range(index1 + 1, len(phashes)):
            h1, h2 = phashes[index1]["phash"], phashes[index2]["phash"]
            if h1 and h2 and phash_hamming_distance(h1, h2) <= 4:
                union(phashes[index1]["group_id"], phashes[index2]["group_id"])

    # Protocol B: Gộp theo cùng plate_identity / plate_text nếu có
    plate_col = "plate_identity" if "plate_identity" in frame.columns else ("plate_text" if "plate_text" in frame.columns else None)
    if plate_col:
        for groups in frame.groupby(plate_col)["group_id"].unique():
            if len(groups) > 1:
                first = groups[0]
                for group_id in groups[1:]:
                    union(first, group_id)

    result = frame.copy()
    result["group_id"] = result["group_id"].map(find)
    return result


def audit_manifest(frame: pd.DataFrame) -> dict:
    """Thống kê chi tiết số lượng ảnh, đối tượng, nhóm nguồn và kiểm tra rò rỉ dữ liệu."""
    crossing_groups = frame.groupby("group_id")["original_split"].nunique()
    crossing_hashes = frame.groupby("md5")["original_split"].nunique()

    plate_col = "plate_identity" if "plate_identity" in frame.columns else ("plate_text" if "plate_text" in frame.columns else None)
    crossing_plates = frame.groupby(plate_col)["original_split"].nunique() if plate_col else None

    # Tính số cặp near-duplicates
    near_dup_pairs = 0
    if "phash" in frame.columns:
        phashes = frame["phash"].tolist()
        for idx1 in range(len(phashes)):
            for idx2 in range(idx1 + 1, len(phashes)):
                if phashes[idx1] and phashes[idx2] and phash_hamming_distance(phashes[idx1], phashes[idx2]) <= 4:
                    near_dup_pairs += 1

    exact_duplicates = int(frame["md5"].duplicated().sum())
    split_counts = frame.get("split", frame["original_split"]).value_counts().to_dict()

    return {
        "images": len(frame),
        "objects": int(frame["n_objects"].sum()),
        "groups": int(frame["group_id"].nunique()),
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": near_dup_pairs,
        "groups_crossing_splits": int((crossing_groups > 1).sum()),
        "duplicate_hashes_crossing_splits": int((crossing_hashes > 1).sum()),
        "plates_crossing_splits": int((crossing_plates > 1).sum()) if crossing_plates is not None else 0,
        "train_images": int(split_counts.get("train", 0)),
        "validation_images": int(split_counts.get("val", split_counts.get("valid", 0))),
        "test_images": int(split_counts.get("test", 0)),
    }



def _score(frame: pd.DataFrame) -> float:
    """Hàm đánh giá độ lệch giữa phân bố ngẫu nhiên và tỷ lệ mục tiêu 70/15/15."""
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
    """Tìm cách chia Group-Safe Split tối ưu nhất thông qua phương pháp thử nghiệm nhiều hạt giống (trials).

    Args:
        frame (pd.DataFrame): Manifest dữ liệu đã gộp nhóm.
        seed (int): Hạt giống ngẫu nhiên khởi tạo.
        trials (int): Số lần thử nghiệm tìm phương án tối ưu.

    Returns:
        pd.DataFrame: Bảng manifest chứa cột 'split' mới (train, val, test).
    """
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
        raise RuntimeError("Không thể tạo split theo nhóm mà vẫn bảo đảm đủ dữ liệu cho mỗi tập.")
    assert best.groupby("group_id")["split"].nunique().max() == 1
    return best


def materialize_split(frame: pd.DataFrame, destination: Path) -> Path:
    """Sao chép tệp ảnh và tệp nhãn vào thư mục phân chia mới và tạo tệp cấu hình data.yaml.

    Args:
        frame (pd.DataFrame): Manifest đã gán phân chia 'split'.
        destination (Path): Thư mục đích lưu dataset mới.

    Returns:
        Path: Đường dẫn tới tệp data.yaml vừa được khởi tạo.
    """
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
        "path": str(destination.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }, sort_keys=False), encoding="utf-8")
    return yaml_path
