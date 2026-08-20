# Nhận diện biển số xe Việt Nam — YOLOv8 + EasyOCR

Pipeline Computer Vision nhận diện biển số xe Việt Nam từ ảnh: **YOLOv8** định vị biển số, **EasyOCR** đọc ký tự, sau đó hậu xử lý theo bố cục một/hai dòng và mẫu định dạng biển số. Project được tái cấu trúc từ notebook nghiên cứu thành source code có thể kiểm thử và chạy lại, phù hợp để trình bày trong CV/portfolio.

> Kết quả được báo cáo trên test split tách theo nguồn video. Không sử dụng kết quả validation cũ có rò rỉ frame. OCR và end-to-end chưa được công bố metric vì bộ dữ liệu hiện chỉ có bounding box, chưa có nhãn ký tự được kiểm chứng thủ công.

## Kết quả nổi bật

| Hạng mục | Kết quả |
|---|---:|
| Dữ liệu gốc | 498 ảnh, 780 bounding box |
| Số nhóm nguồn | 50 |
| Group-safe split | 351 train / 72 validation / 75 test |
| Test Precision | **0.9830** |
| Test Recall | **0.9657** |
| Test mAP@0.50 | **0.9730** |
| Test mAP@0.50:0.95 | **0.8693** |

Các metric detector trên được lấy từ lần chạy `yolov8n_grouped_20260819_110337`, với 120 instances trong 75 ảnh test. Đây là kết quả của detector một lớp `license_plate`, không phải độ chính xác đọc toàn bộ chuỗi ký tự.

## Tại sao project này đáng chú ý?

- Phát hiện và loại bỏ **data leakage**: split gốc có hai nhóm video xuất hiện ở nhiều tập và một ảnh trùng hash giữa các tập.
- Tách lại dữ liệu theo `source_group`, giúp các frame liên tiếp của cùng video/xe chỉ nằm trong một split.
- Phân biệt rõ ba bài toán: detection, OCR-only và end-to-end.
- OCR hỗ trợ biển một dòng/hai dòng, nhiều biến thể tiền xử lý và hiệu chỉnh ký tự có ràng buộc định dạng.
- Không công bố metric OCR giả khi chưa có transcription ground truth.
- Source code được module hóa, có CLI, cấu hình, type hint và unit test.

## Kiến trúc pipeline

```text
Ảnh đầu vào
    │
    ▼
YOLOv8n detector ──► bounding box biển số
    │
    ▼
Crop + padding ──► suy luận bố cục 1 dòng / 2 dòng
    │
    ▼
Gray / CLAHE / Otsu / Adaptive Threshold
    │
    ▼
EasyOCR ──► sắp xếp token theo hình học
    │
    ▼
Chuẩn hóa + kiểm tra mẫu biển số ──► kết quả và confidence
```

YOLO chỉ học một lớp `license_plate`. Bố cục một dòng/hai dòng không phải detector class; nó được suy luận từ tỷ lệ crop để sắp xếp token OCR đúng thứ tự.

## Cấu trúc thư mục

```text
.
├── src/
│   ├── dataset.py        # audit, phát hiện leakage, group-safe split
│   ├── metrics.py        # IoU, Levenshtein/CER helper
│   ├── ocr.py            # preprocessing, layout, OCR post-processing
│   └── pipeline.py       # detector + OCR end-to-end
├── tests/test_core.py
├── configs/data.example.yaml
├── prepare_dataset.py
├── train.py
├── evaluate_detector.py
├── predict.py
├── requirements.txt
└── README.md
```

## Cài đặt

Yêu cầu Python 3.10+; nên dùng GPU NVIDIA/CUDA khi train.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

EasyOCR sẽ tải model ở lần chạy đầu tiên. Ultralytics cũng tải pretrained `yolov8n.pt` nếu file chưa có trên máy.

## Chuẩn bị dữ liệu

Dataset dùng định dạng YOLO:

```text
dataset_source/
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

Mỗi label có dạng `class_id x_center y_center width height`, tọa độ được chuẩn hóa về `[0, 1]`. Toàn bộ annotation hiện dùng class `0` (`license_plate`).

Tạo lại split không rò rỉ theo chuỗi frame:

```bash
python prepare_dataset.py --source path/to/dataset_source --output dataset/grouped
```

Script sẽ:

1. Kiểm tra ảnh, label, class ID và khoảng tọa độ.
2. Tính MD5 để phát hiện ảnh trùng giữa các split.
3. Gom các file như `clip13_new_24`, `clip13_new_25` về nhóm `clip13_new`.
4. Tìm split gần tỷ lệ 70/15/15 nhưng không để cùng nhóm xuất hiện ở nhiều tập.
5. Ghi `split_manifest.csv` và `data.yaml` để tái lập thí nghiệm.

Đích output phải chưa tồn tại để tránh vô tình ghi đè dữ liệu.

## Huấn luyện detector

```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 60 --batch 16
```

Cấu hình chính:

- input `640 × 640`, seed `42`, deterministic mode;
- early stopping patience `15`;
- pretrained YOLOv8n;
- tắt horizontal/vertical flip vì chữ bị lật không phản ánh tình huống thực tế;
- augmentation nhẹ: rotation, translation, scale và perspective;
- checkpoint tốt nhất nằm tại `runs/<run-name>/weights/best.pt`.

Đánh giá checkpoint trên test split và lưu metric dạng JSON:

```bash
python evaluate_detector.py --weights runs/<run-name>/weights/best.pt --data data.yaml
```

## Chạy nhận diện

```bash
python predict.py \
  --weights runs/<run-name>/weights/best.pt \
  --source path/to/image.jpg \
  --output outputs/prediction.jpg
```

Thêm `--cpu` nếu muốn buộc chạy bằng CPU. Chương trình xuất ảnh đã vẽ bounding box và JSON gồm:

- tọa độ box;
- confidence của detector;
- raw OCR và text sau hiệu chỉnh;
- confidence OCR;
- bố cục một/hai dòng;
- biến thể preprocessing được chọn;
- trạng thái hợp lệ theo template.

## Đánh giá đúng cách

### 1. Detection

Đánh giá trên test split độc lập bằng Precision, Recall, mAP@0.50 và mAP@0.50:0.95. Không dùng test để chọn threshold hoặc hyperparameter.

### 2. OCR-only

Cắt biển bằng **ground-truth bounding box**, gán thủ công `plate_text`, rồi báo cáo:

- Exact Match Accuracy: toàn bộ chuỗi phải đúng;
- Character Error Rate (CER): tổng edit distance chia tổng số ký tự;
- kết quả riêng theo bố cục một dòng/hai dòng.

### 3. End-to-end

Ghép prediction với ground truth tại IoU ≥ 0.5. Biển bị detector bỏ sót nhận prediction rỗng và phải tính là lỗi. Cách này tránh báo cáo OCR tốt trong khi bỏ qua lỗi detection.

Hiện repository chưa có transcription ground truth nên chưa báo cáo hai nhóm metric OCR. Đây là công việc tiếp theo cần hoàn thành trước khi khẳng định độ chính xác nhận diện toàn chuỗi.

## Kiểm thử

```bash
pytest -q
```

Test hiện bao phủ hiệu chỉnh ký tự theo template, suy luận layout, IoU và Levenshtein distance.

## Hạn chế và hướng phát triển

- Dữ liệu nhỏ (498 ảnh) và phần lớn đến từ video, chưa đại diện đầy đủ tỉnh/thành, điều kiện đêm, mưa, rung/mờ hoặc góc nhìn lớn.
- EasyOCR tổng quát chưa tối ưu riêng cho font biển số Việt Nam.
- Template hiện tập trung vào biển dân sự phổ biến; cần mở rộng cho biển ngoại giao, quân đội, tạm thời và các loại đặc biệt.
- Layout được suy luận bằng aspect ratio nên có thể sai khi bounding box quá rộng/hẹp.
- Chưa có OCR ground truth đã kiểm chứng và chưa đo latency trên thiết bị triển khai thực tế.

Hướng nâng cấp: gán nhãn transcription, huấn luyện recognizer chuyên biệt (CRNN/Transformer), rectification bằng perspective transform, hard-negative mining, tracking cho video và export ONNX/TensorRT.

## Quyền riêng tư và dữ liệu

Biển số có thể là dữ liệu nhạy cảm. Không đưa ZIP gốc, ảnh biển số riêng tư, annotation OCR hoặc model weight vào repository công khai nếu chưa có quyền sử dụng. Hãy xác minh nguồn và giấy phép của dataset trước khi phát hành. `.gitignore` đã chặn dataset, weights, outputs và ảnh theo mặc định.

## Gợi ý mô tả trong CV

**Tiếng Việt**

> Xây dựng pipeline nhận diện biển số xe Việt Nam với YOLOv8 và EasyOCR trên 498 ảnh/780 annotation; phát hiện và khắc phục rò rỉ dữ liệu giữa các frame bằng group-safe split, đạt 0.973 mAP@0.50 và 0.869 mAP@0.50:0.95 trên test set độc lập.

**English**

> Built a Vietnamese license-plate recognition pipeline with YOLOv8 and EasyOCR on 498 images/780 annotations; eliminated video-frame leakage using group-aware splitting and achieved 0.973 mAP@50 and 0.869 mAP@50–95 on an independent test set.

## Nguồn gốc

Source code được tái cấu trúc từ notebook `Nhan_Dien_Bien_So_Xe_CV.ipynb`. Notebook vẫn hữu ích cho EDA và theo dõi thí nghiệm; các module trong `src/` là phiên bản dùng để tái sử dụng, kiểm thử và trình bày portfolio.
