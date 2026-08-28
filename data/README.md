# Data card

Repository không phân phối ảnh hoặc biển số riêng tư. Dữ liệu đầu vào phải ở định dạng YOLO với `train`, `valid`, `test`, mỗi tập có thư mục `images` và `labels`.

Mỗi dòng nhãn gồm `class_id x_center y_center width height`; project hiện chỉ hỗ trợ class `0: license_plate`. Chạy `prepare_dataset.py` để kiểm tra schema, ảnh lỗi, nhãn thiếu, duplicate MD5 và tạo split cố định theo nhóm nguồn. `split_manifest.csv` là artifact cần lưu cùng mỗi thí nghiệm.

Trước khi công bố dữ liệu, phải ghi rõ nguồn, phiên bản, giấy phép, phạm vi đồng ý sử dụng và chính sách ẩn danh. Không đưa ảnh biển số thật lên repository nếu chưa có quyền.
