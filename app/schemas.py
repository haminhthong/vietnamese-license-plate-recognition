"""Định nghĩa Pydantic Schemas đầu ra cho FastAPI REST API của hệ thống nhận diện biển số xe.

Module này cung cấp các mô hình dữ liệu (Response Models):
- `PlatePrediction`: Thông tin chi tiết cho một biển số được phát hiện.
- `PredictionResponse`: Kết quả tổng hợp trả về cho một ảnh upload.
- `HealthResponse`: Trạng thái hoạt động và tình trạng model weight của dịch vụ.
"""

from pydantic import BaseModel, Field


class PlatePrediction(BaseModel):
    """Thông tin kết quả dự đoán của một biển số xe được phát hiện."""

    box: tuple[int, int, int, int] = Field(description="Tọa độ Bounding Box dạng (x1, y1, x2, y2)")
    padded_box: tuple[int, int, int, int] = Field(description="Tọa độ Bounding Box sau khi mở rộng đệm")
    class_id: int = Field(description="Mã lớp định danh đối tượng (0: license_plate)")
    detector_class: str = Field(description="Tên lớp định danh đối tượng")
    detection_confidence: float = Field(ge=0, le=1, description="Độ tin cậy phát hiện của YOLOv8")
    raw_text: str = Field(description="Chuỗi ký tự nhận dạng thô từ OCR")
    text: str = Field(description="Chuỗi ký tự biển số đã qua hậu xử lý hiệu chỉnh")
    format_valid: bool = Field(description="Cờ báo chuỗi có khớp mẫu định dạng biển số Việt Nam hay không")
    template: str | None = Field(default=None, description="Mẫu định dạng đã khớp (ví dụ: 'DDLDDDDD')")
    correction_cost: float = Field(ge=0, description="Chi phí sửa lỗi ký tự (số lần thay thế)")
    ocr_confidence: float = Field(ge=0, le=1, description="Độ tin cậy nhận dạng trung bình của EasyOCR")
    layout: str = Field(description="Bố cục biển số suy luận ('1_line' hoặc '2_line')")
    rectified: bool = Field(description="Cờ báo ảnh crop đã được nắn góc phối cảnh thành công hay chưa")
    variant: str | None = Field(default=None, description="Tên biến thể tiền xử lý ảnh đạt điểm cao nhất (clahe, otsu, v.v.)")
    score: float = Field(description="Điểm số tổng hợp chọn kết quả tối ưu")
    pipeline_latency_ms: float = Field(ge=0, description="Thời gian thực thi pipeline trên ảnh (tính bằng miligiây)")


class PredictionResponse(BaseModel):
    """Phản hồi JSON đầy đủ cho yêu cầu nhận diện ảnh."""

    filename: str | None = Field(default=None, description="Tên tệp ảnh đã tải lên")
    latency_ms: float = Field(ge=0, description="Tổng thời gian xử lý ảnh (ms)")
    predictions: list[PlatePrediction] = Field(description="Danh sách các biển số nhận dạng được")


class HealthResponse(BaseModel):
    """Trạng thái sức khỏe và cấu hình mô hình của dịch vụ API."""

    status: str = Field(description="Trạng thái dịch vụ ('ok' hoặc 'model_missing')")
    model_weights: str = Field(description="Đường dẫn tệp trọng số mô hình")
    model_available: bool = Field(description="Cờ báo tệp trọng số mô hình có sẵn sàng hay không")
