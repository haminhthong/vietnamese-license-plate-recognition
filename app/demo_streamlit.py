"""Ứng dụng Streamlit Web Demo trực quan cho hệ thống Nhận diện Biển số xe Việt Nam.

Chạy ứng dụng bằng lệnh:
    streamlit run app/demo_streamlit.py
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.pipeline import LicensePlateRecognizer, RecognitionConfig, draw_predictions

st.set_page_config(
    page_title="Vietnamese License Plate Recognition",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Vietnamese License Plate Recognition")
st.markdown("Hệ thống nhận diện biển số xe Việt Nam End-to-End dựa trên **YOLOv8**, **OpenCV** và **EasyOCR**.")

# Sidebar cấu hình
st.sidebar.header("⚙️ Cấu Hình Pipeline")
model_path_str = st.sidebar.text_input("Đường dẫn Trọng số YOLOv8 (.pt)", value=os.getenv("MODEL_WEIGHTS", "models/best.pt"))
conf_threshold = st.sidebar.slider("Ngưỡng độ tin cậy (Confidence Threshold)", 0.10, 0.90, 0.25, 0.05)
use_cpu = st.sidebar.checkbox("Ép buộc chạy trên CPU", value=False)

weights_path = Path(model_path_str)


@st.cache_resource
def load_recognizer(path: str, cpu: bool) -> LicensePlateRecognizer:
    return LicensePlateRecognizer(path, gpu=not cpu)


if not weights_path.is_file():
    st.warning(f"⚠️ Chưa tìm thấy tệp trọng số mô hình tại đường dẫn `{weights_path}`. Vui lòng cung cấp tệp `best.pt` hợp lệ để chạy nhận diện.")
else:
    try:
        recognizer = load_recognizer(str(weights_path), use_cpu)
        recognizer.config = RecognitionConfig(detection_confidence=conf_threshold)
        st.sidebar.success("✅ Đã nạp thành công mô hình YOLOv8 & EasyOCR!")
    except Exception as e:
        st.sidebar.error(f"Lỗi nạp mô hình: {e}")
        recognizer = None

    uploaded_file = st.file_uploader("Tải lên ảnh xe hoặc ảnh biển số (JPEG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None and recognizer is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Ảnh Gốc Tải Lên")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.spinner("Đang thực hiện phân tích và đọc biển số..."):
                predictions = recognizer.predict(image_bgr)
                annotated_bgr = draw_predictions(image_bgr, predictions)

            with col2:
                st.subheader("🔍 Kết Quả Trực Quan")
                st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("---")
            st.subheader(f"📊 Chi Tiết Nhận Diện ({len(predictions)} biển số)")
            st.caption(f"⏱️ Tổng thời gian xử lý: **{recognizer.last_latency_ms:.1f} ms**")

            if predictions:
                for idx, pred in enumerate(predictions, 1):
                    with st.expander(f"Biển số #{idx}: {pred['text'] or pred['raw_text'] or 'N/A'}", expanded=True):
                        p_col1, p_col2, p_col3 = st.columns(3)
                        p_col1.metric("Biển số hiệu chỉnh", pred["text"] or "N/A")
                        p_col2.metric("Chuỗi OCR thô", pred["raw_text"] or "N/A")
                        p_col3.metric("Khớp mẫu chuẩn", "Có" if pred["format_valid"] else "Không")

                        p_col4, p_col5, p_col6 = st.columns(3)
                        p_col4.metric("Độ tin cậy YOLO", f"{pred['detection_confidence']*100:.1f}%")
                        p_col5.metric("Độ tin cậy OCR", f"{pred['ocr_confidence']*100:.1f}%")
                        p_col6.metric("Bố cục suy luận", "1 Dòng" if pred["layout"] == "1_line" else "2 Dòng")
            else:
                st.info("Không phát hiện được biển số xe nào thỏa mãn ngưỡng độ tin cậy.")
