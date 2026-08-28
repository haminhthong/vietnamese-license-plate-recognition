"""FastAPI phục vụ nhận diện biển số từ ảnh tải lên."""

import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.pipeline import LicensePlateRecognizer

app = FastAPI(title="Vietnamese License Plate Recognition", version="1.0.0")


@lru_cache(maxsize=1)
def get_recognizer() -> LicensePlateRecognizer:
    weights = os.getenv("MODEL_WEIGHTS", "models/best.pt")
    return LicensePlateRecognizer(weights)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ JPEG, PNG hoặc WebP")
    payload = await image.read()
    if len(payload) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 10 MB")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh không hợp lệ")
    recognizer = get_recognizer()
    predictions = recognizer.predict(decoded)
    return {"filename": image.filename, "latency_ms": recognizer.last_latency_ms, "predictions": predictions}
