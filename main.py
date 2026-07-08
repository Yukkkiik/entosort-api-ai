import base64
import io
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel
import numpy as np

# Load file .env
load_dotenv()

app = FastAPI(title="EntoSort Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = {0: "larva_active", 1: "prepupa_pupa"}

# ---------- Ambil Konfigurasi dari .env ----------
IMG_SIZE = int(os.getenv("IMG_SIZE", 640))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.25))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "models/model_int8.onnx")

# ---------- Lazy-loaded model ----------
_onnx_session = None


def get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # Batasi thread — CPU Railway Trial/Free terbatas, thread berlebih
        # malah bikin overhead/contention, bukan mempercepat
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        _onnx_session = ort.InferenceSession(
            ONNX_MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _onnx_session


# ---------- Request / response schema ----------
class PredictRequest(BaseModel):
    image_base64: str
    conf: float = CONF_THRESHOLD


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class PredictResponse(BaseModel):
    detections: list[Detection]
    inference_time_ms: float
    image_width: int
    image_height: int


# ---------- Helpers ----------
def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        if "," in image_base64 and image_base64.strip().startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 gambar tidak valid: {e}")


def letterbox(image: Image.Image, size: int = IMG_SIZE):
    """Resize dengan padding, kembalikan array + scale + pad buat konversi bbox balik."""
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = image.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def xywh_to_xyxy_original(cx, cy, w, h, scale, pad_x, pad_y):
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale
    return [x1, y1, x2, y2]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """Simple NMS, tanpa dependency tambahan."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess_onnx_output(output: np.ndarray, scale, pad_x, pad_y, conf_threshold: float):
    # output shape: [1, 6, 8400] -> transpose ke [8400, 6]
    preds = output[0].T  # [8400, 4 + num_classes]
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    boxes_xyxy = np.array(
        [
            xywh_to_xyxy_original(cx, cy, w, h, scale, pad_x, pad_y)
            for cx, cy, w, h in boxes_xywh
        ]
    )

    if len(boxes_xyxy) == 0:
        return []

    keep = nms(boxes_xyxy, confidences, IOU_THRESHOLD)

    detections = []
    for idx in keep:
        detections.append(
            Detection(
                class_id=int(class_ids[idx]),
                class_name=CLASS_NAMES.get(int(class_ids[idx]), str(class_ids[idx])),
                confidence=float(confidences[idx]),
                bbox=[float(round(v, 2)) for v in boxes_xyxy[idx]],
            )
        )
    return detections


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "EntoSort Inference API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/onnx", response_model=PredictResponse)
def predict_onnx(req: PredictRequest):
    image = decode_base64_image(req.image_base64)
    w, h = image.size

    canvas, scale, pad_x, pad_y = letterbox(image, IMG_SIZE)
    img_array = np.array(canvas).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, :]  # NCHW

    session = get_onnx_session()
    input_name = session.get_inputs()[0].name

    start = time.time()
    output = session.run(None, {input_name: img_array})[0]
    elapsed_ms = (time.time() - start) * 1000

    detections = postprocess_onnx_output(output, scale, pad_x, pad_y, req.conf)

    return PredictResponse(
        detections=detections,
        inference_time_ms=round(elapsed_ms, 2),
        image_width=w,
        image_height=h,
    )