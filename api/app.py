import os
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    # When running as a package (e.g. `uvicorn api.app:app`)
    from api.tmh_infer import TMHInferencer
except ModuleNotFoundError:
    # When running from within /api (e.g. `uvicorn app:app`)
    from tmh_infer import TMHInferencer


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_A_PATH = os.path.join(ROOT_DIR, "model_a_fixed_iris.pth")
MODEL_B_PATH = os.path.join(ROOT_DIR, "model_b_meniscus_new.pth")


app = FastAPI(title="TMH API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_inferencer: TMHInferencer | None = None
_inferencer_lock = threading.Lock()


def get_inferencer() -> TMHInferencer:
    global _inferencer
    if _inferencer is not None:
        return _inferencer
    with _inferencer_lock:
        if _inferencer is None:
            _inferencer = TMHInferencer(
                model_a_path=MODEL_A_PATH,
                model_b_path=MODEL_B_PATH,
            )
    return _inferencer


@app.get("/health")
def health():
    return {"ok": True, "models_loaded": _inferencer is not None}


@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    result = get_inferencer().infer_from_bgr(img_bgr)
    return {
        "tmh_mm": result.tmh_mm,
        "diagnosis": result.diagnosis,
        "iris_diam_px": result.iris_diam_px,
        "tmh_px_median": result.tmh_px_median,
    }

