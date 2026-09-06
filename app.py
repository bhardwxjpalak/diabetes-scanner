import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pipeline import PipelineService

app = FastAPI(
    title="Ocular Diabetes Screening API",
    description="Offline microvascular conjunctival screening for diabetes risk",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Lazy/Singleton initialization of the ML pipeline service
pipeline_service = None

def get_pipeline():
    global pipeline_service
    if pipeline_service is None:
        pipeline_service = PipelineService()
    return pipeline_service

class Base64ScreenRequest(BaseModel):
    image_base64: str

@app.on_event("startup")
async def startup_event():
    print("[*] Pre-warming Pipeline Service...")
    get_pipeline()
    print("[+] Pipeline Service loaded and ready for inference.")

@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    return FileResponse(index_path)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "conjunctival-diabetes-screener",
        "models": {
            "roi_segmentation": "GhostNet-U-Net (ONNX)",
            "vessel_segmentation": "GhostNet-U-Net (ONNX)",
            "classifier": "Tuned Regularized XGBoost (4 Features)"
        }
    }

@app.post("/api/screen")
async def screen_image(file: UploadFile = File(None), payload: Base64ScreenRequest = None):
    """
    Accepts either multipart file upload or JSON payload containing base64 image.
    Executes the 6-stage screening pipeline.
    """
    service = get_pipeline()
    img_bytes = None

    if file is not None:
        img_bytes = await file.read()
    elif payload is not None and payload.image_base64:
        header_and_data = payload.image_base64.split(",")
        b64_str = header_and_data[1] if len(header_and_data) > 1 else header_and_data[0]
        img_bytes = base64.b64decode(b64_str)
    else:
        raise HTTPException(status_code=400, detail="No image provided. Upload a file or send base64 data.")

    try:
        result = service.run_pipeline(img_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quality-check")
async def check_quality_only(file: UploadFile = File(...)):
    service = get_pipeline()
    img_bytes = await file.read()
    import numpy as np, cv2
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding.")
    res = service.check_quality(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return JSONResponse(content=res)
