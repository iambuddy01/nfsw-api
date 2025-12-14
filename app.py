from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from model import scan_image

app = FastAPI(title="NSFW Detection API")

@app.get("/")
def root():
    return {"status": "NSFW API running"}

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    primary, scores, safe = scan_image(img)

    return {
        "primary": primary,
        "safe": safe,
        "scores": scores
    }

@app.post("/batch-scan")
async def batch_scan(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        primary, scores, safe = scan_image(img)

        results.append({
            "filename": file.filename,
            "primary": primary,
            "safe": safe,
            "scores": scores
        })

    return results
