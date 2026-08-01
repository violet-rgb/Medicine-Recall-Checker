import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ocr import extract_text
from llm_service import analyze_medicine
from vector_store import init_vector_store

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Medicine Backside Checker")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.on_event("startup")
def startup():
    init_vector_store()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analyze")
def analyze_get():
    """Browsers land here via the address bar / refresh after a form POST;
    send them back to the upload form instead of a 405."""
    return RedirectResponse(url="/", status_code=303)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(None),
    manual_text: str = Form(default=""),
    manual_batch: str = Form(default=""),
):
    ocr_result = {"raw_text": "", "lines": [], "engine": "none"}
    uploaded_image_received = False
    
    if image and image.filename:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(400, "Unsupported file type. Upload JPEG, PNG, or WEBP.")
        image_bytes = await image.read()
        if image_bytes:
            uploaded_image_received = True
            ocr_result = extract_text(image_bytes, image.content_type)

    manual_text = manual_text.strip()
    manual_batch = manual_batch.strip()

    if manual_text:
        ocr_result["raw_text"] = f"{manual_text}\n{ocr_result['raw_text']}".strip()
        ocr_result["lines"] = [(manual_text, 1.0)] + ocr_result["lines"]

    print(f"[analyze] OCR result: {ocr_result['raw_text']}... Manual Batch: {manual_batch}")

    if not uploaded_image_received and not ocr_result["raw_text"].strip() and not manual_batch:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": "No input provided. Please upload an image or enter details manually.",
            },
        )

    analysis = analyze_medicine(
        raw_ocr_text=ocr_result["raw_text"],
        manual_batch=manual_batch,
        manual_med=manual_text
    )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "ocr_text": ocr_result["raw_text"],
            "analysis": analysis,
        },
    )


@app.post("/api/analyze")
async def analyze_api(
    image: UploadFile = File(None),
    manual_text: str = Form(default=""),
    manual_batch: str = Form(default=""),
):
    """JSON API equivalent of /analyze, for programmatic / non-template use."""
    ocr_result = {"raw_text": "", "lines": [], "engine": "none"}
    uploaded_image_received = False
    
    if image and image.filename:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(400, "Unsupported file type. Upload JPEG, PNG, or WEBP.")
        image_bytes = await image.read()
        if image_bytes:
            uploaded_image_received = True
            ocr_result = extract_text(image_bytes, image.content_type)

    manual_text = manual_text.strip()
    manual_batch = manual_batch.strip()

    if manual_text:
        ocr_result["raw_text"] = f"{manual_text}\n{ocr_result['raw_text']}".strip()

    if not uploaded_image_received and not ocr_result["raw_text"].strip() and not manual_batch:
        return JSONResponse({"error": "No input provided."}, status_code=422)

    analysis = analyze_medicine(
        raw_ocr_text=ocr_result["raw_text"],
        manual_batch=manual_batch,
        manual_med=manual_text
    )
    return {"ocr_text": ocr_result["raw_text"], "analysis": analysis}
