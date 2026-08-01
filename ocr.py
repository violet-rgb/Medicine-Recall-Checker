"""
OCR pipeline for reading text off the back of medicine strips/boxes.

Problem profile:
- Small font size
- Black text on variable / low-contrast colored backgrounds
- Foil glare, curvature, noise

Strategy:
1. Aggressive OpenCV preprocessing (upscale, denoise, illumination correction, adaptive threshold).
2. Run PaddleOCR (primary — strong on small/low-res text) on the preprocessed image.
3. Run Tesseract as a secondary pass on a differently-thresholded variant.
4. Merge results, dedupe, keep highest-confidence line per text ID, return combined raw text.
"""

import importlib.util
import shutil

import cv2
import numpy as np
import pytesseract
from decouple import config
from google import genai
from google.genai import types
from PIL import Image

OCR_ENGINE = config("OCR_ENGINE", default="").lower()
TESSERACT_CMD = config("TESSERACT_CMD", default="")
GEMINI_API_KEY = config("GEMINI_API_KEY", default="")
GEMINI_MODEL = config("GEMINI_MODEL", default="gemini-2.5-flash")
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in config(
        "GEMINI_FALLBACK_MODELS",
        default="gemini-2.5-flash,gemini-2.5-flash-lite",
    ).split(",")
    if model.strip()
]

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Pick the best available engine unless the user forced one via OCR_ENGINE.
# PaddleOCR is far stronger on foil strips / small rotated text, so prefer it
# whenever it is installed; fall back to tesseract otherwise.
_HAS_PADDLE = importlib.util.find_spec("paddleocr") is not None
_HAS_TESSERACT = bool(TESSERACT_CMD) or shutil.which("tesseract") is not None

if not OCR_ENGINE:
    OCR_ENGINE = "paddle" if _HAS_PADDLE else "tesseract"
elif OCR_ENGINE in {"paddleocr", "paddle"} and not _HAS_PADDLE:
    print("[ocr] paddleocr not installed; falling back to tesseract.")
    OCR_ENGINE = "tesseract"
elif OCR_ENGINE == "tesseract" and not _HAS_TESSERACT:
    print("[ocr] tesseract binary not found; falling back to PaddleOCR."
          " Set TESSERACT_CMD to silence this message.")
    OCR_ENGINE = "paddle"

_paddle_ocr_instance = None
_gemini_client = None


def _get_paddle_ocr():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        from paddleocr import PaddleOCR
        _paddle_ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            det_db_thresh=0.3,       # more sensitive text detection for tiny fonts
            det_db_box_thresh=0.4,
            rec_algorithm="SVTR_LCNet",
        )
    return _paddle_ocr_instance


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _gemini_models_to_try() -> list[str]:
    models = []
    for model in [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]:
        if model and model not in models:
            models.append(model)
    return models


def _generate_with_model_fallback(contents, config):
    last_error = None
    for model in _gemini_models_to_try():
        try:
            return _get_gemini_client().models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            print(f"[ocr] Gemini OCR model {model} failed: {e}")
            if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                break
    raise last_error


def _upscale(img: np.ndarray, target_min_dim: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_min_dim / min(h, w)
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def _correct_illumination(gray: np.ndarray) -> np.ndarray:
    """Removes uneven lighting / background color variation using a large-kernel
    background estimate, then normalizes contrast (CLAHE)."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    diff = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(diff)


def _denoise_sharpen(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(denoised, -1, kernel)


def preprocess_image(image_bytes: bytes) -> dict:
    """Returns a dict of preprocessed variants (numpy arrays, BGR/gray) to feed to OCR engines."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img = _upscale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    illum_corrected = _correct_illumination(gray)
    sharpened = _denoise_sharpen(illum_corrected)

    # Variant A: adaptive threshold (good for uniform small text)
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Variant B: Otsu threshold (good for higher-contrast regions)
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Variant C: raw color-corrected image, upscaled, for PaddleOCR (works better on
    # near-original images than binarized ones in many cases)
    color_enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return {
        "color": img,
        "gray": gray,
        "illum_corrected": illum_corrected,
        "sharpened": sharpened,
        "color_enhanced": color_enhanced,
        "adaptive": adaptive,
        "otsu": otsu,
    }


def _run_paddleocr(variants: dict) -> list[tuple[str, float]]:
    ocr = _get_paddle_ocr()
    results = []
    # Strip photos are often taken with the text running vertically. Paddle's
    # angle classifier only fixes 180° flips, so also feed a 90°-rotated copy;
    # duplicate lines across passes are deduped by the caller.
    for key in ("color", "color_enhanced"):
        for rotation in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            img = variants[key]
            if rotation is not None:
                img = cv2.rotate(img, rotation)
            out = ocr.ocr(img, cls=True)
            if not out or out[0] is None:
                continue
            for line in out[0]:
                text, conf = line[1][0], line[1][1]
                if text.strip():
                    results.append((text.strip(), float(conf)))
    return results


def _to_pil_image(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def _is_useful_ocr_line(text: str) -> bool:
    text = " ".join(text.split()).strip()
    if len(text) < 3:
        return False

    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    if letters < 2 and digits < 2:
        return False

    return True


def _lines_from_tesseract_data(data: dict) -> list[tuple[str, float, int, int]]:
    results = []
    grouped: dict[tuple[int, int, int], list[tuple[int, str, float, int]]] = {}

    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except ValueError:
            continue
        if conf < 10:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(
            (data["left"][i], text, conf / 100.0, data["top"][i])
        )

    for words in grouped.values():
        words.sort(key=lambda item: item[0])
        text = " ".join(word for _, word, _, _ in words)
        text = " ".join(text.split()).strip()
        if not _is_useful_ocr_line(text):
            continue

        avg_conf = sum(conf for _, _, conf, _ in words) / len(words)
        top = min(top for _, _, _, top in words)
        left = min(left for left, _, _, _ in words)
        results.append((text, avg_conf, top, left))

    return sorted(results, key=lambda item: (item[2], item[3]))


def _downscale_for_tesseract(img: np.ndarray, max_dim: int = 2200) -> np.ndarray:
    """Tesseract regularly hits the 20s timeout on full-resolution phone photos;
    cap the longest side before feeding it."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _run_tesseract(variants: dict) -> list[tuple[str, float]]:
    results_with_pos = []
    runs = (
        ("sharpened", None, "--oem 3 --psm 6"),
        ("sharpened", None, "--oem 3 --psm 11"),
        ("adaptive", None, "--oem 3 --psm 11"),
        ("color", None, "--oem 3 --psm 11"),
        # Rotated passes for strips photographed with vertical/upside-down text.
        ("sharpened", cv2.ROTATE_90_CLOCKWISE, "--oem 3 --psm 6"),
        ("sharpened", cv2.ROTATE_90_CLOCKWISE, "--oem 3 --psm 11"),
        ("sharpened", cv2.ROTATE_180, "--oem 3 --psm 11"),
        ("sharpened", cv2.ROTATE_90_COUNTERCLOCKWISE, "--oem 3 --psm 11"),
    )

    for key, rotation, config in runs:
        img = _downscale_for_tesseract(variants[key])
        if rotation is not None:
            img = cv2.rotate(img, rotation)
        pil_img = _to_pil_image(img)
        try:
            data = pytesseract.image_to_data(
                pil_img,
                config=config,
                output_type=pytesseract.Output.DICT,
                timeout=20,
            )
        except RuntimeError as e:
            print(f"[ocr] Tesseract timed out/failed for {key}: {e}")
            continue

        results_with_pos.extend(_lines_from_tesseract_data(data))

    results = []
    seen = set()
    for text, conf, _top, _left in results_with_pos:
        norm = text.lower()
        if norm in seen:
            continue
        seen.add(norm)
        results.append((text, conf))

    return results


def _run_gemini_vision_ocr(image_bytes: bytes, mime_type: str) -> list[tuple[str, float]]:
    prompt = """Read the text visible on this medicine strip or box.
Return only the transcribed text, line by line.
Preserve batch/lot labels, MFG/MFD dates, EXP dates, medicine names, strengths,
and manufacturer text exactly as much as possible.
If no readable medicine packaging text is visible, return nothing."""

    response = _generate_with_model_fallback(
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=1200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    lines = []
    for line in (response.text or "").splitlines():
        text = line.strip().strip("-*` ")
        if text and _is_useful_ocr_line(text):
            lines.append((text, 0.75))
    return lines


def _ocr_priority(text: str) -> int:
    """Generic ranking: medicine-identifying lines first, then manufacturing /
    regulatory details, then everything else."""
    lowered = text.lower()
    if any(term in lowered for term in (
        "tablet", "capsule", "syrup", "injection", "contains", "composition",
        "mg", "ml", "ip", "usp",
    )):
        return 0
    if any(term in lowered for term in (
        "mfg", "mfd", "exp", "batch", "b. no", "b.no", "mrp",
        "warning", "dose", "dosage", "store", "manufactured",
    )):
        return 1
    return 2


def extract_text(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Runs the full OCR pipeline and returns:
    {
        "raw_text": str,           # merged, deduped text block
        "lines": [(text, confidence), ...],
        "engine": "tesseract|paddleocr|local|all|gemini_vision"
    }
    """
    all_lines: list[tuple[str, float]] = []

    variants = preprocess_image(image_bytes)

    if OCR_ENGINE in {"paddleocr", "paddle", "local", "all"}:
        try:
            all_lines.extend(_run_paddleocr(variants))
        except Exception as e:
            print(f"[ocr] PaddleOCR failed: {e}")

    if OCR_ENGINE in {"tesseract", "local", "all"}:
        try:
            all_lines.extend(_run_tesseract(variants))
        except Exception as e:
            print(f"[ocr] Tesseract failed: {e}")

    # Dedupe near-identical lines while preserving useful OCR order. Confidence
    # sorting alone promotes tiny junk words on foil-textured strips.
    seen = set()
    filtered: list[tuple[str, float]] = []
    for text, conf in all_lines:
        norm = text.lower().strip()
        if norm in seen or conf < 0.15 or not _is_useful_ocr_line(text):
            continue
        seen.add(norm)
        filtered.append((text, conf))

    filtered = sorted(filtered, key=lambda item: (_ocr_priority(item[0]), -item[1]))

    engine = OCR_ENGINE
    if not filtered:
        try:
            filtered = _run_gemini_vision_ocr(image_bytes, mime_type)
            engine = "gemini_vision"
        except Exception as e:
            print(f"[ocr] Gemini Vision OCR fallback failed: {e}")

    raw_text = "\n".join(t for t, _ in filtered)

    return {
        "raw_text": raw_text,
        "lines": filtered,
        "engine": engine,
    }
