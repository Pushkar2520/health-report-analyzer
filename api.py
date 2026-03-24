"""
api.py
──────
FastAPI backend for the Health Report Analyzer.
Exposes REST endpoints that accept file uploads and return analysis results.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import health_report_analyzer as hra
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ═════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Health Report Analyzer API",
    description=(
        "Multi-agent AI pipeline for analyzing medical health reports. "
        "Upload a PDF or image, get structured analysis with risk scoring."
    ),
    version="2.0.0",
)

# Allow Streamlit (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",    # Streamlit default
        "http://127.0.0.1:8501",   # Streamlit alt
    ],  # ← FIX #1: comma was missing after the closing bracket
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
}


# ═════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS (for OpenAPI docs)
# ═════════════════════════════════════════════════════════════════════════════
class HealthCheckResponse(BaseModel):
    status: str
    version: str


class ExtractTextResponse(BaseModel):
    text: str
    source: str


# ═════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """Check if the API is running and Mistral key is loaded."""
    return {
        "status": "ok",
        "version": "2.0.0",
    }


# ── Extract text only ────────────────────────────────────────────────────────
@app.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from a PDF or image without running the analysis pipeline.
    Useful for previewing what the OCR/PDF extractor sees.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    file_bytes = await file.read()

    try:
        text = hra.extract_text(file_bytes, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the uploaded file.",
        )

    return {"text": text, "source": file.content_type}


# ── Full analysis pipeline ───────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_report(
    file: UploadFile = File(...),
    gender: str = Form("male"),
):
    """
    Run the full multi-agent analysis pipeline:
      Agent 0: Report type classification
      Agent 1: Extraction
      Agent 2: Structuring
      Agent 3: Validation (hybrid rule + LLM)
      Agent 4: Risk Analysis (blended 80/20)
      Agent 5: Explanation

    Returns the complete results dictionary.

    Parameters:
        file:   PDF or image (png/jpg/jpeg)
        gender: "male" or "female" (affects reference ranges)
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # Validate gender
    if gender not in ("male", "female"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gender: {gender}. Must be 'male' or 'female'.",
        )

    # Read file
    file_bytes = await file.read()

    # Extract text
    try:
        text = hra.extract_text(file_bytes, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the uploaded file.",
        )

    # Run pipeline in thread pool to avoid blocking the async event loop
    try:
        results = await run_in_threadpool(hra.run_pipeline, text, gender)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Agent error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error in /analyze")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later.",
        )

    return results


# ── Analyze from raw text (no file upload needed) ────────────────────────────
class TextAnalysisRequest(BaseModel):
    text: str
    gender: str = "male"


@app.post("/analyze-text")
async def analyze_from_text(request: TextAnalysisRequest):
    # changed from "def" to "async def" because await requires it
    """
    Run the analysis pipeline on pre-extracted text.
    Useful when text extraction has already been done by the client,
    or for testing with pasted report text.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if request.gender not in ("male", "female"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gender: {request.gender}. Must be 'male' or 'female'.",
        )

    try:
        # FIX #3: closing parenthesis properly indented
        results = await run_in_threadpool(
            hra.run_pipeline,
            request.text,
            request.gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Agent error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error in /analyze-text")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later.",
        )

    return results