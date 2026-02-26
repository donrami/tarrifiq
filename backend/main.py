import sys
import os
import json
from dotenv import load_dotenv

load_dotenv("app.env")

# Add the parent directory so we can import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
import pandas as pd
import io

# Import from our pipeline
from pipeline import search_hs_code, step_b_translate_and_map, step_c_generate_csv

try:
    from google import genai
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception:
    gemini_client = None

app = FastAPI(
    title="HS Code Classification API", 
    version="1.0.0",
    docs_url="/docs",  # You could disable this in production for extra security
    redoc_url=None
)

# 1. CORS
frontend_url = os.environ.get("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url] if frontend_url != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 2. Trusted Host Middleware (Prevents Host Header Injection)
# In production, set this to your actual domain name.
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] # Change this to ["yourdomain.com", "localhost"] in production
)

# 3. Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Hide the fact that we are using FastAPI/Uvicorn
        response.headers["Server"] = "Hidden" 
        return response

app.add_middleware(SecurityHeadersMiddleware)

# --- Pydantic Models ---

class InvoiceItem(BaseModel):
    item_description: str = Field(..., min_length=1, max_length=1000)
    quantity: Optional[str] = Field("1", max_length=50)
    country_of_origin: Optional[str] = Field("", max_length=100)

class ClassifyRequest(BaseModel):
    items: List[InvoiceItem] = Field(..., max_items=100) # Limit batch size to 100

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(5, ge=1, le=100)

class HSCodeResult(BaseModel):
    hs_code: str
    desc_en: str
    desc_ar: str
    distance: float
    hierarchy_path: str = ""
    duty_rate: str = ""
    procedures: str = ""

# --- Routes ---

@app.get("/")
def root():
    return {"status": "HS Code Classification API is running"}

@app.post("/search", response_model=List[HSCodeResult])
def search(req: SearchRequest):
    """Directly search the vector DB for HS codes."""
    try:
        results = search_hs_code(req.query, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Classify a list of invoice items using ChromaDB + Gemini.
    Returns enriched items with hs_code, arabic_description, and reasoning.
    """
    try:
        items_as_dicts = [item.model_dump() for item in req.items]
        mapped = step_b_translate_and_map(items_as_dicts, gemini_client=gemini_client)
        return {"results": mapped}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/csv")
async def classify_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with a column 'item_description' and get back a classified CSV.
    """
    try:
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")
            
        df = pd.read_csv(io.BytesIO(contents), dtype=str)
        if "item_description" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have an 'item_description' column")

        items = df.to_dict(orient="records")
        mapped = step_b_translate_and_map(items, gemini_client=gemini_client)

        import tempfile
        from starlette.background import BackgroundTask
        fd, out_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        
        step_c_generate_csv(mapped, output_filename=out_path)
        return FileResponse(out_path, media_type="text/csv", filename="classified_output.csv", background=BackgroundTask(os.remove, out_path))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
def export_last_result():
    """
    Download the last generated CSV.
    WARNING: This serves a global file. In a multi-user environment, 
    this could lead to data leakage. Consider session-based storage.
    """
    path = "processed_invoice.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No processed invoice found. Run classify first.")
    return FileResponse(path, media_type="text/csv", filename="processed_invoice.csv")
