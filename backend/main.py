import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, "app.env"))

# Add the parent directory so we can import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
import pandas as pd
import io
import uuid
import shutil
from pathlib import Path

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import from our pipeline
from pipeline import search_hs_code, step_b_translate_and_map, step_c_generate_csv

try:
    from google import genai
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception:
    gemini_client = None

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="SaHS - Saudi HS Code Classification API", 
    description="""
Welcome to the **SaHS (Saudi HS Code) API**. This service allows you to semantically search and classify items into the official Saudi Arabian Harmonized System.

## Quick Start

### 1. Base URL
All API requests should be made to:
*   **Production**: `https://sahs-c7oy.onrender.com:8000`
*   **Local**: `http://localhost:8000`

### 2. Authentication
Currently, the API is **Open Access**. No API keys are required for standard usage. However, requests are tracked by IP address for rate limiting.

### 3. Quick Start (cURL)
To classify an item, simply send a POST request to `/classify`:
```bash
curl -X POST "https://sahs-c7oy.onrender.com:8000/classify" \\
     -H "Content-Type: application/json" \\
     -d '{"items": [{"item_description": "Wireless Bluetooth Headphones"}]}'
```

### 4. Key Endpoints
*   **POST `/search`**: Semantic search of the HS dictionary.
*   **POST `/classify`**: Automatic mapping of invoice items to codes.
*   **GET `/export/{id}`**: Download results as CSV.

### 5. Rate Limiting
To ensure stability, the following limits apply per IP:
- **Search**: 60 req/min
- **Classification**: 10 req/min
- **Exports**: 30 req/min

---
*For more detailed implementation details, see the individual endpoint documentation below.*
    """,
    version="1.1.0",
    docs_url=None, # Disable default to use custom route
    redoc_url=None,
    root_path=os.environ.get("ROOT_PATH", ""),
    servers=[
        {"url": "https://sahs-c7oy.onrender.com:8000", "description": "Production server"},
        {"url": "http://localhost:8000", "description": "Local development server"}
    ]
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Ensure exports directory exists
EXPORT_DIR = Path(__file__).parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

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
        # Allow CDN assets for Swagger UI and ReDoc
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "img-src 'self' data: fastly.jsdelivr.net; "
            "frame-ancestors 'none';"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Hide the fact that we are using FastAPI/Uvicorn
        response.headers["Server"] = "Hidden" 
        return response

app.add_middleware(SecurityHeadersMiddleware)

# --- Pydantic Models ---

class InvoiceItem(BaseModel):
    item_description: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="The description of the item as it appears on the invoice (e.g., 'Kugellager')",
        examples=["Kugellager für Industriemotor"]
    )
    quantity: Optional[str] = Field(
        "1", 
        max_length=50,
        description="Optional quantity or count"
    )
    country_of_origin: Optional[str] = Field(
        "", 
        max_length=100,
        description="ISO country code or name"
    )

class ClassifyRequest(BaseModel):
    items: List[InvoiceItem] = Field(
        ..., 
        max_items=100,
        description="A list of items to classify. Limited to 100 items per batch."
    )

class SearchRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="The search query text (supports multiple languages).",
        examples=["Water pump"]
    )
    top_k: Optional[int] = Field(
        5, 
        ge=1, 
        le=100,
        description="Number of candidate matches to return."
    )

class HSCodeResult(BaseModel):
    hs_code: str = Field(..., description="The matching Saudi HS code.")
    desc_en: str = Field(..., description="English description from the official tariff.")
    desc_ar: str = Field(..., description="Arabic description from the official tariff.")
    distance: float = Field(..., description="Vector distance (lower is closer).")
    hierarchy_path: str = Field("", description="Full breadcrumb path in the tariff hierarchy.")
    duty_rate: str = Field("", description="Applicable English duty rate.")
    procedures: str = Field("", description="Customs procedures/restrictions.")

class ClassifyResponse(BaseModel):
    results: List[dict] = Field(..., description="Enriched items including classification reasoning.")
    export_id: Optional[str] = Field(None, description="UUID for downloading results as CSV.")

# --- Routes ---

@app.get("/", include_in_schema=False)
def root():
    return {"status": "SaHS API is active", "version": "1.1.0"}

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    # Use relative path if possible, otherwise rely on root_path fallback
    openapi_url = "./openapi.json"
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html(request: Request):
    openapi_url = "./openapi.json"
    return get_redoc_html(
        openapi_url=openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

@app.post("/search", response_model=List[HSCodeResult], tags=["Search"])
@limiter.limit("60/minute")
def search(req: SearchRequest, request: Request):
    """
    Search the Saudi HS Code database using semantic vector search.
    Bridging informal language to formal tariff terminology using query expansion.
    """
    try:
        results = search_hs_code(req.query, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
@limiter.limit("10/minute")
def classify(req: ClassifyRequest, request: Request):
    """
    Classify a list of invoice items using ChromaDB + Gemini.
    Returns enriched items with HS codes, descriptions, and AI reasoning.
    """
    try:
        items_as_dicts = [item.model_dump() for item in req.items]
        mapped = step_b_translate_and_map(items_as_dicts, gemini_client=gemini_client)
        
        # Save to temporary file for export
        export_id = str(uuid.uuid4())
        out_path = EXPORT_DIR / f"{export_id}.csv"
        step_c_generate_csv(mapped, output_filename=str(out_path))
        
        return {"results": mapped, "export_id": export_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/csv", tags=["Classification"])
@limiter.limit("5/minute")
async def classify_from_csv(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV file featuring an 'item_description' column.
    Returns a UUID to download the processed CSV.
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

        export_id = str(uuid.uuid4())
        out_path = EXPORT_DIR / f"{export_id}.csv"
        step_c_generate_csv(mapped, output_filename=str(out_path))
        
        return {
            "message": "Classification complete",
            "export_id": export_id,
            "download_url": f"/export/{export_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/{export_id}", tags=["Classification"])
@limiter.limit("30/minute")
def export_result(export_id: str, request: Request):
    """
    Download a classified CSV result using its UUID.
    """
    # Security: Validate UUID format
    try:
        uuid.UUID(export_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid export ID format")

    path = EXPORT_DIR / f"{export_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found or expired.")
    
    return FileResponse(
        path, 
        media_type="text/csv", 
        filename=f"sahs_classification_{export_id[:8]}.csv"
    )
