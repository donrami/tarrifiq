# SaHS API Developer Guide

Welcome to the **SaHS (Saudi HS Code) API**. This service allows you to semantically search and classify items into the official Saudi Arabian Harmonized System.

## Base URL
The API is currently available at the root of your deployment (e.g., `http://localhost:8000`).

## Interactive Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## Features
- **Semantic Search**: Search using natural language; the API expands your query to match tariff terminology.
- **AI-Powered Classification**: Map invoice items to HS codes using a combination of vector search and Gemini LLM.
- **Batch Processing**: Send lists of items for classification.
- **CSV Export**: Get results in a ready-to-use CSV format.

---

## Quick Start Examples

### 1. Semantic Search (cURL)
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "industrial water pump", "top_k": 3}'
```

### 2. Classify Items (Python)
```python
import requests

url = "http://localhost:8000/classify"
payload = {
    "items": [
        {"item_description": "Kugellager", "quantity": "10"}
    ]
}

response = requests.post(url, json=payload)
data = response.json()

print(f"HS Code: {data['results'][0]['hs_code']}")
print(f"Export ID: {data['export_id']}")
```

### 3. Download CSV
```bash
# Use the export_id from the classify response
curl -O "http://localhost:8000/export/<EXPORT_ID>"
```

---

## Rate Limits
To ensure fair usage, we apply public rate limits based on your IP address:
- `search`: 60 requests per minute
- `classify`: 10 requests per minute
- `classify/csv`: 5 requests per minute
- `export`: 30 requests per minute
