## TMH Backend (FastAPI)

### Run (Windows / PowerShell)

```bash
cd api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Endpoints

- `GET /health`
- `POST /analyze` (multipart form-data con campo `image`)

