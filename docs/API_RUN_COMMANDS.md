# API Run Commands

## 1) Start locally (Windows PowerShell)

```powershell
cd "c:\Users\asus\Desktop\forvis mazar\OCR-endpoint-project"
.\.venv\Scripts\uvicorn cv_api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open Swagger UI:
- `http://localhost:8080/docs`

Note:
- `--host 0.0.0.0` means "listen on all interfaces".
- In browser, still use `http://localhost:8080/...` (not `http://0.0.0.0:8080/...`).

## 2) Optional auth

If you want Bearer auth enabled:

```powershell
$env:API_AUTH_TOKEN = "your-token"
```

Then call endpoints with:
- `Authorization: Bearer your-token`

## 3) Async API usage

Create job:

```powershell
curl -X POST "http://localhost:8080/v1/jobs" `
  -H "accept: application/json" `
  -F "file=@ground_truth_database/cv/cv005.pdf"
```

Get status/result:

```powershell
curl "http://localhost:8080/v1/jobs/{job_id}"
```

## 4) Sync API usage

```powershell
curl -X POST "http://localhost:8080/v1/parse-cv" `
  -H "accept: application/json" `
  -F "file=@ground_truth_database/cv/cv005.pdf"
```

## 5) Docker (API image)

Build:

```powershell
docker build -f Dockerfile.api -t cv-parsing-api:latest .
```

Run:

```powershell
docker run --rm -p 8080:8080 `
  -e MISTRAL_API_KEY="..." `
  -e REPLICATE_API_TOKEN="..." `
  cv-parsing-api:latest
```

## 6) App test (Svelte UI)

Run (from project root):

```bash
cd "app test"
npm install
npm run dev
```

Open: `http://localhost:5173`

This UI calls the API and lets you upload a CV (PDF), click Parse CV, and view/copy the parsed JSON. Ensure the API is running on port 8080.

## 7) Cloud Run deploy (automated script)

```powershell
.\scripts\cloud_run_deploy.ps1 `
  -ProjectId "YOUR_GCP_PROJECT_ID" `
  -Region "europe-west1" `
  -ServiceName "cv-parsing-api" `
  -ArtifactRepo "cv-api-repo" `
  -MistralApiKey "YOUR_MISTRAL_API_KEY" `
  -ReplicateApiToken "YOUR_REPLICATE_API_TOKEN" `
  -ApiAuthToken "YOUR_INTERNAL_BEARER_TOKEN" `
  -AllowUnauthenticated
```

Validate deployed service:

```powershell
.\scripts\cloud_run_validate.ps1 `
  -ServiceUrl "https://YOUR_SERVICE_URL" `
  -SampleFilePath ".\ground_truth_database\cv\cv005.pdf" `
  -ApiAuthToken "YOUR_INTERNAL_BEARER_TOKEN"
```
