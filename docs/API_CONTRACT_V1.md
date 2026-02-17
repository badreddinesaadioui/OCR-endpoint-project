# API Contract V1 (CV Parsing Endpoint)

This document defines the API you will build next, based on:
- OCR model: `Mistral OCR 3`
- LLM parsing model: `Claude 4.5 Haiku`
- Output structure aligned with `pages/llm_parsing_benchmark.py` schema (`RESUME_EXTRACTION_SCHEMA`)

## 1) Versioning
- Base path: `/v1`
- Content type: `application/json` for metadata, `multipart/form-data` for file upload

## 2) Recommended Endpoints (async-first)

### 2.1 Create parsing job
`POST /v1/jobs`

Request (`multipart/form-data`):
- `file` (required): CV file (`pdf|png|jpg|jpeg|docx`)
- `language_hint` (optional): e.g. `fr`, `en`, `ar`
- `callback_url` (optional): webhook to notify completion

Response `202 Accepted`:
```json
{
  "job_id": "job_01JXYZ...",
  "status": "queued",
  "created_at": "2026-02-17T00:00:00Z",
  "estimated_sla_seconds": 45
}
```

### 2.2 Get job status/result
`GET /v1/jobs/{job_id}`

Response while running (`200 OK`):
```json
{
  "job_id": "job_01JXYZ...",
  "status": "processing",
  "created_at": "2026-02-17T00:00:00Z",
  "started_at": "2026-02-17T00:00:03Z",
  "progress": {
    "stage": "ocr",
    "percent": 40
  }
}
```

Response on success (`200 OK`):
```json
{
  "job_id": "job_01JXYZ...",
  "status": "succeeded",
  "created_at": "2026-02-17T00:00:00Z",
  "started_at": "2026-02-17T00:00:03Z",
  "finished_at": "2026-02-17T00:00:14Z",
  "timings": {
    "total_seconds": 14.1,
    "ocr_seconds": 6.2,
    "parsing_seconds": 7.5
  },
  "models": {
    "ocr": "mistral-ocr-latest",
    "llm_parser": "claude-4.5-haiku"
  },
  "result": {
    "linkedin_url": null,
    "name": "Jane Doe",
    "location": "Paris, France",
    "about": null,
    "open_to_work": null,
    "experiences": [
      {
        "position_title": "Data Scientist",
        "institution_name": "Company A",
        "linkedin_url": null,
        "from_date": "2022",
        "to_date": "Present",
        "duration": null,
        "location": "Paris, France",
        "description": null
      }
    ],
    "educations": [],
    "skills": [],
    "projects": [],
    "interests": [],
    "accomplishments": [],
    "contacts": []
  },
  "quality": {
    "json_valid": true,
    "schema_valid": true
  }
}
```

Response on failure (`200 OK`, terminal state):
```json
{
  "job_id": "job_01JXYZ...",
  "status": "failed",
  "error": {
    "code": "PARSING_SCHEMA_VALIDATION_FAILED",
    "message": "Model output did not satisfy resume schema."
  }
}
```

### 2.3 Optional sync endpoint (MVP)
`POST /v1/parse-cv`

Same upload format, but returns final result directly.
Use this only for small files and low traffic.

## 3) Job state machine
- `queued`
- `processing`
- `succeeded`
- `failed`
- `canceled` (optional future)

## 4) Validation rules
- Allowed file types: `pdf`, `png`, `jpg`, `jpeg`, `docx`
- Max file size (recommended): `<= 10 MB` (adjust as needed)
- Max page count (recommended): `<= 10 pages` for sync, larger only in async mode
- Final `result` must pass strict schema validation
- Unknown keys in result are removed or rejected

## 5) Error model
Every error response body:
```json
{
  "error": {
    "code": "STRING_CODE",
    "message": "Human readable message",
    "details": {}
  },
  "request_id": "req_01JXYZ..."
}
```

Recommended error codes:
- `INVALID_FILE_TYPE`
- `FILE_TOO_LARGE`
- `UNAUTHORIZED`
- `RATE_LIMITED`
- `OCR_PROVIDER_ERROR`
- `LLM_PROVIDER_ERROR`
- `PARSING_JSON_INVALID`
- `PARSING_SCHEMA_VALIDATION_FAILED`
- `INTERNAL_ERROR`

## 6) Security contract
- Auth header: `Authorization: Bearer <token>`
- Enforce rate limiting per client key
- Strip PII from application logs
- Keep provider API keys only in environment/secret manager

## 7) Observability contract
Include IDs in every response:
- `request_id` (always)
- `job_id` (for async endpoints)

Track metrics:
- request count and error rate
- p50/p95 latency
- queue depth (if async worker)
- provider failure rate (OCR, LLM)
- schema validation pass rate

## 8) Success criteria for V1 go-live
- `>= 99%` API availability
- `>= 99%` schema-valid outputs on accepted requests
- clear terminal state for every async job
- reproducible deployment via Docker
