# Cloud Run Deployment Guide (Step-by-step)

This is the operational guide to deploy the API outside local machine.

Scope:
- Deploy `cv_api.main:app` to Google Cloud Run
- Use Secret Manager for provider keys
- Validate the deployed endpoint

## 1) Pre-requisites

1. Google Cloud project (billing enabled)
2. `gcloud` CLI installed locally
3. Authenticated gcloud session
4. Permissions on project:
   - Cloud Run Admin
   - Artifact Registry Admin
   - Cloud Build Editor
   - Secret Manager Admin
5. A test CV file (`pdf/png/jpg/jpeg/docx`)

## 2) Install and authenticate gcloud (Windows)

Option A (recommended):
1. Install "Google Cloud CLI" from Google official installer
2. Restart terminal

Option B (if available in your environment):
```powershell
winget install Google.CloudSDK
```

Then authenticate:
```powershell
gcloud auth login
gcloud auth application-default login
```

## 3) One-command deployment script (recommended)

From repo root:

```powershell
cd "c:\Users\asus\Desktop\forvis mazar\OCR-endpoint-project"
```

Run deploy script:

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

What this script does:
1. Enables required Google APIs
2. Creates Artifact Registry repository if missing
3. Builds container image with Cloud Build (`cloudbuild.api.yaml`)
4. Creates/updates secrets in Secret Manager
5. Deploys Cloud Run service
6. Prints deployed service URL

Files used:
- `Dockerfile.api`
- `cloudbuild.api.yaml`
- `scripts/cloud_run_deploy.ps1`

## 4) Validate deployed service

After deployment, run:

```powershell
.\scripts\cloud_run_validate.ps1 `
  -ServiceUrl "https://YOUR_SERVICE_URL" `
  -SampleFilePath ".\ground_truth_database\cv\cv005.pdf" `
  -ApiAuthToken "YOUR_INTERNAL_BEARER_TOKEN"
```

What it validates:
1. `/health`
2. `POST /v1/jobs`
3. `GET /v1/jobs/{job_id}` polling until terminal state

File used:
- `scripts/cloud_run_validate.ps1`

## 5) Deployment modes

### Public endpoint mode
Use `-AllowUnauthenticated` in deploy script.

Recommended to keep app-level bearer token (`API_AUTH_TOKEN`) enabled.

### Private endpoint mode
Do not use `-AllowUnauthenticated`.

Then callers need Google IAM auth token (Cloud Run Invoker role).

## 6) Runtime defaults used in deploy script

- CPU: `1`
- Memory: `1Gi`
- Concurrency: `10`
- Timeout: `300s`
- Max instances: `10`
- Port: `8080`

Environment variables:
- `APP_ENV=prod`
- `MAX_FILE_SIZE_MB=10`
- `DEFAULT_SLA_SECONDS=45`
- `API_WORKER_THREADS=4`

Secrets mapped:
- `MISTRAL_API_KEY`
- `REPLICATE_API_TOKEN`
- `API_AUTH_TOKEN` (if provided)

## 7) Post-deploy checks

1. Open:
   - `https://YOUR_SERVICE_URL/docs`
2. Test:
   - `GET /health`
   - `POST /v1/jobs`
   - `GET /v1/jobs/{job_id}`
3. Confirm JSON schema-valid responses on successful jobs

## 8) Security notes

1. Rotate keys that were ever exposed in plain text.
2. Keep provider keys only in Secret Manager.
3. Add rate limits and monitoring for production traffic.
4. Avoid logging full CV content in production logs.

## 9) Current architecture limit

Current async jobs are in-memory in API process.

Implication:
- if service restarts, in-flight job state is lost.

Next recommended hardening:
1. move jobs to PostgreSQL
2. use Redis queue + worker service
3. keep API stateless
