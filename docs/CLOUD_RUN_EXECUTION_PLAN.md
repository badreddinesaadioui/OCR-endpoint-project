# Cloud Run Execution Plan

This plan is the practical roadmap from local API to usable cloud endpoint.

## Phase 0 - Inputs

Required inputs:
1. GCP project id
2. region (example: `europe-west1`)
3. Mistral API key
4. Replicate API token
5. optional app bearer token (`API_AUTH_TOKEN`)

Deliverable:
- values ready for `scripts/cloud_run_deploy.ps1`

## Phase 1 - Local readiness

Checklist:
1. local API works (`/health` returns 200)
2. local async flow works (`POST /v1/jobs` + `GET /v1/jobs/{job_id}`)
3. schema-valid JSON returned on a sample CV

Deliverable:
- confidence before cloud deployment

## Phase 2 - Cloud foundation

Checklist:
1. install `gcloud`
2. `gcloud auth login`
3. set project and verify billing
4. ensure IAM permissions

Deliverable:
- operator workstation ready to deploy

## Phase 3 - Build and deploy

Execution:
1. run `scripts/cloud_run_deploy.ps1`
2. collect service URL output
3. verify service revision is healthy

Deliverable:
- deployed Cloud Run URL

## Phase 4 - Functional validation

Execution:
1. run `scripts/cloud_run_validate.ps1`
2. test `/docs`
3. test one async job end-to-end

Exit criteria:
1. `/health` OK
2. async job reaches terminal status
3. successful job returns schema-valid JSON

## Phase 5 - POC integration

Use API from:
1. `app test/` (Svelte UI: drag-drop CV → Parse CV → JSON)
2. any frontend/backend app via HTTPS

Deliverable:
- POC demo with remote cloud endpoint

## Phase 6 - Hardening for production

Next actions:
1. move job storage to PostgreSQL
2. add queue + worker (Redis/Celery/RQ)
3. enforce auth/rate limiting
4. add dashboards and alerts
5. define SLOs and incident runbook

Deliverable:
- production-grade service plan
