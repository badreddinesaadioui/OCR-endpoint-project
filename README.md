# OCR Endpoint Project

End-to-end project for CV parsing:
1. benchmark OCR and LLM parsing options
2. choose best stack from measured results
3. expose a usable API endpoint for CV -> structured JSON

Current selected stack (from benchmark exports):
1. OCR: `Mistral OCR 3`
2. LLM parsing: `Claude 4.5 Haiku`

## Repository Structure

- `app.py`, `pages/`  
  Streamlit benchmark application.

- `cv_api/`  
  FastAPI implementation of the CV parsing API.

- `app test/`  
  Svelte app to test the API (upload CV → Parse CV → parsed JSON).

- `docs/`  
  Functional, API, Swagger, and deployment documentation.

- `scripts/`  
  Utility scripts for export/reporting and Cloud Run deploy/validation.

- `ground_truth_database/`  
  CV dataset, parsed text, and benchmark databases.

## How to run each app

From the project root (after `pip install -r requirements.txt` and `npm install` in `app test/` if needed):

| App | Command | URL |
|-----|--------|-----|
| **API** (FastAPI) | `./api` or `./run_api.sh` | http://localhost:8080 — docs: http://localhost:8080/docs |
| **App test** (Svelte) | `cd "app test" && npm run dev` | http://localhost:5173 (needs API on 8080) |
| **Benchmark** (Streamlit) | `streamlit run app.py` | http://localhost:8501 |

Quick start: run `./api`, then in another terminal `cd "app test" && npm run dev` to try the API from the UI.

## API Endpoints

1. `GET /health`
2. `POST /v1/jobs` (async)
3. `GET /v1/jobs/{job_id}` (async polling)
4. `POST /v1/parse-cv` (sync)

## Key Documentation

- Docs index: `docs/README.md`
- API contract: `docs/API_CONTRACT_V1.md`
- Swagger step-by-step: `docs/SWAGGER_STEP_BY_STEP.md`
- Cloud Run deployment: `docs/CLOUD_RUN_DEPLOYMENT_GUIDE.md`
- Cloud Run plan: `docs/CLOUD_RUN_EXECUTION_PLAN.md`
- Commands cheatsheet: `docs/API_RUN_COMMANDS.md`

## Deployment (Cloud Run)

Automated scripts:
- `scripts/cloud_run_deploy.ps1`
- `scripts/cloud_run_validate.ps1`

Cloud build config:
- `cloudbuild.api.yaml`

## Notes Before PR

1. Keep secrets out of Git (`.streamlit/secrets.toml` already ignored).
2. Generated analysis artifacts under `exports/` are local (ignored).
3. Verify local API smoke test before opening PR.
