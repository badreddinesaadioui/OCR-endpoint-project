# CV Parser – POC

Svelte app to test the CV Parsing API: drag-and-drop a PDF, click **Parse CV**, see the structured result.

## Run with local API

1. **Start the API** (in the project root):
   ```bash
   uvicorn cv_api.main:app --host 0.0.0.0 --port 8080 --reload
   ```

2. **Install and run this app**:
   ```bash
   cd poc
   npm install
   npm run dev
   ```
   Open http://localhost:5173

3. **Optional**: Copy `.env.example` to `.env` and set `VITE_API_BASE_URL=http://localhost:8080` (default).

## Use the deployed API (Cloud Run)

To point the POC at your **deployed** API instead of localhost:

1. Copy `.env.example` to `.env` (if you don’t have `.env` yet).
2. Set `VITE_API_BASE_URL` to your Cloud Run service URL (e.g. `https://cv-parsing-api-xxxxx-ew.a.run.app`). You get this URL after running `scripts/cloud_run_deploy.ps1`.
3. If you deployed with an `API_AUTH_TOKEN`, set `VITE_API_AUTH_TOKEN` in `.env` to the same value.
4. Run the app: `npm run dev` (or `npm run build` then serve `dist/` for production).

In **dev**, the app uses a Vite proxy (`/api` → Cloud Run) so CORS is avoided; ensure `poc/.env` has `VITE_API_BASE_URL` and `VITE_API_AUTH_TOKEN`. To validate the API from the terminal: `curl -s "https://YOUR_SERVICE_URL/health" -H "Authorization: Bearer YOUR_TOKEN"` and `curl -X POST "https://YOUR_SERVICE_URL/v1/parse-cv" -H "Authorization: Bearer YOUR_TOKEN" -F "file=@path/to/cv.pdf"` (see `docs/API_INTEGRATION_README.md`).

**You do not need a Google API key in this app.** The browser calls your Cloud Run URL directly. If the service is deployed with “Allow unauthenticated”, no extra auth is required; if you use Bearer auth, only `VITE_API_AUTH_TOKEN` is needed.

## What you need to deploy the API (one-time)

To deploy the API to Cloud Run (so you can use the POC against it), you need:

- **Google Cloud**: A GCP project with billing enabled, and `gcloud` CLI installed and logged in (`gcloud auth login` + `gcloud auth application-default login`). No “Google API key” file—auth is via gcloud.
- **Provider keys**: `MISTRAL_API_KEY` and `REPLICATE_API_TOKEN` (passed to the deploy script; they are stored in Secret Manager).
- **Optional**: `API_AUTH_TOKEN` for Bearer auth on the API.

See `docs/CLOUD_RUN_DEPLOYMENT_GUIDE.md` and `scripts/cloud_run_deploy.ps1`.

## Logo

Put the Forvis Mazars logo as `public/logo.png`. If the file is missing, the app shows a text fallback.
