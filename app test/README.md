# CV Parser – App Test

Svelte app to test the CV Parsing API: drag-and-drop a PDF, click **Parse CV**, see the structured result.

## Run locally

1. **Start the API** (in the project root):
   ```bash
   uvicorn cv_api.main:app --host 0.0.0.0 --port 8080 --reload
   ```

2. **Install and run this app**:
   ```bash
   cd "app test"
   npm install
   npm run dev
   ```
   Open http://localhost:5173

3. **Optional**: Copy `.env.example` to `.env` and set:
   - `VITE_API_BASE_URL` – API base URL (default: `http://localhost:8080`)
   - `VITE_API_AUTH_TOKEN` – if the API uses Bearer auth

When the API is deployed, set `VITE_API_BASE_URL` to the deployed URL (and rebuild: `npm run build`).

## Logo

Put the Forvis Mazars logo as `public/logo.png`. If the file is missing, the app shows a text fallback.
