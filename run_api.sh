#!/usr/bin/env bash
# Run the CV Parsing API (loads .env from project root, no export needed)
cd "$(dirname "$0")"
.venv/bin/uvicorn cv_api.main:app --host 0.0.0.0 --port 8080 --reload
