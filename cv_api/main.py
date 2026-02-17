"""
FastAPI app implementing the API contract in docs/API_CONTRACT_V1.md.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env from project root so you don't need export commands (optional; falls back to .streamlit/secrets.toml)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .pipeline import (
    LLM_MODEL_ID,
    OCR_MODEL_ID,
    PipelineError,
    prepare_pdf_bytes,
    resolve_provider_keys,
    run_llm_stage,
    run_ocr_stage,
)


ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx"}
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
DEFAULT_SLA_SECONDS = int(os.environ.get("DEFAULT_SLA_SECONDS", "45"))
WORKER_THREADS = int(os.environ.get("API_WORKER_THREADS", "4"))


class APIError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


app = FastAPI(
    title="CV Parsing API",
    version="1.0.0",
    description="Async CV parsing API: Mistral OCR 3 + Claude 4.5 Haiku",
)

# CORS: allow frontends (poc on localhost, or set CORS_ORIGINS env for deployed API)
_default_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
_cors_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_origins_env:
    allow_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    allow_origins = _default_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_job_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", f"req_{uuid4().hex[:12]}")


def _ext(filename: str) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def _build_error_payload(code: str, message: str, request_id: str, details: Optional[dict] = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
        "request_id": request_id,
    }


def _new_job(
    filename: str,
    content_type: Optional[str],
    size_bytes: int,
    language_hint: Optional[str],
    callback_url: Optional[str],
) -> Dict[str, Any]:
    return {
        "job_id": f"job_{uuid4().hex[:16]}",
        "status": "queued",
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "progress": {"stage": "queued", "percent": 0},
        "input": {
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "language_hint": language_hint,
            "callback_url": callback_url,
        },
        "timings": {
            "total_seconds": None,
            "ocr_seconds": None,
            "parsing_seconds": None,
        },
        "models": {
            "ocr": OCR_MODEL_ID,
            "llm_parser": LLM_MODEL_ID,
        },
        "quality": {
            "json_valid": False,
            "schema_valid": False,
        },
        "result": None,
        "error": None,
        "callback_delivery": None,
    }


def _update_job(job_id: str, **changes: Any) -> None:
    with _job_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(changes)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _job_lock:
        item = _jobs.get(job_id)
        return dict(item) if item else None


def _render_job_response(job: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    status = job["status"]
    payload: Dict[str, Any] = {
        "job_id": job["job_id"],
        "status": status,
        "created_at": job["created_at"],
        "request_id": request_id,
    }
    if job.get("started_at"):
        payload["started_at"] = job["started_at"]

    if status in {"queued", "processing"}:
        payload["progress"] = job["progress"]
        return payload

    if status == "succeeded":
        payload["finished_at"] = job.get("finished_at")
        payload["timings"] = job.get("timings")
        payload["models"] = job.get("models")
        payload["result"] = job.get("result")
        payload["quality"] = job.get("quality")
        return payload

    if status == "failed":
        payload["finished_at"] = job.get("finished_at")
        payload["error"] = job.get("error")
        return payload

    payload["progress"] = job.get("progress")
    return payload


def _notify_callback_if_needed(job_snapshot: Dict[str, Any]) -> None:
    callback_url = (job_snapshot.get("input") or {}).get("callback_url")
    if not callback_url:
        return

    delivery = {
        "attempted_at": _utc_now_iso(),
        "ok": False,
        "status_code": None,
        "error": None,
    }

    try:
        import requests

        response = requests.post(callback_url, json=job_snapshot, timeout=10)
        delivery["status_code"] = response.status_code
        delivery["ok"] = 200 <= response.status_code < 300
    except Exception as exc:  # noqa: BLE001
        delivery["error"] = str(exc)

    _update_job(job_snapshot["job_id"], callback_delivery=delivery)


def _run_job_pipeline(
    job_id: str,
    filename: str,
    file_bytes: bytes,
) -> None:
    start = time.perf_counter()
    _update_job(
        job_id,
        status="processing",
        started_at=_utc_now_iso(),
        progress={"stage": "preparing", "percent": 5},
    )

    try:
        mistral_key, replicate_key = resolve_provider_keys()
        pdf_bytes, api_filename = prepare_pdf_bytes(file_bytes, filename)

        _update_job(job_id, progress={"stage": "ocr", "percent": 35})
        ocr_stage = run_ocr_stage(pdf_bytes, api_filename, mistral_key)

        _update_job(job_id, progress={"stage": "llm_parsing", "percent": 75})
        llm_stage = run_llm_stage(ocr_stage["text"], replicate_key)

        total_seconds = time.perf_counter() - start
        _update_job(
            job_id,
            status="succeeded",
            finished_at=_utc_now_iso(),
            progress={"stage": "completed", "percent": 100},
            timings={
                "total_seconds": round(total_seconds, 4),
                "ocr_seconds": round(float(ocr_stage["time_seconds"]), 4),
                "parsing_seconds": round(float(llm_stage["time_seconds"]), 4),
            },
            quality={
                "json_valid": bool(llm_stage.get("json_valid")),
                "schema_valid": bool(llm_stage.get("schema_valid")),
            },
            result=llm_stage["parsed"],
            error=None,
        )
    except PipelineError as exc:
        _update_job(
            job_id,
            status="failed",
            finished_at=_utc_now_iso(),
            progress={"stage": "failed", "percent": 100},
            timings={
                "total_seconds": round(time.perf_counter() - start, 4),
                "ocr_seconds": None,
                "parsing_seconds": None,
            },
            error={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _update_job(
            job_id,
            status="failed",
            finished_at=_utc_now_iso(),
            progress={"stage": "failed", "percent": 100},
            timings={
                "total_seconds": round(time.perf_counter() - start, 4),
                "ocr_seconds": None,
                "parsing_seconds": None,
            },
            error={
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "details": {},
            },
        )

    snapshot = _get_job(job_id)
    if snapshot:
        _notify_callback_if_needed(snapshot)


def _validate_auth(request: Request) -> None:
    expected_token = os.environ.get("API_AUTH_TOKEN")
    if not expected_token:
        return

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise APIError(code="UNAUTHORIZED", message="Missing Bearer token", status_code=401)

    token = header[len("Bearer ") :].strip()
    if token != expected_token:
        raise APIError(code="UNAUTHORIZED", message="Invalid Bearer token", status_code=401)


def require_auth(request: Request) -> None:
    _validate_auth(request)


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    filename = file.filename or ""
    if not filename:
        raise APIError(code="INVALID_FILE_TYPE", message="Missing filename", status_code=415)

    extension = _ext(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise APIError(
            code="INVALID_FILE_TYPE",
            message=f"Allowed file types: {sorted(ALLOWED_EXTENSIONS)}",
            status_code=415,
            details={"received_extension": extension},
        )

    data = await file.read()
    if not data:
        raise APIError(code="INVALID_FILE_TYPE", message="Uploaded file is empty", status_code=415)

    if len(data) > MAX_FILE_SIZE_BYTES:
        raise APIError(
            code="FILE_TOO_LARGE",
            message=f"File exceeds limit of {MAX_FILE_SIZE_MB} MB",
            status_code=413,
            details={"size_bytes": len(data), "max_bytes": MAX_FILE_SIZE_BYTES},
        )

    return data


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or f"req_{uuid4().hex[:12]}"
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_payload(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=_request_id(request),
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_build_error_payload(
            code="INTERNAL_ERROR",
            message="Validation error",
            details={"errors": exc.errors()},
            request_id=_request_id(request),
        ),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):  # noqa: BLE001
    return JSONResponse(
        status_code=500,
        content=_build_error_payload(
            code="INTERNAL_ERROR",
            message=str(exc),
            details={},
            request_id=_request_id(request),
        ),
    )


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "time": _utc_now_iso(),
        "request_id": _request_id(request),
    }


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(require_auth)])
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    language_hint: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
):
    file_bytes = await _read_and_validate_upload(file)
    job = _new_job(
        filename=file.filename or "document",
        content_type=file.content_type,
        size_bytes=len(file_bytes),
        language_hint=language_hint,
        callback_url=callback_url,
    )

    with _job_lock:
        _jobs[job["job_id"]] = job

    _executor.submit(
        _run_job_pipeline,
        job["job_id"],
        file.filename or "document",
        file_bytes,
    )

    return {
        "job_id": job["job_id"],
        "status": "queued",
        "created_at": job["created_at"],
        "estimated_sla_seconds": DEFAULT_SLA_SECONDS,
        "request_id": _request_id(request),
    }


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_auth)])
def get_job(job_id: str, request: Request):
    job = _get_job(job_id)
    if not job:
        raise APIError(
            code="JOB_NOT_FOUND",
            message=f"Job not found: {job_id}",
            status_code=404,
        )
    return _render_job_response(job, _request_id(request))


@app.post("/v1/parse-cv", dependencies=[Depends(require_auth)])
async def parse_cv_sync(
    request: Request,
    file: UploadFile = File(...),
    language_hint: Optional[str] = Form(None),
):
    _ = language_hint  # reserved for future prompt tuning
    file_bytes = await _read_and_validate_upload(file)
    filename = file.filename or "document"

    total_start = time.perf_counter()
    try:
        mistral_key, replicate_key = resolve_provider_keys()
        pdf_bytes, api_filename = prepare_pdf_bytes(file_bytes, filename)
        ocr_stage = run_ocr_stage(pdf_bytes, api_filename, mistral_key)
        llm_stage = run_llm_stage(ocr_stage["text"], replicate_key)
    except PipelineError as exc:
        raise APIError(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        ) from exc

    return {
        "status": "succeeded",
        "created_at": _utc_now_iso(),
        "finished_at": _utc_now_iso(),
        "timings": {
            "total_seconds": round(time.perf_counter() - total_start, 4),
            "ocr_seconds": round(float(ocr_stage["time_seconds"]), 4),
            "parsing_seconds": round(float(llm_stage["time_seconds"]), 4),
        },
        "models": {
            "ocr": OCR_MODEL_ID,
            "llm_parser": LLM_MODEL_ID,
        },
        "result": llm_stage["parsed"],
        "quality": {
            "json_valid": bool(llm_stage["json_valid"]),
            "schema_valid": bool(llm_stage["schema_valid"]),
        },
        "request_id": _request_id(request),
    }


@app.get("/")
def root():
    return {
        "service": "cv-parsing-api",
        "version": "1.0.0",
        "docs": "/docs",
    }
