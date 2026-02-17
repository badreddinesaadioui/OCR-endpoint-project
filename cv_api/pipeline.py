"""
Pipeline logic for CV parsing endpoint:
file -> PDF -> OCR (Mistral) -> structured parsing (Claude) -> schema validation.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import ocr_common

from .resume_schema import RESUME_EXTRACTION_SCHEMA, parse_json_from_response, validate_resume_schema


OCR_MODEL_ID = "mistral-ocr-latest"
LLM_MODEL_ID = "claude-4.5-haiku"
REPLICATE_CLAUDE_MODEL = "anthropic/claude-4.5-haiku"
REPLICATE_WAIT_SECONDS = 60
REPLICATE_COST_ESTIMATE = 0.002


class PipelineError(Exception):
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


def _load_local_secrets_file() -> Dict[str, str]:
    """
    Lightweight parser for .streamlit/secrets.toml key="value" pairs.
    Keeps API runnable locally without forcing env vars.
    """
    root = Path(__file__).resolve().parents[1]
    path = root / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return {}

    out: Dict[str, str] = {}
    line_re = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$')
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = line_re.match(line.strip())
            if match:
                out[match.group(1)] = match.group(2)
    except Exception:
        return {}
    return out


def resolve_provider_keys() -> Tuple[str, str]:
    secrets_data = _load_local_secrets_file()

    mistral_key = os.environ.get("MISTRAL_API_KEY") or secrets_data.get("MISTRAL_API_KEY")
    replicate_key = (
        os.environ.get("REPLICATE_API_TOKEN")
        or os.environ.get("REPLICATE_API_KEY")
        or secrets_data.get("REPLICATE_API_TOKEN")
        or secrets_data.get("REPLICATE_API_KEY")
    )

    if not mistral_key:
        raise PipelineError(
            code="INTERNAL_ERROR",
            message="Missing MISTRAL_API_KEY",
            status_code=500,
        )
    if not replicate_key:
        raise PipelineError(
            code="INTERNAL_ERROR",
            message="Missing REPLICATE_API_TOKEN/REPLICATE_API_KEY",
            status_code=500,
        )
    return mistral_key, replicate_key


def _ext_from_filename(filename: str) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def prepare_pdf_bytes(file_bytes: bytes, filename: str) -> Tuple[bytes, str]:
    ext = _ext_from_filename(filename)
    base = Path(filename).stem or "document"
    api_filename = f"{base}.pdf"

    if ext == "pdf":
        return file_bytes, api_filename

    if ext in {"png", "jpg", "jpeg"}:
        image_type = "png" if ext == "png" else "jpeg"
        pdf_bytes = ocr_common._image_to_pdf(file_bytes, filetype=image_type)
        if pdf_bytes is None:
            raise PipelineError(
                code="INTERNAL_ERROR",
                message="Could not convert image to PDF",
                status_code=500,
            )
        return pdf_bytes, api_filename

    if ext == "docx":
        pdf_bytes, convert_error = ocr_common._convert_docx_to_pdf(file_bytes)
        if pdf_bytes is None:
            raise PipelineError(
                code="INTERNAL_ERROR",
                message="Could not convert DOCX to PDF",
                status_code=500,
                details={"conversion_error": convert_error or "unknown"},
            )
        return pdf_bytes, api_filename

    raise PipelineError(
        code="INVALID_FILE_TYPE",
        message=f"Unsupported file extension: .{ext or 'unknown'}",
        status_code=415,
    )


def run_ocr_stage(pdf_bytes: bytes, api_filename: str, mistral_key: str) -> Dict:
    raw = ocr_common.run_mistral_ocr(pdf_bytes, api_filename, mistral_key)
    if raw.get("error"):
        raise PipelineError(
            code="OCR_PROVIDER_ERROR",
            message=str(raw.get("error")),
            status_code=502,
        )

    text = (raw.get("text") or "").strip()
    if not text:
        raise PipelineError(
            code="OCR_PROVIDER_ERROR",
            message="OCR returned empty text",
            status_code=502,
        )

    return {
        "text": text,
        "time_seconds": float(raw.get("time_seconds") or 0.0),
        "cost_usd": float(raw.get("cost_usd") or 0.0),
    }


def _replicate_text(raw) -> str:
    if isinstance(raw, list):
        return "".join(str(x) for x in raw).strip()
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()


def run_llm_stage(ocr_text: str, replicate_key: str) -> Dict:
    import json

    import replicate

    schema_desc = json.dumps(RESUME_EXTRACTION_SCHEMA, indent=2)
    system_prompt = (
        "You are a precise resume parser. Extract structured data from the given OCR text. "
        "Respond with one JSON object only. Use null for missing values. "
        "Do not add explanations."
    )
    user_prompt = (
        "Extract resume data from the following text into this exact JSON schema.\n\n"
        f"Schema:\n{schema_desc}\n\n"
        f"Text to parse:\n{ocr_text}"
    )

    start = time.perf_counter()
    previous = os.environ.get("REPLICATE_API_TOKEN")
    try:
        os.environ["REPLICATE_API_TOKEN"] = replicate_key
        raw = replicate.run(
            REPLICATE_CLAUDE_MODEL,
            input={
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_tokens": 4096,
            },
            wait=REPLICATE_WAIT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(
            code="LLM_PROVIDER_ERROR",
            message=str(exc),
            status_code=502,
        ) from exc
    finally:
        if previous is None:
            os.environ.pop("REPLICATE_API_TOKEN", None)
        else:
            os.environ["REPLICATE_API_TOKEN"] = previous

    elapsed = time.perf_counter() - start
    text = _replicate_text(raw)
    parsed, parse_error = parse_json_from_response(text)
    if parse_error:
        raise PipelineError(
            code="PARSING_JSON_INVALID",
            message=parse_error,
            status_code=422,
        )

    valid, schema_error, cleaned = validate_resume_schema(parsed)
    if not valid or cleaned is None:
        raise PipelineError(
            code="PARSING_SCHEMA_VALIDATION_FAILED",
            message=schema_error or "Schema validation failed",
            status_code=422,
        )

    return {
        "parsed": cleaned,
        "text": text,
        "time_seconds": elapsed,
        "cost_usd": REPLICATE_COST_ESTIMATE,
        "json_valid": True,
        "schema_valid": True,
    }
