"""
OCR Benchmark: Upload PDF + ground truth → parallel OCR (Mistral, OpenAI Vision, Replicate).
Metrics: word recall, CER, WER, layout accuracy, cost, speed. Cost from API usage or list price.
Multi-criteria: Borda count + Condorcet. APIs are called fresh each time.
"""
import base64
import csv
import io
import json
import os
import re
import time
from threading import Thread
from typing import Optional

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _raw_response_to_json(obj):
    """Turn API response objects into JSON-serializable dict/list/str for display."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _raw_response_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_raw_response_to_json(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return _raw_response_to_json(obj.model_dump())
    if hasattr(obj, "dict"):
        return _raw_response_to_json(obj.dict())
    if hasattr(obj, "__dict__"):
        return _raw_response_to_json(vars(obj))
    return str(obj)

# ---------------------------------------------------------------------------
# Secrets (Streamlit reads .streamlit/secrets.toml)
# ---------------------------------------------------------------------------
def _get_secret(key: str, alt_key: str = None):
    try:
        return st.secrets.get(key) or st.secrets.get(alt_key or "")
    except Exception:
        return None

def get_openai_key():
    return _get_secret("OPENAI_API_KEY")

def get_mistral_key():
    return _get_secret("MISTRAL_API_KEY")

def get_replicate_key():
    # Replicate client expects REPLICATE_API_TOKEN; we support REPLICATE_API_KEY too
    return _get_secret("REPLICATE_API_KEY") or _get_secret("REPLICATE_API_TOKEN")

# ---------------------------------------------------------------------------
# Database (same as our_database.py: CVtheque library)
# ---------------------------------------------------------------------------
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ground_truth_database")
CV_DIR = os.path.join(DB_DIR, "cv")
PARSING_TXT_DIR = os.path.join(DB_DIR, "parsed")
SCREENSHOTS_DIR = os.path.join(DB_DIR, "screenshots")
METADATA_PATH = os.path.join(DB_DIR, "metadata.csv")

@st.cache_data
def load_db_metadata():
    """Load CV library metadata (filename, extension, language, layout_type, ...)."""
    if not os.path.isfile(METADATA_PATH):
        return []
    rows = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filename"):
                rows.append(row)
    return rows

def _screenshot_path(base: str) -> str:
    return os.path.join(SCREENSHOTS_DIR, f"{base}.png")

# ---------------------------------------------------------------------------
# Ground truth vs prediction: metrics (word recall, CER, WER, layout)
# ---------------------------------------------------------------------------
def normalize_words(text: str) -> list:
    if not (text and text.strip()):
        return []
    clean = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return [w for w in clean.split() if w]

def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance between two sequences (lists of tokens or chars)."""
    R, H = len(ref), len(hyp)
    if R == 0:
        return H
    if H == 0:
        return R
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[R][H]

def cer(ground_truth: str, predicted: str) -> float:
    """Character Error Rate: edit_distance(ref_chars, pred_chars) / len(ref_chars). Lower is better; 0 = perfect."""
    ref_chars = list((ground_truth or "").strip())
    if not ref_chars:
        return 0.0
    hyp_chars = list((predicted or "").strip())
    ed = _edit_distance(ref_chars, hyp_chars)
    return (ed / len(ref_chars)) * 100.0

def wer(ground_truth: str, predicted: str) -> float:
    """Word Error Rate: word-level edit distance / num_gt_words. Lower is better; 0 = perfect."""
    ref_words = normalize_words(ground_truth)
    if not ref_words:
        return 0.0
    hyp_words = normalize_words(predicted)
    ed = _edit_distance(ref_words, hyp_words)
    return (ed / len(ref_words)) * 100.0

def layout_accuracy(ground_truth: str, predicted: str) -> float:
    """Layout accuracy: % of GT section titles found in prediction. Uses only real titles (no '--' or layout decoration)."""
    pred_lower = (predicted or "").lower()
    lines = [ln.strip() for ln in (ground_truth or "").splitlines() if ln.strip()]
    section_headers = []
    for ln in lines:
        if len(ln) > 60:
            continue
        # Skip separator/decoration lines (dashes, underscores, etc.)
        if "--" in ln or "—" in ln:
            continue
        letters = sum(1 for c in ln if c.isalpha())
        if letters < 2 or letters / max(len(ln), 1) < 0.4:  # skip layout-only lines (e.g. "---", "___")
            continue
        # Title-like: all caps or starts with capital, no trailing sentence punctuation
        if ln.isupper() or (ln and ln[0].isupper() and not ln.rstrip().endswith(",") and not ln.rstrip().endswith(".")):
            section_headers.append(ln.strip())
    section_headers = list(dict.fromkeys(section_headers))[:30]
    if not section_headers:
        return 100.0
    found = sum(1 for h in section_headers if h.lower() in pred_lower)
    return (found / len(section_headers)) * 100.0

def word_metrics(ground_truth_text: str, predicted_text: str) -> dict:
    """Accuracy = recall (% of ground-truth words found). Plus CER, WER, layout_accuracy, missing/extra lists."""
    gt_words = normalize_words(ground_truth_text)
    pred_words = normalize_words(predicted_text)
    pred_set = set(pred_words)
    gt_set = set(gt_words)
    if not gt_words:
        return {
            "accuracy_pct": 100.0,
            "total_gt_words": 0,
            "found": 0,
            "missing_words": [],
            "extra_words": list(dict.fromkeys(pred_words)),
            "cer_pct": 0.0,
            "wer_pct": 0.0,
            "layout_accuracy_pct": 100.0,
        }
    found = sum(1 for w in gt_words if w in pred_set)
    missing = list(dict.fromkeys(w for w in gt_words if w not in pred_set))
    extra = list(dict.fromkeys(w for w in pred_words if w not in gt_set))
    return {
        "accuracy_pct": (found / len(gt_words)) * 100.0,
        "total_gt_words": len(gt_words),
        "found": found,
        "missing_words": missing,
        "extra_words": extra,
        "cer_pct": cer(ground_truth_text, predicted_text),
        "wer_pct": wer(ground_truth_text, predicted_text),
        "layout_accuracy_pct": layout_accuracy(ground_truth_text, predicted_text),
    }

# ---------------------------------------------------------------------------
# PDF → images (for OpenAI Vision)
# ---------------------------------------------------------------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    """Return list of PNG image bytes, one per page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install pymupdf: pip install pymupdf")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        images.append(pix.tobytes("png"))
    doc.close()
    return images

def pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0

# ---------------------------------------------------------------------------
# Image (PNG/JPEG) → single-page PDF (for library image support)
# ---------------------------------------------------------------------------
def _image_to_pdf(image_bytes: bytes, filetype: str = "png") -> Optional[bytes]:
    """Convert image bytes (PNG or JPEG) to a single-page PDF using PyMuPDF. Returns PDF bytes or None."""
    try:
        import fitz
        img_doc = fitz.open(stream=image_bytes, filetype=filetype)
        if len(img_doc) == 0:
            img_doc.close()
            return None
        w, h = img_doc[0].rect.width, img_doc[0].rect.height
        img_doc.close()
        doc = fitz.open()
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=image_bytes)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except Exception:
        return None

# ---------------------------------------------------------------------------
# DOCX → PDF (for library DOCX support; requires LibreOffice headless)
# ---------------------------------------------------------------------------
def _convert_docx_to_pdf(docx_bytes: bytes) -> tuple[Optional[bytes], Optional[str]]:
    """Convert DOCX bytes to PDF using LibreOffice headless. Returns (pdf_bytes, None) on success, (None, error_message) on failure."""
    import subprocess
    import tempfile
    tmpdir = os.path.abspath(tempfile.mkdtemp())
    docx_path = os.path.join(tmpdir, "document.docx")
    pdf_path = os.path.join(tmpdir, "document.pdf")
    try:
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        # Try LibreOffice: system PATH first, then macOS app bundle
        candidates = [
            "libreoffice",
            "soffice",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
        last_err = None
        for cmd in candidates:
            if cmd.startswith("/") and not os.path.isfile(cmd):
                continue
            try:
                result = subprocess.run(
                    [cmd, "--headless", "--convert-to", "pdf", "--outdir", tmpdir, docx_path],
                    capture_output=True,
                    timeout=90,
                    text=True,
                    cwd=tmpdir,
                )
                if result.returncode != 0:
                    last_err = (result.stderr or result.stdout or f"Exit code {result.returncode}").strip()[:200]
                    continue
                if os.path.isfile(pdf_path):
                    with open(pdf_path, "rb") as f:
                        return (f.read(), None)
                last_err = "LibreOffice did not create document.pdf"
            except FileNotFoundError:
                last_err = f"Command not found: {cmd}"
                continue
            except subprocess.TimeoutExpired:
                last_err = "Conversion timed out (90s)"
                continue
            except Exception as e:
                last_err = str(e)[:200]
                continue
        return (None, last_err or "LibreOffice not found. Install LibreOffice and add it to PATH, or on Mac install from https://www.libreoffice.org")
    except Exception as e:
        return (None, str(e)[:200])
    finally:
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Mistral OCR 3
# ---------------------------------------------------------------------------
MISTRAL_COST_PER_PAGE = 0.002  # $2/1000 pages

def run_mistral_ocr(pdf_bytes: bytes, filename: str, api_key: str) -> dict:
    """Returns { "text", "cost_usd", "time_seconds", "error", "api_summary", "raw_api_response" }."""
    from mistralai import Mistral
    out = {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": None, "api_summary": None, "raw_api_response": None}
    t0 = time.perf_counter()
    try:
        client = Mistral(api_key=api_key)
        uploaded = client.files.upload(
            file={"file_name": filename, "content": pdf_bytes},
            purpose="ocr",
        )
        url = client.files.get_signed_url(file_id=uploaded.id).url
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": url},
        )
        out["raw_api_response"] = _raw_response_to_json(response)
        parts = []
        if hasattr(response, "pages"):
            for p in response.pages:
                parts.append(getattr(p, "markdown", "") or str(p))
        out["text"] = "\n\n".join(parts).strip()
        pages = pdf_page_count(pdf_bytes) or 1
        out["cost_usd"] = pages * MISTRAL_COST_PER_PAGE
        out["api_summary"] = {
            "provider": "Mistral",
            "model": "mistral-ocr-latest",
            "cost_usd": out["cost_usd"],
            "cost_note": "Estimated: $2/1000 pages (list price).",
            "input_pages": pages,
            "output_chars": len(out["text"]),
        }
    except Exception as e:
        out["error"] = str(e)
    out["time_seconds"] = time.perf_counter() - t0
    if out["api_summary"]:
        out["api_summary"]["time_seconds"] = out["time_seconds"]
        out["api_summary"]["speed_note"] = "Wall-clock time from request start to response."
    return out

# ---------------------------------------------------------------------------
# OpenAI Vision OCR (Chat Completions gpt-4o or Responses API gpt-4.1-mini)
# ---------------------------------------------------------------------------
OPENAI_GPT4O_INPUT_PER_1M = 2.50
OPENAI_GPT4O_OUTPUT_PER_1M = 10.0
OPENAI_GPT41MINI_INPUT_PER_1M = 0.40   # approximate
OPENAI_GPT41MINI_OUTPUT_PER_1M = 1.60  # approximate

def run_openai_vision_ocr(pdf_bytes: bytes, api_key: str, use_fast_model: bool = False) -> dict:
    """Returns { "text", "cost_usd", "time_seconds", "error", "api_summary", "raw_api_response" }. Uses Chat Completions (gpt-4o-mini or gpt-4o)."""
    out = {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": None, "api_summary": None, "raw_api_response": None}
    t0 = time.perf_counter()
    try:
        images = pdf_to_images(pdf_bytes)
        if not images:
            out["error"] = "No pages in PDF"
            out["time_seconds"] = time.perf_counter() - t0
            return out
        client = OpenAI(api_key=api_key)
        all_text = []
        total_input = 0
        total_output = 0
        raw_responses = []
        prompt = "Extract all visible text from this document image. Preserve order and structure. Output only the extracted text, no commentary."
        model = "gpt-4o-mini" if use_fast_model else "gpt-4o"
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ]
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
            )
            raw_responses.append(_raw_response_to_json(resp))
            if resp.choices and resp.choices[0].message.content:
                all_text.append(resp.choices[0].message.content.strip())
            if getattr(resp, "usage", None):
                total_input += getattr(resp.usage, "prompt_tokens", 0) or 0
                total_output += getattr(resp.usage, "completion_tokens", 0) or 0
        out["text"] = "\n\n".join(all_text).strip()
        out["raw_api_response"] = raw_responses if len(raw_responses) != 1 else raw_responses[0]
        if use_fast_model:
            out["cost_usd"] = (total_input / 1e6) * OPENAI_GPT41MINI_INPUT_PER_1M + (total_output / 1e6) * OPENAI_GPT41MINI_OUTPUT_PER_1M
        else:
            out["cost_usd"] = (total_input / 1e6) * OPENAI_GPT4O_INPUT_PER_1M + (total_output / 1e6) * OPENAI_GPT4O_OUTPUT_PER_1M
        out["api_summary"] = {
            "provider": "OpenAI",
            "model": model,
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
            "cost_usd": out["cost_usd"],
            "cost_note": "From API usage: input/output token counts × published price per 1M tokens.",
            "input_pages": len(images),
            "output_chars": len(out["text"]),
            "time_seconds": time.perf_counter() - t0,
            "speed_note": "Wall-clock time from request start to response.",
        }
    except Exception as e:
        out["error"] = str(e)
    out["time_seconds"] = time.perf_counter() - t0
    return out

# ---------------------------------------------------------------------------
# Replicate OCR models
# ---------------------------------------------------------------------------
# Why these 3:
# - text-extract-ocr: cheap baseline (~$0.0001/run), plain text, 90M+ runs.
# - deepseek-ocr: strong on tables/layouts/markdown (~$0.006/run), good for CVs.
# - marker: PDF→markdown/JSON, complex layouts (~$0.01/run), optional.
REPLICATE_COST_TEXT_EXTRACT = 0.0001
REPLICATE_COST_DEEPSEEK_OCR = 0.006
REPLICATE_COST_MARKER = 0.01

# Replicate: single request per model; no retries (user can send multiple in parallel).
# API allows wait between 1 and 60 only; cannot use >60 for slow models like deepseek-ocr.
REPLICATE_WAIT_SECONDS = 60
# HTTP read timeout must be >= wait; default client uses 30s and causes "The read operation timed out" for slow models.
REPLICATE_HTTP_READ_TIMEOUT = 120.0

def _replicate_run(model: str, input_dict: dict, api_token: str, cost_usd: float = 0.0) -> dict:
    """Run Replicate model once; returns { "text", "cost_usd", "time_seconds", "error", "api_summary", "raw_api_response" }. No retries—show result or error as soon as it returns."""
    wait = REPLICATE_WAIT_SECONDS
    out = {"text": "", "cost_usd": cost_usd, "time_seconds": 0.0, "error": None, "api_summary": None, "raw_api_response": None}
    t0 = time.perf_counter()
    prev = os.environ.get("REPLICATE_API_TOKEN")
    try:
        os.environ["REPLICATE_API_TOKEN"] = api_token
        import replicate
        import httpx
        client = replicate.Client(
            api_token=api_token,
            timeout=httpx.Timeout(REPLICATE_HTTP_READ_TIMEOUT, read=REPLICATE_HTTP_READ_TIMEOUT),
        )
        raw = client.run(model, input=input_dict, wait=wait)
        out["raw_api_response"] = _raw_response_to_json(raw)
        if isinstance(raw, str):
            out["text"] = raw.strip()
        elif isinstance(raw, dict) and "markdown" in raw:
            out["text"] = (raw["markdown"] or "").strip()
        elif isinstance(raw, (list, tuple)):
            out["text"] = "\n".join(str(x) for x in raw).strip()
        elif hasattr(raw, "output") and isinstance(getattr(raw, "output"), str):
            out["text"] = getattr(raw, "output").strip()
        elif hasattr(raw, "markdown"):
            out["text"] = (getattr(raw, "markdown") or "").strip()
        else:
            out["text"] = str(raw).strip()
    except Exception as e:
        out["error"] = str(e).strip()
    finally:
        if prev is None:
            os.environ.pop("REPLICATE_API_TOKEN", None)
        else:
            os.environ["REPLICATE_API_TOKEN"] = prev
    out["time_seconds"] = time.perf_counter() - t0
    out["api_summary"] = {
        "provider": "Replicate",
        "model": model,
        "cost_usd": out["cost_usd"],
        "cost_note": "Fixed estimate per run from Replicate list price (not from API response).",
        "output_chars": len(out["text"]),
        "time_seconds": out["time_seconds"],
        "speed_note": "Wall-clock time from request start to response.",
    }
    return out

# Replicate data URI limit (1MB); we target ~700KB to leave margin
REPLICATE_DATA_URI_MAX = 1_000_000
REPLICATE_IMAGE_TARGET_BYTES = 700_000

def _replicate_image_data_uri_under_limit(image_bytes: bytes) -> Optional[str]:
    """Build a data URI for Replicate under 1MB by resizing/compressing the image if needed."""
    data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")
    if len(data_uri) <= REPLICATE_DATA_URI_MAX:
        return data_uri
    try:
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    for scale in [0.7, 0.5, 0.35, 0.25, 0.2, 0.15]:
        nw, nh = int(w * scale), int(h * scale)
        if nw < 100 or nh < 100:
            break
        out = io.BytesIO()
        img.resize((nw, nh), Image.Resampling.LANCZOS).save(out, format="PNG", optimize=True)
        if out.tell() <= REPLICATE_IMAGE_TARGET_BYTES:
            data_uri = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("utf-8")
            if len(data_uri) <= REPLICATE_DATA_URI_MAX:
                return data_uri
    for quality in [85, 70, 55, 40]:
        out = io.BytesIO()
        img.resize((int(w * 0.5), int(h * 0.5)), Image.Resampling.LANCZOS).save(out, format="JPEG", quality=quality, optimize=True)
        if out.tell() <= REPLICATE_IMAGE_TARGET_BYTES:
            data_uri = "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("utf-8")
            if len(data_uri) <= REPLICATE_DATA_URI_MAX:
                return data_uri
    return None

# Versioned model IDs (required by Replicate API; unversioned can return 404)
REPLICATE_TEXT_EXTRACT_OCR = "abiruyt/text-extract-ocr:a524caeaa23495bc9edc805ab08ab5fe943afd3febed884a4f3747aa32e9cd61"

def run_replicate_text_extract_ocr(pdf_bytes: bytes, api_token: str) -> dict:
    """abiruyt/text-extract-ocr: image → plain text. Uses first page; resized if needed to stay under 1MB data URI."""
    images = pdf_to_images(pdf_bytes)
    if not images:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "No pages in PDF"}
    data_uri = _replicate_image_data_uri_under_limit(images[0])
    if not data_uri:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "First page image could not be compressed under 1MB. Install Pillow (pip install Pillow) or try a smaller PDF."}
    input_dict = {"image": data_uri}
    return _replicate_run(REPLICATE_TEXT_EXTRACT_OCR, input_dict, api_token, REPLICATE_COST_TEXT_EXTRACT)

# Versioned model ID from Replicate (stable)
REPLICATE_DEEPSEEK_OCR_VERSION = "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5"

def run_replicate_deepseek_ocr(pdf_bytes: bytes, api_token: str) -> dict:
    """lucataco/deepseek-ocr: image → markdown. Uses first page; resized if needed to stay under 1MB data URI."""
    images = pdf_to_images(pdf_bytes)
    if not images:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "No pages in PDF"}
    data_uri = _replicate_image_data_uri_under_limit(images[0])
    if not data_uri:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "First page image could not be compressed under 1MB. Install Pillow (pip install Pillow) or try a smaller PDF."}
    input_dict = {"image": data_uri, "task_type": "Convert to Markdown"}
    return _replicate_run(REPLICATE_DEEPSEEK_OCR_VERSION, input_dict, api_token, REPLICATE_COST_DEEPSEEK_OCR)

def run_replicate_marker(pdf_bytes: bytes, api_token: str) -> dict:
    """datalab-to/marker: PDF → markdown. Full document."""
    input_dict = {"file": io.BytesIO(pdf_bytes), "mode": "fast"}
    return _replicate_run("datalab-to/marker", input_dict, api_token, REPLICATE_COST_MARKER)

# ---------------------------------------------------------------------------
# OCR models (for selector: id -> label; build tasks from selected ids)
# ---------------------------------------------------------------------------
def _ocr_models_available(has_replicate: bool) -> list[tuple[str, str]]:
    """Return list of (id, display_label) for the multiselect. Short labels avoid truncation in chips."""
    out = [
        ("mistral-ocr-3", "Mistral OCR 3"),
        ("openai-gpt4o-mini", "OpenAI GPT-4o-mini"),
        ("openai-gpt4o", "OpenAI GPT-4o"),
    ]
    if has_replicate:
        out.extend([
            ("replicate-text-extract", "text-extract-ocr"),
            ("replicate-deepseek", "deepseek-ocr"),
            ("replicate-marker", "marker"),
        ])
    return out


def _ocr_build_tasks(selected_ids: list[str], pdf_bytes: bytes, filename: str, openai_key: str, mistral_key: str, replicate_key: Optional[str]) -> list:
    """Build list of (name, fn, args) for selected model ids."""
    tasks = []
    for mid in selected_ids:
        if mid == "mistral-ocr-3":
            tasks.append(("Mistral OCR 3", run_mistral_ocr, (pdf_bytes, filename, mistral_key)))
        elif mid == "openai-gpt4o-mini":
            tasks.append(("OpenAI GPT-4o-mini Vision", run_openai_vision_ocr, (pdf_bytes, openai_key, True)))
        elif mid == "openai-gpt4o":
            tasks.append(("OpenAI GPT-4o Vision", run_openai_vision_ocr, (pdf_bytes, openai_key, False)))
        elif mid == "replicate-text-extract" and replicate_key:
            tasks.append(("Replicate text-extract-ocr", run_replicate_text_extract_ocr, (pdf_bytes, replicate_key)))
        elif mid == "replicate-deepseek" and replicate_key:
            tasks.append(("Replicate deepseek-ocr", run_replicate_deepseek_ocr, (pdf_bytes, replicate_key)))
        elif mid == "replicate-marker" and replicate_key:
            tasks.append(("Replicate marker", run_replicate_marker, (pdf_bytes, replicate_key)))
    return tasks


# ---------------------------------------------------------------------------
# Multi-criteria: Borda and Condorcet
# ---------------------------------------------------------------------------
def borda_rank(scores_per_criterion: list[list[float]], weights: list[float]) -> list[float]:
    """
    scores_per_criterion: list of length n_criteria; each element is list of scores (one per model), higher = better.
    weights: length n_criteria.
    Returns Borda score per model (weighted).
    For each criterion: rank 1 gets (n_models-1) points, rank 2 gets (n_models-2), etc. Then multiply by weight and sum.
    """
    n_models = len(scores_per_criterion[0])
    n_crit = len(scores_per_criterion)
    points = [0.0] * n_models
    for c in range(n_crit):
        w = weights[c] if c < len(weights) else 1.0
        # Order indices by score descending (highest first)
        order = sorted(range(n_models), key=lambda i: scores_per_criterion[c][i], reverse=True)
        for rank, idx in enumerate(order):
            points[idx] += (n_models - 1 - rank) * w
    return points

def condorcet_wins(scores_per_criterion: list[list[float]], higher_better: list[bool]) -> list[int]:
    """
    For each model, count how many criteria it wins (best on that criterion).
    higher_better[i] = True if higher score is better for criterion i.
    Returns list of win counts per model (criteria where this model is best).
    """
    n_models = len(scores_per_criterion[0])
    n_crit = len(scores_per_criterion)
    wins = [0] * n_models
    for c in range(n_crit):
        vals = scores_per_criterion[c]
        hb = higher_better[c] if c < len(higher_better) else True
        best = max(vals) if hb else min(vals)
        for i in range(n_models):
            if vals[i] == best:
                wins[i] += 1
    return wins

# ---------------------------------------------------------------------------
# UI: one model card (metrics + text + API expander)
# ---------------------------------------------------------------------------
def render_model_result(name: str, m: dict, r: dict, key_prefix: str) -> None:
    """Render one column: Raw API response expander, then metrics (CER, WER, etc.), then text areas."""
    if r.get("error"):
        st.error(r.get("error", "Unknown error"))
        return
    # 1) Raw API response first (what the API actually returned)
    raw_api_response = r.get("raw_api_response")
    if raw_api_response is not None:
        with st.expander("Raw API response"):
            if isinstance(raw_api_response, (dict, list)):
                st.json(raw_api_response)
            else:
                # Replicate (and others) can return plain text or non-JSON; show as code block (read-only, not grayed out)
                st.code(str(raw_api_response), language=None)
    # 2) Metrics shown directly (no expander)
    st.metric("CER", f"{m['cer_pct']:.1f}%")
    st.metric("WER", f"{m['wer_pct']:.1f}%")
    st.metric("Layout accuracy", f"{m['layout_accuracy_pct']:.1f}%")
    speed_str = "≥60 (timed out)" if (m.get("error") and "timed out" in (m.get("error") or "").lower()) else f"{m['time_seconds']:.2f} s"
    st.metric("Speed", speed_str)
    st.metric("Cost", f"${m['cost_usd']:.4f}")
    # 3) Extracted text (and missing words) in expander
    with st.expander("Extracted text"):
        if m.get("missing_words"):
            missing_str = " ".join(m["missing_words"][:200]) + (" …" if len(m["missing_words"]) > 200 else "")
            st.caption("Missing words (in GT, not in output)")
            st.code(missing_str, language=None)
        extracted = (m.get("text") or "")[:8000] + ("…" if len(m.get("text") or "") > 8000 else "")
        st.code(extracted, language=None)

# ---------------------------------------------------------------------------
# Shared state for progressive results (threads write here; survives st.rerun())
# ---------------------------------------------------------------------------
def _ocr_thread_target(idx: int, fn, args: tuple, results_list: list) -> None:
    """Run fn(*args) and store result in results_list[idx]."""
    try:
        results_list[idx] = fn(*args)
    except Exception as e:
        results_list[idx] = {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": str(e), "api_summary": None}

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("OCR Benchmark: Mistral, OpenAI Vision, Replicate")

openai_key = get_openai_key()
mistral_key = get_mistral_key()
replicate_key = get_replicate_key()
if not openai_key:
    st.error("Missing OPENAI_API_KEY in `.streamlit/secrets.toml`.")
if not mistral_key:
    st.error("Missing MISTRAL_API_KEY in `.streamlit/secrets.toml`.")
st.caption("All OCR requests are sent in parallel (one thread per model). No model waits for another. Each column keeps loading until that model returns—success or error—then updates immediately.")

# Model selector (like LLM Parsing Benchmark)
ocr_models_options = _ocr_models_available(replicate_key is not None and bool(replicate_key))
ocr_default = [m[0] for m in ocr_models_options if m[0] != "openai-gpt4o"]
if not ocr_default:
    ocr_default = [m[0] for m in ocr_models_options]
selected_ocr_models = st.multiselect(
    "Models to run (in parallel, 1 thread per model)",
    options=[m[0] for m in ocr_models_options],
    default=ocr_default,
    format_func=lambda x: next((m[1] for m in ocr_models_options if m[0] == x), x),
    key="ocr_models_select",
)

with st.expander("Metrics & multi-criteria weights (Borda / ranking)"):
    st.markdown("""
| Metric | Meaning | How we get it |
| --- | --- | --- |
| **CER** (Character Error Rate) | % of character-level errors | Levenshtein distance (ref vs pred characters) / ref length × 100. Lower is better. |
| **WER** (Word Error Rate) | % of incorrect words | Word-level edit distance / ref word count × 100. Lower is better. |
| **Layout accuracy** | Ability to keep sections | % of GT section headers found in output. |
| **Speed** | Time waited | Wall-clock seconds from request start to response. |
| **Cost** | Price of the run | **OpenAI/Mistral:** from API usage (tokens or pages × list price). **Replicate:** fixed estimate per run (list price). |
    """)
    st.markdown("**Weights for ranking (all 5 metrics):**")
    w_cer = st.slider("Weight: CER (lower is better)", 0.0, 1.0, 0.20, 0.05, key="w_cer")
    w_wer = st.slider("Weight: WER (lower is better)", 0.0, 1.0, 0.20, 0.05, key="w_wer")
    w_layout = st.slider("Weight: Layout accuracy (higher is better)", 0.0, 1.0, 0.20, 0.05, key="w_layout")
    w_speed = st.slider("Weight: Speed (faster is better)", 0.0, 1.0, 0.20, 0.05, key="w_speed")
    w_cost = st.slider("Weight: Cost (lower is better)", 0.0, 1.0, 0.20, 0.05, key="w_cost")
    total_w = w_cer + w_wer + w_layout + w_speed + w_cost
    if total_w <= 0:
        total_w = 1.0
    st.caption(f"Sum = {total_w:.2f}. Borda uses these weights; Condorcet uses majority of criteria.")

input_mode = st.radio(
    "Input",
    options=["Upload PDF", "Use from library"],
    horizontal=True,
    key="ocr_input_mode",
)

uploaded_file = None
ground_truth = ""
pdf_bytes = None
filename = None
library_row = None

if input_mode == "Upload PDF":
    col_left, col_right = st.columns(2)
    with col_left:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf")
    with col_right:
        ground_truth = st.text_area(
            "Ground truth (perfect OCR text)",
            height=120,
            placeholder="Paste the reference text here (exact words to compare against).",
            key="gt",
        )
    if uploaded_file:
        pdf_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name or "document.pdf"
else:
    # Use from library (same CVtheque as our_database.py)
    db_metadata = load_db_metadata()
    if not db_metadata:
        st.warning("No CV library found (ground_truth_database/metadata.csv). Use **Upload PDF** or add the database.")
    else:
        options = [f"{r['filename']} ({r.get('language', '')}, {r.get('layout_type', '')})" for r in db_metadata]
        choice = st.selectbox("Select a CV from the library", options=options, key="library_cv")
        if choice:
            idx = options.index(choice)
            library_row = db_metadata[idx]
            filename = library_row["filename"]
            base, ext = os.path.splitext(filename)
            base = base.strip()
            ext = ext.lower().lstrip(".")
            doc_path = os.path.join(CV_DIR, filename)
            txt_path = os.path.join(PARSING_TXT_DIR, f"{base}.txt")
            shot_path = _screenshot_path(base)

            # Show display + transcription (same as our_database modal)
            st.subheader("Document & transcription")
            col_doc, col_txt = st.columns([1, 1])
            with col_doc:
                st.caption("Document")
                if os.path.isfile(shot_path):
                    img_path = shot_path
                elif os.path.isfile(doc_path) and ext in ("png", "jpg", "jpeg"):
                    img_path = doc_path
                else:
                    img_path = None
                if img_path:
                    mime = "image/jpeg" if img_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    st.markdown(
                        f'<div style="max-width: 320px;"><img src="data:{mime};base64,{b64}" style="width: 100%; height: auto;" alt="CV" /></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(f"No preview for {filename}")
            with col_txt:
                st.caption("Transcription (ground truth)")
                if os.path.isfile(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        ground_truth = f.read()
                    # Key per CV so changing selection updates the text area
                    st.text_area("Ground truth", ground_truth, height=400, disabled=False, key=f"gt_library_{base}")
                else:
                    st.info(f"No {base}.txt in ground_truth_database/parsed.")
                    ground_truth = ""

            # OCR benchmark: PDF directly; DOCX → PDF (LibreOffice); PNG/JPG → single-page PDF (PyMuPDF)
            if ext == "pdf" and os.path.isfile(doc_path):
                try:
                    with open(doc_path, "rb") as f:
                        pdf_bytes = f.read()
                except Exception:
                    pdf_bytes = None
                    st.error(f"Could not read {filename}.")
            elif ext == "docx" and os.path.isfile(doc_path):
                try:
                    with open(doc_path, "rb") as f:
                        docx_bytes = f.read()
                    pdf_bytes, conv_err = _convert_docx_to_pdf(docx_bytes)
                    if pdf_bytes is not None:
                        filename = f"{base}.pdf"  # OCR APIs receive PDF content with .pdf name
                    elif os.path.isfile(shot_path):
                        # Fallback: use library screenshot (no LibreOffice needed)
                        with open(shot_path, "rb") as f:
                            image_bytes = f.read()
                        pdf_bytes = _image_to_pdf(image_bytes, filetype="png")
                        if pdf_bytes is not None:
                            filename = f"{base}.pdf"
                            st.caption("Using library screenshot for this DOCX (no LibreOffice needed).")
                        else:
                            st.error("DOCX → PDF conversion failed: " + (conv_err or "Unknown error"))
                            st.caption("Screenshot fallback also failed. Install **LibreOffice** (e.g. Mac: `brew install --cask libreoffice`) for DOCX support.")
                    else:
                        st.error("DOCX → PDF conversion failed: " + (conv_err or "Unknown error"))
                        st.caption("To use DOCX: install **LibreOffice** (e.g. Mac: `brew install --cask libreoffice`, or download from libreoffice.org). Or add a screenshot for this CV in `ground_truth_database/screenshots/{base}.png` for a fallback.")
                except Exception as e:
                    pdf_bytes = None
                    st.error(f"Could not read or convert {filename}: {e}")
            elif ext in ("png", "jpg", "jpeg") and os.path.isfile(doc_path):
                try:
                    with open(doc_path, "rb") as f:
                        image_bytes = f.read()
                    filetype = "png" if ext == "png" else "jpeg"
                    pdf_bytes = _image_to_pdf(image_bytes, filetype=filetype)
                    if pdf_bytes is None:
                        st.error(f"Could not convert {filename} (image) to PDF.")
                    else:
                        filename = f"{base}.pdf"
                except Exception as e:
                    pdf_bytes = None
                    st.error(f"Could not read or convert {filename}: {e}")
            else:
                pdf_bytes = None
                st.caption(f"OCR benchmark supports **PDF**, **DOCX**, and **PNG/JPG** from the library. This CV is **{ext}**. Choose a supported file or upload a PDF.")

analyze = st.button("Analyze", type="primary")

if analyze and openai_key and mistral_key:
    if input_mode == "Use from library":
        # Same key as the text area (per-CV) so we read the current selection's value
        library_choice = st.session_state.get("library_cv", "")
        lib_base = os.path.splitext((library_choice.split(" ")[0] or "").strip())[0] if library_choice else ""
        ground_truth = st.session_state.get(f"gt_library_{lib_base}", ground_truth or "")
    if not selected_ocr_models:
        st.warning("Select at least one OCR model to run.")
    elif pdf_bytes is None or not filename:
        st.warning("Provide a PDF, or select a PDF / DOCX / PNG / JPG from the library (or upload a PDF).")
    elif not (ground_truth and ground_truth.strip()):
        st.warning("Ground truth is required. Paste it (Upload mode) or select a CV that has a transcription in the library.")
    else:
        tasks = _ocr_build_tasks(selected_ocr_models, pdf_bytes, filename, openai_key, mistral_key, replicate_key)
        if not tasks:
            st.warning("No valid tasks for selected models (Replicate models require REPLICATE_API_KEY).")
        else:
            models = [t[0] for t in tasks]
            n_tasks = len(tasks)
            results = [None] * n_tasks
            threads = []
            for i, (name, fn, args) in enumerate(tasks):
                t = Thread(target=_ocr_thread_target, args=(i, fn, args, results))
                t.daemon = True
                t.start()
                threads.append(t)
            st.subheader("Results")
            st.caption("Running all models in parallel. Waiting for API responses…")
            with st.spinner("Waiting for API response…"):
                for t in threads:
                    t.join()
            metrics_list = []
            for i, (name, r) in enumerate(zip(models, results)):
                m = word_metrics(ground_truth, r.get("text", "") or "")
                m["name"] = name
                m["cost_usd"] = r.get("cost_usd", 0.0)
                m["time_seconds"] = r.get("time_seconds", 0.0)
                m["error"] = r.get("error")
                m["text"] = r.get("text", "")
                metrics_list.append(m)
            st.session_state["ocr_results"] = {
                "models": models,
                "results": results,
                "metrics_list": metrics_list,
            }
            st.rerun()

else:
    # Show persisted results (summary table + per-model cards + Borda/Condorcet)
    if "ocr_results" in st.session_state:
        data = st.session_state["ocr_results"]
        st.subheader("Results")
        # Summary table (like LLM Parsing Benchmark)
        table_rows = []
        for i, name in enumerate(data["models"]):
            m = data["metrics_list"][i]
            is_timeout = m.get("error") and "timed out" in (m.get("error") or "").lower()
            time_display = "≥60 (timed out)" if is_timeout else f"{m['time_seconds']:.2f}"
            table_rows.append({
                "Model": name,
                "Time (s)": time_display,
                "Cost ($)": f"{m['cost_usd']:.4f}",
                "CER (%)": f"{m['cer_pct']:.1f}",
                "WER (%)": f"{m['wer_pct']:.1f}",
                "Layout (%)": f"{m['layout_accuracy_pct']:.1f}",
                "Error": m.get("error") or "—",
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        # Per-model cards (metrics + raw API + extracted text)
        n_cols = min(5, len(data["models"]))
        cols = st.columns(n_cols)
        for i in range(len(data["models"])):
            with cols[i]:
                st.markdown(f"### {data['models'][i]}")
                render_model_result(data["models"][i], data["metrics_list"][i], data["results"][i], "res_%s_%s" % (data["models"][i], i))
        st.divider()
        st.subheader("Best model (Borda / Condorcet)")
        metrics_list = data["metrics_list"]
        valid = [m for m in metrics_list if m["error"] is None]
        if len(valid) >= 2:
            total_w = st.session_state.get("w_cer", 0.2) + st.session_state.get("w_wer", 0.2) + st.session_state.get("w_layout", 0.2) + st.session_state.get("w_speed", 0.2) + st.session_state.get("w_cost", 0.2)
            if total_w <= 0:
                total_w = 1.0
            w_cer, w_wer = st.session_state.get("w_cer", 0.2), st.session_state.get("w_wer", 0.2)
            w_layout, w_speed, w_cost = st.session_state.get("w_layout", 0.2), st.session_state.get("w_speed", 0.2), st.session_state.get("w_cost", 0.2)
            weights_norm = [w_cer / total_w, w_wer / total_w, w_layout / total_w, w_speed / total_w, w_cost / total_w]
            cer_scores = [-m["cer_pct"] for m in valid]
            wer_scores = [-m["wer_pct"] for m in valid]
            layout_scores = [m["layout_accuracy_pct"] for m in valid]
            speed_scores = [-m["time_seconds"] for m in valid]
            cost_scores = [-m["cost_usd"] for m in valid]
            scores_per_criterion = [cer_scores, wer_scores, layout_scores, speed_scores, cost_scores]
            higher_better = [True, True, True, True, True]
            borda_scores = borda_rank(scores_per_criterion, weights_norm)
            condorcet_counts = condorcet_wins(scores_per_criterion, higher_better)
            names_valid = [m["name"] for m in valid]
            best_borda_idx = max(range(len(borda_scores)), key=lambda i: borda_scores[i])
            best_condorcet_idx = max(range(len(condorcet_counts)), key=lambda i: condorcet_counts[i])
            best_borda_name = names_valid[best_borda_idx]
            best_condorcet_name = names_valid[best_condorcet_idx]
            st.markdown("**Borda (weighted ranking)**")
            st.markdown(f"Best model: **{best_borda_name}** (score: {borda_scores[best_borda_idx]:.2f}).")
            st.markdown("Scores by model (higher = better):")
            for i in range(len(valid)):
                st.markdown(f"- {names_valid[i]}: **{borda_scores[i]:.2f}**")
            st.caption("Borda: each criterion contributes weighted points by rank; scores are summed.")
            st.markdown("---")
            st.markdown("**Condorcet (majority of criteria)**")
            st.markdown(f"Best model: **{best_condorcet_name}** (won {condorcet_counts[best_condorcet_idx]} of 5 criteria).")
            st.markdown("Criteria won per model (each of CER, WER, Layout, Speed, Cost):")
            for i in range(len(valid)):
                st.markdown(f"- {names_valid[i]}: **{condorcet_counts[i]}** criteria")
            st.caption("Condorcet: for each criterion the best model gets a win; the model that wins the most criteria is the Condorcet winner.")
            st.markdown("---")
            st.markdown("**Analysis**")
            if best_borda_name == best_condorcet_name:
                st.info(f"**{best_borda_name}** is both the Borda and Condorcet winner: it ranks first on weighted score and wins the majority of criteria. It is the recommended choice for this document and weights.")
            else:
                st.info(f"**Borda** favours **{best_borda_name}** (best weighted overall score). **Condorcet** favours **{best_condorcet_name}** (won most criteria). Use Borda for a single balanced recommendation, or Condorcet if you prefer the model that wins the most individual metrics.")
        else:
            st.info("Need at least two successful runs to compare.")
