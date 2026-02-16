"""
Shared OCR logic for OCR Benchmark and Parallel OCR Test.
No UI here; pages import and use these helpers.
"""
import base64
import csv
import io
import os
import re
import time
from typing import Optional

# Optional Streamlit for secrets only (so parallel_ocr_test can run without full UI deps on import)
try:
    import streamlit as st
except Exception:
    st = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(_ROOT, "ground_truth_database")
CV_DIR = os.path.join(DB_DIR, "cv")  # CV documents (pdf, docx, etc.)
PARSING_TXT_DIR = os.path.join(DB_DIR, "parsed")  # transcription .txt files
PARSING_DIR = os.path.join(DB_DIR, "json_parsed")  # ground truth JSON
SCREENSHOTS_DIR = os.path.join(DB_DIR, "screenshots")
METADATA_PATH = os.path.join(DB_DIR, "metadata.csv")


def load_db_metadata() -> list[dict]:
    """Load CV library metadata (filename, extension, language, layout_type, ...). No cache."""
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


def get_cv_pdf_and_ground_truth(metadata_row: dict) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Resolve PDF bytes and ground truth for one CV from metadata.
    Returns (pdf_bytes, filename_for_api, ground_truth) or (None, None, None) if unavailable.
    """
    filename = metadata_row.get("filename", "")
    base, ext = os.path.splitext(filename)
    base = base.strip()
    ext = (ext or "").lower().lstrip(".")
    doc_path = os.path.join(CV_DIR, filename)
    txt_path = os.path.join(PARSING_TXT_DIR, f"{base}.txt")
    shot_path = _screenshot_path(base)

    ground_truth = ""
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read()
    if not (ground_truth and ground_truth.strip()):
        return (None, None, None)

    api_filename = f"{base}.pdf"
    if ext == "pdf" and os.path.isfile(doc_path):
        try:
            with open(doc_path, "rb") as f:
                return (f.read(), api_filename, ground_truth)
        except Exception:
            return (None, None, None)

    if ext == "docx" and os.path.isfile(doc_path):
        try:
            with open(doc_path, "rb") as f:
                docx_bytes = f.read()
            pdf_bytes, _ = _convert_docx_to_pdf(docx_bytes)
            if pdf_bytes is not None:
                return (pdf_bytes, api_filename, ground_truth)
            if os.path.isfile(shot_path):
                with open(shot_path, "rb") as f:
                    image_bytes = f.read()
                pdf_bytes = _image_to_pdf(image_bytes, filetype="png")
                if pdf_bytes is not None:
                    return (pdf_bytes, api_filename, ground_truth)
        except Exception:
            pass
        return (None, None, None)

    if ext in ("png", "jpg", "jpeg") and os.path.isfile(doc_path):
        try:
            with open(doc_path, "rb") as f:
                image_bytes = f.read()
            filetype = "png" if ext == "png" else "jpeg"
            pdf_bytes = _image_to_pdf(image_bytes, filetype=filetype)
            if pdf_bytes is not None:
                return (pdf_bytes, api_filename, ground_truth)
        except Exception:
            pass
        return (None, None, None)

    return (None, None, None)


# ---------------------------------------------------------------------------
# Metrics (CER, WER, layout, word_metrics)
# ---------------------------------------------------------------------------
def normalize_words(text: str) -> list:
    if not (text and text.strip()):
        return []
    clean = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return [w for w in clean.split() if w]


def _edit_distance(ref: list, hyp: list) -> int:
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
    ref_chars = list((ground_truth or "").strip())
    if not ref_chars:
        return 0.0
    hyp_chars = list((predicted or "").strip())
    return (_edit_distance(ref_chars, hyp_chars) / len(ref_chars)) * 100.0


def wer(ground_truth: str, predicted: str) -> float:
    ref_words = normalize_words(ground_truth)
    if not ref_words:
        return 0.0
    hyp_words = normalize_words(predicted)
    return (_edit_distance(ref_words, hyp_words) / len(ref_words)) * 100.0


def layout_accuracy(ground_truth: str, predicted: str) -> float:
    pred_lower = (predicted or "").lower()
    lines = [ln.strip() for ln in (ground_truth or "").splitlines() if ln.strip()]
    section_headers = []
    for ln in lines:
        if len(ln) > 60:
            continue
        if "--" in ln or "—" in ln:
            continue
        letters = sum(1 for c in ln if c.isalpha())
        if letters < 2 or letters / max(len(ln), 1) < 0.4:
            continue
        if ln.isupper() or (ln and ln[0].isupper() and not ln.rstrip().endswith(",") and not ln.rstrip().endswith(".")):
            section_headers.append(ln.strip())
    section_headers = list(dict.fromkeys(section_headers))[:30]
    if not section_headers:
        return 100.0
    found = sum(1 for h in section_headers if h.lower() in pred_lower)
    return (found / len(section_headers)) * 100.0


def word_metrics(ground_truth_text: str, predicted_text: str) -> dict:
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
# PDF / images / DOCX
# ---------------------------------------------------------------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    try:
        import fitz
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


def _image_to_pdf(image_bytes: bytes, filetype: str = "png") -> Optional[bytes]:
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


def _convert_docx_to_pdf(docx_bytes: bytes) -> tuple[Optional[bytes], Optional[str]]:
    import subprocess
    import tempfile
    tmpdir = os.path.abspath(tempfile.mkdtemp())
    docx_path = os.path.join(tmpdir, "document.docx")
    pdf_path = os.path.join(tmpdir, "document.pdf")
    try:
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
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
                    last_err = (result.stderr or result.stdout or f"Exit {result.returncode}").strip()[:200]
                    continue
                if os.path.isfile(pdf_path):
                    with open(pdf_path, "rb") as f:
                        return (f.read(), None)
                last_err = "LibreOffice did not create document.pdf"
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                last_err = str(e)[:200]
                continue
        return (None, last_err or "LibreOffice not found")
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
MISTRAL_COST_PER_PAGE = 0.002


def _raw_response_to_json(obj):
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


def run_mistral_ocr(pdf_bytes: bytes, filename: str, api_key: str) -> dict:
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
        out["api_summary"] = {"provider": "Mistral", "model": "mistral-ocr-latest", "cost_usd": out["cost_usd"], "input_pages": pages, "output_chars": len(out["text"])}
    except Exception as e:
        out["error"] = str(e)
    out["time_seconds"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Replicate text-extract-ocr
# ---------------------------------------------------------------------------
# Why PDF to Mistral but images to Replicate?
# - Mistral OCR API accepts a document (PDF) directly, so we send pdf_bytes.
# - Replicate's abiruyt/text-extract-ocr accepts only a single image per call (no PDF).
# So we render the PDF to one image per page, call the API once per page, then
# concatenate the text so CER/WER are computed on the full document.
REPLICATE_COST_TEXT_EXTRACT = 0.0001
REPLICATE_WAIT_SECONDS = 60
REPLICATE_DATA_URI_MAX = 1_000_000
REPLICATE_IMAGE_TARGET_BYTES = 700_000
REPLICATE_TEXT_EXTRACT_OCR = "abiruyt/text-extract-ocr:a524caeaa23495bc9edc805ab08ab5fe943afd3febed884a4f3747aa32e9cd61"


def _replicate_run(model: str, input_dict: dict, api_token: str, cost_usd: float = 0.0) -> dict:
    out = {"text": "", "cost_usd": cost_usd, "time_seconds": 0.0, "error": None, "api_summary": None, "raw_api_response": None}
    t0 = time.perf_counter()
    prev = os.environ.get("REPLICATE_API_TOKEN")
    try:
        os.environ["REPLICATE_API_TOKEN"] = api_token
        import replicate
        raw = replicate.run(model, input=input_dict, wait=REPLICATE_WAIT_SECONDS)
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
    return out


def _replicate_image_data_uri_under_limit(image_bytes: bytes) -> Optional[str]:
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


def run_replicate_text_extract_ocr(pdf_bytes: bytes, api_token: str) -> dict:
    """Run Replicate text-extract-ocr on all pages (one API call per page), then concatenate.
    The model accepts only one image per call, so we send each page as an image for fair CER/WER."""
    images = pdf_to_images(pdf_bytes)
    if not images:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "No pages in PDF"}
    parts = []
    total_cost = 0.0
    total_time = 0.0
    first_error = None
    for i, img_bytes in enumerate(images):
        data_uri = _replicate_image_data_uri_under_limit(img_bytes)
        if not data_uri:
            parts.append(f"[page {i + 1} skipped: image could not be compressed under 1MB]")
            continue
        result = _replicate_run(
            REPLICATE_TEXT_EXTRACT_OCR,
            {"image": data_uri},
            api_token,
            REPLICATE_COST_TEXT_EXTRACT,
        )
        total_cost += result.get("cost_usd") or 0.0
        total_time += result.get("time_seconds") or 0.0
        if result.get("error"):
            if first_error is None:
                first_error = result["error"]
            parts.append(f"[page {i + 1} error: {result['error']}]")
        else:
            parts.append((result.get("text") or "").strip())
    text = "\n\n".join(parts).strip()
    return {
        "text": text,
        "cost_usd": total_cost,
        "time_seconds": total_time,
        "error": first_error if not text and first_error else None,
        "api_summary": None,
        "raw_api_response": None,
    }


# ---------------------------------------------------------------------------
# Secrets (for parallel_ocr_test; benchmark can use its own)
# ---------------------------------------------------------------------------
def get_mistral_key() -> Optional[str]:
    if st is None:
        return os.environ.get("MISTRAL_API_KEY")
    try:
        return st.secrets.get("MISTRAL_API_KEY") or ""
    except Exception:
        return os.environ.get("MISTRAL_API_KEY")


def get_replicate_key() -> Optional[str]:
    if st is None:
        return os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY")
    try:
        return st.secrets.get("REPLICATE_API_KEY") or st.secrets.get("REPLICATE_API_TOKEN") or ""
    except Exception:
        return os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY")
