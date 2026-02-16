"""
OCR Benchmark: Upload PDF + ground truth → parallel OCR (Mistral, OpenAI Vision, Replicate).
Metrics: word recall, CER, WER, layout accuracy, cost, speed. Cost from API usage or list price.
Multi-criteria: Borda count + Condorcet. Results cached locally by CV (filename + content hash).
"""
import base64
import hashlib
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
# Local cache (by CV: filename + content hash)
# ---------------------------------------------------------------------------
def _cache_dir():
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ocr_cache")
    os.makedirs(d, exist_ok=True)
    return d

def _cache_key(pdf_bytes: bytes, filename: str) -> str:
    """Stable key: content hash so same PDF reuses cache even if renamed."""
    h = hashlib.sha256(pdf_bytes).hexdigest()[:24]
    safe = re.sub(r"[^\w\-.]", "_", (filename or "document.pdf")[:80])
    return f"{safe}_{h}"

def _cache_path(cache_key: str) -> str:
    return os.path.join(_cache_dir(), cache_key + ".json")

def _raw_response_to_json(obj):
    """Turn API response objects into JSON-serializable dict/list/str for display and cache."""
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

def load_cached_results(pdf_bytes: bytes, filename: str) -> Optional[dict]:
    """Returns cached payload or None. Payload: { filename, content_hash, ground_truth?, results: { model_name: { text, cost_usd, time_seconds, error } }, models: [...] }."""
    key = _cache_key(pdf_bytes, filename)
    path = _cache_path(key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_cached_results(pdf_bytes: bytes, filename: str, ground_truth: str, models: list, results: list):
    """Save results keyed by this CV (filename + content hash)."""
    key = _cache_key(pdf_bytes, filename)
    path = _cache_path(key)
    def _serialize_result(r):
        d = {"text": r.get("text", ""), "cost_usd": r.get("cost_usd", 0.0), "time_seconds": r.get("time_seconds", 0.0), "error": r.get("error")}
        if r.get("api_summary") is not None:
            d["api_summary"] = r["api_summary"]
        if r.get("raw_api_response") is not None:
            d["raw_api_response"] = r["raw_api_response"]
        return d
    payload = {
        "filename": filename or "document.pdf",
        "content_hash": hashlib.sha256(pdf_bytes).hexdigest()[:24],
        "ground_truth": ground_truth,
        "models": models,
        "results": {name: _serialize_result(r) for name, r in zip(models, results)},
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

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
    """Layout accuracy: % of GT section headers found in prediction. Sections = lines that look like headers (all caps or short title)."""
    pred_lower = (predicted or "").lower()
    lines = [ln.strip() for ln in (ground_truth or "").splitlines() if ln.strip()]
    section_headers = []
    for ln in lines:
        if len(ln) > 60:
            continue
        # All caps or title-like (starts with capital, no trailing punctuation)
        if ln.isupper() or (ln and ln[0].isupper() and not ln.rstrip().endswith(("," or "."))):
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

# Replicate wait: max 60s per request. Retry on timeout (e.g. cold start).
REPLICATE_WAIT_SECONDS = 60
REPLICATE_MAX_RETRIES = 3
REPLICATE_RETRY_DELAY = 5

def _replicate_run(model: str, input_dict: dict, api_token: str, cost_usd: float = 0.0) -> dict:
    """Run Replicate model; returns { "text", "cost_usd", "time_seconds", "error", "api_summary", "raw_api_response" }. Retries on timeout."""
    out = {"text": "", "cost_usd": cost_usd, "time_seconds": 0.0, "error": None, "api_summary": None, "raw_api_response": None}
    t0 = time.perf_counter()
    prev = os.environ.get("REPLICATE_API_TOKEN")
    for attempt in range(REPLICATE_MAX_RETRIES):
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
            break
        except Exception as e:
            last_error = str(e).strip()
            is_timeout = "timed out" in last_error.lower() or "timeout" in last_error.lower()
            if is_timeout and attempt < REPLICATE_MAX_RETRIES - 1:
                time.sleep(REPLICATE_RETRY_DELAY)
                continue
            out["error"] = "Replicate timed out after %d tries." % REPLICATE_MAX_RETRIES if is_timeout else last_error
            break
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

def _replicate_image_data_uri(image_bytes: bytes) -> str:
    """Build data URI for Replicate (accepts data URI for images <1MB)."""
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

# Versioned model IDs (required by Replicate API; unversioned can return 404)
REPLICATE_TEXT_EXTRACT_OCR = "abiruyt/text-extract-ocr:a524caeaa23495bc9edc805ab08ab5fe943afd3febed884a4f3747aa32e9cd61"

def run_replicate_text_extract_ocr(pdf_bytes: bytes, api_token: str) -> dict:
    """abiruyt/text-extract-ocr: image → plain text. Uses first page; pass data URI (no upload needed)."""
    images = pdf_to_images(pdf_bytes)
    if not images:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "No pages in PDF"}
    data_uri = _replicate_image_data_uri(images[0])
    if len(data_uri) > 1_000_000:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "First page image too large for data URI (>1MB). Try a smaller PDF."}
    input_dict = {"image": data_uri}
    return _replicate_run(REPLICATE_TEXT_EXTRACT_OCR, input_dict, api_token, REPLICATE_COST_TEXT_EXTRACT)

# Versioned model ID from Replicate (stable)
REPLICATE_DEEPSEEK_OCR_VERSION = "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5"

def run_replicate_deepseek_ocr(pdf_bytes: bytes, api_token: str) -> dict:
    """lucataco/deepseek-ocr: image → markdown. Uses first page; pass data URI (no upload)."""
    images = pdf_to_images(pdf_bytes)
    if not images:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "No pages in PDF"}
    data_uri = _replicate_image_data_uri(images[0])
    if len(data_uri) > 1_000_000:
        return {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": "First page image too large for data URI (>1MB). Try a smaller PDF."}
    input_dict = {"image": data_uri, "task_type": "Convert to Markdown"}
    return _replicate_run(REPLICATE_DEEPSEEK_OCR_VERSION, input_dict, api_token, REPLICATE_COST_DEEPSEEK_OCR)

def run_replicate_marker(pdf_bytes: bytes, api_token: str) -> dict:
    """datalab-to/marker: PDF → markdown. Full document."""
    input_dict = {"file": io.BytesIO(pdf_bytes), "mode": "fast"}
    return _replicate_run("datalab-to/marker", input_dict, api_token, REPLICATE_COST_MARKER)

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
    st.metric("Speed", f"{m['time_seconds']:.2f} s")
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
if replicate_key:
    st.caption("All models run in parallel (one thread per model). Results appear as each finishes.")

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

force_rerun = st.checkbox("Force re-run (ignore cache)", value=False, help="Re-call all APIs instead of loading from local cache for this CV.")
analyze = st.button("Analyze", type="primary")

if analyze and openai_key and mistral_key:
    if not uploaded_file:
        st.warning("Upload a PDF first.")
    elif not (ground_truth and ground_truth.strip()):
        st.warning("Paste ground truth text to compute accuracy.")
    else:
        pdf_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name or "document.pdf"

        # Try cache first (same CV = same filename + content hash)
        loaded_from_cache = False
        if not force_rerun:
            cached = load_cached_results(pdf_bytes, filename)
            if cached and cached.get("models") and cached.get("results"):
                models = cached["models"]
                results = [cached["results"].get(name, {}) for name in models]
                loaded_from_cache = True
                st.info(f"Results loaded from local cache for: **{cached.get('filename', filename)}**. Check **Force re-run** to call APIs again.")

        if not loaded_from_cache:
            # Worker thread: runs OCR in its own thread, stores result on self (no st.* in thread)
            class OCRWorkerThread(Thread):
                def __init__(self, name, fn, args):
                    super().__init__()
                    self.name = name
                    self.fn = fn
                    self.args = args
                    self.return_value = None

                def run(self):
                    try:
                        self.return_value = self.fn(*self.args)
                    except Exception as e:
                        self.return_value = {"text": "", "cost_usd": 0.0, "time_seconds": 0.0, "error": str(e), "api_summary": None}

            # Build all tasks (each will run in its own thread)
            tasks = [
                ("Mistral OCR 3", run_mistral_ocr, (pdf_bytes, filename, mistral_key)),
                ("OpenAI GPT-4o-mini Vision", run_openai_vision_ocr, (pdf_bytes, openai_key, True)),
            ]
            if replicate_key:
                tasks.extend([
                    ("Replicate text-extract-ocr", run_replicate_text_extract_ocr, (pdf_bytes, replicate_key)),
                    ("Replicate deepseek-ocr", run_replicate_deepseek_ocr, (pdf_bytes, replicate_key)),
                    ("Replicate marker", run_replicate_marker, (pdf_bytes, replicate_key)),
                ])
            models = [t[0] for t in tasks]
            n_tasks = len(tasks)
            n_cols = min(5, n_tasks)
            cols = st.columns(n_cols)

            st.subheader("Results")
            # One placeholder per task (Streamlit pattern: create placeholders before starting threads)
            placeholders = []
            for i in range(n_tasks):
                with cols[i]:
                    st.markdown(f"### {models[i]}")
                    placeholders.append(st.empty())

            # Start one thread per task
            threads = [OCRWorkerThread(name, fn, args) for name, fn, args in tasks]
            for t in threads:
                t.start()

            # Poll from main script thread: each column gets its own spinner until its thread finishes
            results = [None] * n_tasks
            thread_done = [False] * n_tasks
            loop_count = 0
            while not all(thread_done):
                for i in range(n_tasks):
                    if not thread_done[i] and not threads[i].is_alive():
                        results[i] = threads[i].return_value
                        thread_done[i] = True
                    with placeholders[i]:
                        if results[i] is not None:
                            m = word_metrics(ground_truth, results[i].get("text", "") or "")
                            m["time_seconds"] = results[i].get("time_seconds", 0.0)
                            m["cost_usd"] = results[i].get("cost_usd", 0.0)
                            m["text"] = results[i].get("text", "")
                            render_model_result(models[i], m, results[i], "run_%s_%s" % (i, models[i]))
                        else:
                            with st.spinner(f"Running {models[i]}…"):
                                st.caption("API call in progress…")
                                time.sleep(0.1)  # brief pause so spinner is visible; avoid blocking UI
                loop_count += 1
                time.sleep(0.2)
            for t in threads:
                t.join()

            save_cached_results(pdf_bytes, filename, ground_truth, models, results)

        # Build metrics_list for Best model section
        metrics_list = []
        for i, (name, r) in enumerate(zip(models, results)):
            m = word_metrics(ground_truth, r.get("text", "") or "")
            m["name"] = name
            m["cost_usd"] = r.get("cost_usd", 0.0)
            m["time_seconds"] = r.get("time_seconds", 0.0)
            m["error"] = r.get("error")
            m["text"] = r.get("text", "")
            metrics_list.append(m)

        # Persist results so they stay visible after Streamlit reruns (button is True only once)
        st.session_state["ocr_results"] = {
            "models": models,
            "results": results,
            "metrics_list": metrics_list,
        }

        # When loaded from cache we didn't use placeholders; show Results here (metrics + raw API expander)
        if loaded_from_cache:
            st.subheader("Results")
            n_cols = min(5, len(models))
            cols = st.columns(n_cols)
            for i in range(len(models)):
                with cols[i]:
                    st.markdown(f"### {models[i]}")
                    render_model_result(models[i], metrics_list[i], results[i], "cached_%s_%s" % (models[i], i))

        # Dedicated section: Best model (Borda / Condorcet) with title, counts, and analysis
        st.divider()
        st.subheader("Best model (Borda / Condorcet)")
        valid = [m for m in metrics_list if m["error"] is None]
        if len(valid) >= 2:
            cer_scores = [-m["cer_pct"] for m in valid]
            wer_scores = [-m["wer_pct"] for m in valid]
            layout_scores = [m["layout_accuracy_pct"] for m in valid]
            speed_scores = [-m["time_seconds"] for m in valid]
            cost_scores = [-m["cost_usd"] for m in valid]
            weights_norm = [w_cer / total_w, w_wer / total_w, w_layout / total_w, w_speed / total_w, w_cost / total_w]
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

        if not replicate_key:
            st.caption("REPLICATE_API_KEY not set. Add it in secrets to run text-extract-ocr, deepseek-ocr, and marker.")

else:
    # Show persisted results (metrics + raw API expander) so they stay visible after reruns
    if "ocr_results" in st.session_state:
        data = st.session_state["ocr_results"]
        st.subheader("Results")
        n_cols = min(5, len(data["models"]))
        cols = st.columns(n_cols)
        for i in range(len(data["models"])):
            with cols[i]:
                st.markdown(f"### {data['models'][i]}")
                render_model_result(data["models"][i], data["metrics_list"][i], data["results"][i], "res_%s_%s" % (data["models"][i], i))
