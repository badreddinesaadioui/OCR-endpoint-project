"""
LLM Parsing Benchmark: OCR text → structured resume JSON.
Compare GPT-4.1 nano, GPT-5 mini (OpenAI), Gemini 2.5 Flash & Claude Haiku (Replicate).
Uses OPENAI_API_KEY and REPLICATE_API_KEY from .streamlit/secrets.toml (same as OCR Benchmark).
OpenAI: GPT-4.1 nano uses Responses API (client.responses.create); others use Chat Completions
  with structured output and refusal handling. Replicate: prompt + parse; input/output schemas available
  via model openapi_schema (Replicate API) if needed.
Runs selected models in parallel using threads (one thread per model).
Layout: OCR output (left) | Ground truth JSON (right). Results: accuracy vs ground truth, cost, speed.
"""
import json
import os
import time
from threading import Thread
from typing import Any, Optional

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Secrets (same as pages/2_OCR_Benchmark.py)
# ---------------------------------------------------------------------------
def _get_secret(key: str, alt_key: str = None):
    try:
        return st.secrets.get(key) or st.secrets.get(alt_key or "")
    except Exception:
        return None

def get_openai_key():
    return _get_secret("OPENAI_API_KEY")

def get_replicate_key():
    return _get_secret("REPLICATE_API_KEY") or _get_secret("REPLICATE_API_TOKEN")

# ---------------------------------------------------------------------------
# Resume extraction schema (strict JSON for validation)
# ---------------------------------------------------------------------------
RESUME_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "linkedin_url": {"type": ["string", "null"]},
        "name": {"type": "string"},
        "location": {"type": ["string", "null"]},
        "about": {"type": ["string", "null"]},
        "open_to_work": {"type": ["boolean", "null"]},
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position_title": {"type": "string"},
                    "institution_name": {"type": "string"},
                    "linkedin_url": {"type": ["string", "null"]},
                    "from_date": {"type": ["string", "null"]},
                    "to_date": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                },
                "required": [
                    "position_title", "institution_name", "linkedin_url",
                    "from_date", "to_date", "duration", "location", "description",
                ],
                "additionalProperties": False,
            },
        },
        "educations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "institution_name": {"type": "string"},
                    "linkedin_url": {"type": ["string", "null"]},
                    "from_date": {"type": ["string", "null"]},
                    "to_date": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                },
                "required": [
                    "degree", "institution_name", "linkedin_url",
                    "from_date", "to_date", "duration", "location", "description",
                ],
                "additionalProperties": False,
            },
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["category", "items"],
                "additionalProperties": False,
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string"},
                    "role": {"type": ["string", "null"]},
                    "from_date": {"type": ["string", "null"]},
                    "to_date": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "technologies": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                },
                "required": [
                    "project_name", "role", "from_date", "to_date",
                    "duration", "technologies", "description", "url",
                ],
                "additionalProperties": False,
            },
        },
        "interests": {"type": "array", "items": {"type": "string"}},
        "accomplishments": {"type": "array", "items": {"type": "string"}},
        "contacts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "linkedin_url", "name", "location", "about", "open_to_work",
        "experiences", "educations", "skills", "projects",
        "interests", "accomplishments", "contacts",
    ],
    "additionalProperties": False,
}

# Mock resume text (example input for testing)
MOCK_RESUME_TEXT = """JEAN DUPONT
Paris, France | jean.dupont@email.fr | +33 6 12 34 56 78
LinkedIn: linkedin.com/in/jeandupont

ABOUT
Data engineer with 5 years of experience in Python, SQL, and cloud (AWS). Passionate about building reliable ETL pipelines and ML infrastructure.

EXPERIENCE
Senior Data Engineer | TechCorp SAS | 2021 – Present
Paris, France
Design and maintain batch and streaming pipelines. Lead migration to Apache Airflow. Tech stack: Python, Spark, Snowflake.

Data Analyst | StartupXYZ | 2019 – 2021
Lyon, France
Built dashboards and reports. Wrote SQL and Python scripts for data quality checks.

EDUCATION
Master in Computer Science | Université Paris-Saclay | 2017 – 2019
Paris, France
Thesis on distributed systems.

Bachelor in Mathematics | Sorbonne University | 2014 – 2017
Paris, France

SKILLS
Programming: Python, SQL, Scala, Bash
Data: Spark, Airflow, dbt, Snowflake, BigQuery
DevOps: Docker, Kubernetes, CI/CD

PROJECTS
ETL Framework (2023) – Open-source Python library for configurable pipelines. Role: maintainer. Tech: Python, pytest.
Internal Dashboard (2022) – Real-time metrics for the product team. Tech: React, FastAPI, PostgreSQL.

INTERESTS
Open source, running, reading.

ACCOMPLISHMENTS
Speaker at Data Summit 2023. Certified AWS Solutions Architect.
"""

# Example ground truth JSON (perfect reference to compare model outputs against)
MOCK_GROUND_TRUTH_JSON = {
    "linkedin_url": "linkedin.com/in/jeandupont",
    "name": "Jean Dupont",
    "location": "Paris, France",
    "about": "Data engineer with 5 years of experience in Python, SQL, and cloud (AWS). Passionate about building reliable ETL pipelines and ML infrastructure.",
    "open_to_work": None,
    "experiences": [
        {
            "position_title": "Senior Data Engineer",
            "institution_name": "TechCorp SAS",
            "linkedin_url": None,
            "from_date": "2021",
            "to_date": "Present",
            "duration": None,
            "location": "Paris, France",
            "description": "Design and maintain batch and streaming pipelines. Lead migration to Apache Airflow. Tech stack: Python, Spark, Snowflake.",
        },
        {
            "position_title": "Data Analyst",
            "institution_name": "StartupXYZ",
            "linkedin_url": None,
            "from_date": "2019",
            "to_date": "2021",
            "duration": None,
            "location": "Lyon, France",
            "description": "Built dashboards and reports. Wrote SQL and Python scripts for data quality checks.",
        },
    ],
    "educations": [
        {"degree": "Master in Computer Science", "institution_name": "Université Paris-Saclay", "linkedin_url": None, "from_date": "2017", "to_date": "2019", "duration": None, "location": "Paris, France", "description": "Thesis on distributed systems."},
        {"degree": "Bachelor in Mathematics", "institution_name": "Sorbonne University", "linkedin_url": None, "from_date": "2014", "to_date": "2017", "duration": None, "location": "Paris, France", "description": None},
    ],
    "skills": [
        {"category": "Programming", "items": ["Python", "SQL", "Scala", "Bash"]},
        {"category": "Data", "items": ["Spark", "Airflow", "dbt", "Snowflake", "BigQuery"]},
        {"category": "DevOps", "items": ["Docker", "Kubernetes", "CI/CD"]},
    ],
    "projects": [
        {"project_name": "ETL Framework", "role": "maintainer", "from_date": "2023", "to_date": None, "duration": None, "technologies": ["Python", "pytest"], "description": "Open-source Python library for configurable pipelines.", "url": None},
        {"project_name": "Internal Dashboard", "role": None, "from_date": "2022", "to_date": None, "duration": None, "technologies": ["React", "FastAPI", "PostgreSQL"], "description": "Real-time metrics for the product team.", "url": None},
    ],
    "interests": ["Open source", "running", "reading"],
    "accomplishments": ["Speaker at Data Summit 2023", "Certified AWS Solutions Architect"],
    "contacts": ["jean.dupont@email.fr", "+33 6 12 34 56 78"],
}
MOCK_GROUND_TRUTH_JSON_STR = json.dumps(MOCK_GROUND_TRUTH_JSON, indent=2, ensure_ascii=False)

# Models: provider "openai" | "replicate"; use_responses_api = True => Responses API (e.g. nano models)
MODELS = {
    "gpt-4.1-nano": {
        "label": "GPT-4.1 nano",
        "provider": "openai",
        "input_per_1m": 0.05,
        "output_per_1m": 0.40,
        "structured_output": True,
        "max_completion_tokens": 2048,
        "use_responses_api": True,
    },
    "gpt-5-mini": {
        "label": "GPT-5 mini",
        "provider": "openai",
        "input_per_1m": 0.25,
        "output_per_1m": 2.00,
        "structured_output": True,
        "max_completion_tokens": 4096,
    },
    "replicate/gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash (Replicate)",
        "provider": "replicate",
        "replicate_id": "google/gemini-2.5-flash",
        "cost_estimate_per_run": 0.001,  # approximate; Replicate bills per run
        "structured_output": False,
    },
    "replicate/claude-4.5-haiku": {
        "label": "Claude 4.5 Haiku (Replicate)",
        "provider": "replicate",
        "replicate_id": "anthropic/claude-4.5-haiku",
        "cost_estimate_per_run": 0.002,
        "structured_output": False,
    },
}


def _parse_json_from_response(text: str) -> tuple[Optional[dict], Optional[str]]:
    """Extract JSON from model output. Returns (parsed_dict, error_message)."""
    if not (text and text.strip()):
        return None, "Empty response"
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    for start in ("```json", "```"):
        if start in text:
            idx = text.find(start) + len(start)
            end = text.find("```", idx)
            if end != -1:
                try:
                    return json.loads(text[idx:end].strip()), None
                except json.JSONDecodeError:
                    pass
    # Try extracting a single JSON object: first { to matching }
    i = text.find("{")
    if i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i : j + 1]), None
                    except json.JSONDecodeError:
                        pass
                    break
    return None, "No valid JSON found in response"


def _strip_extra_keys_to_schema(data: Any, schema: dict) -> Any:
    """Return a copy of data with only keys allowed by schema (removes e.g. 'additionalProperties' from model output)."""
    if schema.get("type") == "object" and isinstance(data, dict):
        props = schema.get("properties") or {}
        return {k: _strip_extra_keys_to_schema(v, props[k]) for k, v in data.items() if k in props}
    if schema.get("type") == "array" and isinstance(data, list):
        item_schema = schema.get("items") or {}
        return [_strip_extra_keys_to_schema(item, item_schema) for item in data]
    return data


def _validate_schema(data: dict) -> tuple[bool, Optional[str], Optional[dict]]:
    """Returns (valid, error_message, cleaned_dict). When valid, use cleaned_dict (schema-only keys) for storage."""
    try:
        import jsonschema
        # Strip keys not in schema so model output like "additionalProperties": {} doesn't fail
        cleaned = _strip_extra_keys_to_schema(data, RESUME_EXTRACTION_SCHEMA)
        jsonschema.validate(instance=cleaned, schema=RESUME_EXTRACTION_SCHEMA)
        return True, None, cleaned
    except Exception as e:
        return False, str(e), None


def _flatten_leaves(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    """Return list of (path, leaf_value) for all leaves in obj."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten_leaves(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_flatten_leaves(v, f"{path}[{i}]"))
    else:
        out.append((path, obj))
    return out


def _normalize_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().lower()


def _accuracy_vs_ground_truth(predicted: dict, ground_truth: dict) -> float:
    """Compare predicted JSON to ground truth; return 0–100 (%% of leaf values that match)."""
    gt_leaves = _flatten_leaves(ground_truth)
    if not gt_leaves:
        return 100.0
    pred_flat = {p: v for p, v in _flatten_leaves(predicted)}
    matches = 0
    for path, gt_val in gt_leaves:
        pred_val = pred_flat.get(path)
        if _normalize_value(pred_val) == _normalize_value(gt_val):
            matches += 1
    return 100.0 * matches / len(gt_leaves)


# ---------------------------------------------------------------------------
# Borda and Condorcet (multi-criteria ranking)
# ---------------------------------------------------------------------------
def _borda_rank(scores_per_criterion: list, weights: list) -> list:
    """Weighted Borda: each criterion gives points by rank (rank 1 = n-1 pts, etc.); higher total = better."""
    n_models = len(scores_per_criterion[0])
    n_crit = len(scores_per_criterion)
    points = [0.0] * n_models
    for c in range(n_crit):
        w = weights[c] if c < len(weights) else 1.0
        order = sorted(range(n_models), key=lambda i: scores_per_criterion[c][i], reverse=True)
        for rank, idx in enumerate(order):
            points[idx] += (n_models - 1 - rank) * w
    return points


def _condorcet_wins(scores_per_criterion: list, higher_better: list) -> list:
    """Count how many criteria each model wins (is best on that criterion)."""
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


def _condorcet_wins_detail(scores_per_criterion: list, higher_better: list, criterion_names: list) -> list:
    """For each model, return the list of criterion names it won."""
    n_models = len(scores_per_criterion[0])
    n_crit = len(scores_per_criterion)
    detail = [[] for _ in range(n_models)]
    for c in range(n_crit):
        vals = scores_per_criterion[c]
        hb = higher_better[c] if c < len(higher_better) else True
        best = max(vals) if hb else min(vals)
        name = criterion_names[c] if c < len(criterion_names) else f"Criterion {c}"
        for i in range(n_models):
            if vals[i] == best:
                detail[i].append(name)
    return detail


def _run_openai_responses_api(client, model_id: str, system: str, user: str, max_output_tokens: int):
    """Call nano models via Responses API (POST /v1/responses)."""
    # Input: instructions = system prompt; input = user message(s) as list of items
    input_messages = [
        {"role": "user", "content": [{"type": "input_text", "text": user}]},
    ]
    text_format = {
        "format": {
            "type": "json_schema",
            "name": "resume_extraction",
            "strict": True,
            "schema": RESUME_EXTRACTION_SCHEMA,
        },
        "verbosity": "medium",
    }
    kwargs = {
        "model": model_id,
        "instructions": system,
        "input": input_messages,
        "text": text_format,
    }
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens
    return client.responses.create(**kwargs)


def _extract_responses_api_output_text(resp) -> str:
    """Get aggregated text from Responses API output (output_text items)."""
    if hasattr(resp, "output_text") and resp.output_text:
        return (resp.output_text or "").strip()
    output = getattr(resp, "output", None) or []
    parts = []

    def collect_text(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            if obj.get("type") == "output_text":
                parts.append(obj.get("text", ""))
            elif obj.get("type") == "refusal":
                parts.append(obj.get("refusal", ""))
            for c in obj.get("content") or []:
                collect_text(c)
            return
        if hasattr(obj, "type"):
            if getattr(obj, "type", None) == "output_text":
                parts.append(getattr(obj, "text", "") or "")
            for c in (getattr(obj, "content", None) or []):
                collect_text(c)
            return
        if hasattr(obj, "text"):
            parts.append(getattr(obj, "text", "") or "")

    for item in output:
        collect_text(item)
    result = " ".join(str(p) for p in parts).strip()
    if result:
        return result
    # Fallback: dump response to dict and recursively find output_text.text (SDK 2.x shape)
    try:
        if hasattr(resp, "model_dump"):
            data = resp.model_dump()
        elif hasattr(resp, "dict"):
            data = resp.dict()
        else:
            data = {}
        found = []

        def walk(d):
            if isinstance(d, dict):
                if d.get("type") == "output_text":
                    found.append(d.get("text") or "")
                for v in d.values():
                    walk(v)
            elif isinstance(d, list):
                for x in d:
                    walk(x)

        walk(data)
        if found:
            return " ".join(str(t) for t in found).strip()
    except Exception:
        pass
    return ""


def _get_refusal_from_output(output) -> Optional[str]:
    """Extract refusal message from Responses API output array if present."""
    for item in output or []:
        if isinstance(item, dict) and item.get("type") == "refusal":
            return item.get("refusal", "")
        if getattr(item, "type", None) == "refusal":
            return getattr(item, "refusal", None)
    return None


def _run_openai(model_id: str, ocr_text: str, api_key: str, result_holder: list, index: int) -> None:
    """OpenAI with native structured output. Nano models use Responses API; others use Chat Completions."""
    info = MODELS[model_id]
    label = info["label"]
    out = {
        "model_id": model_id,
        "label": label,
        "text": "",
        "parsed": None,
        "time_seconds": 0.0,
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "json_valid": False,
        "schema_valid": False,
        "structured_output_used": True,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        client = OpenAI(api_key=api_key)
        system = (
            "You are a precise resume parser. Extract structured data from the given OCR/text of a resume or CV. "
            "Use null for missing values. Output only the JSON object that matches the schema."
        )
        user = "Extract resume data from the following text.\n\nText to parse:\n" + ocr_text
        max_tok = info.get("max_completion_tokens", 4096)

        # Nano-style models: use Responses API (POST /v1/responses), not Chat Completions
        if info.get("use_responses_api"):
            resp = _run_openai_responses_api(client, model_id, system, user, max_tok)
            out["time_seconds"] = time.perf_counter() - t0
            content = _extract_responses_api_output_text(resp)
            out["text"] = content
            if getattr(resp, "usage", None):
                inp = getattr(resp.usage, "input_tokens", 0) or 0
                out_tok = getattr(resp.usage, "output_tokens", 0) or getattr(resp.usage, "completion_tokens", 0) or 0
                out["input_tokens"] = inp
                out["output_tokens"] = out_tok
                out["cost_usd"] = (inp / 1e6) * info["input_per_1m"] + (out_tok / 1e6) * info["output_per_1m"]
            refusal = getattr(resp, "refusal", None) or (getattr(resp, "output", None) and _get_refusal_from_output(resp.output))
            if refusal:
                out["error"] = f"Model refused: {refusal}"
            else:
                parsed, err = _parse_json_from_response(content)
                if err:
                    out["error"] = err
                    parsed = None
                if parsed is not None:
                    out["json_valid"] = True
                    ok, schema_err, cleaned = _validate_schema(parsed)
                    out["schema_valid"] = ok
                    if ok and cleaned is not None:
                        out["parsed"] = cleaned
                    else:
                        out["parsed"] = parsed
                    if not ok:
                        out["error"] = schema_err
            result_holder[index] = out
            return

        # Chat Completions API (GPT-5 mini, etc.)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "resume_extraction",
                "strict": True,
                "schema": RESUME_EXTRACTION_SCHEMA,
            },
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = dict(
            model=model_id,
            messages=messages,
            response_format=response_format,
            max_completion_tokens=max_tok,
        )
        parse_fn = getattr(client.chat.completions, "parse", None)
        if parse_fn is not None:
            resp = parse_fn(**kwargs)
        else:
            resp = client.chat.completions.create(**kwargs)
        out["time_seconds"] = time.perf_counter() - t0
        msg = resp.choices[0].message
        raw_content = msg.content
        # Normalize: API can return content as string, list of parts, or SDK object
        def _text_from_part(part):
            if isinstance(part, str):
                return part
            if isinstance(part, dict):
                if part.get("type") == "refusal":
                    return part.get("refusal", "")
                for key in ("text", "value", "content", "input_text"):
                    if key in part:
                        v = part[key]
                        return v.get("value", v) if isinstance(v, dict) else str(v)
                return ""
            if hasattr(part, "text"):
                t = getattr(part, "text", None)
                return getattr(t, "value", t) if t is not None and hasattr(t, "value") else (str(t) if t is not None else "")
            if hasattr(part, "value"):
                return str(getattr(part, "value", ""))
            return str(part) if part is not None else ""

        if isinstance(raw_content, list):
            content = " ".join(_text_from_part(p) for p in raw_content).strip()
        elif isinstance(raw_content, str):
            content = (raw_content or "").strip()
        else:
            # SDK object (e.g. content block with .value or .content)
            content = (_text_from_part(raw_content) if raw_content is not None else "").strip()
            if not content and raw_content is not None:
                content = str(raw_content).strip()
        out["text"] = content
        if getattr(resp, "usage", None):
            inp = getattr(resp.usage, "prompt_tokens", 0) or 0
            out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
            out["input_tokens"] = inp
            out["output_tokens"] = out_tok
            out["cost_usd"] = (inp / 1e6) * info["input_per_1m"] + (out_tok / 1e6) * info["output_per_1m"]

        # Explicit refusals (safety): programmatically detectable with Structured Outputs
        refusal = getattr(msg, "refusal", None)
        if refusal:
            out["error"] = f"Model refused: {refusal}"
        else:
            # Prefer .parsed when available (from parse()); else parse content
            raw_parsed = getattr(msg, "parsed", None)
            if raw_parsed is not None:
                if isinstance(raw_parsed, dict):
                    parsed = raw_parsed
                elif hasattr(raw_parsed, "model_dump"):
                    parsed = raw_parsed.model_dump()
                else:
                    parsed = raw_parsed
            else:
                # If content still empty, try message as dict (some SDKs expose content there)
                if not content and hasattr(msg, "model_dump"):
                    try:
                        d = msg.model_dump()
                        content = (d.get("content") or "")
                        if isinstance(content, list):
                            content = " ".join(_text_from_part(p) for p in content).strip()
                        else:
                            content = (content or "").strip()
                        out["text"] = content
                    except Exception:
                        pass
                parsed, err = _parse_json_from_response(content)
                if err:
                    out["error"] = err
                    if out.get("output_tokens") and "Empty response" in (err or ""):
                        out["error"] = (
                            "Empty content from API (model may use a different response shape or returned no text). "
                            f"content type: {type(raw_content).__name__}"
                        )
                    parsed = None
            if parsed is not None:
                out["json_valid"] = True
                ok, schema_err, cleaned = _validate_schema(parsed)
                out["schema_valid"] = ok
                if ok and cleaned is not None:
                    out["parsed"] = cleaned
                else:
                    out["parsed"] = parsed
                if not ok:
                    out["error"] = schema_err
    except Exception as e:
        out["time_seconds"] = time.perf_counter() - t0
        out["error"] = str(e)
    result_holder[index] = out


def _run_replicate(model_id: str, ocr_text: str, replicate_token: str, result_holder: list, index: int) -> None:
    """Replicate: prompt-based JSON (no native structured output). Input/output schemas via API: model openapi_schema."""
    info = MODELS[model_id]
    label = info["label"]
    replicate_model = info["replicate_id"]
    out = {
        "model_id": model_id,
        "label": label,
        "text": "",
        "parsed": None,
        "time_seconds": 0.0,
        "cost_usd": info.get("cost_estimate_per_run", 0.001),
        "input_tokens": 0,
        "output_tokens": 0,
        "json_valid": False,
        "schema_valid": False,
        "structured_output_used": False,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        import replicate as repl
        schema_desc = json.dumps(RESUME_EXTRACTION_SCHEMA, indent=2)
        system = (
            "You are a precise resume parser. Extract structured data from the given OCR/text. "
            "Respond with a single JSON object that strictly follows the schema provided. Use null for missing values. "
            "Do not add any commentary, only the JSON object."
        )
        user = (
            "Extract resume data from the following text into this exact JSON schema.\n\n"
            "Schema:\n" + schema_desc + "\n\n"
            "Text to parse:\n" + ocr_text
        )
        prev = os.environ.get("REPLICATE_API_TOKEN")
        try:
            os.environ["REPLICATE_API_TOKEN"] = replicate_token
            wait_seconds = 120  # avoid "The read operation timed out" on slow runs
            if "gemini" in replicate_model.lower():
                run_input = {
                    "prompt": user,
                    "system_instruction": system,
                    "max_output_tokens": 4096,
                }
            else:
                run_input = {
                    "prompt": user,
                    "system_prompt": system,
                    "max_tokens": 4096,
                }
            # Retry on SSL/TLS errors (e.g. TLSV1_ALERT_DECODE_ERROR) – often transient with Replicate/Gemini
            raw = None
            last_err = None
            for attempt in range(3):
                try:
                    raw = repl.run(replicate_model, input=run_input, wait=wait_seconds)
                    break
                except Exception as e:
                    last_err = e
                    is_ssl = "SSL" in type(e).__name__ or "ssl" in str(e).lower() or "TLS" in str(e).upper()
                    if is_ssl and attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    raise
            if raw is None and last_err is not None:
                raise last_err
        finally:
            if prev is None:
                os.environ.pop("REPLICATE_API_TOKEN", None)
            else:
                os.environ["REPLICATE_API_TOKEN"] = prev

        out["time_seconds"] = time.perf_counter() - t0
        if isinstance(raw, list):
            content = "".join(str(x) for x in raw).strip()
        elif isinstance(raw, str):
            content = raw.strip()
        else:
            content = str(raw).strip()
        out["text"] = content

        parsed, err = _parse_json_from_response(content)
        if err:
            out["error"] = err
        else:
            out["json_valid"] = True
            ok, schema_err, cleaned = _validate_schema(parsed)
            out["schema_valid"] = ok
            if ok and cleaned is not None:
                out["parsed"] = cleaned
            else:
                out["parsed"] = parsed
            if not ok:
                out["error"] = schema_err
    except Exception as e:
        out["time_seconds"] = time.perf_counter() - t0
        out["error"] = str(e)
    result_holder[index] = out


def _run_one_model(
    model_id: str,
    ocr_text: str,
    openai_key: str,
    replicate_key: Optional[str],
    result_holder: list,
    index: int,
) -> None:
    info = MODELS.get(model_id, {})
    if info.get("provider") == "replicate":
        if not replicate_key:
            result_holder[index] = {
                "model_id": model_id,
                "label": info.get("label", model_id),
                "text": "",
                "parsed": None,
                "time_seconds": 0.0,
                "cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "json_valid": False,
                "schema_valid": False,
                "structured_output_used": False,
                "error": "REPLICATE_API_KEY not set",
            }
            return
        _run_replicate(model_id, ocr_text, replicate_key, result_holder, index)
    else:
        _run_openai(model_id, ocr_text, openai_key, result_holder, index)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LLM Parsing Benchmark", layout="wide")
st.title("LLM Parsing Benchmark")
st.markdown("Compare **GPT-4.1 nano**, **GPT-5 mini** (OpenAI), **Gemini 2.5 Flash** and **Claude 4.5 Haiku** (Replicate). "
            "Uses **OPENAI_API_KEY** and **REPLICATE_API_KEY** from secrets (same as OCR Benchmark). "
            "**OpenAI models use native structured output** (response_format json_schema); Replicate uses prompt + JSON parse. "
            "Selected models run **in parallel** (one thread per model).")

openai_key = get_openai_key()
replicate_key = get_replicate_key()
if not openai_key:
    st.error("Missing **OPENAI_API_KEY** in `.streamlit/secrets.toml`.")
    st.stop()
if not replicate_key:
    st.warning("**REPLICATE_API_KEY** not set. You can still run OpenAI models (GPT-4.1 nano, GPT-5 mini).")

# Model selection: default all 4 (each runs in its own thread)
default_models = ["gpt-4.1-nano", "gpt-5-mini", "replicate/gemini-2.5-flash", "replicate/claude-4.5-haiku"]
selected = st.multiselect(
    "Models to run (in parallel, 1 thread per model)",
    options=list(MODELS.keys()),
    default=default_models,
    format_func=lambda x: MODELS[x]["label"],
    key="llm_parsing_models",
)

with st.expander("Metrics & multi-criteria weights (Borda / Condorcet)"):
    st.markdown("""
| Metric | Meaning | How we get it |
| --- | --- | --- |
| **Accuracy** | Match to ground truth | % of leaf values in parsed JSON that match the reference (right panel). Higher is better. |
| **Speed** | Time to finish | Wall-clock seconds from request start to response. Lower is better. |
| **Cost** | Price of the run | **OpenAI:** from token usage × list price. **Replicate:** fixed estimate per run. Lower is better. |
""")
    st.markdown("**Weights for ranking (3 metrics, sum = 1):**")
    w_accuracy = st.slider("Weight: Accuracy (higher is better)", 0.0, 1.0, 0.40, 0.05, key="llm_w_accuracy")
    # Speed max = remainder so that Accuracy + Speed <= 1; Cost = rest (weights always sum to 1)
    speed_max = max(0.0, 1.0 - w_accuracy)
    w_speed = st.slider("Weight: Speed (faster is better)", 0.0, speed_max, min(0.30, speed_max), 0.05, key="llm_w_speed")
    w_cost = 1.0 - w_accuracy - w_speed
    st.session_state["llm_w_cost"] = w_cost
    st.metric("Weight: Cost (lower is better)", f"{w_cost:.2f}", help="Remainder so that Accuracy + Speed + Cost = 1")
    st.caption("Weights always sum to 1. Cost is set automatically from the remainder.")

# First section: two inputs — OCR (left) and ground truth JSON (right)
st.subheader("Input")
col_left, col_right = st.columns(2)
with col_left:
    st.caption("OCR or raw resume text (what the models will parse)")
    ocr_text = st.text_area(
        "OCR output (input)",
        value=MOCK_RESUME_TEXT,
        height=380,
        placeholder="Paste the text extracted from a CV/resume.",
        key="llm_ocr_input",
    )
with col_right:
    st.caption("Perfect / reference JSON — we compare each model’s output to this for accuracy")
    ground_truth_json_str = st.text_area(
        "Ground truth JSON (perfect data)",
        value=MOCK_GROUND_TRUTH_JSON_STR,
        height=380,
        placeholder="Paste the expected JSON (same schema as resume extraction).",
        key="llm_ground_truth_json",
    )

st.divider()
run_btn = st.button("Run parsing (parallel)", type="primary", key="llm_run_parsing")
if run_btn and selected and ocr_text and ocr_text.strip():
    results = [None] * len(selected)
    threads = []
    wall_start = time.perf_counter()
    # Start all threads first (no join yet) — all API calls run concurrently
    for i, model_id in enumerate(selected):
        t = Thread(
            target=_run_one_model,
            args=(model_id, ocr_text, openai_key, replicate_key, results, i),
        )
        t.start()
        threads.append(t)
    with st.spinner("Running models in parallel…"):
        for t in threads:
            t.join()
    wall_seconds = time.perf_counter() - wall_start
    st.session_state["llm_parsing_last_results"] = results
    st.session_state["llm_parsing_wall_clock_seconds"] = wall_seconds
    st.rerun()

if run_btn and (not ocr_text or not ocr_text.strip()):
    st.warning("Paste some OCR/resume text in the left panel first.")
if run_btn and not selected:
    st.warning("Select at least one model.")

# Results (after run): metrics + accuracy vs ground truth, then view parsed JSON
last_results = st.session_state.get("llm_parsing_last_results")
ground_truth_parsed = None
if ground_truth_json_str and ground_truth_json_str.strip():
    gt_parsed, _ = _parse_json_from_response(ground_truth_json_str.strip())
    if gt_parsed is not None:
        ground_truth_parsed = gt_parsed

if last_results:
    st.divider()
    st.subheader("Results")
    wall_s = st.session_state.get("llm_parsing_wall_clock_seconds")
    if wall_s is not None:
        max_model_s = max((r["time_seconds"] for r in last_results), default=0)
        st.caption(
            f"**Total wall-clock:** {wall_s:.2f} s (parallel run). "
            f"≈ slowest model ({max_model_s:.2f} s) ⇒ calls ran in parallel. "
            f"If sequential, total would be ~{sum(r['time_seconds'] for r in last_results):.1f} s."
        )
    # Compute accuracy vs ground truth for each model
    rows = []
    for r in last_results:
        accuracy_pct = None
        if ground_truth_parsed is not None and r.get("parsed"):
            accuracy_pct = _accuracy_vs_ground_truth(r["parsed"], ground_truth_parsed)
            r["accuracy_pct"] = accuracy_pct
        rows.append({
            "Model": r["label"],
            "Time (s)": f"{r['time_seconds']:.2f}",
            "Cost ($)": f"{r['cost_usd']:.4f}",
            "Accuracy (%)": f"{accuracy_pct:.1f}" if accuracy_pct is not None else "—",
            "Input tok": r.get("input_tokens") or "—",
            "Output tok": r.get("output_tokens") or "—",
            "JSON valid": "Yes" if r["json_valid"] else "No",
            "Schema valid": "Yes" if r["schema_valid"] else "No",
            "Structured out": "Yes" if r.get("structured_output_used") else "No",
            "Error": r.get("error") or "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if ground_truth_parsed is None:
        st.caption("Provide valid ground truth JSON (right panel) to see Accuracy % vs that reference.")
    valid = [r for r in last_results if r["schema_valid"]]
    if valid:
        best = min(valid, key=lambda x: (-x.get("accuracy_pct") or 0, x["cost_usd"], x["time_seconds"]))
        acc = f", accuracy {best.get('accuracy_pct', 0):.1f}%" if best.get("accuracy_pct") is not None else ""
        st.success(f"**Best schema-compliant:** {best['label']} (cost ${best['cost_usd']:.4f}, {best['time_seconds']:.2f}s{acc}).")
    else:
        st.warning("No model returned schema-valid JSON for this input. Check errors above.")

    # View parsed JSON from a chosen model
    st.subheader("View parsed JSON (model output)")
    chosen = st.radio(
        "Show extracted JSON from",
        options=[r["label"] for r in last_results],
        key="llm_view_which",
        horizontal=True,
    )
    for r in last_results:
        if r["label"] == chosen:
            if r.get("parsed") is not None:
                st.json(r["parsed"])
            else:
                st.code(r.get("text", "") or (r.get("error") or "No output"), language="json" if r.get("json_valid") else None)
            st.caption("Structured output: Yes" if r.get("structured_output_used") else "Structured output: No (prompt + parse)")
            break

    # Borda and Condorcet: best model from weighted ranking
    st.divider()
    st.subheader("Borda and Condorcet")
    if len(last_results) >= 2:
        total_w = st.session_state.get("llm_w_accuracy", 0.4) + st.session_state.get("llm_w_speed", 0.3) + st.session_state.get("llm_w_cost", 0.3)
        if total_w <= 0:
            total_w = 1.0
        w_acc = st.session_state.get("llm_w_accuracy", 0.4) / total_w
        w_spd = st.session_state.get("llm_w_speed", 0.3) / total_w
        w_cst = st.session_state.get("llm_w_cost", 0.3) / total_w
        # Scores: higher = better (so negate speed and cost)
        accuracy_scores = [(r.get("accuracy_pct") or 0.0) for r in last_results]
        speed_scores = [-r["time_seconds"] for r in last_results]
        cost_scores = [-r["cost_usd"] for r in last_results]
        scores_per_criterion = [accuracy_scores, speed_scores, cost_scores]
        weights_norm = [w_acc, w_spd, w_cst]
        higher_better = [True, True, True]
        borda_scores = _borda_rank(scores_per_criterion, weights_norm)
        condorcet_counts = _condorcet_wins(scores_per_criterion, higher_better)
        criterion_names = ["Accuracy", "Speed", "Cost"]
        condorcet_detail = _condorcet_wins_detail(scores_per_criterion, higher_better, criterion_names)
        names = [r["label"] for r in last_results]
        best_borda_idx = max(range(len(borda_scores)), key=lambda i: borda_scores[i])
        best_condorcet_idx = max(range(len(condorcet_counts)), key=lambda i: condorcet_counts[i])
        best_borda_name = names[best_borda_idx]
        best_condorcet_name = names[best_condorcet_idx]
        st.markdown("**Borda (weighted ranking)**")
        st.markdown(f"Best model: **{best_borda_name}** (score: {borda_scores[best_borda_idx]:.2f}).")
        st.markdown("Scores by model (higher = better):")
        for i in range(len(last_results)):
            st.markdown(f"- {names[i]}: **{borda_scores[i]:.2f}**")
        st.caption("Borda: each criterion contributes weighted points by rank; scores are summed.")
        st.markdown("---")
        st.markdown("**Condorcet (majority of criteria)**")
        st.markdown(f"Best model: **{best_condorcet_name}** (won {condorcet_counts[best_condorcet_idx]} of 3 criteria).")
        st.markdown("Criteria won per model (which criterion each model won):")
        for i in range(len(last_results)):
            won = condorcet_detail[i]
            n = len(won)
            if n == 0:
                detail_str = "—"
            else:
                detail_str = ", ".join(won)
            st.markdown(f"- **{names[i]}**: {n} criteria — **{detail_str}**")
        st.caption("Condorcet: for each criterion the best model gets a win; the model that wins the most criteria is the Condorcet winner.")
        st.markdown("---")
        st.markdown("**Analysis**")
        if best_borda_name == best_condorcet_name:
            st.info(f"**{best_borda_name}** is both the Borda and Condorcet winner: best weighted score and wins the majority of criteria. Recommended for this input and weights.")
        else:
            st.info(f"**Borda** favours **{best_borda_name}** (best weighted overall). **Condorcet** favours **{best_condorcet_name}** (won most criteria). Use Borda for a single balanced pick, or Condorcet if you prefer the model that wins the most individual metrics.")
    else:
        st.info("Need at least two models in the run to compare with Borda/Condorcet.")
