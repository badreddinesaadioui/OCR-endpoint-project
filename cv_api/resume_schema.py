"""
Resume extraction schema and JSON parsing/validation helpers.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple


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
                    "position_title",
                    "institution_name",
                    "linkedin_url",
                    "from_date",
                    "to_date",
                    "duration",
                    "location",
                    "description",
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
                    "degree",
                    "institution_name",
                    "linkedin_url",
                    "from_date",
                    "to_date",
                    "duration",
                    "location",
                    "description",
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
                    "project_name",
                    "role",
                    "from_date",
                    "to_date",
                    "duration",
                    "technologies",
                    "description",
                    "url",
                ],
                "additionalProperties": False,
            },
        },
        "interests": {"type": "array", "items": {"type": "string"}},
        "accomplishments": {"type": "array", "items": {"type": "string"}},
        "contacts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "linkedin_url",
        "name",
        "location",
        "about",
        "open_to_work",
        "experiences",
        "educations",
        "skills",
        "projects",
        "interests",
        "accomplishments",
        "contacts",
    ],
    "additionalProperties": False,
}


def parse_json_from_response(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Extract JSON object from model text response.
    Returns (parsed_dict, error_message).
    """
    if not (text and text.strip()):
        return None, "Empty response"
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "JSON response is not an object"
    except json.JSONDecodeError:
        pass

    for fence_start in ("```json", "```"):
        if fence_start in text:
            start_index = text.find(fence_start) + len(fence_start)
            end_index = text.find("```", start_index)
            if end_index != -1:
                block = text[start_index:end_index].strip()
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict):
                        return parsed, None
                    return None, "JSON response is not an object"
                except json.JSONDecodeError:
                    pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed, None
                        return None, "JSON response is not an object"
                    except json.JSONDecodeError:
                        break

    return None, "No valid JSON found in response"


def _strip_extra_keys_to_schema(data: Any, schema: dict) -> Any:
    if schema.get("type") == "object" and isinstance(data, dict):
        props = schema.get("properties") or {}
        return {
            key: _strip_extra_keys_to_schema(value, props[key])
            for key, value in data.items()
            if key in props
        }
    if schema.get("type") == "array" and isinstance(data, list):
        item_schema = schema.get("items") or {}
        return [_strip_extra_keys_to_schema(item, item_schema) for item in data]
    return data


def validate_resume_schema(data: dict) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Validate parsed dict against RESUME_EXTRACTION_SCHEMA.
    Returns: (is_valid, error_message, cleaned_data)
    """
    try:
        import jsonschema

        cleaned = _strip_extra_keys_to_schema(data, RESUME_EXTRACTION_SCHEMA)
        jsonschema.validate(instance=cleaned, schema=RESUME_EXTRACTION_SCHEMA)
        return True, None, cleaned
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), None
