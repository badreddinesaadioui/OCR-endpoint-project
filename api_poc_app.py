"""
Local Streamlit POC UI for CV Parsing API.

Usage:
  streamlit run api_poc_app.py
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import requests
import streamlit as st


SUPPORTED_TYPES = ("pdf", "png", "jpg", "jpeg", "docx")


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;800&display=swap');

:root {
  --bg-a: #f7efe3;
  --bg-b: #e7f0f5;
  --ink: #1f2937;
  --muted: #5b6673;
  --accent: #c84f2f;
  --accent-2: #267c7c;
  --card: #ffffffde;
  --line: #d9dde3;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 8% -10%, #ffd8b7 0%, transparent 55%),
    radial-gradient(1000px 450px at 95% 5%, #b8efe6 0%, transparent 50%),
    linear-gradient(140deg, var(--bg-a) 0%, var(--bg-b) 100%);
}

section.main > div {
  max-width: 1100px;
}

h1, h2, h3 {
  font-family: "Space Grotesk", sans-serif !important;
  letter-spacing: -0.01em;
  color: var(--ink);
}

p, li, .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, .stRadio {
  font-family: "Manrope", sans-serif !important;
  color: var(--ink);
}

.hero {
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 20px 22px;
  margin: 8px 0 14px;
  background: linear-gradient(130deg, #fff 0%, #fff8ef 55%, #eefaf8 100%);
  box-shadow: 0 8px 24px rgba(20, 24, 31, 0.08);
}

.card {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px 16px;
  background: var(--card);
  box-shadow: 0 6px 14px rgba(0, 0, 0, 0.04);
}

.kpi {
  display: inline-block;
  border: 1px solid #d9d9d9;
  border-radius: 999px;
  padding: 6px 12px;
  margin-right: 8px;
  background: #fff;
  color: var(--muted);
  font-size: 0.85rem;
}

.badge-ok {
  color: #0f5132;
  background: #d1e7dd;
  border: 1px solid #badbcc;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
}

.badge-warn {
  color: #664d03;
  background: #fff3cd;
  border: 1px solid #ffecb5;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _headers(token: str) -> Dict[str, str]:
    if token.strip():
        return {"Authorization": f"Bearer {token.strip()}"}
    return {}


def _extract_error(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except Exception:
        return f"HTTP {resp.status_code}: {resp.text[:500]}"
    err = body.get("error") or {}
    code = err.get("code", "UNKNOWN")
    msg = err.get("message", "No message")
    return f"HTTP {resp.status_code} - {code}: {msg}"


def _call_health(base_url: str, token: str) -> tuple[bool, str]:
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/health",
            headers=_headers(token),
            timeout=10,
        )
        if resp.ok:
            return True, "API reachable"
        return False, _extract_error(resp)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _create_job(
    base_url: str,
    token: str,
    uploaded_file,
    language_hint: str,
) -> Dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {}
    if language_hint.strip():
        data["language_hint"] = language_hint.strip()

    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/jobs",
        headers=_headers(token),
        files=files,
        data=data,
        timeout=90,
    )
    if not resp.ok:
        raise RuntimeError(_extract_error(resp))
    return resp.json()


def _get_job(base_url: str, token: str, job_id: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{base_url.rstrip('/')}/v1/jobs/{job_id}",
        headers=_headers(token),
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(_extract_error(resp))
    return resp.json()


def _parse_sync(
    base_url: str,
    token: str,
    uploaded_file,
    language_hint: str,
) -> Dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {}
    if language_hint.strip():
        data["language_hint"] = language_hint.strip()

    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/parse-cv",
        headers=_headers(token),
        files=files,
        data=data,
        timeout=300,
    )
    if not resp.ok:
        raise RuntimeError(_extract_error(resp))
    return resp.json()


def _status_badge(status: str) -> str:
    if status == "succeeded":
        return '<span class="badge-ok">succeeded</span>'
    if status in {"queued", "processing"}:
        return '<span class="badge-warn">%s</span>' % status
    return '<span class="badge-warn">%s</span>' % status


def main() -> None:
    st.set_page_config(
        page_title="CV Parsing API POC",
        page_icon=":material/task_alt:",
        layout="wide",
    )
    _inject_styles()

    st.markdown(
        """
<div class="hero">
  <h1 style="margin:0 0 6px;">CV Parsing API - Local POC</h1>
  <p style="margin:0 0 10px;color:#4b5563;">
    Upload un CV, envoie-le a l'API, recupere le JSON parse.
    Deux modes: <b>Async (2 etapes)</b> et <b>Sync (one-shot)</b>.
  </p>
  <span class="kpi">OCR: Mistral OCR 3</span>
  <span class="kpi">LLM: Claude 4.5 Haiku</span>
  <span class="kpi">Formats: pdf/png/jpg/jpeg/docx</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        c1, c2, c3 = st.columns([2.2, 1.4, 1.2])
        with c1:
            base_url = st.text_input(
                "API base URL",
                value=st.session_state.get("api_base_url", "http://localhost:8080"),
                placeholder="http://localhost:8080",
            )
            st.session_state["api_base_url"] = base_url
        with c2:
            auth_token = st.text_input(
                "Bearer token (optional)",
                value=st.session_state.get("api_auth_token", ""),
                type="password",
                placeholder="if API_AUTH_TOKEN is enabled",
            )
            st.session_state["api_auth_token"] = auth_token
        with c3:
            st.write("")
            st.write("")
            if st.button("Check API health", use_container_width=True):
                ok, msg = _call_health(base_url, auth_token)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    col_left, col_right = st.columns([1.2, 1.8], vertical_alignment="top")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        mode = st.radio(
            "Execution mode",
            options=["Async (2-step)", "Sync (one-shot)"],
            index=0,
        )
        language_hint = st.text_input(
            "Language hint (optional)",
            value=st.session_state.get("language_hint", ""),
            placeholder="fr, en, ar...",
        )
        st.session_state["language_hint"] = language_hint
        uploaded_file = st.file_uploader(
            "Upload CV",
            type=SUPPORTED_TYPES,
            accept_multiple_files=False,
        )
        submit = st.button(
            "Run with API",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")
        st.caption("Use the selected mode and inspect the parsed JSON below.")

        if "last_job_payload" not in st.session_state:
            st.session_state["last_job_payload"] = None
        if "last_sync_payload" not in st.session_state:
            st.session_state["last_sync_payload"] = None
        if "active_job_id" not in st.session_state:
            st.session_state["active_job_id"] = ""

        if submit and uploaded_file is not None:
            try:
                if mode.startswith("Async"):
                    payload = _create_job(base_url, auth_token, uploaded_file, language_hint)
                    st.session_state["active_job_id"] = payload["job_id"]
                    st.session_state["last_job_payload"] = payload
                    st.success(f"Job created: {payload['job_id']}")
                else:
                    with st.spinner("Processing in one-shot mode..."):
                        payload = _parse_sync(base_url, auth_token, uploaded_file, language_hint)
                    st.session_state["last_sync_payload"] = payload
                    st.success("Sync parsing completed.")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        if mode.startswith("Async"):
            active_job_id = st.session_state.get("active_job_id", "")
            manual_job_id = st.text_input(
                "Job ID",
                value=active_job_id,
                placeholder="Paste job_id if needed",
            )
            c_poll1, c_poll2 = st.columns([1.2, 1.6])
            with c_poll1:
                refresh_status = st.button(
                    "Refresh status",
                    use_container_width=True,
                    disabled=not manual_job_id.strip(),
                )
            with c_poll2:
                wait_until_done = st.button(
                    "Wait until done (max 60s)",
                    use_container_width=True,
                    disabled=not manual_job_id.strip(),
                )

            if refresh_status and manual_job_id.strip():
                try:
                    job_payload = _get_job(base_url, auth_token, manual_job_id.strip())
                    st.session_state["last_job_payload"] = job_payload
                    st.session_state["active_job_id"] = manual_job_id.strip()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

            if wait_until_done and manual_job_id.strip():
                try:
                    with st.spinner("Polling job status..."):
                        deadline = time.time() + 60
                        final_payload = None
                        while time.time() < deadline:
                            current = _get_job(base_url, auth_token, manual_job_id.strip())
                            final_payload = current
                            if current.get("status") in {"succeeded", "failed"}:
                                break
                            time.sleep(2)
                    if final_payload:
                        st.session_state["last_job_payload"] = final_payload
                        st.session_state["active_job_id"] = manual_job_id.strip()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

            job_payload = st.session_state.get("last_job_payload")
            if job_payload:
                status = job_payload.get("status", "unknown")
                st.markdown(
                    f"Status: {_status_badge(status)}",
                    unsafe_allow_html=True,
                )
                st.json(job_payload)

                if status == "succeeded" and isinstance(job_payload.get("result"), dict):
                    parsed_json_str = json.dumps(job_payload["result"], indent=2, ensure_ascii=False)
                    st.download_button(
                        "Download parsed JSON",
                        data=parsed_json_str.encode("utf-8"),
                        file_name=f"{job_payload.get('job_id', 'cv')}_parsed.json",
                        mime="application/json",
                    )

        else:
            sync_payload = st.session_state.get("last_sync_payload")
            if sync_payload:
                st.json(sync_payload)
                if sync_payload.get("status") == "succeeded" and isinstance(sync_payload.get("result"), dict):
                    parsed_json_str = json.dumps(sync_payload["result"], indent=2, ensure_ascii=False)
                    st.download_button(
                        "Download parsed JSON",
                        data=parsed_json_str.encode("utf-8"),
                        file_name="parsed_cv.json",
                        mime="application/json",
                    )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
