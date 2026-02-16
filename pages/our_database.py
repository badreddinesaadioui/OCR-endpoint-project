"""Our CVtheque page: database description + grid of 16 CVs with modal (CV + transcription)."""
import base64
import os
import csv
import streamlit as st

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
SCREENSHOTS_DIR = os.path.join(DB_DIR, "screenshots")
METADATA_PATH = os.path.join(DB_DIR, "metadata.csv")

def screenshot_path(base: str) -> str:
    """Path to database/screenshots/{base}.png if it exists."""
    return os.path.join(SCREENSHOTS_DIR, f"{base}.png")

st.set_page_config(page_title="Our CVtheque", layout="wide")

st.title("Our CVtheque")

st.write(
    "We prepared a **database of 16 CVs** with a variety of types: **multi-column**, **infographic**, and **mono-column** layouts; "
    "languages **Arabic**, **French**, and **English**; **is_scanned** (0 = digital/native, 1 = scanned or image); "
    "**extensions** docx, pdf, png, jpg; **layout type** (multi-column, mono-column, infographic); **has_tables**; "
    "**is_rtl** (1 = right-to-left for Arabic, 0 = left-to-right). We parsed them and this data is used for our tests "
    "to take a decision about the **best OCR model** and the **best LLM parsing model** to use."
)

# Load metadata
@st.cache_data
def load_metadata():
    rows = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filename"):
                rows.append(row)
    return rows

metadata = load_metadata()
if not metadata:
    st.warning("No metadata found in database/metadata.csv.")
    st.stop()

def _layout_label(layout_type: str) -> str:
    """Map layout_type to readable label."""
    m = {"mono": "mono-column", "multi": "multi-column", "infographic": "infographic"}
    return m.get(layout_type, layout_type)


def _yes_no(value) -> str:
    """Display 0/1 as No/Yes."""
    return "Yes" if str(value).strip() == "1" else "No"


@st.dialog("CV view", width="large")
def show_cv_modal(row):
    """Modal: transcription in text_area (scrollable); no custom CSS to avoid clipping."""
    filename = row["filename"]
    base, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip(".")
    txt_path = os.path.join(DB_DIR, f"{base}.txt")
    doc_path = os.path.join(DB_DIR, filename)
    shot_path = screenshot_path(base)

    # Document info: multi-line, user-friendly labels
    ext_display = row.get("extension", ext)
    lang = row.get("language", "")
    layout = _layout_label(row.get("layout_type", ""))
    with st.container(border=True):
        st.markdown("**Document info**")
        st.markdown(
            f"""
- **Extension of file:** {ext_display}
- **Language:** {lang}
- **Layout:** {layout}
- **Number of columns:** {row.get('num_columns', '')}
- **Is scanned or image:** {_yes_no(row.get('is_scanned', ''))}
- **Has tables:** {_yes_no(row.get('has_tables', ''))}
- **Has icons:** {_yes_no(row.get('has_icons', ''))}
- **Has graphics:** {_yes_no(row.get('has_graphics', ''))}
- **Is right-to-left:** {_yes_no(row.get('is_rtl', ''))}
"""
        )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Document")
        img_path = shot_path if os.path.isfile(shot_path) else (doc_path if os.path.isfile(doc_path) and ext in ("png", "jpg", "jpeg") else None)
        if img_path:
            st.image(img_path, use_container_width=True)
        elif os.path.isfile(doc_path) and ext == "pdf":
            try:
                with open(doc_path, "rb") as f:
                    st.download_button("Download PDF", data=f.read(), file_name=filename, mime="application/pdf", key=f"dl_{base}")
            except Exception:
                st.caption(f"PDF: {filename}")
        else:
            st.caption(f"No preview: {filename}")

    with col_right:
        st.subheader("Transcription")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            st.text_area("Text", text, height=1000, disabled=False, key=f"txt_{base}")
        else:
            st.info(f"No {base}.txt in database.")

# Grid: 3 per row; each card = large thumbnail + label + View
st.subheader("CVs")
THUMB_HEIGHT = 320  # Large thumbnail so the CV preview is clearly visible
N_COLS = 4
for i in range(0, len(metadata), N_COLS):
    cols = st.columns(N_COLS)
    for j, col in enumerate(cols):
        idx = i + j
        if idx >= len(metadata):
            break
        row = metadata[idx]
        filename = row["filename"]
        base, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip(".")
        doc_path = os.path.join(DB_DIR, filename)
        shot_path = screenshot_path(base)
        with col:
            with st.container(border=True):
                has_preview = os.path.isfile(shot_path) or (os.path.isfile(doc_path) and ext in ("png", "jpg", "jpeg"))
                if has_preview:
                    img_path = shot_path if os.path.isfile(shot_path) else doc_path
                    mime = "image/jpeg" if img_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    st.markdown(
                        f'<div style="min-height:{THUMB_HEIGHT}px; background:#fafafa; border-radius:8px; overflow:hidden;">'
                        f'<img src="data:{mime};base64,{b64}" style="width:100%; height:{THUMB_HEIGHT}px; object-fit:contain; display:block;" alt="{base}" /></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="height:{THUMB_HEIGHT}px; display:flex; align-items:center; justify-content:center; '
                        f'background:#f0f2f6; border-radius:8px; font-size:4rem;">📄</div>',
                        unsafe_allow_html=True,
                    )
                st.caption(f"**{base}**")
                if st.button("View", key=f"btn_{base}", use_container_width=True):
                    show_cv_modal(row)
