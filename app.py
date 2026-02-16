"""
Multipage app entrypoint. Uses st.navigation + st.Page with custom sidebar.
Run with: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="OCR & LLM Benchmark",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define all pages with st.Page
page_our_database = st.Page(
    page="pages/our_database.py",
    title="Our database",
)
page_ocr_benchmark = st.Page(
    page="pages/2_OCR_Benchmark.py",
    title="OCR Benchmark",
)
page_parallel_ocr_test = st.Page(
    page="pages/parallel_ocr_test.py",
    title="Parallel OCR test",
)
page_llm_parsing_benchmark = st.Page(
    page="pages/llm_parsing_benchmark.py",
    title="LLM Parsing benchmark",
)
page_parallel_llm_parsing_test = st.Page(
    page="pages/parallel_llm_parsing_test.py",
    title="Parallel LLM Parsing test",
)
page_results_summary = st.Page(
    page="pages/results_summary.py",
    title="Results summary",
)

# Custom sidebar navigation
with st.sidebar.container(border=False):
    st.write("## Navigation")
    st.page_link(
        page=page_our_database,
        label="Our database",
        icon=":material/folder:",
    )
    st.page_link(
        page=page_ocr_benchmark,
        label="OCR Benchmark",
        icon=":material/document_scanner:",
    )
    st.page_link(
        page=page_parallel_ocr_test,
        label="Parallel OCR test",
        icon=":material/speed:",
    )
    st.page_link(
        page=page_llm_parsing_benchmark,
        label="LLM Parsing benchmark",
        icon=":material/psychology:",
    )
    st.page_link(
        page=page_parallel_llm_parsing_test,
        label="Parallel LLM Parsing test",
        icon=":material/bolt:",
    )
    st.page_link(
        page=page_results_summary,
        label="Results summary",
        icon=":material/summarize:",
    )

pages = [
    page_our_database,
    page_ocr_benchmark,
    page_parallel_ocr_test,
    page_llm_parsing_benchmark,
    page_parallel_llm_parsing_test,
    page_results_summary,
]

# Hide default nav; custom sidebar above handles navigation
st.navigation(pages, position="hidden").run()
