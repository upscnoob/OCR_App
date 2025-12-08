import streamlit as st
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
from pathlib import Path
import requests
import re
import base64 
from streamlit_local_storage import LocalStorage

# --- Constants ---
SUPPORTED_IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')
SUPPORTED_PDF_TYPES = ('.pdf',)
SUPPORTED_FILE_TYPES = SUPPORTED_IMAGE_TYPES + SUPPORTED_PDF_TYPES

# --- Helper Functions ---

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replaces image placeholders with base64 data URIs."""
    for img_id, base64_data_uri in images_dict.items():
        placeholder_pattern = re.compile(rf"!\[\s*{re.escape(img_id)}\s*\]\(\s*{re.escape(img_id)}\s*\)")
        markdown_str = placeholder_pattern.sub(f"![{img_id}]({base64_data_uri})", markdown_str)
    return markdown_str

def get_combined_markdown_optimized(ocr_response: OCRResponse) -> str:
    """Combines OCR text and images into a single markdown document."""
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        if page.images:
            for img in page.images:
                if img.image_base64 and not img.image_base64.startswith('data:image'):
                    image_data[img.id] = f"data:image/png;base64,{img.image_base64}"
                elif img.image_base64:
                    image_data[img.id] = img.image_base64
        page_md = page.markdown if hasattr(page, 'markdown') and page.markdown else ""
        markdowns.append(replace_images_in_markdown(page_md, image_data))
    return "\n\n".join(markdowns)

def display_pdf(pdf_data: bytes, height: int = 600) -> str:
    """Generates an HTML iframe tag to embed a PDF from bytes."""
    base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    pdf_html = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="{height}px" type="application/pdf"></iframe>'
    return pdf_html

# --- NEW FUNCTION: Generate Standalone HTML ---
def create_html_content(markdown_text: str) -> str:
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Output</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
        <style>
            .markdown-body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; }}
            @media (max-width: 767px) {{ .markdown-body {{ padding: 15px; }} }}
            img {{ max-width: 100%; }}
        </style>
        <script>MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }} }};</script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    </head>
    <body class="markdown-body">
        <div id="raw-markdown" style="display:none;">{markdown_text}</div>
        <div id="content"></div>
        <script>
            const rawMarkdown = document.getElementById('raw-markdown').textContent;
            document.getElementById('content').innerHTML = marked.parse(rawMarkdown);
            if (typeof MathJax !== 'undefined') {{ MathJax.typesetPromise(); }}
        </script>
    </body>
    </html>
    """
    return html_template

# --- Caching Functions ---

@st.cache_data(show_spinner="Downloading file from URL...")
def get_data_from_url(url: str) -> bytes:
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_data(show_spinner="Uploading file and running OCR...")
def get_ocr_result(api_key: str, file_data: bytes, file_name_stem: str, is_image: bool) -> str:
    client = None
    mistral_uploaded_file = None
    try:
        client = Mistral(api_key=api_key)
        upload_file_name = f"{file_name_stem}.tmp"
        mistral_uploaded_file = client.files.upload(
            file={"file_name": upload_file_name, "content": file_data},
            purpose="ocr"
        )
        signed_url = client.files.get_signed_url(file_id=mistral_uploaded_file.id, expiry=60)
        ocr_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        return get_combined_markdown_optimized(ocr_response)
    except Exception as e:
        raise e
    finally:
        if client and mistral_uploaded_file:
            try:
                client.files.delete(mistral_uploaded_file.id)
            except Exception:
                pass

# --- Helper for Toolbar ---
def append_syntax(syntax: str):
    """Appends markdown syntax to the current session state text."""
    if st.session_state.combined_markdown:
        st.session_state.combined_markdown += syntax
    else:
        st.session_state.combined_markdown = syntax

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Mistral OCR Interface")

localS = LocalStorage() 

# --- Initialize Session State ---
if 'combined_markdown' not in st.session_state: st.session_state.combined_markdown = None
if 'ocr_error' not in st.session_state: st.session_state.ocr_error = None
if 'current_file_name_stem' not in st.session_state: st.session_state.current_file_name_stem = "ocr_result"
if 'uploaded_file_data' not in st.session_state: st.session_state.uploaded_file_data = None
if 'is_image' not in st.session_state: st.session_state.is_image = False
if 'is_pdf' not in st.session_state: st.session_state.is_pdf = False

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    api_key_from_storage = localS.getItem("mistral_api_key")
    api_key = st.text_input("Enter Mistral API Key:", type="password", value=api_key_from_storage if api_key_from_storage else "")
    if api_key and api_key != api_key_from_storage:
        localS.setItem("mistral_api_key", api_key)
        st.success("API Key saved.")

    st.subheader("Upload File")
    upload_option = st.radio("Input method:", ("Upload from Computer", "Enter URL"), key="input_method")
    
    uploaded_file_data = None
    file_name_stem = "ocr_result"
    
    if upload_option == "Upload from Computer":
        uploaded_file = st.file_uploader("Choose PDF/Image", type=[ext.lstrip('.') for ext in SUPPORTED_FILE_TYPES])
        if uploaded_file:
            uploaded_file_data = uploaded_file.getvalue()
            st.session_state.uploaded_file_data = uploaded_file_data
            file_name_stem = Path(uploaded_file.name).stem
    else:
        file_url = st.text_input("Enter URL:")
        if file_url:
            uploaded_file_data = get_data_from_url(file_url)
            st.session_state.uploaded_file_data = uploaded_file_data
            file_name_stem = "file_from_url"

    if uploaded_file_data:
        st.session_state.current_file_name_stem = file_name_stem
        st.session_state.is_image = False # Simplified logic for checking type later
        if file_name_stem == "file_from_url" and file_url:
             st.session_state.is_image = any(file_url.lower().endswith(ext) for ext in SUPPORTED_IMAGE_TYPES)
             st.session_state.is_pdf = file_url.lower().endswith('.pdf')
        elif uploaded_file:
             st.session_state.is_image = uploaded_file.name.lower().endswith(SUPPORTED_IMAGE_TYPES)
             st.session_state.is_pdf = uploaded_file.name.lower().endswith('.pdf')

    # Process Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Run OCR", disabled=(not api_key or not uploaded_file_data), use_container_width=True):
            st.session_state.combined_markdown = None
            st.session_state.ocr_error = None
            if "markdown_editor" in st.session_state: del st.session_state.markdown_editor
            try:
                st.session_state.combined_markdown = get_ocr_result(api_key, uploaded_file_data, file_name_stem, st.session_state.is_image)
                st.success("OCR Complete!")
            except Exception as e:
                st.session_state.ocr_error = str(e)

    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.combined_markdown = None
            st.session_state.uploaded_file_data = None
            if "markdown_editor" in st.session_state: del st.session_state.markdown_editor
            st.rerun()

# --- Main Interface ---

if st.session_state.get('ocr_error'):
    st.error(f"OCR Error: {st.session_state.ocr_error}")

col_input, col_output = st.columns(2)

with col_input:
    st.subheader("Your File")
    if st.session_state.uploaded_file_data:
        if st.session_state.is_image:
            st.image(st.session_state.uploaded_file_data, use_container_width=True)
        elif st.session_state.is_pdf:
            st.markdown(display_pdf(st.session_state.uploaded_file_data), unsafe_allow_html=True)
    else:
        st.info("Upload a file to see it here.")

with col_output:
    st.subheader("OCR Results")
    if st.session_state.get('combined_markdown'):
        
        tab_edit, tab_preview = st.tabs(["✏️ Edit & Correct", "👁️ Preview"])
        
        with tab_edit:
            # --- TOOLBAR ---
            # We create small columns for the buttons
            t_col1, t_col2, t_col3, t_col4, t_col5, t_col6, t_col7 = st.columns([1,1,1,1,1,1,4])
            
            # Note: Because Streamlit can't track cursor position, these append to the END.
            if t_col1.button("B", help="Append Bold syntax"): append_syntax(" **bold text** ")
            if t_col2.button("I", help="Append Italic syntax"): append_syntax(" *italic text* ")
            if t_col3.button("H1", help="Append Header 1"): append_syntax("\n\n# Heading 1\n")
            if t_col4.button("H2", help="Append Header 2"): append_syntax("\n\n## Heading 2\n")
            if t_col5.button("List", help="Append Bullet List"): append_syntax("\n- List item\n")
            if t_col6.button("Num", help="Append Numbered List"): append_syntax("\n1. List item\n")

            # --- EDITOR ---
            edited_text = st.text_area(
                "Markdown Editor",
                value=st.session_state.combined_markdown,
                height=600,
                key="markdown_editor",
                label_visibility="collapsed",
                help="Type here to edit. Use toolbar buttons to append syntax to the end."
            )
            
            # Sync changes back to session state
            if edited_text != st.session_state.combined_markdown:
                st.session_state.combined_markdown = edited_text
        
        with tab_preview:
            with st.container(height=600):
                st.markdown(st.session_state.combined_markdown, unsafe_allow_html=True)

        # --- Download ---
        st.markdown("---")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button("⬇️ Download Markdown", st.session_state.combined_markdown, f"{st.session_state.current_file_name_stem}.md", "text/markdown", use_container_width=True)
        with d_col2:
            st.download_button("🌐 Download HTML", create_html_content(st.session_state.combined_markdown), f"{st.session_state.current_file_name_stem}.html", "text/html", use_container_width=True)
