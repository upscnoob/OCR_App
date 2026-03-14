import streamlit as st
from mistralai import Mistral
from mistralai.models import OCRResponse
from pathlib import Path
import requests
import re
import io
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
    """
    Wraps the markdown in a standalone HTML template with MathJax and Marked.js 
    for perfect rendering in any browser.
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Output</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
        <style>
            .markdown-body {{
                box-sizing: border-box;
                min-width: 200px;
                max-width: 980px;
                margin: 0 auto;
                padding: 45px;
            }}
            @media (max-width: 767px) {{
                .markdown-body {{
                    padding: 15px;
                }}
            }}
            /* Ensure images fit within the container */
            img {{
                max-width: 100%;
            }}
        </style>
        <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
            }}
        }};
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    </head>
    <body class="markdown-body">
        <div id="raw-markdown" style="display:none;">{markdown_text}</div>
        
        <div id="content"></div>

        <script>
            // Get raw markdown
            const rawMarkdown = document.getElementById('raw-markdown').textContent;
            
            // Render Markdown to HTML
            document.getElementById('content').innerHTML = marked.parse(rawMarkdown);
            
            // Trigger MathJax to process the new content
            if (typeof MathJax !== 'undefined') {{
                MathJax.typesetPromise();
            }}
        </script>
    </body>
    </html>
    """
    return html_template

# --- Caching Functions ---

@st.cache_data(show_spinner="Downloading file from URL...")
def get_data_from_url(url: str) -> bytes:
    """Downloads file from URL and caches the result."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"URL download error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected URL error: {e}")
        return None

@st.cache_data(show_spinner="Uploading file and running OCR... This may take a moment.")
def get_ocr_result(api_key: str, file_data: bytes, file_name_stem: str, is_image: bool) -> str:
    """
    Runs the full OCR process and CACHES the result.
    CRITICALLY: Cleans up the uploaded file from Mistral servers.
    """
    client = None
    mistral_uploaded_file = None
    try:
        client = Mistral(api_key=api_key)
        
        # 1. Upload file to Mistral (temporary)
        upload_file_name = f"{file_name_stem}.tmp"
        mistral_uploaded_file = client.files.upload(
            file={"file_name": upload_file_name, "content": file_data},
            purpose="ocr"
        )
        
        # 2. Get a short-lived URL for processing
        signed_url = client.files.get_signed_url(file_id=mistral_uploaded_file.id, expiry=60)
        
        # 3. Run OCR
        ocr_response = client.ocr.process(
            document={
                "type": "document_url",
                "document_url": signed_url.url
            },
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        
        # 4. Get the final markdown
        return get_combined_markdown_optimized(ocr_response)

    except Exception as e:
        raise e
        
    finally:
        if client and mistral_uploaded_file:
            try:
                client.files.delete(mistral_uploaded_file.id)
                print(f"Successfully deleted temporary file: {mistral_uploaded_file.id}")
            except Exception as e:
                print(f"Error deleting file {mistral_uploaded_file.id}: {e}")


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
    
    # --- API Key Input ---
    api_key_from_storage = localS.getItem("mistral_api_key")
    api_key = st.text_input(
        "Enter your Mistral API Key:",
        type="password",
        value=api_key_from_storage if api_key_from_storage else ""
    )
    if api_key and api_key != api_key_from_storage:
        localS.setItem("mistral_api_key", api_key)
        st.success("API Key saved to browser.")

    # --- File Input ---
    st.subheader("Upload File or Enter URL")
    upload_option = st.radio("Choose input method:", ("Upload from Computer", "Enter URL"), key="input_method")
    
    uploaded_file_data = None
    file_name_stem = "ocr_result"
    is_image = False
    is_pdf = False
    original_name_or_url = ""

    if upload_option == "Upload from Computer":
        uploaded_file = st.file_uploader(
            "Choose a PDF or Image file",
            type=[ext.lstrip('.') for ext in SUPPORTED_FILE_TYPES],
            key="file_uploader"
        )
        if uploaded_file is not None:
            uploaded_file_data = uploaded_file.getvalue()
            original_name_or_url = uploaded_file.name
            st.session_state.uploaded_file_data = uploaded_file_data 
    
    elif upload_option == "Enter URL":
        file_url = st.text_input("Enter File URL (PDF or Image):", key="url_input")
        if file_url:
            if not (file_url.startswith("http://") or file_url.startswith("https://")):
                st.error("Invalid URL format. Must start with http:// or https://")
            else:
                uploaded_file_data = get_data_from_url(file_url) 
                original_name_or_url = file_url
                st.session_state.uploaded_file_data = uploaded_file_data 

    # --- Process File Name and Type ---
    if uploaded_file_data:
        try:
            path_part = original_name_or_url.split('/')[-1].split('?')[0]
            file_name_stem = Path(path_part).stem if path_part else "file_from_url"
        except Exception:
            file_name_stem = "file_from_url"
        
        is_image = any(original_name_or_url.lower().endswith(ext) for ext in SUPPORTED_IMAGE_TYPES)
        is_pdf = original_name_or_url.lower().endswith('.pdf')
        
        st.session_state.current_file_name_stem = file_name_stem
        st.session_state.is_image = is_image
        st.session_state.is_pdf = is_pdf
    else:
        st.session_state.uploaded_file_data = None


    # --- Process & Clear Buttons ---
    col1, col2 = st.columns(2)
    
    with col1:
        run_disabled = (not api_key or not uploaded_file_data)
        if st.button("🚀 Run OCR", disabled=run_disabled, key="run_button", use_container_width=True):
            st.session_state.combined_markdown = None
            st.session_state.ocr_error = None
            
            # --- CRITICAL FIX: Reset the editor state so it picks up the new text ---
            if "markdown_editor" in st.session_state:
                del st.session_state.markdown_editor

            try:
                st.session_state.combined_markdown = get_ocr_result(
                    api_key,
                    uploaded_file_data,
                    file_name_stem,
                    is_image 
                )
                st.success("OCR Complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.ocr_error = str(e)

    with col2:
        if st.button("🧹 Clear", key="clear_button", use_container_width=True):
            st.session_state.combined_markdown = None
            st.session_state.ocr_error = None
            st.session_state.current_file_name_stem = "ocr_result"
            st.session_state.uploaded_file_data = None
            st.session_state.is_image = False
            st.session_state.is_pdf = False
            if "markdown_editor" in st.session_state:
                del st.session_state.markdown_editor
            st.rerun()

# --- Display Results ---

if st.session_state.get('ocr_error'):
    st.error(f"Last OCR attempt failed: {st.session_state.ocr_error}")

col_input, col_output = st.columns(2)

with col_input:
    st.subheader("Your File")
    if st.session_state.uploaded_file_data:
        if st.session_state.is_image:
            st.image(st.session_state.uploaded_file_data, use_container_width=True)
            
        elif st.session_state.is_pdf:
            pdf_html = display_pdf(st.session_state.uploaded_file_data, height=600)
            st.markdown(pdf_html, unsafe_allow_html=True)
            
    else:
        st.info("Your uploaded file will be displayed here.")

with col_output:
    st.subheader("OCR Results (Editable)")
    if st.session_state.get('combined_markdown'):
        
        # --- NEW: Tabs for Editing vs Previewing ---
        tab_edit, tab_preview = st.tabs(["✏️ Edit & Correct", "👁️ Preview"])
        
        with tab_edit:
            # The Text Area. We use 'markdown_editor' key to track state.
            # The 'value' is set to the OCR result initially.
            # On subsequent edits, Streamlit updates 'markdown_editor' automatically.
            edited_text = st.text_area(
                "Edit the markdown below to fix typos:",
                value=st.session_state.combined_markdown,
                height=600,
                key="markdown_editor",
                label_visibility="collapsed"
            )
            
            # --- CRITICAL: Update the main session state with your edits ---
            if edited_text != st.session_state.combined_markdown:
                st.session_state.combined_markdown = edited_text

        with tab_preview:
            with st.container(height=600): 
                st.markdown(st.session_state.combined_markdown, unsafe_allow_html=True)

        # --- Download Buttons (Now use the EDITED text) ---
        st.markdown("---")
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            st.download_button(
                label="⬇️ Download Markdown (.md)",
                data=st.session_state.combined_markdown,
                file_name=f"{st.session_state.current_file_name_stem}_ocr_result.md",
                mime="text/markdown",
                key="download_markdown_button_top",
                help="Downloads the raw Markdown code (including your edits).",
                use_container_width=True
            )
        
        with btn_col2:
            # Generate the HTML content using the EDITED text
            html_content = create_html_content(st.session_state.combined_markdown)
            
            st.download_button(
                label="🌐 Download HTML Report",
                data=html_content,
                file_name=f"{st.session_state.current_file_name_stem}_ocr_result.html",
                mime="text/html",
                key="download_html_button_top",
                help="Downloads a standalone HTML file with your corrections.",
                use_container_width=True
            )
    else:
        st.info("OCR results will be rendered here.")
