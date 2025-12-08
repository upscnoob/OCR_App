import streamlit as st
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk
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

# --- NEW FUNCTION: Generate Publish-Ready HTML (Fixed Color Issue) ---
def create_html_content(markdown_text: str, page_title: str) -> str:
    """
    Wraps the markdown in a standalone HTML template.
    Forces Light Mode styles to prevent white-on-white text in Dark Mode.
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="generator" content="Mistral OCR">
        <title>{page_title}</title>
        
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
        
        <style>
            /* Force Light Mode Variables to override system Dark Mode */
            :root {{
                --color-canvas-default: #ffffff;
                --color-canvas-subtle: #f6f8fa;
                --color-border-default: #d0d7de;
                --color-fg-default: #24292f;
                --color-fg-muted: #57606a;
                --color-accent-fg: #0969da;
            }}

            body {{
                background-color: var(--color-canvas-subtle);
                font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif;
                margin: 0;
                padding: 20px;
                color: var(--color-fg-default); /* Force dark text */
            }}
            .markdown-body {{
                box-sizing: border-box;
                min-width: 200px;
                max-width: 980px;
                margin: 0 auto;
                padding: 45px;
                background-color: var(--color-canvas-default);
                border: 1px solid var(--color-border-default);
                border-radius: 6px;
                box-shadow: 0 3px 6px rgba(140, 149, 159, 0.15);
                color: #24292f !important; /* CRITICAL: Force text to be black/dark gray */
            }}
            @media (max-width: 767px) {{
                .markdown-body {{
                    padding: 15px;
                }}
                body {{
                    padding: 10px;
                }}
            }}
            img {{
                max-width: 100%;
                display: block;
                margin: 1em auto;
            }}
            @media print {{
                body {{ background-color: white; }}
                .markdown-body {{ border: none; box-shadow: none; padding: 0; }}
            }}
        </style>
        
        <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true
            }}
        }};
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    </head>
    <body>
        <div id="raw-markdown" style="display:none;">{markdown_text}</div>
        
        <article class="markdown-body" id="content">
            <p style="color:#666; text-align:center;">Loading content...</p>
        </article>

        <script>
            const rawMarkdown = document.getElementById('raw-markdown').textContent;
            document.getElementById('content').innerHTML = marked.parse(rawMarkdown);
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
            document=DocumentURLChunk(document_url=signed_url.url),
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
    
    # 3. Try to get the key from browser storage
    api_key_from_storage = localS.getItem("mistral_api_key")
    
    # 4. Use the stored key as the default value for the text input
    api_key = st.text_input(
        "Enter your Mistral API Key:",
        type="password",
        value=api_key_from_storage if api_key_from_storage else ""
    )

    # 5. If the user entered a new key, save it to storage
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
            original_name_or_url
