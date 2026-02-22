import streamlit as st
from pathlib import Path
from src.util.vectorstore import get_all_collection_names
import re
import os
from src.ingest.simple_ingest import simple_ingest
from src.ingest.advanced_ingest import advanced_ingest

def sanitize_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    clean_name = re.sub(r'[^a-z0-9]', '_', name.lower())
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    return f"{clean_name}{ext}"

def is_valid_name(name: str) -> bool:
    pattern = r'^[a-z0-9_]+$'
    return bool(re.match(pattern, name))

st.set_page_config(page_title="Ingest Textbook")

st.title("Textbook Processing")
st.markdown("Upload the book pdf to process and add it to your Vector Database.")


INGEST_METHODS = {
    "Simple Chunking (Fixed Size)": "simple",
    "Advanced chunking (chapter-based)": "chapter",
    # "Recursive Character (Paragraphs)": "recursive"
}

with st.container(border=True):
    uploaded_file = st.file_uploader("Choose a PDF file of the Textbook", type="pdf")
    
    selected_method = st.selectbox(
        "Select Ingestion Strategy:",
        options=list(INGEST_METHODS.keys())
    )
    
    collection_name = st.text_input(
        "Collection Name:", 
        placeholder="e.g., simple_500, chapter_12_recursive",
        help="This is the name that will appear in the chat dropdown."
    )
    st.caption("Use only **lowercase letters, numbers, and underscores** (e.g., `my_data_01`). No spaces.")
    clean_name = collection_name.strip()

    if st.button("Start Ingestion", use_container_width=True):
        if not uploaded_file:
            st.error("Please upload the textbook first.")
        elif not clean_name:
            st.error("Please provide a collection name.")
        elif not is_valid_name(clean_name):
            st.error("Invalid format! Please use only lowercase letters, numbers, and underscores (no spaces or special characters).")
        else:
            original_name = uploaded_file.name    
            safe_name = sanitize_filename(original_name)

            existing_collections = get_all_collection_names()
            
            if collection_name in existing_collections:
                st.error(
                    f"The collection '{collection_name}' already exists. "
                    "To prevent strategy mixing, please choose a new name or delete the existing collection manually."
                )
            else:
                with st.status("Processing document...", expanded=True) as status:
                    temp_dir = Path("data/raw")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    file_path = temp_dir / safe_name
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.write(f"File saved locally: `{uploaded_file.name}`")
                    
                    method_key = INGEST_METHODS[selected_method]
                    st.write(f"Running `{method_key}` strategy...")
                    
                    try:
                        if method_key == "simple":
                            num_chunks = simple_ingest(file_path, collection_name)
                        elif method_key == "chapter":
                            num_chunks = advanced_ingest(file_path,collection_name)
                            
                        status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                        st.success(f"Ingested **{num_chunks}** chunks into the collection: `{clean_name}`.")
                        st.balloons()
                        
                    except Exception as e:
                        status.update(label="Ingestion Failed", state="error")
                        st.error(f"Error: {str(e)}")