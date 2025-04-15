import os
import re
import traceback
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
# This must be the first Streamlit command
st.set_page_config(page_title="RAG Dashboard")
st.title("RAG Document Assistant")

import fitz  # PyMuPDF
import docx
import chromadb
from ollama import Client as OllamaClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ----------------------------
# Caching Resources
# ----------------------------

@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

@st.cache_resource
def load_tokenizer(model_choice: str) -> AutoTokenizer:
    """
    Load and cache the tokenizer based on the given model choice.
    """
    tokenizer_map = {
        "mistral": "intfloat/e5-mistral-7b-instruct",
        "gemma3": "google/generative-ai",
        "cogito": "NousResearch/Llama-2-7b-hf",
        "ALIENTELLIGENCE/contractanalyzerv2": "sentence-transformers/all-MiniLM-L6-v2",  # Fallback; update if a dedicated tokenizer is available.
    }
    model_id = tokenizer_map.get(model_choice, "sentence-transformers/all-MiniLM-L6-v2")
    return AutoTokenizer.from_pretrained(model_id)

# Load the embedding model once for the session
model = load_embedding_model()

# Setup persistent Chroma client and Ollama client
persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)
ollama = OllamaClient()


# ----------------------------
# Helper Functions
# ----------------------------

def extract_text_from_pdf(file) -> Tuple[str, bool]:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text, True
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}", False

def extract_text_from_txt(file) -> Tuple[str, bool]:
    """Extract text from a TXT file."""
    try:
        return file.read().decode("utf-8"), True
    except Exception as e:
        return f"Error extracting text from TXT: {str(e)}", False

def extract_text_from_docx(file) -> Tuple[str, bool]:
    """Extract text from a DOCX file using python-docx."""
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs]), True
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}", False

def smart_chunk(text: str, min_length: int = 40) -> List[str]:
    """
    Chunk text using a simple heuristic that splits on two or more newlines
    and filters out chunks that are too short.
    """
    chunks = [chunk.strip() for chunk in re.split(r'\n{2,}', text) if len(chunk.strip()) > min_length]
    return chunks

def smart_chunk_markdown(text: str) -> List[str]:
    """
    Chunk markdown text by splitting on patterns where list items or headers occur.
    """
    return [chunk.strip() for chunk in re.split(r'\n(?=(\d+\.\s|\-|\#))', text) if chunk.strip()]

def process_file(file, collection, file_type: str) -> Dict[str, Any]:
    """
    Process a single file and add it to the collection.
    Returns a dictionary with processing results.
    """
    result = {
        "filename": file.name,
        "success": False,
        "chunks": 0,
        "error": None
    }

    try:
        # Extract text based on file type
        if file_type == "pdf":
            text, success = extract_text_from_pdf(file)
        elif file_type == "txt":
            text, success = extract_text_from_txt(file)
        elif file_type == "docx":
            text, success = extract_text_from_docx(file)
        elif file_type == "markdown":
            text, success = extract_text_from_txt(file)
        else:
            return {**result, "error": f"Unsupported file type: {file_type}"}

        if not success:
            return {**result, "error": text}  # text contains error message

        # Split the text into semantic chunks
        chunks = smart_chunk_markdown(text) if file_type == "markdown" else smart_chunk(text)

        if not chunks:
            return {**result, "error": "No valid chunks extracted"}

        # Generate embeddings
        embeddings = model.encode(chunks)
        base = os.path.splitext(file.name)[0]
        ids = [f"{base}_{file_type}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file.name, "type": file_type} for _ in chunks]

        # Add to collection
        collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)

        return {
            **result,
            "success": True,
            "chunks": len(chunks)
        }

    except Exception as e:
        error_details = traceback.format_exc()
        return {**result, "error": f"{str(e)}\n{error_details}"}

# ----------------------------
# Application Tabs
# ----------------------------

tab_query, tab_ingest_pdf, tab_ingest_md, tab_verify, tab_errors = st.tabs([
    "Ask Questions",
    "Ingest PDF Files",
    "Ingest Markdown Files",
    "Verify Document Count",
    "Ingestion Errors"
])

# Store errors in session state for persistence
if 'ingestion_errors' not in st.session_state:
    st.session_state.ingestion_errors = []

# ----------------------------
# Tab 1: Ask Questions
# ----------------------------

with tab_query:
    st.header("Ask Questions About Your Documents")
    question = st.text_input("Enter your question:")
    top_n = st.slider("How many chunks to retrieve?", 1, 10, 3)
    model_choice = st.selectbox("Choose LLM model", ["mistral", "cogito", "gemma3", "ALIENTELLIGENCE/contractanalyzerv2"])

    if question:
        collection = client.get_or_create_collection("tutorial_docs")
        # Generate embedding for the query
        embedding = model.encode(question)
        results = collection.query(query_embeddings=[embedding.tolist()], n_results=top_n)
        docs = results.get("documents", [[]])[0]

        if not docs:
            st.warning("No matching documents found.")
        else:
            # Build the prompt from retrieved document chunks
            ranked_chunks = "\n\n".join([f"[Chunk {i+1}]\n{doc}" for i, doc in enumerate(docs)])
            prompt = (
                "You are a highly helpful assistant.\n\n"
                "Use the following retrieved text chunks to answer the question.\n\n"
                "Chunks:\n\"\"\"\n" +
                ranked_chunks +
                "\n\"\"\"\n\n" +
                f"Question: {question}\n" +
                "Answer:"
            )

            # Load and cache the tokenizer for the chosen model
            try:
                tokenizer = load_tokenizer(model_choice)
                token_count = len(tokenizer.encode(prompt))
            except Exception as e:
                st.warning(f"Tokenizer error: {e}")
                token_count = 0

            # Generate the response using the LLM via Ollama
            response = ollama.chat(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"].strip()

            # Display the answer and additional information
            st.subheader("Answer")
            st.write(answer)
            st.markdown(f"Prompt token count: {token_count}")

            with st.expander("Show retrieved context"):
                for i, doc in enumerate(docs):
                    st.markdown(f"Chunk {i+1}")
                    st.code(doc)

            with st.expander("Show prompt sent to model"):
                st.code(prompt)

# ----------------------------
# Tab 2: Ingest PDF/TXT/DOCX Files
# ----------------------------

with tab_ingest_pdf:
    st.header("Ingest PDF, TXT, or DOCX Documents")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    if uploaded_files:
        collection = client.get_or_create_collection("tutorial_docs")

        # Create a progress bar for overall progress
        progress_bar = st.progress(0.0)

        # Create a status area for current file
        current_file_text = st.empty()

        # Process files with progress tracking
        total_files = len(uploaded_files)
        successful_files = 0
        failed_files = 0
        total_chunks = 0

        for i, file in enumerate(uploaded_files):
            # Update current file display
            current_file_text.text(f"Ingesting: {file.name} ({i+1}/{total_files})")

            # Update progress bar
            progress_bar.progress((i) / total_files)

            # Determine file type
            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                file_type = "pdf"
            elif ext == ".txt":
                file_type = "txt"
            elif ext == ".docx":
                file_type = "docx"
            else:
                failed_files += 1
                st.session_state.ingestion_errors.append({
                    "filename": file.name,
                    "error": f"Unsupported file type: {ext}"
                })
                continue

            # Process the file
            result = process_file(file, collection, file_type)

            if result["success"]:
                successful_files += 1
                total_chunks += result["chunks"]
            else:
                failed_files += 1
                st.session_state.ingestion_errors.append({
                    "filename": file.name,
                    "error": result["error"]
                })

            # Update progress bar
            progress_bar.progress((i+1) / total_files)

        # Clear the current file text
        current_file_text.empty()

        # Complete the progress bar
        progress_bar.progress(1.0)

        # Show final summary
        st.info(f"Ingestion complete. Successfully processed {successful_files} of {total_files} files ({total_chunks} total chunks).")

        if failed_files > 0:
            st.warning(f"{failed_files} files failed to process. See the 'Ingestion Errors' tab for details.")

# ----------------------------
# Tab 3: Ingest Markdown Files
# ----------------------------

with tab_ingest_md:
    st.header("Ingest Markdown Documents")
    uploaded_md_files = st.file_uploader("Upload Markdown files", type="md", accept_multiple_files=True)

    if uploaded_md_files:
        collection = client.get_or_create_collection("tutorial_docs")

        # Create a progress bar for overall progress
        progress_bar = st.progress(0.0)

        # Create a status area for current file
        current_file_text = st.empty()

        # Process files with progress tracking
        total_files = len(uploaded_md_files)
        successful_files = 0
        failed_files = 0
        total_chunks = 0

        for i, file in enumerate(uploaded_md_files):
            # Update current file display
            current_file_text.text(f"Ingesting: {file.name} ({i+1}/{total_files})")

            # Update progress bar
            progress_bar.progress((i) / total_files)

            # Process the file
            result = process_file(file, collection, "markdown")

            if result["success"]:
                successful_files += 1
                total_chunks += result["chunks"]
            else:
                failed_files += 1
                st.session_state.ingestion_errors.append({
                    "filename": file.name,
                    "error": result["error"]
                })

            # Update progress bar
            progress_bar.progress((i+1) / total_files)

        # Clear the current file text
        current_file_text.empty()

        # Complete the progress bar
        progress_bar.progress(1.0)

        # Show final summary
        st.info(f"Ingestion complete. Successfully processed {successful_files} of {total_files} files ({total_chunks} total chunks).")

        if failed_files > 0:
            st.warning(f"{failed_files} files failed to process. See the 'Ingestion Errors' tab for details.")

# ----------------------------
# Tab 4: Verify Document Count
# ----------------------------

with tab_verify:
    st.header("Verify Stored Document Count")

    try:
        all_collections = client.list_collections()
        st.subheader(f"Total collections: {len(all_collections)}")

        for coll_info in all_collections:
            name = coll_info.name
            col_key = f"delete_{name}"

            with st.container():
                col = client.get_or_create_collection(name)
                count = col.count()

                st.markdown(f"**Collection:** `{name}` — **Documents:** {count}")

                if st.button(f"Delete Collection '{name}'", key=col_key):
                    client.delete_collection(name)
                    st.success(f"Collection '{name}' deleted.")
                    st.experimental_rerun()

                # Load up to 100 documents for filtering and preview
                preview = col.get(include=["metadatas", "documents"], limit=100)
                all_metadatas = preview.get("metadatas", [])
                available_keys = {k for meta in all_metadatas if isinstance(meta, dict) for k in meta.keys()}

                # Optional filtering by metadata key/value
                if available_keys:
                    filter_key = st.selectbox(f"Filter by metadata key in '{name}'", sorted(available_keys), key=f"fk_{name}")
                    filter_value = st.text_input(f"Value to match for `{filter_key}`:", key=f"fv_{name}")

                    filtered_docs = []
                    filtered_ids = []
                    for i, meta in enumerate(all_metadatas):
                        if meta.get(filter_key) == filter_value:
                            filtered_docs.append(preview["documents"][i])
                            filtered_ids.append(preview["ids"][i])
                else:
                    filtered_docs = preview["documents"]
                    filtered_ids = preview["ids"]

                with st.expander(f"Show preview for collection '{name}'"):
                    for i in range(min(5, len(filtered_docs))):
                        st.markdown(f"**ID:** `{filtered_ids[i]}`")
                        st.code(filtered_docs[i])
                        if i < len(all_metadatas):
                            meta = all_metadatas[i]
                            if meta:
                                st.markdown("Metadata:")
                                st.json(meta)

    except Exception as e:
        st.error("Error retrieving collection info.")
        st.code(str(e))

# ----------------------------
# Tab 5: Ingestion Errors
# ----------------------------

with tab_errors:
    st.header("Ingestion Errors")

    if not st.session_state.ingestion_errors:
        st.info("No errors recorded during ingestion.")
    else:
        error_count = len(st.session_state.ingestion_errors)
        st.warning(f"{error_count} errors recorded during ingestion.")

        if st.button("Clear Error Log"):
            st.session_state.ingestion_errors = []
            st.experimental_rerun()

        # Group errors by type for a summary view
        error_types = {}
        for error in st.session_state.ingestion_errors:
            error_msg = error["error"].split('\n')[0]  # Get first line of error
            if error_msg in error_types:
                error_types[error_msg].append(error["filename"])
            else:
                error_types[error_msg] = [error["filename"]]

        # Display error summary
        st.subheader("Error Summary")
        for error_msg, filenames in error_types.items():
            with st.expander(f"{error_msg} ({len(filenames)} files)"):
                st.write("Affected files:")
                for filename in filenames:
                    st.write(f"- {filename}")

        # Option to view detailed errors
        if st.checkbox("Show detailed error logs"):
            for i, error in enumerate(st.session_state.ingestion_errors):
                with st.expander(f"Error details for: {error['filename']}"):
                    st.code(error['error'])
