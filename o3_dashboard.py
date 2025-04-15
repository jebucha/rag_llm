import os
import re

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
from typing import List

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
        "mistral": "mistralai/Mistral-7B-v0.1",
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

def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_txt(file) -> str:
    """Extract text from a TXT file."""
    return file.read().decode("utf-8")

def extract_text_from_docx(file) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

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

# ----------------------------
# Application Tabs
# ----------------------------

tab_query, tab_ingest_pdf, tab_ingest_md, tab_verify = st.tabs([
    "Ask Questions",
    "Ingest PDF Files",
    "Ingest Markdown Files",
    "Verify Document Count"
])

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

            # (Optional) Check prompt token count against model's limits and adjust if necessary.
            # e.g., if token_count > MAX_TOKENS: truncate or summarize the prompt.

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
        for file in uploaded_files:
            st.write(f"Processing: {file.name}")
            ext = os.path.splitext(file.name)[1].lower()

            if ext == ".pdf":
                text = extract_text_from_pdf(file)
                file_type = "pdf"
            elif ext == ".txt":
                text = extract_text_from_txt(file)
                file_type = "txt"
            elif ext == ".docx":
                text = extract_text_from_docx(file)
                file_type = "docx"
            else:
                st.warning(f"Unsupported file type: {ext}")
                continue

            # Split the text into semantic chunks
            chunks = smart_chunk(text)
            embeddings = model.encode(chunks)
            base = os.path.splitext(file.name)[0]
            ids = [f"{base}_{file_type}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file.name, "type": file_type} for _ in chunks]

            collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)
            st.success(f"Ingested {len(chunks)} chunks from {file.name}")

            with st.expander("Show preview chunks"):
                for i, chunk in enumerate(chunks[:5]):
                    st.markdown(f"Chunk {i+1}")
                    st.code(chunk)

        st.info("Ingestion complete. You can now query these documents using the 'Ask Questions' tab.")

# ----------------------------
# Tab 3: Ingest Markdown Files
# ----------------------------

with tab_ingest_md:
    st.header("Ingest Markdown Documents")
    uploaded_md_files = st.file_uploader("Upload Markdown files", type="md", accept_multiple_files=True)

    if uploaded_md_files:
        collection = client.get_or_create_collection("tutorial_docs")
        for file in uploaded_md_files:
            st.write(f"Processing: {file.name}")
            text = file.read().decode("utf-8")
            chunks = smart_chunk_markdown(text)
            embeddings = model.encode(chunks)
            base = os.path.splitext(file.name)[0]
            ids = [f"{base}_md_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": file.name, "type": "markdown"} for _ in chunks]

            collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)
            st.success(f"Ingested {len(chunks)} chunks from {file.name}")

            with st.expander("Show preview chunks"):
                for i, chunk in enumerate(chunks[:5]):
                    st.markdown(f"Chunk {i+1}")
                    st.code(chunk)

        st.info("Ingestion complete. You can now query these documents using the 'Ask Questions' tab.")

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
