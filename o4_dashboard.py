#!/usr/bin/env python3
import os
import re
import json
from typing import List

import streamlit as st  # pip install streamlit
import fitz  # PyMuPDF, pip install pymupdf
import docx  # pip install python-docx
import chromadb  # pip install chromadb
from ollama import Client as OllamaClient  # pip install ollama
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
from transformers import AutoTokenizer  # pip install transformers

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="RAG Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("RAG Document Assistant")

# ----------------------------
# Constants
# ----------------------------
PERSIST_PATH   = "./chroma_db"
INPUT_TYPES    = ["pdf", "txt", "docx", "md"]
MAX_TOKENS     = 100
STRIDE_TOKENS  = 25
MODEL_CHOICES  = ["mistral", "cogito", "gemma3"]

# ----------------------------
# Caching Resources
# ----------------------------
@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

@st.cache_resource
def load_chunk_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")

@st.cache_resource
def load_tokenizer(model_choice: str) -> AutoTokenizer:
    tokenizer_map = {
        "mistral": "intfloat/e5-mistral-7b-instruct",
        "cogito": "NousResearch/Llama-2-7b-hf",
    }
    model_id = tokenizer_map.get(model_choice, "sentence-transformers/all-MiniLM-L6-v2")
    return AutoTokenizer.from_pretrained(model_id)

# ----------------------------
# Helper Functions
# ----------------------------
def chunk_text_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = MAX_TOKENS,
    stride: int = STRIDE_TOKENS
) -> List[str]:
    token_ids = tokenizer.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunks.append(tokenizer.decode(token_ids[start:end]).strip())
        if end >= len(token_ids):
            break
        start = end - stride
    return chunks


def split_json_and_plain(text: str) -> (List[str], str):
    json_lines: List[str] = []
    plain_lines: List[str] = []
    for line in text.splitlines():
        try:
            json.loads(line)
            json_lines.append(line)
        except json.JSONDecodeError:
            plain_lines.append(line)
    return json_lines, "\n".join(plain_lines)


def super_chunk(text: str, tokenizer: AutoTokenizer) -> List[str]:
    json_lines, plain = split_json_and_plain(text)
    chunks: List[str] = []
    for jl in json_lines:
        chunks.extend(chunk_text_by_tokens(jl, tokenizer))
    chunks.extend(chunk_text_by_tokens(plain, tokenizer))
    return [c for c in chunks if c]


def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8")


def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

# ----------------------------
# Initialize Clients & Resources
# ----------------------------
model = load_embedding_model()
chunk_tokenizer = load_chunk_tokenizer()
client = chromadb.PersistentClient(path=PERSIST_PATH)
ollama = OllamaClient()

# ----------------------------
# Application Tabs
# ----------------------------
tab_query, tab_ingest, tab_verify = st.tabs([
    "Ask Questions",
    "Ingest Documents",
    "Verify Collections"
])

# ----------------------------
# Tab 1: Query / RAG
# ----------------------------
with tab_query:
    st.header("Ask Questions About Your Documents")
    question     = st.text_input("Enter your question:")
    top_n        = st.slider("How many chunks to retrieve?", 1, 10, 3)
    model_choice = st.selectbox("Choose LLM model", MODEL_CHOICES)

    if question:
        collection = client.get_or_create_collection("tutorial_docs")
        embedding  = model.encode(question)
        results    = collection.query(query_embeddings=[embedding.tolist()], n_results=top_n)
        docs       = results.get("documents", [[]])[0]

        if not docs:
            st.warning("No matching documents found.")
        else:
            ranked = "\n\n".join(f"[Chunk {i+1}]\n{d}" for i, d in enumerate(docs))
            prompt = (
                "You are a highly helpful assistant.\n\n"
                "Use the following retrieved text chunks to answer the question.\n\n"
                "Chunks:\n\"\"\"\n" + ranked + "\n\"\"\"\n\n"
                f"Question: {question}\nAnswer:"
            )

            try:
                tokenizer  = load_tokenizer(model_choice)
                token_count = len(tokenizer.encode(prompt))
            except Exception as e:
                st.warning(f"Tokenizer error: {e}")
                token_count = 0

            response = ollama.chat(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"].strip()

            st.subheader("Answer")
            st.write(answer)
            st.markdown(f"**Prompt tokens:** {token_count}")

            with st.expander("Retrieved Chunks"):
                for i, d in enumerate(docs):
                    st.markdown(f"Chunk {i+1}")
                    st.code(d)

            with st.expander("Full Prompt"):
                st.code(prompt)

# ----------------------------
# Tab 2: Ingest Documents (PDF, TXT, DOCX, MD)
# ----------------------------
with tab_ingest:
    st.header("Ingest PDF, TXT, DOCX, or Markdown")
    uploaded = st.file_uploader(
        "Upload files", type=INPUT_TYPES, accept_multiple_files=True
    )
    if uploaded:
        collection = client.get_or_create_collection("tutorial_docs")
        for file in uploaded:
            name = file.name
            st.write(f"Processing: {name}")
            ext = os.path.splitext(name)[1].lower()
            if ext == ".pdf":
                text = extract_text_from_pdf(file)
                ftype = "pdf"
            elif ext == ".txt":
                text = extract_text_from_txt(file)
                ftype = "txt"
            elif ext == ".docx":
                text = extract_text_from_docx(file)
                ftype = "docx"
            elif ext == ".md":
                text = file.read().decode("utf-8")
                ftype = "markdown"
            else:
                st.warning(f"Unsupported file type: {ext}")
                continue

            chunks = super_chunk(text, chunk_tokenizer)
            embeddings = model.encode(chunks)
            base = os.path.splitext(name)[0]
            ids = [f"{base}_{i}" for i in range(len(chunks))]
            metas = [{"source": name, "type": ftype} for _ in chunks]

            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metas
            )
            st.success(f"Ingested {len(chunks)} chunks from {name}")

            with st.expander("Preview Chunks"):
                for i, c in enumerate(chunks[:5]):
                    st.markdown(f"Chunk {i+1}")
                    st.code(c)
        st.info("Ingestion complete. Switch to 'Ask Questions' to query.")

# ----------------------------
# Tab 3: Verify Collections
# ----------------------------
with tab_verify:
    st.header("Verify Stored Collections")
    try:
        cols = client.list_collections()
        st.subheader(f"Total: {len(cols)} collections")
        for ci in cols:
            name = ci.name
            col  = client.get_or_create_collection(name)
            count = col.count()
            st.markdown(f"**{name}** — {count} documents")
            if st.button(f"Delete '{name}'", key=f"del_{name}"):
                client.delete_collection(name)
                st.success(f"Deleted '{name}'")
                st.experimental_rerun()

            preview = col.get(include=["documents","metadatas"], limit=100)
            metas = preview.get("metadatas", [])
            docs  = preview.get("documents", [])
            ids   = preview.get("ids", [])
            available = {k for m in metas if isinstance(m, dict) for k in m.keys()}

            if available:
                fk = st.selectbox(f"Filter key ({name})", sorted(available), key=f"fk_{name}")
                fv = st.text_input(f"Value for `{fk}`", key=f"fv_{name}")
                filtered = [
                    (ids[i], docs[i], metas[i]) for i in range(len(ids)) if metas[i].get(fk) == fv
                ]
            else:
                filtered = list(zip(ids, docs, metas))

            with st.expander(f"Preview '{name}'"):
                for idx, doc, md in filtered[:5]:
                    st.markdown(f"**ID:** `{idx}`")
                    st.code(doc)
                    if md:
                        st.json(md)

    except Exception as e:
        st.error("Error listing collections")
        st.code(str(e))
