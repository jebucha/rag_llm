import os
import re
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb

# Load model and Chroma collection
model = SentenceTransformer("all-MiniLM-L6-v2")
persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection("tutorial_docs")

st.set_page_config(page_title="Ingest PDF Documents")
st.title("Ingest PDF Files into Vector Database")

uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def smart_chunk(text):
    chunks = re.split(r'\n{2,}', text)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 40]

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Processing: {file.name}")

        text = extract_text_from_pdf(file)
        chunks = smart_chunk(text)
        embeddings = model.encode(chunks)

        base = os.path.splitext(file.name)[0]
        ids = [f"{base}_chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=ids
        )

        st.success(f"Ingested {len(chunks)} chunks from {file.name}")

        with st.expander("Show preview chunks"):
            for i, chunk in enumerate(chunks[:5]):
                st.markdown(f"Chunk {i+1}")
                st.code(chunk)

    st.info("Ingestion complete. You can now query these documents using the RAG UI.")
