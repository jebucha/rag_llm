import os
import re
import streamlit as st
import fitz  # PyMuPDF
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb
from ollama import Client as OllamaClient

# Load model and Chroma client
model = SentenceTransformer("all-MiniLM-L6-v2")
persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection("tutorial_docs")
ollama = OllamaClient()

# App layout
st.set_page_config(page_title="RAG Dashboard")
st.title("RAG Document Assistant")

# Tabs
tab_query, tab_ingest_pdf, tab_ingest_md = st.tabs(["Ask Questions", "Ingest PDF Files", "Ingest Markdown Files"])

# ----------------------------
# Tab 1: Ask Questions
# ----------------------------
with tab_query:
    st.header("Ask Questions About Your Documents")
    question = st.text_input("Enter your question:")
    top_n = st.slider("How many chunks to retrieve?", 1, 10, 3)
    model_choice = st.selectbox("Choose LLM model", ["mistral", "cogito", "gemma3"])

    if question:
        embedding = model.encode(question)
        results = collection.query(query_embeddings=[embedding.tolist()], n_results=top_n)

        docs = results.get("documents", [[]])[0]
        if not docs:
            st.warning("No matching documents found.")
        else:
            ranked_chunks = "\n\n".join([f"[Chunk {i+1}]\n{doc}" for i, doc in enumerate(docs)])
            prompt = f"""
You are a helpful assistant.

Use the following retrieved text chunks to answer the question.

Chunks:
\"\"\"
{ranked_chunks}
\"\"\"

Question: {question}
Answer:
"""

            # Estimate token count
            try:
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except:
                enc = tiktoken.get_encoding("cl100k_base")
            token_count = len(enc.encode(prompt))

            # Call LLM
            response = ollama.chat(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"].strip()

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
# Tab 2: Ingest PDF Files
# ----------------------------
with tab_ingest_pdf:
    st.header("Ingest PDF Documents")
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

        st.info("Ingestion complete. You can now query these documents using the 'Ask Questions' tab.")

# ----------------------------
# Tab 3: Ingest Markdown Files
# ----------------------------
with tab_ingest_md:
    st.header("Ingest Markdown Documents")
    uploaded_md_files = st.file_uploader("Upload one or more Markdown files", type="md", accept_multiple_files=True)

    def smart_chunk_markdown(text):
        # Split on headers, bullets, numbered items
        chunks = re.split(r'\n(?=(\d+\.\s|\-|\#))', text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    if uploaded_md_files:
        for file in uploaded_md_files:
            st.write(f"Processing: {file.name}")
            text = file.read().decode("utf-8")
            chunks = smart_chunk_markdown(text)
            embeddings = model.encode(chunks)

            base = os.path.splitext(file.name)[0]
            ids = [f"{base}_md_chunk_{i}" for i in range(len(chunks))]

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

        st.info("Ingestion complete. You can now query these documents using the 'Ask Questions' tab.")

