
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
ollama = OllamaClient()

# App layout
st.set_page_config(page_title="RAG Dashboard")
st.title("RAG Document Assistant")

# Tabs
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
    model_choice = st.selectbox("Choose LLM model", ["mistral", "cogito", "gemma3"])

if question:
    collection = client.get_or_create_collection("tutorial_docs")
    embedding = model.encode(question)
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=top_n)

    docs = results.get("documents", [[]])[0]
    if not docs:
        st.warning("No matching documents found.")
    else:
        # Create prompt text from ranked chunks
        ranked_chunks = "\n\n".join([f"[Chunk {i+1}]\n{doc}" for i, doc in enumerate(docs)])
        prompt = f"""You are a helpful assistant.

Use the following retrieved text chunks to answer the question.

Chunks:
\"\"\"
{ranked_chunks}
\"\"\"

Question: {question}
Answer:"""

        try:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(prompt))

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
# Tab 2: Ingest PDF/TXT/DOCX Files
# ----------------------------
with tab_ingest_pdf:
    st.header("Ingest PDF, TXT, or DOCX Documents")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    def extract_text_from_pdf(file):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])

    def extract_text_from_txt(file):
        return file.read().decode("utf-8")

    def extract_text_from_docx(file):
        import docx
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    def smart_chunk(text):
        return [chunk.strip() for chunk in re.split(r'\n{2,}', text) if len(chunk.strip()) > 40]

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

    def smart_chunk_markdown(text):
        return [chunk.strip() for chunk in re.split(r'\n(?=(\d+\.\s|\-|\#))', text) if chunk.strip()]

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

                # Optional filter by metadata key/value
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

                # Show chunk preview
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
