import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from ollama import Client as OllamaClient

# Load embedding model and Chroma
model = SentenceTransformer("all-MiniLM-L6-v2")
persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection("tutorial_docs")

ollama = OllamaClient()

# UI
st.set_page_config(page_title="Ask Your Markdown")
st.title("Ask Questions About Your Markdown Data")

question = st.text_input("Enter your question:")

if question:
    # Embed and search
    embedding = model.encode(question)
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=3)

    if not results["documents"] or not results["documents"][0]:
        st.warning("No matching documents found.")
    else:
        retrieved = "\n\n".join(results["documents"][0])

        # Build prompt
        prompt = f"""
Use the following documents to answer the question.

Documents:
\"\"\"
{retrieved}
\"\"\"

Question: {question}
Answer:
"""

        # Call local LLM
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            st.code(retrieved)
