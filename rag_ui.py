import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from ollama import Client as OllamaClient
import tiktoken

# Load embedding model and Chroma DB
model = SentenceTransformer("all-MiniLM-L6-v2")
persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection("tutorial_docs")
ollama = OllamaClient()

# Streamlit UI setup
st.set_page_config(page_title="Ask Your Markdown")
st.title("Ask Questions About Your Markdown Data")

question = st.text_input("Enter your question:")

# Select how many results to retrieve
top_n = st.slider("How many top matching chunks to retrieve?", min_value=1, max_value=10, value=3)

# Model selection
model_choice = st.selectbox("Choose Ollama model", ["mistral", "llama2", "gemma"])

# Run search
if question:
    embedding = model.encode(question)
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=top_n)

    docs = results.get("documents", [[]])[0]
    if not docs:
        st.warning("No matching documents found.")
        st.stop()

    # Build display string and prompt
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

    # Token count (using tiktoken for estimate)
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback
    token_count = len(enc.encode(prompt))

    # LLM response
    response = ollama.chat(
        model=model_choice,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"].strip()

    # Display
    st.subheader("Answer")
    st.write(answer)

    st.markdown(f"**Prompt token count:** {token_count}")

    with st.expander("Show retrieved context (ranked)"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(doc)

    with st.expander("Show full prompt sent to model"):
        st.code(prompt)
