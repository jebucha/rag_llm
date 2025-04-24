import os
# Add this line at the very top to disable tokenizers parallelism and suppress the fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#!/usr/bin/env python3
import re
import json
from typing import List, Dict, Any

import streamlit as st  # pip install streamlit
import fitz  # PyMuPDF, pip install pymupdf
import docx  # pip install python-docx
import chromadb  # pip install chromadb
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
from transformers import AutoTokenizer  # pip install transformers
import requests

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
# Further reduced chunk size based on previous discussion and 21k token document example
# This helps ensure the *combined* retrieved chunks + query + instructions
# fit within the LLM's context window (likely 8192 for Llama 3 1B).
MAX_TOKENS     = 100 # Adjusted for safety
STRIDE_TOKENS  = 40 # Adjusted for overlap relative to new MAX_TOKENS
# Updated model key for clarity
MODEL_CHOICES  = ["mistral", "cogito", "gemma3", "llama3-1b-instruct"]

# LLM Server URL (for the generative model)
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

# Define the actual context window for the Llama 3 1B model
# Adjust this value if your specific model/server setup has a different limit.
LLM_CONTEXT_WINDOW = 8192 # Llama 3 1B Instruct default is 8192

# ----------------------------
# Caching Resources
# ----------------------------
@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Loads the sentence transformer model for creating embeddings."""
    st.info(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop() # Stop execution if essential model fails to load

@st.cache_resource
def load_tokenizer_for_chunking() -> AutoTokenizer:
    """Loads a general-purpose tokenizer for text chunking."""
    # Using GPT-2 tokenizer for chunking is common and reasonably effective.
    # The warning about sequence length exceeding 1024 originates here
    # when encoding the full document text if it's > 1024 tokens.
    # This warning is expected for long documents and doesn't prevent chunking.
    st.info("Loading tokenizer for chunking (gpt2)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        st.success("Chunking tokenizer loaded.")
        return tokenizer
    except Exception as e:
        st.error(f"Error loading chunking tokenizer: {e}")
        st.stop() # Stop execution if essential resource fails

@st.cache_resource
def load_tokenizer_for_llm(model_choice: str) -> AutoTokenizer:
    """Loads the specific tokenizer for the chosen generative LLM."""
    tokenizer_map = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "cogito": "NousResearch/Llama-2-7b-hf",
        "gemma3": "google/gemma-2b-it",
        "llama3-1b-instruct": "hugging-quants/Llama-3.2-1B-Instruct-Q4_0-GGUF",
    }
    model_id = tokenizer_map.get(model_choice)
    if not model_id:
         st.error(f"Unknown LLM model choice: {model_choice}")
         # Fallback or raise error
         model_id = "gpt2" # Fallback to a basic tokenizer
         st.warning(f"Using fallback tokenizer: {model_id}")


    st.info(f"Loading tokenizer for LLM ({model_choice} - {model_id})...")
    try:
        # Add revision="main" or specific commit hash if needed
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        st.success("LLM tokenizer loaded.")
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer for {model_choice} ({model_id}): {e}")
        # Return a fallback or raise error depending on desired behavior
        st.warning("Could not load specific LLM tokenizer, falling back to gpt2. Prompt token count may be inaccurate.")
        return AutoTokenizer.from_pretrained("gpt2")


# ----------------------------
# Helper Functions
# ----------------------------
def chunk_text_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = MAX_TOKENS,
    stride: int = STRIDE_TOKENS
) -> List[str]:
    """Splits text into token-based chunks with overlap."""
    if not text:
        return []
    # tokenizer.encode will issue a warning if text > tokenizer.model_max_length (e.g., 1024 for gpt2),
    # but it still returns the full token IDs when truncation=False.
    # This warning during ingest is generally safe to ignore as the code
    # correctly slices the token_ids list afterwards.
    try:
        token_ids = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    except Exception as e:
        st.error(f"Error encoding text for chunking: {e}")
        return []

    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        # Decode the chunk token IDs back to text
        chunk_text = tokenizer.decode(chunk_ids).strip()
        if chunk_text: # Only add non-empty chunks
             chunks.append(chunk_text)

        if end >= len(token_ids):
            break
        # Move back by stride for overlap
        start = end - stride
        # Ensure start doesn't go below 0 or beyond text length - stride
        start = max(0, start)
        if start >= len(token_ids): # Avoid infinite loop in edge cases
            break
        # Also handle case where stride is very large, preventing progress
        if start == end - stride and stride >= max_tokens:
             st.warning(f"Stride ({stride}) is larger than or equal to max_tokens ({max_tokens}). Chunking might not progress.")
             break # Prevent infinite loop

    return chunks


# Simplified extraction functions - assuming standard text content
def extract_text_from_pdf(file) -> str:
    """Extracts text from a PDF file."""
    try:
        # Read file content into memory stream for PyMuPDF
        file_content = file.read()
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_txt(file) -> str:
    """Extracts text from a TXT file."""
    try:
        # Ensure file pointer is at the beginning
        file.seek(0)
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text from TXT: {e}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extracts text from a DOCX file."""
    try:
        # python-docx can read directly from the UploadedFile object
        doc = docx.Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_md(file) -> str:
    """Extracts text from a Markdown file."""
    try:
        # Ensure file pointer is at the beginning
        file.seek(0)
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text from MD: {e}")
        return ""


# ----------------------------
# Initialize Clients & Resources
# ----------------------------
# Load the embedding model
model = load_embedding_model()
# Load the tokenizer specifically for chunking
chunk_tokenizer = load_tokenizer_for_chunking()
# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path=PERSIST_PATH)
except Exception as e:
    st.error(f"Error initializing ChromaDB client: {e}")
    st.stop() # Stop if database cannot be initialized


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
def query_llama_server(llm_url: str, messages: list) -> str:
    """Queries the LLM server with a list of messages."""
    payload = {
        # NOTE: The model name here might need to match what the server expects
        # depending on your llama.cpp server configuration.
        "model": "llama3-1b-instruct", # Use a consistent model name if the server needs it
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024, # Limit response length from the LLM
        # Add other parameters like top_p, stop sequences if needed
    }
    try:
        resp = requests.post(llm_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Safely access the content, handling potential missing keys
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            st.warning("LLM server returned an empty response.")
        return content

    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to LLM server at {llm_url}. Please ensure the server is running.")
        return "Error: Could not connect to the LLM server."
    except requests.exceptions.Timeout:
        st.error(f"Timeout Error: The request to the LLM server at {llm_url} timed out.")
        return "Error: LLM server request timed out."
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying LLM server at {llm_url}: {e}")
        return "Error: Could not get a response from the LLM server."
    except KeyError:
        st.error("Error: Unexpected response format from LLM server.")
        return "Error: Unexpected response format from LLM server."
    except Exception as e:
        st.error(f"An unexpected error occurred while querying LLM: {e}")
        return "Error: An unexpected error occurred."


with tab_query:
    st.header("Ask Questions About Your Documents")
    question       = st.text_input("Enter your question:")
    # Reduced max value for top_n to help control prompt length
    # 6 chunks * 100 tokens/chunk = 600 tokens, which should be safe
    # within an 8192-token context window after adding query/instructions.
    top_n          = st.slider("How many chunks to retrieve?", 1, 6, 3)
    model_choice = st.selectbox("Choose LLM model", MODEL_CHOICES, index=MODEL_CHOICES.index("llama3-1b-instruct") if "llama3-1b-instruct" in MODEL_CHOICES else 0)

    # Use a button to trigger the query
    if st.button("Get Answer", key="query_button") and question:
        if model_choice != "llama3-1b-instruct":
             st.warning(f"Please select the 'llama3-1b-instruct' model choice to query the local server URL specified.")
             # Optionally add logic here to handle querying other models if desired
        else:
            with st.spinner("Processing your question..."):
                try:
                    # 1) Retrieve embeddings + docs
                    collection = client.get_or_create_collection("tutorial_docs")
                    if collection.count() == 0:
                         st.warning("Database is empty. Please ingest documents first.")
                         st.stop()

                    embedding  = model.encode(question)
                    results    = collection.query(
                        query_embeddings=[embedding.tolist()],
                        n_results=top_n,
                        include=["documents", "metadatas"] # Include metadatas for display/debugging
                    )
                    docs = results.get("documents", [[]])[0]
                    metas = results.get("metadatas", [[]])[0]

                    if not docs:
                        st.warning("No matching documents found in the database.")
                    else:
                        # 2) Build prompt using chat template
                        ranked_chunks = "\n\n".join(f"[Chunk {i+1}] (Source: {metas[i].get('source', 'N/A')}, Index: {metas[i].get('chunk_index', 'N/A')})\n{d}" for i, (d, metas) in enumerate(zip(docs, metas)))

                        # Load the specific tokenizer for the LLM for correct formatting and counting
                        llm_tokenizer = load_tokenizer_for_llm(model_choice)

                        # Use the chat template recommended by Hugging Face for instruction models
                        # This is the standard way to format prompts for models like Llama 3 Instruct
                        messages = [
                            {"role": "system", "content": "You are a highly helpful assistant."},
                            {"role": "user", "content": f"Use the following retrieved text chunks to answer the question. Cite the source and chunk index for each piece of information used in your answer.\n\nChunks:\n\"\"\"\n{ranked_chunks}\n\"\"\"\n\nQuestion: {question}\nAnswer:"}
                        ]

                        # Apply the chat template to get the final prompt string
                        # add_generation_prompt=True adds the final prompt token(s) for the model to start generating
                        # Setting tokenize=False ensures we get the string representation
                        prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


                        # 3) Count tokens using the LLM's tokenizer
                        # The token count is based on the string generated by apply_chat_template
                        try:
                            token_count = len(llm_tokenizer.encode(prompt))
                        except Exception as e:
                            st.warning(f"Could not encode prompt for token count: {e}. Token count may be inaccurate.")
                            token_count = -1 # Indicate unknown count


                        # Optional: Add a check if token_count exceeds the LLM context window
                        if token_count != -1 and token_count > LLM_CONTEXT_WINDOW:
                             st.warning(f"Warning: Prompt ({token_count} tokens) exceeds the assumed LLM context window ({LLM_CONTEXT_WINDOW} tokens). The LLM may truncate the input, affecting the quality of the answer.")
                             # You might choose to truncate the prompt here before sending,
                             # but handling truncation client-side based on tokenizer limits
                             # is complex and often best left to the server if possible.
                             # The current setup sends the full prompt and relies on the server to handle it.


                        # 4) Query llama‑server
                        # Pass the single formatted prompt string within a user message
                        # Note: The llama.cpp /v1/chat/completions endpoint expects the messages structure,
                        # but the content itself will be the string from apply_chat_template.
                        answer = query_llama_server(
                            LLAMA_SERVER_URL,
                            [{"role": "user", "content": prompt}]
                        )

                        # 5) Display results
                        st.subheader("Answer")
                        st.write(answer)
                        if token_count != -1:
                            st.markdown(f"**Prompt tokens:** {token_count}") # Display token count
                        else:
                            st.markdown("**Prompt tokens:** Could not count")


                        with st.expander("Retrieved Chunks and Metadata"):
                            for i, (d, meta) in enumerate(zip(docs, metas)):
                                st.markdown(f"**Chunk {i+1}**")
                                st.code(d)
                                if meta:
                                     st.json(meta)

                        with st.expander("Full Prompt Sent to LLM"):
                             st.code(prompt)

                except Exception as e:
                    st.error(f"An error occurred during the query process: {e}")
                    # Optional: Display traceback for debugging
                    # import traceback
                    # st.code(traceback.format_exc())


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
        # Check if the collection is empty or if user wants to overwrite
        if collection.count() > 0:
            st.warning(f"Collection '{collection.name}' already contains {collection.count()} documents.")
            col_ingest_overwrite, col_ingest_clear = st.columns(2)
            with col_ingest_clear:
                 if st.button("Clear Existing Documents Before Ingesting"):
                     st.info("Clearing existing documents...")
                     try:
                         client.delete_collection("tutorial_docs")
                         collection = client.get_or_create_collection("tutorial_docs")
                         st.success("Existing documents cleared.")
                         # No rerun needed if we just cleared, the UI updates
                         # automatically based on the new collection state.
                     except Exception as e:
                         st.error(f"Error clearing collection: {e}")

            # Option to continue without clearing
            if st.checkbox("Add new documents to the existing collection", value=True, key="add_to_existing_checkbox"):
                 pass # Continue with ingestion
            else:
                 st.info("Upload new files to ingest them into a fresh or existing collection.")
                 uploaded = [] # Clear uploaded list if not adding to existing

        # Process uploaded files if any remain after checking overwrite/clear
        if uploaded:
            for file in uploaded:
                name = file.name
                st.write(f"Processing: :blue[{name}]") # Use markdown for color
                ext = os.path.splitext(name)[1].lower()
                text = ""
                ftype = "unknown"

                # Ensure the file pointer is reset for each file if reading multiple times
                file.seek(0)

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
                    text = extract_text_from_md(file)
                    ftype = "markdown"
                else:
                    st.warning(f"Unsupported file type: {ext}")
                    continue

                if not text:
                    st.warning(f"Could not extract text from :orange[{name}]. Skipping.") # Use markdown for color
                    continue

                # Use the general chunking function with the loaded chunk tokenizer
                st.info(f"Chunking text from :blue[{name}] using MAX_TOKENS={MAX_TOKENS}, STRIDE_TOKENS={STRIDE_TOKENS}...")
                # This is where the "Token indices sequence length" warning might appear
                # if the full 'text' is longer than the chunk_tokenizer's model_max_length (e.g., 1024).
                # This is expected for long documents and doesn't prevent chunking.
                chunks = chunk_text_by_tokens(text, chunk_tokenizer, max_tokens=MAX_TOKENS, stride=STRIDE_TOKENS)

                if not chunks:
                     st.warning(f"No chunks generated from :orange[{name}]. Skipping.") # Use markdown for color
                     continue

                try:
                    st.info(f"Creating embeddings for {len(chunks)} chunks from :blue[{name}]...")
                    # Using batch size can help with memory for very large documents
                    # The SentenceTransformer.encode method handles batching internally, but you
                    # can control it with the `batch_size` parameter if needed.
                    embeddings = model.encode(chunks, show_progress_bar=True)
                    st.success(f"Embeddings created for :blue[{name}].")

                    base = os.path.splitext(name)[0]
                    # Generate unique IDs based on filename and chunk index
                    # Ensure IDs are truly unique across ingestion runs if adding to existing collection
                    # A simple approach is to add a timestamp or UUID if overwriting isn't guaranteed.
                    # For this example, file_base_index is assumed unique enough if files have distinct names.
                    ids = [f"{base}_{i}" for i in range(len(chunks))]
                    # Add chunk index and original text length to metadata
                    metas = [{"source": name, "type": ftype, "chunk_index": i, "original_text_len": len(text), "chunk_len": len(chunks[i])} for i in range(len(chunks))]

                    st.info(f"Adding {len(chunks)} chunks to ChromaDB for :blue[{name}]...")
                    # ChromaDB handles batches internally, but explicit batching can be done if needed
                    # for extremely large numbers of chunks or memory issues.
                    collection.add(
                        documents=chunks,
                        embeddings=embeddings.tolist(),
                        ids=ids,
                        metadatas=metas
                    )
                    st.success(f"Successfully ingested {len(chunks)} chunks from :green[{name}].")

                    with st.expander(f"Preview First 10 Chunks from {name}"):
                        for i, (c, meta) in enumerate(zip(chunks[:10], metas[:10])):
                            st.markdown(f"**Chunk {i+1}** (ID: `{ids[i]}`)")
                            st.code(c)
                            if meta:
                                 st.json(meta)

                except Exception as e:
                    st.error(f"An error occurred during ingestion of :orange[{name}]: {e}")
                    # Optional: Display traceback for debugging
                    # import traceback
                    # st.code(traceback.format_exc())


            st.info("Ingestion process complete. Switch to 'Ask Questions' to query.")

# ----------------------------
# Tab 3: Verify Collections
# ----------------------------
with tab_verify:
    st.header("Verify Stored Collections")
    try:
        cols = client.list_collections()
        st.subheader(f"Total: {len(cols)} collections")
        if not cols:
            st.info("No collections found in the database.")
        else:
            for ci in cols:
                name = ci.name
                col  = client.get_or_create_collection(name)
                count = col.count()
                st.markdown(f"**{name}** — {count} documents")

                col1, col2 = st.columns([1, 3]) # Adjust column widths
                with col1:
                    if st.button(f"Delete '{name}'", key=f"del_{name}"):
                        st.info(f"Deleting '{name}'...")
                        try:
                            client.delete_collection(name)
                            st.success(f"Deleted '{name}'")
                            st.experimental_rerun() # Rerun to update the list
                        except Exception as e:
                             st.error(f"Error deleting collection: {e}")


                with col2:
                     # Display number of items in the collection
                     st.write("") # Placeholder for alignment

                # Display preview and filter options only if the collection is not about to be deleted
                # Check session_state to see if the delete button was clicked in this run
                if not st.session_state.get(f"del_{name}", False):
                    preview = col.get(include=["documents","metadatas"], limit=100)
                    metas = preview.get("metadatas", [])
                    docs  = preview.get("documents", [])
                    ids   = preview.get("ids", [])
                    # Collect all unique keys from metadatas for filtering
                    available_keys = sorted({k for m in metas if isinstance(m, dict) for k in m.keys()})

                    filtered_data = list(zip(ids, docs, metas))

                    if available_keys:
                        # Filtering UI
                        st.markdown("Filter Collection Preview:")
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            fk = st.selectbox(f"Filter key ({name})", ["All"] + available_keys, key=f"fk_{name}")
                        with filter_col2:
                            fv = st.text_input(f"Value for '{fk}'", key=f"fv_{name}", disabled=(fk == "All"))

                        if fk != "All" and fv:
                            filtered_data = [
                                (ids[i], docs[i], metas[i]) for i in range(len(ids))
                                if isinstance(metas[i], dict) and metas[i].get(fk) == fv
                            ]

                    with st.expander(f"Preview '{name}' (showing {len(filtered_data)} of {len(ids)} items)"):
                        if not filtered_data:
                            st.info("No items to display based on current filter or limit.")
                        for idx, doc, md in filtered_data[:10]: # Limit preview to 10 items
                            st.markdown(f"**ID:** `{idx}`")
                            st.code(doc)
                            if md:
                                 st.json(md)

    except Exception as e:
        st.error("Error listing or interacting with collections")
        st.code(str(e))