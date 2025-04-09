import os
import glob
import re
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Directory containing Markdown files
md_directory = "/home/john/repos/lists"  # Adjust this to your directory path
persist_path = "./chroma_db"

# Find all Markdown files (*.md) in directory
md_files = glob.glob(os.path.join(md_directory, "*.md"))
assert md_files, f"No Markdown files found in directory: {md_directory}"

# Setup Chroma DB
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection("tutorial_docs")

# Optional: clear existing data
try:
    client.delete_collection("tutorial_docs")
    collection = client.get_or_create_collection("tutorial_docs")
except ValueError:
    pass

# Process each Markdown file
total_chunks = 0
for md_file in md_files:
    with open(md_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Smart chunking (headers, lists, numbers)
    chunks = re.split(r'\n(?=(\d+\.\s|\-|\#))', full_text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Embed chunks
    embeddings = model.encode(chunks)

    # Generate unique IDs
    base_filename = os.path.splitext(os.path.basename(md_file))[0]
    ids = [f"{base_filename}_chunk_{i}" for i in range(len(chunks))]

    # Add to collection
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    total_chunks += len(chunks)
    print(f"Ingested {len(chunks)} chunks from {md_file}")

print(f"Finished ingesting {total_chunks} total chunks from {len(md_files)} files.")
