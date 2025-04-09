from sentence_transformers import SentenceTransformer
import chromadb
from ollama import Client as OllamaClient

# question = "What is the relatioin of llm prompts and the Pareto Principle?"


# Prompt user for input
question = input("Enter your question: ").strip()

# SAME model used during ingestion
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(question)

persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection("tutorial_docs")

results = collection.query(query_embeddings=[embedding.tolist()], n_results=3)

if not results["documents"] or not results["documents"][0]:
    print("No matching documents found.")
    exit()

retrieved = "\n\n".join(results["documents"][0])

ollama = OllamaClient()
prompt = f"""
Use the following documents to answer the question.

Documents:
\"\"\"
{retrieved}
\"\"\"

Question: {question}
Answer:
"""

response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}]
)

print("Answer:", response["message"]["content"].strip())
