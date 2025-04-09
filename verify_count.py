import chromadb

persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection("tutorial_docs")

print("Document count in vector store:", collection.count())
