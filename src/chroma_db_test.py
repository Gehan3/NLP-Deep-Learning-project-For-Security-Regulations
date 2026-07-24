import chromadb
from pathlib import Path

DB_PATH = Path("iso27002_chroma_db")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection("iso27002_controls")
results = collection.peek(limit=5) 

print("IDs:", results["ids"])
print("\nMetadatas:", results["metadatas"])
print("\nDocuments (Text):", results["documents"])