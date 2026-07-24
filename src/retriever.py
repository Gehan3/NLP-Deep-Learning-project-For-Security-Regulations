from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DB_PATH = Path("iso27002_chroma_db")
COLLECTION_NAME = "iso27002_controls"

MODEL_NAME = "BAAI/bge-m3"

class ISO27002Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"Loaded Chroma collection '{COLLECTION_NAME}' with {self.collection.count()} chunks.")
        
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, k: int = 4) -> list[dict]:
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        formatted_results = []
        if results and results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
            
            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                score = 1.0 - dist if dist <= 1.0 else 0.0
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "control_id": meta.get("control_id", ""),
                    "section": meta.get("section", ""),
                    "parent_id": meta.get("parent_id", ""),
                    "score": score
                })
                
        return formatted_results

    def build_context(self, question: str, k=5, max_sources=4):
        rows = self.search(question, k=k)
        
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)

        selected = []
        seen_controls = set()

        for row in rows:
            if row["score"] <= 0:
                continue
            control_key = row["control_id"]
            if control_key in seen_controls:
                continue
                
            selected.append(row)
            seen_controls.add(control_key)
            
            if len(selected) == max_sources:
                break

        context = ""
        for source_number, row in enumerate(selected, start=1):
            context += f"[Source {source_number}] ISO 27002 Control {row['control_id']} - Section: {row['section'].upper()}\n{row['text']}\n\n"

        return context.strip(), selected

if __name__ == "__main__":
    retriever = ISO27002Retriever()

    sample_question = "Unauthorized access or insider escalation leading to unauthorized users gaining elevated privileges , which may result in the compromise of sensitive data, unauthorized system modifications, disruption of critical services, and potential regulatory non-compliance."
    #"How to manage privileged access rights?"
    ctx, sources = retriever.build_context(sample_question)
    print("\n--- Built Context ---")
    print(ctx)
    print(f"\nTotal sources selected: {len(sources)}")