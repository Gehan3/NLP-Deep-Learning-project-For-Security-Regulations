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
            #Problem 2: Wrong distance metric assumption in ChromaDB retriever
            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                # With cosine distance metric: distance = 1 - cosine_similarity.
                # score = 1 - dist recovers the cosine similarity directly.
                # No clamping — negative similarity (dist > 1) is a valid signal
                # and already filtered by the score <= 0 check in build_context.
                score = 1.0 - dist
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
        #update for retriev diff answer
        """selected = []
        seen_controls = set() 

        for row in rows:
            if row["score"] <= 0:
                continue
            control_key = row["control_id"]
            if control_key in seen_controls:  # <--- وهنا يمنع الكنترول بالكامل
                continue"""
         
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)

        selected = []
        seen_chunks = set()
        seen_texts = set()  
        for row in rows:
            if row["score"] <= 0:
                continue
            
            chunk_key = row["chunk_id"] 
            doc_text = row["text"].strip()

            if chunk_key in seen_chunks or doc_text in seen_texts:
                continue
                
            selected.append(row)
            seen_chunks.add(chunk_key)
            seen_texts.add(doc_text)
            
            if len(selected) == max_sources:
                break

        context = ""
        for source_number, row in enumerate(selected, start=1):
            context += f"[Source {source_number}] ISO 27002 Control {row['control_id']} - Section: {row['section'].upper()}\n{row['text']}\n\n"

        return context.strip(), selected

if __name__ == "__main__":
    retriever = ISO27002Retriever()

    print("\n--- Diagnostic Check ---")
    res = retriever.collection.get(include=["documents", "metadatas"])
    print(f"Total records in DB: {len(res['ids'])}")

    seen_docs = set()
    duplicates_count = 0
    for doc in res["documents"]:
        if doc in seen_docs:
            duplicates_count += 1
        else:
            seen_docs.add(doc)

    print(f"Number of exact duplicate documents found in DB: {duplicates_count}")
    print("-" * 30)
    # -----------------------------------------------------

    sample_question = "Due to manual breach escalation processes, mandatory personal data breach notifications to authorities may miss the statutory 72-hour regulatory reporting window."
    
    ctx, sources = retriever.build_context(sample_question)
    print("\n--- Built Context ---")
    print(ctx)
    print(f"\nTotal sources selected: {len(sources)}")