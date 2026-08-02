from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer ,CrossEncoder

DB_PATH = Path("iso27002_chroma_db")
COLLECTION_NAME = "iso27002_controls"

MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
class ISO27002Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_PATH), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"Loaded Chroma collection '{COLLECTION_NAME}' with {self.collection.count()} chunks.")
        
        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
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

    # [Anti-Hallucination Update]
    # Why: Removed dead code that was placed after a return statement (THRESHOLD = -1.0 check).
    # The threshold logic is now properly implemented in build_context() via the min_rerank_score
    # parameter — see that method for the live version.

    def rerank_results(self, query: str, initial_results: list[dict], top_k: int = 4) -> list[dict]:
        if not initial_results:
            return []

        pairs = [[query, res["text"]] for res in initial_results]
        
        scores = self.reranker.predict(pairs)
        
        for i, res in enumerate(initial_results):
            res["rerank_score"] = float(scores[i])
            
        reranked = sorted(initial_results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    

    def build_context(self, question: str, k=10, max_sources=4, min_rerank_score: float = -2.0):
        initial_rows = self.search(question, k=k)

        if not initial_rows:
            return "", []

        reranked_rows = self.rerank_results(question, initial_rows, top_k=max_sources)

        selected = []
        seen_chunks = set()
        seen_texts = set()
        for row in reranked_rows:
        
            if row.get("rerank_score", 0.0) < min_rerank_score:
                continue

            chunk_key = row["chunk_id"]
            doc_text = row["text"].strip()

            if chunk_key in seen_chunks or doc_text in seen_texts:
                continue

            selected.append(row)
            seen_chunks.add(chunk_key)
            seen_texts.add(doc_text)

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