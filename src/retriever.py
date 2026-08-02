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

        meta = self.collection.metadata or {}
        self.space = meta.get("hnsw:space", "l2")  # chroma's own default
        if self.space not in ("cosine", "l2", "ip"):
            print(f"WARNING: unrecognized hnsw:space '{self.space}', assuming l2")
            self.space = "l2"
        print(f"Collection distance space: {self.space}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def _distance_to_score(self, dist: float) -> float:
        if self.space == "cosine":
            return 1.0 - dist                    # chroma cosine distance = 1 - cos_sim
        elif self.space == "l2":
            return 1.0 - (dist / 2.0)             # squared L2 (normalized) = 2 - 2*cos_sim
        elif self.space == "ip":
            return 1.0 - dist                    # chroma ip distance = 1 - inner_product
        return 1.0 - dist

    def search(self, query: str, k: int = 4) -> list[dict]:
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
            #Problem 2: Wrong distance metric assumption in ChromaDB retriever
            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):  
                score = self._distance_to_score(dist)
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "control_id": meta.get("control_id", ""),
                    "section": meta.get("section", ""),
                    "parent_id": meta.get("parent_id", ""),
                    "score": score
                })
                
        return formatted_results

   

    def rerank_results(self, query: str, initial_results: list[dict], top_k: int = 4) -> list[dict]:
        if not initial_results:
            return []

        pairs = [[query, res["text"]] for res in initial_results]
        
        scores = self.reranker.predict(pairs)
        
        for i, res in enumerate(initial_results):
            res["rerank_score"] = float(scores[i])
            
        reranked = sorted(initial_results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    

    def build_context(
        self,
        question: str,
        k=10,
        max_sources=4,
        min_rerank_score: float = 0.0,   # add here: was -2.0, far too permissive
        min_score_ratio: float = 0.60,   # add here: new relative gate
    ):
        initial_rows = self.search(question, k=k)

        if not initial_rows:
            return "", []

        reranked_rows = self.rerank_results(question, initial_rows, top_k=max_sources)

        best_score = reranked_rows[0]["rerank_score"] if reranked_rows else 0.0

        selected = []
        seen_chunks = set()
        seen_texts = set()
        for row in reranked_rows:

        
            if row.get("rerank_score", 0.0) < min_rerank_score:
                continue
        
            if best_score > 0 and row.get("rerank_score", 0.0) < best_score * min_score_ratio:
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
            control_num = row['control_id']
            section_name = row['section'].upper()
            text_content = row['text']

            header = f"--- SOURCE {source_number} ---\n"
            ctrl_line = f"ISO 27002 Control Number: {control_num}\n"
            sec_line = f"Section: {section_name}\n"
            content_line = f"Content: {text_content}\n\n"

            context += header + ctrl_line + sec_line + content_line

        return context.strip(), selected


if __name__ == "__main__":
    retriever = ISO27002Retriever()

    print("\n--- Diagnostic Check ---")
    res = retriever.collection.get(include=["documents", "metadatas"])
    print(f"Total records in DB: {len(res['ids'])}")
    print(f"Collection hnsw:space in use: {retriever.space}")

    seen_docs = set()
    duplicates_count = 0
    for doc in res["documents"]:
        if doc in seen_docs:
            duplicates_count += 1
        else:
            seen_docs.add(doc)

    print(f"Number of exact duplicate documents found in DB: {duplicates_count}")
    print("-" * 30)

    
    sample_question = "Due to manual breach escalation processes, mandatory personal data breach notifications to authorities may miss the statutory 72-hour regulatory reporting window."
    ctx, sources = retriever.build_context(sample_question)
    print("\n--- Built Context (relevant question) ---")
    print(ctx)
    print(f"\nTotal sources selected: {len(sources)}")

    
    off_topic_question = "i use ml"
    ctx2, sources2 = retriever.build_context(off_topic_question)
    print("\n--- Built Context (off-topic question) ---")
    print(repr(ctx2))
    print(f"Total sources selected: {len(sources2)}")