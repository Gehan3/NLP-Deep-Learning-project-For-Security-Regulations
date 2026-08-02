from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("ISO27002_DB_PATH", str(BASE_DIR / "iso27002_chroma_db")))
COLLECTION_NAME = os.getenv("ISO27002_COLLECTION_NAME", "iso27002_controls")

MODEL_NAME = os.getenv("ISO27002_EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("ISO27002_RERANKER_MODEL", "BAAI/bge-reranker-large")


class ISO27002Retriever:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(
            f"Loaded Chroma collection '{COLLECTION_NAME}' "
            f"with {self.collection.count()} chunks."
        )

        metadata = self.collection.metadata or {}
        self.space = str(metadata.get("hnsw:space", "l2")).lower()
        if self.space not in {"cosine", "l2", "ip"}:
            print(
                f"WARNING: unrecognized hnsw:space '{self.space}', "
                "assuming l2"
            )
            self.space = "l2"

        print(f"Collection distance space: {self.space}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Convert a raw reranker logit into a bounded diagnostic score."""
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _distance_to_score(self, distance: float) -> float:
        
        if self.space in {"cosine", "ip"}:
            return 1.0 - distance
        if self.space == "l2":
            return 1.0 - (distance / 2.0)
        return 1.0 - distance

    def search(self, query: str, k: int = 12) -> list[dict[str, Any]]:
        clean_query = query.strip()
        if not clean_query:
            return []

        collection_size = self.collection.count()
        if collection_size <= 0:
            return []

        requested_k = max(1, min(int(k), collection_size))

        # update for retriever: use the same normalized BGE-M3 query embedding
        # configuration expected by a normalized Chroma collection.
        query_embedding = self.model.encode(
            clean_query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=requested_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            document = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 0.0
            metadata = metadata or {}

            formatted_results.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": str(document or ""),
                    "control_id": str(metadata.get("control_id", "")),
                    "section": str(metadata.get("section", "")),
                    "parent_id": str(metadata.get("parent_id", "")),
                    "metadata_context": str(metadata.get("metadata_context", "")),
                    "distance": distance,
                    "embedding_score": float(self._distance_to_score(distance)),
                    # Backward-compatible field used by existing UI/code.
                    "score": float(self._distance_to_score(distance)),
                }
            )

        return formatted_results

    def rerank_results(
        self,
        query: str,
        initial_results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not initial_results:
            return []

        pairs = [[query, result["text"]] for result in initial_results]
        raw_scores = self.reranker.predict(pairs)

        
        reranked: list[dict[str, Any]] = []
        for result, raw_score in zip(initial_results, raw_scores):
            row = dict(result)
            logit = float(raw_score)
            row["rerank_logit"] = logit
            row["rerank_score"] = self._sigmoid(logit)
            reranked.append(row)

        reranked.sort(key=lambda row: row["rerank_logit"], reverse=True)

        for rank, row in enumerate(reranked, start=1):
            row["rerank_rank"] = rank

        if top_k is None:
            return reranked
        return reranked[: max(0, int(top_k))]

    def build_context(
        self,
        question: str,
        k: int = 12,
        max_sources: int = 4,
        min_rerank_score: float | None = None,
        min_score_ratio: float | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Retrieve, rerank, filter, deduplicate, and format source chunks.

        ``min_rerank_score`` refers to the bounded sigmoid diagnostic score,
        not the raw CrossEncoder logit. Both thresholds are optional because
        uncalibrated global cutoffs can incorrectly remove relevant chunks.
        """
        clean_question = question.strip()
        if not clean_question:
            return "", []

        initial_rows = self.search(clean_question, k=k)
        if not initial_rows:
            return "", []

        # update for retriever: rerank the entire candidate pool first. Do not
        # truncate to max_sources before filtering and deduplication.
        reranked_rows = self.rerank_results(
            clean_question,
            initial_rows,
            top_k=None,
        )