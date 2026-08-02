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
        """
        Convert Chroma distance into a readable diagnostic similarity score.

        This value is not treated as a calibrated relevance probability. The
        L2 conversion is cosine-equivalent only when both stored and query
        embeddings were normalized during indexing and querying.
        """
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

        # update for retriever: preserve the raw BGE reranker logit for sorting,
        # and expose a sigmoid score only for diagnostics/optional filtering.
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

        best_normalized_score = (
            reranked_rows[0]["rerank_score"] if reranked_rows else None
        )

        selected: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        seen_texts: set[str] = set()

        for row in reranked_rows:
            normalized_score = float(row.get("rerank_score", 0.0))

            if (
                min_rerank_score is not None
                and normalized_score < min_rerank_score
            ):
                continue

            if (
                min_score_ratio is not None
                and best_normalized_score is not None
                and normalized_score < best_normalized_score * min_score_ratio
            ):
                continue

            chunk_key = str(row.get("chunk_id", ""))
            document_text = str(row.get("text", "")).strip()
            normalized_text_key = " ".join(document_text.casefold().split())

            if not document_text:
                continue
            if chunk_key in seen_chunks or normalized_text_key in seen_texts:
                continue

            selected_row = dict(row)
            selected_row["selected"] = True
            selected_row["selection_reason"] = (
                f"Top reranked result (rank {row.get('rerank_rank', '?')})"
            )
            selected.append(selected_row)
            seen_chunks.add(chunk_key)
            seen_texts.add(normalized_text_key)

            # update for retriever: stop only after valid, unique sources have
            # been selected, allowing lower-ranked candidates to backfill.
            if len(selected) >= max(1, int(max_sources)):
                break

        if not selected:
            return "", []

        context_blocks: list[str] = []
        for source_number, row in enumerate(selected, start=1):
            control_number = row.get("control_id") or "Not specified"
            section_name = (row.get("section") or "Not specified").upper()
            parent_id = row.get("parent_id") or "Not specified"
            text_content = row.get("text", "")

            # update for retriever: the marker now exactly matches the prompt's
            # required citation syntax: [SOURCE N].
            context_blocks.append(
                "\n".join(
                    [
                        f"[SOURCE {source_number}]",
                        f"ISO 27002 Control Number: {control_number}",
                        f"Section: {section_name}",
                        f"Parent ID: {parent_id}",
                        f"Content: {text_content}",
                    ]
                )
            )

        return "\n\n".join(context_blocks).strip(), selected


if __name__ == "__main__":
    retriever = ISO27002Retriever()

    print("\n--- Diagnostic Check ---")
    records = retriever.collection.get(include=["documents", "metadatas"])
    print(f"Total records in DB: {len(records['ids'])}")
    print(f"Collection hnsw:space in use: {retriever.space}")

    seen_documents: set[str] = set()
    duplicate_count = 0
    for document in records.get("documents", []):
        if document in seen_documents:
            duplicate_count += 1
        else:
            seen_documents.add(document)

    print(f"Number of exact duplicate documents found in DB: {duplicate_count}")
    print("-" * 30)

    sample_question = "unauthorized access"
    context, sources = retriever.build_context(sample_question)
    print("\n--- Built Context (relevant question) ---")
    print(context)
    print(f"\nTotal sources selected: {len(sources)}")

    for number, source in enumerate(sources, start=1):
        print(
            f"[SOURCE {number}] Control {source['control_id']} "
            f"({source['section']}) | distance={source['distance']:.4f} "
            f"| rerank_logit={source['rerank_logit']:.4f} "
            f"| rerank_score={source['rerank_score']:.4f}"
        )