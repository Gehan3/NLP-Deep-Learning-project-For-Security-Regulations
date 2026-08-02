from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.nn import Identity


COLLECTION_NAME = "iso27002_controls"
MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

DEFAULT_MIN_EMBEDDING_SCORE = float(
    os.getenv("ISO27002_MIN_EMBEDDING_SCORE", "0.45")
)

DEFAULT_MIN_RERANK_SCORE = float(
    os.getenv("ISO27002_MIN_RERANK_SCORE", "0.05")
)


def resolve_db_path() -> Path:
    env_path = os.getenv("ISO27002_DB_PATH", "").strip()

    if env_path:
        return Path(env_path).expanduser().resolve()

    project_root = Path(__file__).resolve().parent.parent
    project_db_path = project_root / "iso27002_chroma_db"

    if project_db_path.exists():
        return project_db_path.resolve()

    return (Path.cwd() / "iso27002_chroma_db").resolve()


class ISO27002Retriever:
    def __init__(self):
        self.db_path = resolve_db_path()

        print(f"Current working directory: {Path.cwd()}")
        print(f"Resolved ChromaDB path: {self.db_path}")
        print(f"Database directory exists: {self.db_path.exists()}")

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        available_collections = self.client.list_collections()

        collection_names = [
            collection.name if hasattr(collection, "name") else str(collection)
            for collection in available_collections
        ]

        print(f"Available Chroma collections: {collection_names}")

        if COLLECTION_NAME not in collection_names:
            raise RuntimeError(
                "\nChroma collection was not found.\n"
                f"Expected collection: {COLLECTION_NAME}\n"
                f"Resolved database path: {self.db_path}\n"
                f"Available collections: {collection_names}\n\n"
                "Run the Chroma database builder from the project root:\n"
                "python .\\src\\chroma_db.py"
            )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Loaded Chroma collection '{COLLECTION_NAME}' "
            f"with {self.collection.count()} chunks."
        )

        metadata = self.collection.metadata or {}

        self.space = str(
            metadata.get("hnsw:space", "l2")
        ).lower()

        if self.space not in {"cosine", "l2", "ip"}:
            self.space = "l2"

        print(f"Collection distance space: {self.space}")

        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    @staticmethod
    def _sigmoid(value: float) -> float:
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

    def search(
        self,
        query: str,
        k: int = 12,
    ) -> list[dict[str, Any]]:
        clean_query = query.strip()

        if not clean_query:
            return []

        collection_size = self.collection.count()

        if collection_size <= 0:
            return []

        requested_k = max(
            1,
            min(int(k), collection_size),
        )

        query_embedding = self.model.encode(
            clean_query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=requested_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        if (
            not results
            or not results.get("ids")
            or not results["ids"][0]
        ):
            return []

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results: list[dict[str, Any]] = []

        for index, chunk_id in enumerate(ids):
            document = (
                documents[index]
                if index < len(documents)
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                float(distances[index])
                if index < len(distances)
                else 0.0
            )

            metadata = metadata or {}

            embedding_score = self._distance_to_score(
                distance
            )

            formatted_results.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": str(document or ""),
                    "control_id": str(
                        metadata.get("control_id", "")
                    ),
                    "section": str(
                        metadata.get("section", "")
                    ),
                    "parent_id": str(
                        metadata.get("parent_id", "")
                    ),
                    "metadata_context": str(
                        metadata.get(
                            "metadata_context",
                            "",
                        )
                    ),
                    "distance": distance,
                    "embedding_score": float(
                        embedding_score
                    ),
                    "score": float(embedding_score),
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

        pairs = [
            [query, result["text"]]
            for result in initial_results
        ]

        raw_scores = self.reranker.predict(
            pairs,
            activation_fn=Identity(),
        )

        reranked: list[dict[str, Any]] = []

        for result, raw_score in zip(
            initial_results,
            raw_scores,
        ):
            row = dict(result)

            rerank_logit = float(raw_score)
            rerank_score = self._sigmoid(
                rerank_logit
            )

            row["rerank_logit"] = rerank_logit
            row["rerank_score"] = rerank_score

            reranked.append(row)

        reranked.sort(
            key=lambda row: row["rerank_logit"],
            reverse=True,
        )

        for rank, row in enumerate(
            reranked,
            start=1,
        ):
            row["rerank_rank"] = rank

        if top_k is None:
            return reranked

        return reranked[:max(0, int(top_k))]

    def build_context(
        self,
        question: str,
        k: int = 12,
        max_sources: int = 4,
        min_embedding_score: float = DEFAULT_MIN_EMBEDDING_SCORE,
        min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
        min_score_ratio: float | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        clean_question = question.strip()

        if not clean_question:
            return "", []

        initial_rows = self.search(
            clean_question,
            k=k,
        )

        if not initial_rows:
            return "", []

        reranked_rows = self.rerank_results(
            clean_question,
            initial_rows,
            top_k=None,
        )

        if not reranked_rows:
            return "", []

        best_result = reranked_rows[0]

        best_embedding_score = float(
            best_result.get(
                "embedding_score",
                float("-inf"),
            )
        )

        best_rerank_score = float(
            best_result.get(
                "rerank_score",
                0.0,
            )
        )

        if (
            best_embedding_score
            < min_embedding_score
            and best_rerank_score
            < min_rerank_score
        ):
            return "", []

        selected: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        seen_texts: set[str] = set()

        for row in reranked_rows:
            embedding_score = float(
                row.get(
                    "embedding_score",
                    float("-inf"),
                )
            )

            rerank_score = float(
                row.get(
                    "rerank_score",
                    0.0,
                )
            )

            if (
                embedding_score
                < min_embedding_score
                and rerank_score
                < min_rerank_score
            ):
                continue

            if (
                min_score_ratio is not None
                and best_rerank_score > 0
                and rerank_score
                < best_rerank_score
                * min_score_ratio
            ):
                continue

            chunk_key = str(
                row.get("chunk_id", "")
            ).strip()

            document_text = str(
                row.get("text", "")
            ).strip()

            normalized_text_key = " ".join(
                document_text.casefold().split()
            )

            if not document_text:
                continue

            if chunk_key in seen_chunks:
                continue

            if normalized_text_key in seen_texts:
                continue

            selected_row = dict(row)

            selected_row["selected"] = True
            selected_row["selection_reason"] = (
                f"Relevant reranked result "
                f"(rank {row.get('rerank_rank', '?')})"
            )

            selected.append(selected_row)

            seen_chunks.add(chunk_key)
            seen_texts.add(normalized_text_key)

            if len(selected) >= max(
                1,
                int(max_sources),
            ):
                break

        if not selected:
            return "", []

        context_blocks: list[str] = []

        for source_number, row in enumerate(
            selected,
            start=1,
        ):
            control_number = (
                row.get("control_id")
                or "Not specified"
            )

            section_name = (
                row.get("section")
                or "Not specified"
            ).upper()

            parent_id = (
                row.get("parent_id")
                or "Not specified"
            )

            text_content = str(
                row.get("text", "")
            ).strip()

            context_blocks.append(
                "\n".join(
                    [
                        f"[SOURCE {source_number}]",
                        (
                            "ISO 27002 Control Number: "
                            f"{control_number}"
                        ),
                        f"Section: {section_name}",
                        f"Parent ID: {parent_id}",
                        f"Content: {text_content}",
                    ]
                )
            )

        context = "\n\n".join(
            context_blocks
        ).strip()

        return context, selected


if __name__ == "__main__":
    retriever = ISO27002Retriever()

    relevant_question = "unauthorized access"

    relevant_context, relevant_sources = retriever.build_context(
        relevant_question
    )

    print("\nRelevant query:")
    print(f"Question: {relevant_question}")
    print(relevant_context)
    print(f"Sources: {len(relevant_sources)}")

    outside_question = "Ml definition"

    outside_context, outside_sources = retriever.build_context(
        outside_question
    )

    print("\nOutside-scope query:")
    print(f"Question: {outside_question}")
    print(repr(outside_context))
    print(f"Sources: {len(outside_sources)}")