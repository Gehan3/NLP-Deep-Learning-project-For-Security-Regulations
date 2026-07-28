from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb.config import Settings
from collections import Counter

import vector_representation

DB_PATH = Path("iso27002_chroma_db")
COLLECTION_NAME = "iso27002_controls"


def build_embed_text(row) -> str:
    """Combine metadata context + raw text into a single string for embedding.

    Prefixing the chunk with control ID, title, and section ensures the
    embedding vector carries structural signals that queries like
    "control 5.15 Access Control" can match on semantically.
    """
    meta_prefix = row.get("metadata_context", "")
    raw_text = row.get("text", "")
    if meta_prefix:
        return f"{meta_prefix}\n{raw_text}"
    return raw_text


def create_vector_store():

    chunks_df, _ = vector_representation.get_vector_store_data()
    #Problem 1: Metadata is NOT embedded — control IDs are invisible to semantic search
    # Build enriched text that embeds metadata into the vector representation.
    enriched_texts = [build_embed_text(row) for _, row in chunks_df.iterrows()]
    # Re-embed using the enriched text so the vectors carry structural context.
    print("Re-embedding with metadata-enriched text...")
    chunk_embeddings = vector_representation.embedding_model.encode(
        enriched_texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    #Problem 2: Wrong distance metric assumption in ChromaDB retriever
    # Use cosine distance — matches the normalized embeddings from bge-m3.
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hf:space": "cosine"},
    )

    # Prevent Repeated IDs
    ids = chunks_df["child_id"].astype(str).tolist()
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        print(f"Warning: Found duplicate IDs, making them unique automatically...")
        ids = [f"{cid}_{i}" for i, cid in enumerate(ids)]

    collection.upsert(
        ids=ids,
        documents=enriched_texts,  # store enriched text as the document
        metadatas=[
            {
                "parent_id": str(row.get("parent_id", "")),
                "control_id": str(row.get("control_id", "")),
                "section": str(row.get("section", "")),
                "token_count": int(row.get("token_count", 0)),
                "metadata_context": str(row.get("metadata_context", "")),
            }
            for _, row in chunks_df.iterrows()
        ],
        embeddings=chunk_embeddings.tolist(),
    )

    print(f"Successfully stored {len(ids)} chunks with metadata-enriched BGE-M3 embeddings in ChromaDB!")
    return collection


if __name__ == "__main__":
    create_vector_store()