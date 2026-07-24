from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb.config import Settings
from collections import Counter 

import vector_representation

DB_PATH = Path("iso27002_chroma_db")
COLLECTION_NAME = "iso27002_controls"


def create_vector_store():
    
    chunks_df, chunk_embeddings = vector_representation.get_vector_store_data()
    
    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Prevent Repeated IDs
    ids = chunks_df["child_id"].astype(str).tolist()
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        print(f"Warning: Found duplicate IDs, making them unique automatically...")
        ids = [f"{cid}_{i}" for i, cid in enumerate(ids)]
        
    collection.upsert(
        ids=ids,
        documents=chunks_df["text"].tolist(),
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

    print(f"Successfully stored {len(ids)} chunks with pre-computed BGE-M3 embeddings in ChromaDB!")
    return collection


if __name__ == "__main__":
    create_vector_store()