import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import string
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    'shall', 'should', 'may', 'ensure', 'also', 'use', 'using', 'include', 
    'including', 'the', 'and', 'to', 'of', 'in', 'for', 'is', 'on', 'that', 'by'
})

def advanced_tokenize(text: str) -> list[str]:
    """توحيد النص، إزالة علامات الترقيم، إزالة الكلمات المهملة، والتجذير للـ BM25."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [
        stemmer.stem(word) 
        for word in tokens 
        if word not in CUSTOM_STOP_WORDS and len(word) > 2
    ]


print("Loading BAAI/bge-m3 model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

def generate_chunk_embeddings(chunks_df):
    chunk_texts = chunks_df["text"].tolist()
    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        batch_size=16, #PREVENT OOM out of memory
        convert_to_numpy=True,
        normalize_embeddings=True, #cos similarity يسرع
        show_progress_bar=True
    )
    
    return chunk_embeddings
#Problem 3: BM25 tokenization is too naive for a security/c compliance domain
def retrieve_hybrid_bge(query, chunks_df, chunk_embeddings, bm25_index, alpha=0.6, k=3):
    tokenized_query = advanced_tokenize(query)
    bm25_raw_scores = bm25_index.get_scores(tokenized_query)
    bm25_scores = min_max_normalize(bm25_raw_scores)
    
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embedding_raw_scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    embedding_scores = min_max_normalize(embedding_raw_scores)
    hybrid_scores = ((1 - alpha) * bm25_scores) + (alpha * embedding_scores)
   
    ranking = np.argsort(hybrid_scores)[::-1][:k]
    valid_indices = [idx for idx in ranking if hybrid_scores[idx] >= 0.5][:k]
    
  
    results = chunks_df.iloc[valid_indices].copy()
    results["score"] = hybrid_scores[valid_indices]
    results["retriever"] = "BAAI/bge-m3 + BM25(Optimized)"
    
    return results[[
        "retriever", "child_id", "parent_id", "control_id", 
        "section", "score", "metadata_context", "text"
    ]].reset_index(drop=True)

# --- UPDATE FOR CHROMA_DB: 
def get_vector_store_data():
    chunks_df = pd.read_json("data/chunks/iso27002_children.json")
    chunk_embeddings = generate_chunk_embeddings(chunks_df)
    return chunks_df, chunk_embeddings

if __name__ == "__main__":
    chunks_df = pd.read_json("data/chunks/iso27002_children.json")
    
    print(f"Loaded {len(chunks_df)} chunks successfully!")

    #K for term frequency saturation b for document length normalization
    tokenized_chunks = [advanced_tokenize(str(text)) for text in chunks_df["text"].tolist()]
    bm25_index = BM25Okapi(tokenized_chunks, k1=1.2, b=0.5)
    chunk_embeddings = generate_chunk_embeddings(chunks_df)
    
    test_query = "Unauthorized access or insider escalation leading to unauthorized users gaining elevated privileges , which may result in the compromise of sensitive data, unauthorized system modifications, disruption of critical services, and potential regulatory non-compliance."
    #"Malware encrypts business-critical data causing service disruption"
    results = retrieve_hybrid_bge(test_query, chunks_df, chunk_embeddings, bm25_index, alpha=0.6, k=5)

    pd.set_option('display.max_columns', None)  # عرض كل الأعمدة
    pd.set_option('display.width', 1000)
    print(results)
    pass