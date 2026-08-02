import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHILDREN_PATH = os.path.join(_CURRENT_DIR, "..", "data", "chunks", "iso27002_children.json")


stemmer = PorterStemmer()
CUSTOM_STOP_WORDS = ENGLISH_STOP_WORDS.union({
    'shall', 'should', 'may', 'ensure', 'also', 'use', 'using', 'include', 
    'including', 'the', 'and', 'to', 'of', 'in', 'for', 'is', 'on', 'that', 'by'
})

def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)      # إزالة علامات الترقيم
    tokens = text.split()
    
    # تنقية الكلمات المهملة وتطبيق التجذير (Stemming)
    stemmed_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word not in CUSTOM_STOP_WORDS and len(word) > 2
    ]
    return stemmed_tokens
class LexicalRetriever:


    def __init__(
        self,
        children_path: str | None = None,
        tfidf_weight: float = 0.5,
        score_threshold: float = 0.0,
    ):
        self.children_path = children_path or _CHILDREN_PATH
        self.tfidf_weight = tfidf_weight
        self.score_threshold = score_threshold

        # ---- Load chunks --------------------------------------------------
        with open(self.children_path, "r", encoding="utf-8") as f:
            chunks_raw = json.load(f)
        self.chunks_df = pd.DataFrame(chunks_raw)
        print(f"Loaded {len(self.chunks_df)} child chunks from {self.children_path}")

        # ---- Prepare tokenised corpus -------------------------------------
        self.corpus_texts = self.chunks_df["text"].astype(str).tolist()
        self.tokenized_corpus: list[list[str]] = [tokenize(t) for t in self.corpus_texts]

        # ---- Build TF-IDF index -------------------------------------------
        #   analyzer=tokenize  →  use our custom tokenizer instead of the default
        #   sublinear_tf=True  →  dampens high-frequency term saturation (1 + log tf)
        #   norm="l2"          →  unit-length vectors so cosine_similarity works
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer=tokenize,
            sublinear_tf=True,
            norm="l2",
            ngram_range=(1, 2),
            min_df=1
        )
        # fit_transform on the raw texts; the analyzer calls tokenize() internally
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # ---- Build BM25 index ---------------------------------------------
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        print("BM25 index built successfully")

    # ------------------------------------------------------------------
    # Internal: min-max normalise a 1-D array to [0, 1]
    # ------------------------------------------------------------------
    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Normalise scores to [0, 1].  Returns zeros when all scores are equal."""
        scores = np.asarray(scores, dtype=float)
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    # ------------------------------------------------------------------
    # search: core retrieval logic (mirrors ISO27002Retriever.search)
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[dict]:
  
        # ---- TF-IDF scores (cosine similarity) ---------------------------
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_raw = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        tfidf_norm = self._min_max_normalize(tfidf_raw)

        # ---- BM25 scores -------------------------------------------------
        tokenized_query = tokenize(query)
        bm25_raw = self.bm25_index.get_scores(tokenized_query)
        bm25_norm = self._min_max_normalize(bm25_raw)

        # ---- Hybrid score ------------------------------------------------
        alpha = self.tfidf_weight
        hybrid_scores = (alpha * tfidf_norm) + ((1 - alpha) * bm25_norm)

        # ---- Rank and select top-k above threshold -----------------------
        ranking = np.argsort(hybrid_scores)[::-1]       # descending
        results: list[dict] = []

        for idx in ranking:
            if len(results) >= k:
                break
            score = float(hybrid_scores[idx])
            if score <= self.score_threshold:
                continue

            row = self.chunks_df.iloc[idx]
            results.append({
                "chunk_id": row["child_id"],
                "text": row["text"],
                "control_id": row["control_id"],
                "section": row["section"],
                "parent_id": row["parent_id"],
                "metadata_context": row.get("metadata_context", ""),
                "score": score,
            })

        return results

    # ------------------------------------------------------------------
    # build_context: deduplicated, formatted context string
    # ------------------------------------------------------------------
    def build_context(
        self,
        question: str,
        k: int = 5,
        max_sources: int = 4,
    ) -> tuple[str, list[dict]]:
        rows = self.search(question, k=k + 2)  # retrieve a few extra for dedup headroom
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)

        selected: list[dict] = []
        seen_chunks: set[str] = set()
        seen_texts: set[str] = set()

        for row in rows:
            if row["score"] <= 0:
                continue

            chunk_key = row["chunk_id"]
            doc_text = row["text"].strip()

            # Skip exact duplicates (same chunk_id or identical text)
            if chunk_key in seen_chunks or doc_text in seen_texts:
                continue

            selected.append(row)
            seen_chunks.add(chunk_key)
            seen_texts.add(doc_text)

            if len(selected) == max_sources:
                break

        # Format context string (same layout as the semantic retriever)
        context = ""
        for source_number, row in enumerate(selected, start=1):
            context += (
                f"[Source {source_number}] ISO 27002 Control {row['control_id']} "
                f"- Section: {row['section'].upper()}\n{row['text']}\n\n"
            )

        return context.strip(), selected

    # ------------------------------------------------------------------
    # Convenience: search with TF-IDF only (no BM25)
    # ------------------------------------------------------------------
    def search_tfidf_only(self, query: str, k: int = 5) -> list[dict]:
        """Return top-*k* results using TF-IDF cosine similarity alone."""
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        ranking = np.argsort(scores)[::-1]
        results: list[dict] = []

        for idx in ranking:
            if len(results) >= k:
                break
            score = float(scores[idx])
            if score <= self.score_threshold:
                continue

            row = self.chunks_df.iloc[idx]
            results.append({
                "chunk_id": row["child_id"],
                "text": row["text"],
                "control_id": row["control_id"],
                "section": row["section"],
                "parent_id": row["parent_id"],
                "metadata_context": row.get("metadata_context", ""),
                "score": score,
            })

        return results

    # ------------------------------------------------------------------
    # Convenience: search with BM25 only (no TF-IDF)
    # ------------------------------------------------------------------
    def search_bm25_only(self, query: str, k: int = 5) -> list[dict]:
        """Return top-*k* results using BM25 scores alone."""
        tokenized_query = tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)

        ranking = np.argsort(scores)[::-1]
        results: list[dict] = []

        for idx in ranking:
            if len(results) >= k:
                break
            score = float(scores[idx])
            if score <= self.score_threshold:
                continue

            row = self.chunks_df.iloc[idx]
            results.append({
                "chunk_id": row["child_id"],
                "text": row["text"],
                "control_id": row["control_id"],
                "section": row["section"],
                "parent_id": row["parent_id"],
                "metadata_context": row.get("metadata_context", ""),
                "score": score,
            })

        return results


# ---------------------------------------------------------------------------
# Quick demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    retriever = LexicalRetriever(tfidf_weight=0.7)

    test_query = (
        "Due to manual breach escalation processes, mandatory personal data "
        "breach notifications to authorities may miss the statutory 72-hour "
        "regulatory reporting window."
    )

    print("\n" + "=" * 70)
    print("HYBRID SEARCH (TF-IDF + BM25)")
    print("=" * 70)
    context, sources = retriever.build_context(test_query, k=6, max_sources=4)
    print(context)
    print(f"\nTotal sources selected: {len(sources)}")

    print("\n" + "=" * 70)
    print("TF-IDF ONLY")
    print("=" * 70)
    for r in retriever.search_tfidf_only(test_query, k=4):
        print(f"  [{r['score']:.4f}] Control {r['control_id']} | {r['section']} | {r['text'][:80]}...")

    print("\n" + "=" * 70)
    print("BM25 ONLY")
    print("=" * 70)
    for r in retriever.search_bm25_only(test_query, k=4):
        print(f"  [{r['score']:.4f}] Control {r['control_id']} | {r['section']} | {r['text'][:80]}...")
