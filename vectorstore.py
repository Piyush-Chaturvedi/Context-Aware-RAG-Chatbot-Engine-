"""
Lightweight vector store built on TF-IDF + cosine similarity.

Why TF-IDF instead of a neural embedding model?
- Zero API cost, no GPU, no large model download -> runs anywhere instantly.
- Good enough for keyword/technical-term-heavy content like manuals and
  engineering standards (which is exactly what this project deals with).
- The retrieval interface (`build_index` / `query`) is written so it can be
  swapped for a real embedding model (e.g. sentence-transformers + FAISS)
  later with minimal changes -- see README "Upgrading retrieval" section.
"""
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingest import load_and_chunk_folder


class VectorStore:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.chunks = []  # list of {"text":..., "source":...}

    def build_index(self, folder_path):
        self.chunks = load_and_chunk_folder(folder_path)
        if not self.chunks:
            raise ValueError(f"No documents found/parsed in {folder_path}")

        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "matrix": self.matrix, "chunks": self.chunks},
                f,
            )

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.chunks = data["chunks"]
        return self

    def query(self, query_text, top_k=3):
        """
        Returns a list of (chunk_dict, score) sorted by descending relevance.
        """
        if self.vectorizer is None:
            raise RuntimeError("Index not built/loaded yet.")
        q_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]
