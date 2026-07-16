import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Retriever:
    """In-memory semantic search over document chunks."""

    def __init__(self, chunks: list[str]):
        if not chunks:
            raise ValueError("Нужен хотя бы один фрагмент документа")

        self.chunks = chunks
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
        )

    def search(self, question: str, top_k: int = 2) -> list[tuple[str, float]]:
        if not question.strip():
            return []

        question_embedding = self.model.encode(
            question,
            normalize_embeddings=True,
        )

        scores = self.embeddings @ question_embedding
        limit = min(max(top_k, 1), len(self.chunks))
        best_indices = np.argsort(scores)[::-1][:limit]

        return [
            (self.chunks[index], float(scores[index]))
            for index in best_indices
        ]
