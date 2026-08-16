import os
from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv

load_dotenv()

class BaseEmbeddingService(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass


class HuggingFaceEmbeddingService(BaseEmbeddingService):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


def get_embedding_provider() -> BaseEmbeddingService:
    return HuggingFaceEmbeddingService()