import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class HuggingFaceEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "multi-qa-MiniLM-L6-cos-v1"
        self.config.embedding_dims = self.config.embedding_dims or 384
        if self.config.huggingface_base_url:
            import httpx
            self.client = httpx.Client(base_url=self.config.huggingface_base_url)
            self._use_remote = True
        else:
            from sentence_transformers import SentenceTransformer
            self.client = SentenceTransformer(self.config.model, **self.config.model_kwargs)
            self._use_remote = False

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        if self._use_remote:
            response = self.client.post("/embed", json={"inputs": text})
            return response.json()[0]
        return self.client.encode(text).tolist()

    def embed_batch(self, texts, memory_action="add"):
        texts = [t.replace("\n", " ") for t in texts]
        if self._use_remote:
            response = self.client.post("/embed", json={"inputs": texts})
            return response.json()
        return self.client.encode(texts).tolist()
