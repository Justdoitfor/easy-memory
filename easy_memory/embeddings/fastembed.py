from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class FastEmbedEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "BAAI/bge-small-en-v1.5"
        from fastembed import TextEmbedding
        self.client = TextEmbedding(model_name=self.config.model)

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        embeddings = list(self.client.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts, memory_action="add"):
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = list(self.client.embed(texts))
        return [e.tolist() for e in embeddings]
