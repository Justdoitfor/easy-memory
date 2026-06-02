from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class LangchainEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        embeddings = self.config.model
        return embeddings.embed_query(text)

    def embed_batch(self, texts, memory_action="add"):
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.config.model
        return embeddings.embed_documents(texts)
