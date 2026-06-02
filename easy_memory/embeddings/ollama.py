import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class OllamaEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "nomic-embed-text"
        import ollama
        self.client = ollama.Client(host=self.config.ollama_base_url or "http://localhost:11434")

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        response = self.client.embeddings(model=self.config.model, prompt=text)
        return response["embedding"]
