import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class GoogleGenAIEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "text-embedding-004"
        self.config.embedding_dims = self.config.embedding_dims or 768
        from google import genai
        self.client = genai.Client(api_key=self.config.api_key or os.getenv("GOOGLE_API_KEY"))

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        response = self.client.models.embed_content(model=self.config.model, contents=text)
        return response.embeddings[0].values
