import json
import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class VertexAIEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "text-embedding-004"
        self.config.embedding_dims = self.config.embedding_dims or 768
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        if self.config.vertex_credentials_json:
            credentials_info = json.loads(self.config.vertex_credentials_json)
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            vertexai.init(credentials=credentials)
        self.client = TextEmbeddingModel.from_pretrained(self.config.model)

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        embeddings = self.client.get_embeddings([text])
        return embeddings[0].values
