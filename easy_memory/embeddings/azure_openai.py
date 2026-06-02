import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class AzureOpenAIEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536
        from openai import AzureOpenAI
        azure = self.config.azure_kwargs
        self.client = AzureOpenAI(api_key=azure.api_key or self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                                  azure_endpoint=azure.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                                  api_version=azure.api_version or os.getenv("OPENAI_API_VERSION", "2024-02-01"))

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        kwargs = {"input": [text], "model": self.config.model}
        if self.config.embedding_dims:
            kwargs["dimensions"] = self.config.embedding_dims
        return self.client.embeddings.create(**kwargs).data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        texts = [t.replace("\n", " ") for t in texts]
        kwargs = {"input": texts, "model": self.config.model}
        if self.config.embedding_dims:
            kwargs["dimensions"] = self.config.embedding_dims
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
