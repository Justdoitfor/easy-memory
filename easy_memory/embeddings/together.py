import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class TogetherEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "M2-BERT-80M-2K-Retrieval"
        self.config.embedding_dims = self.config.embedding_dims or 768
        from openai import OpenAI
        self.client = OpenAI(api_key=self.config.api_key or os.getenv("TOGETHER_API_KEY"),
                             base_url="https://api.together.xyz/v1")

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.config.model).data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(input=texts, model=self.config.model)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
