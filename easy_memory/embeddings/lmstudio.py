import os
from typing import Literal, Optional
from easy_memory.configs.embeddings.base import BaseEmbedderConfig
from easy_memory.embeddings.base import EmbeddingBase


class LMStudioEmbedding(EmbeddingBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.config.model = self.config.model or "nomic-embed"
        from openai import OpenAI
        self.client = OpenAI(api_key="lm-studio", base_url=self.config.lmstudio_base_url)

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.config.model).data[0].embedding
