""" 定义了文本向量化的接口 """

from abc import ABC, abstractmethod
from typing import Literal, Optional

from easy_memory.configs.embeddings.base import BaseEmbedderConfig


class EmbeddingBase(ABC):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        if config is None:
            config = BaseEmbedderConfig()
        else:
            self.config = config

    @abstractmethod
    def embed(self, text, memory_action=Optional[Literal["add", "search", "update"]]):
        pass

    def embed_batch(self, texts, memory_action="add"):
        return [self.embed(text, memory_action) for text in texts]
