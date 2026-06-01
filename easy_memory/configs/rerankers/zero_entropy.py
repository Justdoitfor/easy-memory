from typing import Optional
from pydantic import Field
from easy_memory.configs.rerankers.base import BaseRerankerConfig


class ZeroEntropyRerankerConfig(BaseRerankerConfig):
    model: str = Field(default="zerank-1", description="Model to use for reranking")
    api_key: Optional[str] = Field(default=None, description="Zero Entropy API key")
    top_k: Optional[int] = Field(default=None, description="Number of top documents to return after reranking")
