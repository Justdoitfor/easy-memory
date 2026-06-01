from typing import Optional
from pydantic import BaseModel, Field


class BaseRerankerConfig(BaseModel):
    provider: Optional[str] = Field(default=None, description="The reranker provider to use")
    model: Optional[str] = Field(default=None, description="The reranker model to use")
    api_key: Optional[str] = Field(default=None, description="The API key for the reranker service")
    top_k: Optional[int] = Field(default=None, description="Maximum number of documents to return after reranking")
