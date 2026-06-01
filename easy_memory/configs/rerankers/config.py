from typing import Optional
from pydantic import BaseModel, Field


class RerankerConfig(BaseModel):
    provider: str = Field(description="Reranker provider (e.g., 'cohere', 'sentence_transformer')", default="cohere")
    config: Optional[dict] = Field(description="Provider-specific reranker configuration", default=None)
    model_config = {"extra": "forbid"}
