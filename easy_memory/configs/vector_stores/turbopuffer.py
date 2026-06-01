from typing import Optional
from pydantic import BaseModel, Field


class TurbopufferConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    api_key: Optional[str] = Field(None, description="API key for Turbopuffer")
