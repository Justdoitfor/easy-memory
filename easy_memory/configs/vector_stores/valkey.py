from typing import Optional
from pydantic import BaseModel, Field

class ValkeyConfig(BaseModel):
    collection_name: str = Field("easy-memory", description="Name of the collection")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    redis_url: Optional[str] = Field(None, description="Valkey/Redis connection URL")